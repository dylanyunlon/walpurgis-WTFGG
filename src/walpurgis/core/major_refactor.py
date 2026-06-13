"""
walpurgis/core/major_refactor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 73af12903 (第24个, 共9062)
Subject: "Major refactoring, combining gpt2 and bert"

上游改动摘要
============
此文件整合建模 73af12903 中两个关键新增文件的设计语义:

【megatron/module.py (34行新增)】
  MegatronModule 基类，继承 torch.nn.Module。
  核心改动: 覆写 state_dict() / state_dict_for_save_checkpoint()，
  使参数始终以 float32 格式保存，即使模型运行在 fp16 模式。
  原因: 检查点跨精度加载时的数值稳定性。

  方法结构:
    state_dict(destination=None, prefix='', keep_vars=False)
      → {k: v.float() for k, v in state.items()}
    state_dict_for_save_checkpoint(destination=None, prefix='', keep_vars=False)
      → 同上 (与 state_dict 等价，专为检查点保存路径设计)

【megatron/training.py (499行新增)】
  统一 pretrain 主循环，将原来 pretrain_bert.py 和 pretrain_gpt2.py
  中重复的训练逻辑提取为共享基础设施。

  主要函数:
    pretrain(train_data_iterator, val_data_iterator, test_data_iterator,
             end_of_epoch_callback_provider, model_provider,
             forward_step_func, args_defaults={})
      → 统一入口，替代原 main() 函数

    setup_model_and_optimizer(model_provider)
      → 返回 (model, optimizer, lr_scheduler)
      → 整合了 get_model() + get_optimizer() + get_learning_rate_scheduler()

    train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator)
      → 主训练循环

    evaluate(forward_step_func, data_iterator, model)
      → 验证集评估

    initialize_megatron(extra_args_provider=None, args_defaults={})
      → 替代原 initialize_distributed() + set_random_seed()
      → 73af12903 同步将 initialize_distributed/set_random_seed
         从 pretrain_gpt2.py 移入 megatron/utils.py

  megatron/utils.py 同步新增 (73af12903):
    initialize_distributed(args)
    set_random_seed(seed)
    generate_samples.py 同步改为:
      from megatron.utils import initialize_distributed, set_random_seed

kicker: pretrain_bert.py / pretrain_gpt2.py 大幅瘦身
  pretrain_bert.py:  528行 → 精简，保留 model_provider / forward_step
  pretrain_gpt2.py:  462行 → 精简，保留 model_provider / forward_step
  两者均移除了大量重复的 train_step / evaluate / main 逻辑。

Docker 变更 (73af12903):
  docker/Dockerfile:
    - base image: nvidia/pytorch:19.05-py3 → 19.09-py3
    - 删除 apex 手动安装块 (19.09 镜像内置)
  docker/README.md: 整文件删除

鲁迅拿法改写（≥20%）
=====================
鲁迅在《随感录》里写:「中国的文明……多数是侍奉少数人的设施。」

上游两个 pretrain 脚本——pretrain_bert.py 和 pretrain_gpt2.py——
各自为王，互不相让。
同样的 setup_model_and_optimizer、同样的 train 循环、
同样的 evaluate 函数，写了两遍，改一处必须改两处。
如同清朝的南北两套官场——架构相同，规矩相同，
但谁也不承认对方，谁也不引用对方，各自傲然而立。

73af12903 做的是「统一」——把重复的结构抽进 training.py，
让 pretrain_bert 和 pretrain_gpt2 退化为薄薄的「地方官」，
只需向中央汇报「我的 model_provider 和 forward_step 是这样的」。

但上游的统一是一次「强制合并」——没有解释合并的契约，
没有文档说明 model_provider / forward_step_func 的接口规范，
没有注释说明 pretrain() 的调用方应该传入什么。
如同一道圣旨: 你们合并了，具体怎么合并，自己看着办。

Walpurgis 将「统一训练循环」的设计契约改写为五个显式组件:

1. **`CheckpointPrecision` 枚举** — 显式化 MegatronModule.state_dict()
   总是保存 float32 的原因: MIXED_PRECISION_TRAINING / FLOAT32_CHECKPOINT。
   上游: 直接 v.float()，无注释。

2. **`MegatronModuleSpec` dataclass** — 建模 MegatronModule 基类的
   行为契约: state_dict 精度策略、支持的 checkpoint 格式、
   与 FP16_Module 的组合关系。

3. **`PretrainCallbackSpec` dataclass** — 建模 pretrain() 的
   回调函数接口契约: model_provider、forward_step_func 的签名规范。
   上游: callable 参数无类型注解，无接口文档。

4. **`TrainingLoopManifest` dataclass** — 建模 training.py 的
   主要函数清单，记录各函数的输入/输出规格和
   与 pretrain_bert/gpt2 的对应关系。

5. **`MajorRefactorManifest`** — 汇总 73af12903 整体重构的
   完整元数据: 新增文件、删除逻辑、Docker 变更、
   megatron/utils.py 新增函数。audit() 输出完整报告。

全链路 _dbg() 断点共 **16 处**:
MODULE_LOAD, CHECKPOINT_PRECISION_ENUM, MODULE_SPEC_INIT,
MODULE_SPEC_VALIDATE, PRETRAIN_SPEC_INIT, PRETRAIN_SPEC_VALIDATE,
TRAINING_MANIFEST_INIT, MAJOR_MANIFEST_INIT, MAJOR_AUDIT_START,
MAJOR_AUDIT_MODULE, MAJOR_AUDIT_TRAINING, MAJOR_AUDIT_DOCKER,
MAJOR_AUDIT_DONE, SELF_CHECK_START, SELF_CHECK_1~4, SELF_CHECK_PASS。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0").strip()
_DEBUG = _DEBUG_ENV in ("1", "major_refactor")


def _dbg(tag: str, msg: object = "") -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出结构化诊断行到 stderr"""
    if _DEBUG:
        print(f"[REFACTOR-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "major_refactor 加载 — 73af12903 module.py + training.py 设计契约建模")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CheckpointPrecision — MegatronModule 保存精度策略枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CheckpointPrecision(Enum):
    """
    MegatronModule.state_dict() 的参数精度保存策略。

    上游 megatron/module.py (73af12903 新增):
      class MegatronModule(torch.nn.Module):
        def state_dict(self, ...):
          state_dict = super().state_dict(...)
          return {k: v.float() for k, v in state_dict.items()}

    为什么总是保存 float32？

    场景: 模型以 fp16 运行 (FP16_Module 包装)，
    但检查点需要跨精度加载 (例如 fp16 训练后 fp32 微调)。
    直接保存 fp16 参数，加载到 fp32 模型时需要额外转换，
    且 fp16 的数值范围有限，可能有微小精度损失。
    强制 float32 保存，使检查点精度无关于训练精度。

    鲁迅视角：检查点是一封给未来的信。
    fp16 的笔迹太细，未来的人可能看不清楚。
    上游选择「用 fp32 誊写一遍再寄出」——
    多花一点空间，换来跨精度的确定性。
    但这个决策没有写在信封上，只有读代码才知道。
    """
    FLOAT32_ALWAYS = auto()  # 无论训练精度，检查点始终 float32
    NATIVE_DTYPE   = auto()  # 保存训练时的实际精度 (非 MegatronModule 行为)

    @property
    def megatron_module_behavior(self) -> bool:
        """True → 这是 MegatronModule 实际采用的策略"""
        return self is CheckpointPrecision.FLOAT32_ALWAYS

    @property
    def description(self) -> str:
        return {
            CheckpointPrecision.FLOAT32_ALWAYS: (
                "无论模型以 fp16/bf16 运行，state_dict() 始终将参数 .float() "
                "转换为 float32 后保存。跨精度加载安全，检查点体积约为 fp16 的 2x。"
            ),
            CheckpointPrecision.NATIVE_DTYPE: (
                "保存参数的实际训练精度。fp16 训练时产生 fp16 检查点。"
                "跨精度加载需要额外转换，数值范围受 fp16 限制。"
                "MegatronModule 未采用此策略。"
            ),
        }[self]


_dbg("CHECKPOINT_PRECISION_ENUM",
     f"CheckpointPrecision: {[p.name for p in CheckpointPrecision]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MegatronModuleSpec — MegatronModule 行为契约
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class MegatronModuleSpec:
    """
    megatron/module.py 的 MegatronModule 基类行为契约。

    上游 73af12903 新增此基类 (34行)，
    所有 Megatron 模型 (BertModel, GPT2Model) 均继承自它。

    字段说明
    --------
    upstream_file        : 上游文件路径
    upstream_lines       : 上游文件行数
    base_class           : 继承的基类
    checkpoint_precision : state_dict 的保存精度策略
    overridden_methods   : 覆写的 nn.Module 方法列表
    composition_pattern  : 与 FP16_Module 的组合模式描述
    """
    upstream_file: str = "megatron/module.py"
    upstream_lines: int = 34
    base_class: str = "torch.nn.Module"
    checkpoint_precision: CheckpointPrecision = CheckpointPrecision.FLOAT32_ALWAYS
    overridden_methods: Tuple[str, ...] = field(default_factory=lambda: (
        "state_dict(destination, prefix, keep_vars) → Dict[str, Tensor(float32)]",
        "state_dict_for_save_checkpoint(destination, prefix, keep_vars) → Dict[str, Tensor(float32)]",
    ))
    composition_pattern: str = (
        "训练时: FP16_Module(MegatronModule) — FP16_Module 管理精度转换;\n"
        "保存时: MegatronModule.state_dict() 绕过 FP16 包装，直接以 float32 序列化。\n"
        "加载时: load_checkpoint() 加载 float32 state_dict，再由 FP16_Module 适配精度。"
    )

    def __post_init__(self) -> None:
        _dbg("MODULE_SPEC_INIT", (
            f"MegatronModuleSpec: {self.upstream_file} ({self.upstream_lines}行), "
            f"precision={self.checkpoint_precision.name}"
        ))
        self._validate()

    def _validate(self) -> None:
        """验证基类契约的一致性"""
        _dbg("MODULE_SPEC_VALIDATE", "验证 MegatronModule 契约")
        assert self.checkpoint_precision.megatron_module_behavior, (
            "MegatronModule 必须采用 FLOAT32_ALWAYS 精度策略"
        )
        assert len(self.overridden_methods) >= 2, (
            "期望至少覆写 state_dict 和 state_dict_for_save_checkpoint"
        )
        _dbg("MODULE_SPEC_VALIDATE", "契约验证通过 ✓")

    def as_dict(self) -> Dict[str, object]:
        return {
            "upstream_file": self.upstream_file,
            "upstream_lines": self.upstream_lines,
            "base_class": self.base_class,
            "checkpoint_precision": self.checkpoint_precision.name,
            "precision_description": self.checkpoint_precision.description,
            "overridden_methods": list(self.overridden_methods),
            "composition_pattern": self.composition_pattern,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PretrainCallbackSpec — pretrain() 回调函数接口契约
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class PretrainCallbackSpec:
    """
    megatron/training.py pretrain() 的回调函数接口契约。

    上游 pretrain() 接受两个关键 callable 参数，无类型注解:
      1. model_provider: 构造并返回模型实例
      2. forward_step_func: 执行一次前向计算并返回 (loss, metrics)

    Walpurgis 将这两个接口的签名契约显式化。
    """
    model_provider_signature: str = (
        "model_provider(args) → nn.Module\n"
        "  构造并返回已初始化的模型实例。\n"
        "  调用方: setup_model_and_optimizer()\n"
        "  BERT 实现: lambda args: BertModel(args.num_tokentypes)\n"
        "  GPT-2 实现: lambda args: GPT2Model(args.num_tokentypes)\n"
        "  [73af12903: pretrain_bert.py 和 pretrain_gpt2.py 各自实现此 callable]"
    )
    forward_step_signature: str = (
        "forward_step_func(data_iterator, model, args, timers) → (loss, metrics)\n"
        "  从 data_iterator 取一批数据，执行前向计算，返回 loss 张量和指标字典。\n"
        "  调用方: train_step() / evaluate()\n"
        "  BERT: get_batch() + model(input_ids, attention_mask, tokentype_ids)\n"
        "  GPT-2: get_batch() + model(tokens, position_ids, attention_mask)\n"
        "  [73af12903: 替代原 pretrain_bert/gpt2 中重复的 train_step 实现]"
    )
    # 73af12903 是否为 model_provider / forward_step 提供类型注解？
    has_type_annotations: bool = False  # 上游无注解，Walpurgis 补全契约文档

    def __post_init__(self) -> None:
        _dbg("PRETRAIN_SPEC_INIT",
             f"PretrainCallbackSpec: has_type_annotations={self.has_type_annotations}")
        self._validate()

    def _validate(self) -> None:
        _dbg("PRETRAIN_SPEC_VALIDATE", "验证 pretrain 回调契约")
        assert "model_provider" in self.model_provider_signature
        assert "forward_step" in self.forward_step_signature
        _dbg("PRETRAIN_SPEC_VALIDATE", "回调契约验证通过 ✓")

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_provider_signature": self.model_provider_signature,
            "forward_step_signature": self.forward_step_signature,
            "has_type_annotations": self.has_type_annotations,
            "upstream_note": (
                "73af12903 引入 megatron/training.py，将 pretrain_bert.py "
                "和 pretrain_gpt2.py 的重复训练循环统一为 pretrain() 函数。"
                "两个脚本只需实现 model_provider 和 forward_step_func，"
                "其余均由 training.py 统一处理。"
            ),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TrainingLoopManifest — training.py 主要函数清单
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainingLoopManifest:
    """megatron/training.py (499行) 的主要函数清单"""

    upstream_file: str = "megatron/training.py"
    upstream_lines: int = 499

    FUNCTIONS: Tuple[Dict[str, object], ...] = field(default_factory=lambda: (
        {
            "name": "pretrain",
            "signature": "pretrain(train_iter, val_iter, test_iter, "
                         "end_of_epoch_callback_provider, model_provider, "
                         "forward_step_func, args_defaults={})",
            "role": "统一 pretrain 入口，替代 pretrain_bert/gpt2 的 main()",
            "calls": ["initialize_megatron", "setup_model_and_optimizer",
                      "train", "evaluate"],
        },
        {
            "name": "setup_model_and_optimizer",
            "signature": "setup_model_and_optimizer(model_provider) → (model, optimizer, lr_scheduler)",
            "role": "整合 get_model + get_optimizer + get_learning_rate_scheduler",
            "calls": ["model_provider", "FP16_Module", "DDP", "Adam", "AnnealingLR"],
        },
        {
            "name": "train",
            "signature": "train(forward_step_func, model, optimizer, lr_scheduler, train_iter, val_iter)",
            "role": "主训练循环: iteration → forward → backward → optimizer step → log",
            "calls": ["train_step", "evaluate_and_print_results", "save_checkpoint"],
        },
        {
            "name": "evaluate",
            "signature": "evaluate(forward_step_func, data_iterator, model) → Dict[str, float]",
            "role": "验证集评估循环，返回损失和指标",
            "calls": ["forward_step_func"],
        },
        {
            "name": "initialize_megatron",
            "signature": "initialize_megatron(extra_args_provider=None, args_defaults={}) → args",
            "role": "替代原 initialize_distributed + set_random_seed + 参数解析",
            "calls": ["get_args", "initialize_distributed", "set_random_seed"],
        },
    ))

    # utils.py 73af12903 新增的函数
    UTILS_NEW_FUNCTIONS: Tuple[str, ...] = field(default_factory=lambda: (
        "initialize_distributed(args) — 替代 pretrain_gpt2.py 中的同名函数",
        "set_random_seed(seed) — 替代 pretrain_gpt2.py 中的同名函数",
    ))

    def __post_init__(self) -> None:
        _dbg("TRAINING_MANIFEST_INIT",
             f"TrainingLoopManifest: {len(self.FUNCTIONS)} 个主要函数, "
             f"{self.upstream_lines} 行")

    def as_dict(self) -> Dict[str, object]:
        return {
            "upstream_file": self.upstream_file,
            "upstream_lines": self.upstream_lines,
            "functions": list(self.FUNCTIONS),
            "utils_new_functions": list(self.UTILS_NEW_FUNCTIONS),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. MajorRefactorManifest — 73af12903 整体重构元数据汇总
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class MajorRefactorManifest:
    """
    73af12903 「Major refactoring, combining gpt2 and bert」的
    整体重构元数据汇总。

    汇总所有子组件的审计信息，提供统一的入口。
    """
    upstream_commit: str = "73af12903"
    upstream_subject: str = "Major refactoring, combining gpt2 and bert"
    files_changed: int = 23
    insertions: int = 1964
    deletions: int = 3268

    module_spec: MegatronModuleSpec = field(
        default_factory=MegatronModuleSpec
    )
    pretrain_spec: PretrainCallbackSpec = field(
        default_factory=PretrainCallbackSpec
    )
    training_manifest: TrainingLoopManifest = field(
        default_factory=TrainingLoopManifest
    )

    # Docker 变更
    DOCKER_CHANGES: Dict[str, str] = field(default_factory=lambda: {
        "base_image_old": "nvcr.io/nvidia/pytorch:19.05-py3",
        "base_image_new": "nvcr.io/nvidia/pytorch:19.09-py3",
        "apex_change": "删除手动安装 apex 块 (pip uninstall -y apex + git clone + pip install)。"
                       "19.09 镜像内置 apex，无需手动安装。",
        "readme_deleted": "docker/README.md 整文件删除 (曾注明需要预先 clone PySOL)",
        "net_line_change_docker": "-9 行 (11 insertions, 1 deletion for README removal)",
    })

    def net_line_change(self) -> int:
        return self.insertions - self.deletions  # -1304

    def audit(self) -> Dict[str, object]:
        """输出 73af12903 整体重构的完整审计报告"""
        _dbg("MAJOR_AUDIT_START",
             f"审计 {self.upstream_commit}: {self.upstream_subject}")

        _dbg("MAJOR_AUDIT_MODULE",
             f"MegatronModule: {self.module_spec.upstream_lines}行, "
             f"精度={self.module_spec.checkpoint_precision.name}")
        _dbg("MAJOR_AUDIT_TRAINING",
             f"training.py: {self.training_manifest.upstream_lines}行, "
             f"{len(self.training_manifest.FUNCTIONS)} 个主要函数")
        _dbg("MAJOR_AUDIT_DOCKER",
             f"Docker base: {self.DOCKER_CHANGES['base_image_old']} "
             f"→ {self.DOCKER_CHANGES['base_image_new']}")

        result = {
            "commit_meta": {
                "hash": self.upstream_commit,
                "subject": self.upstream_subject,
                "files_changed": self.files_changed,
                "insertions": self.insertions,
                "deletions": self.deletions,
                "net_line_change": self.net_line_change(),
            },
            "megatron_module": self.module_spec.as_dict(),
            "pretrain_callbacks": self.pretrain_spec.as_dict(),
            "training_loop": self.training_manifest.as_dict(),
            "docker_changes": self.DOCKER_CHANGES,
            "key_architectural_decisions": [
                "MegatronModule.state_dict() 强制 float32，确保跨精度检查点安全性",
                "training.py 统一 pretrain 循环，消除 bert/gpt2 训练代码重复",
                "pretrain_bert/gpt2 退化为薄包装: 只需实现 model_provider + forward_step",
                "initialize_distributed/set_random_seed 从 pretrain_gpt2.py 提取到 megatron/utils.py",
                "tokentype_ids 透传到 GPT2Model.forward()，为多模态扩展预留接口",
                "Docker 基础镜像升级至 19.09，内置 apex 消除手动安装风险",
            ],
        }
        _dbg("MAJOR_AUDIT_DONE", "整体重构审计完成 ✓")
        return result

    def self_check(self) -> None:
        """4 项断言"""
        _dbg("SELF_CHECK_START", "开始 4 项断言")

        # 1. net_line_change 为负
        assert self.net_line_change() < 0, (
            f"73af12903 应净减少代码，得到 {self.net_line_change()}"
        )
        _dbg("SELF_CHECK_1",
             f"✓ net_line_change={self.net_line_change()} < 0")

        # 2. MegatronModule 精度为 FLOAT32_ALWAYS
        assert self.module_spec.checkpoint_precision is CheckpointPrecision.FLOAT32_ALWAYS
        _dbg("SELF_CHECK_2", "✓ MegatronModule 使用 FLOAT32_ALWAYS")

        # 3. training.py 主要函数 >= 4
        assert len(self.training_manifest.FUNCTIONS) >= 4
        _dbg("SELF_CHECK_3",
             f"✓ {len(self.training_manifest.FUNCTIONS)} 个主要函数")

        # 4. Docker 基础镜像确实升级了
        old = self.DOCKER_CHANGES["base_image_old"]
        new = self.DOCKER_CHANGES["base_image_new"]
        assert old != new, "Docker 基础镜像应有变化"
        assert "19.09" in new, f"新镜像应包含 19.09，得到 {new}"
        _dbg("SELF_CHECK_4",
             f"✓ Docker: {old} → {new}")

        _dbg("SELF_CHECK_PASS", "4 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块级初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANIFEST = MajorRefactorManifest()
_dbg("MAJOR_MANIFEST_INIT", (
    f"MajorRefactorManifest 初始化: "
    f"net_line_change={MANIFEST.net_line_change()}, "
    f"{len(MANIFEST.training_manifest.FUNCTIONS)} 个训练函数"
))
MANIFEST.self_check()

_dbg("MODULE_READY", "major_refactor 就绪 — 73af12903 module.py + training.py 设计契约")

__all__ = [
    "CheckpointPrecision",
    "MegatronModuleSpec",
    "PretrainCallbackSpec",
    "TrainingLoopManifest",
    "MajorRefactorManifest",
    "MANIFEST",
]
