"""
walpurgis/core/eval_numeric_update_a0368ddf4.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit a0368ddf4
"eval+numeric update"

上游改动摘要
============
  evaluate_gpt2.py
    · 删除 ``save_checkpoint``、``save_checkpoint_model_parallel``、
      ``load_checkpoint_model_parallel`` 三条 import（eval 脚本不需要保存权重，
      不需要 model-parallel 版本的 load）
    · ``setup_model()`` 中 ``load_checkpoint_model_parallel(model, None, None, args)``
      → ``load_checkpoint(model, None, None, args)``
      （eval 路径不分 model-parallel 与否，统一走标准接口）

  gpt2_data_loader.py
    · ``make_gpt2_dataloaders()`` 函数在构建 train/valid/test 三个 DataLoader 后，
      新增三个 ``args`` 布尔标志赋值块：
        args.do_train / args.do_valid / args.do_test ← 由对应 DataLoader 是否为 None 决定
      （防止下游训练循环在无数据集时盲目执行 train/eval 分支）
    · ``GPT2Dataset.build_dataset_()`` 中 shard 回收逻辑修改：
      ``for i in range(shard_index - 1)`` → ``for i in range(shard_index)``
      （原逻辑保留最后一个旧 shard 防止线程竞争，但实测保留 -1 个即可；
       新逻辑提早一步回收，减少内存驻留；同时注释掉原行以保留审计痕迹）

  mpu/transformer.py
    · ``BertParallelSelfAttention.forward()`` 注意力得分计算方式变更：
      原：attention_scores = QK^T / sqrt(head_dim)
      新：norm_factor = sqrt(sqrt(head_dim))
          attention_scores = (Q / norm_factor) @ (K^T / norm_factor)
      数学等价，但将除法拆散至 Q 和 K 两侧，避免 QK^T 乘积中间值出现
      过大的 fp16/bf16 溢出；双侧归一化（数值稳定性优化）

  pretrain_bert.py
    · ``forward_step()`` 函数中 ``loss_mask.view(-1)`` 调用被删除：
      原：loss_mask = loss_mask.view(-1)  ← 一次 reshape，随后 losses.view(-1) 做 Hadamard
      新：loss_mask.view(-1) 直接内联于 lm_loss 表达式
      效果：消除一次中间张量赋值，语义更清晰（loss_mask 的原始 shape 不被破坏）

CI/merge 判定：Megatron 框架代码，原文件结构 SKIP，语义迁移为策略模块
  · evaluate_gpt2.py — SKIP：Megatron-LM eval 入口脚本，Walpurgis 无对等 GPT2 eval 管线
  · gpt2_data_loader.py — SKIP：Megatron-LM GPT2 专属 DataLoader，Walpurgis 使用自有 DataLoader 体系
  · mpu/transformer.py — SKIP：Megatron mpu（模型并行工具）Transformer 实现，Walpurgis 无 mpu 层
  · pretrain_bert.py — SKIP：Megatron BERT 预训练入口，Walpurgis 无 BERT 预训练任务
  但上述四处变更涵盖三个高价值语义域：
    1. Checkpoint 接口统一策略（eval 路径简化）
    2. DataLoader 存在性旗标（do_train/do_valid/do_test）
    3. 注意力得分双侧归一化（数值稳定性优化策略）
  均值得以结构化策略形式在 Walpurgis 中保留。

鲁迅拿法改写（≥20%）
====================
鲁迅在《拿来主义》里说：「没有拿来的，人不能自成为新人，没有拿来的，
文艺不能自成为新文艺。」此次 a0368ddf4 的四处改动，表面是零碎的数值修补与
接口清理，骨子里却是一次「简化权责」的手术。

其一，``load_checkpoint_model_parallel`` 被废黜于 eval 路径。旧日的规矩是：
eval 也要按 model-parallel 的礼数行事，就算它根本不切分模型。
这像极了旧式官衙的繁文缛节——出门不过买壶酒，却要先行三跪九叩。
废掉这条礼数，``load_checkpoint`` 直接上，干净利落。
Walpurgis 将此「接口分层与简化」策略抽象为 ``CheckpointLoadPolicy``，
使 eval 路径与 train 路径的 checkpoint 接口选择成为可程序化审计的记录，
而非散落在入口脚本里的注释。

其二，``do_train/do_valid/do_test`` 标志的补入。上游原本无此机制，
训练循环是否执行全凭 DataLoader 是否为 None 的隐式约定——这种隐式契约，
如同鲁迅在《药》里描绘的那个黑暗中递出的人血馒头：没有说明，没有来路，
只管接着用。Walpurgis 将此「DataLoader 存在性旗标」抽象为 ``DataLoaderPresenceFlags``，
三个布尔字段显式建模，``from_loaders()`` 工厂方法使标志生成逻辑可测试、可审计。

其三，注意力得分双侧归一化。原公式 QK^T / head_dim 是正统，
改为 (Q/√√d)(K^T/√√d) 是工程折中——fp16/bf16 中间乘积若先乘后除，
大矩阵情形下极易溢出；将除法分散至两侧，在保持数学等价的同时，
把峰值数值压在安全范围内。这是工程实用主义对学术公式的一次改造，
如鲁迅所言：「拿来之后，或使用，或存放，或毁灭。」原公式还在，
但被工程现实「改造」了一遍。Walpurgis 将此建模为 ``AttentionNormStrategy``，
携带公式等价性证明与 fp16 溢出分析，使「为何要双侧归一化」成为可检索的知识。

其四，``loss_mask.view(-1)`` 的行内化。这是最小的改动，也是最能说明
「旧代码习惯」的一处：多写一次赋值，是因为「感觉更清楚」；
删掉这次赋值，是因为「内联才真正清楚」。两者都是主观的清晰，
却走向了相反的方向。Walpurgis 将此「中间张量消除」策略建模为 ``LossMaskReshapePolicy``，
使「何时应该内联 view」有了可查询的历史依据。

四个结构合计，Walpurgis 将上游的零碎修补升格为可程序化审计的策略知识库，
以「拿来主义」的精神：不是照单全收，而是取其精华，赋予新形。

全链路 WALPURGIS_DEBUG=1 断点 print 共 16 处：
  MODULE_LOAD × 2、CHECKPOINT_POLICY × 3、DATALOADER_FLAGS × 3、
  ATTENTION_NORM × 4、LOSS_MASK_POLICY × 2、SELF_CHECK × 2
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ── 调试开关 ─────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点：WALPURGIS_DEBUG=1 时输出结构化诊断信息。"""
    if _DEBUG:
        print(f"[eval_numeric_update_a0368ddf4] [{tag}] {msg}")


_dbg("MODULE_LOAD", "eval_numeric_update_a0368ddf4 开始加载（来自 Megatron-LM a0368ddf4）")


# ════════════════════════════════════════════════════════════════════════════
# 1. CheckpointLoadPolicy  —  evaluate_gpt2.py 改动语义
#    上游变更：eval 路径统一使用 load_checkpoint，废弃 load_checkpoint_model_parallel
# ════════════════════════════════════════════════════════════════════════════

class CheckpointLoadMode(Enum):
    """
    Checkpoint 加载接口选择模式。

    上游 evaluate_gpt2.py 在 a0368ddf4 中将 eval 路径从 MODEL_PARALLEL 切换至 STANDARD。
    历史背景：model-parallel 版本的 load_checkpoint 在 eval 场景下引入了不必要的
    rank 协商开销，而 eval 仅需读取权重，不需要 pipeline/tensor 并行语义。
    """
    STANDARD = "load_checkpoint"
    """标准接口：适用于 eval、单卡、或无需精细 MP rank 协商的场景。"""

    MODEL_PARALLEL = "load_checkpoint_model_parallel"
    """Model-parallel 接口：适用于分布式训练 checkpoint 恢复，含 TP/PP rank 协商。"""

    @property
    def is_eval_appropriate(self) -> bool:
        """eval 路径是否应选用此模式。a0368ddf4 后答案固定为 STANDARD。"""
        return self == CheckpointLoadMode.STANDARD

    @property
    def deprecated_in_eval(self) -> bool:
        """此模式是否在 eval 路径中被 a0368ddf4 废弃。"""
        return self == CheckpointLoadMode.MODEL_PARALLEL


@dataclass(frozen=True)
class CheckpointLoadPolicy:
    """
    Checkpoint 加载策略记录。

    封装 evaluate_gpt2.py 中 setup_model() 的 checkpoint 接口选择逻辑。
    上游在 a0368ddf4 前后的接口差异被显式建模，使 eval 路径的简化有可查询的历史。

    上游 diff 核心：
      - 删除 import: save_checkpoint, save_checkpoint_model_parallel,
                      load_checkpoint_model_parallel
      - setup_model(): load_checkpoint_model_parallel → load_checkpoint
    """
    mode: CheckpointLoadMode
    """当前选用的加载模式。"""

    upstream_commit: str = "a0368ddf4"
    """引入此策略简化的上游 commit SHA（前缀）。"""

    rationale: str = (
        "eval 路径不切分模型，无需 model-parallel checkpoint 协商开销；"
        "统一走 load_checkpoint 接口，消除 save_checkpoint 等无用 import。"
    )
    """选择当前模式的技术理由。"""

    deprecated_imports: tuple[str, ...] = (
        "save_checkpoint",
        "save_checkpoint_model_parallel",
        "load_checkpoint_model_parallel",
    )
    """a0368ddf4 中从 evaluate_gpt2.py 删除的 import 名称列表。"""

    def audit(self) -> dict[str, Any]:
        """输出策略审计报告。"""
        _dbg("CHECKPOINT_POLICY", f"audit(): mode={self.mode.value}, "
             f"is_eval_appropriate={self.mode.is_eval_appropriate}, "
             f"deprecated_imports={len(self.deprecated_imports)}")
        return {
            "mode": self.mode.value,
            "is_eval_appropriate": self.mode.is_eval_appropriate,
            "deprecated_in_eval": self.mode.deprecated_in_eval,
            "upstream_commit": self.upstream_commit,
            "rationale": self.rationale,
            "deprecated_imports": list(self.deprecated_imports),
        }

    @classmethod
    def for_eval(cls) -> "CheckpointLoadPolicy":
        """构造 eval 场景下的推荐策略（a0368ddf4 后）。"""
        policy = cls(mode=CheckpointLoadMode.STANDARD)
        _dbg("CHECKPOINT_POLICY", f"for_eval() → mode={policy.mode.value}, "
             f"deprecated_imports={policy.deprecated_imports}")
        return policy

    @classmethod
    def pre_a0368ddf4_eval(cls) -> "CheckpointLoadPolicy":
        """构造 a0368ddf4 之前 eval 路径的旧策略（已废弃）。"""
        return cls(
            mode=CheckpointLoadMode.MODEL_PARALLEL,
            rationale="历史遗留：eval 路径沿用 train 的 model-parallel load 接口，存在不必要开销。",
            deprecated_imports=(),
        )


_CHECKPOINT_LOAD_POLICY = CheckpointLoadPolicy.for_eval()
_dbg("CHECKPOINT_POLICY", f"全局策略已初始化：{_CHECKPOINT_LOAD_POLICY.mode.value}")


# ════════════════════════════════════════════════════════════════════════════
# 2. DataLoaderPresenceFlags  —  gpt2_data_loader.py 改动语义（do_train/do_valid/do_test）
#    上游变更：make_gpt2_dataloaders() 新增显式存在性标志；shard 回收 range 修正
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DataLoaderPresenceFlags:
    """
    DataLoader 存在性旗标。

    对应 gpt2_data_loader.py 中 a0368ddf4 新增的三行：
      args.do_train = train is not None
      args.do_valid = valid is not None
      args.do_test  = test is not None

    上游在此之前，训练/验证/测试循环是否执行完全依赖调用方隐式检查 DataLoader 是否为 None。
    引入显式旗标后，下游循环可直接通过 args.do_train 等布尔值做分支，
    消除多处重复的 ``if dataloader is not None`` 模式。

    Walpurgis 将此抽象为独立 dataclass，使旗标的语义与生成方式可独立测试。
    """
    do_train: bool
    """训练集 DataLoader 是否存在。"""

    do_valid: bool
    """验证集 DataLoader 是否存在。"""

    do_test: bool
    """测试集 DataLoader 是否存在。"""

    upstream_commit: str = field(default="a0368ddf4", compare=False)

    @classmethod
    def from_loaders(
        cls,
        train: Optional[Any],
        valid: Optional[Any],
        test: Optional[Any],
    ) -> "DataLoaderPresenceFlags":
        """
        工厂方法：由三个 DataLoader（可为 None）构造旗标对象。

        与上游 make_gpt2_dataloaders() 中新增逻辑完全等价：
          args.do_train = train is not None
          args.do_valid = valid is not None
          args.do_test  = test is not None
        """
        flags = cls(
            do_train=(train is not None),
            do_valid=(valid is not None),
            do_test=(test is not None),
        )
        _dbg(
            "DATALOADER_FLAGS",
            f"from_loaders() → do_train={flags.do_train}, "
            f"do_valid={flags.do_valid}, do_test={flags.do_test}",
        )
        return flags

    def active_splits(self) -> list[str]:
        """返回当前存在的数据集划分名称列表（用于日志/审计）。"""
        splits = []
        if self.do_train:
            splits.append("train")
        if self.do_valid:
            splits.append("valid")
        if self.do_test:
            splits.append("test")
        _dbg("DATALOADER_FLAGS", f"active_splits() → {splits}")
        return splits

    def audit(self) -> dict[str, Any]:
        """输出旗标审计报告。"""
        return {
            "do_train": self.do_train,
            "do_valid": self.do_valid,
            "do_test": self.do_test,
            "active_splits": self.active_splits(),
            "upstream_commit": self.upstream_commit,
        }


class ShardRecyclePolicy(Enum):
    """
    GPT2Dataset shard 回收策略。

    对应 gpt2_data_loader.py build_dataset_() 中的 range 修正：
      原：for i in range(shard_index - 1)  ← 保留最后一个旧 shard 防竞争
      新：for i in range(shard_index)      ← 提早回收，减少内存驻留

    上游将原行注释保留（``#for i in range(shard_index - 1)``），
    Walpurgis 将两种策略显式建模为枚举，使回收边界的选择有可查询的历史。
    """

    CONSERVATIVE = "range(shard_index - 1)"
    """保守策略（a0368ddf4 之前）：保留最近一个旧 shard，防止慢线程尚未读取时被回收。"""

    AGGRESSIVE = "range(shard_index)"
    """激进策略（a0368ddf4 之后）：立即回收已越过的全部 shard，减少内存驻留。"""

    def recycle_bound(self, shard_index: int) -> int:
        """计算应回收至哪个 shard index（exclusive）。"""
        if self == ShardRecyclePolicy.CONSERVATIVE:
            return max(0, shard_index - 1)
        return shard_index

    @property
    def introduced_in(self) -> str:
        """此策略引入/废弃的 commit 参考。"""
        if self == ShardRecyclePolicy.CONSERVATIVE:
            return "pre-a0368ddf4"
        return "a0368ddf4"


_dbg("DATALOADER_FLAGS",
     f"ShardRecyclePolicy 已加载：CONSERVATIVE={ShardRecyclePolicy.CONSERVATIVE.value}, "
     f"AGGRESSIVE={ShardRecyclePolicy.AGGRESSIVE.value}")


# ════════════════════════════════════════════════════════════════════════════
# 3. AttentionNormStrategy  —  mpu/transformer.py 改动语义（数值稳定性）
#    上游变更：注意力得分从单侧除法改为双侧归一化
# ════════════════════════════════════════════════════════════════════════════

class AttentionNormMode(Enum):
    """
    注意力得分归一化模式。

    上游 BertParallelSelfAttention.forward() 在 a0368ddf4 中切换了归一化策略：
      原：scores = (Q @ K^T) / sqrt(d)          ← 先乘后除，中间值可能溢出
      新：nf = sqrt(sqrt(d))
          scores = (Q/nf) @ (K^T/nf)           ← 双侧归一化，中间值峰值压缩

    两者数学等价（nf² = sqrt(d)，故 (Q/nf)(K^T/nf) = QK^T / sqrt(d)），
    但 fp16/bf16 精度下，先乘再除的中间值更易饱和。
    """

    POST_MATMUL_DIVIDE = "scores = (Q @ K^T) / sqrt(d)"
    """先乘后除（a0368ddf4 之前）：中间乘积数值较大，fp16 下有溢出风险。"""

    BILATERAL_NORMALIZE = "scores = (Q/√√d) @ (K^T/√√d)"
    """双侧归一化（a0368ddf4 之后）：Q 与 K 各自预除 √√d，峰值数值受控。"""

    @property
    def is_numerically_stable(self) -> bool:
        """此模式在 fp16/bf16 下是否具有更好的数值稳定性。"""
        return self == AttentionNormMode.BILATERAL_NORMALIZE

    @property
    def norm_factor_formula(self) -> str:
        """归一化因子的数学表达式（Python 风格）。"""
        if self == AttentionNormMode.POST_MATMUL_DIVIDE:
            return "math.sqrt(head_dim)"
        return "math.sqrt(math.sqrt(head_dim))"  # a0368ddf4 引入


@dataclass(frozen=True)
class AttentionNormStrategy:
    """
    注意力得分归一化策略记录。

    封装 mpu/transformer.py BertParallelSelfAttention 中归一化模式的选择逻辑，
    提供等价性证明与 fp16 溢出分析接口。

    数学等价性证明：
      令 d = hidden_size_per_attention_head
      nf = √(√d) = d^(1/4)
      (Q/nf) @ (K^T/nf) = (Q @ K^T) / nf² = (Q @ K^T) / √d  ∎
    """
    mode: AttentionNormMode
    head_dim: int
    upstream_commit: str = "a0368ddf4"

    def norm_factor(self) -> float:
        """计算当前模式下的归一化因子数值。"""
        if self.mode == AttentionNormMode.POST_MATMUL_DIVIDE:
            nf = math.sqrt(self.head_dim)
        else:
            nf = math.sqrt(math.sqrt(self.head_dim))
        _dbg("ATTENTION_NORM",
             f"norm_factor(): mode={self.mode.name}, head_dim={self.head_dim}, nf={nf:.6f}")
        return nf

    def effective_scale(self) -> float:
        """
        计算等效的最终缩放因子（两种模式应相同）。

        BILATERAL 模式：nf² = √head_dim，与 POST_MATMUL 等价。
        """
        nf = self.norm_factor()
        if self.mode == AttentionNormMode.BILATERAL_NORMALIZE:
            scale = nf * nf  # = √head_dim
        else:
            scale = nf
        _dbg("ATTENTION_NORM",
             f"effective_scale(): mode={self.mode.name}, scale={scale:.6f}, "
             f"expected=√{self.head_dim}={math.sqrt(self.head_dim):.6f}")
        return scale

    def fp16_peak_estimate(self, seq_len: int = 512) -> dict[str, float]:
        """
        估算两种归一化模式下 fp16 中间值的峰值量级。

        假设 Q/K 元素服从 N(0, 1)，QK^T 期望量级为 O(√d)，
        POST_MATMUL 模式下中间乘积峰值 ∝ d（head_dim），
        BILATERAL 模式下各因子峰值 ∝ 1（归一化后接近单位量级）。
        """
        fp16_max = 65504.0
        post_matmul_peak = float(self.head_dim) * seq_len  # 粗估
        bilateral_peak = math.sqrt(float(self.head_dim)) * seq_len  # 粗估（归一化后）
        result = {
            "fp16_max": fp16_max,
            "post_matmul_intermediate_peak_estimate": post_matmul_peak,
            "bilateral_intermediate_peak_estimate": bilateral_peak,
            "post_matmul_overflow_risk": post_matmul_peak > fp16_max * 0.5,
            "bilateral_overflow_risk": bilateral_peak > fp16_max * 0.5,
        }
        _dbg("ATTENTION_NORM",
             f"fp16_peak_estimate(seq_len={seq_len}): "
             f"post={post_matmul_peak:.0f}, bilateral={bilateral_peak:.0f}, "
             f"fp16_max={fp16_max}")
        return result

    @classmethod
    def for_bert(cls, head_dim: int) -> "AttentionNormStrategy":
        """构造 BERT 注意力层推荐策略（a0368ddf4 后）。"""
        strategy = cls(mode=AttentionNormMode.BILATERAL_NORMALIZE, head_dim=head_dim)
        _dbg("ATTENTION_NORM",
             f"for_bert(): head_dim={head_dim}, mode={strategy.mode.name}, "
             f"nf={strategy.norm_factor():.6f}")
        return strategy


# ════════════════════════════════════════════════════════════════════════════
# 4. LossMaskReshapePolicy  —  pretrain_bert.py 改动语义
#    上游变更：loss_mask.view(-1) 赋值行删除，内联至 lm_loss 表达式
# ════════════════════════════════════════════════════════════════════════════

class LossMaskReshapeStyle(Enum):
    """
    loss_mask reshape 风格。

    pretrain_bert.py forward_step() 在 a0368ddf4 中删除了中间赋值：
      原：loss_mask = loss_mask.view(-1)                     ← 独立赋值，修改原变量
          lm_loss = sum(losses.view(-1) * loss_mask.float()) / loss_mask.sum()
      新：lm_loss = sum(losses.view(-1) * loss_mask.view(-1).float()) / loss_mask.sum()
          （loss_mask 原始 shape 不被破坏，view(-1) 只在计算时内联）

    语义差异：中间赋值风格会破坏 loss_mask 的原始 shape，
    若后续还有依赖 loss_mask 原始维度的代码，会静默产生 bug；
    内联风格保持 loss_mask 不变，语义更严格。
    """

    INTERMEDIATE_ASSIGN = "loss_mask = loss_mask.view(-1)"
    """中间赋值（a0368ddf4 之前）：就地修改 loss_mask shape，存在潜在副作用。"""

    INLINE_VIEW = "loss_mask.view(-1) 内联于表达式"
    """内联视图（a0368ddf4 之后）：不修改 loss_mask，语义更严格，消除中间张量赋值。"""

    @property
    def preserves_original_shape(self) -> bool:
        """此风格是否保持 loss_mask 的原始 shape 不被修改。"""
        return self == LossMaskReshapeStyle.INLINE_VIEW

    @property
    def has_side_effect_risk(self) -> bool:
        """此风格是否存在 loss_mask shape 被意外修改的风险。"""
        return self == LossMaskReshapeStyle.INTERMEDIATE_ASSIGN


@dataclass(frozen=True)
class LossMaskReshapePolicy:
    """
    loss_mask reshape 策略记录。

    封装 pretrain_bert.py forward_step() 中 loss_mask reshape 风格的演变，
    使「为何内联」有可查询的技术理由与历史依据。
    """
    style: LossMaskReshapeStyle
    upstream_commit: str = "a0368ddf4"
    rationale: str = (
        "内联 view(-1) 保持 loss_mask 原始 shape，消除中间赋值的隐式副作用；"
        "lm_loss 表达式自洽，不依赖外部变量 shape 变更。"
    )

    def audit(self) -> dict[str, Any]:
        """输出策略审计报告。"""
        _dbg("LOSS_MASK_POLICY",
             f"audit(): style={self.style.name}, "
             f"preserves_shape={self.style.preserves_original_shape}, "
             f"side_effect_risk={self.style.has_side_effect_risk}")
        return {
            "style": self.style.value,
            "preserves_original_shape": self.style.preserves_original_shape,
            "has_side_effect_risk": self.style.has_side_effect_risk,
            "upstream_commit": self.upstream_commit,
            "rationale": self.rationale,
        }

    @classmethod
    def current(cls) -> "LossMaskReshapePolicy":
        """构造当前推荐策略（a0368ddf4 后）。"""
        policy = cls(style=LossMaskReshapeStyle.INLINE_VIEW)
        _dbg("LOSS_MASK_POLICY",
             f"current(): style={policy.style.name}, "
             f"preserves_shape={policy.style.preserves_original_shape}")
        return policy


_LOSS_MASK_POLICY = LossMaskReshapePolicy.current()


# ════════════════════════════════════════════════════════════════════════════
# 5. MigrationManifest  —  本次迁移的全局审计入口
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MigrationManifest:
    """
    a0368ddf4 迁移清单。

    汇总本次 eval+numeric update 迁移的四个策略结构，
    提供统一的审计接口与自检入口。
    """
    upstream_commit: str = "a0368ddf4"
    upstream_subject: str = "eval+numeric update"
    files_changed: int = 4
    insertions: int = 17
    deletions: int = 10

    def audit_all(self) -> dict[str, Any]:
        """输出全量审计报告。"""
        return {
            "upstream_commit": self.upstream_commit,
            "upstream_subject": self.upstream_subject,
            "diff_summary": {
                "files_changed": self.files_changed,
                "insertions": self.insertions,
                "deletions": self.deletions,
            },
            "policies": {
                "checkpoint_load": _CHECKPOINT_LOAD_POLICY.audit(),
                "loss_mask_reshape": _LOSS_MASK_POLICY.audit(),
                "shard_recycle": {
                    "before": ShardRecyclePolicy.CONSERVATIVE.value,
                    "after": ShardRecyclePolicy.AGGRESSIVE.value,
                    "introduced_in": ShardRecyclePolicy.AGGRESSIVE.introduced_in,
                },
                "attention_norm": {
                    "mode": AttentionNormMode.BILATERAL_NORMALIZE.name,
                    "formula": AttentionNormMode.BILATERAL_NORMALIZE.norm_factor_formula,
                    "is_numerically_stable": AttentionNormMode.BILATERAL_NORMALIZE.is_numerically_stable,
                },
            },
        }

    def self_check(self) -> bool:
        """运行自检断言，验证所有策略结构的一致性。"""
        _dbg("SELF_CHECK", "self_check() 开始")

        # 1. Checkpoint policy 一致性
        assert _CHECKPOINT_LOAD_POLICY.mode.is_eval_appropriate, \
            "eval 策略应选用 STANDARD 模式"
        assert not _CHECKPOINT_LOAD_POLICY.mode.deprecated_in_eval, \
            "STANDARD 模式不应在 eval 中被标记为废弃"
        assert len(_CHECKPOINT_LOAD_POLICY.deprecated_imports) == 3, \
            "a0368ddf4 共删除 3 个 import"

        # 2. Shard recycle 边界一致性
        assert ShardRecyclePolicy.AGGRESSIVE.recycle_bound(5) == 5
        assert ShardRecyclePolicy.CONSERVATIVE.recycle_bound(5) == 4

        # 3. Attention norm 数学等价性
        strategy = AttentionNormStrategy.for_bert(head_dim=64)
        nf = strategy.norm_factor()
        expected_scale = math.sqrt(64)  # = 8.0
        actual_scale = strategy.effective_scale()
        assert abs(actual_scale - expected_scale) < 1e-9, \
            f"双侧归一化等效缩放因子应为 √64=8.0，实为 {actual_scale}"

        # 4. Loss mask policy 一致性
        assert _LOSS_MASK_POLICY.style.preserves_original_shape
        assert not _LOSS_MASK_POLICY.style.has_side_effect_risk

        # 5. DataLoader 旗标工厂方法
        flags_all = DataLoaderPresenceFlags.from_loaders(
            train=object(), valid=object(), test=object()
        )
        assert flags_all.do_train and flags_all.do_valid and flags_all.do_test
        flags_none = DataLoaderPresenceFlags.from_loaders(
            train=None, valid=None, test=None
        )
        assert not flags_none.do_train
        assert flags_none.active_splits() == []

        _dbg("SELF_CHECK", "self_check() 通过：5 项断言全部验证成功")
        return True


# ── 模块加载时执行自检（DEBUG 模式下）──────────────────────────────────────
_MANIFEST = MigrationManifest()

if _DEBUG:
    _MANIFEST.self_check()

_dbg("MODULE_LOAD",
     f"eval_numeric_update_a0368ddf4 加载完成："
     f"upstream={_MANIFEST.upstream_commit}, "
     f"subject='{_MANIFEST.upstream_subject}', "
     f"policies=checkpoint_load/dataloader_flags/shard_recycle/attention_norm/loss_mask")
