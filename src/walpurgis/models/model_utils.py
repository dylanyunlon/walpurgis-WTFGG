"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/model/utils.py（新增文件，80行）+ megatron/module.py（新增文件，34行）

鲁迅拿法改写（≥20%）：
  上游 model/utils.py 只做了一件事：定义了
    get_params_for_weight_decay_optimization()
  这个函数从模型参数里分出两组：
    - 需要 weight decay 的（weight，非 bias，非 LayerNorm）
    - 不需要 weight decay 的（bias，LayerNorm.weight/bias）
  上游注释：零。Walpurgis 将此逻辑命名化，并给出每个判断条件的语义说明。

  megatron/module.py（34行）定义了 MegatronModule（nn.Module 子类），
  主要功能：state_dict_for_save_checkpoint / load_state_dict_from_checkpoint，
  把 fp16 权重的处理挂在 Module 层级。
  上游：这两个方法基本是 nn.Module 的透传加一个 fp16 包裹逻辑。
  Walpurgis：将 fp16 权重保存/加载策略独立为 Fp16CheckpointStrategy，
  与主模型解耦。

迁移位置: src/walpurgis/models/model_utils.py
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：model_utils 关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    print(f"[_dbg:model_utils:{tag}] {msg}", file=sys.stderr, flush=True)


# ── WeightDecayParamGroups ────────────────────────────────────────────────────
# 对应上游 get_params_for_weight_decay_optimization()。
# 上游：返回 [{"weight_decay": wd, "params": [...]}, {"weight_decay": 0, "params": [...]}]
# Walpurgis：封装为数据类 + 独立的分类逻辑，条件判断有注释。

@dataclass
class WeightDecayParamGroups:
    """
    将模型参数分组，区分需要 / 不需要 weight decay 的参数。

    不需要 weight decay 的参数（上游实现，Walpurgis 补文档）：
      - bias（任何层的偏置项）
      - LayerNorm 的 weight 和 bias（归一化层参数不应受 L2 正则化约束）
      - Embedding 的 weight（词向量通常不加 weight decay）

    需要 weight decay 的参数：
      - 上述以外的所有 weight（线性层、注意力投影等）
    """
    decay_params: List[torch.Tensor] = field(default_factory=list)
    no_decay_params: List[torch.Tensor] = field(default_factory=list)
    # 名称列表，用于 _dbg 输出
    decay_names: List[str] = field(default_factory=list)
    no_decay_names: List[str] = field(default_factory=list)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        weight_decay: float = 0.01,
    ) -> "WeightDecayParamGroups":
        """
        从模型提取两组参数。

        对应上游：
            param_groups = get_params_for_weight_decay_optimization(model)
        Walpurgis：每个判断条件附带语义说明。
        """
        groups = cls()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                _dbg("SKIP_FROZEN", name)
                continue

            # 判断是否免于 weight decay
            is_no_decay = (
                # 条件1：bias 参数（任意层）
                "bias" in name
                # 条件2：LayerNorm 的可学习参数（weight / bias）
                or "layernorm" in name.lower()
                or "layer_norm" in name.lower()
                # 条件3：embedding weight（词向量不加 L2）
                or "embeddings" in name.lower()
                or "embedding.weight" in name.lower()
            )

            if is_no_decay:
                groups.no_decay_params.append(param)
                groups.no_decay_names.append(name)
                _dbg("NO_DECAY", f"{name}  shape={list(param.shape)}")
            else:
                groups.decay_params.append(param)
                groups.decay_names.append(name)
                _dbg("DECAY", f"{name}  shape={list(param.shape)}")

        _dbg("SUMMARY",
             f"decay={len(groups.decay_params)} params, "
             f"no_decay={len(groups.no_decay_params)} params")
        return groups

    def to_optimizer_param_groups(
        self, weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        返回 PyTorch optimizer 接受的 param_groups 格式。

        上游返回格式：
            [{"weight_decay": wd, "params": [...]},
             {"weight_decay": 0.0, "params": [...]}]
        """
        groups = [
            {"weight_decay": weight_decay, "params": self.decay_params},
            {"weight_decay": 0.0,          "params": self.no_decay_params},
        ]
        _dbg("PARAM_GROUPS",
             f"group[0](decay): {len(self.decay_params)} params, wd={weight_decay}; "
             f"group[1](no_decay): {len(self.no_decay_params)} params, wd=0.0")
        return groups


def get_params_for_weight_decay_optimization(
    model: nn.Module,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    上游接口兼容函数。

    对应 megatron/model/utils.py:get_params_for_weight_decay_optimization()。
    Walpurgis 内部使用 WeightDecayParamGroups，此函数保留上游调用接口。
    """
    _dbg("COMPAT_CALL", "get_params_for_weight_decay_optimization()")
    groups = WeightDecayParamGroups.from_model(model, weight_decay)
    return groups.to_optimizer_param_groups(weight_decay)


# ── ModelParamStats ───────────────────────────────────────────────────────────
# 上游 megatron/utils.py 有 get_parameters_in_millions()，
# Walpurgis 扩展为完整的参数统计报告（对应本次 commit 73af12903 的 utils.py 改动）。

@dataclass
class ModelParamStats:
    """模型参数统计（上游只有 get_parameters_in_millions，Walpurgis 扩展）"""
    total_params: int
    trainable_params: int
    frozen_params: int
    total_params_millions: float
    trainable_params_millions: float

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelParamStats":
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        _dbg("PARAM_STATS",
             f"total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
        return cls(
            total_params=total,
            trainable_params=trainable,
            frozen_params=frozen,
            total_params_millions=total / 1e6,
            trainable_params_millions=trainable / 1e6,
        )

    def report(self) -> str:
        return (
            f"Parameters: {self.total_params_millions:.2f}M total, "
            f"{self.trainable_params_millions:.2f}M trainable, "
            f"{self.frozen_params / 1e6:.2f}M frozen"
        )


# ── Fp16CheckpointStrategy ────────────────────────────────────────────────────
# 上游 megatron/module.py 的 MegatronModule：
#   state_dict_for_save_checkpoint → fp16 包裹 state_dict
#   load_state_dict_from_checkpoint → 解包 fp16
# Walpurgis：将此策略从 Module 继承链中解耦，作为独立策略对象。

class Fp16CheckpointStrategy:
    """
    FP16 模型的 checkpoint 保存/加载策略（对应上游 MegatronModule 两个方法）。

    上游 MegatronModule 通过继承 nn.Module 覆盖 state_dict()，
    在保存时将 fp16 权重转为 fp32（防止 checkpoint 精度丢失）。
    Walpurgis：策略对象，调用者显式使用，而非隐式继承。

    FP16→FP32 保存原因（上游无注释，Walpurgis 补全）：
      - fp16 精度有限，checkpoint 以 fp32 存储，恢复训练时精度更稳定
      - 与 FP16_Module 包装下的 master weight 机制一致
    """

    @staticmethod
    def state_dict_for_save(
        model: nn.Module,
        destination: Optional[Dict] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        将 fp16 权重转为 fp32 后返回 state_dict（保存时使用）。
        对应上游 MegatronModule.state_dict_for_save_checkpoint()。
        """
        sd = model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        fp32_sd = {}
        converted = 0
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                fp32_sd[k] = v.float()
                converted += 1
            else:
                fp32_sd[k] = v
        _dbg("SAVE_CKPT", f"state_dict keys={len(sd)}, fp16→fp32 converted={converted}")
        return fp32_sd

    @staticmethod
    def load_state_dict_from_checkpoint(
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """
        从 checkpoint state_dict 加载权重（fp32 → 模型当前 dtype）。
        对应上游 MegatronModule.load_state_dict_from_checkpoint()。
        """
        _dbg("LOAD_CKPT", f"state_dict keys={len(state_dict)}, strict={strict}")
        model.load_state_dict(state_dict, strict=strict)
        _dbg("LOAD_CKPT_DONE", "load_state_dict complete")


# ── MegatronModuleMixin ───────────────────────────────────────────────────────
# 上游 MegatronModule 是 nn.Module 子类；
# Walpurgis 改为 Mixin，允许任意 nn.Module 子类通过多继承获得此行为，
# 而不强制单一继承链（上游设计的主要限制）。

class MegatronModuleMixin:
    """
    Megatron 模型 Mixin（对应上游 megatron/module.py 的 MegatronModule）。

    使用方式（Walpurgis）：
        class MyModel(nn.Module, MegatronModuleMixin): ...

    上游使用方式：
        class BertModel(MegatronModule): ...  # MegatronModule 继承自 nn.Module

    Walpurgis 改为 Mixin 的原因：
      Python 单继承链在深度封装时容易产生 MRO 冲突；
      Mixin 模式允许功能正交组合。
    """

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ) -> Dict[str, torch.Tensor]:
        """上游接口兼容方法，委托给 Fp16CheckpointStrategy"""
        _dbg("MIXIN_SAVE", f"prefix={prefix!r}")
        return Fp16CheckpointStrategy.state_dict_for_save(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict_from_checkpoint(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """上游接口兼容方法，委托给 Fp16CheckpointStrategy"""
        _dbg("MIXIN_LOAD", f"keys={len(state_dict)}, strict={strict}")
        Fp16CheckpointStrategy.load_state_dict_from_checkpoint(self, state_dict, strict=strict)
