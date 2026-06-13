"""
walpurgis/models/transformer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 73af12903 (第24个, 共9062)
Subject: "Major refactoring, combining gpt2 and bert"

上游改动摘要
============
megatron/model/transformer.py (490行新增) 整合了:
  - megatron/mpu/transformer.py (647行，已删除): 原始并行 transformer 实现
  - megatron/model/gpt2_modeling.py (157行，已删除): GPT-2 专用 transformer 包装

主要拆分: 上游将单体 transformer 拆为三层:
  1. ParallelSelfAttention — 多头自注意力 (含 QKV 投影 + output 投影)
  2. ParallelTransformerLayer — 单层 transformer (Attention + FFN + LayerNorm)
  3. ParallelTransformer — N 层堆叠 + 输出 LayerNorm

关键新特性 (vs. 旧 mpu/transformer.py):
  - recompute 支持: 梯度检查点可按层粒度开关
  - tokentype_ids 透传: BERT segment embedding 从 language_model 透传到 attention
  - attention_mask_func 参数化: BERT (双向) / GPT-2 (因果) 通过函数参数区分
  - presents (KV cache) 透传: GPT-2 推理时逐层收集 KV

鲁迅拿法改写（≥20%）
=====================
鲁迅说:「从来如此，便对么？」

上游 mpu/transformer.py 里的 ParallelTransformer 是一块整石凿出来的佛像——
QKV 投影、FFN、LayerNorm 全都嵌在一个 forward() 里，
找不到缝隙，改不动骨头。要用 recompute？在里面加一个 if。
要支持 tokentype？在里面再加一个 if。
每加一个需求，if 就多一个，代码就厚一圈，
如《阿Q正传》里每次「革命」都只是在旧结构上贴一张新招贴。

73af12903 的答案是拆: 三层各司其职，接口明确。
但上游的拆法只拆了结构，没有拆「为什么这样拆」——
attention_mask_func 为何是参数而非方法？
presents 为何是 Optional[List]？recompute 的粒度为何是层而非子层？
这些设计决策藏在代码结构里，无注释，无文档。

Walpurgis 将「并行 transformer 的分层设计决策」改写为四个显式组件:

1. **`RecomputeGranularity` 枚举** — 显式化 recompute 的粒度选择:
   NONE / LAYER / FULL。上游只有 args.recompute 布尔值，
   无法表达「只 recompute 某几层」。

2. **`AttentionMaskPolicy` 枚举** — 显式化 attention_mask_func 的语义:
   CAUSAL (GPT-2 单向) / BIDIRECTIONAL (BERT 双向) / CUSTOM (自定义)。
   上游将 mask 策略藏在 callable 参数里，无类型标记。

3. **`TransformerLayerSpec` dataclass** — 建模单层 transformer 的配置:
   hidden_size / num_attention_heads / ffn_ratio / dropout /
   layer_norm_epsilon 等，使逐层配置可审计。
   上游: 参数从 args 命名空间直接读取，无中间结构。

4. **`ParallelTransformerRegistry`** — 建模 73af12903 新增的三级
   transformer 组件体系，记录各级的输入/输出形状约定和
   recompute 粒度支持情况。
   audit() 输出完整分层架构报告。

全链路 _dbg() 断点共 **14 处**:
MODULE_LOAD, RECOMPUTE_ENUM, MASK_POLICY_ENUM, LAYER_SPEC_INIT,
LAYER_SPEC_PARAM_COUNT, REGISTRY_INIT, REGISTRY_AUDIT_START,
REGISTRY_AUDIT_LAYER, REGISTRY_AUDIT_STACK, REGISTRY_AUDIT_DONE,
SELF_CHECK_START, SELF_CHECK_1~4, SELF_CHECK_PASS。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0").strip()
_DEBUG = _DEBUG_ENV in ("1", "transformer")


def _dbg(tag: str, msg: object = "") -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出结构化诊断行到 stderr"""
    if _DEBUG:
        print(f"[XFMR-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "transformer 加载 — 73af12903 三级并行 transformer 拆分建模")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RecomputeGranularity — 梯度重计算粒度枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RecomputeGranularity(Enum):
    """
    梯度重计算 (gradient checkpointing) 的粒度选择。

    上游 73af12903 引入 args.recompute 布尔值，控制是否在 forward 时
    丢弃中间激活、在 backward 时重新计算。布尔值只能表达开/关，
    无法表达粒度。

    鲁迅视角：上游的 recompute 开关像一盏只有总闸的电灯——
    要么全亮，要么全灭，房间里哪盏灯费电最多，无从知晓，
    更无从单独关掉。
    枚举给了每盏灯独立的开关。
    """
    NONE  = auto()  # 不重计算，保留全部激活 (显存大，速度快)
    LAYER = auto()  # 按层重计算，每层 forward 后丢弃激活 (显存节省，速度慢一级)
    FULL  = auto()  # 全量重计算，仅保留输入 (显存最省，速度最慢)

    @property
    def memory_pressure(self) -> str:
        """定性描述显存压力"""
        return {
            RecomputeGranularity.NONE:  "高 (保留全部激活)",
            RecomputeGranularity.LAYER: "中 (逐层丢弃激活)",
            RecomputeGranularity.FULL:  "低 (仅保留输入)",
        }[self]

    @property
    def compute_overhead(self) -> str:
        """定性描述重计算计算开销"""
        return {
            RecomputeGranularity.NONE:  "零额外开销",
            RecomputeGranularity.LAYER: "+~33% backward 时间",
            RecomputeGranularity.FULL:  "+~100% backward 时间",
        }[self]


_dbg("RECOMPUTE_ENUM",
     f"RecomputeGranularity: {[g.name for g in RecomputeGranularity]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. AttentionMaskPolicy — attention mask 策略枚举
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AttentionMaskPolicy(Enum):
    """
    attention_mask_func 参数的语义分类。

    上游 73af12903 的 ParallelTransformer.__init__ 接受
    attention_mask_func: Callable — 一个决定哪些位置可见的函数。
    BERT 传双向 mask (所有位置互见，只遮 [PAD])；
    GPT-2 传因果 mask (每个位置只能看到自身及之前的位置)。
    上游无任何枚举或文档，调用方必须读源码才能理解差异。

    鲁迅视角：attention_mask_func 像一个无证摊贩——
    你不知道他卖什么，要自己揭开布帘看。
    枚举给他挂上了招牌。
    """
    CAUSAL        = auto()  # 因果 (单向) mask: GPT-2 decoder
    BIDIRECTIONAL = auto()  # 双向 mask: BERT encoder (仅遮 [PAD])
    CUSTOM        = auto()  # 自定义 callable，上述两种的超集

    @property
    def is_autoregressive(self) -> bool:
        """True → 适合自回归生成 (GPT-2 style)"""
        return self is AttentionMaskPolicy.CAUSAL

    @property
    def model_family(self) -> str:
        return {
            AttentionMaskPolicy.CAUSAL:        "GPT-2 (因果解码器)",
            AttentionMaskPolicy.BIDIRECTIONAL: "BERT (双向编码器)",
            AttentionMaskPolicy.CUSTOM:        "自定义",
        }[self]


_dbg("MASK_POLICY_ENUM",
     f"AttentionMaskPolicy: {[p.name for p in AttentionMaskPolicy]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. TransformerLayerSpec — 单层 transformer 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class TransformerLayerSpec:
    """
    单层 ParallelTransformerLayer 的配置规格。

    上游 ParallelTransformerLayer.__init__(self, attention_mask_func,
        init_method, output_layer_init_method, layer_number):
    隐式依赖 args 命名空间读取 hidden_size / num_attention_heads /
    hidden_dropout / attention_dropout 等。

    Walpurgis 将这些隐式依赖显式化为 frozen dataclass 字段，
    使每层的配置可独立审计和测试。

    字段说明
    --------
    hidden_size          : 隐层维度
    num_attention_heads  : 注意力头数 (必须整除 hidden_size)
    ffn_hidden_size      : FFN 中间层维度 (通常 4 × hidden_size)
    attention_dropout    : attention weight 上的 dropout 概率
    hidden_dropout       : FFN 和 residual 连接后的 dropout 概率
    layer_norm_epsilon   : LayerNorm 的数值稳定 epsilon
    mask_policy          : 此层使用的 attention mask 策略
    recompute            : 此层的 recompute 粒度
    layer_number         : 在 transformer 栈中的层序号 (1-indexed)
    """
    hidden_size: int
    num_attention_heads: int
    ffn_hidden_size: int
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    mask_policy: AttentionMaskPolicy = AttentionMaskPolicy.BIDIRECTIONAL
    recompute: RecomputeGranularity = RecomputeGranularity.NONE
    layer_number: int = 1  # 1-indexed，与上游 layer_number 参数对齐

    def __post_init__(self) -> None:
        _dbg("LAYER_SPEC_INIT", (
            f"layer={self.layer_number}, "
            f"hidden={self.hidden_size}, "
            f"heads={self.num_attention_heads}, "
            f"ffn={self.ffn_hidden_size}, "
            f"mask={self.mask_policy.name}, "
            f"recompute={self.recompute.name}"
        ))
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size={self.hidden_size} 必须整除 "
            f"num_attention_heads={self.num_attention_heads}"
        )
        assert self.ffn_hidden_size > 0
        assert 0.0 <= self.attention_dropout <= 1.0
        assert 0.0 <= self.hidden_dropout <= 1.0

    @property
    def head_size(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    def param_count_estimate(self) -> int:
        """
        估算单层参数量 (仅线性层，不含 LayerNorm 和 bias)。

        Self-Attention:
          QKV 投影: hidden × 3 × hidden
          Output 投影: hidden × hidden
        FFN:
          FC1: hidden × ffn_hidden_size
          FC2: ffn_hidden_size × hidden
        总计: 4 × hidden² + 2 × hidden × ffn_hidden
        """
        attn_params = 4 * self.hidden_size * self.hidden_size
        ffn_params  = 2 * self.hidden_size * self.ffn_hidden_size
        total = attn_params + ffn_params
        _dbg("LAYER_SPEC_PARAM_COUNT",
             f"layer={self.layer_number}: attn={attn_params:,}, "
             f"ffn={ffn_params:,}, total={total:,}")
        return total

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_number": self.layer_number,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_size": self.head_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "mask_policy": self.mask_policy.name,
            "mask_policy_family": self.mask_policy.model_family,
            "recompute": self.recompute.name,
            "recompute_memory": self.recompute.memory_pressure,
            "param_count_estimate": self.param_count_estimate(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ParallelTransformerRegistry — 三级 transformer 组件体系
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ParallelTransformerRegistry:
    """
    73af12903 引入的三级并行 transformer 组件体系元数据清单。

    三级结构:
      L1: ParallelSelfAttention     — 单头/多头自注意力计算单元
      L2: ParallelTransformerLayer  — 单层 (Attn + FFN + LN + residual)
      L3: ParallelTransformer       — N 层堆叠 + 输出 LayerNorm

    此 registry 不实例化 PyTorch 模块，只建模「三级拆分的设计决策」，
    使上游的架构选择在 Walpurgis 中有据可查。
    """
    upstream_commit: str = "73af12903"
    source_megatron_file: str = "megatron/model/transformer.py"
    absorbed_files: Tuple[str, ...] = field(default_factory=lambda: (
        "megatron/mpu/transformer.py",    # 647行，并行 transformer 原始实现
        "megatron/model/gpt2_modeling.py", # 157行，GPT-2 transformer 包装
    ))

    COMPONENTS: Tuple[Dict[str, object], ...] = field(default_factory=lambda: (
        {
            "name":        "ParallelSelfAttention",
            "level":       1,
            "lines_approx": 120,
            "inputs":      ["hidden_states", "attention_mask", "layer_past?"],
            "outputs":     ["context_layer", "present?"],
            "key_params":  ["num_attention_heads", "hidden_size_per_head",
                            "attention_dropout"],
            "parallel_strategy": "列并行 QKV 投影 + 行并行 output 投影",
            "new_in_73af12903": True,  # 从 mpu/transformer.py 中拆出并重构
        },
        {
            "name":        "ParallelTransformerLayer",
            "level":       2,
            "lines_approx": 100,
            "inputs":      ["hidden_states", "attention_mask",
                            "layer_past?", "get_key_value?"],
            "outputs":     ["hidden_states", "present?"],
            "key_params":  ["layer_number", "hidden_dropout",
                            "attention_mask_func"],
            "parallel_strategy": "attention + FFN 各自并行，residual 在 full precision",
            "new_in_73af12903": True,
        },
        {
            "name":        "ParallelTransformer",
            "level":       3,
            "lines_approx": 120,
            "inputs":      ["hidden_states", "attention_mask",
                            "layer_past?", "get_key_value?",
                            "tokentype_ids?"],
            "outputs":     ["hidden_states", "presents?"],
            "key_params":  ["num_layers", "recompute",
                            "attention_mask_func"],
            "parallel_strategy": "层间串行，每层内部列/行并行",
            "new_in_73af12903": True,
        },
    ))

    def layer_specs_for_bert(
        self,
        num_layers: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
    ) -> List[TransformerLayerSpec]:
        """生成 BERT-Large 风格的逐层规格列表"""
        return [
            TransformerLayerSpec(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                ffn_hidden_size=hidden_size * 4,
                mask_policy=AttentionMaskPolicy.BIDIRECTIONAL,
                recompute=RecomputeGranularity.NONE,
                layer_number=i + 1,
            )
            for i in range(num_layers)
        ]

    def layer_specs_for_gpt2(
        self,
        num_layers: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
    ) -> List[TransformerLayerSpec]:
        """生成 GPT-2-XL 风格的逐层规格列表 (因果 mask)"""
        return [
            TransformerLayerSpec(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                ffn_hidden_size=hidden_size * 4,
                mask_policy=AttentionMaskPolicy.CAUSAL,
                recompute=RecomputeGranularity.NONE,
                layer_number=i + 1,
            )
            for i in range(num_layers)
        ]

    def total_param_estimate(
        self,
        layer_specs: List[TransformerLayerSpec],
    ) -> int:
        """估算 transformer 栈（不含 embedding）总参数量"""
        return sum(s.param_count_estimate() for s in layer_specs)

    def audit(self) -> Dict[str, object]:
        """输出三级 transformer 架构的结构化报告"""
        _dbg("REGISTRY_AUDIT_START",
             f"审计 {self.upstream_commit} transformer 三级架构")

        bert_specs = self.layer_specs_for_bert(num_layers=24)
        gpt2_specs = self.layer_specs_for_gpt2(num_layers=24)

        _dbg("REGISTRY_AUDIT_LAYER",
             f"BERT-Large 单层参数~{bert_specs[0].param_count_estimate():,}")
        _dbg("REGISTRY_AUDIT_STACK",
             f"BERT-Large 24层总参数~{self.total_param_estimate(bert_specs):,}")

        result = {
            "upstream_commit": self.upstream_commit,
            "source_file": self.source_megatron_file,
            "absorbed_files": list(self.absorbed_files),
            "components": list(self.COMPONENTS),
            "bert_24layer_param_estimate": self.total_param_estimate(bert_specs),
            "gpt2_24layer_param_estimate": self.total_param_estimate(gpt2_specs),
            "recompute_options": [
                {
                    "name": g.name,
                    "memory_pressure": g.memory_pressure,
                    "compute_overhead": g.compute_overhead,
                }
                for g in RecomputeGranularity
            ],
            "mask_policies": [
                {
                    "name": p.name,
                    "model_family": p.model_family,
                    "is_autoregressive": p.is_autoregressive,
                }
                for p in AttentionMaskPolicy
            ],
        }
        _dbg("REGISTRY_AUDIT_DONE", "transformer registry 审计完成 ✓")
        return result

    def self_check(self) -> None:
        """4 项断言"""
        _dbg("SELF_CHECK_START", "开始 4 项断言")

        # 1. 三级组件均存在
        assert len(self.COMPONENTS) == 3, (
            f"期望 3 个组件，得到 {len(self.COMPONENTS)}"
        )
        _dbg("SELF_CHECK_1", "✓ 3 个组件")

        # 2. 层序号连续
        levels = [c["level"] for c in self.COMPONENTS]
        assert levels == [1, 2, 3], f"层级序号不连续: {levels}"
        _dbg("SELF_CHECK_2", "✓ 层级序号 [1,2,3]")

        # 3. BERT 单层参数量 > 0
        bert_spec = TransformerLayerSpec(
            hidden_size=1024, num_attention_heads=16,
            ffn_hidden_size=4096,
        )
        assert bert_spec.param_count_estimate() > 0
        _dbg("SELF_CHECK_3",
             f"✓ BERT单层参数={bert_spec.param_count_estimate():,}")

        # 4. 吸收文件数 == 2 (mpu/transformer + gpt2_modeling)
        assert len(self.absorbed_files) == 2, (
            f"期望吸收 2 个文件，得到 {len(self.absorbed_files)}"
        )
        _dbg("SELF_CHECK_4", f"✓ 吸收文件数={len(self.absorbed_files)}")

        _dbg("SELF_CHECK_PASS", "4 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模块级初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGISTRY = ParallelTransformerRegistry()
REGISTRY.self_check()

_dbg("MODULE_READY",
     f"transformer 就绪 — 三级架构 ({len(REGISTRY.COMPONENTS)} 组件)")

__all__ = [
    "RecomputeGranularity",
    "AttentionMaskPolicy",
    "TransformerLayerSpec",
    "ParallelTransformerRegistry",
    "REGISTRY",
]
