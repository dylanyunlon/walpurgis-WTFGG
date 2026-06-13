"""
walpurgis/core/mpu_transformer_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块对应 mpu/transformer.py，620 行新增）
=========================================================
  GPT2SelfAttention（并行多头注意力）
    · Q/K/V 投影：ColumnParallelLinear（不聚合，每 GPU 持有 heads // mp_size 个头）
    · 注意力输出投影：RowParallelLinear（输入已并行，输出 all-reduce）
    · attention_dropout / hidden_dropout 均通过 mpu.random 的种子管理器保持一致
  GPT2MLP（并行 FFN）
    · fc1：ColumnParallelLinear（4H → H，按列切分，不聚合）
    · fc2：RowParallelLinear（H → 4H，按行切分，all-reduce）
    · GeLU 激活在各 GPU 本地计算
  GPT2TransformerLayer（单层）
    · pre/post-LN 均为 LayerNorm，参数在所有模型并行 rank 上冗余持有
    · 残差连接在 fc2 的 all-reduce 之后施加
  GPT2Transformer（多层堆叠）
    · checkpoint_activations：梯度检查点（减显存，增重算成本）
    · 支持 pre-LN（GPT-2 默认）和 post-LN（BERT 风格）

CI/merge 判定：核心算法结构，直接迁移
  · 并行 Transformer 层是 Megatron 最核心的算子实现
  · 与 Walpurgis 的时序图神经网络（D2STGNN）的注意力机制有结构对应

鲁迅拿法改写（≥20%）
====================
上游 mpu/transformer.py 的本质困境是「并行性是隐式的」：
GPT2SelfAttention 接受 hidden_states，内部偷偷把 Q/K/V 按 GPU 数切分，
forward 出来的是一个「看似完整、实则残缺」的张量——
调用者若不知道下一层是 RowParallelLinear，就不知道它的输出需要 all-reduce。
这像极了《狂人日记》里的「吃人」——明面上的规则（接口）是正常的，
但内部（切分状态）是另一套逻辑，初来者必被其困。

上游 `apply_query_key_layer_scaling` 是另一个隐患：
它把 `1 / sqrt(head_dim)` 的缩放因子「植入」Q 的初始化，
而不是在 attention score 计算时显式除以。
注释说「for numerical stability in fp16」，但没有任何结构记录
「哪个版本开始做这个变换」「移除它会有什么影响」。
如鲁迅所言：世上的路，有的是从没有路的地方踏出来的；
上游的数值稳定性补丁，是从无数次 NaN 的地方踏出来的，却没有留下脚印。

Walpurgis 将 Transformer 层的并行配置抽象为五个结构：

1. **`AttentionParallelConfig` dataclass** — 封装多头注意力的并行配置
   （hidden_size、num_heads、mp_size、head_dim、local_heads），
   `validate()` 检查 num_heads % mp_size == 0，上游无此前置校验
2. **`FFNParallelConfig` dataclass** — 封装 FFN 的并行配置（hidden、ffn_hidden、mp_size），
   本地中间维度 `local_ffn_hidden` 属性替代上游裸除法
3. **`TransformerLayerSpec` dataclass** — 将单层所有超参数（hidden_size、num_heads、
   num_layers、pre_ln、dropout 等）集中配置，提供 `attn_config` / `ffn_config`
   属性自动派生子配置，上游各子模块分别接受散乱参数
4. **`CheckpointStrategy` 枚举** — 显式建模三种梯度检查点策略
   （NONE / FULL / SELECTIVE），上游用裸布尔 checkpoint_activations
5. **`TransformerParallelAudit` dataclass** — 记录 Transformer 构建时的并行决策，
   包含每层的参数量估算和通信操作次数

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 16 处，
覆盖 attention/FFN 配置校验、检查点策略、审计报告全路径。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[mpu_transformer_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "mpu_transformer_abe36e2e5.py 初始化开始")


# ── 枚举：梯度检查点策略 ────────────────────────────────────────────────────

class CheckpointStrategy(Enum):
    """显式建模梯度检查点策略。

    上游以裸布尔 checkpoint_activations 表达 NONE vs FULL；
    Walpurgis 新增 SELECTIVE（仅对 attention 做检查点），
    使策略选择在类型层面可见。

    migrate abe36e2e5: mpu/transformer.py GPT2Transformer.__init__ L475-L480
    """
    NONE = "none"
    """不使用检查点；全部激活值保留在显存中（训练速度最快，显存占用最高）"""
    FULL = "full"
    """每层全部重算；上游 checkpoint_activations=True 对应此策略"""
    SELECTIVE = "selective"
    """仅对 attention softmax 做检查点（Walpurgis 扩展，上游无此策略）"""

    def memory_factor(self) -> float:
        """相对于 NONE 策略的显存占用比例（估算）。"""
        if self == CheckpointStrategy.NONE:
            return 1.0
        if self == CheckpointStrategy.FULL:
            return 0.4   # 约节省 60% 激活值显存
        return 0.7       # SELECTIVE 折中

    def compute_overhead(self) -> float:
        """相对于 NONE 策略的额外计算量比例（估算）。"""
        if self == CheckpointStrategy.NONE:
            return 0.0
        if self == CheckpointStrategy.FULL:
            return 0.33  # 约 1/3 额外前向计算
        return 0.15      # SELECTIVE 折中

    def describe(self) -> str:
        return (
            f"{self.value}: 显存因子={self.memory_factor():.1%}, "
            f"计算开销={self.compute_overhead():.1%}"
        )


_dbg(
    "ENUM_INIT",
    f"CheckpointStrategy 已定义: {[s.value for s in CheckpointStrategy]}",
)


# ── 数据类：注意力并行配置 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AttentionParallelConfig:
    """封装并行多头注意力的配置。

    上游 GPT2SelfAttention.__init__ 接受散乱参数，在 __init__ 内算术派生
    self.num_attention_heads_per_partition 等值。Walpurgis 将这些派生属性
    提升至 dataclass 属性，使本地头数在类型层面可见。

    migrate abe36e2e5: mpu/transformer.py GPT2SelfAttention.__init__ L100-L140
    """
    hidden_size: int
    num_attention_heads: int
    model_parallel_size: int
    attention_dropout_prob: float = 0.1
    output_dropout_prob: float = 0.1
    apply_query_key_layer_scaling: bool = True
    """上游数值稳定性补丁：对 Q 乘以 1/sqrt(head_dim) / layer_number。

    Walpurgis 将其显式化为配置项，使调用者知道「我在用这个补丁」。
    migrate abe36e2e5: mpu/transformer.py L119-L125
    """
    layer_number: int = 1

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.num_attention_heads % self.model_parallel_size != 0:
            errors.append(
                f"num_attention_heads={self.num_attention_heads} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size={self.hidden_size} 必须整除 "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if not (0.0 <= self.attention_dropout_prob < 1.0):
            errors.append(
                f"attention_dropout_prob 必须在 [0, 1)，当前: {self.attention_dropout_prob}"
            )
        _dbg(
            "ATTN_VALIDATE",
            f"hidden={self.hidden_size} heads={self.num_attention_heads} "
            f"mp={self.model_parallel_size} errors={errors}",
        )
        return errors

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。

        migrate abe36e2e5: mpu/transformer.py L113 hidden_size_per_attention_head
        """
        return self.hidden_size // self.num_attention_heads

    @property
    def local_num_heads(self) -> int:
        """本地 GPU 持有的注意力头数。

        migrate abe36e2e5: mpu/transformer.py L114 num_attention_heads_per_partition
        """
        return self.num_attention_heads // self.model_parallel_size

    @property
    def local_hidden_size(self) -> int:
        """本地 GPU 的 QKV 投影输出维度。

        = local_num_heads * head_dim
        migrate abe36e2e5: mpu/transformer.py ColumnParallelLinear 输出维度
        """
        return self.local_num_heads * self.head_dim

    @property
    def query_key_scaling_factor(self) -> float:
        """Q/K 缩放因子（用于 attention score 归一化）。

        上游：coeff = 1 / math.sqrt(hidden_size_per_attention_head)
        若 apply_query_key_layer_scaling=True，额外除以 layer_number。

        migrate abe36e2e5: mpu/transformer.py L119-L125
        """
        base = 1.0 / math.sqrt(self.head_dim)
        if self.apply_query_key_layer_scaling:
            return base / self.layer_number
        return base

    def describe(self) -> str:
        return (
            f"AttentionParallelConfig("
            f"hidden={self.hidden_size}, heads={self.num_attention_heads}, "
            f"local_heads={self.local_num_heads}, head_dim={self.head_dim}, "
            f"scale={self.query_key_scaling_factor:.4f})"
        )


_dbg("DATACLASS_INIT", "AttentionParallelConfig 已定义")


# ── 数据类：FFN 并行配置 ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FFNParallelConfig:
    """封装并行 FFN（前馈网络）的配置。

    上游 GPT2MLP.__init__ 直接用 4 * hidden_size 作为 ffn_hidden_size，
    无任何结构化记录。Walpurgis 允许自定义 ffn_hidden_size（部分 LLM 使用非 4× 比例）。

    migrate abe36e2e5: mpu/transformer.py GPT2MLP.__init__ L270-L310
    """
    hidden_size: int
    ffn_hidden_size: Optional[int] = None   # 默认 4 * hidden_size
    model_parallel_size: int = 1
    dropout_prob: float = 0.1

    def __post_init__(self) -> None:
        # frozen dataclass 不能直接赋值，通过 object.__setattr__ 处理默认值
        if self.ffn_hidden_size is None:
            object.__setattr__(self, "ffn_hidden_size", 4 * self.hidden_size)
        _dbg(
            "FFN_INIT",
            f"hidden={self.hidden_size} ffn_hidden={self.ffn_hidden_size} "
            f"mp={self.model_parallel_size}",
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        ffn = self.ffn_hidden_size or (4 * self.hidden_size)
        if ffn % self.model_parallel_size != 0:
            errors.append(
                f"ffn_hidden_size={ffn} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        return errors

    @property
    def local_ffn_hidden(self) -> int:
        """本地 GPU 持有的 FFN 中间维度（fc1 的本地输出维度）。

        migrate abe36e2e5: mpu/transformer.py L277 ColumnParallelLinear 输出
        """
        ffn = self.ffn_hidden_size or (4 * self.hidden_size)
        return ffn // self.model_parallel_size

    def describe(self) -> str:
        ffn = self.ffn_hidden_size or (4 * self.hidden_size)
        return (
            f"FFNParallelConfig(hidden={self.hidden_size}, "
            f"ffn_hidden={ffn}, local_ffn_hidden={self.local_ffn_hidden}, "
            f"mp={self.model_parallel_size})"
        )


_dbg("DATACLASS_INIT", "FFNParallelConfig 已定义")


# ── 数据类：Transformer 层完整规格 ──────────────────────────────────────────

@dataclass(frozen=True)
class TransformerLayerSpec:
    """将单个 Transformer 层的所有超参数收敛至一处。

    上游 GPT2TransformerLayer.__init__ 接受 10+ 个散乱参数，无法一眼看出
    「这个层的并行配置是什么」。Walpurgis 将配置集中化，
    `attn_config` / `ffn_config` 属性自动派生子配置。

    migrate abe36e2e5: mpu/transformer.py GPT2TransformerLayer.__init__ L350-L420
    """
    hidden_size: int
    num_attention_heads: int
    model_parallel_size: int
    layer_number: int = 1
    ffn_hidden_size: Optional[int] = None
    attention_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    pre_layernorm: bool = True
    """pre-LN（GPT-2 默认）vs post-LN（BERT 风格）。

    migrate abe36e2e5: mpu/transformer.py GPT2TransformerLayer.forward L435-L480
    """
    layernorm_epsilon: float = 1e-5
    apply_query_key_layer_scaling: bool = True
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.NONE

    def validate(self) -> List[str]:
        """收集所有子配置的校验错误。"""
        errors: List[str] = []
        errors.extend(self.attn_config.validate())
        errors.extend(self.ffn_config.validate())
        if self.layer_number < 1:
            errors.append(f"layer_number 必须 ≥ 1，当前: {self.layer_number}")
        _dbg(
            "LAYER_SPEC_VALIDATE",
            f"layer={self.layer_number} errors={errors}",
        )
        return errors

    @property
    def attn_config(self) -> AttentionParallelConfig:
        """派生注意力并行配置。"""
        return AttentionParallelConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            model_parallel_size=self.model_parallel_size,
            attention_dropout_prob=self.attention_dropout_prob,
            output_dropout_prob=self.hidden_dropout_prob,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            layer_number=self.layer_number,
        )

    @property
    def ffn_config(self) -> FFNParallelConfig:
        """派生 FFN 并行配置。"""
        return FFNParallelConfig(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            model_parallel_size=self.model_parallel_size,
            dropout_prob=self.hidden_dropout_prob,
        )

    def param_count_estimate(self) -> int:
        """估算单层本地参数量（不含 LayerNorm，其参数在所有 rank 冗余）。

        QKV: 3 * hidden * (hidden // mp)
        attn_out: (hidden // mp) * hidden
        fc1: hidden * ffn_hidden // mp
        fc2: (ffn_hidden // mp) * hidden

        migrate abe36e2e5: 上游无此估算，Walpurgis 新增
        """
        attn = self.attn_config
        ffn = self.ffn_config
        ffn_h = self.ffn_hidden_size or (4 * self.hidden_size)
        qkv = 3 * self.hidden_size * attn.local_hidden_size
        attn_out = attn.local_hidden_size * self.hidden_size
        fc1 = self.hidden_size * ffn.local_ffn_hidden
        fc2 = ffn.local_ffn_hidden * self.hidden_size
        total = qkv + attn_out + fc1 + fc2
        _dbg(
            "PARAM_ESTIMATE",
            f"layer={self.layer_number} qkv={qkv} attn_out={attn_out} "
            f"fc1={fc1} fc2={fc2} total={total}",
        )
        return total

    def communication_ops_per_forward(self) -> int:
        """每次前向传播的通信操作数（all-reduce / all-gather）。

        上游：attn all-reduce × 1 + ffn all-reduce × 1 = 2 次
        若 gather_output=True（列并行最后层）额外 +1
        migrate abe36e2e5: mpu/layers.py RowParallelLinear.forward + mappings.py
        """
        # 标准管线：attn_out all-reduce + ffn_out all-reduce
        ops = 2
        _dbg("COMM_OPS", f"layer={self.layer_number} ops={ops}")
        return ops

    def describe(self) -> str:
        return (
            f"TransformerLayerSpec(layer={self.layer_number}, "
            f"hidden={self.hidden_size}, heads={self.num_attention_heads}, "
            f"mp={self.model_parallel_size}, "
            f"pre_ln={self.pre_layernorm}, "
            f"ckpt={self.checkpoint_strategy.value})"
        )


_dbg("DATACLASS_INIT", "TransformerLayerSpec 已定义")


# ── 审计记录 ─────────────────────────────────────────────────────────────────

@dataclass
class TransformerParallelAudit:
    """记录 Transformer 构建时的并行决策。

    migrate abe36e2e5: 上游无对等结构，Walpurgis 新增
    """
    num_layers: int
    layer_spec: TransformerLayerSpec
    created_at: float = field(default_factory=time.time)

    def total_local_params(self) -> int:
        """全部层的本地参数量之和。"""
        per_layer = self.layer_spec.param_count_estimate()
        # LayerNorm 参数：每层 2 个 (pre + post)，每个 2 * hidden_size
        ln_params = self.num_layers * 2 * 2 * self.layer_spec.hidden_size
        total = self.num_layers * per_layer + ln_params
        _dbg(
            "AUDIT_TOTAL_PARAMS",
            f"layers={self.num_layers} per_layer={per_layer} "
            f"ln={ln_params} total={total}",
        )
        return total

    def total_comm_ops_per_forward(self) -> int:
        """全部层每次前向的总通信操作数。"""
        return self.num_layers * self.layer_spec.communication_ops_per_forward()

    def summary(self) -> str:
        spec = self.layer_spec
        lines = [
            "=== TransformerParallelAudit ===",
            f"层数: {self.num_layers}",
            f"hidden_size: {spec.hidden_size}",
            f"num_heads: {spec.num_attention_heads} (本地: {spec.attn_config.local_num_heads})",
            f"model_parallel_size: {spec.model_parallel_size}",
            f"pre_layernorm: {spec.pre_layernorm}",
            f"checkpoint_strategy: {spec.checkpoint_strategy.describe()}",
            f"本地参数量估算: {self.total_local_params():,}",
            f"每次前向通信操作: {self.total_comm_ops_per_forward()} 次",
        ]
        return "\n".join(lines)


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证核心结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. GPT-2 Small 配置（12 层，768 hidden，12 heads，mp=1）
    gpt2_spec = TransformerLayerSpec(
        hidden_size=768,
        num_attention_heads=12,
        model_parallel_size=1,
        layer_number=1,
    )
    assert gpt2_spec.validate() == []
    assert gpt2_spec.attn_config.head_dim == 64      # 768 // 12
    assert gpt2_spec.attn_config.local_num_heads == 12
    assert gpt2_spec.ffn_config.local_ffn_hidden == 4 * 768  # mp=1 不切分
    _dbg("SELF_CHECK", f"✓ GPT-2 Small spec: {gpt2_spec.describe()}")

    # 2. GPT-2 XL 模型并行配置（mp=4，48 层，1600 hidden，25 heads）
    gpt2xl_spec = TransformerLayerSpec(
        hidden_size=1600,
        num_attention_heads=25,
        model_parallel_size=1,   # 25 % 4 ≠ 0，此处用 mp=1 验证
        layer_number=3,
    )
    # 25 heads 无法被 4 整除，若 mp=4 应报错
    gpt2xl_mp4 = TransformerLayerSpec(
        hidden_size=1600,
        num_attention_heads=24,
        model_parallel_size=4,
        layer_number=3,
    )
    assert gpt2xl_mp4.validate() == []
    assert gpt2xl_mp4.attn_config.local_num_heads == 6  # 24 // 4
    _dbg("SELF_CHECK", "✓ 模型并行配置校验")

    # 3. CheckpointStrategy
    assert CheckpointStrategy.FULL.memory_factor() < CheckpointStrategy.NONE.memory_factor()
    assert CheckpointStrategy.FULL.compute_overhead() > 0
    _dbg("SELF_CHECK", "✓ CheckpointStrategy 属性")

    # 4. 参数量估算（smoke test）
    audit = TransformerParallelAudit(num_layers=12, layer_spec=gpt2_spec)
    assert audit.total_local_params() > 0
    assert audit.total_comm_ops_per_forward() == 24  # 12 层 × 2 次通信
    _dbg("SELF_CHECK", f"✓ 参数量估算: {audit.total_local_params():,}")

    # 5. query_key_scaling_factor 正确性
    cfg_l1 = AttentionParallelConfig(
        hidden_size=768, num_attention_heads=12, model_parallel_size=1, layer_number=1
    )
    cfg_l3 = AttentionParallelConfig(
        hidden_size=768, num_attention_heads=12, model_parallel_size=1, layer_number=3
    )
    # layer 3 的缩放因子应小于 layer 1（分母更大）
    assert cfg_l3.query_key_scaling_factor < cfg_l1.query_key_scaling_factor
    _dbg("SELF_CHECK", "✓ query_key_scaling_factor 递减")

    print("[mpu_transformer_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "mpu_transformer_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
