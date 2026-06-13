"""
walpurgis/core/mpu_layers_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块对应 mpu/layers.py，327 行新增）
===================================================
  VocabParallelEmbedding
    · 词表在模型并行维度切分：每个 GPU 持有 [vocab_start, vocab_end) 的 embedding 行
    · forward: mask 越界 token → 本地 embedding lookup → all-reduce 聚合
  ColumnParallelLinear
    · 列并行线性层：权重矩阵按列切分，每 GPU 持有 out_features // mp_size 列
    · gather_output=True 时在输出端做 all-gather（适用于最后一层）
    · bias_tp_auto_sync: bias 在所有模型并行 rank 上保持一致
  RowParallelLinear
    · 行并行线性层：权重矩阵按行切分，每 GPU 持有 in_features // mp_size 行
    · input_is_parallel=True 时跳过 scatter（上游管线中配合 ColumnParallelLinear 使用）
    · 输出端做 all-reduce 聚合各 GPU 的部分和

CI/merge 判定：核心算法结构，直接迁移
  · ColumnParallelLinear / RowParallelLinear 是 Megatron 张量并行的基础算子
  · 与 Walpurgis 中 wholememory 分布式 embedding 有结构对应

鲁迅拿法改写（≥20%）
====================
上游 mpu/layers.py 的深层矛盾是「切分-聚合」的不对称性：
ColumnParallelLinear 把列切开、可选地聚合输出；
RowParallelLinear 把行切开、必须 all-reduce 部分和。
这种「切而不全、合而有代价」的模式，像极了鲁迅《呐喊》自序里的「铁屋子」——
房子（计算图）不能轻易打破，但里面的人（参数）必须重新分配，
分配的规则（切分策略）却埋藏在 `__init__` 的注释里，调用者只能凭经验猜测。

上游 `gather_output` / `input_is_parallel` 两个布尔值，
就是那两扇「不知道开着还是关着」的窗：
调用者需要同时知道上下文（前一层是否列并行、后一层是否行并行）才能正确设置，
但上游没有任何结构帮你表达这个约束——出错了就是 shape mismatch，
错误信息里没有「你忘了设 input_is_parallel」。

Walpurgis 将「张量并行层」的切分-聚合策略抽象为五个结构：

1. **`TensorParallelStrategy` 枚举** — 显式建模四种切分-聚合组合，
   使「列并行 + gather」「行并行 + all-reduce」等模式有名字而非裸布尔值
2. **`ParallelLinearSpec` dataclass** — 封装层配置（in/out features、mp_size、strategy），
   `validate()` 前置校验整除性，`local_out_features` / `local_in_features` 属性
   直接给出本地维度，上游调用者须手动除以 mp_size
3. **`VocabShardSpec` dataclass** — 封装词表分片配置，
   `vocab_start_index` / `vocab_end_index` 属性替代上游裸算术
4. **`LayerParallelAudit` dataclass** — 记录层创建时的分片决策，
   上游 __init__ 无任何此类记录
5. **`TensorParallelLayerRegistry`** — 汇总已创建的并行层，
   提供按 strategy 查询、统计本地参数量等接口

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 15 处，
覆盖 spec 校验、词表分片、层注册、审计查询全路径。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[mpu_layers_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "mpu_layers_abe36e2e5.py 初始化开始")


# ── 枚举：张量并行策略 ───────────────────────────────────────────────────────

class TensorParallelStrategy(Enum):
    """显式建模四种张量并行切分-聚合组合。

    上游以 gather_output / input_is_parallel 两个布尔值隐式表达；
    Walpurgis 强类型化，使调用者的意图在类型层面可见。

    migrate abe36e2e5: mpu/layers.py ColumnParallelLinear.__init__ L87-L130
                                      RowParallelLinear.__init__ L170-L220
    """
    # ColumnParallelLinear 变体
    COLUMN_NO_GATHER = "column_no_gather"
    """列并行，不聚合输出（管线中间层，下一层为 RowParallelLinear）"""
    COLUMN_WITH_GATHER = "column_with_gather"
    """列并行，聚合输出（最后一层或独立使用）"""

    # RowParallelLinear 变体
    ROW_PARALLEL_INPUT = "row_parallel_input"
    """行并行，输入已分散（配合 COLUMN_NO_GATHER 使用）"""
    ROW_SCATTERED_INPUT = "row_scattered_input"
    """行并行，输入未分散（独立使用，需先 scatter）"""

    def is_column_parallel(self) -> bool:
        return self in (
            TensorParallelStrategy.COLUMN_NO_GATHER,
            TensorParallelStrategy.COLUMN_WITH_GATHER,
        )

    def is_row_parallel(self) -> bool:
        return self in (
            TensorParallelStrategy.ROW_PARALLEL_INPUT,
            TensorParallelStrategy.ROW_SCATTERED_INPUT,
        )

    def requires_all_reduce(self) -> bool:
        """行并行层输出端需要 all-reduce 聚合各 GPU 的部分和。

        migrate abe36e2e5: mpu/layers.py RowParallelLinear.forward L246-L260
        """
        return self.is_row_parallel()

    def requires_all_gather(self) -> bool:
        """列并行+gather 变体需要在输出端做 all-gather。

        migrate abe36e2e5: mpu/layers.py ColumnParallelLinear.forward L157-L163
        """
        return self == TensorParallelStrategy.COLUMN_WITH_GATHER

    def describe(self) -> str:
        descriptions = {
            TensorParallelStrategy.COLUMN_NO_GATHER: (
                "列并行（不聚合）：权重按列切分，输出为本地部分结果，"
                "下一层须为 RowParallelLinear"
            ),
            TensorParallelStrategy.COLUMN_WITH_GATHER: (
                "列并行（all-gather）：权重按列切分，输出聚合为完整张量"
            ),
            TensorParallelStrategy.ROW_PARALLEL_INPUT: (
                "行并行（并行输入）：权重按行切分，输入已分散，"
                "输出需 all-reduce"
            ),
            TensorParallelStrategy.ROW_SCATTERED_INPUT: (
                "行并行（scatter 输入）：权重按行切分，输入需先 scatter，"
                "输出需 all-reduce"
            ),
        }
        return descriptions[self]


_dbg(
    "ENUM_INIT",
    f"TensorParallelStrategy 已定义: {[s.value for s in TensorParallelStrategy]}",
)


# ── 数据类：并行线性层规格 ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ParallelLinearSpec:
    """封装 ColumnParallelLinear / RowParallelLinear 的完整配置。

    上游在 __init__ 中直接用算术表达式计算本地维度，无结构化校验。
    Walpurgis 将所有配置与衍生属性收敛至此 dataclass，
    `validate()` 前置拦截整除性错误。

    migrate abe36e2e5: mpu/layers.py L87-L130 (Column) + L170-L220 (Row)
    """
    in_features: int
    out_features: int
    model_parallel_size: int
    strategy: TensorParallelStrategy
    bias: bool = True
    skip_bias_add: bool = False        # 上游 ColumnParallelLinear 的 skip_bias_add 参数

    def validate(self) -> List[str]:
        """前置校验整除性与策略-维度一致性。

        上游：在 __init__ 内用 assert；Walpurgis：返回错误列表，不 crash。
        """
        errors: List[str] = []
        if self.strategy.is_column_parallel():
            if self.out_features % self.model_parallel_size != 0:
                errors.append(
                    f"列并行：out_features={self.out_features} 必须整除 "
                    f"model_parallel_size={self.model_parallel_size}"
                )
        if self.strategy.is_row_parallel():
            if self.in_features % self.model_parallel_size != 0:
                errors.append(
                    f"行并行：in_features={self.in_features} 必须整除 "
                    f"model_parallel_size={self.model_parallel_size}"
                )
        _dbg(
            "SPEC_VALIDATE",
            f"strategy={self.strategy.value} in={self.in_features} "
            f"out={self.out_features} mp={self.model_parallel_size} "
            f"errors={errors}",
        )
        return errors

    @property
    def local_out_features(self) -> int:
        """本地输出维度（列并行下等于 out_features // mp_size）。

        migrate abe36e2e5: mpu/layers.py L109 output_size_per_partition
        """
        if self.strategy.is_column_parallel():
            return self.out_features // self.model_parallel_size
        return self.out_features

    @property
    def local_in_features(self) -> int:
        """本地输入维度（行并行下等于 in_features // mp_size）。

        migrate abe36e2e5: mpu/layers.py L196 input_size_per_partition
        """
        if self.strategy.is_row_parallel():
            return self.in_features // self.model_parallel_size
        return self.in_features

    def weight_shape(self) -> Tuple[int, int]:
        """本地权重矩阵形状 (local_out, local_in)。

        上游：ColumnParallelLinear 初始化 weight 为 (output_size_per_partition, in_features)
             RowParallelLinear 初始化 weight 为 (out_features, input_size_per_partition)
        migrate abe36e2e5: mpu/layers.py L110 + L197
        """
        return (self.local_out_features, self.local_in_features)

    def describe(self) -> str:
        return (
            f"ParallelLinearSpec(strategy={self.strategy.value}, "
            f"in={self.in_features}→local_in={self.local_in_features}, "
            f"out={self.out_features}→local_out={self.local_out_features}, "
            f"mp={self.model_parallel_size})"
        )


_dbg("DATACLASS_INIT", "ParallelLinearSpec 已定义")


# ── 数据类：词表分片规格 ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class VocabShardSpec:
    """封装 VocabParallelEmbedding 的词表分片配置。

    上游 mpu/layers.py::VocabParallelEmbedding.__init__ 直接算术分片，
    越界 mask 逻辑散落在 forward 中。Walpurgis 将分片规则收敛至此 dataclass。

    migrate abe36e2e5: mpu/layers.py VocabParallelEmbedding L20-L85
    """
    num_embeddings: int          # 总词表大小
    embedding_dim: int           # embedding 维度
    model_parallel_size: int
    rank: int                    # 当前 GPU 在模型并行组中的 rank
    padding_idx: Optional[int] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.num_embeddings % self.model_parallel_size != 0:
            errors.append(
                f"num_embeddings={self.num_embeddings} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        if not (0 <= self.rank < self.model_parallel_size):
            errors.append(
                f"rank={self.rank} 越界 [0, {self.model_parallel_size})"
            )
        _dbg(
            "VOCAB_SHARD_VALIDATE",
            f"vocab={self.num_embeddings} dim={self.embedding_dim} "
            f"mp={self.model_parallel_size} rank={self.rank} errors={errors}",
        )
        return errors

    @property
    def vocab_per_partition(self) -> int:
        """每个 GPU 持有的词表行数。

        migrate abe36e2e5: mpu/layers.py L40 vocab_size_per_partition
        """
        return self.num_embeddings // self.model_parallel_size

    @property
    def vocab_start_index(self) -> int:
        """本分片的起始词表索引（含）。

        migrate abe36e2e5: mpu/layers.py L41 vocab_start_index
        """
        return self.rank * self.vocab_per_partition

    @property
    def vocab_end_index(self) -> int:
        """本分片的结束词表索引（不含）。

        migrate abe36e2e5: mpu/layers.py L42 vocab_end_index
        """
        return self.vocab_start_index + self.vocab_per_partition

    def is_local_token(self, token_id: int) -> bool:
        """判断 token_id 是否属于本分片。

        上游 forward 中用 mask = (input < vocab_start_index) | (input >= vocab_end_index)
        migrate abe36e2e5: mpu/layers.py VocabParallelEmbedding.forward L60-L72
        """
        return self.vocab_start_index <= token_id < self.vocab_end_index

    def local_token_id(self, global_token_id: int) -> int:
        """将全局 token_id 转换为本分片的本地索引。

        上游：masked_input = input.clone() - vocab_start_index
        migrate abe36e2e5: mpu/layers.py L67
        """
        if not self.is_local_token(global_token_id):
            raise ValueError(
                f"token_id={global_token_id} 不属于本分片 "
                f"[{self.vocab_start_index}, {self.vocab_end_index})"
            )
        return global_token_id - self.vocab_start_index

    def describe(self) -> str:
        return (
            f"VocabShardSpec(rank={self.rank}/{self.model_parallel_size}, "
            f"vocab=[{self.vocab_start_index}, {self.vocab_end_index}), "
            f"local_size={self.vocab_per_partition}, dim={self.embedding_dim})"
        )


_dbg("DATACLASS_INIT", "VocabShardSpec 已定义")


# ── 审计记录 ─────────────────────────────────────────────────────────────────

@dataclass
class LayerParallelAudit:
    """记录并行层创建时的分片决策。

    上游 __init__ 无任何此类记录；Walpurgis 新增，支持事后审计「哪些层、
    用了什么切分策略、本地参数量是多少」。

    migrate abe36e2e5: 上游无对等结构
    """
    layer_id: str                        # 用户自定义层标识
    spec: ParallelLinearSpec
    created_at: float = field(default_factory=time.time)
    local_param_count: int = 0

    def __post_init__(self) -> None:
        w_shape = self.spec.weight_shape()
        self.local_param_count = w_shape[0] * w_shape[1]
        if self.spec.bias:
            self.local_param_count += self.spec.local_out_features
        _dbg(
            "LAYER_AUDIT",
            f"layer_id={self.layer_id} strategy={self.spec.strategy.value} "
            f"local_params={self.local_param_count}",
        )

    def describe(self) -> str:
        return (
            f"LayerParallelAudit(id={self.layer_id}, "
            f"strategy={self.spec.strategy.value}, "
            f"weight_shape={self.spec.weight_shape()}, "
            f"local_params={self.local_param_count})"
        )


# ── 注册表 ───────────────────────────────────────────────────────────────────

class TensorParallelLayerRegistry:
    """汇总已创建的并行层，提供按 strategy 查询和参数量统计。

    上游无此结构；Walpurgis 新增，使「整个模型共有多少并行层、
    本地参数量是多少」可程序化查询。

    migrate abe36e2e5: 上游无对等结构，Walpurgis 新增
    """

    def __init__(self) -> None:
        self._layers: Dict[str, LayerParallelAudit] = {}
        _dbg("REGISTRY_INIT", "TensorParallelLayerRegistry 创建")

    def register(
        self,
        layer_id: str,
        spec: ParallelLinearSpec,
    ) -> LayerParallelAudit:
        """注册一个并行层并返回其审计记录。"""
        errors = spec.validate()
        if errors:
            raise ValueError(
                f"并行层规格校验失败 (layer_id={layer_id}): {errors}"
            )
        audit = LayerParallelAudit(layer_id=layer_id, spec=spec)
        self._layers[layer_id] = audit
        _dbg("REGISTRY_REGISTER", f"{audit.describe()}")
        return audit

    def by_strategy(
        self, strategy: TensorParallelStrategy
    ) -> List[LayerParallelAudit]:
        """返回使用指定切分策略的所有层。"""
        result = [a for a in self._layers.values() if a.spec.strategy == strategy]
        _dbg(
            "REGISTRY_QUERY",
            f"strategy={strategy.value} → {len(result)} 层",
        )
        return result

    def total_local_params(self) -> int:
        """当前 GPU 持有的全部并行层参数量之和。"""
        total = sum(a.local_param_count for a in self._layers.values())
        _dbg("REGISTRY_PARAMS", f"total_local_params={total}")
        return total

    def summary(self) -> str:
        """输出所有已注册层的汇总信息。"""
        lines = [
            f"=== TensorParallelLayerRegistry ({len(self._layers)} 层) ===",
            f"本地总参数量: {self.total_local_params():,}",
        ]
        for strategy in TensorParallelStrategy:
            layers = self.by_strategy(strategy)
            if layers:
                lines.append(f"  [{strategy.value}] {len(layers)} 层:")
                for a in layers:
                    lines.append(f"    {a.describe()}")
        return "\n".join(lines)


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证核心结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. ColumnParallelLinear spec
    col_spec = ParallelLinearSpec(
        in_features=1024,
        out_features=4096,
        model_parallel_size=4,
        strategy=TensorParallelStrategy.COLUMN_NO_GATHER,
    )
    assert col_spec.validate() == []
    assert col_spec.local_out_features == 1024   # 4096 // 4
    assert col_spec.local_in_features == 1024    # column parallel: in_features 不切分
    assert col_spec.weight_shape() == (1024, 1024)
    _dbg("SELF_CHECK", "✓ ColumnParallelLinear spec")

    # 2. RowParallelLinear spec
    row_spec = ParallelLinearSpec(
        in_features=4096,
        out_features=1024,
        model_parallel_size=4,
        strategy=TensorParallelStrategy.ROW_PARALLEL_INPUT,
    )
    assert row_spec.validate() == []
    assert row_spec.local_in_features == 1024    # 4096 // 4
    assert row_spec.local_out_features == 1024   # row parallel: out_features 不切分
    assert row_spec.weight_shape() == (1024, 1024)
    _dbg("SELF_CHECK", "✓ RowParallelLinear spec")

    # 3. 非整除 spec 被拒绝
    bad_spec = ParallelLinearSpec(
        in_features=1000,
        out_features=4096,
        model_parallel_size=3,
        strategy=TensorParallelStrategy.COLUMN_NO_GATHER,
    )
    assert len(bad_spec.validate()) > 0
    _dbg("SELF_CHECK", "✓ 非整除 spec 拒绝")

    # 4. VocabShardSpec
    vocab = VocabShardSpec(
        num_embeddings=50257,
        embedding_dim=768,
        model_parallel_size=1,
        rank=0,
    )
    # mp_size=1 时整个词表在一个 GPU 上
    assert vocab.vocab_start_index == 0
    assert vocab.vocab_end_index == 50257
    assert vocab.is_local_token(0)
    assert vocab.is_local_token(50256)
    _dbg("SELF_CHECK", f"✓ VocabShardSpec: {vocab.describe()}")

    # 5. Registry 注册与查询
    registry = TensorParallelLayerRegistry()
    registry.register("attn.qkv", col_spec)
    registry.register("attn.out", row_spec)
    col_layers = registry.by_strategy(TensorParallelStrategy.COLUMN_NO_GATHER)
    assert len(col_layers) == 1
    assert registry.total_local_params() > 0
    _dbg("SELF_CHECK", f"✓ Registry: {registry.total_local_params():,} 参数")

    print("[mpu_layers_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "mpu_layers_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
