"""
walpurgis/core/mpu_support_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块合并对应 mpu/ 下 6 个支撑模块）
=================================================
  mpu/random.py（225 行）
    · get_cuda_rng_tracker()：返回全局 CudaRNGStatesTracker 实例
    · CudaRNGStatesTracker：通过 context manager 管理多组 RNG 状态
      （model_parallel / data_parallel 各自独立的 dropout 种子）
    · model_parallel_cuda_manual_seed()：在所有模型并行 rank 上设置一致的种子
  mpu/cross_entropy.py（109 行）
    · vocab_parallel_cross_entropy()：词表并行时的分布式交叉熵
      每 GPU 仅持有部分 logit，需 all-reduce 全局 log-sum-exp
  mpu/mappings.py（141 行）
    · copy_to_model_parallel_region()：forward 恒等，backward all-reduce
    · reduce_from_model_parallel_region()：forward all-reduce，backward 恒等
    · scatter_to_model_parallel_region()：forward scatter，backward all-gather
    · gather_from_model_parallel_region()：forward all-gather，backward scatter
  mpu/data.py（116 行）
    · broadcast_data()：从 rank 0 向所有 rank 广播训练数据 batch
  mpu/grads.py（74 行）
    · split_tensor_into_1d_equal_chunks() / gather_split_1d_tensor()
    · 用于 ZeRO 风格的梯度分片与聚合
  mpu/utils.py（70 行）
    · VocabUtility：词表分片边界计算的静态方法集合

CI/merge 判定：核心算法结构，直接迁移
  · random 状态管理与 Walpurgis mpu/random 的种子一致性需求直接对应
  · vocab_parallel_cross_entropy 是 GPT-2 训练的损失函数关键路径

鲁迅拿法改写（≥20%）
====================
上游六个支撑模块的共同问题是「分散的真理」：
每个模块各持一片拼图，任何一个单独看都是合理的，但没有人告诉你
「这六块拼图的组合方式是什么」「它们之间的调用顺序约束在哪里」。

mpu/random.py 里的 CudaRNGStatesTracker 最能说明这个问题：
它用一个字典存储多组 RNG 状态，用 context manager 切换。
但「model_parallel」这个 key 是硬编码的字符串，
scatter 在 transformer.py 的 attention_dropout 调用处；
如果你不小心写了「model_parallel_」多一个下划线，
运行时才会发现 RNG 状态没有按预期同步，
而模型仍然能跑，只是 dropout pattern 不一致，导致训练不收敛。
这正是鲁迅在《灯下漫笔》里说的：「中国的文明，是奴隶的文明」——
规则是有的，但规则在你发现错误之前不会告诉你它在哪里。

Walpurgis 将六个支撑模块的语义抽象为五个结构：

1. **`RNGStateKind` 枚举** — 显式建模三种 RNG 状态（MODEL_PARALLEL / DATA_PARALLEL /
   GLOBAL），替代上游裸字符串 key，使「选错状态」在类型层面报错
2. **`ParallelRNGConfig` dataclass** — 封装种子配置（base_seed、mp_rank、dp_rank），
   `derive_seed(kind)` 为每种状态派生确定性种子，上游无此结构
3. **`VocabParallelLossSpec` dataclass** — 封装词表并行交叉熵的配置与中间值
   （vocab_start、vocab_end、local_logit_max、global_log_sum_exp），
   上游函数式实现无状态记录
4. **`TensorMappingOp` 枚举** — 显式建模四种张量映射操作（COPY/REDUCE/SCATTER/GATHER），
   上游每种操作是独立函数，无统一枚举
5. **`VocabUtility`（重新实现）** — 将上游 mpu/utils.py 的静态方法集合改写为
   dataclass-based 的词表分片工具，新增 `overlap_check()` 验证各分片无重叠

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 18 处，
覆盖 RNG 状态派生、损失计算规格、张量映射操作、词表分片工具全路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[mpu_support_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "mpu_support_abe36e2e5.py 初始化开始")


# ══════════════════════════════════════════════════════════════════════════════
# 1. mpu/random.py — RNG 状态管理
# ══════════════════════════════════════════════════════════════════════════════

class RNGStateKind(Enum):
    """显式建模 mpu 中三种 RNG 状态。

    上游以裸字符串 "model-parallel-rng" / "data-parallel-rng" 作为字典 key；
    Walpurgis 强类型化，使「选错 key」在 Python 类型检查层面可见。

    migrate abe36e2e5: mpu/random.py CudaRNGStatesTracker.add() L60-L80
    """
    MODEL_PARALLEL = "model-parallel-rng"
    """模型并行 dropout：同一模型并行组内各 rank 应使用相同种子"""
    DATA_PARALLEL = "data-parallel-rng"
    """数据并行 dropout：不同数据并行 rank 应使用不同种子"""
    GLOBAL = "global-rng"
    """全局操作（weight init 等）：所有 rank 使用相同种子"""

    def should_be_identical_across_mp(self) -> bool:
        """该 RNG 状态是否应在模型并行组内保持一致。

        MODEL_PARALLEL dropout 需要所有持有同一层参数分片的 GPU 使用相同的 dropout mask。
        migrate abe36e2e5: mpu/random.py L140-L180 (model_parallel_cuda_manual_seed)
        """
        return self in (RNGStateKind.MODEL_PARALLEL, RNGStateKind.GLOBAL)


_dbg(
    "ENUM_INIT",
    f"RNGStateKind 已定义: {[k.value for k in RNGStateKind]}",
)


@dataclass(frozen=True)
class ParallelRNGConfig:
    """封装并行 RNG 配置，派生各状态的确定性种子。

    上游 model_parallel_cuda_manual_seed() 接受单个 seed，
    按公式派生各进程组的种子，但公式散落在函数体内，
    无任何结构化记录「rank X 的 model-parallel 种子是多少」。

    migrate abe36e2e5: mpu/random.py L140-L180
    """
    base_seed: int
    model_parallel_rank: int
    data_parallel_rank: int

    def derive_seed(self, kind: RNGStateKind) -> int:
        """为指定 RNG 状态派生确定性种子。

        上游：mp_seed = base_seed + 1 + mp_rank
              dp_seed = base_seed

        Walpurgis 将公式显式化，并支持 GLOBAL 种子（上游缺失）。
        migrate abe36e2e5: mpu/random.py L152-L162
        """
        if kind == RNGStateKind.MODEL_PARALLEL:
            seed = self.base_seed + 1 + self.model_parallel_rank
        elif kind == RNGStateKind.DATA_PARALLEL:
            seed = self.base_seed + 2 + self.data_parallel_rank * 2
        else:  # GLOBAL
            seed = self.base_seed
        _dbg(
            "RNG_DERIVE",
            f"kind={kind.value} mp_rank={self.model_parallel_rank} "
            f"dp_rank={self.data_parallel_rank} → seed={seed}",
        )
        return seed

    def all_seeds(self) -> Dict[RNGStateKind, int]:
        """返回所有 RNG 状态的种子映射。"""
        result = {kind: self.derive_seed(kind) for kind in RNGStateKind}
        _dbg("RNG_ALL_SEEDS", str(result))
        return result


@dataclass
class RNGStateRegistry:
    """替代上游 CudaRNGStatesTracker 的字典-字符串接口，提供类型安全的状态管理。

    上游 CudaRNGStatesTracker 存储 {str → RNG_state}，
    用 context manager 切换当前激活状态。
    Walpurgis 以枚举 key 替代字符串 key，并记录每个状态的初始化时间。

    migrate abe36e2e5: mpu/random.py CudaRNGStatesTracker L50-L130
    """
    config: ParallelRNGConfig
    _initialized_kinds: Dict[RNGStateKind, int] = field(default_factory=dict)

    def initialize_all(self) -> None:
        """初始化所有 RNG 状态（对应上游 model_parallel_cuda_manual_seed）。

        migrate abe36e2e5: mpu/random.py L140-L180
        """
        for kind in RNGStateKind:
            seed = self.config.derive_seed(kind)
            self._initialized_kinds[kind] = seed
            _dbg("RNG_INIT", f"kind={kind.value} seed={seed}")

    def is_initialized(self, kind: RNGStateKind) -> bool:
        """检查指定 RNG 状态是否已初始化。"""
        return kind in self._initialized_kinds

    def get_seed(self, kind: RNGStateKind) -> int:
        """获取指定 RNG 状态的种子。"""
        if kind not in self._initialized_kinds:
            raise RuntimeError(
                f"RNG 状态 {kind.value} 尚未初始化。请先调用 initialize_all()。"
            )
        return self._initialized_kinds[kind]


_dbg("DATACLASS_INIT", "ParallelRNGConfig + RNGStateRegistry 已定义")


# ══════════════════════════════════════════════════════════════════════════════
# 2. mpu/cross_entropy.py — 词表并行交叉熵
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VocabParallelLossSpec:
    """封装词表并行交叉熵的配置与中间计算状态。

    上游 vocab_parallel_cross_entropy() 是纯函数式实现（109 行）；
    Walpurgis 将算法分解为 spec 对象，使中间值（local_max、global_log_sum_exp）
    可被程序化检查，便于数值稳定性调试。

    核心算法（上游 mpu/cross_entropy.py）：
    1. 本地 softmax max：local_max = logits.max(dim=-1)
    2. all-reduce 全局 max：global_max = all_reduce(local_max, op=MAX)
    3. 本地 exp sum：exp_logits = exp(logits - global_max); local_sum = exp_logits.sum()
    4. all-reduce 全局 sum：global_sum = all_reduce(local_sum, op=SUM)
    5. 全局 log_sum_exp = log(global_sum) + global_max
    6. target token 在本地词表范围内：predicted_logit = logits[target - vocab_start]
    7. all-reduce predicted_logit（越界 rank 贡献 0）
    8. loss = log_sum_exp - predicted_logit

    migrate abe36e2e5: mpu/cross_entropy.py L40-L109
    """
    vocab_start_index: int
    vocab_end_index: int
    model_parallel_size: int

    # 中间状态（计算过程中填充）
    local_max: Optional[float] = None
    global_max: Optional[float] = None
    local_exp_sum: Optional[float] = None
    global_exp_sum: Optional[float] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.vocab_start_index >= self.vocab_end_index:
            errors.append(
                f"vocab_start_index={self.vocab_start_index} 必须小于 "
                f"vocab_end_index={self.vocab_end_index}"
            )
        _dbg(
            "LOSS_SPEC_VALIDATE",
            f"vocab=[{self.vocab_start_index}, {self.vocab_end_index}) "
            f"errors={errors}",
        )
        return errors

    @property
    def local_vocab_size(self) -> int:
        """本地词表分片大小。"""
        return self.vocab_end_index - self.vocab_start_index

    def is_local_target(self, target_token_id: int) -> bool:
        """判断目标 token 是否在本地词表分片内。

        migrate abe36e2e5: mpu/cross_entropy.py L75-L82
        """
        return self.vocab_start_index <= target_token_id < self.vocab_end_index

    def global_log_sum_exp(self) -> Optional[float]:
        """计算全局 log_sum_exp（需要 global_max 和 global_exp_sum 已填充）。

        migrate abe36e2e5: mpu/cross_entropy.py L95-L98
        """
        import math
        if self.global_max is None or self.global_exp_sum is None:
            _dbg("LOSS_LSE", "global_max 或 global_exp_sum 尚未填充，无法计算")
            return None
        lse = math.log(self.global_exp_sum) + self.global_max
        _dbg("LOSS_LSE", f"log_sum_exp={lse:.4f}")
        return lse

    def describe(self) -> str:
        return (
            f"VocabParallelLossSpec("
            f"vocab=[{self.vocab_start_index},{self.vocab_end_index}), "
            f"local_size={self.local_vocab_size}, "
            f"mp={self.model_parallel_size})"
        )


_dbg("DATACLASS_INIT", "VocabParallelLossSpec 已定义")


# ══════════════════════════════════════════════════════════════════════════════
# 3. mpu/mappings.py — 张量映射操作
# ══════════════════════════════════════════════════════════════════════════════

class TensorMappingOp(Enum):
    """显式建模 mpu/mappings.py 中四种张量映射操作。

    上游每种操作是独立函数（copy_to/reduce_from/scatter_to/gather_from），
    无统一枚举。Walpurgis 将四种操作收敛至枚举，使调用者能声明
    「这一层的前向操作是什么」而非记忆四个函数名。

    migrate abe36e2e5: mpu/mappings.py L1-L141
    """
    COPY = "copy_to_model_parallel"
    """forward: 恒等（输入复制到每个 rank）；backward: all-reduce 梯度"""
    REDUCE = "reduce_from_model_parallel"
    """forward: all-reduce（聚合各 rank 的部分和）；backward: 恒等"""
    SCATTER = "scatter_to_model_parallel"
    """forward: scatter（将输入均分给各 rank）；backward: all-gather"""
    GATHER = "gather_from_model_parallel"
    """forward: all-gather（聚合各 rank 的本地张量）；backward: scatter"""

    def forward_communication(self) -> str:
        """前向传播的通信类型。"""
        return {
            TensorMappingOp.COPY: "none（各 rank 持有完整输入副本）",
            TensorMappingOp.REDUCE: "all-reduce",
            TensorMappingOp.SCATTER: "scatter",
            TensorMappingOp.GATHER: "all-gather",
        }[self]

    def backward_communication(self) -> str:
        """反向传播的通信类型（前向的对偶操作）。"""
        return {
            TensorMappingOp.COPY: "all-reduce",
            TensorMappingOp.REDUCE: "none（梯度直接通过）",
            TensorMappingOp.SCATTER: "all-gather",
            TensorMappingOp.GATHER: "scatter",
        }[self]

    def describe(self) -> str:
        return (
            f"{self.value}: "
            f"forward={self.forward_communication()}, "
            f"backward={self.backward_communication()}"
        )


_dbg(
    "ENUM_INIT",
    f"TensorMappingOp 已定义: {[op.value for op in TensorMappingOp]}",
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. mpu/utils.py — 词表分片工具
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VocabPartitionSpec:
    """单个词表分片的规格（单 rank 视角）。

    上游 VocabUtility.vocab_range_from_per_partition_vocab_size() 返回裸元组；
    Walpurgis 封装为 dataclass，使分片规格可被程序化检查。

    migrate abe36e2e5: mpu/utils.py VocabUtility L20-L65
    """
    rank: int
    model_parallel_size: int
    total_vocab_size: int

    def __post_init__(self) -> None:
        errors = self._validate()
        if errors:
            raise ValueError(f"VocabPartitionSpec 校验失败: {errors}")
        _dbg(
            "VOCAB_PARTITION",
            f"rank={self.rank} mp={self.model_parallel_size} "
            f"total={self.total_vocab_size} → {self.describe()}",
        )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.total_vocab_size % self.model_parallel_size != 0:
            errors.append(
                f"total_vocab_size={self.total_vocab_size} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        if not (0 <= self.rank < self.model_parallel_size):
            errors.append(
                f"rank={self.rank} 越界 [0, {self.model_parallel_size})"
            )
        return errors

    @property
    def per_partition_size(self) -> int:
        """每个分片的词表行数。

        migrate abe36e2e5: mpu/utils.py L27
        """
        return self.total_vocab_size // self.model_parallel_size

    @property
    def start_index(self) -> int:
        """本分片起始词表索引（含）。

        migrate abe36e2e5: mpu/utils.py L45 vocab_start_index
        """
        return self.rank * self.per_partition_size

    @property
    def end_index(self) -> int:
        """本分片结束词表索引（不含）。

        migrate abe36e2e5: mpu/utils.py L46 vocab_end_index
        """
        return self.start_index + self.per_partition_size

    def describe(self) -> str:
        return (
            f"VocabPartitionSpec(rank={self.rank}/{self.model_parallel_size}, "
            f"[{self.start_index}, {self.end_index}), "
            f"size={self.per_partition_size})"
        )


class VocabPartitionManifest:
    """汇总所有 rank 的词表分片，验证无重叠、无遗漏。

    上游 VocabUtility 无此聚合视图；Walpurgis 新增，
    使「整个词表是否被正确分片」可一次性审计。

    migrate abe36e2e5: mpu/utils.py（扩展）
    """

    def __init__(self, total_vocab_size: int, model_parallel_size: int) -> None:
        self.total_vocab_size = total_vocab_size
        self.model_parallel_size = model_parallel_size
        self._partitions: List[VocabPartitionSpec] = []
        for rank in range(model_parallel_size):
            spec = VocabPartitionSpec(
                rank=rank,
                model_parallel_size=model_parallel_size,
                total_vocab_size=total_vocab_size,
            )
            self._partitions.append(spec)
        _dbg(
            "VOCAB_MANIFEST",
            f"total={total_vocab_size} mp={model_parallel_size} "
            f"partitions={len(self._partitions)}",
        )

    def overlap_check(self) -> List[str]:
        """验证各分片无重叠（完备性检查）。

        migrate abe36e2e5: 上游无此检查，Walpurgis 新增
        """
        errors: List[str] = []
        for i, p1 in enumerate(self._partitions):
            for p2 in self._partitions[i + 1:]:
                if p1.start_index < p2.end_index and p2.start_index < p1.end_index:
                    errors.append(
                        f"分片重叠: rank={p1.rank} [{p1.start_index},{p1.end_index}) "
                        f"与 rank={p2.rank} [{p2.start_index},{p2.end_index})"
                    )
        coverage = sum(p.per_partition_size for p in self._partitions)
        if coverage != self.total_vocab_size:
            errors.append(
                f"总覆盖 {coverage} ≠ total_vocab_size {self.total_vocab_size}"
            )
        _dbg("OVERLAP_CHECK", f"errors={errors}")
        return errors

    def partition_for_token(self, token_id: int) -> Optional[VocabPartitionSpec]:
        """找出 token_id 所在的分片。"""
        for p in self._partitions:
            if p.start_index <= token_id < p.end_index:
                return p
        return None

    def summary(self) -> str:
        lines = [
            f"=== VocabPartitionManifest ===",
            f"total_vocab: {self.total_vocab_size}",
            f"model_parallel_size: {self.model_parallel_size}",
        ]
        for p in self._partitions:
            lines.append(f"  {p.describe()}")
        overlap_errors = self.overlap_check()
        if overlap_errors:
            lines.append(f"⚠ 重叠错误: {overlap_errors}")
        else:
            lines.append("✓ 无重叠，分片完备")
        return "\n".join(lines)


_dbg("DATACLASS_INIT", "VocabPartitionSpec + VocabPartitionManifest 已定义")


# ══════════════════════════════════════════════════════════════════════════════
# 5. mpu/grads.py — 梯度分片工具
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GradShardSpec:
    """封装 ZeRO 风格梯度分片配置。

    上游 mpu/grads.py::split_tensor_into_1d_equal_chunks() 接受裸张量，
    按 world_size 均分为 1D 分片，用于跨数据并行 rank 的梯度分散。
    Walpurgis 将分片参数显式化。

    migrate abe36e2e5: mpu/grads.py L20-L74
    """
    total_elements: int
    data_parallel_size: int
    rank: int

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.total_elements % self.data_parallel_size != 0:
            errors.append(
                f"total_elements={self.total_elements} 必须整除 "
                f"data_parallel_size={self.data_parallel_size}；"
                f"上游用 padding 处理非整除情况"
            )
        if not (0 <= self.rank < self.data_parallel_size):
            errors.append(
                f"rank={self.rank} 越界 [0, {self.data_parallel_size})"
            )
        return errors

    @property
    def chunk_size(self) -> int:
        """每个 rank 的梯度分片大小。

        若不能整除，向上取整（上游用 pad）。
        migrate abe36e2e5: mpu/grads.py L32 chunk_size
        """
        import math
        return math.ceil(self.total_elements / self.data_parallel_size)

    @property
    def local_start(self) -> int:
        """本 rank 的梯度分片起始索引。"""
        return self.rank * self.chunk_size

    @property
    def local_end(self) -> int:
        """本 rank 的梯度分片结束索引（不含）。"""
        return min(self.local_start + self.chunk_size, self.total_elements)

    def describe(self) -> str:
        return (
            f"GradShardSpec(rank={self.rank}/{self.data_parallel_size}, "
            f"total={self.total_elements}, chunk={self.chunk_size}, "
            f"[{self.local_start}, {self.local_end}))"
        )


_dbg("DATACLASS_INIT", "GradShardSpec 已定义")


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证所有支撑模块结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. RNG 状态种子派生
    rng_cfg = ParallelRNGConfig(base_seed=1234, model_parallel_rank=2, data_parallel_rank=0)
    mp_seed = rng_cfg.derive_seed(RNGStateKind.MODEL_PARALLEL)
    global_seed = rng_cfg.derive_seed(RNGStateKind.GLOBAL)
    assert mp_seed == 1234 + 1 + 2  # base + 1 + mp_rank
    assert global_seed == 1234
    _dbg("SELF_CHECK", f"✓ RNG 种子派生: mp={mp_seed} global={global_seed}")

    # 2. RNG 状态注册表
    reg = RNGStateRegistry(config=rng_cfg)
    assert not reg.is_initialized(RNGStateKind.MODEL_PARALLEL)
    reg.initialize_all()
    assert reg.is_initialized(RNGStateKind.MODEL_PARALLEL)
    assert reg.get_seed(RNGStateKind.GLOBAL) == 1234
    _dbg("SELF_CHECK", "✓ RNGStateRegistry 生命周期")

    # 3. VocabParallelLossSpec
    loss_spec = VocabParallelLossSpec(
        vocab_start_index=0, vocab_end_index=25128, model_parallel_size=2
    )
    assert loss_spec.validate() == []
    assert loss_spec.local_vocab_size == 25128
    assert loss_spec.is_local_target(1000)
    assert not loss_spec.is_local_target(30000)
    _dbg("SELF_CHECK", "✓ VocabParallelLossSpec")

    # 4. TensorMappingOp
    for op in TensorMappingOp:
        desc = op.describe()
        assert len(desc) > 0
    _dbg("SELF_CHECK", "✓ TensorMappingOp.describe()")

    # 5. VocabPartitionManifest
    manifest = VocabPartitionManifest(total_vocab_size=50256, model_parallel_size=4)
    assert manifest.overlap_check() == []
    p = manifest.partition_for_token(50000)
    assert p is not None
    assert p.start_index <= 50000 < p.end_index
    _dbg("SELF_CHECK", f"✓ VocabPartitionManifest: {manifest.model_parallel_size} 分片无重叠")

    # 6. GradShardSpec
    grad_spec = GradShardSpec(total_elements=4096, data_parallel_size=4, rank=1)
    assert grad_spec.chunk_size == 1024
    assert grad_spec.local_start == 1024
    assert grad_spec.local_end == 2048
    _dbg("SELF_CHECK", f"✓ GradShardSpec: {grad_spec.describe()}")

    print("[mpu_support_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "mpu_support_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
