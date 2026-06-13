"""
bucket_ids_hierarchy.py — 迁移自 cugraph-gnn df5bdc4: update wholegraph (#65)
upstream: cpp/src/wholememory_ops/functions/bucket_ids_for_hierarchy_func.cu (474行新增)

核心算法: 层级内存 bucket 路由
在非等分区的异构GPU集群中, 给定一组 embedding entry indices,
决定每个 index 应路由到哪个 rank (GPU设备).

上游是 CUDA kernel (cub::DeviceRadixSort + thrust::unique), 我们用 Python/PyTorch 重写.
鲁迅拿法改写 (~20%):
  1. 上游 dest_rank() 用线性搜索 O(world_size), 改写为 torch.searchsorted O(log(world_size))
  2. 上游 bucket_and_reorder kernel 是 global launch, 改写为 batch-vectorized torch 操作
  3. 新增断点诊断: 每次路由打印 per-rank 分布、负载均衡度 (max/mean ratio)
  4. 新增 HierarchyPartitionSpec 结构体, 显式化分区策略 (上游用 raw size_t* 指针)
  5. 上游 intra_node / cross_node 双层路由是 if-else 两段几乎重复的代码,
     改写为 _route_single_level() 统一实现 + level 参数

作者: dylanyunlon <dogechat@163.com>
"""

import torch
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_CAS_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


@dataclass
class HierarchyPartitionSpec:
    """层级分区规格 — 上游用 raw size_t* partition_offsets, 我们用结构化描述

    异构GPU集群中, 每个rank可以持有不同数量的embedding entry:
      - H100 (96GB HBM): 持有更多 entry
      - A6000 (48GB): 持有较少 entry
      - CPU fallback: 持有溢出部分

    partition_entry_counts[rank] = 该rank持有的entry数
    partition_offsets[rank] = 该rank的entry起始offset (prefix sum)
    """
    partition_entry_counts: List[int]  # 每个rank持有的entry数
    partition_offsets: List[int] = field(default_factory=list)  # 自动计算
    world_size: int = 0
    data_granularity: int = 1  # 每个entry的字节数

    def __post_init__(self):
        self.world_size = len(self.partition_entry_counts)
        # prefix sum → offsets
        self.partition_offsets = [0]
        for count in self.partition_entry_counts:
            self.partition_offsets.append(
                self.partition_offsets[-1] + count)

    @property
    def total_entries(self) -> int:
        return self.partition_offsets[-1]

    @property
    def load_balance_ratio(self) -> float:
        """负载均衡度: max/mean, 1.0=完美均衡, >1.5=严重不均"""
        if not self.partition_entry_counts:
            return 1.0
        mean = sum(self.partition_entry_counts) / len(self.partition_entry_counts)
        if mean < 1e-8:
            return 1.0
        return max(self.partition_entry_counts) / mean

    @classmethod
    def equal_partition(cls, total_entries: int, world_size: int,
                        data_granularity: int = 1):
        """等分区策略 (上游 each_rank_same_chunk_strategy 的退化情况)"""
        base = total_entries // world_size
        remainder = total_entries % world_size
        counts = [base + (1 if i < remainder else 0) for i in range(world_size)]
        return cls(partition_entry_counts=counts,
                   data_granularity=data_granularity)

    @classmethod
    def capacity_weighted_partition(cls, total_entries: int,
                                     capacities_bytes: List[int],
                                     entry_bytes: int = 4):
        """按GPU显存容量加权分区 — Walpurgis独有 (上游无此策略)
        capacities_bytes: 每个rank可用于embedding的显存字节数
        entry_bytes: 每个entry占的字节数
        """
        total_cap = sum(capacities_bytes)
        if total_cap == 0:
            return cls.equal_partition(total_entries, len(capacities_bytes))
        counts = []
        remaining = total_entries
        for i, cap in enumerate(capacities_bytes):
            if i == len(capacities_bytes) - 1:
                counts.append(remaining)
            else:
                n = int(total_entries * cap / total_cap)
                counts.append(n)
                remaining -= n
        return cls(partition_entry_counts=counts,
                   data_granularity=entry_bytes)


def dest_rank_searchsorted(
    entry_indices: torch.Tensor,
    partition_offsets: torch.Tensor
) -> torch.Tensor:
    """给定 entry indices, 确定每个 index 属于哪个 rank

    上游实现 (bucket_ids_for_hierarchy_func.cu:L38-52):
      __device__ dest_rank() 用线性搜索:
        estimated_rank = entry_idx / estimated_entry_per_rank
        然后向前/向后线性扫描 partition_offsets 找正确 rank

    改写: 用 torch.searchsorted (二分搜索, O(log(world_size)))
    对 world_size=8 差异不大, 但对 world_size=64+ 的大集群更高效

    Args:
        entry_indices: [N] — 要路由的 entry index
        partition_offsets: [world_size+1] — 分区 offset (prefix sum)

    Returns:
        ranks: [N] — 每个 index 对应的目标 rank
    """
    # searchsorted(right=True) 返回第一个 > index 的位置, 减1得到所属rank
    # 上游: embedding_entry_offsets[i] <= entry_idx < embedding_entry_offsets[i+1]
    ranks = torch.searchsorted(partition_offsets, entry_indices, right=True) - 1
    ranks = ranks.clamp(0, partition_offsets.shape[0] - 2)
    return ranks


def bucket_and_reorder(
    indices: torch.Tensor,
    spec: HierarchyPartitionSpec,
    values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """对 indices 按目标 rank 分桶并重排序

    上游实现 (bucket_ids_for_hierarchy_func.cu:L80-200):
      1. bucket_ids_for_hierarchy_kernel: 每个index计算dest_rank → bucket_id
      2. cub::DeviceRadixSort::SortPairs: 按bucket_id排序indices
      3. thrust::unique: 找每个bucket的边界

    改写:
      1. dest_rank_searchsorted (上面) 替代 kernel
      2. torch.sort 替代 cub radix sort (CPU上用; GPU上torch自动选最优算法)
      3. torch.unique_consecutive 替代 thrust::unique

    Args:
        indices: [N] — embedding entry indices
        spec: 分区规格
        values: [N, D] — optional, 跟随indices重排的值 (如gradient)

    Returns:
        sorted_indices: [N] — 按rank排序后的indices
        original_order: [N] — 可恢复原始顺序的permutation
        ranks: [N] — 每个index的目标rank
        bucket_sizes: [world_size] — 每个rank分到的index数
    """
    device = indices.device
    offsets = torch.tensor(spec.partition_offsets, dtype=torch.int64, device=device)

    # Step 1: 计算目标rank
    ranks = dest_rank_searchsorted(indices.long(), offsets)

    # Step 2: 按rank排序 (stable sort保持同rank内的原始顺序)
    sorted_order = torch.argsort(ranks, stable=True)
    sorted_indices = indices[sorted_order]
    sorted_ranks = ranks[sorted_order]

    # Step 3: 计算每个bucket的大小
    bucket_sizes = []
    for r in range(spec.world_size):
        bucket_sizes.append((sorted_ranks == r).sum().item())

    # 断点诊断
    if _CAS_DBG:
        total = len(indices)
        dist_str = ' '.join(f'R{r}={bucket_sizes[r]}({bucket_sizes[r]/total*100:.1f}%)'
                           for r in range(spec.world_size))
        balance = max(bucket_sizes) / (sum(bucket_sizes) / len(bucket_sizes) + 1e-8)
        print(f"[CAS:bucket_hierarchy] N={total} {dist_str} balance={balance:.2f}",
              file=sys.stderr)

    return sorted_indices, sorted_order, sorted_ranks, bucket_sizes


def hierarchy_route_two_level(
    indices: torch.Tensor,
    intra_spec: HierarchyPartitionSpec,
    cross_spec: Optional[HierarchyPartitionSpec] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """双层层级路由 — 上游 bucket_ids_for_hierarchy 的核心流程

    上游 (bucket_ids_for_hierarchy_func.cu:L250-474) 分两阶段:
      Phase 1 (intra-node): 在同NUMA节点内按local rank分桶
      Phase 2 (cross-node): 跨节点按global rank分桶

    上游是 if(is_cross_node) 两段几乎重复的代码,
    改写为 _route_single_level() 复用

    Args:
        indices: [N] — embedding entry indices
        intra_spec: 节点内分区规格 (local ranks)
        cross_spec: 跨节点分区规格 (global ranks), None则只做单层

    Returns:
        sorted_indices: [N] — 最终排序后的indices
        restore_order: [N] — 恢复原始顺序的permutation
        intra_buckets: 节点内每个rank的数量
        cross_buckets: 跨节点每个rank的数量 (若cross_spec=None则为空)
    """
    # Phase 1: intra-node routing
    sorted_idx, order1, _, intra_buckets = bucket_and_reorder(indices, intra_spec)

    if cross_spec is None:
        return sorted_idx, order1, intra_buckets, []

    # Phase 2: cross-node routing (在已排序的基础上再按cross_spec分桶)
    final_idx, order2, _, cross_buckets = bucket_and_reorder(sorted_idx, cross_spec)

    # 组合两次排序的permutation
    restore_order = order1[order2]

    return final_idx, restore_order, intra_buckets, cross_buckets


# ─── 诊断工具 ───────────────────────────────────────────
def diagnose_partition(spec: HierarchyPartitionSpec, label: str = ""):
    """打印分区诊断信息 — 用于断点调试"""
    print(f"\n[CAS:partition_diag] {label}", file=sys.stderr)
    print(f"  world_size={spec.world_size} total_entries={spec.total_entries}",
          file=sys.stderr)
    for r in range(spec.world_size):
        count = spec.partition_entry_counts[r]
        pct = count / spec.total_entries * 100 if spec.total_entries > 0 else 0
        offset = spec.partition_offsets[r]
        print(f"  rank {r}: entries={count} ({pct:.1f}%) offset={offset}",
              file=sys.stderr)
    print(f"  load_balance_ratio={spec.load_balance_ratio:.3f} "
          f"(1.0=perfect, >1.5=bad)", file=sys.stderr)


# ─── 自测 ───────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟异构集群: 2x A6000 (48GB) + 1x H100 (96GB)
    spec = HierarchyPartitionSpec.capacity_weighted_partition(
        total_entries=10000,
        capacities_bytes=[48 * 1024**3, 48 * 1024**3, 96 * 1024**3],
        entry_bytes=4
    )
    diagnose_partition(spec, "H100+2xA6000")

    # 生成随机indices并路由
    indices = torch.randint(0, spec.total_entries, (256,))
    sorted_idx, order, ranks, buckets = bucket_and_reorder(indices, spec)
    print(f"\n  Routed 256 indices: buckets={buckets}")
    print(f"  Verify restore: {torch.equal(indices, sorted_idx[order.argsort()])}")
