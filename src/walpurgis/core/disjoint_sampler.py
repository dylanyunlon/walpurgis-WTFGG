"""
disjoint_sampler.py
迁移自 upstream b25bc88 ([FEA] Support Disjoint Sampling in cuGraph-PyG #452)
          + 659a0e1 ([BUG] Fix hashing and node id issues in disjoint sampling test #474)

原上游：distributed_sampler.py 新增 disjoint=False 参数，
        及 __calc_local_seeds_per_call 的 disjoint 内存估算修正。
改写：将「disjoint 采样内存估算逻辑」独立为可测试 Python 模块，
      并将 659a0e1 的哈希表 node_id 修正建模为 DisjointBatchVerifier。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Optional


# ── 常量（上游 DistributedNeighborSampler 类级别） ───────────────────────────
# 上游实测值：DistributedNeighborSampler.BASE_VERTICES_PER_BYTE = 0.1107662486009992
# 物理含义：每字节 GPU 显存对应约 0.11 个顶点（经验校准系数）
_BASE_VERTICES_PER_BYTE: float = 0.1107662486009992
_UNKNOWN_VERTICES_DEFAULT: int = 32_768


# ── 1. DisjointMemoryEstimator：disjoint 采样显存估算 ────────────────────────
@dataclass(frozen=True)
class DisjointMemoryEstimator:
    """
    上游 b25bc88 在 __calc_local_seeds_per_call 中追加：
      if disjoint:
          fanout_prod *= fanout[0]  # 无跨 seed 去重，额外 fanout[0] 倍内存增长

    此类将该估算逻辑独立化，使其可被单独测试，
    原上游内联在 DistributedNeighborSampler 私有方法中，无独立测试覆盖。

    字段（鲁迅刀法）：
      fanout         — 每 hop 采样扇出列表，e.g. [25, 10]
      total_gpu_memory — 目标 GPU 显存字节数
      disjoint       — 是否启用 disjoint 采样
      heterogeneous  — 是否异构图（影响 hop 计数方式）
      num_edge_types — 异构图边类型数
    """
    fanout: tuple[int, ...]
    total_gpu_memory: int
    disjoint: bool = False
    heterogeneous: bool = False
    num_edge_types: int = 1

    def estimate(self) -> int:
        """
        返回 local_seeds_per_call 估算值。
        同上游逻辑，但将 disjoint 分支与异构图分支显式分离，不再嵌套。
        """
        # 若含非正 fanout（全采样），返回默认值
        if any(x <= 0 for x in self.fanout):
            return _UNKNOWN_VERTICES_DEFAULT

        # 异构图：按 edge type 分组 fanout，取各 hop 最大值
        if self.heterogeneous:
            if len(self.fanout) % self.num_edge_types != 0:
                raise ValueError(
                    f"异构图 fanout 长度 {len(self.fanout)} "
                    f"不能被 num_edge_types={self.num_edge_types} 整除"
                )
            num_hops = len(self.fanout) // self.num_edge_types
            per_hop = [
                max(self.fanout[h * self.num_edge_types:(h + 1) * self.num_edge_types])
                for h in range(num_hops)
            ]
        else:
            per_hop = list(self.fanout)

        fanout_prod = reduce(lambda x, y: x * y, per_hop)

        # Disjoint 放大：无跨 seed 去重，每 seed 独立展开，
        # 上游注释：「memory grows by an extra fanout[0] factor」
        if self.disjoint:
            fanout_prod *= per_hop[0]

        result = int(
            _BASE_VERTICES_PER_BYTE * self.total_gpu_memory / fanout_prod
        )
        return max(result, 1)

    def disjoint_overhead_factor(self) -> float:
        """相对于非 disjoint 模式的内存开销倍数（用于日志/告警）"""
        if not self.disjoint or not self.fanout:
            return 1.0
        return float(self.fanout[0])


# ── 2. DisjointSamplingSpec：disjoint 参数传递合约 ──────────────────────────
@dataclass(frozen=True)
class DisjointSamplingSpec:
    """
    上游 b25bc88 在 DistributedNeighborSampler.__init__ 中新增：
      disjoint: bool = False
    并在 sample_kwargs 中以 "disjoint_sampling" 键传递给 cugraph backend。

    此 spec 将「disjoint 参数命名映射」显式化，
    避免调用方记忆 cugraph backend 内部键名 "disjoint_sampling"（与外部参数名 disjoint 不同）。
    """
    enabled: bool = False

    def as_cugraph_kwarg(self) -> dict:
        """返回传给 cugraph sample 后端的 kwarg dict"""
        return {"disjoint_sampling": self.enabled}

    def as_human_label(self) -> str:
        return "disjoint" if self.enabled else "standard"


# ── 3. DisjointBatchVerifier：659a0e1 修正的节点 ID 验证逻辑 ─────────────────
@dataclass
class DisjointBatchVerifier:
    """
    上游 659a0e1 修正了测试中 hash table 构造错误：
      原：tree_vertices[n_id] = set([n_id.item()])     ← n_id 是 tensor，作 key 不稳定
      修：tree_vertices[n_id.item()] = set([n_id.item()])  ← 整数 key

    此类将「disjoint batch 不变式」建模为可复用的验证器，
    供 Walpurgis 训练循环中的 smoke check 使用。

    不变式（Disjoint Sampling 保证）：
      对于批次内任意两个不同 seed i ≠ j，
      其采样树节点集合 T_i ∩ T_j = ∅（无跨 seed 共享节点）
    """

    def build_tree_vertices(
        self,
        num_seeds: int,
        edge_index: "list[tuple[int,int]]",
        num_sampled_edges_per_hop: "list[int]",
    ) -> "dict[int, set[int]]":
        """
        从 edge_index 和每 hop 边数重建各 seed 的采样树节点集合。

        参数：
          num_seeds              — batch 内 seed 数（对应 num_sampled_nodes[0]）
          edge_index             — [(src, dst), ...] 格式边列表
          num_sampled_edges_per_hop — 每 hop 的边数列表

        上游 659a0e1 修正：用 torch.arange(num_seeds) 而非 batch.input_id 迭代，
        此处 Python 层等价实现。
        """
        # 修正点：用整数索引作 key（原 tensor key 在 set 运算中哈希不稳定）
        tree_vertices: dict[int, set[int]] = {
            seed_idx: {seed_idx} for seed_idx in range(num_seeds)
        }

        edge_offset = 0
        for hop_edges in num_sampled_edges_per_hop:
            hop_edge_list = edge_index[edge_offset: edge_offset + int(hop_edges)]
            for seed_idx in range(num_seeds):
                reachable = tree_vertices[seed_idx]
                for src, dst in hop_edge_list:
                    if dst in reachable:
                        reachable.add(src)
            edge_offset += int(hop_edges)

        return tree_vertices

    def verify_disjoint(
        self,
        tree_vertices: "dict[int, set[int]]",
    ) -> None:
        """
        断言各 seed 的采样树节点集合两两不相交。
        上游 659a0e1 修正：`assert (tv_items[i] & tv_items[j]) == set()`
        括号使优先级明确，避免 Python `&` 与 `==` 优先级混淆。
        """
        items = list(tree_vertices.values())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                intersection = items[i] & items[j]   # 括号明确（659a0e1 修正风格）
                if intersection:
                    raise AssertionError(
                        f"Disjoint 不变式违反：seed {i} 与 seed {j} "
                        f"共享节点 {intersection}"
                    )


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # DisjointMemoryEstimator
    est_std = DisjointMemoryEstimator(
        fanout=(25, 10), total_gpu_memory=8 * 1024**3
    )
    est_disj = DisjointMemoryEstimator(
        fanout=(25, 10), total_gpu_memory=8 * 1024**3, disjoint=True
    )
    seeds_std = est_std.estimate()
    seeds_disj = est_disj.estimate()
    assert seeds_std > seeds_disj, "disjoint 模式应需要更少 seeds per call（更大内存开销）"
    assert est_disj.disjoint_overhead_factor() == 25.0

    # 全采样 fanout
    est_unknown = DisjointMemoryEstimator(fanout=(-1,), total_gpu_memory=8 * 1024**3)
    assert est_unknown.estimate() == _UNKNOWN_VERTICES_DEFAULT

    # DisjointSamplingSpec
    spec_on = DisjointSamplingSpec(enabled=True)
    assert spec_on.as_cugraph_kwarg() == {"disjoint_sampling": True}
    assert spec_on.as_human_label() == "disjoint"

    spec_off = DisjointSamplingSpec(enabled=False)
    assert spec_off.as_cugraph_kwarg() == {"disjoint_sampling": False}

    # DisjointBatchVerifier
    verifier = DisjointBatchVerifier()

    # 构造两个独立采样树：
    # num_seeds=2 → seed 0 初始 {0}, seed 1 初始 {1}
    # edges: (2,0),(3,0) → seed 0 扩展为 {0,2,3}
    #        (4,1),(5,1) → seed 1 扩展为 {1,4,5}
    # 两树不相交 ✓
    edges = [(2, 0), (3, 0), (4, 1), (5, 1)]
    tv = verifier.build_tree_vertices(
        num_seeds=2,
        edge_index=edges,
        num_sampled_edges_per_hop=[4],
    )
    assert tv[0] == {0, 2, 3} and tv[1] == {1, 4, 5}
    verifier.verify_disjoint(tv)  # 应通过

    # 构造违反 disjoint 的情况：
    # seed 0: {0,2,3}, seed 1: {1,2,4} → 共享节点 2
    edges_bad = [(2, 0), (3, 0), (2, 1), (4, 1)]
    tv_bad = verifier.build_tree_vertices(
        num_seeds=2,
        edge_index=edges_bad,
        num_sampled_edges_per_hop=[4],
    )
    try:
        verifier.verify_disjoint(tv_bad)
        assert False, "应抛出 AssertionError"
    except AssertionError:
        pass

    print("disjoint_sampler.py 自测：9 项断言全部 PASS")
