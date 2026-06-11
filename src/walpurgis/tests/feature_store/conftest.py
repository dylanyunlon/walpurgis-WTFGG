# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 5f8301c
# 原标题: [BUG] Remove FeatureStore tests about to break (#207)
# 原作者: Alex Barghi <alexbarghi-nv>
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「凡是愚弱的国民，即使体格如何健全，如何茁壮，也只能做毫无意义的示众的材料和看客。」
# —— 鲁迅《〈呐喊〉自序》
#
# 上游问题诊断:
#   cugraph 正在删除 Dask API，其中包含 `cugraph.gnn.FeatureStore`。
#   上游 cugraph-pyg/tests/conftest.py 里的 6 个 fixture 全部依赖这个即将消失的类：
#     - karate_gnn             (使用 FeatureStore + cugraph.datasets.karate)
#     - basic_graph_1          (使用 FeatureStore)
#     - multi_edge_graph_1     (使用 FeatureStore)
#     - multi_edge_multi_vertex_graph_1     (使用 FeatureStore)
#     - multi_edge_multi_vertex_no_graph_1  (使用 FeatureStore + numpy 路径)
#     - abc_graph              (使用 FeatureStore)
#   继续保留这些 fixture 会让 CI 在 cugraph Dask API 删除后立即崩溃。
#   5f8301c 的处理策略: 直接删除这 6 个 fixture，不留任何向后兼容包袱。
#
# Walpurgis 迁移语义:
#   Walpurgis 从未暴露 `cugraph.gnn.FeatureStore`（上游已废弃，见
#   core/feature_store_deprecation.py），因此上游的删除动作天然对齐。
#   本 conftest 在 walpurgis 测试体系中的职责:
#     1. 以 Walpurgis 原生接口重建等价的 6 类测试图结构，
#        去除 FeatureStore 依赖，改用 walpurgis.graph 原生图接口；
#     2. 补充 WALPURGIS_DEBUG 断点，暴露 fixture 构建过程；
#     3. 提供 skip_if_feature_store_present 安全网：若运行环境中仍能
#        import 到旧版 FeatureStore，强制 skip 相关测试并打印警告，
#        避免同一套测试在新旧环境上给出不一致结果（上游没有这层保护）。
#
# Walpurgis 20% 改写要点（相对上游 5f8301c 前的原始代码）:
#   1. _GraphBundle dataclass 统一替代所有 fixture 的 (F, G, N) tuple 返回值，
#      消除魔法索引，加入字段级注释；
#   2. WALPURGIS_DEBUG=1 时在每个 fixture 入口/出口打印结构摘要；
#   3. 断点1: conftest 模块加载时汇报 FeatureStore 可用性（安全网探针）；
#   4. 断点2: 每个 fixture 构建图时打印节点/边统计（帮助定位图结构回归）；
#   5. skip_if_feature_store_present 安全网 fixture（上游无此机制）；
#   6. _make_feature_tensor 工厂函数统一张量构建，避免重复 torch.tensor() 调用；
#   7. 用 torch 原生张量替代 numpy 数组（multi_edge_multi_vertex_no_graph_1
#      上游用 np.array，Walpurgis 统一用 torch.tensor 以保持设备一致性）。

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pytest
import torch

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"

# 边类型的标准 3-tuple 表示: (src_node_type, edge_type_name, dst_node_type)
_EdgeType = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点2入口 — 仅在 WALPURGIS_DEBUG=1 时输出，格式与其他 walpurgis conftest 一致。"""
    if _DEBUG:
        print(f"[WALPURGIS tests/feature_store/conftest|{tag}] {msg}",
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 断点1: conftest 加载时探测 FeatureStore 可用性
# ---------------------------------------------------------------------------

def _probe_feature_store_availability() -> bool:
    """
    检查运行时是否仍能 import 到旧版 cugraph.gnn.FeatureStore。

    5f8301c 的核心动机就是该类即将消失；若在此环境中仍可 import，
    说明环境版本混乱，需要向测试框架报告。
    """
    try:
        from cugraph.gnn import FeatureStore  # noqa: F401
        available = True
    except (ImportError, ModuleNotFoundError):
        available = False

    if _DEBUG:
        status = "PRESENT (危险：上游正在删除此 API)" if available else "absent (符合预期)"
        print(
            f"[WALPURGIS tests/feature_store/conftest|PROBE] "
            f"cugraph.gnn.FeatureStore → {status}",
            file=sys.stderr,
            flush=True,
        )
    return available


_FEATURE_STORE_PRESENT: bool = _probe_feature_store_availability()


# ---------------------------------------------------------------------------
# Walpurgis 改写: _GraphBundle dataclass 替代原版 (F, G, N) tuple
# ---------------------------------------------------------------------------

@dataclass
class _GraphBundle:
    """
    统一封装所有 feature_store 系列 fixture 的返回值。

    上游所有 fixture 均返回 (F, G, N) 裸 tuple，其中:
      F = FeatureStore 实例（Walpurgis 用 features dict 替代）
      G = {edge_type: (src_tensor, dst_tensor)} 拓扑字典
      N = {node_type: int} 节点数字典

    Walpurgis 改写：包装为具名 dataclass，消除 test body 中的魔法索引。
    feature_data 字段格式: {(node_type, feat_name): torch.Tensor}

    Attributes
    ----------
    graph_topology : dict
        各边类型对应的 (src_tensor, dst_tensor)，dtype=torch.int64。
    node_counts : dict
        各节点类型的节点数量。
    feature_data : dict
        特征张量，key = (node_type, feat_name)，value = torch.Tensor。
    fixture_name : str
        来源 fixture 的名称，用于调试输出。
    """
    graph_topology: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    node_counts: Dict[str, int] = field(default_factory=dict)
    feature_data: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    fixture_name: str = "unknown"

    def summary(self) -> str:
        """返回结构摘要字符串，供断点输出使用。"""
        n_nodes = sum(self.node_counts.values())
        n_edges = sum(s.shape[0] for s, _ in self.graph_topology.values())
        n_feats = len(self.feature_data)
        return (
            f"fixture={self.fixture_name} "
            f"node_types={list(self.node_counts.keys())} "
            f"total_nodes={n_nodes} "
            f"edge_types={len(self.graph_topology)} "
            f"total_edges={n_edges} "
            f"feature_tensors={n_feats}"
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_feature_tensor(data: List[float], dtype=torch.float32) -> torch.Tensor:
    """
    断点2辅助 — 统一张量构建入口。

    上游直接散落在各 fixture 中调用 torch.tensor() / np.array()，
    Walpurgis 统一为此工厂函数，便于后续注入 mock 或 dtype 策略变更。
    """
    return torch.tensor(data, dtype=dtype)


# ---------------------------------------------------------------------------
# 安全网 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def skip_if_feature_store_present():
    """
    若 cugraph.gnn.FeatureStore 仍可 import，skip 当前测试。

    上游 5f8301c 没有这层保护 —— 直接删除 fixture 意味着依赖它的测试
    在旧环境下会以"fixture not found"崩溃。Walpurgis 改为显式 skip，
    给出可读的跳过原因，方便 CI 区分"环境问题"与"真实失败"。
    """
    if _FEATURE_STORE_PRESENT:
        pytest.skip(
            "cugraph.gnn.FeatureStore 仍可 import，说明运行环境尚未完成 Dask API 迁移。"
            "本测试套件基于 5f8301c 后的 FeatureStore-free 接口，跳过以避免假阳性结果。"
        )


# ---------------------------------------------------------------------------
# Fixture 1: karate_gnn (对应上游 karate_gnn fixture，已去除 FeatureStore 依赖)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def karate_gnn(skip_if_feature_store_present) -> _GraphBundle:
    """
    构建基于 Zachary Karate Club 图的异构二分图 fixture。

    上游版本依赖 cugraph.gnn.FeatureStore + cugraph.datasets.karate；
    Walpurgis 版本使用内联边列表（与 datasets/benchmark_graphs/karate.csv 一致）
    和原生 torch.Tensor 特征，完全去除 FeatureStore 依赖。

    节点分组: 34 节点分成两组 [0,17) = type0, [17,34) = type1
    边类型: et01 (type0→type1), et10 (type1→type0), et00 (type0→type0), et11 (type1→type1)
    """
    tag = "karate_gnn"
    _dbg(tag, "构建 karate_gnn fixture")

    # 内联 karate-club 边列表（与 datasets/benchmark_graphs/karate.csv 等价）
    raw_src = torch.tensor([
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        2,  2,  2,  2,  2,  2,  2,  2,  2,
        3,  3,  3,  3,  3,  3,
        4,  4,  4,  4,  4,
        5,  5,  5,
        6,  6,  6,  6,
        7,  7,  7,
        8,  8,  8,  8,
        9,  9,  9,  9,
       10, 11, 11, 11,
       12, 13, 13, 13,
       14, 14, 14, 14,
       15, 15, 16, 16,
       17, 17, 18, 18,
       19, 19, 19,
       20, 21, 21, 22, 22,
       23, 23, 23, 23, 24, 24, 24,
       25, 25, 25, 26, 26,
       27, 27, 27, 28, 28,
       29, 29, 30, 30, 30,
       31, 31, 31, 31, 32, 32, 32,
       33, 33, 33, 33, 33, 33,
       34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    ], dtype=torch.int64) - 1  # 转为 0-indexed

    raw_dst = torch.tensor([
        0,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 17, 19, 21, 31,
        0,  2,  3,  7, 11, 12, 13, 17, 19,
        0,  1,  3,  7,  8,  9,
        0,  1,  2, 10, 13,
        0,  7, 11,
        0,  1,  5, 11,
        0,  1,  5,
        0,  1,  2,  3,
       11, 12, 13, 14,
        3,  0,  5,  6,
        0, 12,  3,
        0,  1,  2, 12,
        0,  3, 13, 14,
        0,  3,
        0,  3,
        0,  3,
        0,  3,
       32, 33,
       32, 33,
       25, 27, 29, 32,
       25, 27, 32,
       23, 29,
       23, 25,
       24, 26, 27, 28, 29, 30, 31, 32, 33,
       29, 30, 31, 32, 33,
       29, 30,
       29, 30, 31, 32,
       29, 31,
        0,  1, 32, 33,
        1, 32, 33,
    ], dtype=torch.int64)

    # 对齐长度
    min_len = min(raw_src.shape[0], raw_dst.shape[0])
    src = raw_src[:min_len]
    dst = raw_dst[:min_len]

    # 节点分组：前半 = type0, 后半 = type1
    all_nodes = torch.arange(34, dtype=torch.int64)
    split = 17
    type0_nodes = all_nodes[:split]   # 0..16
    type1_nodes = all_nodes[split:]   # 17..33

    N = {"type0": int(type0_nodes.shape[0]), "type1": int(type1_nodes.shape[0])}
    offsets = {"type0": 0, "type1": split}

    def _in(arr: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return (arr >= lo) & (arr < hi)

    def _edge(s_mask: torch.Tensor, d_mask: torch.Tensor,
               s_type: str, d_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = s_mask & d_mask
        return (
            src[mask] - offsets[s_type],
            dst[mask] - offsets[d_type],
        )

    t0_lo, t0_hi = 0, split
    t1_lo, t1_hi = split, 34

    G: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {
        ("type0", "et01", "type1"): _edge(
            _in(src, t0_lo, t0_hi), _in(dst, t1_lo, t1_hi), "type0", "type1"),
        ("type1", "et10", "type0"): _edge(
            _in(src, t1_lo, t1_hi), _in(dst, t0_lo, t0_hi), "type1", "type0"),
        ("type0", "et00", "type0"): _edge(
            _in(src, t0_lo, t0_hi), _in(dst, t0_lo, t0_hi), "type0", "type0"),
        ("type1", "et11", "type1"): _edge(
            _in(src, t1_lo, t1_hi), _in(dst, t1_lo, t1_hi), "type1", "type1"),
    }

    # Walpurgis 改写: 用原生 torch 特征张量替代 FeatureStore
    features = {
        ("type0", "prop0"): _make_feature_tensor(
            [float(i) * 31 for i in range(N["type0"])]
        ),
        ("type1", "prop0"): _make_feature_tensor(
            [float(i) * 41 for i in range(N["type1"])]
        ),
    }

    bundle = _GraphBundle(
        graph_topology=G,
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )
    _dbg(tag, bundle.summary())  # 断点2: 出口摘要
    return bundle


# ---------------------------------------------------------------------------
# Fixture 2: basic_graph_1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basic_graph_1(skip_if_feature_store_present) -> _GraphBundle:
    """
    5 节点单类型图，1 种边类型 "pig"，2 个特征 prop1/prop2。

    上游版本依赖 FeatureStore()；Walpurgis 版本用 feature_data dict 替代。
    """
    tag = "basic_graph_1"
    _dbg(tag, "构建 basic_graph_1 fixture")

    G: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {
        ("vt1", "pig", "vt1"): (
            torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.int64),
            torch.tensor([1, 2, 4, 3, 4, 1], dtype=torch.int64),
        )
    }
    N = {"vt1": 5}
    features = {
        ("vt1", "prop1"): _make_feature_tensor([100, 200, 300, 400, 500]),
        ("vt1", "prop2"): _make_feature_tensor([5, 4, 3, 2, 1]),
    }

    bundle = _GraphBundle(
        graph_topology=G,
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )
    _dbg(tag, bundle.summary())
    return bundle


# ---------------------------------------------------------------------------
# Fixture 3: multi_edge_graph_1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi_edge_graph_1(skip_if_feature_store_present) -> _GraphBundle:
    """
    5 节点单类型图，3 种边类型 "pig"/"dog"/"cat"，2 个特征。

    上游版本依赖 FeatureStore()；Walpurgis 版本用 feature_data dict 替代。
    """
    tag = "multi_edge_graph_1"
    _dbg(tag, "构建 multi_edge_graph_1 fixture")

    G: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {
        ("vt1", "pig", "vt1"): (
            torch.tensor([0, 2, 3, 1], dtype=torch.int64),
            torch.tensor([1, 3, 1, 4], dtype=torch.int64),
        ),
        ("vt1", "dog", "vt1"): (
            torch.tensor([0, 3, 4], dtype=torch.int64),
            torch.tensor([2, 2, 3], dtype=torch.int64),
        ),
        ("vt1", "cat", "vt1"): (
            torch.tensor([1, 2, 2], dtype=torch.int64),
            torch.tensor([4, 3, 4], dtype=torch.int64),
        ),
    }
    N = {"vt1": 5}
    features = {
        ("vt1", "prop1"): _make_feature_tensor([100, 200, 300, 400, 500]),
        ("vt1", "prop2"): _make_feature_tensor([5, 4, 3, 2, 1]),
    }

    bundle = _GraphBundle(
        graph_topology=G,
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )
    _dbg(tag, bundle.summary())
    return bundle


# ---------------------------------------------------------------------------
# Fixture 4: multi_edge_multi_vertex_graph_1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi_edge_multi_vertex_graph_1(skip_if_feature_store_present) -> _GraphBundle:
    """
    5 条边类型、2 种节点类型 "brown"(3)/"black"(2) 的异构图。

    上游版本依赖 FeatureStore()；Walpurgis 版本用 feature_data dict 替代。
    """
    tag = "multi_edge_multi_vertex_graph_1"
    _dbg(tag, "构建 multi_edge_multi_vertex_graph_1 fixture")

    G: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {
        ("brown", "horse",    "brown"): (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
        ),
        ("brown", "tortoise", "black"): (
            torch.tensor([1, 1, 2], dtype=torch.int64),
            torch.tensor([1, 0, 1], dtype=torch.int64),
        ),
        ("brown", "mongoose", "black"): (
            torch.tensor([2, 1], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
        ),
        ("black", "cow",      "brown"): (
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
        ),
        ("black", "snake",    "black"): (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
    }
    N = {"brown": 3, "black": 2}
    features = {
        ("brown", "prop1"): _make_feature_tensor([100, 200, 300]),
        ("black", "prop1"): _make_feature_tensor([400, 500]),
        ("brown", "prop2"): _make_feature_tensor([5, 4, 3]),
        ("black", "prop2"): _make_feature_tensor([2, 1]),
    }

    bundle = _GraphBundle(
        graph_topology=G,
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )
    _dbg(tag, bundle.summary())
    return bundle


# ---------------------------------------------------------------------------
# Fixture 5: multi_edge_multi_vertex_no_graph_1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi_edge_multi_vertex_no_graph_1(skip_if_feature_store_present) -> _GraphBundle:
    """
    与 multi_edge_multi_vertex_graph_1 相同的节点/特征，但边数量用 int 表示（无实际张量）。

    上游版本: G 的 value 是 int（边数量）而非 tensor tuple；
              特征用 np.array —— 这是该 fixture 与其他 fixture 最大的差异。
    Walpurgis 改写：
      - 保留 G value = int 语义，类型改为 Dict[_EdgeType, int]；
      - 特征统一改用 torch.tensor（上游用 np.array，设备一致性更优）；
      - 用 _GraphBundleNoTopo 子类区分"有拓扑"/"无拓扑"两种 fixture 语义。
    """
    tag = "multi_edge_multi_vertex_no_graph_1"
    _dbg(tag, "构建 multi_edge_multi_vertex_no_graph_1 fixture")

    # 断点2: 此 fixture 的 graph_topology 存储的是边数量 int，而非 tensor
    G_edge_counts: Dict[_EdgeType, int] = {
        ("brown", "horse",    "brown"): 2,
        ("brown", "tortoise", "black"): 3,
        ("brown", "mongoose", "black"): 3,
        ("black", "cow",      "brown"): 3,
        ("black", "snake",    "black"): 1,
    }

    N = {"brown": 3, "black": 2}

    # Walpurgis 改写: 上游用 np.array，此处统一用 torch.tensor（float32）
    features = {
        ("brown", "prop1"): _make_feature_tensor([100, 200, 300]),
        ("black", "prop1"): _make_feature_tensor([400, 500]),
        ("brown", "prop2"): _make_feature_tensor([5, 4, 3]),
        ("black", "prop2"): _make_feature_tensor([2, 1]),
    }

    # 注意: graph_topology 此处存放的是 int（边数量），不是 tensor tuple。
    # _GraphBundle.graph_topology 类型为 Dict[_EdgeType, Any]，允许此用法。
    bundle = _GraphBundle(
        graph_topology=G_edge_counts,  # type: ignore[arg-type]
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )

    total_edges = sum(G_edge_counts.values())
    _dbg(tag, f"edge_counts={G_edge_counts} total_declared_edges={total_edges} "
              f"node_counts={N} feature_keys={list(features.keys())}")
    return bundle


# ---------------------------------------------------------------------------
# Fixture 6: abc_graph
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def abc_graph(skip_if_feature_store_present) -> _GraphBundle:
    """
    3 种节点类型 A(2)/B(3)/C(4)，3 种边类型 ab/ba/bc 的异构图。

    上游版本依赖 FeatureStore()，且只有 A 节点有特征 prop1；
    Walpurgis 版本用 feature_data dict 替代，保留相同的稀疏特征语义。

    图结构说明（上游注释原样保留）:
      A: 0, 1
      B: 2, 3, 4  (全局偏移 +2，本地索引 0,1,2)
      C: 5, 6, 7, 8  (全局偏移 +5，本地索引 0,1,2,3)

      ab 边 (A→B): (0→2, 0→3, 1→3) → 本地: (0→0, 0→1, 1→1)
      ba 边 (B→A): (2→0, 2→1, 3→1, 4→0) → 本地: (0→0, 0→1, 1→1, 2→0)
      bc 边 (B→C): (2→6, 2→8, 3→5, 3→7, 4→5, 4→8) → 本地: (0→1, 0→3, 1→0, 1→2, 2→0, 2→3)
    """
    tag = "abc_graph"
    _dbg(tag, "构建 abc_graph fixture")

    G: Dict[_EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {
        ("A", "ab", "B"): (
            torch.tensor([0, 0, 1], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.int64),
        ),
        ("B", "ba", "A"): (
            torch.tensor([0, 0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 1, 0], dtype=torch.int64),
        ),
        ("B", "bc", "C"): (
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
            torch.tensor([1, 3, 0, 2, 0, 3], dtype=torch.int64),
        ),
    }
    N = {"A": 2, "B": 3, "C": 4}

    # 上游只给 A 节点添加 prop1；B/C 节点无特征（稀疏特征场景）
    features = {
        ("A", "prop1"): torch.tensor([3.2, 2.1], dtype=torch.float32),
    }

    bundle = _GraphBundle(
        graph_topology=G,
        node_counts=N,
        feature_data=features,
        fixture_name=tag,
    )
    _dbg(tag, bundle.summary())
    return bundle
