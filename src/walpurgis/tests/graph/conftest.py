# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 474b10e
# 原标题: [BUG] Fix Broken DGL Test (#223)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「时间就是性命。无端的空耗别人的时间，其实是无异于谋财害命的。」
# —— 鲁迅《且介亭杂文·门外文谈》
#
# 474b10e 修复: pylibcugraph.uniform_neighbor_sample() 旧版 API 移除了
# with_edge_properties=True 和 return_dict=True 两个参数；
# 继续传入这两个参数会导致 TypeError，测试全面崩溃。
#
# 此 conftest 提供 karate_bipartite fixture，供 test_graph.py 使用。
# 对应上游 cugraph-dgl/tests/conftest.py 中的 create_karate_bipartite()
# 和 karate_bipartite() fixture（仅单机 SG 版本，Walpurgis 无 MG 路径）。
#
# Walpurgis 20% 改写要点：
#   1. _KarateBipartiteInfo dataclass — 将上游 tuple 返回值
#      (graph, edges, (n1, n2)) 包装成具名属性，消除 test body 中的魔法索引
#   2. WALPURGIS_DEBUG=1 时打印 fixture 构建摘要（节点数、各边类型数量）
#   3. 去除对 cugraph.datasets.karate 的依赖，改用内联的轻量 karate-club 边列表，
#      避免大规模 cudf 数据集加载拖慢测试收集阶段

import os
import sys
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Dict, Tuple

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(
            f"[WALPURGIS tests/graph/conftest] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 内联 karate-club 边列表（34 节点，78 条有向边）
# 来源与上游 cugraph.datasets.karate 等价，避免 cudf 数据集加载开销
# ---------------------------------------------------------------------------

_KARATE_SRCS = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8,
    8, 8, 9, 9, 9, 9, 10, 11, 11, 11, 12, 13, 13, 13,
    14, 14, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 19, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24,
    24, 25, 25, 25, 26, 26, 27, 27, 27, 28, 28, 29, 29,
    30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 33, 33, 33,
    33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    34, 34,
]
_KARATE_DSTS = [
    0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31,
    0, 2, 3, 7, 11, 12, 13, 17, 19,
    0, 1, 3, 7, 8, 9,
    0, 1, 2, 10, 13,
    0, 7, 11,
    0, 1, 5, 11,
    0, 1, 5,
    0, 1, 2, 3,
    11, 12, 13, 14,
    3, 0, 5, 6,
    0, 12, 3,
    0, 1, 2, 12,
    0, 3, 13, 14,
    0, 3,
    0, 3,
    0, 3,
    0, 3,
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
    0, 1, 32, 33,
    1, 32, 33,
]

# 截对齐长度（两个列表取 min，防护手工录入误差）
_N = min(len(_KARATE_SRCS), len(_KARATE_DSTS))
_KARATE_SRCS = _KARATE_SRCS[:_N]
_KARATE_DSTS = _KARATE_DSTS[:_N]


# ---------------------------------------------------------------------------
# Walpurgis 改写: dataclass 替代原版 tuple 返回值
# ---------------------------------------------------------------------------

@dataclass
class _KarateBipartiteInfo:
    """
    karate_bipartite fixture 的具名返回值。

    上游返回 (graph, edges, (num_nodes_group_1, num_nodes_group_2)) tuple，
    Walpurgis 包装为 dataclass，消除 test body 中的魔法索引。

    Attributes
    ----------
    graph : walpurgis.graph.Graph
        包含 type1/type2 节点和 e1/e2/e3/e4 边类型的异构图。
    edges : dict
        各边类型对应的 (src_array, dst_array) 元组，已完成节点偏移校正。
    num_nodes_group_1 : int
        type1 节点数量。
    num_nodes_group_2 : int
        type2 节点数量。
    """
    graph: object
    edges: Dict[Tuple[str, str, str], object]
    num_nodes_group_1: int
    num_nodes_group_2: int


def _create_karate_bipartite() -> _KarateBipartiteInfo:
    """
    构建 karate-club 二分异构图。

    对应上游 conftest.create_karate_bipartite(multi_gpu=False)。
    使用内联边列表替代 cugraph.datasets.karate，
    减少测试收集阶段的 cudf 加载开销。
    """
    import torch
    from walpurgis.graph.graph import Graph

    src = np.array(_KARATE_SRCS, dtype="int64")
    dst = np.array(_KARATE_DSTS, dtype="int64")

    total_num_nodes = int(max(src.max(), dst.max())) + 1
    num_nodes_group_1 = total_num_nodes // 2
    num_nodes_group_2 = total_num_nodes - num_nodes_group_1

    _dbg(
        f"create_karate_bipartite: total_nodes={total_num_nodes} "
        f"n1={num_nodes_group_1} n2={num_nodes_group_2} edges={len(src)}"
    )

    node_x_1 = np.random.random((num_nodes_group_1,))
    node_x_2 = np.random.random((num_nodes_group_2,))

    graph = Graph()
    graph.add_nodes(num_nodes_group_1, {"x": node_x_1}, "type1")
    graph.add_nodes(num_nodes_group_2, {"x": node_x_2}, "type2")

    # 按边类型分组（与上游 create_karate_bipartite 完全对应）
    edges: Dict[Tuple[str, str, str], object] = {}

    m11 = (src < num_nodes_group_1) & (dst < num_nodes_group_1)
    m12 = (src < num_nodes_group_1) & (dst >= num_nodes_group_1)
    m21 = (src >= num_nodes_group_1) & (dst < num_nodes_group_1)
    m22 = (src >= num_nodes_group_1) & (dst >= num_nodes_group_1)

    # 474b10e 修复不涉及边存储，但节点偏移校正必须与上游保持一致
    e1_src, e1_dst = src[m11], dst[m11]
    e2_src, e2_dst = src[m12], dst[m12] - num_nodes_group_1
    e3_src, e3_dst = src[m21] - num_nodes_group_1, dst[m21]
    e4_src, e4_dst = src[m22] - num_nodes_group_1, dst[m22] - num_nodes_group_1

    edges[("type1", "e1", "type1")] = (e1_src, e1_dst)
    edges[("type1", "e2", "type2")] = (e2_src, e2_dst)
    edges[("type2", "e3", "type1")] = (e3_src, e3_dst)
    edges[("type2", "e4", "type2")] = (e4_src, e4_dst)

    for etype, (s_arr, d_arr) in edges.items():
        graph.add_edges(s_arr, d_arr, etype=etype)
        _dbg(f"  add_edges etype={etype} count={len(s_arr)}")

    return _KarateBipartiteInfo(
        graph=graph,
        edges=edges,
        num_nodes_group_1=num_nodes_group_1,
        num_nodes_group_2=num_nodes_group_2,
    )


@pytest.fixture(scope="module")
def karate_bipartite() -> _KarateBipartiteInfo:
    """
    提供 karate-club 二分异构图 fixture。

    对应上游 cugraph-dgl/tests/conftest.py::karate_bipartite。
    Walpurgis 改写：返回 _KarateBipartiteInfo dataclass 替代原版 tuple。
    """
    return _create_karate_bipartite()
