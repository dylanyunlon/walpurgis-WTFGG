# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 474b10e
# 原标题: [BUG] Fix Broken DGL Test (#223)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「穷人的孩子，蚤熟世事，也许倒不是幸福。」—— 鲁迅《故乡》
#
# 474b10e 修复: pylibcugraph.uniform_neighbor_sample() 新版 API
# 废弃并移除了 with_edge_properties=True 和 return_dict=True 两个参数。
# 上游测试在调用中仍然传入这两个参数，导致 TypeError 崩溃。
# 修复方式：直接删除这两个参数，新版 API 默认返回带边属性的 dict 格式。
#
# 原始变更文件:
#   - python/cugraph-dgl/cugraph_dgl/tests/test_graph.py    (2 deletions)
#   - python/cugraph-dgl/cugraph_dgl/tests/test_graph_mg.py (2 deletions)
#
# Walpurgis 迁移：
#   - 使用 walpurgis.graph.Graph 替代 cugraph_dgl.Graph
#   - 使用 conftest.karate_bipartite (_KarateBipartiteInfo dataclass)
#     替代上游 tuple 返回值，消除魔法索引
#   - test_graph_mg.py 路径（多 GPU）在 Walpurgis 架构中暂不支持，
#     对应测试标记 @pytest.mark.skip 并注释说明
#
# Walpurgis 20% 改写要点（鲁迅拿法）：
#   1. _SamplingOutputInspector dataclass — 封装 sampling_output 验证逻辑，
#      把上游散落在 test body 内的 for 循环 + 3个 assert 提取为
#      assert_all_edges_valid() 方法，含 WALPURGIS_DEBUG 输出每条边的校验结果
#   2. test body 内的关键步骤加 WALPURGIS_DEBUG 断点 print
#   3. 同构图测试保留完整 pylibcugraph.degrees 验证路径（上游 test_graph.py 同款）
#   4. 474b10e 修复点在注释中明确标注：「474b10e: 删除 with_edge_properties / return_dict」

import os
import sys
import numpy as np
import pytest
import pylibcugraph
import cupy

from cugraph.utilities.utils import import_optional, MissingModule

from .conftest import _KarateBipartiteInfo

torch = import_optional("torch")
dgl = import_optional("dgl")

os.environ.setdefault("WALPURGIS_DEBUG", "1")

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含文件前缀。"""
    if _DEBUG:
        print(
            f"[WALPURGIS test_graph:{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Walpurgis 改写: SamplingOutputInspector
# 上游 test_graph_make_heterogeneous_graph 在 test body 里内联了
#   for i, etype in enumerate(sampling_output["edge_type"].tolist()): ...
# 难以单独 DEBUG。此 dataclass 封装校验逻辑，加断点打印每条边的校验结果。
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


@dataclass
class _SamplingOutputInspector:
    """
    封装 pylibcugraph.uniform_neighbor_sample 输出的边校验逻辑。

    migrate 474b10e: 新版 API 去掉 with_edge_properties/return_dict 后，
    sampling_output 仍然是包含 majors/minors/edge_id/edge_type 的 dict；
    此类验证输出格式正确且所有采样边均合法。

    Attributes
    ----------
    sampling_output : dict
        pylibcugraph.uniform_neighbor_sample 的返回值（dict 格式）。
    expected_etypes : dict
        edge_type int → etype string，如 {0: "e1", 1: "e2", ...}。
    expected_offsets : dict
        edge_type int → (src_offset, dst_offset)，用于还原全局节点 ID。
    direction : str
        采样方向 "in" 或 "out"，决定 majors/minors 对应 src/dst 的映射。
    """
    sampling_output: Dict[str, Any]
    expected_etypes: Dict[int, str]
    expected_offsets: Dict[int, Tuple[int, int]]
    direction: str

    @property
    def src_col(self) -> str:
        """474b10e 修复后 dict 键名不变：direction=in 时 majors=dst，minors=src。"""
        return "minors" if self.direction == "in" else "majors"

    @property
    def dst_col(self) -> str:
        return "majors" if self.direction == "in" else "minors"

    def assert_output_is_dict(self) -> None:
        """
        474b10e 核心验证：新版 API 删除 return_dict=True 后，
        输出应自动为 dict 格式（而非旧版 namedtuple）。
        """
        assert isinstance(self.sampling_output, dict), (
            f"474b10e: uniform_neighbor_sample 应返回 dict，"
            f"实际类型: {type(self.sampling_output).__name__}"
        )
        required_keys = {"majors", "minors", "edge_id", "edge_type"}
        missing = required_keys - set(self.sampling_output.keys())
        assert not missing, (
            f"474b10e: sampling_output 缺少键: {missing}"
        )
        _dbg(
            "assert_output_is_dict",
            f"✓ output 为 dict，keys={sorted(self.sampling_output.keys())}",
        )

    def assert_all_edges_valid(self, graph) -> None:
        """
        校验所有采样边在原图中合法。

        上游 test body 的 for 循环 + 3 个 assert 被提取至此，
        加 WALPURGIS_DEBUG 断点输出每条边的校验摘要。
        """
        edge_types = self.sampling_output["edge_type"].tolist()
        total = len(edge_types)

        _dbg(
            "assert_all_edges_valid",
            f"direction={self.direction!r} total_sampled_edges={total}",
        )

        for i, etype_int in enumerate(edge_types):
            eid = int(self.sampling_output["edge_id"][i])
            etype_str = self.expected_etypes[etype_int]
            src_off, dst_off = self.expected_offsets[etype_int]

            srcs, dsts, eids = graph.edges("all", etype=etype_str, device="cpu")

            # eids 断言
            assert eids[eid] == eid, (
                f"边 #{i} etype={etype_str} eid={eid}: eids[eid]={eids[eid]} ≠ {eid}"
            )
            # src 断言
            expected_src = int(self.sampling_output[self.src_col][i]) - src_off
            assert srcs[eid] == expected_src, (
                f"边 #{i} etype={etype_str} eid={eid}: "
                f"srcs[eid]={int(srcs[eid])} ≠ {expected_src} "
                f"(raw={int(self.sampling_output[self.src_col][i])} off={src_off})"
            )
            # dst 断言
            expected_dst = int(self.sampling_output[self.dst_col][i]) - dst_off
            assert dsts[eid] == expected_dst, (
                f"边 #{i} etype={etype_str} eid={eid}: "
                f"dsts[eid]={int(dsts[eid])} ≠ {expected_dst} "
                f"(raw={int(self.sampling_output[self.dst_col][i])} off={dst_off})"
            )

            if _DEBUG and i < 5:
                _dbg(
                    "assert_all_edges_valid",
                    f"  edge #{i}: etype={etype_str} eid={eid} "
                    f"src={int(srcs[eid])} dst={int(dsts[eid])} ✓",
                )

        _dbg(
            "assert_all_edges_valid",
            f"✓ {total} 条采样边全部校验通过",
        )


# ---------------------------------------------------------------------------
# test: 同构图构建
# 迁移自上游 test_graph.py::test_graph_make_homogeneous_graph
# 使用 pylibcugraph.degrees 验证图结构（不涉及 474b10e 修复的参数）
# ---------------------------------------------------------------------------

@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_homogeneous_graph(direction):
    """
    验证 walpurgis.graph.Graph 构建同构图后，节点度数与参考图一致。

    migrate 474b10e: 同构图路径不调用 uniform_neighbor_sample，
    无需删参数，但作为基础健康检查一并迁移。
    使用 pylibcugraph.degrees 替代采样调用验证图结构合法性。
    """
    from walpurgis.graph.graph import Graph

    # 轻量内联图（避免 cugraph.datasets.karate 加载开销）
    srcs = np.array([0, 1, 2, 3, 4, 0, 2, 4], dtype="int64")
    dsts = np.array([1, 2, 3, 4, 0, 3, 4, 1], dtype="int64")
    num_nodes = 5
    wgt = np.random.random((len(srcs),))
    node_x = np.random.random((num_nodes,))

    graph = Graph()
    graph.add_nodes(
        num_nodes,
        data={
            "num": torch.arange(num_nodes, dtype=torch.int64),
            "x": node_x,
        },
    )
    graph.add_edges(srcs, dsts, {"weight": wgt})

    _dbg(
        "test_homogeneous",
        f"direction={direction!r} num_nodes={num_nodes} num_edges={len(srcs)}",
    )

    plc_graph = graph._graph(direction=direction)

    # 基本断言
    assert graph.num_nodes() == num_nodes, f"期望 {num_nodes} 节点，实际 {graph.num_nodes()}"
    assert graph.num_edges() == len(srcs), f"期望 {len(srcs)} 边，实际 {graph.num_edges()}"
    assert graph.is_homogeneous, "同构图 is_homogeneous 应为 True"
    assert not graph.is_multi_gpu, "单机测试 is_multi_gpu 应为 False"

    # 节点 ID 验证
    assert (
        graph.nodes() == torch.arange(num_nodes, dtype=torch.int64, device="cuda")
    ).all(), "graph.nodes() 应为连续整数"

    # 节点特征验证
    emb = graph.nodes[None]["x"]
    assert emb is not None
    assert (emb() == torch.as_tensor(node_x, device="cuda")).all(), "节点特征 x 不一致"

    # 使用 pylibcugraph.degrees 验证图结构（474b10e 修复无关路径）
    verts_cp = cupy.arange(num_nodes, dtype="int64")
    v_actual, d_in_actual, d_out_actual = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_graph,
        source_vertices=verts_cp,
        do_expensive_check=True,
    )

    # 参考图（正向）
    ref_src = srcs if direction == "out" else dsts
    ref_dst = dsts if direction == "out" else srcs
    plc_ref = pylibcugraph.SGGraph(
        pylibcugraph.ResourceHandle(),
        pylibcugraph.GraphProperties(is_multigraph=True, is_symmetric=False),
        cupy.asarray(ref_src),
        cupy.asarray(ref_dst),
        vertices_array=verts_cp,
    )
    v_exp, d_in_exp, d_out_exp = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_ref,
        source_vertices=verts_cp,
        do_expensive_check=True,
    )

    assert (v_actual == v_exp).all(), "节点 ID 度数表不一致"
    assert (d_in_actual == d_in_exp).all(), f"入度不一致 direction={direction!r}"
    assert (d_out_actual == d_out_exp).all(), f"出度不一致 direction={direction!r}"

    _dbg("test_homogeneous", f"✓ direction={direction!r} 全部断言通过")


# ---------------------------------------------------------------------------
# test: 异构图构建 + 采样 API 修复验证
# 迁移自上游 test_graph.py::test_graph_make_heterogeneous_graph
# 核心：474b10e 删除 with_edge_properties=True 和 return_dict=True
# ---------------------------------------------------------------------------

@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_heterogeneous_graph(direction, karate_bipartite):
    """
    验证 walpurgis.graph.Graph 构建异构图正确，
    并以 pylibcugraph.uniform_neighbor_sample 采样验证边合法性。

    474b10e 修复核心：
      旧版调用（已废弃，会触发 TypeError）：
        pylibcugraph.uniform_neighbor_sample(
            ...,
            with_edge_properties=True,   ← 474b10e 删除此行
            prior_sources_behavior="exclude",
            return_dict=True,             ← 474b10e 删除此行
        )
      新版调用（本测试使用）：
        pylibcugraph.uniform_neighbor_sample(
            ...,
            prior_sources_behavior="exclude",
        )
      新版 API 默认返回 dict 且包含边属性，无需显式传入上述两参数。

    Walpurgis 改写：_SamplingOutputInspector 封装边校验逻辑。
    """
    info: _KarateBipartiteInfo = karate_bipartite
    graph = info.graph
    num_nodes_group_1 = info.num_nodes_group_1
    num_nodes_group_2 = info.num_nodes_group_2

    _dbg(
        "test_heterogeneous",
        f"direction={direction!r} n1={num_nodes_group_1} n2={num_nodes_group_2}",
    )

    # 基本图结构断言
    assert not graph.is_homogeneous, "异构图 is_homogeneous 应为 False"
    assert not graph.is_multi_gpu, "单机测试 is_multi_gpu 应为 False"

    total_nodes = num_nodes_group_1 + num_nodes_group_2

    assert (
        graph.nodes()
        == torch.arange(total_nodes, dtype=torch.int64, device="cuda")
    ).all(), "全局节点 ID 不连续"

    assert (
        graph.nodes("type1")
        == torch.arange(num_nodes_group_1, dtype=torch.int64, device="cuda")
    ).all(), "type1 节点 ID 不正确"

    assert (
        graph.nodes("type2")
        == torch.arange(num_nodes_group_2, dtype=torch.int64, device="cuda")
    ).all(), "type2 节点 ID 不正确"

    # 各边类型 eid 验证
    for etype_key in [
        ("type1", "e1", "type1"),
        ("type1", "e2", "type2"),
        ("type2", "e3", "type1"),
        ("type2", "e4", "type2"),
    ]:
        eids_actual = graph.edges("eid", etype=etype_key)
        expected_len = len(info.edges[etype_key][0])  # (src_arr, dst_arr)
        assert (
            eids_actual == torch.arange(expected_len, dtype=torch.int64, device="cuda")
        ).all(), f"边类型 {etype_key} eid 序列不正确"
        _dbg(
            "test_heterogeneous",
            f"  etype={etype_key} n_edges={expected_len} eid 验证 ✓",
        )

    # -----------------------------------------------------------------------
    # 474b10e 修复核心：调用 uniform_neighbor_sample，不传废弃参数
    # -----------------------------------------------------------------------
    plc_graph = graph._graph(direction)

    _dbg(
        "test_heterogeneous",
        f"调用 uniform_neighbor_sample: start_list size={total_nodes} "
        f"h_fan_out=[1,1] — 474b10e: 不传 with_edge_properties / return_dict",
    )

    # 474b10e: 删除了 with_edge_properties=True 和 return_dict=True
    sampling_output = pylibcugraph.uniform_neighbor_sample(
        pylibcugraph.ResourceHandle(),
        plc_graph,
        start_list=cupy.arange(total_nodes, dtype="int64"),
        h_fan_out=np.array([1, 1], dtype="int32"),
        with_replacement=False,
        do_expensive_check=True,
        prior_sources_behavior="exclude",
        # with_edge_properties=True,  ← 474b10e: 已删除（API 废弃）
        # return_dict=True,           ← 474b10e: 已删除（API 废弃）
    )

    _dbg(
        "test_heterogeneous",
        f"采样完成，output 类型={type(sampling_output).__name__} "
        f"keys={sorted(sampling_output.keys()) if isinstance(sampling_output, dict) else 'N/A'}",
    )

    # -----------------------------------------------------------------------
    # 使用 _SamplingOutputInspector 校验所有采样边
    # -----------------------------------------------------------------------
    inspector = _SamplingOutputInspector(
        sampling_output=sampling_output,
        expected_etypes={
            0: "e1",
            1: "e2",
            2: "e3",
            3: "e4",
        },
        expected_offsets={
            0: (0, 0),
            1: (0, num_nodes_group_1),
            2: (num_nodes_group_1, 0),
            3: (num_nodes_group_1, num_nodes_group_1),
        },
        direction=direction,
    )

    inspector.assert_output_is_dict()
    inspector.assert_all_edges_valid(graph)

    _dbg("test_heterogeneous", f"✓ direction={direction!r} 全部断言通过")


# ---------------------------------------------------------------------------
# test: MG (多 GPU) 路径说明
# 对应上游 test_graph_mg.py::run_test_graph_make_heterogeneous_graph_mg
# Walpurgis 架构暂不支持 MG 路径，记录原因并 skip
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason=(
        "474b10e MG 路径（test_graph_mg.py）在 Walpurgis 架构中暂不支持。"
        "Walpurgis 单机单 GPU，不依赖 torch.distributed + MG 图对象。"
        "上游 MG 修复（删除 with_edge_properties/return_dict）已在 "
        "test_graph_make_heterogeneous_graph 的 SG 路径中完整覆盖。"
    )
)
def test_graph_make_heterogeneous_graph_mg():
    """占位：474b10e MG 测试路径，Walpurgis 架构暂不支持，见 skip reason。"""
    pass
