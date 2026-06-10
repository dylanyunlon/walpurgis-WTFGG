# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg
# Walpurgis 迁移测试: 分布式邻居采样器 (单机单 GPU)
#
# 迁移自 cugraph-gnn 03292cf:
#   python/cugraph-pyg/cugraph_pyg/tests/sampler/test_distributed_sampler.py
#
# 20% 改写要点：
#   - 加 WALPURGIS_DEBUG=1 固定，确保每次 pytest 都输出断点信息（可按需关闭）
#   - 拆出 _build_hetero_graph() fixture，避免 test body 中 30 行图构建代码重复
#   - 每个关键断言前加 print 注释，便于 CI 失败时快速定位
#   - 保持与上游完全一致的数值断言

import os
import pytest
import cupy

os.environ.setdefault("WALPURGIS_DEBUG", "1")  # 测试期间开启断点输出

from walpurgis.sampler import DistributedNeighborSampler

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")


# ---------------------------------------------------------------------------
# Fixture：构建异构测试图（迁移20%改写：抽取为 fixture）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hetero_sg_graph():
    """
    构建一个 10 节点、14 条边、2 种边类型的有向异构图。

    节点：0-9
    边类型 0：src 在 [4,9]，dst 在 [0,3]  (bipartite-ish)
    边类型 1：src/dst 均在 [4,9]            (intra-type)
    """
    props = GraphProperties(is_symmetric=False, is_multigraph=True)
    handle = ResourceHandle()

    srcs = cupy.array([4, 5, 6, 7, 8, 9, 8, 9, 8, 7, 6, 5, 4, 5])
    dsts = cupy.array([0, 1, 2, 3, 3, 0, 4, 5, 6, 8, 7, 8, 9, 9])
    eids = cupy.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7])
    etps = cupy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int32")

    graph = SGGraph(
        handle,
        props,
        srcs,
        dsts,
        vertices_array=cupy.arange(10),
        edge_id_array=eids,
        edge_type_array=etps,
        weight_array=cupy.ones((14,), dtype="float32"),
    )

    print(
        f"\n[fixture] hetero_sg_graph 构建完毕 | "
        f"nodes=10 edges=14 edge_types=2",
        flush=True,
    )
    return graph


# ---------------------------------------------------------------------------
# 测试：异构图节点采样
# ---------------------------------------------------------------------------

@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_hetero_from_nodes(hetero_sg_graph):
    """
    验证 DistributedNeighborSampler 在异构图上进行全邻居采样（fanout=-1）的正确性。

    断言依据：给定种子节点 [4, 5]，2 跳全邻居，期望采样到的边 id 及端点集合确定。
    """
    print(f"\n[test] 构建 DistributedNeighborSampler ...", flush=True)

    sampler = DistributedNeighborSampler(
        hetero_sg_graph,
        fanout=[-1, -1, -1, -1],
        compression="COO",
        heterogeneous=True,
        vertex_type_offsets=cupy.array([0, 4, 10]),
        num_edge_types=2,
        deduplicate_sources=True,
        biased=False,
    )

    print(f"[test] 执行 sample_from_nodes | seeds=[4,5] input_id=[5,10]", flush=True)

    out_iter = sampler.sample_from_nodes(
        nodes=cupy.array([4, 5]),
        input_id=cupy.array([5, 10]),
    )

    out = list(out_iter)
    print(f"[test] 迭代完毕 | out 长度={len(out)}", flush=True)

    assert len(out) == 1, f"期望 1 个 call_group 输出，实际 {len(out)}"
    out, _, _ = out[0]

    lho = out["label_type_hop_offsets"]
    print(f"[test] label_type_hop_offsets={lho.tolist()}", flush=True)

    # ---------------------------------------------------------------
    # 边类型 0 断言
    # ---------------------------------------------------------------
    emap_0 = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][0] : out["edge_renumber_map_offsets"][1]
    ]
    smap = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]
    dmap = out["map"][out["renumber_map_offsets"][0] : out["renumber_map_offsets"][1]]

    # 边类型 0，hop 0
    hs, he = int(lho[0]), int(lho[1])
    print(f"[test] 边类型0 hop0: offset=[{hs},{he}]", flush=True)
    assert he - hs == 2, f"边类型0 hop0: 期望2条边，实际 {he - hs}"
    e = emap_0[out["edge_id"][hs:he]]
    assert sorted(e.tolist()) == [0, 1], f"边类型0 hop0 edge_id 错误: {sorted(e.tolist())}"
    s = cupy.asarray(smap[out["majors"][hs:he]])
    d = cupy.asarray(dmap[out["minors"][hs:he]])
    assert sorted(s.tolist()) == [4, 5], f"边类型0 hop0 majors 错误: {sorted(s.tolist())}"
    assert sorted(d.tolist()) == [0, 1], f"边类型0 hop0 minors 错误: {sorted(d.tolist())}"

    # 边类型 0，hop 1
    hs, he = int(lho[1]), int(lho[2])
    print(f"[test] 边类型0 hop1: offset=[{hs},{he}]", flush=True)
    assert he - hs == 2, f"边类型0 hop1: 期望2条边，实际 {he - hs}"
    e = emap_0[out["edge_id"][hs:he]]
    assert sorted(e.tolist()) == [4, 5], f"边类型0 hop1 edge_id 错误: {sorted(e.tolist())}"
    s = cupy.asarray(smap[out["majors"][hs:he]])
    d = cupy.asarray(dmap[out["minors"][hs:he]])
    assert sorted(s.tolist()) == [8, 9], f"边类型0 hop1 majors 错误: {sorted(s.tolist())}"
    assert sorted(d.tolist()) == [0, 3], f"边类型0 hop1 minors 错误: {sorted(d.tolist())}"

    # ---------------------------------------------------------------
    # 边类型 1 断言
    # ---------------------------------------------------------------
    emap_1 = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][1] : out["edge_renumber_map_offsets"][2]
    ]
    smap_1 = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]
    dmap_1 = smap_1  # 边类型1 src/dst 同属一个节点类型分区

    # 边类型 1，hop 0
    hs, he = int(lho[2]), int(lho[3])
    print(f"[test] 边类型1 hop0: offset=[{hs},{he}]", flush=True)
    assert he - hs == 3, f"边类型1 hop0: 期望3条边，实际 {he - hs}"
    e = emap_1[out["edge_id"][hs:he]]
    assert sorted(e.tolist()) == [5, 6, 7], f"边类型1 hop0 edge_id 错误: {sorted(e.tolist())}"
    s = cupy.asarray(smap_1[out["majors"][hs:he]])
    d = cupy.asarray(dmap_1[out["minors"][hs:he]])
    assert sorted(s.tolist()) == [4, 5, 5], f"边类型1 hop0 majors 错误: {sorted(s.tolist())}"
    assert sorted(d.tolist()) == [8, 9, 9], f"边类型1 hop0 minors 错误: {sorted(d.tolist())}"

    # 边类型 1，hop 1
    hs, he = int(lho[3]), int(lho[4])
    print(f"[test] 边类型1 hop1: offset=[{hs},{he}]", flush=True)
    assert he - hs == 3, f"边类型1 hop1: 期望3条边，实际 {he - hs}"
    e = emap_1[out["edge_id"][hs:he]]
    assert sorted(e.tolist()) == [0, 1, 2], f"边类型1 hop1 edge_id 错误: {sorted(e.tolist())}"
    s = cupy.asarray(smap_1[out["majors"][hs:he]])
    d = cupy.asarray(dmap_1[out["minors"][hs:he]])
    assert sorted(s.tolist()) == [8, 8, 9], f"边类型1 hop1 majors 错误: {sorted(s.tolist())}"
    assert sorted(d.tolist()) == [4, 5, 6], f"边类型1 hop1 minors 错误: {sorted(d.tolist())}"

    print("[test] ✓ 全部断言通过", flush=True)
