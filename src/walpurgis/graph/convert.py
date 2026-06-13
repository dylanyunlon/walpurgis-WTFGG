# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — cugraph_dgl/convert.py 新增
#         cugraph_dgl_graph_from_heterograph()
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「凡是愚弱的国民，即使体格如何健全，如何茁壮，也只能做毫无意义的示众的材料
#   和看客，病死多少是不必以为不幸的。」
# —— 鲁迅《呐喊·自序》
#
# 上游 convert.py 在 f4ca484 中新增了 cugraph_dgl_graph_from_heterograph——
# 将 dgl.DGLGraph 转换为 cugraph_dgl.Graph 的工厂函数。
# 单独的一个函数，逻辑很直观，但上游没有任何入口断点可以观察转换过程。
#
# Walpurgis 改写：
#   1. 将函数重命名为 graph_from_heterograph（去掉冗余的 cugraph_dgl_ 前缀）
#      并保留 cugraph_dgl_graph_from_heterograph 别名向后兼容
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - 函数入口：input_graph 的 ntypes/etypes/num_nodes/num_edges
#      - 同构/异构分支选择
#      - add_nodes/add_edges 调用前的参数摘要
#
# adb4006 迁移注记 (fix circular import):
#   上游 cugraph_dgl/convert.py 在 456d5a2 后出现循环导入:
#     import cugraph_dgl                          # 触发 cugraph_dgl/__init__.py
#     from cugraph_dgl import CuGraphStorage      # __init__.py 又 import convert.py
#   修复: 改为直接导入底层模块，绕过 __init__.py:
#     from cugraph_dgl.cugraph_storage import CuGraphStorage  # 直接路径，无循环
#   Walpurgis 中 convert.py 从未使用 cugraph_dgl 包路径（Graph 来自 walpurgis.graph），
#   故不存在对应的循环导入问题；adb4006 的修复模式已在本文件架构设计时规避。
#   详见 src/walpurgis/core/circular_import_fix.py

import os as _os
import sys as _sys
import time as _time
from typing import Optional

from walpurgis.utils.imports import import_optional
from walpurgis.graph.graph import Graph

dgl = import_optional("dgl")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-CONVERT:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


def graph_from_heterograph(
    input_graph: "dgl.DGLGraph",
    single_gpu: bool = True,
    ndata_storage: str = "torch",
    edata_storage: str = "torch",
    **kwargs,
) -> Graph:
    """
    将 dgl.DGLGraph 转换为 walpurgis.graph.Graph。

    f4ca484 新增：原名 cugraph_dgl_graph_from_heterograph。
    Walpurgis 简化名称：graph_from_heterograph。

    Parameters
    ----------
    input_graph : dgl.DGLGraph
        待转换的 DGL 图。
    single_gpu : bool (default=True)
        若 False，则启用多 GPU 分布式存储。
    ndata_storage : str (default='torch')
        节点特征存储后端（'torch' 或 'wholegraph'）。
    edata_storage : str (default='torch')
        边特征存储后端（'torch' 或 'wholegraph'）。
    **kwargs
        传递给 WholeFeatureStore 的可选参数。

    Returns
    -------
    Graph
        转换后的 walpurgis.graph.Graph 对象。
    """
    _dbg(
        "graph_from_heterograph",
        f"ntypes={input_graph.ntypes} etypes={input_graph.etypes} "
        f"num_nodes={input_graph.num_nodes()} num_edges={input_graph.num_edges()} "
        f"single_gpu={single_gpu}",
    )

    output_graph = Graph(
        is_multi_gpu=(not single_gpu),
        ndata_storage=ndata_storage,
        edata_storage=edata_storage,
        **kwargs,
    )

    # ---------------------------------------------------------------
    # 节点：同构图（ntypes <= 1）直接添加；异构图逐类型添加
    # ---------------------------------------------------------------
    if len(input_graph.ntypes) <= 1:
        ntype = input_graph.ntypes[0]
        _dbg(
            "graph_from_heterograph",
            f"同构节点 ntype={ntype!r} num_nodes={input_graph.num_nodes()}",
        )
        output_graph.add_nodes(
            input_graph.num_nodes(),
            data=input_graph.ndata,
            ntype=ntype,
        )
    else:
        _dbg(
            "graph_from_heterograph",
            f"异构节点 ntypes={input_graph.ntypes}",
        )
        for ntype in input_graph.ntypes:
            # 筛出本 ntype 的特征，格式：{feat_name: tensor}
            data = {
                k: v_dict[ntype]
                for k, v_dict in input_graph.ndata.items()
                if ntype in v_dict
            }
            _dbg(
                "graph_from_heterograph",
                f"  add_nodes ntype={ntype!r} "
                f"num={input_graph.num_nodes(ntype)} data_keys={list(data.keys())}",
            )
            output_graph.add_nodes(
                input_graph.num_nodes(ntype), data=data, ntype=ntype
            )

    # ---------------------------------------------------------------
    # 边：同构图（canonical_etypes <= 1）直接添加；异构图逐类型添加
    # ---------------------------------------------------------------
    if len(input_graph.canonical_etypes) <= 1:
        can_etype = input_graph.canonical_etypes[0]
        src_t, dst_t = input_graph.edges(form="uv", etype=can_etype)
        _dbg(
            "graph_from_heterograph",
            f"同构边 etype={can_etype!r} num_edges={src_t.shape[0]}",
        )
        output_graph.add_edges(src_t, dst_t, input_graph.edata, etype=can_etype)
    else:
        _dbg(
            "graph_from_heterograph",
            f"异构边 canonical_etypes={input_graph.canonical_etypes}",
        )
        for can_etype in input_graph.canonical_etypes:
            data = {
                k: v_dict[can_etype]
                for k, v_dict in input_graph.edata.items()
                if can_etype in v_dict
            }
            src_t, dst_t = input_graph.edges(form="uv", etype=can_etype)
            _dbg(
                "graph_from_heterograph",
                f"  add_edges etype={can_etype!r} "
                f"num={src_t.shape[0]} data_keys={list(data.keys())}",
            )
            output_graph.add_edges(src_t, dst_t, data=data, etype=can_etype)

    return output_graph


# 向后兼容别名（与上游 cugraph_dgl 命名对齐）
cugraph_dgl_graph_from_heterograph = graph_from_heterograph

# ── 61a370e: cugraph_storage_from_heterograph 墓碑 ───────────────────────────
#
# 上游 61a370e (Remove Dask API from cuGraph-DGL, PR #199) 从 convert.py 删除了:
#
#   def cugraph_storage_from_heterograph(g: dgl.DGLGraph, single_gpu: bool = True):
#       num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
#       edges_dict = get_edges_dict_from_dgl_HeteroGraph(g, single_gpu)
#       gs = CuGraphStorage(data_dict=edges_dict, num_nodes_dict=num_nodes_dict,
#                           single_gpu=single_gpu, idtype=g.idtype)
#       add_ndata_from_dgl_HeteroGraph(gs, g)
#       add_edata_from_dgl_HeteroGraph(gs, g)
#       return gs
#
# Walpurgis 处置：
#   - cugraph_storage_from_heterograph 从未迁移（f4ca484 迁移时即 SKIP）
#     因为 CuGraphStorage 自身从未迁移（依赖 dask_cudf，不属于 walpurgis 架构）
#   - 此处提供显式 RuntimeError 墓碑函数，比「AttributeError: 不存在」更友好
#   - 详见 src/walpurgis/core/dgl_dask_removal.py


def cugraph_storage_from_heterograph(*args, **kwargs) -> None:
    """
    cugraph_storage_from_heterograph 墓碑函数（61a370e）。

    上游此函数将 dgl.DGLGraph 转换为 CuGraphStorage。
    因 CuGraphStorage 整体基于 dask_cudf，从未迁移到 walpurgis。
    上游 61a370e 删除了此函数，walpurgis 提供此墓碑避免静默 AttributeError。

    断点: cugraph_storage_from_heterograph 调用检测
    """
    _dbg(
        "cugraph_storage_from_heterograph",
        f"检测到对已删除函数的调用。"
        f"args={[type(a).__name__ for a in args]}",
    )
    raise RuntimeError(
        "cugraph_storage_from_heterograph 已在 commit 61a370e 中从上游删除，\n"
        "且从未迁移到 walpurgis（依赖 CuGraphStorage / dask_cudf）。\n"
        "\n"
        "替代（walpurgis 原生接口）:\n"
        "  from walpurgis.graph import graph_from_heterograph\n"
        "  wg = graph_from_heterograph(dgl_g, single_gpu=True)\n"
        "\n"
        "graph_from_heterograph 返回 walpurgis.graph.Graph，\n"
        "支持同构/异构图，节点/边特征均可迁移。\n"
        "详见: src/walpurgis/graph/convert.py"
    )
