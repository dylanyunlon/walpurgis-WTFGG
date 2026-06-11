# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts (引入 cugraph-dgl Graph/features/view/convert)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「从来如此，便对么？」 —— 鲁迅《狂人日记》
# f4ca484 合并引入了 cugraph_dgl.Graph 这一核心图对象，以及
# WholeFeatureStore、HeteroView、convert 工具——
# 上游把这些分散在 cugraph_dgl 根目录，Walpurgis 将其聚合为独立 graph 子包。

import os as _os
import sys as _sys
import time as _time

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg_import(msg: str) -> None:
    if _DEBUG:
        print(
            f"[WALPURGIS-GRAPH:__init__][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


_dbg_import(">>> graph 子包开始加载")

from walpurgis.graph.graph import Graph, HOMOGENEOUS_NODE_TYPE, HOMOGENEOUS_EDGE_TYPE
from walpurgis.graph.features import WholeFeatureStore
from walpurgis.graph.convert import (
    graph_from_heterograph,
)

_dbg_import(
    f"<<< graph 子包加载完毕 | "
    f"Graph={Graph.__module__} "
    f"WholeFeatureStore={WholeFeatureStore.__module__}"
)

__all__ = [
    "Graph",
    "HOMOGENEOUS_NODE_TYPE",
    "HOMOGENEOUS_EDGE_TYPE",
    "WholeFeatureStore",
    "graph_from_heterograph",
]
