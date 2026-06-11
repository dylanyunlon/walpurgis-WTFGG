# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 539d0ad: Expose cugraph_pyg.tensor Subpackage
# Walpurgis 迁移: 暴露 walpurgis.tensor 子包
# 原文不过是把门推开了一道缝——却没想到屋里早已另有乾坤。
# 鲁迅拿法20%改写: 去掉官僚式的平铺直叙, 改用"按需暴露"思路:
#   - 凡已被 walpurgis 自研版本替换的符号, 用本地版本; 否则沿用上游。
#   - 断点调试全链路: import 时即打印 tensor 子包已载入 (WALPURGIS_DEBUG=1)

import os as _os
import sys as _sys
import time as _time

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg_import(msg: str) -> None:
    """断点: tensor 子包初始化诊断 (WALPURGIS_DEBUG=1 时输出)"""
    if _DEBUG:
        print(
            f"[WALPURGIS-TENSOR:__init__][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


_dbg_import(">>> tensor 子包开始加载")

from walpurgis.tensor.dist_tensor import DistTensor, DistEmbedding
from walpurgis.tensor.dist_matrix import DistMatrix
from walpurgis.tensor.utils import is_empty, empty
from walpurgis.tensor.sparse_graph import SparseGraph, compress_ids, decompress_ids
from walpurgis.tensor.embedding_view import EmbeddingView

_dbg_import(
    f"<<< tensor 子包加载完毕 | "
    f"符号: DistTensor={DistTensor.__module__}, "
    f"DistMatrix={DistMatrix.__module__}, "
    f"is_empty={is_empty.__module__}, "
    f"SparseGraph={SparseGraph.__module__}"
)

__all__ = [
    "DistTensor",
    "DistEmbedding",
    "DistMatrix",
    "is_empty",
    "empty",
    "SparseGraph",
    "compress_ids",
    "decompress_ids",
    "EmbeddingView",
]
