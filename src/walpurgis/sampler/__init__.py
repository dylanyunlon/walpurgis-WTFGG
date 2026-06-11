# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg
# Walpurgis 迁移: sampler 子包 — 分布式图采样
#
# 「横眉冷对千夫指，俯首甘为孺子牛。」
# 此处只做一件事：把正确的名字送到正确的地方。

import os as _os
import sys as _sys
import time as _time

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg_import(msg: str) -> None:
    """断点：sampler 子包初始化诊断 (WALPURGIS_DEBUG=1 时输出)"""
    if _DEBUG:
        print(
            f"[WALPURGIS-SAMPLER:__init__][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


_dbg_import(">>> sampler 子包开始加载")

from .sampler import BaseSampler, SampleIterator
from .distributed_sampler import DistributedNeighborSampler, BaseDistributedSampler
# f4ca484 迁移新增：DGL 架构采样器基类和 NeighborSampler
from .dgl_sampler import Sampler, SampleReader, HomogeneousSampleReader
from .dgl_neighbor_sampler import NeighborSampler as DGLNeighborSampler
from .sampling_csc_helpers import (
    create_homogeneous_sampled_graphs_from_dataframe_csc,
    _process_sampled_df_csc,
    _create_homogeneous_sparse_graphs_from_csc,
)


# sampler_utils: neg_sample (dd543dc fix) + filter_cugraph_pyg_store + output builders
# 延迟 import 避免无 GPU 环境崩溃（sampler.py 内部也是延迟 import）
try:
    from .sampler_utils import (
        neg_sample,
        neg_cat,
        filter_cugraph_pyg_store,
        HopIndexer,
        SamplerResultValidator,
    )
    _dbg_import("sampler_utils 加载成功: neg_sample / neg_cat / filter_cugraph_pyg_store")
except ImportError as _e:
    _dbg_import(f"sampler_utils 加载跳过 (无 GPU 依赖): {_e}")

_dbg_import(
    f"<<< sampler 子包加载完毕 | "
    f"符号: BaseSampler={BaseSampler.__module__}, "
    f"DistributedNeighborSampler={DistributedNeighborSampler.__module__}, "
    f"BaseDistributedSampler={BaseDistributedSampler.__module__}, "
    f"create_homogeneous_sampled_graphs_from_dataframe_csc 已加载"
)

__all__ = [
    "BaseSampler",
    "SampleIterator",
    "DistributedNeighborSampler",
    "BaseDistributedSampler",
    "create_homogeneous_sampled_graphs_from_dataframe_csc",
    "_process_sampled_df_csc",
    "_create_homogeneous_sparse_graphs_from_csc",
    "neg_sample",
    "neg_cat",
    "filter_cugraph_pyg_store",
    "HopIndexer",
    "SamplerResultValidator",
]
