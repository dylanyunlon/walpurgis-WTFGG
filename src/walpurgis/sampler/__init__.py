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

_dbg_import(
    f"<<< sampler 子包加载完毕 | "
    f"符号: BaseSampler={BaseSampler.__module__}, "
    f"DistributedNeighborSampler={DistributedNeighborSampler.__module__}, "
    f"BaseDistributedSampler={BaseDistributedSampler.__module__}"
)

__all__ = [
    "BaseSampler",
    "SampleIterator",
    "DistributedNeighborSampler",
    "BaseDistributedSampler",
]
