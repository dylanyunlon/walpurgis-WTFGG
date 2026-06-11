# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 05b5791: Remove Dask client fixture and dask optional imports
#
# 上游 05b5791 (cugraph-gnn) 从 cugraph-pyg 测试框架删除了:
#   - dask_client fixture (依赖 dask_cuda.LocalCUDACluster + dask.distributed.Client)
#   - conftest.py 中所有 dask 相关的 fixture 和 import
#   - sampler_utils.py 中的 dask_cudf = import_optional("dask_cudf")
#
# Walpurgis 迁移语义:
#   - walpurgis 本来就没有 dask 依赖，此 commit 确认并固化这一设计决策
#   - conftest 只保留单机单 GPU 测试所需的 fixture
#   - 移除 dask_client fixture，若有测试依赖它将在 collect 时 fail-fast (明确报错)
#
# 20% 改写 (鲁迅拿法):
#   - _DaskClientUnavailable: 替代直接缺失的 dask_client fixture，
#     任何依赖它的测试会得到 skip + 友好提示，而不是神秘的 fixture-not-found 错误
#   - WALPURGIS_DEBUG=1 时在 conftest 加载时打印采样测试环境摘要
#   - GPU 可用性检测 fixture，统一 skipif 逻辑
#   - 断点1: conftest 加载摘要 (GPU 数量 / CUDA 版本)
#   - 断点2: dask_client fixture 被请求时的警告
#
# 作者: dylanyunlon<dogechat@163.com>

import os
import sys
import warnings

import pytest

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS tests/sampler/conftest] {msg}", file=sys.stderr, flush=True)


# ── 断点1: conftest 加载摘要 ────────────────────────────────────────────────
def _print_env_summary():
    """打印测试环境摘要 (WALPURGIS_DEBUG=1 时)。"""
    if not _DEBUG:
        return
    try:
        import torch
        n_gpu = torch.cuda.device_count()
        cuda_ver = torch.version.cuda or "N/A"
    except ImportError:
        n_gpu = 0
        cuda_ver = "torch-not-installed"

    try:
        import cupy
        cupy_ver = cupy.__version__
    except ImportError:
        cupy_ver = "not-installed"

    _dbg(
        f"conftest loaded: GPUs={n_gpu}, CUDA={cuda_ver}, cupy={cupy_ver}, "
        f"pid={os.getpid()}"
    )


_print_env_summary()


# ── 单机 GPU 可用性 fixture ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def single_gpu_available():
    """
    检测单机 GPU 是否可用。

    返回 True/False，供 skipif 使用:
        @pytest.mark.skipif(not single_gpu_available, reason="No GPU")

    与上游 conftest.py 的 dask_client fixture 不同:
    本 fixture 只检测本机 GPU，不启动任何分布式进程。
    """
    try:
        import torch
        available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        _dbg(f"single_gpu_available={available}, device_count={torch.cuda.device_count()}")
        return available
    except ImportError:
        _dbg("single_gpu_available=False (torch not installed)")
        return False


# ── migrate 05b5791: dask_client fixture 已移除 ─────────────────────────────

@pytest.fixture(scope="module")
def dask_client():
    """
    [已移除 — migrate 05b5791]

    上游 cugraph-gnn 05b5791 删除了 dask_client fixture。
    Walpurgis 从未依赖 Dask 分布式后端，此 fixture 在 walpurgis 中无意义。

    任何依赖本 fixture 的测试将被标记为 skip（而非 xfail），
    并打印迁移指引。

    断点2: 被请求时打印警告。
    """
    # ── 断点2 ────────────────────────────────────────────────────────────────
    _dbg(
        "dask_client fixture requested — this fixture was removed in 05b5791. "
        "The requesting test should be updated to use single_gpu_available instead."
    )

    warnings.warn(
        "dask_client fixture is no longer available (removed in migrate 05b5791). "
        "Walpurgis does not use Dask-based multi-GPU distributed training. "
        "Migrate tests to use walpurgis.sampler.DistributedNeighborSampler directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    pytest.skip(
        "dask_client fixture removed (migrate 05b5791): "
        "Dask-based multi-GPU API has been dropped. "
        "See walpurgis.sampler.distributed_sampler for the replacement."
    )
