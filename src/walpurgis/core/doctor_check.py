# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Walpurgis 运行时自检入口——仿"拉斯柯尔尼科夫自我审判"之格式：
每一道断言皆是一次灵魂拷问，失败即抛出带现场坐标的 RuntimeError。

上游来源: rapidsai/cugraph-gnn@daf857d
  (Add RAPIDS Doctor Check for cuGraph-PyG and pylibwholegraph, #418)

改写摘要（≥20%）:
  - 原 _doctor_check.py × 2 合并为单文件，按"诊断上下文"对象（DoctorCtx）统一封装
  - 所有 warnings.warn → structured WalpurgisWarning 子类，可被上层捕获过滤
  - GraphStore 断言从裸 if/raise 改为 _assert_edge_shape() 工具函数（可复用）
  - 分布式初始化从无超时 → 带 60 s 守卫的 _init_pg_with_timeout()
  - 每个关键分支插入 breakpoint_trace()（调试钩子，生产环境 no-op）
  - pyproject.toml entry-point 不在此文件处理——见项目根 pyproject.toml 说明
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import patch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 0. 调试钩子：WALPURGIS_DEBUG=1 时激活 pdb
# ──────────────────────────────────────────────

def breakpoint_trace(label: str, ctx: Optional[dict] = None) -> None:
    """
    断点调试工具。WALPURGIS_DEBUG=1 时暂停进入 pdb；
    否则仅写 DEBUG 日志，生产零开销。

    用法（嵌入在关键路径）::

        breakpoint_trace("after_import", {"module": "cugraph_pyg", "ver": ver})
    """
    if os.environ.get("WALPURGIS_DEBUG", "0") != "1":
        logger.debug("[doctor_check] %s ctx=%s", label, ctx)
        return
    import pdb  # noqa: T100

    print(f"\n[doctor_check:breakpoint] {label}  ctx={ctx}")  # noqa: T201
    pdb.set_trace()  # noqa: T100


# ──────────────────────────────────────────────
# 1. 结构化告警
# ──────────────────────────────────────────────

class WalpurgisWarning(UserWarning):
    """Walpurgis doctor 检查的基础告警类，可被上层统一 filter。"""


class CudaUnavailableWarning(WalpurgisWarning):
    """CUDA 不可用或 PyTorch 缺失时发出。"""


class WholegraphWarning(WalpurgisWarning):
    """pylibwholegraph 依赖告警。"""


# ──────────────────────────────────────────────
# 2. 诊断上下文（数据类）
# ──────────────────────────────────────────────

@dataclass
class DoctorCtx:
    """
    单次 doctor 检查的执行上下文。
    集中持有所有动态探测到的状态，避免各子函数重复 import 探测。
    """
    cugraph_pyg_version: str = ""
    pylibwholegraph_version: str = ""
    torch_available: bool = False
    cuda_available: bool = False
    distributed_initialized: bool = False
    edge_shape_ok: bool = False
    warnings_issued: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, msg: str, category: type = WalpurgisWarning) -> None:
        warnings.warn(msg, category, stacklevel=3)
        self.warnings_issued.append(msg)

    def record_error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error("[doctor_check] %s", msg)


# ──────────────────────────────────────────────
# 3. 工具函数
# ──────────────────────────────────────────────

_DIST_ENV_DEFAULTS: dict[str, str] = {
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "29505",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_WORLD_SIZE": "1",
    "RANK": "0",
}


def _init_pg_with_timeout(backend: str = "nccl", timeout_s: int = 60) -> None:
    """
    带超时守卫的分布式进程组初始化。
    原实现无超时，悬挂风险高；此处注入 timedelta 守卫。
    """
    import datetime
    import torch.distributed as dist

    timeout = datetime.timedelta(seconds=timeout_s)
    try:
        dist.init_process_group(backend, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            f"分布式进程组初始化失败（backend={backend}, timeout={timeout_s}s）。"
            "请确认 PyTorch 与 NCCL 安装正确。"
        ) from exc


def _assert_edge_shape(edge_index: Any, expected: tuple[int, int]) -> None:
    """
    断言 edge_index 形状符合预期，失败时附带实际形状信息。
    原实现内联在 smoke check 中，此处抽取为可复用函数。
    """
    import torch

    expected_size = torch.Size(list(expected))
    if edge_index.shape != expected_size:
        raise AssertionError(
            f"edge index 形状断言失败：期望 {list(expected)}，"
            f"实际 {list(edge_index.shape)}"
        )


# ──────────────────────────────────────────────
# 4. cugraph-pyg smoke check（entry-point 函数）
# ──────────────────────────────────────────────

def cugraph_pyg_smoke_check(**kwargs: Any) -> DoctorCtx:
    """
    cugraph-pyg 运行时自检。

    成功返回填充后的 DoctorCtx；失败抛出 ImportError / RuntimeError。
    调用者可检查 ctx.warnings_issued 了解软性告警。
    """
    ctx = DoctorCtx()
    breakpoint_trace("cugraph_pyg_smoke_check:start", {"kwargs": kwargs})

    # ── 4a. 导入 & 版本探测 ──────────────────
    try:
        import cugraph_pyg
        import cugraph_pyg.data  # noqa: F401  触发 submodule 加载
        import cugraph_pyg.tensor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cugraph-pyg 或其依赖无法导入。"
            "提示：`pip install --extra-index-url=https://pypi.nvidia.com cugraph-pyg-cu13`"
            " 或使用 RAPIDS conda 环境。"
        ) from exc

    ver = getattr(cugraph_pyg, "__version__", "")
    if not ver:
        raise AssertionError("cugraph-pyg smoke check 失败：__version__ 为空或缺失")
    ctx.cugraph_pyg_version = ver
    breakpoint_trace("cugraph_pyg_smoke_check:imported", {"version": ver})

    # ── 4b. torch / CUDA 可用性探测 ─────────
    from cugraph_pyg.utils.imports import import_optional, MissingModule

    torch = import_optional("torch")
    if isinstance(torch, MissingModule) or not torch.cuda.is_available():
        ctx.warn(
            "cuGraph-PyG 需要支持 CUDA 的 PyTorch。"
            "请从 PyPI 或 Conda-Forge 安装 PyTorch（CUDA 版）。",
            CudaUnavailableWarning,
        )
        breakpoint_trace("cugraph_pyg_smoke_check:cuda_unavailable")
        return ctx

    ctx.torch_available = True
    ctx.cuda_available = True

    # ── 4c. 分布式 GraphStore 端到端校验 ────
    from cugraph_pyg.data import GraphStore

    with patch.dict(os.environ, _DIST_ENV_DEFAULTS, clear=False):
        try:
            _init_pg_with_timeout()
            ctx.distributed_initialized = True
            breakpoint_trace("cugraph_pyg_smoke_check:pg_initialized")

            gs = GraphStore()
            gs.put_edge_index(
                torch.tensor([[0, 1], [1, 2]]),
                ("person", "knows", "person"),
                "coo",
                False,
                (3, 3),
            )
            edge_index = gs.get_edge_index(("person", "knows", "person"), "coo")
            _assert_edge_shape(edge_index, (2, 2))
            ctx.edge_shape_ok = True
            breakpoint_trace("cugraph_pyg_smoke_check:edge_ok", {"shape": list(edge_index.shape)})

        finally:
            if ctx.distributed_initialized:
                torch.distributed.destroy_process_group()

    return ctx


# ──────────────────────────────────────────────
# 5. pylibwholegraph smoke check（entry-point 函数）
# ──────────────────────────────────────────────

def pylibwholegraph_smoke_check(**kwargs: Any) -> DoctorCtx:
    """
    pylibwholegraph 运行时自检。

    成功返回填充后的 DoctorCtx；失败抛出 ImportError。
    """
    ctx = DoctorCtx()
    breakpoint_trace("pylibwholegraph_smoke_check:start", {"kwargs": kwargs})

    # ── 5a. 导入 & 版本探测 ──────────────────
    try:
        import pylibwholegraph
    except ImportError as exc:
        raise ImportError(
            "pylibwholegraph 或其依赖无法导入。"
            "提示：`pip install --extra-index-url=https://pypi.nvidia.com pylibwholegraph-cu13`"
            " 或使用 RAPIDS conda 环境。"
        ) from exc

    ver = getattr(pylibwholegraph, "__version__", "")
    if not ver:
        raise AssertionError("pylibwholegraph smoke check 失败：__version__ 为空或缺失")
    ctx.pylibwholegraph_version = ver
    breakpoint_trace("pylibwholegraph_smoke_check:imported", {"version": ver})

    # ── 5b. torch / CUDA 软性检查 ────────────
    try:
        import torch

        if not torch.cuda.is_available():
            raise AssertionError("CUDA 不可用")
        ctx.torch_available = True
        ctx.cuda_available = True

    except (ImportError, AssertionError) as exc:
        ctx.warn(
            "PyTorch（含 CUDA 支持）或其依赖无法导入/使用。"
            "pylibwholegraph 需要 PyTorch（CUDA 版），请从 PyPI 或 Conda-Forge 安装。",
            WholegraphWarning,
        )
        breakpoint_trace("pylibwholegraph_smoke_check:cuda_unavailable", {"reason": str(exc)})

    return ctx
