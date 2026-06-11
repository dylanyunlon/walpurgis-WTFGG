"""
loader_deprecation.py — 431801c 迁移: 废弃 DaskNeighborLoader / BulkSampleLoader

migrate 431801c: Deprecate the Dask API in cuGraph-PyG

上游变化 (431801c, cugraph-gnn / python/cugraph-pyg/cugraph_pyg/loader/__init__.py):

1. DaskNeighborLoader 废弃:
   旧: from cugraph_pyg.loader.dask_node_loader import DaskNeighborLoader
   新:
       from cugraph_pyg.loader.dask_node_loader import (
           DaskNeighborLoader as DEPRECATED__DaskNeighborLoader,
       )
       def DaskNeighborLoader(*args, **kwargs):
           warnings.warn("DaskNeighborLoader and the Dask API are deprecated. ...", FutureWarning)
           return DEPRECATED__DaskNeighborLoader(*args, **kwargs)

2. BulkSampleLoader 废弃:
   旧: from cugraph_pyg.loader.dask_node_loader import BulkSampleLoader
   新:
       def BulkSampleLoader(*args, **kwargs):
           warnings.warn("BulkSampleLoader and the Dask API are deprecated. ...", FutureWarning)
           return DEPRECATED__BulkSampleLoader(*args, **kwargs)

设计背景:
    - DaskNeighborLoader: 基于 Dask 的分布式邻居采样加载器，是旧 API 的主入口。
      新 API 用 cugraph_pyg.loader.NodeLoader / NeighborLoader 替代。
    - BulkSampleLoader: 批量采样加载器，依赖 Dask + 磁盘 IO。
      新 API 淘汰 unbuffered/bulk sampling，改用 streaming 内存采样。
    - wrapper 函数模式保持向后兼容（不破坏现有用户代码），FutureWarning 催迁移。

Walpurgis 改写 20%（鲁迅拿法）:
- 复用 core/feature_store_deprecation.py 的 DeprecationPolicy / DeprecationGate 机制
  上游 __init__.py 模式: 每个废弃对象写一个独立 wrapper 函数，无统计，无可观测性。
  Walpurgis 模式: DeprecationGate 统一管理，call_count 可查，WALPURGIS_DEBUG 可观测。
- LoaderDeprecationRegistry 单例: 集中注册所有 loader 侧废弃对象，
  与 feature_store_deprecation.py 的 DeprecationPolicy 并列，覆盖不同模块。
- BulkSampleDeprecation 加额外警告: 上游仅 FutureWarning「deprecated」，
  Walpurgis 额外注明「unbuffered sampling 将被完全移除（含新 API）」。
- 全链路 WALPURGIS_DEBUG=1 断点 print: 触发时打印调用来源模块 + 参数类型摘要。

作者: dylanyunlon <dogechat@163.com>
"""

import os
import sys
import warnings
from typing import Any, Callable, Dict, Optional

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LoaderDeprecationGate — 单个废弃 loader 的 wrapper 封装
# ─────────────────────────────────────────────────────────────────────────────

class LoaderDeprecationGate:
    """
    封装单个废弃 loader 的 FutureWarning + 透传构造。

    对应上游每个 wrapper 函数:
        def DaskNeighborLoader(*args, **kwargs):
            warnings.warn("...", FutureWarning)
            return DEPRECATED__DaskNeighborLoader(*args, **kwargs)

    LoaderDeprecationGate 等价但加:
    - call_count: 可查询该废弃入口被调用了多少次
    - WALPURGIS_DEBUG=1 时打印调用参数类型摘要（不泄露值）
    - extra_note: 额外警告内容（BulkSampleLoader 需要更强的警告）
    """

    def __init__(
        self,
        name: str,
        wrapped_cls,
        warning_msg: str,
        extra_note: str = "",
        stacklevel: int = 3,
    ):
        self._name = name
        self._wrapped_cls = wrapped_cls
        self._warning_msg = warning_msg
        self._extra_note = extra_note
        self._stacklevel = stacklevel
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(self, *args, **kwargs):
        self._call_count += 1

        full_msg = self._warning_msg
        if self._extra_note:
            full_msg = full_msg + "  " + self._extra_note

        warnings.warn(full_msg, FutureWarning, stacklevel=self._stacklevel)

        _dbg(
            f"LoaderDeprecationGate:{self._name}",
            f"called (call_count={self._call_count})",
            args_types=str([type(a).__name__ for a in args]),
            kwargs_keys=str(list(kwargs.keys())),
        )

        return self._wrapped_cls(*args, **kwargs)

    def __repr__(self):
        return (
            f"LoaderDeprecationGate(name={self._name!r}, "
            f"wrapped={self._wrapped_cls.__name__!r}, "
            f"call_count={self._call_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LoaderDeprecationRegistry — 集中管理所有 loader 侧废弃对象
# ─────────────────────────────────────────────────────────────────────────────

class LoaderDeprecationRegistry:
    """
    统一注册和查找废弃 loader 对象。

    与 feature_store_deprecation.py 的 DeprecationPolicy 并列，
    覆盖 loader 模块侧的废弃 API（DaskNeighborLoader / BulkSampleLoader）。

    断点1: register() — 注册时打印 name / wrapped class
    断点2: get() — 查找时打印 name + 是否命中
    断点3: summary() — 打印所有已注册对象的 call_count
    """

    def __init__(self):
        self._gates: Dict[str, LoaderDeprecationGate] = {}

    def register(
        self,
        name: str,
        wrapped_cls,
        warning_msg: str,
        extra_note: str = "",
        stacklevel: int = 3,
    ) -> LoaderDeprecationGate:
        gate = LoaderDeprecationGate(name, wrapped_cls, warning_msg, extra_note, stacklevel)
        self._gates[name] = gate
        _dbg(
            "LoaderDeprecationRegistry",
            f"registered {name!r}",
            wrapped=wrapped_cls.__name__,
        )
        return gate

    def get(self, name: str) -> Optional[LoaderDeprecationGate]:
        gate = self._gates.get(name)
        _dbg(
            "LoaderDeprecationRegistry",
            f"get {name!r}",
            found=gate is not None,
        )
        return gate

    def summary(self) -> str:
        lines = ["LoaderDeprecationRegistry summary:"]
        for name, gate in self._gates.items():
            lines.append(f"  {name}: call_count={gate.call_count}")
        return "\n".join(lines)

    def __repr__(self):
        return f"LoaderDeprecationRegistry(registered={list(self._gates.keys())})"


# ─────────────────────────────────────────────────────────────────────────────
# 全局 registry 实例 + 废弃对象注册
# ─────────────────────────────────────────────────────────────────────────────

# 全局单例，供外部查询 call_count / summary
loader_deprecation_registry = LoaderDeprecationRegistry()

# 延迟构建实际废弃入口，避免在无 cugraph_pyg 环境 import 时崩溃
_DaskNeighborLoader_gate: Optional[LoaderDeprecationGate] = None
_BulkSampleLoader_gate: Optional[LoaderDeprecationGate] = None


def _get_or_build_gates():
    """
    延迟 import cugraph_pyg.loader.dask_node_loader，
    避免在无 GPU 环境 import 本模块时引发 ImportError。

    51fa4e8: 首次调用时发出模块级 Dask 废弃警告 (参见 _emit_dask_module_warning)。
    """
    global _DaskNeighborLoader_gate, _BulkSampleLoader_gate

    # 51fa4e8: emit once-per-process module-level warning on first real use
    _emit_dask_module_warning()

    if _DaskNeighborLoader_gate is not None:
        return

    try:
        from cugraph_pyg.loader.dask_node_loader import (
            DaskNeighborLoader as DEPRECATED__DaskNeighborLoader,
            BulkSampleLoader as DEPRECATED__BulkSampleLoader,
        )
    except ImportError:
        _dbg("loader_deprecation", "cugraph_pyg not available — gates not built")
        return

    _DaskNeighborLoader_gate = loader_deprecation_registry.register(
        name="DaskNeighborLoader",
        wrapped_cls=DEPRECATED__DaskNeighborLoader,
        warning_msg=(
            "DaskNeighborLoader and the Dask API are deprecated. "
            "Consider switching to the new API "
            "(cugraph_pyg.loader.NodeLoader, cugraph_pyg.loader.NeighborLoader)."
        ),
        stacklevel=3,
    )

    _BulkSampleLoader_gate = loader_deprecation_registry.register(
        name="BulkSampleLoader",
        wrapped_cls=DEPRECATED__BulkSampleLoader,
        warning_msg=(
            "BulkSampleLoader and the Dask API are deprecated. "
            "Unbuffered sampling in cuGraph-PyG will be completely deprecated "
            "and removed, including in the new API."
        ),
        extra_note=(
            "Switch to streaming NeighborLoader instead of bulk pre-sampling."
        ),
        stacklevel=3,
    )

    _dbg("loader_deprecation", "gates built", gates=list(loader_deprecation_registry._gates.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API — 与上游 __init__.py wrapper 函数等价
# ─────────────────────────────────────────────────────────────────────────────

def DaskNeighborLoader(*args, **kwargs):
    """
    废弃入口: DaskNeighborLoader — 请改用 cugraph_pyg.loader.NeighborLoader。

    等价于上游 431801c 中的 wrapper 函数，但通过 LoaderDeprecationGate 路由，
    可查询 call_count 和 WALPURGIS_DEBUG 断点输出。
    """
    _get_or_build_gates()
    if _DaskNeighborLoader_gate is None:
        raise ImportError("cugraph_pyg.loader.dask_node_loader not available")
    return _DaskNeighborLoader_gate(*args, **kwargs)


def BulkSampleLoader(*args, **kwargs):
    """
    废弃入口: BulkSampleLoader — 请改用 streaming NeighborLoader。

    等价于上游 431801c 中的 wrapper 函数，但通过 LoaderDeprecationGate 路由，
    额外包含「unbuffered sampling 将被完全移除」的警告。
    """
    _get_or_build_gates()
    if _BulkSampleLoader_gate is None:
        raise ImportError("cugraph_pyg.loader.dask_node_loader not available")
    return _BulkSampleLoader_gate(*args, **kwargs)


def CuGraphNeighborLoader(*args, **kwargs):
    """
    历史兼容名称 — 已被 DaskNeighborLoader 替代，后者也已废弃。
    发出双重 FutureWarning。
    """
    warnings.warn(
        "CuGraphNeighborLoader has been renamed to DaskNeighborLoader, "
        "which is also deprecated. Please use NeighborLoader instead.",
        FutureWarning,
        stacklevel=2,
    )
    _dbg("CuGraphNeighborLoader", "redirecting to DaskNeighborLoader")
    return DaskNeighborLoader(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 51fa4e8 / ee58e32: 模块级废弃警告 (DGL pattern 迁移)
#
# 上游 51fa4e8 在 cugraph_dgl/__init__.py 顶层加了：
#   warnings.warn("cuGraph-DGL is no longer under active development. ...")
# 上游 ee58e32 把同样的警告作为正式 PR 合并进 branch-25.06。
#
# Walpurgis 对应位置: Dask-based loader API (DaskNeighborLoader / BulkSampleLoader)
# 与 cuGraph-DGL 同属「待移除的旧 API 层」。
# 区别在于我们不在模块 import 时就触发 (避免噪声)，而是在首次调用 _get_or_build_gates()
# 时通过 _emit_dask_module_warning() 发出一次性模块级警告。
# 相当于 DGL 的「import 即警告」但延迟到实际使用，减少无关代码路径的噪声。
#
# WALPURGIS_DEBUG=1 时额外打印触发调用栈摘要，定位遗留代码位置。
# ─────────────────────────────────────────────────────────────────────────────

_dask_module_warning_emitted: bool = False


def _emit_dask_module_warning() -> None:
    """
    51fa4e8: 发出一次性模块级 Dask loader 废弃警告。

    上游在 cugraph_dgl/__init__.py 顶层用裸 warnings.warn() 对整个 DGL 子包发警告；
    我们在首次实际调用 Dask loader 入口时触发，语义等价但无 import 噪声。

    Knuth 视角: 上游的模块级警告在 CI 环境会导致每次 import cugraph_dgl 都打印
    FutureWarning，而 test_dask_dataloader.py 里有多处 import，每次都触发。
    延迟到首次调用更干净；但 stacklevel 必须足够深才能指向用户代码而不是这里。
    """
    global _dask_module_warning_emitted
    if _dask_module_warning_emitted:
        return

    _dask_module_warning_emitted = True

    warnings.warn(
        "The Dask-based distributed loader API (DaskNeighborLoader, BulkSampleLoader) "
        "in Walpurgis is no longer under active development and will be removed in a "
        "future release. We strongly recommend migrating to the WholeGraph-backed "
        "NeighborLoader / NodeLoader launched via torchrun.",
        FutureWarning,
        stacklevel=3,
    )

    if os.environ.get("WALPURGIS_DEBUG", "0") == "1":
        import traceback
        import sys
        print(
            "[WALPURGIS-LOADER:loader_deprecation][51fa4e8] "
            "Dask module-level warning emitted. Call stack:",
            file=sys.stderr, flush=True,
        )
        traceback.print_stack(file=sys.stderr, limit=8)
