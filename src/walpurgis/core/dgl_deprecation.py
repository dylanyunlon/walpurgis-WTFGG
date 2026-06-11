"""
dgl_deprecation.py — migrate 456d5a2: Add deprecation warnings for DGL classes

上游 456d5a2 (cugraph-gnn) 改动:
  - cugraph_dgl/__init__.py:
      * 把 CuGraphStorage 改为 DEPRECATED__CuGraphStorage 内部别名
      * 新增 CuGraphStorage wrapper 函数 (FutureWarning)
      * 新增模块级 warnings.warn: cuGraph-DGL 不再主动维护，建议迁移到 cuGraph-PyG
  - cugraph_dgl/dataloading/__init__.py:
      * 把 DaskDataLoader 改为 DEPRECATED__DaskDataLoader 内部别名
      * DataLoader wrapper 改为也指向 DEPRECATED__DaskDataLoader (同样 FutureWarning)

Walpurgis 迁移语义:
  - walpurgis 基于 PyG 而非 DGL，DGL wrapper 对应的是 walpurgis.dataloader.DataLoader
    以及 walpurgis.sampler 中任何依赖 DGL 接口的路径
  - 本模块统一管理 DGL 兼容层的废弃状态:
      * CuGraphStorageCompat: DGL CuGraphStorage API → walpurgis UnifiedStore 迁移桥
      * DaskDataLoaderCompat: DGL DaskDataLoader API → walpurgis DataLoader 迁移桥
  - 模块加载时发出全局警告: DGL 接口在 walpurgis 中已进入 legacy 维护模式

20% 改写 (鲁迅拿法):
  - 用 DeprecationPolicy 替代裸 warnings.warn wrapper 函数 (与 feature_store_deprecation 一致)
  - _DglLegacyBanner: 模块级弃用公告，替代上游的裸 warnings.warn 模块顶部调用，
    增加 WALPURGIS_DEBUG 细节输出 + 调用次数去重
  - 断点1: 模块加载时的全局弃用公告
  - 断点2: CuGraphStorageCompat.__call__ 入口 (参数摘要)
  - 断点3: DaskDataLoaderCompat.__call__ 入口 (参数摘要 + 推荐替代)

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import warnings
from typing import Any

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"

_MODULE_WARN_ISSUED = False  # 模块级警告去重标志


def _dbg(msg: str) -> None:
    """WALPURGIS_DEBUG=1 时打印调试信息。"""
    if _DEBUG:
        print(f"[WALPURGIS dgl_deprecation] {msg}", file=sys.stderr, flush=True)


def _summarize_args(args, kwargs) -> str:
    """生成参数类型摘要（不打印值）。"""
    parts = [type(a).__name__ for a in args]
    kw_parts = [f"{k}={type(v).__name__}" for k, v in kwargs.items()]
    return f"args=({', '.join(parts)}) kwargs={{" + ", ".join(kw_parts) + "}"


# ── 断点1: 模块级 DGL legacy 公告 ────────────────────────────────────────────

class _DglLegacyBanner:
    """
    对应上游 456d5a2 在 cugraph_dgl/__init__.py 顶部新增的模块级 warnings.warn:

        warnings.warn(
            "cuGraph-DGL is no longer under active development. "
            "We strongly recommend migrating to cuGraph-PyG.",
        )

    Walpurgis 改写:
    - 去重: 同一进程中只触发一次 (上游裸 warn 每次 import 都可能触发)
    - WALPURGIS_DEBUG 下输出详细信息
    - issue() 方法可在测试中手动重置 (reset_issued=True)

    断点1: issue() 调用时。
    """

    def __init__(self):
        self._issued = False

    def issue(self, reset_issued: bool = False) -> None:
        """
        发出模块级 DGL legacy 警告。

        参数
        ----
        reset_issued: 测试用，强制重新发出警告。
        """
        if reset_issued:
            self._issued = False

        if self._issued:
            _dbg("_DglLegacyBanner.issue(): 已发出，跳过重复 (reset_issued=False)")
            return

        self._issued = True

        # ── 断点1 ────────────────────────────────────────────────────────
        _dbg(
            "_DglLegacyBanner.issue(): 发出 DGL legacy 警告 "
            "(对应上游 456d5a2 模块级 warnings.warn)"
        )

        warnings.warn(
            "cuGraph-DGL compatible interfaces in walpurgis are in legacy maintenance mode "
            "and will be removed in a future release. "
            "We strongly recommend migrating to the PyG-based walpurgis.dataloader.DataLoader "
            "and walpurgis.sampler.DistributedNeighborSampler.",
            FutureWarning,
            stacklevel=3,
        )

    def reset(self) -> None:
        """重置发出标志 (测试用)。"""
        self._issued = False

    @property
    def issued(self) -> bool:
        return self._issued

    def __repr__(self):
        return f"_DglLegacyBanner(issued={self._issued})"


#: 模块级 legacy banner 单例
DGL_LEGACY_BANNER = _DglLegacyBanner()

# 模块加载时自动发出一次
DGL_LEGACY_BANNER.issue()


# ── CuGraphStorageCompat: 对应上游 CuGraphStorage wrapper ────────────────────

class CuGraphStorageCompat:
    """
    DGL CuGraphStorage API 兼容层。

    对应上游 456d5a2:
        def CuGraphStorage(*args, **kwargs):
            warnings.warn(
                "CuGraphStorage and the rest of the dask-based API are deprecated"
                "and will be removed in release 25.08.",
                FutureWarning,
            )
            return DEPRECATED__CuGraphStorage(*args, **kwargs)

    Walpurgis 改写:
    - callable class 替代裸 wrapper 函数，持有调用统计
    - 推荐迁移路径: walpurgis.core.unified_store.UnifiedStore
    - 断点2: __call__ 入口

    用法:
        store = CuGraphStorageCompat(g, feat_tensor)  # 触发 FutureWarning
    """

    def __init__(self):
        self.call_count: int = 0
        self.__name__ = "CuGraphStorage"

    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1

        # ── 断点2 ────────────────────────────────────────────────────────
        _dbg(
            f"CuGraphStorageCompat.__call__(): "
            f"call_count={self.call_count}, "
            f"{_summarize_args(args, kwargs)}"
        )

        warnings.warn(
            "CuGraphStorage and the rest of the DGL-based API are deprecated "
            "and will be removed in a future walpurgis release. "
            "Migrate to: walpurgis.core.unified_store.UnifiedStore",
            FutureWarning,
            stacklevel=2,
        )

        # 尝试从可选的 cugraph-dgl 包调用原始类
        try:
            from cugraph_dgl.cugraph_storage import CuGraphStorage as _Impl
            _dbg(
                f"  cugraph_dgl.CuGraphStorage found, constructing "
                f"(type={_Impl.__name__})"
            )
            return _Impl(*args, **kwargs)
        except ImportError as e:
            raise ImportError(
                "CuGraphStorage requires cugraph-dgl which is not installed. "
                "Consider migrating to walpurgis.core.unified_store.UnifiedStore. "
                f"Original error: {e}"
            ) from e

    def reset_count(self) -> None:
        self.call_count = 0

    def __repr__(self):
        return f"CuGraphStorageCompat(call_count={self.call_count})"


# ── DaskDataLoaderCompat: 对应上游 DataLoader/DaskDataLoader wrapper ──────────

class DaskDataLoaderCompat:
    """
    DGL DaskDataLoader / DataLoader API 兼容层。

    对应上游 456d5a2 dataloading/__init__.py 改动:
        def DataLoader(*args, **kwargs):
            warnings.warn(
                "CuGraphStorage and the rest of the dask-based API are deprecated"
                "and will be removed in release 25.08.",
                FutureWarning,
            )
            return DEPRECATED__DaskDataLoader(*args, **kwargs)

    Walpurgis 改写:
    - callable class，持有调用统计
    - 推荐迁移路径: walpurgis.dataloader.DataLoader
    - 断点3: __call__ 入口

    用法:
        loader = DaskDataLoaderCompat(graph_store, train_ids, sampler, ...)
    """

    def __init__(self):
        self.call_count: int = 0
        self.__name__ = "DaskDataLoader"

    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1

        # ── 断点3 ────────────────────────────────────────────────────────
        _dbg(
            f"DaskDataLoaderCompat.__call__(): "
            f"call_count={self.call_count}, "
            f"{_summarize_args(args, kwargs)}, "
            f"建议替代: walpurgis.dataloader.DataLoader"
        )

        warnings.warn(
            "DaskDataLoader and the DGL-based dataloader API are deprecated "
            "and will be removed in a future walpurgis release. "
            "Migrate to: walpurgis.dataloader.DataLoader",
            FutureWarning,
            stacklevel=2,
        )

        # 尝试从可选的 cugraph-dgl 包调用原始类
        try:
            from cugraph_dgl.dataloading.dask_dataloader import (
                DaskDataLoader as _Impl,
            )
            _dbg(f"  cugraph_dgl.DaskDataLoader found, constructing (type={_Impl.__name__})")
            return _Impl(*args, **kwargs)
        except ImportError as e:
            raise ImportError(
                "DaskDataLoader requires cugraph-dgl which is not installed. "
                "Consider migrating to walpurgis.dataloader.DataLoader. "
                f"Original error: {e}"
            ) from e

    def reset_count(self) -> None:
        self.call_count = 0

    def __repr__(self):
        return f"DaskDataLoaderCompat(call_count={self.call_count})"


# ── 模块级单例 ────────────────────────────────────────────────────────────────

#: CuGraphStorage DGL 兼容层 callable（等同上游 wrapper 函数）
CuGraphStorage = CuGraphStorageCompat()

#: DaskDataLoader DGL 兼容层 callable（等同上游 wrapper 函数）
DaskDataLoader = DaskDataLoaderCompat()

_dbg(
    f"dgl_deprecation module loaded: "
    f"CuGraphStorage={CuGraphStorage!r}, "
    f"DaskDataLoader={DaskDataLoader!r}, "
    f"DGL_LEGACY_BANNER={DGL_LEGACY_BANNER!r}"
)


__all__ = [
    "DGL_LEGACY_BANNER",
    "CuGraphStorageCompat",
    "DaskDataLoaderCompat",
    "CuGraphStorage",
    "DaskDataLoader",
]
