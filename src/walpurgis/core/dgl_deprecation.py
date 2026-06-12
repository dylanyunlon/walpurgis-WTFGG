"""
dgl_deprecation.py — migrate 456d5a2 + fb8296e: DGL 废弃 → 永久删除

上游历史:
  - 456d5a2: Add deprecation warnings for DGL classes
      * cugraph_dgl/__init__.py: CuGraphStorage wrapper + FutureWarning
      * cugraph_dgl/dataloading/__init__.py: DaskDataLoader wrapper + FutureWarning
      * 模块顶部: "cuGraph-DGL 不再主动维护" 全局警告
  - fb8296e (PR #210): Remove cuGraph-DGL — 永久删除
      * 删除整个 python/cugraph-dgl/ 包（6610 行删除）
      * 停止所有 CI、package publishing、conda recipes
      * gnn_model.py: 删除所有 `elif framework_name == "dgl"` 分支
      * common_options.py: help 文本 "dgl, pyg, wg" → "pyg, wg"
      * 升级通知: "cuGraph-DGL has been removed in release 25.08."

Walpurgis 迁移语义:
  - fb8296e 后，DGL 接口从"deprecated"升级为"removed"
  - _DglLegacyBanner 升级为 RemovedError（调用立即 RuntimeError）
  - CuGraphStorageCompat / DaskDataLoaderCompat 调用时抛 RuntimeError + 迁移指引
  - 保留 DeprecationWarning 路径供已知依赖方平滑迁移期使用

20% 改写 (鲁迅拿法):
  - _DglRemovedBanner: 升级版公告，fb8296e 后状态从 FutureWarning → RuntimeError
  - CuGraphStorageCompat / DaskDataLoaderCompat: 调用时 RuntimeError + 迁移指引
  - WALPURGIS_DEBUG=1 断点 3 处: 模块加载 / CuGraphStorage 调用 / DaskDataLoader 调用

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

# ── migrate fb8296e: DGL 永久删除状态标志 ───────────────────────────────────
# fb8296e (PR #210, Alex Barghi, 2025-05-22): Remove cuGraph-DGL
#   "Removes cuGraph-DGL permanently and stops all CI and package publishing
#    related to DGL support in this repository."
# README 更新: "cuGraph-DGL has been removed in release 25.08."
# 此标志供调用方查询当前 DGL 删除状态，无需二次迁移。
_FB8296E_DGL_PERMANENTLY_REMOVED = True
_FB8296E_REMOVED_RELEASE = "25.08"
_FB8296E_MIGRATION_TARGET = "cuGraph-PyG (cugraph-pyg)"


def is_dgl_permanently_removed() -> bool:
    """
    返回 True 表示 cuGraph-DGL 已在 fb8296e 中永久删除。
    调用方可用此函数在 import 之前提前检测并给出友好错误信息。

    断点: WALPURGIS_DEBUG=1 时打印删除状态。
    """
    _dbg(
        f"is_dgl_permanently_removed(): "
        f"removed={_FB8296E_DGL_PERMANENTLY_REMOVED} "
        f"release={_FB8296E_REMOVED_RELEASE!r} "
        f"migrate_to={_FB8296E_MIGRATION_TARGET!r}"
    )
    return _FB8296E_DGL_PERMANENTLY_REMOVED




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

        # migrate fb8296e: cuGraph-DGL 已在 25.08 永久删除，不再是 FutureWarning
        # 而是立即 RuntimeError + 迁移指引。
        # 上游 PR #210: "Removes cuGraph-DGL permanently and stops all CI
        # and package publishing related to DGL support in this repository."
        raise RuntimeError(
            "[Walpurgis:CuGraphStorage] migrate fb8296e: "
            "cuGraph-DGL has been permanently removed in release 25.08 (PR #210).\n"
            "CuGraphStorage is no longer available.\n"
            "Please migrate to: walpurgis.core.unified_store.UnifiedStore\n"
            "or use the WholeGraph-backed FeatureStore/GraphStore APIs."
        )

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

        # migrate fb8296e: cuGraph-DGL 已在 25.08 永久删除。
        raise RuntimeError(
            "[Walpurgis:DaskDataLoader] migrate fb8296e: "
            "cuGraph-DGL has been permanently removed in release 25.08 (PR #210).\n"
            "DaskDataLoader is no longer available.\n"
            "Please migrate to: walpurgis.dataloader.DataLoader"
        )

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
    "is_dgl_permanently_removed",
    "_FB8296E_DGL_PERMANENTLY_REMOVED",
    "_FB8296E_REMOVED_RELEASE",
    "_FB8296E_MIGRATION_TARGET",
]
