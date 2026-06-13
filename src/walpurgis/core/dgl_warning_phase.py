"""
dgl_warning_phase.py — 456d5a2 迁移: 为 DGL 类添加废弃警告

migrate 456d5a2: add deprecation warnings for DGL classes

上游变化 (456d5a2, cugraph-gnn):
  2 files changed, 26 insertions(+), 8 deletions(-)

核心变化:
  1. cugraph_dgl/__init__.py:
     - import warnings 新增
     - CuGraphStorage 重命名为 DEPRECATED__CuGraphStorage
     - 新增 CuGraphStorage wrapper 函数 (FutureWarning)
     - 新增模块级警告: "cuGraph-DGL is no longer under active development"

  2. cugraph_dgl/dataloading/__init__.py:
     - DaskDataLoader 重命名为 DEPRECATED__DaskDataLoader
     - DataLoader wrapper 更新警告文本:
       旧: "DataLoader has been renamed to DaskDataLoader. In Release 24.10..."
       新: "CuGraphStorage and the rest of the dask-based API are deprecated
            and will be removed in release 25.08."
     - DataLoader 现在返回 DEPRECATED__DaskDataLoader（而非 DaskDataLoader）

456d5a2 在 DGL 废弃时间线中的位置:
  阶段1 — 废弃警告 (456d5a2): FutureWarning + 模块级公告，API 仍可调用
  阶段2 — 完全删除 (61a370e / fb8296e): 整个 cugraph-dgl 包被移除

与 dgl_deprecation.py 的关系:
  - dgl_deprecation.py: 覆盖了 456d5a2 + fb8296e 的合并效果（废弃→删除）
  - dgl_warning_phase.py (本文件): 单独记录 456d5a2 的「警告阶段」语义，
    作为废弃时间线的第一阶段文档，供历史追溯和回归测试参考

Walpurgis 改写 20%（鲁迅拿法）:
- DglWarningPhaseSpec(frozen dataclass): 结构化记录警告阶段的完整参数
  上游: 裸 warnings.warn() 散布在 __init__.py 各处；
  Walpurgis: DglWarningPhaseSpec 统一携带警告文本、目标版本、替代 API
- DglModuleBanner: 封装模块级公告（"cuGraph-DGL 不再主动维护"），
  区别于 API 级 FutureWarning，是针对整个子包的存续状态公告
- DglDeprecationTimeline: 枚举 456d5a2 在整个 DGL 废弃时间线中的位置，
  比 dgl_deprecation.py 的「合并效果」描述更细粒度
- WarnOnlyGate: 仅发出警告（不抛错）的轻量 wrapper，
  对应 456d5a2 阶段（「deprecated but callable」），
  与 dgl_deprecation.py 的 RuntimeError gate 形成对比

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:dgl_warning_phase:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DglDeprecationTimeline — 枚举 DGL 废弃时间线的阶段
# ─────────────────────────────────────────────────────────────────────────────

class DglDeprecationStage(Enum):
    """
    cuGraph-DGL 废弃时间线的阶段枚举。

    对应上游两个关键 commit:
    - 456d5a2: WARN_PHASE — FutureWarning + 模块级公告，API 仍可调用
    - 61a370e/fb8296e: REMOVAL_PHASE — 整个包删除，调用即 RuntimeError/ImportError
    """
    WARN_PHASE = auto()      # 456d5a2: deprecated but callable
    REMOVAL_PHASE = auto()   # 61a370e: package deleted


@dataclass(frozen=True)
class DglDeprecationTimelineEntry:
    """DGL 废弃时间线中的单个阶段记录。"""
    stage: DglDeprecationStage
    upstream_commit: str
    description: str
    affected_apis: Tuple[str, ...]
    target_removal_version: Optional[str]   # 预计删除版本（warn 阶段）
    actual_removal_version: Optional[str]   # 实际删除版本（removal 阶段）


DGL_DEPRECATION_TIMELINE: Tuple[DglDeprecationTimelineEntry, ...] = (
    DglDeprecationTimelineEntry(
        stage=DglDeprecationStage.WARN_PHASE,
        upstream_commit="456d5a2",
        description="Add deprecation warnings for DGL classes",
        affected_apis=("CuGraphStorage", "DataLoader", "DaskDataLoader"),
        target_removal_version="25.08",
        actual_removal_version=None,
    ),
    DglDeprecationTimelineEntry(
        stage=DglDeprecationStage.REMOVAL_PHASE,
        upstream_commit="61a370e/fb8296e",
        description="Remove Dask API from cuGraph-DGL / Remove entire package",
        affected_apis=("cugraph_dgl.*",),
        target_removal_version=None,
        actual_removal_version="25.08",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# DglWarningPhaseSpec — 结构化记录 456d5a2 警告阶段的完整参数
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DglWarningPhaseSpec:
    """
    单个 DGL 废弃 API 在 456d5a2 警告阶段的完整规格。

    上游 456d5a2 中每个废弃 wrapper 函数的参数分散在代码中；
    DglWarningPhaseSpec 将这些参数结构化，便于：
    1. 测试框架验证警告文本是否符合规范
    2. 审计工具追踪所有 DGL 废弃 API 的状态
    3. 未来版本升级时批量更新版本号
    """
    api_name: str
    deprecated_via: str           # 触发废弃的 upstream class/function name
    warning_text: str             # 456d5a2 中的 FutureWarning 文本（近似）
    removal_version: str          # 预计删除版本
    migration_hint: str           # 迁移建议
    is_module_level: bool = False # True 表示这是模块级警告（非 API 调用时触发）


#: 456d5a2 引入的废弃 API 规格清单
DGL_WARN_PHASE_SPECS: Tuple[DglWarningPhaseSpec, ...] = (
    DglWarningPhaseSpec(
        api_name="CuGraphStorage",
        deprecated_via="DEPRECATED__CuGraphStorage",
        warning_text=(
            "CuGraphStorage and the rest of the dask-based API are deprecated"
            "and will be removed in release 25.08."
        ),
        removal_version="25.08",
        migration_hint="Use walpurgis.graph.Graph instead of CuGraphStorage.",
        is_module_level=False,
    ),
    DglWarningPhaseSpec(
        api_name="DataLoader (DaskDataLoader alias)",
        deprecated_via="DEPRECATED__DaskDataLoader",
        warning_text=(
            "CuGraphStorage and the rest of the dask-based API are deprecated"
            "and will be removed in release 25.08."
        ),
        removal_version="25.08",
        migration_hint="Use walpurgis.dataloader.DataLoader instead.",
        is_module_level=False,
    ),
    DglWarningPhaseSpec(
        api_name="cugraph_dgl (module)",
        deprecated_via="module-level warnings.warn",
        warning_text=(
            "cuGraph-DGL is no longer"
            "under active development.  We strongly recommend migrating to"
            "cuGraph-PyG."
        ),
        removal_version="25.08",
        migration_hint="Migrate to cuGraph-PyG / walpurgis.dataloader.",
        is_module_level=True,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# DglModuleBanner — 封装模块级公告
# ─────────────────────────────────────────────────────────────────────────────

class DglModuleBanner:
    """
    封装 456d5a2 引入的模块级公告:
        warnings.warn("cuGraph-DGL is no longer under active development. ...")

    上游在 cugraph_dgl/__init__.py 顶层直接调用 warnings.warn()；
    Walpurgis 封装为 DglModuleBanner，加：
    - issued: 追踪是否已发出（防重复）
    - stage: 当前阶段（WARN_PHASE vs REMOVAL_PHASE）
    - emit_for_stage(): 根据阶段选择 FutureWarning 或 RuntimeError
    """

    def __init__(self, stage: DglDeprecationStage = DglDeprecationStage.WARN_PHASE):
        self._stage = stage
        self._issued = False

    @property
    def issued(self) -> bool:
        return self._issued

    def emit(self) -> None:
        """发出模块级公告（仅发一次）。"""
        if self._issued:
            return
        self._issued = True

        if self._stage == DglDeprecationStage.WARN_PHASE:
            warnings.warn(
                "cuGraph-DGL is no longer under active development. "
                "We strongly recommend migrating to cuGraph-PyG / "
                "walpurgis.dataloader.DataLoader.",
                FutureWarning,
                stacklevel=3,
            )
        else:
            # REMOVAL_PHASE: 模块不可用，发 RuntimeError
            raise RuntimeError(
                "cuGraph-DGL has been removed. "
                "Please migrate to walpurgis.dataloader.DataLoader."
            )

        _dbg(
            "DglModuleBanner",
            "emitted",
            stage=self._stage.name,
        )

    def __repr__(self):
        return f"DglModuleBanner(stage={self._stage.name}, issued={self._issued})"


# ─────────────────────────────────────────────────────────────────────────────
# WarnOnlyGate — 仅发出警告（不抛错）的轻量 wrapper
#
# 对应 456d5a2 阶段的「deprecated but callable」语义，
# 与 dgl_deprecation.py 的 RuntimeError gate 形成对比：
#
#   456d5a2 阶段 (WarnOnlyGate): warn → 调用原始构造函数 → 返回实例
#   61a370e 阶段 (dgl_deprecation.py): warn → RuntimeError（不返回）
# ─────────────────────────────────────────────────────────────────────────────

class WarnOnlyGate:
    """
    「仅警告」包装器，对应 456d5a2 的废弃阶段（API 仍可调用）。

    上游 456d5a2 中每个 wrapper 函数:
        def CuGraphStorage(*args, **kwargs):
            warnings.warn("...", FutureWarning)
            return DEPRECATED__CuGraphStorage(*args, **kwargs)

    WarnOnlyGate 等价但加 call_count + WALPURGIS_DEBUG 支持。
    与 dgl_deprecation.py 的 CuGraphStorageCompat/DaskDataLoaderCompat 区别:
    - WarnOnlyGate: 仍然返回实例（warn only, 对应 456d5a2）
    - Compat 类: 抛 RuntimeError（对应 61a370e/fb8296e 删除后）
    """

    def __init__(
        self,
        name: str,
        wrapped_cls,
        spec: DglWarningPhaseSpec,
        stacklevel: int = 3,
    ):
        self._name = name
        self._wrapped_cls = wrapped_cls
        self._spec = spec
        self._stacklevel = stacklevel
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        warnings.warn(
            self._spec.warning_text,
            FutureWarning,
            stacklevel=self._stacklevel,
        )
        _dbg(
            f"WarnOnlyGate:{self._name}",
            f"called (call_count={self._call_count})",
            migration_hint=self._spec.migration_hint,
            args_types=str([type(a).__name__ for a in args]),
        )
        return self._wrapped_cls(*args, **kwargs)

    def __repr__(self):
        return (
            f"WarnOnlyGate(name={self._name!r}, "
            f"wrapped={self._wrapped_cls.__name__!r}, "
            f"stage=WARN_PHASE, "
            f"call_count={self._call_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 全局模块级公告实例（对应 456d5a2 阶段）
# ─────────────────────────────────────────────────────────────────────────────

#: 456d5a2 阶段的模块级公告（WARN_PHASE，不抛错）
dgl_module_banner_warn_phase = DglModuleBanner(stage=DglDeprecationStage.WARN_PHASE)

#: 61a370e 阶段的模块级公告（REMOVAL_PHASE，抛 RuntimeError）
dgl_module_banner_removal_phase = DglModuleBanner(stage=DglDeprecationStage.REMOVAL_PHASE)


def get_current_dgl_timeline_entry() -> DglDeprecationTimelineEntry:
    """
    返回当前最终生效的 DGL 废弃时间线条目。

    由于上游 61a370e/fb8296e 已完成删除，当前阶段是 REMOVAL_PHASE。
    此函数供测试和审计工具查询「DGL 现在处于哪个废弃阶段」。
    """
    # 当前（fb8296e 后）处于 REMOVAL_PHASE
    for entry in DGL_DEPRECATION_TIMELINE:
        if entry.stage == DglDeprecationStage.REMOVAL_PHASE:
            return entry
    return DGL_DEPRECATION_TIMELINE[-1]


_dbg(
    "module",
    "dgl_warning_phase loaded",
    specs=len(DGL_WARN_PHASE_SPECS),
    timeline_entries=len(DGL_DEPRECATION_TIMELINE),
)
