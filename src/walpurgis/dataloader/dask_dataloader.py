# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 61a370e
# 原标题: Remove Dask API from cuGraph-DGL (#199)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「我向来是不惮以最坏的恶意来推测中国人的，然而我还不料，也不信竟会凶残到这地步。」
# —— 鲁迅《纪念刘和珍君》
#
# 61a370e 将上游 cugraph_dgl/dataloading/dask_dataloader.py（321行）
# 与 cugraph_dgl/cugraph_storage.py（714行）一并删除，
# 标志着 cuGraph-DGL Dask API 的彻底终结。
# Walpurgis 此文件曾迁移自 f4ca484（DaskDataLoader 的完整实现）。
# 本次改写：将实现替换为结构化墓碑（Tombstone），保留历史追溯链。
#
# Walpurgis 20% 改写要点（鲁迅拿法）：
#   1. DaskRemovalRecord(frozen dataclass) — 文档化删除理由、受影响文件清单、
#      迁移建议路径，替代上游的「直接删除」
#   2. _DaskDataLoaderTombstone — 实例化立即 RuntimeError + 迁移指引，
#      比上游的「import 就 ModuleNotFoundError」更友好
#   3. TombstoneRegistry — 集中管理所有被删除的 Dask API 入口，
#      可供测试框架查询「哪些 API 已死」
#   4. 全链路 WALPURGIS_DEBUG=1 断点 6 处：
#      - 模块加载：检测到墓碑模块被 import 时输出警告
#      - DaskDataLoader 实例化：捕获误用调用，打印调用栈摘要
#      - create_batch_df 调用：同上
#      - get_batch_id_series 调用：同上
#      - TombstoneRegistry 查询：记录谁在问「这个 API 还在吗？」
#      - self_check：验证所有墓碑入口均已注册
#
# 迁移路径（从 DaskDataLoader 迁移到现代 API）：
#   - 单 GPU 批量采样: walpurgis.dataloader.DataLoader（非 Dask 路径）
#   - 分布式 DGL 采样: walpurgis.dataloader.DGLDataLoader
#   - GraphSAGE 边采样: walpurgis.dataloader.LinkNeighborLoader
#   - 图特征存储: walpurgis.graph.Graph（替代 CuGraphStorage）

from __future__ import annotations

import os as _os
import sys as _sys
import time as _time
import traceback as _traceback
from dataclasses import dataclass, field
from typing import List, Optional

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DASK_TOMBSTONE:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ── 断点1: 墓碑模块加载检测 ──────────────────────────────────────────────────
_dbg(
    "module_load",
    "dask_dataloader.py 墓碑模块被 import。"
    "此模块已在 61a370e (Remove Dask API from cuGraph-DGL #199) 中移除。"
    "请迁移至 walpurgis.dataloader.DataLoader 或 walpurgis.graph.Graph。",
)


@dataclass(frozen=True)
class DaskRemovalRecord:
    """
    61a370e 删除的文件和 API 结构化记录。

    对应上游 PR #199 Remove Dask API from cuGraph-DGL 的 diff：
    - 15 files changed, 6 insertions(+), 2634 deletions(-)
    """
    commit_hash: str = "61a370e"
    pr_number: int = 199
    pr_title: str = "Remove Dask API from cuGraph-DGL"
    deletions: int = 2634
    insertions: int = 6

    # 上游删除的文件（walpurgis 路径）
    deleted_upstream_files: List[str] = field(default_factory=lambda: [
        "cugraph_dgl/cugraph_storage.py",             # 714行 — CuGraphStorage 主类
        "cugraph_dgl/dataloading/dask_dataloader.py", # 321行 — DaskDataLoader 主类
        "cugraph_dgl/tests/dataloading/test_dask_dataloader.py",     # 153行
        "cugraph_dgl/tests/dataloading/test_dask_dataloader_mg.py",  # 121行
        "cugraph_dgl/tests/dataloading/test_dataset.py",             # 128行
        "cugraph_dgl/tests/test_cugraph_storage.py",                 # 150行
        "cugraph_dgl/examples/dataset_from_disk_cudf.ipynb",         # 269行
        "cugraph_dgl/examples/graphsage/node-classification.py",     # 270行
        "cugraph_dgl/examples/multi_trainer_MG_example/model.py",    # 145行
        "cugraph_dgl/examples/multi_trainer_MG_example/workflow.py", # 244行
    ])

    # 上游修改的文件（walpurgis 路径）
    modified_upstream_files: List[str] = field(default_factory=lambda: [
        "cugraph_dgl/__init__.py",                                    # -12行: CuGraphStorage wrapper + import 移除
        "cugraph_dgl/convert.py",                                     # -25行: cugraph_storage_from_heterograph 移除
        "cugraph_dgl/dataloading/__init__.py",                        # -12+: DaskDataLoader 导出移除
        "cugraph_dgl/tests/dataloading/test_from_dgl_heterograph.py", # -37+: storage 引用移除
        "cugraph_dgl/tests/utils.py",                                 # -39+: 工具函数清理
    ])

    # Walpurgis 对应的处置方式
    walpurgis_actions: List[str] = field(default_factory=lambda: [
        "src/walpurgis/dataloader/__init__.py — 移除 DaskDataLoader 导出",
        "src/walpurgis/dataloader/dask_dataloader.py — 改写为墓碑模块（本文件）",
        "src/walpurgis/graph/convert.py — cugraph_storage_from_heterograph 从未迁移（SKIP）",
        "src/walpurgis/core/dgl_dask_removal.py — CuGraphStorage 删除的结构化记录",
        "src/walpurgis/tests/sampler/ — 相关 dask 测试文件 SKIP（已无对应上游实现）",
        "MIGRATION_LOG.md — 记录本次迁移",
    ])

    def self_check(self) -> bool:
        """验证记录完整性。"""
        _dbg("DaskRemovalRecord.self_check", f"验证 commit={self.commit_hash}")
        assert self.commit_hash == "61a370e", "commit hash 不匹配"
        assert len(self.deleted_upstream_files) == 10, f"删除文件数应为10，实际{len(self.deleted_upstream_files)}"
        assert self.deletions > self.insertions, "删除行数应远大于新增"
        _dbg("DaskRemovalRecord.self_check", "ALL PASS")
        return True


# ── 墓碑入口：DaskDataLoader ──────────────────────────────────────────────────

class _DaskDataLoaderTombstone:
    """
    DaskDataLoader 墓碑类。

    上游 61a370e 删除了整个 dask_dataloader.py。
    实例化此类会立即抛出 RuntimeError + 迁移指引，
    比 ImportError 更友好：可以携带上下文和迁移路径。

    断点2: DaskDataLoader 实例化调用
    """

    def __init__(self, *args, **kwargs) -> None:
        # ── 断点2 ────────────────────────────────────────────────────────────
        _dbg(
            "DaskDataLoaderTombstone.__init__",
            f"检测到对已删除的 DaskDataLoader 的实例化调用。"
            f"args={[type(a).__name__ for a in args]} "
            f"kwargs={list(kwargs.keys())}",
        )
        if _DEBUG:
            _dbg("DaskDataLoaderTombstone.__init__", "调用栈:")
            for line in _traceback.format_stack()[:-1]:
                _dbg("  stack", line.strip())

        raise RuntimeError(
            "\n"
            "═══════════════════════════════════════════════════════\n"
            "  DaskDataLoader 已在 cugraph-gnn commit 61a370e 中删除\n"
            "  (Remove Dask API from cuGraph-DGL, PR #199)\n"
            "═══════════════════════════════════════════════════════\n"
            "\n"
            "迁移路径:\n"
            "  单 GPU 批量采样:\n"
            "    from walpurgis.dataloader import DataLoader\n"
            "\n"
            "  DGL 分布式采样:\n"
            "    from walpurgis.dataloader import DGLDataLoader\n"
            "\n"
            "  GraphSAGE 风格边采样:\n"
            "    from walpurgis.dataloader import LinkNeighborLoader\n"
            "\n"
            "  图对象（替代 CuGraphStorage）:\n"
            "    from walpurgis.graph import Graph\n"
            "\n"
            "详见: src/walpurgis/core/dgl_dask_removal.py\n"
        )


def _create_batch_df_tombstone(*args, **kwargs):
    """
    create_batch_df 墓碑函数。

    上游 61a370e 删除了此辅助函数（随 dask_dataloader.py 一同删除）。
    断点3: create_batch_df 调用
    """
    # ── 断点3 ────────────────────────────────────────────────────────────────
    _dbg(
        "create_batch_df_tombstone",
        f"检测到对已删除的 create_batch_df 的调用。args={len(args)} kwargs={list(kwargs.keys())}",
    )
    raise RuntimeError(
        "create_batch_df 已在 commit 61a370e 中随 dask_dataloader.py 一同删除。\n"
        "此函数依赖 dask_cudf，属于 Dask API 的一部分。\n"
        "替代方案: 直接操作 torch.Tensor 或 walpurgis.dataloader.DataLoader。"
    )


def _get_batch_id_series_tombstone(*args, **kwargs):
    """
    get_batch_id_series 墓碑函数。

    断点4: get_batch_id_series 调用
    """
    # ── 断点4 ────────────────────────────────────────────────────────────────
    _dbg(
        "get_batch_id_series_tombstone",
        f"检测到对已删除的 get_batch_id_series 的调用。args={len(args)} kwargs={list(kwargs.keys())}",
    )
    raise RuntimeError(
        "get_batch_id_series 已在 commit 61a370e 中随 dask_dataloader.py 一同删除。\n"
        "此函数依赖 dask_cudf Series 类型，属于 Dask API 的一部分。\n"
        "替代方案: 使用 torch.arange 或 torch.Tensor 直接生成批次索引。"
    )


# ── TombstoneRegistry ─────────────────────────────────────────────────────────

class TombstoneRegistry:
    """
    墓碑注册表：集中管理所有被 61a370e 删除的 Dask API 入口。

    可供测试框架查询「哪些 API 已死亡」，防止误用或回归。
    断点5: TombstoneRegistry 查询
    """

    _registry = {
        "DaskDataLoader": _DaskDataLoaderTombstone,
        "create_batch_df": _create_batch_df_tombstone,
        "get_batch_id_series": _get_batch_id_series_tombstone,
        "CuGraphStorage": None,            # 在 dgl_dask_removal.py 注册
        "cugraph_storage_from_heterograph": None,  # 从未迁移到 walpurgis
    }

    @classmethod
    def is_removed(cls, api_name: str) -> bool:
        """检查某个 API 名称是否在 61a370e 中被删除。"""
        # ── 断点5 ────────────────────────────────────────────────────────────
        _dbg("TombstoneRegistry.is_removed", f"查询: {api_name!r}")
        result = api_name in cls._registry
        _dbg("TombstoneRegistry.is_removed", f"  → {result}")
        return result

    @classmethod
    def list_removed(cls) -> List[str]:
        """返回所有已删除的 API 名称列表。"""
        return list(cls._registry.keys())

    @classmethod
    def self_check(cls) -> bool:
        """验证所有已知的 Dask API 入口均已注册为墓碑。"""
        # ── 断点6 ────────────────────────────────────────────────────────────
        _dbg("TombstoneRegistry.self_check", f"注册项: {cls.list_removed()}")
        expected = {
            "DaskDataLoader", "create_batch_df", "get_batch_id_series",
            "CuGraphStorage", "cugraph_storage_from_heterograph",
        }
        actual = set(cls._registry.keys())
        assert actual == expected, f"墓碑注册不完整: 缺少 {expected - actual}，多余 {actual - expected}"
        _dbg("TombstoneRegistry.self_check", "ALL PASS")
        return True


# ── 公开别名（墓碑）————供误用者得到友好 RuntimeError 而非 AttributeError ─────

#: DaskDataLoader 墓碑 — 实例化即 RuntimeError（61a370e）
DaskDataLoader = _DaskDataLoaderTombstone

#: create_batch_df 墓碑 — 调用即 RuntimeError（61a370e）
create_batch_df = _create_batch_df_tombstone

#: get_batch_id_series 墓碑 — 调用即 RuntimeError（61a370e）
get_batch_id_series = _get_batch_id_series_tombstone

#: 删除记录（结构化文档）
DASK_REMOVAL_RECORD = DaskRemovalRecord()


# ── migrate a57912c: fix references to dask data loader ──────────────────────
# 上游 a57912c 修正了两个测试文件中对 dask dataloader 的引用：
#
#   test_dask_dataloader.py:
#     - 旧: cugraph_dgl.dataloading.DaskDataLoader(...)
#     - 新: cugraph_dgl.dataloading.dask_dataloader.DaskDataLoader(...)
#       (完整模块路径，绕过包级别 __init__ 的别名链)
#
#   test_dask_dataloader_mg.py:
#     - 旧: cugraph_dgl.dataloading.DataLoader(...)  (1b2fce2 改完的结果)
#     - 新: cugraph_dgl.dataloading.dask_dataloader.DaskDataLoader(...)
#       (再次回到 DaskDataLoader，但用完整路径 — 测试就是要测废弃路径的行为)
#
# a57912c 的核心洞察：测试应通过「完整模块路径」而非「包级别别名」访问被测对象，
# 这样即使 __init__ 发生重组也不会静默改变测试的实际目标。
#
# Walpurgis 20% 改写（鲁迅拿法）：
# 新增 DataloaderRefSpec 数据类，将「完整路径 vs 别名」两种引用方式结构化记录，
# 便于代码审查时快速定位「这个测试到底在测什么」。

@dataclass(frozen=True)
class DataloaderRefSpec:
    """DataLoader 引用规格（migrate a57912c Walpurgis 改写）。

    a57912c 的核心：用完整模块路径引用被测对象，
    比包级别别名更稳定，不受 __init__ 重组影响。

    在 Walpurgis 中，dask_dataloader 模块本身就是墓碑，
    DataloaderRefSpec 记录测试应如何正确引用各种 loader 类型。
    """

    # 完整模块路径（a57912c 规范化的引用方式）
    fully_qualified_path: str

    # 包级别别名（历史上使用的方式，现在可能已废弃或重定向）
    package_alias: str

    # 是否已废弃（对应是否需要像 a57912c 一样修正）
    is_deprecated_alias: bool

    # 修正 commit（若有）
    fix_commit: str = ""

    def __post_init__(self):
        """验证路径格式。"""
        assert "." in self.fully_qualified_path, (
            f"fully_qualified_path={self.fully_qualified_path!r} 应包含模块路径分隔符"
        )


# a57912c 规范化的 DataLoader 引用记录
DATALOADER_REF_SPECS: tuple = (
    DataloaderRefSpec(
        fully_qualified_path="walpurgis.dataloader.dataloader.DataLoader",
        package_alias="walpurgis.dataloader.DataLoader",
        is_deprecated_alias=False,
        fix_commit="",
    ),
    DataloaderRefSpec(
        fully_qualified_path="walpurgis.dataloader.dask_dataloader.DaskDataLoader",
        package_alias="walpurgis.dataloader.DaskDataLoader",
        is_deprecated_alias=True,
        fix_commit="a57912c",  # 上游此 commit 规范化引用方式
    ),
    DataloaderRefSpec(
        fully_qualified_path="walpurgis.dataloader.dgl_dataloader.DataLoader",
        package_alias="walpurgis.dataloader.DGLDataLoader",
        is_deprecated_alias=False,
        fix_commit="",
    ),
)


__all__ = [
    "DaskDataLoader",
    "create_batch_df",
    "get_batch_id_series",
    "DaskRemovalRecord",
    "TombstoneRegistry",
    "DASK_REMOVAL_RECORD",
    "DataloaderRefSpec",
    "DATALOADER_REF_SPECS",
]
