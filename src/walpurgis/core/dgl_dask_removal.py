# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 61a370e
# 原标题: Remove Dask API from cuGraph-DGL (#199)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「沉默呵，沉默呵！不在沉默中爆发，就在沉默中灭亡。」
# —— 鲁迅《纪念刘和珍君》
#
# 61a370e 删除了上游三个核心 Dask API 组件：
#
#   1. cugraph_storage.py（714行）— CuGraphStorage 类
#      上游这是基于 cuDF/dask_cudf 的 DGL GraphStorage 适配层，
#      实现了 DGL 的 DGLGraph-like 接口，底层用 RAPIDS 图存储。
#      删除原因：Dask API 不再维护，推荐使用 cugraph_dgl.Graph。
#
#   2. convert.py 中的 cugraph_storage_from_heterograph（-25行）
#      将 dgl.DGLGraph 转换为 CuGraphStorage 的工厂函数。
#      随 CuGraphStorage 一同删除。
#
#   3. __init__.py 中的 CuGraphStorage wrapper（-12行）
#      上游在 456d5a2 中引入的 FutureWarning 包装器：
#        def CuGraphStorage(*args, **kwargs):
#            warnings.warn("CuGraphStorage and the rest of the dask-based API
#                           are deprecated and will be removed in release 25.08.",
#                          FutureWarning)
#            return DEPRECATED__CuGraphStorage(*args, **kwargs)
#      61a370e 删除了此 wrapper 及其底层 DEPRECATED__CuGraphStorage 导入。
#
# Walpurgis 此前（dgl_deprecation.py）已将 CuGraphStorage 迁移为
# CuGraphStorageCompat（RuntimeError 路径），本文件进一步升级：
#   - StorageRemovalSpec(frozen dataclass): 结构化记录删除元数据
#   - CuGraphStorageGrave: 语义更明确的墓碑类（grave = 坟墓）
#   - cugraph_storage_from_heterograph_grave(): 转换函数墓碑
#   - DaskApiInventory: Dask API 完整清单，供审计使用
#   - self_check(): 验证所有墓碑已注册
#
# 20% 改写（鲁迅拿法）：
#   dgl_deprecation.py 用「compat」命名（暗示兼容层），
#   本文件用「grave/removal」命名（明确是死亡记录），
#   语义上区分「已废弃但可用」与「已彻底删除不可调」。
#
# _dbg 断点 6 处：
#   1. 模块加载
#   2. CuGraphStorageGrave 实例化
#   3. cugraph_storage_from_heterograph_grave 调用
#   4. DaskApiInventory 查询
#   5. StorageRemovalSpec.self_check
#   6. module-level self_check 汇总

from __future__ import annotations

import os as _os
import sys as _sys
import time as _time
import traceback as _traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DGL_DASK_REMOVAL:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ── 断点1: 模块加载 ────────────────────────────────────────────────────────────
_dbg(
    "module_load",
    "dgl_dask_removal.py 加载。"
    "本模块记录 61a370e 删除的 CuGraphStorage / cugraph_storage_from_heterograph / "
    "Dask API 的全量元数据。",
)


# =============================================================================
# StorageRemovalSpec — 结构化删除元数据
# =============================================================================

@dataclass(frozen=True)
class StorageRemovalSpec:
    """
    cugraph_storage.py 和相关 Dask API 的删除规格。

    对应上游 diff：
      python/cugraph-dgl/cugraph_dgl/cugraph_storage.py | 714 deletions
      python/cugraph-dgl/cugraph_dgl/convert.py        |  25 deletions
      python/cugraph-dgl/cugraph_dgl/__init__.py        |  12 deletions
    """
    commit_hash: str = "61a370e"
    pr_number: int = 199
    removed_class: str = "CuGraphStorage"
    removed_function: str = "cugraph_storage_from_heterograph"
    lines_deleted: int = 714          # cugraph_storage.py
    convert_lines_deleted: int = 25   # convert.py 中 cugraph_storage_from_heterograph
    init_lines_deleted: int = 12      # __init__.py 中 wrapper + import

    # CuGraphStorage 的关键公开方法（供迁移参考）
    removed_methods: Tuple[str, ...] = (
        "num_nodes", "num_edges", "ntypes", "etypes", "canonical_etypes",
        "ndata", "edata", "add_nodes", "add_edges",
        "sample_neighbors", "find_edges",
        "node_subgraph", "edge_subgraph",
    )

    # 迁移建议
    migration_targets: Tuple[str, ...] = (
        "walpurgis.graph.Graph — 替代 CuGraphStorage 图存储",
        "walpurgis.graph.graph_from_heterograph — 替代 cugraph_storage_from_heterograph",
        "walpurgis.dataloader.DataLoader — 替代基于 CuGraphStorage 的 DaskDataLoader",
    )

    def self_check(self) -> bool:
        """验证规格完整性。"""
        # ── 断点5 ────────────────────────────────────────────────────────────
        _dbg("StorageRemovalSpec.self_check", f"commit={self.commit_hash}")
        assert self.commit_hash == "61a370e", f"hash 不匹配: {self.commit_hash}"
        assert self.lines_deleted == 714, f"cugraph_storage.py 删除行数应为 714"
        assert self.removed_class == "CuGraphStorage"
        assert self.removed_function == "cugraph_storage_from_heterograph"
        assert len(self.removed_methods) >= 10, "方法清单过短，可能遗漏"
        _dbg("StorageRemovalSpec.self_check", "ALL PASS")
        return True


# =============================================================================
# CuGraphStorageGrave — CuGraphStorage 墓碑类
# =============================================================================

class CuGraphStorageGrave:
    """
    CuGraphStorage 墓碑类（grave = 坟墓，比 compat 更明确的死亡语义）。

    上游 61a370e 删除了 cugraph_storage.py（714行），
    包含 CuGraphStorage 类的全部实现。

    此类替代 dgl_deprecation.py 中的 CuGraphStorageCompat：
    - Compat: 「有 compat 层，也许还能用」
    - Grave:  「这里是坟墓，里面什么都没有」

    实例化立即 RuntimeError + 迁移指引 + 调用栈（DEBUG 模式）。

    断点2: CuGraphStorageGrave 实例化
    """

    def __init__(self, *args, **kwargs) -> None:
        # ── 断点2 ────────────────────────────────────────────────────────────
        _dbg(
            "CuGraphStorageGrave.__init__",
            f"检测到对已删除的 CuGraphStorage 的实例化调用。"
            f"args={[type(a).__name__ for a in args]} "
            f"kwargs={list(kwargs.keys())}",
        )
        if _DEBUG:
            _dbg("CuGraphStorageGrave.__init__", "调用栈（最近5帧）:")
            frames = _traceback.format_stack()
            for frame in frames[-6:-1]:
                _dbg("  stack", frame.strip())

        raise RuntimeError(
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "  CuGraphStorage 已在 cugraph-gnn commit 61a370e 中彻底删除\n"
            "  (Remove Dask API from cuGraph-DGL, PR #199)\n"
            "  删除规模: cugraph_storage.py 714行全部移除\n"
            "═══════════════════════════════════════════════════════════════\n"
            "\n"
            "  此前 (dgl_deprecation.py) 的 FutureWarning 兼容层也已随之删除。\n"
            "  CuGraphStorage 不可用，没有向后兼容路径。\n"
            "\n"
            "迁移路径:\n"
            "  图对象 (替代 CuGraphStorage):\n"
            "    from walpurgis.graph import Graph\n"
            "    g = Graph(is_multi_gpu=False)\n"
            "    g.add_nodes(...); g.add_edges(...)\n"
            "\n"
            "  从 DGLGraph 转换 (替代 cugraph_storage_from_heterograph):\n"
            "    from walpurgis.graph import graph_from_heterograph\n"
            "    wg = graph_from_heterograph(dgl_g, single_gpu=True)\n"
            "\n"
            "  批量采样 (替代 DaskDataLoader + CuGraphStorage):\n"
            "    from walpurgis.dataloader import DataLoader\n"
            "\n"
            "详见: src/walpurgis/core/dgl_dask_removal.py\n"
            "      src/walpurgis/graph/graph.py\n"
        )

    def __repr__(self) -> str:
        return "CuGraphStorageGrave(⚰ 61a370e)"


# =============================================================================
# cugraph_storage_from_heterograph — 转换函数墓碑
# =============================================================================

def cugraph_storage_from_heterograph(*args, **kwargs) -> None:
    """
    cugraph_storage_from_heterograph 墓碑函数。

    上游 61a370e 从 convert.py 中删除了此函数（-25行）：
        def cugraph_storage_from_heterograph(g: dgl.DGLGraph, single_gpu: bool = True):
            num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
            edges_dict = get_edges_dict_from_dgl_HeteroGraph(g, single_gpu)
            gs = CuGraphStorage(data_dict=edges_dict, ...)
            add_ndata_from_dgl_HeteroGraph(gs, g)
            add_edata_from_dgl_HeteroGraph(gs, g)
            return gs

    Walpurgis 中 graph_from_heterograph 是其语义等价替代。
    断点3: cugraph_storage_from_heterograph 调用
    """
    # ── 断点3 ────────────────────────────────────────────────────────────────
    _dbg(
        "cugraph_storage_from_heterograph",
        f"检测到对已删除的 cugraph_storage_from_heterograph 的调用。"
        f"args={[type(a).__name__ for a in args]}",
    )
    raise RuntimeError(
        "cugraph_storage_from_heterograph 已在 commit 61a370e 中删除。\n"
        "替代:\n"
        "  from walpurgis.graph import graph_from_heterograph\n"
        "  wg = graph_from_heterograph(dgl_g, single_gpu=True)\n"
        "\n"
        "graph_from_heterograph 返回 walpurgis.graph.Graph（非 CuGraphStorage）。\n"
        "API 兼容性见 src/walpurgis/graph/convert.py。"
    )


# =============================================================================
# DaskApiInventory — Dask API 完整清单
# =============================================================================

class DaskApiInventory:
    """
    61a370e 删除的 Dask API 完整清单。

    供测试框架和审计工具查询：哪些 API 在此 commit 死亡？
    与 dask_dataloader.py 中的 TombstoneRegistry 互补：
    - TombstoneRegistry: 数据加载器层面的 API
    - DaskApiInventory: 存储层面的 API + 测试文件 + 示例文件

    断点4: DaskApiInventory 查询
    """

    # 上游删除的 Python API（类/函数）
    REMOVED_CLASSES = frozenset({"CuGraphStorage"})
    REMOVED_FUNCTIONS = frozenset({
        "cugraph_storage_from_heterograph",
        "DaskDataLoader",       # dataloading/__init__.py 中移除的导出
        "create_batch_df",      # dask_dataloader.py 中的辅助函数
        "get_batch_id_series",  # dask_dataloader.py 中的辅助函数
    })

    # 上游删除的测试文件（含路径）
    REMOVED_TEST_FILES: Dict[str, int] = {
        "cugraph_dgl/tests/dataloading/test_dask_dataloader.py": 153,
        "cugraph_dgl/tests/dataloading/test_dask_dataloader_mg.py": 121,
        "cugraph_dgl/tests/dataloading/test_dataset.py": 128,
        "cugraph_dgl/tests/test_cugraph_storage.py": 150,
    }

    # 上游删除的示例文件（含路径）
    REMOVED_EXAMPLE_FILES: Dict[str, int] = {
        "cugraph_dgl/examples/dataset_from_disk_cudf.ipynb": 269,
        "cugraph_dgl/examples/graphsage/node-classification.py": 270,
        "cugraph_dgl/examples/multi_trainer_MG_example/model.py": 145,
        "cugraph_dgl/examples/multi_trainer_MG_example/workflow.py": 244,
    }

    @classmethod
    def is_removed_api(cls, name: str) -> bool:
        """检查给定名称是否在 61a370e 中被删除。"""
        # ── 断点4 ────────────────────────────────────────────────────────────
        _dbg("DaskApiInventory.is_removed_api", f"查询: {name!r}")
        result = name in cls.REMOVED_CLASSES or name in cls.REMOVED_FUNCTIONS
        _dbg("DaskApiInventory.is_removed_api", f"  → {result}")
        return result

    @classmethod
    def total_lines_removed(cls) -> int:
        """返回测试 + 示例文件的删除行数总计（不含实现文件）。"""
        test_lines = sum(cls.REMOVED_TEST_FILES.values())
        example_lines = sum(cls.REMOVED_EXAMPLE_FILES.values())
        return test_lines + example_lines

    @classmethod
    def self_check(cls) -> bool:
        """验证清单完整性。"""
        # ── 断点6 ────────────────────────────────────────────────────────────
        _dbg("DaskApiInventory.self_check", "开始验证")
        assert "CuGraphStorage" in cls.REMOVED_CLASSES
        assert "cugraph_storage_from_heterograph" in cls.REMOVED_FUNCTIONS
        assert "DaskDataLoader" in cls.REMOVED_FUNCTIONS
        assert len(cls.REMOVED_TEST_FILES) == 4, "删除的测试文件应为4个"
        assert len(cls.REMOVED_EXAMPLE_FILES) == 4, "删除的示例文件应为4个"
        total = cls.total_lines_removed()
        assert total > 600, f"测试+示例删除行数 {total} 应>600"
        _dbg("DaskApiInventory.self_check", f"ALL PASS (test+example lines={total})")
        return True


# =============================================================================
# 模块级实例 + 公开符号
# =============================================================================

#: CuGraphStorage 墓碑实例（调用即 RuntimeError）
CuGraphStorage = CuGraphStorageGrave

#: 删除规格（结构化文档）
STORAGE_REMOVAL_SPEC = StorageRemovalSpec()

#: Dask API 完整清单
DASK_API_INVENTORY = DaskApiInventory()


def _module_self_check() -> bool:
    """模块级完整性检查，覆盖所有墓碑和规格。"""
    _dbg("_module_self_check", "开始模块级验证")
    STORAGE_REMOVAL_SPEC.self_check()
    DaskApiInventory.self_check()
    _dbg("_module_self_check", "ALL PASS — dgl_dask_removal.py 完整性验证通过")
    return True


__all__ = [
    "StorageRemovalSpec",
    "CuGraphStorageGrave",
    "CuGraphStorage",
    "cugraph_storage_from_heterograph",
    "DaskApiInventory",
    "STORAGE_REMOVAL_SPEC",
    "DASK_API_INVENTORY",
    "_module_self_check",
]
