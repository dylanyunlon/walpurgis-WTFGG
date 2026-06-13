# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 61a370e
# 原标题: Remove Dask API from cuGraph-DGL (#199)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「过去的生命已经死亡。我对于这死亡有大欢喜，因为我借此知道它曾经存活。」
# —— 鲁迅《野草·题辞》
#
# 61a370e 删除了 4 个测试文件（共 552 行）:
#   - cugraph_dgl/tests/dataloading/test_dask_dataloader.py    (153行)
#   - cugraph_dgl/tests/dataloading/test_dask_dataloader_mg.py (121行)
#   - cugraph_dgl/tests/dataloading/test_dataset.py            (128行)
#   - cugraph_dgl/tests/test_cugraph_storage.py                (150行)
#
# Walpurgis 处置：
#   - 上述文件测试的 API（DaskDataLoader, CuGraphStorage, HeteroBulkSamplerDataset）
#     在 walpurgis 中均未迁移（依赖 dask_cudf / CuGraphStorage）
#   - 本文件记录删除轨迹，并提供验证「这些 API 确实不存在」的 pytest 测试
#   - SKIP 的理由文档化为 SkipRecord dataclass，便于审计
#
# 20% 改写（鲁迅拿法）：
#   上游是「直接 git rm」，Walpurgis 是「留下墓志铭」——
#   记录每个测试曾经测试过什么，以及为什么现在可以删除。
#
# _dbg 断点 4 处：
#   1. 模块加载
#   2. 验证 DaskDataLoader 不可导入
#   3. 验证 CuGraphStorage 不可实例化
#   4. 验证 cugraph_storage_from_heterograph 不可调用

import os
import sys
import time

import pytest

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(
            f"[WALPURGIS-TEST-DASK-TOMBSTONE:{tag}][{time.strftime('%H:%M:%S')}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ── 断点1: 模块加载 ────────────────────────────────────────────────────────────
_dbg(
    "module_load",
    "test_dask_storage_tombstone.py 加载。"
    "记录 61a370e 删除的 4 个测试文件的迁移轨迹。",
)

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SkipRecord:
    """SKIP 理由结构化记录。"""
    original_file: str
    lines: int
    what_it_tested: str
    why_skipped: str
    walpurgis_note: str


SKIP_RECORDS: Tuple[SkipRecord, ...] = (
    SkipRecord(
        original_file="cugraph_dgl/tests/dataloading/test_dask_dataloader.py",
        lines=153,
        what_it_tested=(
            "DaskDataLoader 实例化、__iter__ 批量采样、"
            "HomogenousBulkSamplerDataset 构建"
        ),
        why_skipped=(
            "DaskDataLoader 在 61a370e 中删除。"
            "依赖 dask_cudf 和 BulkSampler，walpurgis 无对应实现。"
        ),
        walpurgis_note=(
            "等价功能: walpurgis.dataloader.DataLoader（非 Dask 路径）。"
            "采样逻辑参见 src/walpurgis/sampler/sampler.py。"
        ),
    ),
    SkipRecord(
        original_file="cugraph_dgl/tests/dataloading/test_dask_dataloader_mg.py",
        lines=121,
        what_it_tested=(
            "DaskDataLoader 多 GPU 路径、DDP 分布式采样、"
            "create_homogeneous_bulk_sampler 多进程构建"
        ),
        why_skipped=(
            "DaskDataLoader mg 路径在 61a370e 中删除。"
            "依赖 torch.distributed + dask CUDA 上下文，walpurgis 无此依赖组合。"
        ),
        walpurgis_note=(
            "等价功能: walpurgis.dataloader.DGLDataLoader（多 GPU DGL 采样）。"
            "参见 src/walpurgis/dataloader/dgl_dataloader.py。"
        ),
    ),
    SkipRecord(
        original_file="cugraph_dgl/tests/dataloading/test_dataset.py",
        lines=128,
        what_it_tested=(
            "HomogenousBulkSamplerDataset / HeterogenousBulkSamplerDataset 的"
            "磁盘读写、batch 索引、cuDF DataFrame 构造"
        ),
        why_skipped=(
            "Dataset 类随 dask_dataloader.py 在 61a370e 中删除。"
            "依赖 cuDF/dask_cudf，walpurgis 使用 PyTorch Dataset 替代。"
        ),
        walpurgis_note=(
            "等价功能: walpurgis.dataloader.node_classification.NodeClassificationDataset。"
            "参见 src/walpurgis/dataloader/node_classification.py。"
        ),
    ),
    SkipRecord(
        original_file="cugraph_dgl/tests/test_cugraph_storage.py",
        lines=150,
        what_it_tested=(
            "CuGraphStorage 构造、num_nodes/num_edges/ntypes/etypes 属性、"
            "sample_neighbors、node_subgraph、edata/ndata 访问"
        ),
        why_skipped=(
            "CuGraphStorage 在 61a370e 中删除（714行）。"
            "整个测试文件失去测试对象。"
        ),
        walpurgis_note=(
            "等价功能: walpurgis.graph.Graph。"
            "图操作测试参见 src/walpurgis/tests/graph/test_graph.py。"
        ),
    ),
)


# =============================================================================
# pytest 测试：验证这些 API 确实不可用（逆向测试，确认删除生效）
# =============================================================================


def test_dask_api_inventory_complete():
    """
    验证 DaskApiInventory 包含 61a370e 删除的所有 API 名称。
    断点2 触发点（查询 DastApiInventory）。
    """
    from walpurgis.core.dgl_dask_removal import DaskApiInventory, DaskApiInventory

    # ── 断点2 ────────────────────────────────────────────────────────────────
    _dbg("test", "验证 DaskApiInventory 完整性")

    assert DaskApiInventory.is_removed_api("CuGraphStorage"), (
        "CuGraphStorage 应在 DaskApiInventory.REMOVED_CLASSES 中"
    )
    assert DaskApiInventory.is_removed_api("DaskDataLoader"), (
        "DaskDataLoader 应在 DaskApiInventory.REMOVED_FUNCTIONS 中"
    )
    assert DaskApiInventory.is_removed_api("cugraph_storage_from_heterograph"), (
        "cugraph_storage_from_heterograph 应在 DaskApiInventory.REMOVED_FUNCTIONS 中"
    )
    assert DaskApiInventory.self_check()

    _dbg("test", "PASS: DaskApiInventory 完整性验证通过")


def test_cugraph_storage_raises_runtime_error():
    """
    验证 CuGraphStorage 实例化抛出 RuntimeError（不是 ImportError 或 AttributeError）。

    对应 test_cugraph_storage.py 的逆向验证：
    「CuGraphStorage 不再可用」这一事实本身是可测试的。

    断点3: CuGraphStorage 实例化
    """
    # ── 断点3 ────────────────────────────────────────────────────────────────
    _dbg("test", "验证 CuGraphStorage 实例化抛出 RuntimeError")

    from walpurgis.core.dgl_dask_removal import CuGraphStorage
    with pytest.raises(RuntimeError, match="61a370e"):
        CuGraphStorage(data_dict={}, num_nodes_dict={})

    _dbg("test", "PASS: CuGraphStorage 正确抛出 RuntimeError")


def test_cugraph_storage_from_heterograph_raises_runtime_error():
    """
    验证 cugraph_storage_from_heterograph 调用抛出 RuntimeError。

    对应 test_from_dgl_heterograph.py 中被 61a370e 修改的测试
    （移除 storage 相关断言）的逆向验证。

    断点4: cugraph_storage_from_heterograph 调用
    """
    # ── 断点4 ────────────────────────────────────────────────────────────────
    _dbg("test", "验证 cugraph_storage_from_heterograph 抛出 RuntimeError")

    from walpurgis.graph.convert import cugraph_storage_from_heterograph
    with pytest.raises(RuntimeError, match="61a370e"):
        cugraph_storage_from_heterograph(None, single_gpu=True)

    _dbg("test", "PASS: cugraph_storage_from_heterograph 正确抛出 RuntimeError")


def test_dask_dataloader_raises_runtime_error():
    """
    验证 DaskDataLoader 实例化抛出 RuntimeError。

    对应 test_dask_dataloader.py / test_dask_dataloader_mg.py 的逆向验证。
    """
    _dbg("test", "验证 DaskDataLoader 实例化抛出 RuntimeError")

    from walpurgis.dataloader.dask_dataloader import DaskDataLoader
    with pytest.raises(RuntimeError, match="Dask API"):
        DaskDataLoader(None, None, batch_size=32)

    _dbg("test", "PASS: DaskDataLoader 正确抛出 RuntimeError")


def test_skip_records_complete():
    """验证 SkipRecord 覆盖所有 4 个删除的测试文件。"""
    _dbg("test", f"验证 SkipRecord 数量={len(SKIP_RECORDS)}")
    assert len(SKIP_RECORDS) == 4, f"应有 4 条 SkipRecord，实际 {len(SKIP_RECORDS)}"
    total_lines = sum(r.lines for r in SKIP_RECORDS)
    assert total_lines == 552, f"4 个文件合计应 552 行，实际 {total_lines}"
    _dbg("test", f"PASS: 4 条 SkipRecord，总行数={total_lines}")


def test_graph_from_heterograph_still_works():
    """
    验证 walpurgis.graph.graph_from_heterograph（CuGraphStorage 的替代）仍可导入。

    对应 test_from_dgl_heterograph.py 中保留的非 storage 部分。
    """
    _dbg("test", "验证 graph_from_heterograph 可导入")
    from walpurgis.graph.convert import graph_from_heterograph
    assert callable(graph_from_heterograph)
    _dbg("test", f"PASS: graph_from_heterograph={graph_from_heterograph.__module__}")


def test_storage_removal_spec_self_check():
    """验证 StorageRemovalSpec 自检通过。"""
    from walpurgis.core.dgl_dask_removal import STORAGE_REMOVAL_SPEC
    assert STORAGE_REMOVAL_SPEC.self_check()
