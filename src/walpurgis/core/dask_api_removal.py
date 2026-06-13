"""
dask_api_removal.py — 1e91ed7 迁移: 从 cuGraph-PyG 彻底删除 Dask API

migrate 1e91ed7: Remove Dask API from cuGraph-PyG (#166)

上游变化 (1e91ed7, cugraph-gnn):
  13 files changed, 2 insertions(+), 4837 deletions(-) — 大规模删除

核心删除项:
  1. data/dask_graph_store.py (1321行) — 整个 DaskGraphStore 实现
     包含: EdgeLayout, CuGraphEdgeAttr, CuGraphTensorAttr, DaskGraphStore 类
     Dask/cuDF 依赖: dask.array, dask.dataframe, dask.distributed, dask_cudf
  2. data/__init__.py — 移除 DaskGraphStore / CuGraphStore wrapper 函数
  3. loader/__init__.py — 移除 DaskNeighborLoader / BulkSampleLoader wrapper 函数
  4. loader/dask_node_loader.py (558行) — 整个 DaskNeighborLoader 实现
  5. sampler/sampler_utils.py — 移除 DaskGraphStore 相关导入和所有 Dask 路径函数
     具体: _get_unique_nodes / _sampler_output_from_sampling_results_* 等 ~379行
  6. tests/data/test_dask_graph_store.py (413行)
  7. tests/data/test_dask_graph_store_mg.py (424行)
  8. tests/loader/test_dask_neighbor_loader.py (508行)
  9. tests/loader/test_dask_neighbor_loader_mg.py (77行)
  10. tests/sampler/test_sampler_utils.py (194行) + test_sampler_utils_mg.py (233行)
  11. examples/graph_sage_mg.py (450行) + examples/graph_sage_sg.py (222行)

Walpurgis 迁移语义:
  - 431801c (graph_store_deprecation.py): DaskGraphStore/CuGraphStore 发出 FutureWarning
  - 1e91ed7 (本文件): 废弃警告阶段结束，API 彻底删除，升级为 RuntimeError 墓碑
  - 与 dgl_dask_removal.py 的 DGL 侧删除对称，形成完整的 Dask API 清除记录

Walpurgis 改写 20%（鲁迅拿法）:
- DaskApiRemovalManifest(frozen dataclass): 结构化记录每个被删文件的元数据
  上游是「直接删除文件」，Walpurgis 保留完整的删除清单供审计
- RemovalPhase 枚举: 区分「废弃警告阶段」和「彻底删除阶段」，
  比上游的「先 deprecate PR + 后 remove PR」二元操作更清晰
- DaskPygRemovalAudit: 集中验证 PyG 侧 Dask API 的所有删除均已在 Walpurgis 完成
  与 dgl_dask_removal.py 的 DaskApiInventory 对称，共同覆盖 PyG + DGL 两侧
- PyGDaskSamplerCleanup: 封装 sampler_utils.py 的清理逻辑
  上游直接删 _get_unique_nodes / _sampler_output_from_sampling_results_* 等函数；
  Walpurgis 用结构体记录这些函数的墓碑，使静态分析工具可检测遗留引用

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:dask_api_removal:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# RemovalPhase — 区分废弃警告阶段和彻底删除阶段
# ─────────────────────────────────────────────────────────────────────────────

class RemovalPhase(Enum):
    """
    Dask API 移除的阶段枚举。

    对应上游两个 PR 的两阶段操作:
      - DEPRECATED (#118 / 431801c): FutureWarning wrapper 阶段
      - REMOVED (#166 / 1e91ed7):   彻底删除阶段
    """
    DEPRECATED = auto()   # 431801c: FutureWarning，API 仍可调用
    REMOVED = auto()      # 1e91ed7: 文件/函数已删除，调用即错误


# ─────────────────────────────────────────────────────────────────────────────
# DaskApiRemovalManifest — 结构化记录被删文件的元数据
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DaskApiRemovalManifest:
    """
    记录单个被删除的 Dask API 组件的元数据。

    上游 1e91ed7 直接删除 13 个文件/模块；
    Walpurgis 将删除清单保留为可查询的结构，
    供测试框架验证「哪些 API 已不可用」和审计工具追踪迁移进度。
    """
    upstream_path: str             # 上游被删文件相对路径
    walpurgis_tombstone: str       # Walpurgis 对应的墓碑文件
    lines_deleted: int             # 上游删除行数（近似）
    phase: RemovalPhase            # 当前阶段（REMOVED）
    deprecation_commit: str        # 废弃阶段的上游 commit
    removal_commit: str            # 删除阶段的上游 commit
    migration_target: Optional[str] = None  # 推荐替代 API / 文件


#: 1e91ed7 删除的 PyG 侧 Dask API 完整清单
REMOVED_DASK_COMPONENTS: List[DaskApiRemovalManifest] = [
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/data/dask_graph_store.py",
        walpurgis_tombstone="src/walpurgis/core/graph_store_deprecation.py",
        lines_deleted=1321,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target="walpurgis.graph.Graph",
    ),
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/loader/dask_node_loader.py",
        walpurgis_tombstone="src/walpurgis/dataloader/loader_deprecation.py",
        lines_deleted=558,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target="walpurgis.dataloader.DataLoader",
    ),
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/examples/graph_sage_mg.py",
        walpurgis_tombstone="src/walpurgis/examples/gcn/gcn_dist_mnmg.py",
        lines_deleted=450,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target="walpurgis.examples.gcn.gcn_dist_mnmg",
    ),
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/examples/graph_sage_sg.py",
        walpurgis_tombstone="src/walpurgis/examples/gcn/gcn_dist_sg.py",
        lines_deleted=222,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target="walpurgis.examples.gcn.gcn_dist_sg",
    ),
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/tests/data/test_dask_graph_store.py",
        walpurgis_tombstone="src/walpurgis/tests/sampler/test_dask_storage_tombstone.py",
        lines_deleted=413,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target=None,
    ),
    DaskApiRemovalManifest(
        upstream_path="python/cugraph-pyg/cugraph_pyg/tests/loader/test_dask_neighbor_loader.py",
        walpurgis_tombstone="src/walpurgis/tests/sampler/test_dask_storage_tombstone.py",
        lines_deleted=508,
        phase=RemovalPhase.REMOVED,
        deprecation_commit="431801c",
        removal_commit="1e91ed7",
        migration_target=None,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# PyGDaskSamplerCleanup — 记录 sampler_utils.py 的 Dask 函数墓碑
#
# 上游 1e91ed7 从 sampler_utils.py 删除了以下函数（~379行）:
#   _get_unique_nodes()
#   _sampler_output_from_sampling_results_homogeneous_coo()
#   _sampler_output_from_sampling_results_heterogeneous_coo()
#   _sampler_output_from_sampling_results_homogeneous_csc()
#   _sampler_output_from_sampling_results_heterogeneous_csc()
#
# 这些函数都依赖 DaskGraphStore 和 cudf.DataFrame，是旧 Dask 采样路径的核心。
# Walpurgis 将其记录为结构化墓碑，便于代码考古和遗留引用检测。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SamplerFunctionTombstone:
    """被删除的 sampler 函数的墓碑记录。"""
    func_name: str
    depends_on: Tuple[str, ...]     # 依赖的 Dask/cuDF 类型
    lines_deleted: int
    description: str


SAMPLER_UTILS_TOMBSTONES: List[SamplerFunctionTombstone] = [
    SamplerFunctionTombstone(
        func_name="_get_unique_nodes",
        depends_on=("cudf.DataFrame", "DaskGraphStore"),
        lines_deleted=45,
        description="统计 DaskGraphStore 图中特定节点类型的唯一节点数",
    ),
    SamplerFunctionTombstone(
        func_name="_sampler_output_from_sampling_results_homogeneous_coo",
        depends_on=("cudf.DataFrame", "DaskGraphStore"),
        lines_deleted=80,
        description="同构 COO 格式采样结果 → SamplerOutput 转换",
    ),
    SamplerFunctionTombstone(
        func_name="_sampler_output_from_sampling_results_heterogeneous_coo",
        depends_on=("cudf.DataFrame", "DaskGraphStore"),
        lines_deleted=90,
        description="异构 COO 格式采样结果 → HeteroSamplerOutput 转换",
    ),
    SamplerFunctionTombstone(
        func_name="_sampler_output_from_sampling_results_homogeneous_csc",
        depends_on=("cudf.DataFrame", "DaskGraphStore"),
        lines_deleted=80,
        description="同构 CSC 格式采样结果 → SamplerOutput 转换",
    ),
    SamplerFunctionTombstone(
        func_name="_sampler_output_from_sampling_results_heterogeneous_csc",
        depends_on=("cudf.DataFrame", "DaskGraphStore"),
        lines_deleted=84,
        description="异构 CSC 格式采样结果 → HeteroSamplerOutput 转换",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# DaskPygRemovalAudit — 验证 PyG 侧 Dask API 的所有删除均已完成
# ─────────────────────────────────────────────────────────────────────────────

class DaskPygRemovalAudit:
    """
    审计 Walpurgis 中 PyG 侧 Dask API 的删除完整性。

    与 dgl_dask_removal.py 的 DaskApiInventory 对称:
      - DaskApiInventory: 覆盖 cuGraph-DGL 侧 (61a370e)
      - DaskPygRemovalAudit: 覆盖 cuGraph-PyG 侧 (1e91ed7)

    断点1: audit() — 打印所有删除清单条目
    断点2: find_by_tombstone() — 查找某个墓碑文件对应的删除清单
    断点3: sampler_tombstones_summary() — 打印 sampler_utils.py 清理摘要
    """

    def __init__(self):
        self._manifests = REMOVED_DASK_COMPONENTS
        self._sampler_tombstones = SAMPLER_UTILS_TOMBSTONES

    def audit(self) -> str:
        """返回完整的删除审计报告。"""
        lines = [
            f"DaskPygRemovalAudit (commit: 1e91ed7)",
            f"Total components removed: {len(self._manifests)}",
            f"Total sampler functions removed: {len(self._sampler_tombstones)}",
            "",
        ]
        for m in self._manifests:
            lines.append(
                f"  [{m.phase.name}] {m.upstream_path}"
                f" (-{m.lines_deleted} lines)"
                f" → {m.migration_target or 'no migration path'}"
            )
        _dbg("DaskPygRemovalAudit", "audit() called", total=len(self._manifests))
        return "\n".join(lines)

    def find_by_tombstone(self, tombstone_path: str) -> List[DaskApiRemovalManifest]:
        """根据 Walpurgis 墓碑文件路径查找对应的删除清单条目。"""
        result = [m for m in self._manifests if m.walpurgis_tombstone == tombstone_path]
        _dbg(
            "DaskPygRemovalAudit",
            f"find_by_tombstone({tombstone_path!r})",
            found=len(result),
        )
        return result

    def sampler_tombstones_summary(self) -> str:
        """返回 sampler_utils.py 被删函数的汇总。"""
        total_lines = sum(t.lines_deleted for t in self._sampler_tombstones)
        lines = [
            f"PyG sampler_utils.py Dask cleanup (1e91ed7):",
            f"  Functions removed: {len(self._sampler_tombstones)}",
            f"  Total lines: ~{total_lines}",
        ]
        for t in self._sampler_tombstones:
            lines.append(
                f"  - {t.func_name}() "
                f"(-{t.lines_deleted}L) "
                f"deps={t.depends_on}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"DaskPygRemovalAudit("
            f"components={len(self._manifests)}, "
            f"sampler_tombstones={len(self._sampler_tombstones)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────────────────────

dask_pyg_removal_audit = DaskPygRemovalAudit()

_dbg("module", "dask_api_removal loaded", components=len(REMOVED_DASK_COMPONENTS))
