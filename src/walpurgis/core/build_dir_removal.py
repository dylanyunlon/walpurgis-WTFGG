# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 87455cf
# 原标题: Remove Build Directory (#107)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「什么是路？就是从没有路的地方践踏出来的，从只有荆棘的地方开辟出来的。」
# —— 鲁迅《故乡》
#
# 87455cf 删除了被误提交到版本库的 python/cugraph-pyg/build/ 目录，
# 共 32 文件, 6699 行删除，无任何新增。
#
# build/ 目录包含 pip install -e . 时自动生成的旧版 wheel 构建产物，
# 里面的代码是 cugraph-pyg 历史上两代 API 的混合体：
#   - 第一代：CuGraphStore（单体，直接用 cuDF + Dask + cuPy，无 PyG Store 接口）
#     代表文件: build/lib/cugraph_pyg/data/cugraph_store.py (1215 行)
#   - 第一代采样器: build/lib/cugraph_pyg/sampler/cugraph_sampler.py (438 行)
#     用 cuDF DataFrame 作为采样结果容器，与新版 torch.Tensor 路径不兼容
#   - 第一代 Loader: build/lib/cugraph_pyg/loader/cugraph_node_loader.py (534 行)
#     直接读 parquet 文件，写死目录结构
#
# 与现行 API（GraphStore + FeatureStore + SampleReader）的对比：
#   旧 API (build/):          新 API (src/):
#   CuGraphStore              GraphStore + TensorDictFeatureStore
#   cudf.DataFrame samples    torch.Tensor samples
#   DistSampleWriter(dir=)    writer=None (in-memory)
#   EdgeLayout Enum (local)   PyG EdgeAttr.layout
#   CuGraphSAGEConv (ops)     torch_geometric.nn.SAGEConv
#
# Walpurgis 20% 改写要点：
#   1. LegacyApiTombstone 类 — 封装已删除的旧 API 名称列表，
#      尝试导入时给出明确的迁移映射（旧符号 → 新符号 + 新模块）
#   2. BuildArtifactGuard 类 — 检测 build/ 目录是否仍存在于工作目录，
#      若存在则警告（防止 IDE 导入时拾取旧版本而非 src/ 版本）
#   3. EdgeLayoutLegacy 枚举 — 与旧版 CuGraphEdgeAttr.layout 等价，
#      保留作为迁移指引，新代码用 PyG 原生 EdgeAttr
#   4. api_migration_table() — 返回完整的旧→新 API 映射字典，
#      供 CI 迁移检查脚本使用

from __future__ import annotations

import enum
import os as _os
import sys as _sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 遗留边布局枚举：对应旧版 build/lib/cugraph_pyg/data/cugraph_store.py 中的 EdgeLayout
# ---------------------------------------------------------------------------

class EdgeLayoutLegacy(enum.Enum):
    """
    旧版 CuGraphStore 中的 EdgeLayout 枚举（已在 87455cf 随 build/ 目录一起删除）。

    新代码应使用 PyG 原生的 torch_geometric.data.storage.EdgeAttr 或直接用字符串
    "coo" / "csc" / "csr" 作为 GraphStore key 的 layout 参数。

    保留此枚举仅作迁移参考，不应在新代码中使用。
    """

    COO = "coo"
    CSC = "csc"
    CSR = "csr"

    @classmethod
    def from_legacy_str(cls, s: str) -> "EdgeLayoutLegacy":
        """
        从旧版字符串构造枚举，用于迁移诊断脚本。
        """
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(
                f"[Walpurgis:EdgeLayoutLegacy] 未知 layout '{s}'。"
                f"旧版可用: {[e.value for e in cls]}。"
                f"新版直接使用字符串 'coo'/'csc'/'csr' 即可。"
            )

    def to_new_api_str(self) -> str:
        """
        返回新 API 中等价的 layout 字符串（与枚举值相同，方便迁移）。
        """
        return self.value


# ---------------------------------------------------------------------------
# 旧 API 迁移映射表
# ---------------------------------------------------------------------------

def api_migration_table() -> Dict[str, Dict[str, str]]:
    """
    返回 build/ 目录旧 API → 现行 API 的完整迁移映射。

    格式: { "旧符号路径": { "new_module": "...", "new_symbol": "...", "note": "..." } }

    供 CI 迁移检查脚本或开发者查阅使用。
    """
    table = {
        # 数据层
        "cugraph_pyg.data.CuGraphStore": {
            "new_module": "cugraph_pyg.data",
            "new_symbol": "GraphStore + TensorDictFeatureStore",
            "note": "CuGraphStore 是单体设计，新版拆分为符合 PyG Store 接口的两个类",
        },
        "cugraph_pyg.data.CuGraphEdgeAttr": {
            "new_module": "torch_geometric.data.storage",
            "new_symbol": "EdgeAttr",
            "note": "直接使用 PyG 原生 EdgeAttr，不再需要 cuGraph 自定义类",
        },
        # 采样器层
        "cugraph_pyg.sampler.CuGraphSampler": {
            "new_module": "cugraph_pyg.sampler",
            "new_symbol": "BaseSampler",
            "note": "旧版用 cuDF DataFrame 传递采样结果，新版用 torch.Tensor",
        },
        "cugraph_pyg.sampler.cugraph_sampler._get_unique_nodes": {
            "new_module": "cugraph_pyg.sampler.sampler",
            "new_symbol": "HeterogeneousSampleReader.__decode_coo (内部)",
            "note": "节点唯一化逻辑已内化到 SampleReader",
        },
        # Loader 层
        "cugraph_pyg.loader.CuGraphNodeLoader": {
            "new_module": "cugraph_pyg.loader",
            "new_symbol": "NeighborLoader",
            "note": "旧版读磁盘 parquet，新版支持 in-memory (writer=None)",
        },
        # NN 层（另见 d38b832）
        "cugraph_pyg.nn.conv.SAGEConv": {
            "new_module": "torch_geometric.nn",
            "new_symbol": "SAGEConv",
            "note": "旧版依赖 pylibcugraphops，d38b832 已删除",
        },
        "cugraph_pyg.nn.conv.GATConv": {
            "new_module": "torch_geometric.nn",
            "new_symbol": "GATConv",
            "note": "旧版依赖 pylibcugraphops，d38b832 已删除",
        },
    }

    _dbg("api_migration_table", f"entries={len(table)}")
    return table


# ---------------------------------------------------------------------------
# LegacyApiTombstone：尝试访问已删除 API 时的哨兵
# ---------------------------------------------------------------------------

class LegacyApiTombstone:
    """
    当代码尝试使用已随 build/ 目录一起消失的旧版 API 时，
    提供明确的错误信息和迁移指引。

    用法：
        # 在旧 __init__.py 的兼容层中：
        CuGraphStore = LegacyApiTombstone("cugraph_pyg.data.CuGraphStore")
        # 这样 `from cugraph_pyg.data import CuGraphStore` 能成功 import，
        # 但实例化时会抛出 LegacyApiError 而不是 ImportError
    """

    def __init__(self, old_path: str) -> None:
        self._old_path = old_path
        table = api_migration_table()
        self._hint = table.get(old_path, {})

    def __call__(self, *args, **kwargs):
        new_sym = self._hint.get("new_symbol", "（见迁移文档）")
        new_mod = self._hint.get("new_module", "")
        note = self._hint.get("note", "")
        raise LegacyApiError(
            f"[Walpurgis] {self._old_path} 已在 87455cf 随 build/ 目录移除。\n"
            f"迁移至: {new_mod}.{new_sym}\n"
            + (f"说明: {note}\n" if note else "")
            + f"完整迁移表: walpurgis.core.build_dir_removal.api_migration_table()"
        )

    def __repr__(self) -> str:
        return f"LegacyApiTombstone({self._old_path!r})"


class LegacyApiError(RuntimeError):
    """尝试使用已删除的 build/ 目录 API 时抛出。"""
    pass


# ---------------------------------------------------------------------------
# BuildArtifactGuard：检测工作目录中是否残留 build/ 产物
# ---------------------------------------------------------------------------

class BuildArtifactGuard:
    """
    检查项目目录中是否仍然存在 build/lib/cugraph_pyg/（87455cf 已删除的目录）。

    如果残留，Python import 系统可能会优先加载旧版本而非 src/ 中的新版本，
    造成难以诊断的 API 不匹配问题。

    用法（在 conftest.py 或 pytest 钩子中）：
        BuildArtifactGuard.check_and_warn(project_root="/path/to/cugraph-pyg")
    """

    SENTINEL_PATHS = [
        "build/lib/cugraph_pyg/data/cugraph_store.py",
        "build/lib/cugraph_pyg/sampler/cugraph_sampler.py",
        "build/lib/cugraph_pyg/loader/cugraph_node_loader.py",
    ]

    @classmethod
    def check_and_warn(cls, project_root: str = ".") -> List[str]:
        """
        扫描 project_root 下是否存在旧版 build/ 产物。

        Parameters
        ----------
        project_root : str
            项目根目录，默认当前目录

        Returns
        -------
        List[str]
            发现的旧版文件路径列表（空列表表示干净）
        """
        found = []
        for rel_path in cls.SENTINEL_PATHS:
            full_path = _os.path.join(project_root, rel_path)
            if _os.path.exists(full_path):
                found.append(full_path)
                _dbg("BuildArtifactGuard", f"发现旧版产物: {full_path}")

        if found:
            warnings.warn(
                f"[Walpurgis:BuildArtifactGuard] 发现 {len(found)} 个旧版 build/ 产物:\n"
                + "\n".join(f"  {p}" for p in found)
                + "\n这些文件应在 87455cf 中已删除。"
                + "\n运行 `rm -rf build/` 清理，防止 import 拾取旧版本。",
                UserWarning,
                stacklevel=2,
            )
        else:
            _dbg("BuildArtifactGuard", "工作目录干净，无旧版 build/ 产物")

        return found

    @classmethod
    def assert_clean(cls, project_root: str = ".") -> None:
        """
        断言无旧版 build/ 产物，CI 中使用。
        """
        found = cls.check_and_warn(project_root)
        if found:
            raise AssertionError(
                f"[Walpurgis:BuildArtifactGuard] CI 失败: 发现 {len(found)} 个旧版文件。\n"
                + "\n".join(f"  {p}" for p in found)
            )


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ["WALPURGIS_DEBUG"] = "1"
    print("=== 自测 build_dir_removal.py (migrate 87455cf) ===\n")

    # --- 测试 1: EdgeLayoutLegacy ---
    layout = EdgeLayoutLegacy.from_legacy_str("coo")
    assert layout == EdgeLayoutLegacy.COO
    assert layout.to_new_api_str() == "coo"
    print(f"[OK] 测试1: EdgeLayoutLegacy.from_legacy_str('coo') → {layout}")

    # --- 测试 2: 未知 layout ---
    try:
        EdgeLayoutLegacy.from_legacy_str("dense")
        assert False, "应该 ValueError"
    except ValueError as e:
        assert "dense" in str(e)
        print("[OK] 测试2: 未知 layout ValueError")

    # --- 测试 3: api_migration_table ---
    table = api_migration_table()
    assert "cugraph_pyg.data.CuGraphStore" in table
    assert "cugraph_pyg.loader.CuGraphNodeLoader" in table
    print(f"[OK] 测试3: api_migration_table 包含 {len(table)} 条映射")

    # --- 测试 4: LegacyApiTombstone ---
    CuGraphStore = LegacyApiTombstone("cugraph_pyg.data.CuGraphStore")
    try:
        CuGraphStore(some_arg=1)
        assert False, "应该 LegacyApiError"
    except LegacyApiError as e:
        assert "87455cf" in str(e)
        assert "GraphStore" in str(e)
        print("[OK] 测试4: LegacyApiTombstone 抛出 LegacyApiError 并含迁移提示")

    # --- 测试 5: BuildArtifactGuard 在干净目录不报警 ---
    found = BuildArtifactGuard.check_and_warn("/nonexistent_path_xyz")
    assert found == []
    print("[OK] 测试5: BuildArtifactGuard 干净目录无报警")

    # --- 测试 6: LegacyApiTombstone repr ---
    assert "CuGraphStore" in repr(CuGraphStore)
    print(f"[OK] 测试6: LegacyApiTombstone repr → {repr(CuGraphStore)}")

    print("\n=== 全部自测通过 ===")
