# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 0e88280
# 原标题: Support PyG 2.6 in cuGraph-PyG (#114)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「横眉冷对千夫指，俯首甘为孺子牛。」—— 鲁迅《自嘲》
#
# 0e88280 的核心变化：PyG 2.6 不再接受部分指定的 Store 键。
#
# PyG 2.5 及之前允许：
#   feature_store["person", "feat"] = tensor         # 2-tuple key
#   graph_store[("n","e","n"), "coo"] = edge_index   # 2-tuple key，省略 is_sorted 和 size
#
# PyG 2.6 要求：
#   feature_store["person", "feat", None] = tensor   # 必须指定 group（None = 默认）
#   graph_store[("n","e","n"), "coo", False, (N, M)] = ei  # 必须指定 is_sorted 和 size
#
# 改动文件: 12 个 Python 文件，全部是 key 格式升级，无算法逻辑变化。
# - examples/gcn_dist_{mnmg,sg,snmg}.py — 训练 loop 中 loader 调用前的数据加载
# - examples/rgcn_link_class_{mnmg,sg,snmg}.py — 关系图链接预测示例
# - tests/data/test_feature_store.py, test_graph_store.py — 单元测试
# - tests/loader/test_neighbor_loader.py, test_neighbor_loader_mg.py — 加载器测试
# - pyproject.toml — PyG 版本约束从 >=2.5,<2.6 改为 >=2.6,<2.7
#
# Walpurgis 20% 改写要点：
#   1. StoreKeySpec 数据类 — 将 PyG Store 键的四个组成部分（type, attr, group, size）
#      封装为命名结构，替代裸 tuple 字面量，防止遗漏 None / size 而触发 PyG 2.6 错误
#   2. PyGVersionGuard 上下文管理器 — 检测已安装 PyG 版本是否 >= 2.6，
#      版本不匹配时给出明确的升级指引而非隐晦的 KeyError
#   3. EdgeIndexKey / FeatureKey 工厂函数 — 替代散落在各文件的 key 拼写，
#      统一保证 is_sorted=False 和 group=None 默认值
#   4. WALPURGIS_DEBUG=1 时打印每次 put_edge_index / feature_store 的完整键

from __future__ import annotations

import os as _os
import sys as _sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 数据类：Feature Store 键
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureStoreKey:
    """
    PyG FeatureStore 键的完整规范（PyG 2.6+ 格式）。

    PyG 2.5 (2-tuple):  feature_store["person", "feat"]
    PyG 2.6 (3-tuple):  feature_store["person", "feat", None]

    Fields
    ------
    node_type : str
        节点类型名称（如 "person"）或 canonical edge type tuple（如 ("n","e","n")）
    attr_name : str
        属性名称（如 "feat", "rel"）
    group : Optional[str]
        属性分组，PyG 2.6 新增要求，None 表示默认组
    """

    node_or_edge_type: object  # str 或 Tuple[str,str,str]
    attr_name: str
    group: Optional[str] = None

    def as_key(self) -> Tuple:
        """
        返回 PyG 2.6+ FeatureStore 键 tuple。

        Returns
        -------
        (node_or_edge_type, attr_name, group)
        """
        key = (self.node_or_edge_type, self.attr_name, self.group)
        _dbg("FeatureStoreKey.as_key", str(key))
        return key

    @classmethod
    def node(
        cls,
        node_type: str,
        attr: str,
        group: Optional[str] = None,
    ) -> "FeatureStoreKey":
        """节点特征键工厂方法。"""
        return cls(node_or_edge_type=node_type, attr_name=attr, group=group)

    @classmethod
    def edge(
        cls,
        can_etype: Tuple[str, str, str],
        attr: str,
        group: Optional[str] = None,
    ) -> "FeatureStoreKey":
        """边特征键工厂方法。"""
        return cls(node_or_edge_type=can_etype, attr_name=attr, group=group)


# ---------------------------------------------------------------------------
# 数据类：Edge Index Store 键
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeIndexKey:
    """
    PyG GraphStore 边索引键的完整规范（PyG 2.6+ 格式）。

    PyG 2.5 (2-tuple):  graph_store[("n","e","n"), "coo"]
    PyG 2.6 (4-tuple):  graph_store[("n","e","n"), "coo", False, (N, M)]

    Fields
    ------
    can_etype : Tuple[str, str, str]
        Canonical edge type，如 ("paper", "cites", "paper")
    layout : str
        存储格式，"coo" / "csr" / "csc"
    is_sorted : bool
        是否已排序（大多数情况 False），PyG 2.6 新增要求
    size : Tuple[int, int]
        (num_src_nodes, num_dst_nodes)，PyG 2.6 新增要求
    """

    can_etype: Tuple[str, str, str]
    layout: str
    is_sorted: bool
    size: Tuple[int, int]

    def as_key(self) -> Tuple:
        """
        返回 PyG 2.6+ GraphStore 键 tuple。

        Returns
        -------
        (can_etype, layout, is_sorted, size)
        """
        key = (self.can_etype, self.layout, self.is_sorted, self.size)
        _dbg("EdgeIndexKey.as_key", str(key))
        return key

    @classmethod
    def coo(
        cls,
        can_etype: Tuple[str, str, str],
        num_src: int,
        num_dst: int,
        is_sorted: bool = False,
    ) -> "EdgeIndexKey":
        """COO 格式边索引键工厂方法。"""
        k = cls(
            can_etype=can_etype,
            layout="coo",
            is_sorted=is_sorted,
            size=(num_src, num_dst),
        )
        _dbg(
            "EdgeIndexKey.coo",
            f"etype={can_etype}  num_src={num_src}  num_dst={num_dst}",
        )
        return k

    @classmethod
    def homogeneous_coo(
        cls,
        node_type: str,
        rel: str,
        num_nodes: int,
        is_sorted: bool = False,
    ) -> "EdgeIndexKey":
        """同构图 COO 键工厂方法（src == dst 同一类型）。"""
        return cls.coo(
            (node_type, rel, node_type), num_nodes, num_nodes, is_sorted
        )


# ---------------------------------------------------------------------------
# 版本检查工具
# ---------------------------------------------------------------------------

def get_pyg_version() -> Optional[Tuple[int, int]]:
    """
    获取已安装 torch_geometric 的版本号 (major, minor)。

    Returns
    -------
    (major, minor) 或 None（未安装时）
    """
    try:
        import torch_geometric
        parts = torch_geometric.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def assert_pyg_26_compatible() -> None:
    """
    断言 PyG >= 2.6，否则抛出 RuntimeError 并给出升级指引。

    0e88280 要求 PyG >=2.6,<2.7（后续 PR 进一步放宽上限）。
    """
    ver = get_pyg_version()
    if ver is None:
        import warnings
        warnings.warn(
            "[Walpurgis] torch_geometric 未安装，无法验证 PyG >= 2.6 要求。",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    major, minor = ver
    _dbg("assert_pyg_26_compatible", f"detected PyG {major}.{minor}")

    if (major, minor) < (2, 6):
        raise RuntimeError(
            f"[Walpurgis] 检测到 PyG {major}.{minor}，但需要 >= 2.6。\n"
            f"0e88280 升级了所有 Store 键格式以兼容 PyG 2.6+ 的完整键要求。\n"
            f"请升级: pip install 'torch_geometric>=2.6,<2.7'"
        )


@contextmanager
def pyg_26_store_context(label: str = ""):
    """
    上下文管理器：在进入时断言 PyG >= 2.6，异常时添加迁移提示。

    用法：
        with pyg_26_store_context("图数据加载"):
            graph_store[("n","e","n"), "coo", False, (N, M)] = ei
    """
    assert_pyg_26_compatible()
    _dbg("pyg_26_store_context", f"enter [{label}]")
    try:
        yield
    except TypeError as e:
        raise TypeError(
            f"[Walpurgis:{label}] Store 键格式错误，可能使用了 PyG 2.5 旧格式。\n"
            f"PyG 2.6+ 要求 feature_store[type, attr, None] 和 "
            f"graph_store[(src,rel,dst), 'coo', False, (N,M)]。\n"
            f"原始错误: {e}"
        ) from e


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ["WALPURGIS_DEBUG"] = "1"
    print("=== 自测 pyg26_store_compat.py (migrate 0e88280) ===\n")

    # --- 测试 1: FeatureStoreKey.node ---
    k = FeatureStoreKey.node("person", "feat")
    assert k.as_key() == ("person", "feat", None)
    print(f"[OK] 测试1: FeatureStoreKey.node → {k.as_key()}")

    # --- 测试 2: FeatureStoreKey.edge ---
    k2 = FeatureStoreKey.edge(("n", "e", "n"), "rel")
    assert k2.as_key() == (("n", "e", "n"), "rel", None)
    print(f"[OK] 测试2: FeatureStoreKey.edge → {k2.as_key()}")

    # --- 测试 3: EdgeIndexKey.coo ---
    ek = EdgeIndexKey.coo(("paper", "cites", "paper"), 100, 100)
    assert ek.as_key() == (("paper", "cites", "paper"), "coo", False, (100, 100))
    print(f"[OK] 测试3: EdgeIndexKey.coo → {ek.as_key()}")

    # --- 测试 4: EdgeIndexKey.homogeneous_coo ---
    ek2 = EdgeIndexKey.homogeneous_coo("person", "knows", 34)
    assert ek2.size == (34, 34)
    assert ek2.can_etype == ("person", "knows", "person")
    print(f"[OK] 测试4: EdgeIndexKey.homogeneous_coo → {ek2.as_key()}")

    # --- 测试 5: get_pyg_version ---
    ver = get_pyg_version()
    if ver is None:
        print("[SKIP] 测试5: torch_geometric 未安装，跳过版本检测")
    else:
        print(f"[OK] 测试5: PyG 版本 = {ver[0]}.{ver[1]}")

    # --- 测试 6: 异构图边键（author→paper 不同 src/dst size）---
    ek3 = EdgeIndexKey.coo(("author", "writes", "paper"), num_src=400, num_dst=100)
    assert ek3.size == (400, 100)
    print(f"[OK] 测试6: 异构图边键 → {ek3.as_key()}")

    print("\n=== 全部自测通过 ===")
