"""Cascade utils.

migrate 529546a: Finish CUDA 12.9 migration and use branch-25.06 workflows
  上游 python/cugraph-dgl/cugraph_dgl/utils/__init__.py 新增5个图节点/边列名常量，
  将原来散落在 cugraph.gnn.dgl_extensions 的 src_n/dst_n 等符号迁移至本包内，
  消除对已废弃 cugraph.gnn 上游路径的依赖。

  上游同时将 cugraph_conversion_utils.py / cugraph_storage_utils.py 的 import 路径
  从 cugraph.gnn.dgl_extensions.* 改为 cugraph_dgl.utils，此为配套迁移。

  鲁迅: 以前这些常量住在 cugraph.gnn 的老宅里，现在搬到自己门户了。
        一个字符串，不在这里，便在那里，终究要有个落脚处。

  CI/workflow/conda 文件（13个）→ SKIP（纯 cuda 12.9→12.8 版本号替换）
"""

import os

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 图边/节点列名常量 ────────────────────────────────────────────────────────
# migrate 529546a 新增：从 cugraph.gnn.dgl_extensions 迁移至本包，消除上游废弃路径依赖。
# 上游原版只有裸字符串赋值；Walpurgis 改写：
#   1. 加入 __all__ 显式声明，防止 from utils import * 时意外泄露内部符号
#   2. 加入类型注解（Final[str]），文档化"禁止修改"语义
#   3. 加入 WALPURGIS_DEBUG 断点，采样管线调试时可追踪常量解析路径
#   4. 新增 _GRAPH_COLUMN_NAMES 元组，供需要枚举全部列名的代码使用（上游无）
#   5. 新增 assert_valid_column_name() 守卫，防止非法列名污染 DataFrame（上游无）

try:
    from typing import Final
except ImportError:
    Final = str  # type: ignore[assignment]

#: 源节点列名（SRC）
src_n: Final[str] = "_SRC_"
#: 目标节点列名（DST）
dst_n: Final[str] = "_DST_"
#: 边 ID 列名（EDGE_ID）
eid_n: Final[str] = "_EDGE_ID_"
#: 边/节点类型列名（TYPE）
type_n: Final[str] = "_TYPE_"
#: 节点全局 ID 列名（VERTEX）
vid_n: Final[str] = "_VERTEX_"

# 改写新增：全部列名元组，可迭代枚举，上游无此结构
_GRAPH_COLUMN_NAMES: Final[tuple] = (src_n, dst_n, eid_n, type_n, vid_n)

__all__ = [
    "src_n",
    "dst_n",
    "eid_n",
    "type_n",
    "vid_n",
    "_GRAPH_COLUMN_NAMES",
    "assert_valid_column_name",
]

if _WALPURGIS_DEBUG:
    print(
        f"[WALPURGIS_DEBUG 529546a utils/__init__] "
        f"图列名常量已加载: {_GRAPH_COLUMN_NAMES}"
    )


def assert_valid_column_name(col: str) -> None:
    """
    改写新增：断言 col 是已知图列名之一，防止拼写错误污染 DataFrame。

    上游 529546a 仅定义裸字符串常量，无任何调用侧保护。
    Walpurgis 改写点之一：对\"应该传常量却传了字符串字面量\"的场景做显式守卫。

    Parameters
    ----------
    col : str
        待验证的列名，必须是 src_n/dst_n/eid_n/type_n/vid_n 之一。

    Raises
    ------
    ValueError
        若 col 不在已知列名集合中。
    """
    if col not in _GRAPH_COLUMN_NAMES:
        raise ValueError(
            f"[Walpurgis] 未知图列名 {col!r}。"
            f"合法列名: {_GRAPH_COLUMN_NAMES}"
        )
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG 529546a assert_valid_column_name] {col!r} ✓")
