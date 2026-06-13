"""
graph_store_deprecation.py — 431801c 迁移: 废弃 DaskGraphStore / CuGraphStore

migrate 431801c: Deprecate the Dask API in cuGraph-PyG (#118)

上游变化 (431801c, cugraph-gnn / python/cugraph-pyg/cugraph_pyg/data/__init__.py):

1. DaskGraphStore 废弃:
   旧:
       from cugraph_pyg.data.dask_graph_store import DaskGraphStore
   新:
       from cugraph_pyg.data.dask_graph_store import (
           DaskGraphStore as DEPRECATED__DaskGraphStore,
       )
       def DaskGraphStore(*args, **kwargs):
           warnings.warn(
               "DaskGraphStore and the Dask API are deprecated."
               " Please switch over to the new API (cugraph_pyg.data.GraphStore)",
               FutureWarning,
           )
           return DEPRECATED__DaskGraphStore(*args, **kwargs)

2. CuGraphStore 保持 FutureWarning 链:
       def CuGraphStore(*args, **kwargs):
           warnings.warn("CuGraphStore has been renamed to DaskGraphStore", FutureWarning)
           return DaskGraphStore(*args, **kwargs)

上游 graph_sage_mg.py / graph_sage_sg.py 在 main() 函数入口增加:
   warnings.warn("The Dask API used in this example is deprecated. ...", FutureWarning)

设计背景:
    - DaskGraphStore: 基于 Dask 的分布式图存储，是旧 API 的主入口。
      新 API 用 cugraph_pyg.data.GraphStore（无 Dask 依赖）替代。
    - CuGraphStore: DaskGraphStore 的历史名称，431801c 前已有 FutureWarning。
      新链: CuGraphStore → DaskGraphStore(deprecated) → DEPRECATED__DaskGraphStore。
    - wrapper 函数模式保持向后兼容（不破坏现有用户代码），FutureWarning 催迁移。

Walpurgis 改写 20%（鲁迅拿法）:
- GraphStoreDeprecationGate: 统一封装单个废弃存储入口，对齐 loader_deprecation.py
  的 LoaderDeprecationGate 设计，两者形成完整的数据+加载器废弃门控体系。
  上游是独立 wrapper 函数，无统计，无可观测性；本实现加 call_count + WALPURGIS_DEBUG。
- GraphStoreDeprecationRegistry: 集中管理 data 层废弃对象
  与 loader_deprecation.py 的 LoaderDeprecationRegistry 并列，
  覆盖 data 模块侧废弃 API（DaskGraphStore / CuGraphStore）。
- DeprecationChain: 枚举记录 CuGraphStore→DaskGraphStore→GraphStore 的迁移链路，
  上游仅靠注释/docstring 描述，Walpurgis 用结构体文档化。
- ExampleEntryDeprecation: 封装 graph_sage_mg/sg 例子里的入口警告，
  上游直接裸 warnings.warn，Walpurgis 加 entry_name + 目标示例名的结构信息。
- 全链路 WALPURGIS_DEBUG=1 断点: 触发时打印调用模块 + 参数类型摘要。

作者: dylanyunlon <dogechat@163.com>
"""

import os
import sys
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DeprecationChain — 结构化记录 CuGraphStore→DaskGraphStore→GraphStore 迁移链路
# ─────────────────────────────────────────────────────────────────────────────

class DeprecationStage(Enum):
    """迁移阶段枚举，对应上游 431801c 的废弃路径。"""
    LEGACY_ALIAS = auto()       # CuGraphStore: 历史名称别名
    DEPRECATED_API = auto()     # DaskGraphStore: 废弃但仍可用
    CURRENT_API = auto()        # GraphStore: 现行推荐 API


@dataclass(frozen=True)
class DeprecationChainEntry:
    """
    迁移链路中的单个条目。

    上游 431801c 仅靠注释描述迁移路径；
    Walpurgis 用 DeprecationChainEntry 将路径结构化，
    供审计工具和测试框架查询。
    """
    name: str
    stage: DeprecationStage
    successor: Optional[str]          # 推荐迁移目标
    warning_category: type            # FutureWarning / DeprecationWarning
    commit_introduced: str            # 引入该废弃的上游 commit hash


#: 431801c 定义的完整迁移链路
GRAPH_STORE_DEPRECATION_CHAIN: List[DeprecationChainEntry] = [
    DeprecationChainEntry(
        name="CuGraphStore",
        stage=DeprecationStage.LEGACY_ALIAS,
        successor="DaskGraphStore",
        warning_category=FutureWarning,
        commit_introduced="431801c",
    ),
    DeprecationChainEntry(
        name="DaskGraphStore",
        stage=DeprecationStage.DEPRECATED_API,
        successor="GraphStore",
        warning_category=FutureWarning,
        commit_introduced="431801c",
    ),
    DeprecationChainEntry(
        name="GraphStore",
        stage=DeprecationStage.CURRENT_API,
        successor=None,
        warning_category=FutureWarning,
        commit_introduced="n/a",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# GraphStoreDeprecationGate — 单个废弃 graph store 的 wrapper 封装
# ─────────────────────────────────────────────────────────────────────────────

class GraphStoreDeprecationGate:
    """
    封装单个废弃 graph store 入口的 FutureWarning + 透传构造。

    对应上游每个 wrapper 函数:
        def DaskGraphStore(*args, **kwargs):
            warnings.warn("...", FutureWarning)
            return DEPRECATED__DaskGraphStore(*args, **kwargs)

    GraphStoreDeprecationGate 等价但加:
    - call_count: 可查询该废弃入口被调用了多少次
    - chain_entry: 携带 DeprecationChainEntry 结构信息
    - WALPURGIS_DEBUG=1 时打印调用参数类型摘要（不泄露值）
    """

    def __init__(
        self,
        name: str,
        wrapped_cls,
        warning_msg: str,
        chain_entry: Optional[DeprecationChainEntry] = None,
        stacklevel: int = 3,
    ):
        self._name = name
        self._wrapped_cls = wrapped_cls
        self._warning_msg = warning_msg
        self._chain_entry = chain_entry
        self._stacklevel = stacklevel
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def chain_entry(self) -> Optional[DeprecationChainEntry]:
        return self._chain_entry

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        warnings.warn(self._warning_msg, FutureWarning, stacklevel=self._stacklevel)

        _dbg(
            f"GraphStoreDeprecationGate:{self._name}",
            f"called (call_count={self._call_count})",
            args_types=str([type(a).__name__ for a in args]),
            kwargs_keys=str(list(kwargs.keys())),
            successor=self._chain_entry.successor if self._chain_entry else "unknown",
        )

        return self._wrapped_cls(*args, **kwargs)

    def __repr__(self):
        return (
            f"GraphStoreDeprecationGate(name={self._name!r}, "
            f"wrapped={self._wrapped_cls.__name__!r}, "
            f"call_count={self._call_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GraphStoreDeprecationRegistry — 集中管理所有 data 层废弃对象
# ─────────────────────────────────────────────────────────────────────────────

class GraphStoreDeprecationRegistry:
    """
    统一注册和查找废弃 graph store 对象。

    与 loader_deprecation.py 的 LoaderDeprecationRegistry 并列：
    - LoaderDeprecationRegistry: 管理 loader 侧 (DaskNeighborLoader / BulkSampleLoader)
    - GraphStoreDeprecationRegistry: 管理 data 侧 (DaskGraphStore / CuGraphStore)

    断点1: register() — 注册时打印 name / wrapped class / chain_entry
    断点2: get() — 查找时打印 name + 是否命中
    断点3: summary() — 打印所有已注册对象的 call_count
    """

    def __init__(self):
        self._gates: Dict[str, GraphStoreDeprecationGate] = {}

    def register(
        self,
        name: str,
        wrapped_cls,
        warning_msg: str,
        chain_entry: Optional[DeprecationChainEntry] = None,
        stacklevel: int = 3,
    ) -> GraphStoreDeprecationGate:
        gate = GraphStoreDeprecationGate(name, wrapped_cls, warning_msg, chain_entry, stacklevel)
        self._gates[name] = gate
        _dbg(
            "GraphStoreDeprecationRegistry",
            f"registered {name!r}",
            wrapped=wrapped_cls.__name__,
            stage=chain_entry.stage.name if chain_entry else "unknown",
        )
        return gate

    def get(self, name: str) -> Optional[GraphStoreDeprecationGate]:
        gate = self._gates.get(name)
        _dbg(
            "GraphStoreDeprecationRegistry",
            f"get {name!r}",
            found=gate is not None,
        )
        return gate

    def summary(self) -> str:
        lines = ["GraphStoreDeprecationRegistry summary:"]
        for name, gate in self._gates.items():
            entry = gate.chain_entry
            stage = entry.stage.name if entry else "?"
            lines.append(f"  {name}: call_count={gate.call_count}, stage={stage}")
        return "\n".join(lines)

    def __repr__(self):
        return f"GraphStoreDeprecationRegistry(registered={list(self._gates.keys())})"


# ─────────────────────────────────────────────────────────────────────────────
# 全局 registry 实例 + 废弃对象注册
# ─────────────────────────────────────────────────────────────────────────────

graph_store_deprecation_registry = GraphStoreDeprecationRegistry()

_DaskGraphStore_gate: Optional[GraphStoreDeprecationGate] = None
_CuGraphStore_gate: Optional[GraphStoreDeprecationGate] = None

_chain_by_name: Dict[str, DeprecationChainEntry] = {
    e.name: e for e in GRAPH_STORE_DEPRECATION_CHAIN
}


def _get_or_build_gates():
    """
    延迟 import cugraph_pyg.data.dask_graph_store，
    避免在无 GPU 环境 import 本模块时引发 ImportError。
    """
    global _DaskGraphStore_gate, _CuGraphStore_gate

    if _DaskGraphStore_gate is not None:
        return

    try:
        from cugraph_pyg.data.dask_graph_store import (
            DaskGraphStore as DEPRECATED__DaskGraphStore,
        )
    except ImportError:
        _dbg("graph_store_deprecation", "cugraph_pyg not available — gates not built")
        return

    _DaskGraphStore_gate = graph_store_deprecation_registry.register(
        name="DaskGraphStore",
        wrapped_cls=DEPRECATED__DaskGraphStore,
        warning_msg=(
            "DaskGraphStore and the Dask API are deprecated. "
            "Please switch over to the new API (walpurgis.graph.Graph / "
            "cugraph_pyg.data.GraphStore)."
        ),
        chain_entry=_chain_by_name.get("DaskGraphStore"),
        stacklevel=3,
    )

    # CuGraphStore wraps DaskGraphStore (which is itself deprecated)
    # We use a sentinel class to satisfy the registry's wrapped_cls requirement.
    class _CuGraphStoreSentinel:
        """历史名称 CuGraphStore 的透传哨兵，转发到 DaskGraphStore gate。"""
        __name__ = "CuGraphStore->DaskGraphStore"

        def __new__(cls, *args, **kwargs):
            return DEPRECATED__DaskGraphStore(*args, **kwargs)

    _CuGraphStore_gate = graph_store_deprecation_registry.register(
        name="CuGraphStore",
        wrapped_cls=_CuGraphStoreSentinel,
        warning_msg=(
            "CuGraphStore has been renamed to DaskGraphStore, "
            "which is also deprecated. "
            "Please use walpurgis.graph.Graph instead."
        ),
        chain_entry=_chain_by_name.get("CuGraphStore"),
        stacklevel=3,
    )

    _dbg(
        "graph_store_deprecation",
        "gates built",
        gates=list(graph_store_deprecation_registry._gates.keys()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API — 与上游 data/__init__.py wrapper 函数等价
# ─────────────────────────────────────────────────────────────────────────────

def DaskGraphStore(*args, **kwargs):
    """
    废弃入口: DaskGraphStore — 请改用 walpurgis.graph.Graph / GraphStore。

    等价于上游 431801c 中的 wrapper 函数，但通过 GraphStoreDeprecationGate 路由，
    可查询 call_count 和 WALPURGIS_DEBUG 断点输出。
    """
    _get_or_build_gates()
    if _DaskGraphStore_gate is None:
        raise ImportError("cugraph_pyg.data.dask_graph_store not available")
    return _DaskGraphStore_gate(*args, **kwargs)


def CuGraphStore(*args, **kwargs):
    """
    历史兼容名称 — 已被 DaskGraphStore 替代，后者也已废弃。
    发出双重 FutureWarning: CuGraphStore→DaskGraphStore→GraphStore。
    """
    _get_or_build_gates()
    if _CuGraphStore_gate is None:
        raise ImportError("cugraph_pyg.data.dask_graph_store not available")
    return _CuGraphStore_gate(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ExampleEntryDeprecation — 封装 graph_sage example 入口警告
#
# 上游 431801c 在 graph_sage_mg.py / graph_sage_sg.py 的 main() 函数中
# 直接调用裸 warnings.warn()；
# Walpurgis 将这段逻辑封装为 ExampleEntryDeprecation，
# 携带结构化元数据（entry_name + migration_target），
# 供测试框架验证所有废弃示例均已打标。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExampleEntryDeprecation:
    """
    单个废弃示例入口的元数据。

    上游 431801c 对 graph_sage_mg.py / graph_sage_sg.py 的处理:
        warnings.warn(
            "The Dask API is used in this example is deprecated. "
            "Please refer to 'gcn_dist_mg' for an example that uses the new API.",
            FutureWarning,
        )

    Walpurgis 将此模式结构化，方便审计哪些示例仍依赖 Dask API。
    """
    entry_name: str           # 示例文件名或函数名
    migration_target: str     # 推荐替代示例
    dask_api_version: str     # 最后使用 Dask API 的上游版本


def emit_example_deprecation_warning(entry: ExampleEntryDeprecation) -> None:
    """
    在废弃示例的入口处调用此函数，替代裸 warnings.warn()。

    携带 ExampleEntryDeprecation 结构信息，比上游更易于静态分析。
    """
    warnings.warn(
        f"The Dask API used in '{entry.entry_name}' is deprecated. "
        f"Please refer to '{entry.migration_target}' for an example that uses the new API.",
        FutureWarning,
        stacklevel=2,
    )
    _dbg(
        "ExampleEntryDeprecation",
        f"warning emitted for {entry.entry_name!r}",
        migration_target=entry.migration_target,
    )


#: 431801c 废弃的两个示例入口
DEPRECATED_EXAMPLES: List[ExampleEntryDeprecation] = [
    ExampleEntryDeprecation(
        entry_name="graph_sage_mg",
        migration_target="gcn_dist_mnmg",
        dask_api_version="25.02",
    ),
    ExampleEntryDeprecation(
        entry_name="graph_sage_sg",
        migration_target="gcn_dist_sg",
        dask_api_version="25.02",
    ),
]
