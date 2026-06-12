# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/view.py
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「面具戴久了，就会长到脸上，再想揭下来，除非伤筋动骨扒皮。」
# —— 鲁迅《而已集·小杂感》
#
# 上游 HeteroEdgeDataView/HeteroNodeDataView 把所有分支都塞进
# __getitem__/__setitem__，多类型时没有统一的内部抽象，一旦出错
# 只能在四个方法间来回追溯。
# Walpurgis 改写：
#   1. _as_etype_list() / _as_ntype_list() 私有辅助 — 规范化"单类型/多类型"判断
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - HeteroEdgeDataView.__getitem__ / __setitem__ 类型信息
#      - HeteroNodeDataView.__getitem__ / __setitem__ 类型信息

import os as _os
import sys as _sys
import time as _time
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Union, Dict, List, Tuple

from walpurgis.utils.imports import import_optional

# TensorType 由本包的 typing 模块提供（或 tensor.utils 兼容路径）
try:
    from walpurgis.graph.typing import TensorType
except ImportError:
    from typing import Any as TensorType  # type: ignore

torch = import_optional("torch")
dgl = import_optional("dgl")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-VIEW:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


class HeteroEdgeDataView(MutableMapping):
    """
    DGL HeteroEdgeDataView 的 duck-typed 适配版本。
    用于访问和修改边特征。

    f4ca484 新增：供 cugraph_dgl.Graph.edata 属性返回。
    Walpurgis 改写：_is_multi_etype 属性统一多/单类型分支判断。
    """

    def __init__(
        self,
        graph: "walpurgis.graph.Graph",
        etype: Union[Tuple[str, str, str], List[Tuple[str, str, str]]],
        edges: TensorType,
    ):
        self.__graph = graph
        self.__etype = etype
        self.__edges = edges

    @property
    def _etype(self):
        return self.__etype

    @property
    def _graph(self):
        return self.__graph

    @property
    def _edges(self):
        return self.__edges

    @property
    def _is_multi_etype(self) -> bool:
        """Walpurgis: 统一多/单类型判断入口，替代各方法中重复的 isinstance(list)。"""
        return isinstance(self.__etype, list)

    def __getitem__(self, key: str):
        _dbg("HeteroEdgeDataView.__getitem__", f"key={key!r} multi={self._is_multi_etype}")
        if self._is_multi_etype:
            return {
                t: self.__graph._get_e_emb(t, key, self.__edges)
                for t in self.__etype
                if self.__graph._has_e_emb(t, key)
            }
        return self.__graph._get_e_emb(self.__etype, key, self.__edges)

    def __setitem__(self, key: str, val: Union[TensorType, Dict[str, TensorType]]):
        _dbg("HeteroEdgeDataView.__setitem__", f"key={key!r} multi={self._is_multi_etype}")
        if self._is_multi_etype:
            if not isinstance(val, dict):
                raise ValueError(
                    "[Walpurgis:HeteroEdgeDataView] 多边类型视图要求传入 dict。"
                )
            for t, v in val.items():
                if t not in self.__etype:
                    raise ValueError(
                        f"[Walpurgis:HeteroEdgeDataView] 类型 {t!r} 不在当前视图范围内。"
                    )
                self.__graph.set_e_emb(t, self.__edges, {key: v})
        else:
            if isinstance(val, dict):
                raise ValueError(
                    "[Walpurgis:HeteroEdgeDataView] 单边类型视图要求传入单一张量，而非 dict。"
                )
            self.__graph.set_e_emb(self.__etype, self.__edges, {key: val})

    def __delitem__(self, key: str):
        if self._is_multi_etype:
            for t in self.__etype:
                self.__graph.pop_e_emb(t, key)
        else:
            self.__graph.pop_e_emb(self.__etype, key)

    def _transpose(self, fetch_vals: bool = True):
        if self._is_multi_etype:
            tr = defaultdict(dict)
            for etype in self.__etype:
                for key in self.__graph._get_e_emb_keys(etype):
                    tr[key][etype] = (
                        self.__graph._get_e_emb(etype, key, self.__edges)
                        if fetch_vals
                        else []
                    )
        else:
            tr = {}
            for key in self.__graph._get_e_emb_keys(self.__etype):
                tr[key] = (
                    self.__graph._get_e_emb(self.__etype, key, self.__edges)
                    if fetch_vals
                    else []
                )
        return tr

    def __len__(self):
        return len(self._transpose(fetch_vals=False))

    def __iter__(self):
        return iter(self._transpose())

    def keys(self):
        return self._transpose(fetch_vals=False).keys()

    def values(self):
        return self._transpose().values()

    def __repr__(self):
        return repr(self._transpose(fetch_vals=False))


class HeteroNodeDataView(MutableMapping):
    """
    DGL HeteroNodeDataView 的 duck-typed 适配版本。
    用于访问和修改节点特征。

    f4ca484 新增：供 cugraph_dgl.Graph.ndata 属性返回。
    Walpurgis 改写：_is_multi_ntype 属性统一多/单类型分支判断。
    """

    def __init__(
        self,
        graph: "walpurgis.graph.Graph",
        ntype: Union[str, List[str]],
        nodes: TensorType,
    ):
        self.__graph = graph
        self.__ntype = ntype
        self.__nodes = nodes

    @property
    def _ntype(self):
        return self.__ntype

    @property
    def _graph(self):
        return self.__graph

    @property
    def _nodes(self):
        return self.__nodes

    @property
    def _is_multi_ntype(self) -> bool:
        """Walpurgis: 统一多/单类型判断入口。"""
        return isinstance(self.__ntype, list)

    def __getitem__(self, key: str):
        _dbg("HeteroNodeDataView.__getitem__", f"key={key!r} multi={self._is_multi_ntype}")
        if self._is_multi_ntype:
            return {
                t: self.__graph._get_n_emb(t, key, self.__nodes)
                for t in self.__ntype
                if self.__graph._has_n_emb(t, key)
            }
        return self.__graph._get_n_emb(self.__ntype, key, self.__nodes)

    def __setitem__(self, key: str, val: Union[TensorType, Dict[str, TensorType]]):
        _dbg("HeteroNodeDataView.__setitem__", f"key={key!r} multi={self._is_multi_ntype}")
        if self._is_multi_ntype:
            if not isinstance(val, dict):
                raise ValueError(
                    "[Walpurgis:HeteroNodeDataView] 多节点类型视图要求传入 dict。"
                )
            for t, v in val.items():
                if t not in self.__ntype:
                    raise ValueError(
                        f"[Walpurgis:HeteroNodeDataView] 类型 {t!r} 不在当前视图范围内。"
                    )
                self.__graph._set_n_emb(t, self.__nodes, {key: v})
        else:
            if isinstance(val, dict):
                raise ValueError(
                    "[Walpurgis:HeteroNodeDataView] 单节点类型视图要求传入单一张量，而非 dict。"
                )
            self.__graph._set_n_emb(self.__ntype, self.__nodes, {key: val})

    def __delitem__(self, key: str):
        if self._is_multi_ntype:
            for t in self.__ntype:
                self.__graph._pop_n_emb(t, key)
        else:
            self.__graph._pop_n_emb(self.__ntype, key)

    def _transpose(self, fetch_vals: bool = True):
        if self._is_multi_ntype:
            tr = defaultdict(dict)
            for ntype in self.__ntype:
                for key in self.__graph._get_n_emb_keys(ntype):
                    tr[key][ntype] = (
                        self.__graph._get_n_emb(ntype, key, self.__nodes)
                        if fetch_vals
                        else []
                    )
        else:
            tr = {}
            for key in self.__graph._get_n_emb_keys(self.__ntype):
                tr[key] = (
                    self.__graph._get_n_emb(self.__ntype, key, self.__nodes)
                    if fetch_vals
                    else []
                )
        return tr

    def __len__(self):
        return len(self._transpose(fetch_vals=False))

    def __iter__(self):
        return iter(self._transpose())

    def keys(self):
        return self._transpose(fetch_vals=False).keys()

    def values(self):
        return self._transpose().values()

    def __repr__(self):
        return repr(self._transpose(fetch_vals=False))


class HeteroEdgeView:
    """
    DGL HeteroEdgeView 的 duck-typed 适配版本。
    """

    def __init__(self, graph: "walpurgis.graph.Graph"):
        self.__graph = graph

    @property
    def _graph(self):
        return self.__graph

    def __getitem__(self, key):
        _dbg("HeteroEdgeView.__getitem__", f"key={key!r}")
        if isinstance(key, slice):
            if not (key.start is None and key.stop is None and key.step is None):
                raise ValueError(
                    "[Walpurgis:HeteroEdgeView] 仅支持全量 slice（[:])。"
                )
            edges = dgl.base.ALL
            etype = None
        elif key is None:
            edges = dgl.base.ALL
            etype = None
        elif isinstance(key, tuple):
            if len(key) == 3:
                edges = dgl.base.ALL
                etype = key
            else:
                edges = key
                etype = None
        elif isinstance(key, str):
            edges = dgl.base.ALL
            etype = key
        else:
            edges = key
            etype = None

        return HeteroEdgeDataView(graph=self.__graph, etype=etype, edges=edges)

    def __call__(self, *args, **kwargs):
        if "device" in kwargs:
            return self.__graph.all_edges(*args, **kwargs)
        return self.__graph.all_edges(*args, **kwargs, device="cuda")


class HeteroNodeView:
    """
    DGL HeteroNodeView 的 duck-typed 适配版本。
    """

    def __init__(self, graph: "walpurgis.graph.Graph"):
        self.__graph = graph

    @property
    def _graph(self):
        return self.__graph

    def __getitem__(self, key):
        _dbg("HeteroNodeView.__getitem__", f"key={key!r}")
        if isinstance(key, slice):
            if not (key.start is None and key.stop is None and key.step is None):
                raise ValueError(
                    "[Walpurgis:HeteroNodeView] 仅支持全量 slice（[:])。"
                )
            nodes = dgl.base.ALL
            ntype = None
        elif isinstance(key, tuple):
            nodes, ntype = key
        elif key is None or isinstance(key, str):
            nodes = dgl.base.ALL
            ntype = key
        else:
            nodes = key
            ntype = None

        return HeteroNodeDataView(graph=self.__graph, ntype=ntype, nodes=nodes)

    def __call__(self, ntype=None):
        return torch.arange(
            0,
            self.__graph.num_nodes(ntype),
            dtype=self.__graph.idtype,
            device="cuda",
        )


# b6163b1: 将 EmbeddingView 从 tensor 层 re-export 到 graph.view，
# 与上游 cugraph_dgl/view.py 的 EmbeddingView 位置对应。
from walpurgis.tensor.embedding_view import EmbeddingView  # noqa: F401
