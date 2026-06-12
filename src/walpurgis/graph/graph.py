# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/graph.py (Graph 核心图对象)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「不满是向上的车轮，能够载着不自满的人类，向人道前进。」
# —— 鲁迅《热风·随感录》
#
# f4ca484 合并引入了 cugraph_dgl.Graph——cuGraph 后端的延迟图构建对象，
# 支持单/多 GPU、同/异构图，内含分布式节点/边特征存储。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. _validate_storage_type() 私有方法 — 把 __init__ 中两个 if/raise 提取，
#      便于单独测试和 DEBUG 打印
#   2. _assert_single_call() 私有方法 — 把 add_nodes/add_edges 中
#      "同类型只能调用一次"的重复检查统一
#   3. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - __init__ 参数（storage_type / is_multi_gpu）
#      - add_nodes：global_num_nodes / ntype / data 的 key 列表
#      - add_edges：src_type / dst_type / etype / num_edges
#      - _graph()：方向 / 是否重新构建 / SGGraph/MGGraph
#      - _get_n_emb / _get_e_emb：ntype/etype / indices.shape

import os as _os
import sys as _sys
import time as _time
import warnings
from typing import Union, Optional, Dict, Tuple, List

from walpurgis.utils.imports import import_optional
from walpurgis.graph.features import WholeFeatureStore
from walpurgis.graph.view import (
    HeteroNodeView,
    HeteroNodeDataView,
    HeteroEdgeView,
    HeteroEdgeDataView,
    EmbeddingView,
)
from walpurgis.graph.typing import TensorType

import cupy
import pylibcugraph
from cugraph.gnn import cugraph_comms_get_raft_handle

dgl = import_optional("dgl")
torch = import_optional("torch")
tensordict = import_optional("tensordict")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-GRAPH:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 类型常量
# ---------------------------------------------------------------------------

HOMOGENEOUS_NODE_TYPE: str = "n"
HOMOGENEOUS_EDGE_TYPE: Tuple[str, str, str] = (
    HOMOGENEOUS_NODE_TYPE,
    "e",
    HOMOGENEOUS_NODE_TYPE,
)


def _cast_to_torch_tensor(t: TensorType) -> "torch.Tensor":
    """将各类数组类型统一转为 torch.Tensor。"""
    if isinstance(t, torch.Tensor):
        return t
    try:
        import cupy as cp
        import cudf
        if isinstance(t, (cp.ndarray, cudf.Series)):
            return torch.as_tensor(t, device="cuda")
    except ImportError:
        pass
    try:
        import pandas as pd
        import numpy as np
        if isinstance(t, (pd.Series, np.ndarray)):
            return torch.as_tensor(t, device="cpu")
    except ImportError:
        pass
    return torch.as_tensor(t)


class Graph:
    """
    cuGraph 后端的延迟图构建对象，duck-typed 适配 dgl.DGLGraph。

    f4ca484 核心新增：将 cugraph_dgl.Graph 引入 Walpurgis。
    支持：
    - 单节点/单 GPU、单节点/多 GPU、多节点/多 GPU
    - 同构图与异构图
    - torch 或 wholegraph 分布式特征存储

    图在首次需要时（即创建 DataLoader 时）才真正构建 pylibcugraph 图对象，
    之前对 add_nodes/add_edges 的调用都只是积累边/节点元数据。
    """

    def __init__(
        self,
        is_multi_gpu: bool = False,
        ndata_storage: str = "torch",
        edata_storage: str = "torch",
        **kwargs,
    ):
        """
        Parameters
        ----------
        is_multi_gpu : bool (default=False)
            是否跨多 GPU 分布式存储图。
        ndata_storage : str (default='torch')
            节点特征存储后端：'torch'（复制）或 'wholegraph'（分布式）。
        edata_storage : str (default='torch')
            边特征存储后端：'torch'（复制）或 'wholegraph'（分布式）。
        **kwargs
            传递给 WholeFeatureStore 的可选参数（仅当使用 wholegraph 时有效）。
        """
        self._validate_storage_type("ndata_storage", ndata_storage)
        self._validate_storage_type("edata_storage", edata_storage)

        _dbg(
            "__init__",
            f"is_multi_gpu={is_multi_gpu} "
            f"ndata_storage={ndata_storage!r} edata_storage={edata_storage!r}",
        )

        self.__num_nodes_dict: Dict[str, int] = {}
        self.__num_edges_dict: Dict[Tuple[str, str, str], int] = {}
        self.__edge_indices = tensordict.TensorDict({}, batch_size=(2,))

        self.__graph = None
        self.__vertex_offsets = None
        self.__handle = None
        self.__is_multi_gpu = is_multi_gpu

        self.__ndata_storage_type = (
            WholeFeatureStore
            if ndata_storage == "wholegraph"
            else dgl.storages.pytorch_tensor.PyTorchTensorStorage
        )
        self.__edata_storage_type = (
            WholeFeatureStore
            if edata_storage == "wholegraph"
            else dgl.storages.pytorch_tensor.PyTorchTensorStorage
        )
        self.__ndata_storage: Dict = {}
        self.__edata_storage: Dict = {}
        self.__wg_kwargs = kwargs

    # ------------------------------------------------------------------
    # Walpurgis 改写：_validate_storage_type — 参数校验独立方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_storage_type(param_name: str, value: str) -> None:
        """校验存储类型参数，统一两处重复的 if/raise。"""
        if value not in ("torch", "wholegraph"):
            raise ValueError(
                f"[Walpurgis:Graph] {param_name} 无效（有效值：'torch', 'wholegraph'），"
                f"实际收到：{value!r}"
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.__is_multi_gpu

    def to_canonical_etype(
        self, etype: Union[str, Tuple[str, str, str], None]
    ) -> Tuple[str, str, str]:
        if etype is None:
            if len(self.canonical_etypes) > 1:
                raise ValueError(
                    "[Walpurgis:Graph] 异构图必须指定边类型。"
                )
            return HOMOGENEOUS_EDGE_TYPE

        if isinstance(etype, tuple) and len(etype) == 3:
            return etype

        for src_type, rel_type, dst_type in self.__edge_indices.keys(
            leaves_only=True, include_nested=True
        ):
            if etype == rel_type:
                return (src_type, rel_type, dst_type)

        raise ValueError(f"[Walpurgis:Graph] 未知边关系类型：{etype!r}")

    def add_nodes(
        self,
        global_num_nodes: int,
        data: Optional[Dict[str, TensorType]] = None,
        ntype: Optional[str] = None,
    ) -> None:
        """
        添加指定数量的节点（每种节点类型只能调用一次）。

        Parameters
        ----------
        global_num_nodes : int
            该类型的全局节点总数（所有 worker 传入相同的值）。
        data : Dict[str, TensorType], optional
            节点特征字典。分布式存储时只传本地切片，复制存储时传全量。
        ntype : str, optional
            节点类型名称；同构图可省略。
        """
        if ntype is None:
            if len(self.__num_nodes_dict) > 1:
                raise ValueError("[Walpurgis:Graph] 异构图必须指定节点类型。")
            ntype = HOMOGENEOUS_NODE_TYPE

        self._assert_single_call("节点", ntype, ntype in self.__num_nodes_dict)

        _dbg(
            "add_nodes",
            f"ntype={ntype!r} global_num_nodes={global_num_nodes} "
            f"data_keys={list(data.keys()) if data else []}",
        )

        if self.__is_multi_gpu:
            world_size = torch.distributed.get_world_size()
            local_size = torch.tensor(
                [global_num_nodes], device="cuda", dtype=torch.int64
            )
            ns = torch.empty((world_size,), device="cuda", dtype=torch.int64)
            torch.distributed.all_gather_into_tensor(ns, local_size)
            if not (ns == global_num_nodes).all():
                raise ValueError(
                    "[Walpurgis:Graph] 所有 worker 的 global_num_nodes 必须一致。"
                )

            if data is not None:
                for fname, ft in data.items():
                    feat_size = torch.tensor(
                        [int(ft.shape[0])], device="cuda", dtype=torch.int64
                    )
                    torch.distributed.all_reduce(
                        feat_size, op=torch.distributed.ReduceOp.SUM
                    )
                    if int(feat_size) != global_num_nodes:
                        raise ValueError(
                            f"[Walpurgis:Graph] 特征 {fname!r} 的跨 worker 总长度"
                            f"与 global_num_nodes={global_num_nodes} 不符。"
                        )

        self.__num_nodes_dict[ntype] = global_num_nodes

        if data is not None:
            for fname, ft in data.items():
                self.__ndata_storage[ntype, fname] = self.__ndata_storage_type(
                    _cast_to_torch_tensor(ft), **self.__wg_kwargs
                )

        self.__graph = None
        self.__vertex_offsets = None

    def __check_node_ids(self, ntype: str, ids: "torch.Tensor") -> None:
        """校验节点 ID 是否在合法范围内。"""
        if ntype in self.__num_nodes_dict:
            if ids.max() + 1 > self.num_nodes(ntype):
                raise ValueError(
                    f"[Walpurgis:Graph] 节点 ID 超出类型 {ntype!r} 的合法范围。"
                )
        else:
            raise ValueError(
                f"[Walpurgis:Graph] 类型 {ntype!r} 还未通过 add_nodes() 注册。"
            )

    @staticmethod
    def _assert_single_call(kind: str, type_name: str, already_exists: bool) -> None:
        """
        Walpurgis 改写：统一"同类型只能调用一次"的检查。
        原版在 add_nodes/add_edges 中各自 raise，此处提取为共享断言。
        """
        if already_exists:
            raise ValueError(
                f"[Walpurgis:Graph] {kind}类型 {type_name!r} 已存在，"
                f"cuGraph-DGL 不允许对同一类型多次调用 add_{kind}s()。"
            )

    def add_edges(
        self,
        u: TensorType,
        v: TensorType,
        data: Optional[Dict[str, TensorType]] = None,
        etype: Optional[Union[str, Tuple[str, str, str]]] = None,
    ) -> None:
        """
        添加边（每种边类型只能调用一次）。

        Parameters
        ----------
        u : TensorType
            源节点 ID 张量（本地切片）。
        v : TensorType
            目标节点 ID 张量（本地切片）。
        data : Dict[str, TensorType], optional
            边特征字典。
        etype : Union[str, Tuple[str, str, str]], optional
            边类型；同构图可省略。
        """
        dgl_can_edge_type = self.to_canonical_etype(etype)
        src_type, _, dst_type = dgl_can_edge_type

        existing_keys = list(
            self.__edge_indices.keys(leaves_only=True, include_nested=True)
        )
        self._assert_single_call(
            "边", str(dgl_can_edge_type), dgl_can_edge_type in existing_keys
        )

        u_t = _cast_to_torch_tensor(u)
        v_t = _cast_to_torch_tensor(v)
        self.__check_node_ids(src_type, u_t)
        self.__check_node_ids(dst_type, v_t)

        self.__edge_indices[dgl_can_edge_type] = torch.stack(
            [u_t, v_t]
        ).to(self.idtype)

        if data is not None:
            for attr_name, attr_tensor in data.items():
                self.__edata_storage[
                    dgl_can_edge_type, attr_name
                ] = self.__edata_storage_type(
                    _cast_to_torch_tensor(attr_tensor), **self.__wg_kwargs
                )

        num_edges = self.__edge_indices[dgl_can_edge_type].shape[1]
        if self.__is_multi_gpu:
            ne_t = torch.tensor([num_edges], device="cuda", dtype=torch.int64)
            torch.distributed.all_reduce(ne_t, op=torch.distributed.ReduceOp.SUM)
            num_edges = int(ne_t)

        _dbg(
            "add_edges",
            f"etype={dgl_can_edge_type!r} num_edges={num_edges} "
            f"data_keys={list(data.keys()) if data else []}",
        )

        self.__num_edges_dict[dgl_can_edge_type] = num_edges
        self.__graph = None
        self.__vertex_offsets = None

    def num_nodes(self, ntype: Optional[str] = None) -> int:
        if ntype is None:
            return sum(self.__num_nodes_dict.values())
        return self.__num_nodes_dict[ntype]

    def number_of_nodes(self, ntype: Optional[str] = None) -> int:
        return self.num_nodes(ntype=ntype)

    def num_edges(
        self, etype: Optional[Union[str, Tuple[str, str, str]]] = None
    ) -> int:
        if etype is None:
            return sum(self.__num_edges_dict.values())
        etype = self.to_canonical_etype(etype)
        return self.__num_edges_dict[etype]

    def number_of_edges(
        self, etype: Optional[Union[str, Tuple[str, str, str]]] = None
    ) -> int:
        return self.num_edges(etype=etype)

    @property
    def ntypes(self) -> List[str]:
        return list(self.__num_nodes_dict.keys())

    @property
    def etypes(self) -> List[str]:
        return [et[1] for et in self.__num_edges_dict.keys()]

    @property
    def canonical_etypes(self) -> List[Tuple[str, str, str]]:
        return list(self.__num_edges_dict.keys())

    @property
    def _vertex_offsets(self) -> Dict[str, int]:
        if self.__vertex_offsets is None:
            self.__vertex_offsets = {}
            offset = 0
            for ntype, n in self.__num_nodes_dict.items():
                self.__vertex_offsets[ntype] = offset
                offset += n
        return self.__vertex_offsets

    def __get_edgelist(self, prob_attr=None) -> Dict[str, "torch.Tensor"]:
        # b6163b1: 接受 prob_attr 参数，支持有权重的边列表（BiasedNeighborSampler 路径）
        sorted_keys = sorted(
            self.__edge_indices.keys(leaves_only=True, include_nested=True)
        )

        edge_index = torch.concat(
            [
                torch.stack(
                    [
                        self.__edge_indices[src_type, rel_type, dst_type][0]
                        + self._vertex_offsets[src_type],
                        self.__edge_indices[src_type, rel_type, dst_type][1]
                        + self._vertex_offsets[dst_type],
                    ]
                )
                for (src_type, rel_type, dst_type) in sorted_keys
            ],
            axis=1,
        ).cuda()

        edge_type_array = torch.arange(
            len(sorted_keys), dtype=torch.int32, device="cuda"
        ).repeat_interleave(
            torch.tensor(
                [self.__edge_indices[et].shape[1] for et in sorted_keys],
                device="cuda",
                dtype=torch.int32,
            )
        )

        # b6163b1: 使用 pinned memory 存储 edge_id（加速 CPU/WG 存储访问）
        num_edges_t = torch.tensor(
            [self.__edge_indices[et].shape[1] for et in sorted_keys], device="cuda"
        )

        if self.__is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            num_edges_all_t = torch.empty(
                world_size, num_edges_t.numel(), dtype=torch.int64, device="cuda"
            )
            torch.distributed.all_gather_into_tensor(num_edges_all_t, num_edges_t)

            start_offsets = num_edges_all_t[:rank].T.sum(axis=1)
        else:
            rank = 0
            start_offsets = torch.zeros(
                (len(sorted_keys),), dtype=torch.int64, device="cuda"
            )
            num_edges_all_t = num_edges_t.reshape((1, num_edges_t.numel()))

        # b6163b1: pinned memory 加速 CPU/WG 存储访问
        edge_id_array_per_type = [
            torch.arange(
                start_offsets[i],
                start_offsets[i] + num_edges_all_t[rank][i],
                dtype=torch.int64,
                device="cpu",
            ).pin_memory()
            for i in range(len(sorted_keys))
        ]

        # b6163b1: 从边特征中提取权重（若 prob_attr 不为 None）
        # DGL 隐式要求所有边类型使用相同的 feature name
        if prob_attr is None:
            weights = None
        else:
            if len(sorted_keys) > 1:
                weights = torch.concat(
                    [
                        self.edata[prob_attr][sorted_keys[i]][ix]
                        for i, ix in enumerate(edge_id_array_per_type)
                    ]
                )
            else:
                weights = self.edata[prob_attr][edge_id_array_per_type[0]]

        # b6163b1: 安全移到 cuda（consumer 总会移过去）
        edge_id_array = torch.concat(edge_id_array_per_type).cuda()

        edgelist_dict = {
            "src": edge_index[0],
            "dst": edge_index[1],
            "etp": edge_type_array,
            "eid": edge_id_array,
        }

        if weights is not None:
            edgelist_dict["wgt"] = weights

        return edgelist_dict

    @property
    def is_homogeneous(self) -> bool:
        return (
            len(self.__num_edges_dict) <= 1
            and len(self.__num_nodes_dict) <= 1
        )

    @property
    def idtype(self) -> "torch.dtype":
        return torch.int64

    @property
    def _resource_handle(self):
        if self.__handle is None:
            if self.__is_multi_gpu:
                self.__handle = pylibcugraph.ResourceHandle(
                    cugraph_comms_get_raft_handle().getHandle()
                )
            else:
                self.__handle = pylibcugraph.ResourceHandle()
        return self.__handle

    def _graph(
        self,
        direction: str,
        prob_attr: Optional[str] = None,
    ) -> Union["pylibcugraph.SGGraph", "pylibcugraph.MGGraph"]:
        """
        获取指定方向的 pylibcugraph 图对象（延迟构建）。

        Parameters
        ----------
        direction : str
            'out'（正向）或 'in'（反向采样）。
        prob_attr : str, optional
            b6163b1: 概率/权重边特征名。若提供则使用 BiasedNeighborSampler。
        """
        if direction not in ("out", "in"):
            raise ValueError(
                f"[Walpurgis:Graph._graph] 无效方向 {direction!r}（期望 'in' 或 'out'）。"
            )

        graph_properties = pylibcugraph.GraphProperties(
            is_multigraph=True, is_symmetric=False
        )

        # b6163b1: 改用 dict-based 缓存，区分 direction + prob_attr 两个维度
        # 鲁迅改写：比上游 tuple[0]/tuple[1] 更可读
        if self.__graph is not None:
            if (
                self.__graph["direction"] != direction
                or self.__graph["prob_attr"] != prob_attr
            ):
                self.__graph = None

        if self.__graph is None:
            src_col, dst_col = ("src", "dst") if direction == "out" else ("dst", "src")
            edgelist_dict = self.__get_edgelist(prob_attr=prob_attr)

            _dbg(
                "_graph",
                f"构建 {'MG' if self.__is_multi_gpu else 'SG'}Graph "
                f"direction={direction!r} prob_attr={prob_attr!r} "
                f"num_nodes={self.num_nodes()} "
                f"num_edges={self.num_edges()} "
                f"has_weight={'wgt' in edgelist_dict}",
            )

            if self.__is_multi_gpu:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                vertices_array = cupy.arange(self.num_nodes(), dtype="int64")
                vertices_array = cupy.array_split(vertices_array, world_size)[rank]

                graph = pylibcugraph.MGGraph(
                    self._resource_handle,
                    graph_properties,
                    [cupy.asarray(edgelist_dict[src_col]).astype("int64")],
                    [cupy.asarray(edgelist_dict[dst_col]).astype("int64")],
                    vertices_array=[vertices_array],
                    edge_id_array=[cupy.asarray(edgelist_dict["eid"])],
                    edge_type_array=[cupy.asarray(edgelist_dict["etp"])],
                    weight_array=[cupy.asarray(edgelist_dict["wgt"])]
                    if "wgt" in edgelist_dict
                    else None,
                )
            else:
                graph = pylibcugraph.SGGraph(
                    self._resource_handle,
                    graph_properties,
                    cupy.asarray(edgelist_dict[src_col]).astype("int64"),
                    cupy.asarray(edgelist_dict[dst_col]).astype("int64"),
                    vertices_array=cupy.arange(self.num_nodes(), dtype="int64"),
                    edge_id_array=cupy.asarray(edgelist_dict["eid"]),
                    edge_type_array=cupy.asarray(edgelist_dict["etp"]),
                    weight_array=cupy.asarray(edgelist_dict["wgt"])
                    if "wgt" in edgelist_dict
                    else None,
                )

            self.__graph = {
                "graph": graph,
                "direction": direction,
                "prob_attr": prob_attr,
            }

        return self.__graph["graph"]

    # ------------------------------------------------------------------
    # 节点/边嵌入访问接口
    # ------------------------------------------------------------------

    def _has_n_emb(self, ntype: str, emb_name: str) -> bool:
        return (ntype, emb_name) in self.__ndata_storage

    def _get_n_emb(
        self, ntype: Optional[str], emb_name: str, u: Union[str, TensorType]
    ) -> Union["torch.Tensor", "EmbeddingView"]:
        if ntype is None:
            if len(self.ntypes) == 1:
                ntype = HOMOGENEOUS_NODE_TYPE
            else:
                raise ValueError(
                    "[Walpurgis:Graph._get_n_emb] 异构图必须指定节点类型。"
                )

        # b6163b1: is_all 时返回 EmbeddingView（惰性视图），而非全量 arange
        if dgl.base.is_all(u):
            _dbg(
                "_get_n_emb",
                f"ntype={ntype!r} emb={emb_name!r} → EmbeddingView（惰性）",
            )
            return EmbeddingView(
                self.__ndata_storage[ntype, emb_name], self.num_nodes(ntype)
            )

        _dbg(
            "_get_n_emb",
            f"ntype={ntype!r} emb={emb_name!r} u_shape={tuple(_cast_to_torch_tensor(u).shape)}",
        )

        try:
            return self.__ndata_storage[ntype, emb_name].fetch(
                _cast_to_torch_tensor(u), "cuda"
            )
        except RuntimeError as ex:
            warnings.warn(
                f"[Walpurgis:Graph._get_n_emb] 访问出错，尝试将 index 移至 device: {ex}"
            )
            return self.__ndata_storage[ntype, emb_name].fetch(
                _cast_to_torch_tensor(u).cuda(), "cuda"
            )

    def _has_e_emb(self, etype: Tuple[str, str, str], emb_name: str) -> bool:
        return (etype, emb_name) in self.__edata_storage

    def _get_e_emb(
        self,
        etype: Tuple[str, str, str],
        emb_name: str,
        u: Union[str, TensorType],
    ) -> Union["torch.Tensor", "EmbeddingView"]:
        etype = self.to_canonical_etype(etype)

        # b6163b1: is_all 时返回 EmbeddingView（惰性视图），而非全量 arange
        if dgl.base.is_all(u):
            _dbg(
                "_get_e_emb",
                f"etype={etype!r} emb={emb_name!r} → EmbeddingView（惰性）",
            )
            return EmbeddingView(
                self.__edata_storage[etype, emb_name], self.num_edges(etype)
            )

        _dbg(
            "_get_e_emb",
            f"etype={etype!r} emb={emb_name!r} u_shape={tuple(_cast_to_torch_tensor(u).shape)}",
        )

        try:
            return self.__edata_storage[etype, emb_name].fetch(
                _cast_to_torch_tensor(u), "cuda"
            )
        except RuntimeError as ex:
            warnings.warn(
                f"[Walpurgis:Graph._get_e_emb] 访问出错，尝试将 index 移至 device: {ex}"
            )
            return self.__edata_storage[etype, emb_name].fetch(
                _cast_to_torch_tensor(u).cuda(), "cuda"
            )

    def _set_n_emb(
        self, ntype: str, u: Union[str, TensorType], kv: Dict[str, TensorType]
    ) -> None:
        if not dgl.base.is_all(u):
            raise NotImplementedError(
                "[Walpurgis:Graph._set_n_emb] 暂不支持更新嵌入切片，"
                "请传入 dgl.base.ALL。"
            )
        for k, v in kv.items():
            self.__ndata_storage[ntype, k] = self.__ndata_storage_type(
                _cast_to_torch_tensor(v), **self.__wg_kwargs
            )

    def _set_e_emb(
        self, etype: str, u: Union[str, TensorType], kv: Dict[str, TensorType]
    ) -> None:
        if not dgl.base.is_all(u):
            raise NotImplementedError(
                "[Walpurgis:Graph._set_e_emb] 暂不支持更新嵌入切片，"
                "请传入 dgl.base.ALL。"
            )
        for k, v in kv.items():
            self.__edata_storage[etype, k] = self.__edata_storage_type(
                _cast_to_torch_tensor(v), **self.__wg_kwargs
            )

    def _pop_n_emb(self, ntype: str, key: str) -> "torch.Tensor":
        return self.__ndata_storage.pop((ntype, key))

    def _pop_e_emb(self, etype: str, key: str) -> "torch.Tensor":
        return self.__edata_storage.pop((etype, key))

    def _get_n_emb_keys(self, ntype: str) -> List[str]:
        return [k for (t, k) in self.__ndata_storage if ntype == t]

    def _get_e_emb_keys(self, etype: str) -> List[str]:
        return [k for (t, k) in self.__edata_storage if etype == t]

    def all_edges(
        self,
        form: str = "uv",
        order: str = "eid",
        etype: Optional[Union[str, Tuple[str, str, str]]] = None,
        device: Union[str, int, "torch.device"] = "cpu",
    ):
        """返回所有边（cuGraph-DGL 仅支持 eid 顺序）。"""
        if order != "eid":
            raise NotImplementedError(
                "[Walpurgis:Graph.all_edges] 仅支持 eid 顺序。"
            )
        if etype is None and len(self.canonical_etypes) > 1:
            raise ValueError("[Walpurgis:Graph.all_edges] 异构图必须指定边类型。")

        etype = self.to_canonical_etype(etype)

        if form == "eid":
            return torch.arange(
                0,
                self.__num_edges_dict[etype],
                dtype=self.idtype,
                device=device,
            )
        else:
            if self.__is_multi_gpu:
                raise ValueError(
                    "[Walpurgis:Graph.all_edges] 分布式图不支持 'uv'/'all' 格式。"
                )
            eix = self.__edge_indices[etype].to(device)
            if form == "uv":
                return eix[0], eix[1]
            elif form == "all":
                return (
                    eix[0],
                    eix[1],
                    torch.arange(
                        self.__num_edges_dict[etype], dtype=self.idtype, device=device
                    ),
                )
            else:
                raise ValueError(f"[Walpurgis:Graph.all_edges] 无效 form={form!r}")

    @property
    def ndata(self) -> HeteroNodeDataView:
        if len(self.ntypes) == 1:
            ntype = self.ntypes[0]
            return HeteroNodeDataView(self, ntype, dgl.base.ALL)
        return HeteroNodeDataView(self, self.ntypes, dgl.base.ALL)

    @property
    def edata(self) -> HeteroEdgeDataView:
        if len(self.canonical_etypes) == 1:
            return HeteroEdgeDataView(self, None, dgl.base.ALL)
        return HeteroEdgeDataView(self, self.canonical_etypes, dgl.base.ALL)

    @property
    def nodes(self) -> HeteroNodeView:
        return HeteroNodeView(self)

    @property
    def edges(self) -> HeteroEdgeView:
        return HeteroEdgeView(self)
