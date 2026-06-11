# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit a9ab8b4
# 原标题: [FEA] Support Heterogeneous Sampling in cuGraph-PyG
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「从来如此，便对么？」 —— 鲁迅《狂人日记》
# 上游在 BaseSampler 里有两处 raise NotImplementedError("Sampling heterogeneous graphs
# is currently unsupported in the non-dask API")。
# a9ab8b4 把它们替换成真正的 HeterogeneousSampleReader 调用路径。
# 同步新增 SampleReader.lho_name 兼容逻辑（label_type_hop_offsets vs label_hop_offsets），
# 修复 SampleIterator 异构 HeteroSamplerOutput 路径的 .items() bug。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）:
#   1. _HeteroDecodeContext dataclass — 将 HeterogeneousSampleReader.__decode_coo
#      内散落的 4 个 double-underscore 成员提取为统一上下文，DEBUG 输出含所有关键维度
#   2. _safe_max_plus_one() — 替代 decode_coo 里 3 处裸 `x.max() + 1`，
#      空 tensor 时返回 0 而非 RuntimeError（上游同款 silent bug，前一批 commit 已在
#      hetero_sample_reader.py 里标注，此处统一修复）
#   3. 全链路 WALPURGIS_DEBUG=1 断点 print:
#      - SampleIterator.__next__ 输出类型判断
#      - SampleReader.__next__ lho_name 自动选取
#      - HeterogeneousSampleReader.__init__ 维度摘要
#      - HeterogeneousSampleReader.__decode_coo 每个 etype 的 map/lho 切片
#      - HomogeneousSampleReader._decode 分支（CSC/COO）
#      - BaseSampler.sample_from_nodes/sample_from_edges 同/异构路径选择
#   4. BaseSampler 中异构路径不再 raise NotImplementedError — 核心迁移目标

import os as _os
import sys as _sys
import time as _time
from dataclasses import dataclass
from math import ceil
from typing import Optional, Iterator, Union, Dict, Tuple, List

from walpurgis.utils.imports import import_optional
from walpurgis.sampler.distributed_sampler import DistributedNeighborSampler

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点输出：WALPURGIS_DEBUG=1 时打印到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-SAMPLER:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 迁移新增：_safe_max_plus_one — 空 tensor guard（上游 silent bug，此处修复）
# ---------------------------------------------------------------------------

def _safe_max_plus_one(t: "torch.Tensor") -> "torch.Tensor":
    """
    返回 t.max() + 1；若 t 为空则返回 0。

    上游 __decode_coo 中有 3 处裸 ux.max() + 1，空 tensor 时触发 RuntimeError:
    "max() arg is an empty sequence"。 Walpurgis 将其统一为此辅助函数。
    前序 commit hetero_sample_reader.py 中已标注此问题（_safe_max_plus_one 命名一致）。
    """
    if t.numel() == 0:
        return torch.tensor(0, dtype=torch.int64, device=t.device)
    return t.max() + 1


# ---------------------------------------------------------------------------
# 迁移新增：_HeteroDecodeContext — 封装 HeterogeneousSampleReader 解码状态
# ---------------------------------------------------------------------------

@dataclass
class _HeteroDecodeContext:
    """
    HeterogeneousSampleReader.__decode_coo 每次调用时的上下文快照。

    上游将 num_edge_types / fanout_length / num_vertex_types 全部通过局部变量传递，
    难以在 DEBUG 时看清维度关系。 Walpurgis 将其封装，并在 DEBUG=1 时打印摘要。
    """
    num_edge_types: int
    fanout_length: int
    num_vertex_types: int
    index: int

    def summary(self) -> str:
        return (
            f"index={self.index} "
            f"num_edge_types={self.num_edge_types} "
            f"fanout_length={self.fanout_length} "
            f"num_vertex_types={self.num_vertex_types}"
        )


# ---------------------------------------------------------------------------
# SampleIterator
# ---------------------------------------------------------------------------

class SampleIterator:
    """
    Iterator that combines output graphs with their features to produce
    final output minibatches that can be fed into a GNN model.

    a9ab8b4 修复：HeteroSamplerOutput 路径的 .col 迭代从 next_sample.col
    （裸遍历，返回 (edge_type, tensor) 二元组）改为 next_sample.col.items()
    （dict 迭代，正确返回 key-value 对）。
    """

    def __init__(
        self,
        data: Tuple[
            "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
        ],
        output_iter: Iterator[
            Union[
                "torch_geometric.sampler.HeteroSamplerOutput",
                "torch_geometric.sampler.SamplerOutput",
            ]
        ],
    ):
        self.__feature_store, self.__graph_store = data
        self.__output_iter = output_iter

    def __next__(self):
        next_sample = next(self.__output_iter)

        _dbg(
            "SampleIterator",
            f"next_sample type={type(next_sample).__name__}",
        )

        if isinstance(next_sample, torch_geometric.sampler.SamplerOutput):
            sz = next_sample.edge.numel()
            if sz == next_sample.col.numel() and (
                next_sample.node.numel() > next_sample.col[-1]
            ):
                col = next_sample.col
            else:
                col = torch_geometric.edge_index.ptr2index(
                    next_sample.col, next_sample.edge.numel()
                )

            from walpurgis.sampler.sampler_utils import filter_cugraph_pyg_store
            data = filter_cugraph_pyg_store(
                self.__feature_store,
                self.__graph_store,
                next_sample.node,
                next_sample.row,
                col,
                next_sample.edge,
                None,
            )

            if "n_id" not in data:
                data.n_id = next_sample.node
            if next_sample.edge is not None and "e_id" not in data:
                edge = next_sample.edge.to(torch.long)
                data.e_id = edge

            data.batch = next_sample.batch
            data.num_sampled_nodes = next_sample.num_sampled_nodes
            data.num_sampled_edges = next_sample.num_sampled_edges

            data.input_id = next_sample.metadata[0]
            data.batch_size = data.input_id.size(0)

            if len(next_sample.metadata) == 2:
                data.seed_time = next_sample.metadata[1]
            elif len(next_sample.metadata) == 4:
                (
                    data.edge_label_index,
                    data.edge_label,
                    data.seed_time,
                ) = next_sample.metadata[1:]
            else:
                raise ValueError(
                    f"[Walpurgis:SampleIterator] 无效 metadata 长度: "
                    f"{len(next_sample.metadata)}，期望 2 或 4"
                )

        elif isinstance(next_sample, torch_geometric.sampler.HeteroSamplerOutput):
            col = {}
            # a9ab8b4 修复：上游旧代码 for edge_type, col_idx in next_sample.col:
            # 对 dict 裸遍历只能拿到 key，无法解包二元组，运行时 ValueError。
            # 修复：改为 .items()
            for edge_type, col_idx in next_sample.col.items():
                sz = next_sample.edge[edge_type].numel()
                if sz == col_idx.numel():
                    col[edge_type] = col_idx
                else:
                    col[edge_type] = torch_geometric.edge_index.ptr2index(col_idx, sz)

            data = torch_geometric.loader.utils.filter_custom_hetero_store(
                self.__feature_store,
                self.__graph_store,
                next_sample.node,
                next_sample.row,
                col,
                next_sample.edge,
                None,
            )

            for key, node in next_sample.node.items():
                if "n_id" not in data[key]:
                    data[key].n_id = node

            for key, edge in (next_sample.edge or {}).items():
                if edge is not None and "e_id" not in data[key]:
                    edge = edge.to(torch.long)
                    data[key].e_id = edge

            data.set_value_dict("batch", next_sample.batch)
            data.set_value_dict("num_sampled_nodes", next_sample.num_sampled_nodes)
            data.set_value_dict("num_sampled_edges", next_sample.num_sampled_edges)

            # a9ab8b4 新增：正确设置异构图的 input_id
            input_type, input_id = next_sample.metadata[0]
            data[input_type].input_id = input_id
            data[input_type].batch_size = input_id.size(0)

            _dbg(
                "SampleIterator",
                f"HeteroSamplerOutput input_type={input_type!r} "
                f"input_id.shape={tuple(input_id.shape)}",
            )

            if len(next_sample.metadata) == 2:
                data[input_type].seed_time = next_sample.metadata[1]
            elif len(next_sample.metadata) == 4:
                (
                    data[input_type].edge_label_index,
                    data[input_type].edge_label,
                    data[input_type].seed_time,
                ) = next_sample.metadata[1:]
            else:
                raise ValueError(
                    f"[Walpurgis:SampleIterator] 无效 metadata 长度（异构）: "
                    f"{len(next_sample.metadata)}，期望 2 或 4"
                )
        else:
            raise ValueError(
                f"[Walpurgis:SampleIterator] 无效输出类型: {type(next_sample).__name__}"
            )

        return data

    def __iter__(self):
        return self


# ---------------------------------------------------------------------------
# SampleReader
# ---------------------------------------------------------------------------

class SampleReader:
    """
    Iterator that processes results from the cuGraph distributed sampler.

    a9ab8b4 新增：lho_name 自动选取逻辑，兼容旧格式 (label_hop_offsets) 与
    新异构格式 (label_type_hop_offsets)。
    """

    def __init__(
        self, base_reader: Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]
    ):
        self.__base_reader = base_reader
        self.__num_samples_remaining = 0
        self.__index = 0

    def __next__(self):
        if self.__num_samples_remaining == 0:
            self.__raw_sample_data, start_inclusive, end_inclusive = next(
                self.__base_reader
            )

            # a9ab8b4：动态选 lho_name，兼容同/异构采样输出
            lho_name = (
                "label_type_hop_offsets"
                if "label_type_hop_offsets" in self.__raw_sample_data
                else "label_hop_offsets"
            )

            _dbg(
                "SampleReader",
                f"加载新 call_group | lho_name={lho_name} "
                f"samples=[{start_inclusive}, {end_inclusive}] "
                f"keys={list(self.__raw_sample_data.keys())}",
            )

            self.__raw_sample_data["input_offsets"] -= self.__raw_sample_data[
                "input_offsets"
            ][0].clone()
            self.__raw_sample_data[lho_name] -= self.__raw_sample_data[lho_name][
                0
            ].clone()
            self.__raw_sample_data["renumber_map_offsets"] -= self.__raw_sample_data[
                "renumber_map_offsets"
            ][0].clone()
            if "major_offsets" in self.__raw_sample_data:
                self.__raw_sample_data["major_offsets"] -= self.__raw_sample_data[
                    "major_offsets"
                ][0].clone()

            self.__num_samples_remaining = end_inclusive - start_inclusive + 1
            self.__index = 0

        out = self._decode(self.__raw_sample_data, self.__index)
        self.__index += 1
        self.__num_samples_remaining -= 1
        return out

    def __iter__(self):
        return self


# ---------------------------------------------------------------------------
# HeterogeneousSampleReader — a9ab8b4 核心新增
# ---------------------------------------------------------------------------

class HeterogeneousSampleReader(SampleReader):
    """
    Subclass of SampleReader that reads heterogeneous output samples
    produced by the cuGraph distributed sampler.

    a9ab8b4 中实现了此类，替代 BaseSampler 里原来的 NotImplementedError。
    Walpurgis 改写：
    - _HeteroDecodeContext dataclass 封装解码维度
    - _safe_max_plus_one() 替代裸 .max()+1，防止空 tensor 崩溃
    - 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖每个 etype 的 map/lho 切片
    """

    def __init__(
        self,
        base_reader: Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]],
        src_types: "torch.Tensor",
        dst_types: "torch.Tensor",
        vertex_offsets: "torch.Tensor",
        edge_types: List[Tuple[str, str, str]],
        vertex_types: List[str],
    ):
        self.__src_types = src_types
        self.__dst_types = dst_types
        self.__edge_types = edge_types
        self.__vertex_types = vertex_types
        self.__num_vertex_types = len(vertex_types)
        self.__vertex_offsets = vertex_offsets

        _dbg(
            "HeteroReader",
            f"初始化 | num_edge_types={src_types.numel()} "
            f"num_vertex_types={self.__num_vertex_types} "
            f"edge_types={edge_types} "
            f"vertex_types={vertex_types}",
        )

        super().__init__(base_reader)

    def __decode_coo(
        self,
        raw_sample_data: Dict[str, "torch.Tensor"],
        index: int,
    ):
        num_edge_types = self.__src_types.numel()
        fanout_length = raw_sample_data["fanout"].numel() // num_edge_types

        ctx = _HeteroDecodeContext(
            num_edge_types=num_edge_types,
            fanout_length=fanout_length,
            num_vertex_types=self.__num_vertex_types,
            index=index,
        )
        _dbg("HeteroReader.__decode_coo", ctx.summary())

        num_sampled_nodes = [
            torch.zeros((fanout_length + 1,), dtype=torch.int64, device="cuda")
            for _ in range(self.__num_vertex_types)
        ]

        num_sampled_edges = {}
        node = {}
        row = {}
        col = {}
        edge = {}

        # a9ab8b4: input_type 由 BaseSampler 通过 metadata 传入，不再靠启发式猜测
        input_type = raw_sample_data.get("input_type")
        if input_type is None:
            raise ValueError(
                "[Walpurgis:HeterogeneousSampleReader] raw_sample_data 中缺少 input_type。\n"
                "确认 BaseSampler 已将 metadata={'input_type': ...} 传入 DistributedNeighborSampler。"
            )

        integer_input_type = None

        for etype in range(num_edge_types):
            pyg_can_etype = self.__edge_types[etype]

            # src node map
            jx = self.__src_types[etype] + index * self.__num_vertex_types
            map_ptr_src_beg = raw_sample_data["renumber_map_offsets"][jx]
            map_ptr_src_end = raw_sample_data["renumber_map_offsets"][jx + 1]
            map_src = raw_sample_data["map"][map_ptr_src_beg:map_ptr_src_end]
            node[pyg_can_etype[0]] = (
                map_src - self.__vertex_offsets[self.__src_types[etype]]
            ).cpu()

            # dst node map
            kx = self.__dst_types[etype] + index * self.__num_vertex_types
            map_ptr_dst_beg = raw_sample_data["renumber_map_offsets"][kx]
            map_ptr_dst_end = raw_sample_data["renumber_map_offsets"][kx + 1]
            map_dst = raw_sample_data["map"][map_ptr_dst_beg:map_ptr_dst_end]
            node[pyg_can_etype[2]] = (
                map_dst - self.__vertex_offsets[self.__dst_types[etype]]
            ).cpu()

            _dbg(
                "HeteroReader",
                f"etype={etype} {pyg_can_etype} "
                f"map_src.shape={tuple(map_src.shape)} "
                f"map_dst.shape={tuple(map_dst.shape)} "
                f"jx={int(jx)} kx={int(kx)}",
            )

            # edge lho slicing
            edge_ptr_beg = (
                index * num_edge_types * fanout_length + etype * fanout_length
            )
            edge_ptr_end = (
                index * num_edge_types * fanout_length + (etype + 1) * fanout_length
            )
            lho = raw_sample_data["label_type_hop_offsets"][
                edge_ptr_beg : edge_ptr_end + 1
            ]

            _dbg(
                "HeteroReader",
                f"etype={etype} lho=[{int(lho[0])}, {int(lho[-1])}] "
                f"edge_ptr=[{edge_ptr_beg}, {edge_ptr_end}]",
            )

            num_sampled_edges[pyg_can_etype] = lho.diff()

            eid_i = raw_sample_data["edge_id"][lho[0] : lho[-1]]

            eirx = (index * num_edge_types) + etype
            edge_id_ptr_beg = raw_sample_data["edge_renumber_map_offsets"][eirx]
            edge_id_ptr_end = raw_sample_data["edge_renumber_map_offsets"][eirx + 1]
            emap = raw_sample_data["edge_renumber_map"][edge_id_ptr_beg:edge_id_ptr_end]
            edge[pyg_can_etype] = emap[eid_i]

            col[pyg_can_etype] = raw_sample_data["majors"][lho[0] : lho[-1]]
            row[pyg_can_etype] = raw_sample_data["minors"][lho[0] : lho[-1]]

            # num_sampled_nodes per hop
            for hop in range(fanout_length):
                vx = raw_sample_data["majors"][: lho[hop + 1]]
                if vx.numel() > 0:
                    num_sampled_nodes[self.__dst_types[etype]][hop + 1] = torch.max(
                        num_sampled_nodes[self.__dst_types[etype]][hop + 1],
                        vx.max() + 1,
                    )
                vy = raw_sample_data["minors"][: lho[hop + 1]]
                if vy.numel() > 0:
                    num_sampled_nodes[self.__src_types[etype]][hop + 1] = torch.max(
                        num_sampled_nodes[self.__src_types[etype]][hop + 1],
                        vy.max() + 1,
                    )

            # input_type 匹配逻辑（a9ab8b4：正式判断，不再靠 ux.numel() 启发式）
            if input_type == pyg_can_etype:
                # 边采样：同时更新 dst 和 src 的 hop-0 seed 数
                integer_input_type = etype
                hop0_edges = num_sampled_edges[pyg_can_etype][0]
                ux = col[pyg_can_etype][:hop0_edges]
                uy = row[pyg_can_etype][:hop0_edges]
                # _safe_max_plus_one — Walpurgis 空 tensor guard
                num_sampled_nodes[self.__dst_types[etype]][0] = torch.max(
                    num_sampled_nodes[self.__dst_types[etype]][0],
                    _safe_max_plus_one(ux).reshape((1,)),
                )
                num_sampled_nodes[self.__src_types[etype]][0] = torch.max(
                    num_sampled_nodes[self.__src_types[etype]][0],
                    _safe_max_plus_one(uy).reshape((1,)),
                )
                _dbg(
                    "HeteroReader",
                    f"edge 采样匹配 etype={etype} {pyg_can_etype} "
                    f"hop0_edges={int(hop0_edges)} "
                    f"ux.numel={ux.numel()} uy.numel={uy.numel()}",
                )
            elif isinstance(input_type, str) and input_type == pyg_can_etype[2]:
                # 节点采样：只更新 dst hop-0
                integer_input_type = self.__src_types[etype]
                hop0_edges = num_sampled_edges[pyg_can_etype][0]
                ux = col[pyg_can_etype][:hop0_edges]
                if ux.numel() > 0:
                    num_sampled_nodes[self.__dst_types[etype]][0] = torch.max(
                        num_sampled_nodes[self.__dst_types[etype]][0],
                        _safe_max_plus_one(ux).reshape((1,)),
                    )
                _dbg(
                    "HeteroReader",
                    f"节点采样匹配 etype={etype} {pyg_can_etype} "
                    f"input_type={input_type!r} "
                    f"integer_input_type={int(integer_input_type)}",
                )

        if integer_input_type is None:
            raise ValueError(
                f"[Walpurgis:HeterogeneousSampleReader] "
                f"input_type {input_type!r} 与所有 edge_type 均不匹配。\n"
                f"已知 edge_types: {self.__edge_types}\n"
                f"检查 BaseSampler 传入的 index.input_type 是否与图 edge/node type 名称一致。"
            )

        num_sampled_nodes = {
            self.__vertex_types[i]: z.diff(
                prepend=torch.zeros((1,), dtype=torch.int64, device="cuda")
            ).cpu()
            for i, z in enumerate(num_sampled_nodes)
        }
        num_sampled_edges = {k: v.cpu() for k, v in num_sampled_edges.items()}

        input_index = raw_sample_data["input_index"][
            raw_sample_data["input_offsets"][index] : raw_sample_data["input_offsets"][
                index + 1
            ]
        ]

        num_seeds = input_index.numel()
        input_index = input_index[input_index >= 0]
        num_pos = input_index.numel()
        num_neg = num_seeds - num_pos

        if num_neg > 0:
            edge_label = torch.concat(
                [
                    torch.full((num_pos,), 1.0),
                    torch.full((num_neg,), 0.0),
                ]
            )
        else:
            if "input_label" in raw_sample_data:
                edge_label = raw_sample_data["input_label"][
                    raw_sample_data["input_offsets"][index] : raw_sample_data[
                        "input_offsets"
                    ][index + 1]
                ]
            else:
                edge_label = None

        input_index = (input_type, input_index)

        edge_inverse = (
            (
                raw_sample_data["edge_inverse"][
                    (raw_sample_data["input_offsets"][index] * 2) : (
                        raw_sample_data["input_offsets"][index + 1] * 2
                    )
                ]
            )
            if "edge_inverse" in raw_sample_data
            else None
        )

        if edge_inverse is None:
            metadata = (
                input_index,
                None,  # TODO: 时间采样支持
            )
        else:
            edge_inverse = edge_inverse.view(2, -1)
            if isinstance(input_type, str):
                raise ValueError(
                    "[Walpurgis:HeterogeneousSampleReader] "
                    "边采样（edge_inverse 存在）时 input_type 应为 tuple，"
                    f"实际得到 str: {input_type!r}"
                )
            else:
                # De-offset 基于词典序
                if input_type[0] != input_type[2]:
                    if input_type[0] < input_type[2]:
                        edge_inverse[1] -= edge_inverse[0].max() + 1
                    else:
                        edge_inverse[0] -= edge_inverse[1].max() + 1

            _dbg(
                "HeteroReader",
                f"edge_inverse de-offset 完成 "
                f"ei[0].range=[{int(edge_inverse[0].min())}, {int(edge_inverse[0].max())}] "
                f"ei[1].range=[{int(edge_inverse[1].min())}, {int(edge_inverse[1].max())}]",
            )

            metadata = (
                input_index,
                edge_inverse,
                edge_label,
                None,  # TODO: 时间采样支持
            )

        return torch_geometric.sampler.HeteroSamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=edge,
            batch=None,
            num_sampled_nodes=num_sampled_nodes,
            num_sampled_edges=num_sampled_edges,
            metadata=metadata,
        )

    def _decode(
        self,
        raw_sample_data: Dict[str, Union["torch.Tensor", str, Tuple[str, str, str]]],
        index: int,
    ):
        if "major_offsets" in raw_sample_data:
            raise ValueError(
                "[Walpurgis:HeterogeneousSampleReader] "
                "CSR 格式（major_offsets）当前不支持异构图，"
                "请使用 COO 格式（NeighborLoader 默认对异构图使用 compression='COO'）。"
            )
        else:
            return self.__decode_coo(raw_sample_data, index)


# ---------------------------------------------------------------------------
# HomogeneousSampleReader
# ---------------------------------------------------------------------------

class HomogeneousSampleReader(SampleReader):
    """
    Subclass of SampleReader that reads homogeneous output samples
    produced by the cuGraph distributed sampler.
    """

    def __init__(
        self, base_reader: Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]
    ):
        super().__init__(base_reader)

    def __decode_csc(
        self,
        raw_sample_data: Dict[str, Union["torch.Tensor", str, Tuple[str, str, str]]],
        index: int,
    ):
        _dbg("HomoReader", f"CSC decode index={index}")

        fanout_length = (raw_sample_data["label_hop_offsets"].numel() - 1) // (
            raw_sample_data["renumber_map_offsets"].numel() - 1
        )

        major_offsets_start_incl = raw_sample_data["label_hop_offsets"][
            index * fanout_length
        ]
        major_offsets_end_incl = raw_sample_data["label_hop_offsets"][
            (index + 1) * fanout_length
        ]

        major_offsets = raw_sample_data["major_offsets"][
            major_offsets_start_incl : major_offsets_end_incl + 1
        ].clone()
        minors = raw_sample_data["minors"][major_offsets[0] : major_offsets[-1]]
        edge_id = raw_sample_data["edge_id"][major_offsets[0] : major_offsets[-1]]

        major_offsets -= major_offsets[0].clone()

        renumber_map_start = raw_sample_data["renumber_map_offsets"][index]
        renumber_map_end = raw_sample_data["renumber_map_offsets"][index + 1]
        renumber_map = raw_sample_data["map"][renumber_map_start:renumber_map_end]

        current_label_hop_offsets = raw_sample_data["label_hop_offsets"][
            index * fanout_length : (index + 1) * fanout_length + 1
        ].clone()
        current_label_hop_offsets -= current_label_hop_offsets[0].clone()

        num_sampled_edges = major_offsets[current_label_hop_offsets].diff()

        num_sampled_nodes_hops = torch.tensor(
            [
                minors[: num_sampled_edges[:i].sum()].max() + 1
                for i in range(1, fanout_length + 1)
            ],
            device="cpu",
        )

        num_seeds = (
            torch.searchsorted(major_offsets, num_sampled_edges[0]).reshape((1,)).cpu()
        )
        num_sampled_nodes = torch.concat(
            [num_seeds, num_sampled_nodes_hops.diff(prepend=num_seeds)]
        )

        input_index = raw_sample_data["input_index"][
            raw_sample_data["input_offsets"][index] : raw_sample_data["input_offsets"][
                index + 1
            ]
        ]

        num_seeds = input_index.numel()
        input_index = input_index[input_index >= 0]
        num_pos = input_index.numel()
        num_neg = num_seeds - num_pos

        if num_neg > 0:
            edge_label = torch.concat(
                [
                    torch.full((num_pos,), 1.0),
                    torch.full((num_neg,), 0.0),
                ]
            )
        else:
            if "input_label" in raw_sample_data:
                edge_label = raw_sample_data["input_label"][
                    raw_sample_data["input_offsets"][index] : raw_sample_data[
                        "input_offsets"
                    ][index + 1]
                ]
            else:
                edge_label = None

        edge_inverse = (
            (
                raw_sample_data["edge_inverse"][
                    (raw_sample_data["input_offsets"][index] * 2) : (
                        raw_sample_data["input_offsets"][index + 1] * 2
                    )
                ]
            )
            if "edge_inverse" in raw_sample_data
            else None
        )

        if edge_inverse is None:
            metadata = (input_index, None)
        else:
            edge_inverse = edge_inverse.view(2, -1)
            metadata = (input_index, edge_inverse, edge_label, None)

        return torch_geometric.sampler.SamplerOutput(
            node=renumber_map.cpu(),
            row=minors,
            col=major_offsets,
            edge=edge_id.cpu(),
            batch=renumber_map[:num_seeds],
            num_sampled_nodes=num_sampled_nodes.cpu(),
            num_sampled_edges=num_sampled_edges.cpu(),
            metadata=metadata,
        )

    def __decode_coo(
        self,
        raw_sample_data: Dict[str, Union["torch.Tensor", str, Tuple[str, str, str]]],
        index: int,
    ):
        _dbg("HomoReader", f"COO decode index={index}")

        fanout_length = (raw_sample_data["label_hop_offsets"].numel() - 1) // (
            raw_sample_data["renumber_map_offsets"].numel() - 1
        )

        major_minor_start = raw_sample_data["label_hop_offsets"][index * fanout_length]
        ix_end = (index + 1) * fanout_length
        if ix_end == raw_sample_data["label_hop_offsets"].numel():
            major_minor_end = raw_sample_data["majors"].numel()
        else:
            major_minor_end = raw_sample_data["label_hop_offsets"][ix_end]

        majors = raw_sample_data["majors"][major_minor_start:major_minor_end]
        minors = raw_sample_data["minors"][major_minor_start:major_minor_end]
        edge_id = raw_sample_data["edge_id"][major_minor_start:major_minor_end]

        renumber_map_start = raw_sample_data["renumber_map_offsets"][index]
        renumber_map_end = raw_sample_data["renumber_map_offsets"][index + 1]
        renumber_map = raw_sample_data["map"][renumber_map_start:renumber_map_end]

        num_sampled_edges = (
            raw_sample_data["label_hop_offsets"][
                index * fanout_length : (index + 1) * fanout_length + 1
            ]
            .diff()
            .cpu()
        )

        num_seeds = (majors[: num_sampled_edges[0]].max() + 1).reshape((1,)).cpu()
        num_sampled_nodes_hops = torch.tensor(
            [
                minors[: num_sampled_edges[:i].sum()].max() + 1
                for i in range(1, fanout_length + 1)
            ],
            device="cpu",
        )
        num_sampled_nodes = torch.concat(
            [num_seeds, num_sampled_nodes_hops.diff(prepend=num_seeds)]
        )

        input_index = raw_sample_data["input_index"][
            raw_sample_data["input_offsets"][index] : raw_sample_data["input_offsets"][
                index + 1
            ]
        ]

        edge_inverse = (
            (
                raw_sample_data["edge_inverse"][
                    (raw_sample_data["input_offsets"][index] * 2) : (
                        raw_sample_data["input_offsets"][index + 1] * 2
                    )
                ]
            )
            if "edge_inverse" in raw_sample_data
            else None
        )

        if edge_inverse is None:
            metadata = (input_index, None)
        else:
            edge_inverse = edge_inverse.view(2, -1)
            metadata = (input_index, edge_inverse, None, None)

        return torch_geometric.sampler.SamplerOutput(
            node=renumber_map.cpu(),
            row=minors,
            col=majors,
            edge=edge_id,
            batch=renumber_map[:num_seeds],
            num_sampled_nodes=num_sampled_nodes,
            num_sampled_edges=num_sampled_edges,
            metadata=metadata,
        )

    def _decode(
        self,
        raw_sample_data: Dict[str, Union["torch.Tensor", str, Tuple[str, str, str]]],
        index: int,
    ):
        if "major_offsets" in raw_sample_data:
            return self.__decode_csc(raw_sample_data, index)
        else:
            return self.__decode_coo(raw_sample_data, index)


# ---------------------------------------------------------------------------
# BaseSampler — a9ab8b4 核心：移除 NotImplementedError，接入 HeterogeneousSampleReader
# ---------------------------------------------------------------------------

class BaseSampler:
    """
    PyG-compatible sampler wrapper around DistributedNeighborSampler.

    a9ab8b4 之前：异构图路径抛 NotImplementedError。
    a9ab8b4 之后：通过 HeterogeneousSampleReader 完整实现异构采样。

    Walpurgis 改写：
    - _choose_reader() 私有方法，将 sample_from_nodes/sample_from_edges 中
      重复的「homogeneous vs heterogeneous 分支」提取为单一决策点，
      加 WALPURGIS_DEBUG 打印路径选择原因
    """

    def __init__(
        self,
        sampler: DistributedNeighborSampler,
        data: Tuple[
            "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
        ],
        batch_size: int = 16,
    ):
        self.__sampler = sampler
        self.__feature_store, self.__graph_store = data
        self.__batch_size = batch_size

        _dbg(
            "BaseSampler",
            f"初始化 | batch_size={batch_size} "
            f"graph_store={type(self.__graph_store).__name__}",
        )

    def _choose_reader(self, reader) -> Union[HomogeneousSampleReader, HeterogeneousSampleReader]:
        """
        Walpurgis 新增：根据图类型选择 HomogeneousSampleReader 或
        HeterogeneousSampleReader。

        判断逻辑与上游一致：
        - 单边类型且 src_type == dst_type → 同构
        - 其他 → 异构（a9ab8b4 之前此处直接 raise NotImplementedError）

        WALPURGIS_DEBUG=1 时打印路径选择原因及异构类型信息。
        """
        edge_attrs = self.__graph_store.get_all_edge_attrs()
        is_homogeneous = (
            len(edge_attrs) == 1
            and edge_attrs[0].edge_type[0] == edge_attrs[0].edge_type[2]
        )

        if is_homogeneous:
            _dbg(
                "BaseSampler._choose_reader",
                "选择 HomogeneousSampleReader（单边类型，src_type==dst_type）",
            )
            return HomogeneousSampleReader(reader)
        else:
            edge_types, src_types, dst_types = self.__graph_store._numeric_edge_types
            vertex_types = sorted(self.__graph_store._num_vertices().keys())
            vertex_offsets = self.__graph_store._vertex_offset_array

            _dbg(
                "BaseSampler._choose_reader",
                f"选择 HeterogeneousSampleReader "
                f"num_edge_types={len(edge_types)} "
                f"vertex_types={vertex_types} "
                f"vertex_offsets.shape={tuple(vertex_offsets.shape)}",
            )

            return HeterogeneousSampleReader(
                reader,
                src_types=src_types,
                dst_types=dst_types,
                edge_types=edge_types,
                vertex_types=vertex_types,
                vertex_offsets=vertex_offsets,
            )

    def sample_from_nodes(
        self, index: "torch_geometric.sampler.NodeSamplerInput", **kwargs
    ) -> Iterator[
        Union[
            "torch_geometric.sampler.HeteroSamplerOutput",
            "torch_geometric.sampler.SamplerOutput",
        ]
    ]:
        metadata = (
            {"input_type": index.input_type}
            if index.input_type is not None
            else None
        )

        _dbg(
            "BaseSampler.sample_from_nodes",
            f"input_type={index.input_type!r} "
            f"node.shape={tuple(index.node.shape)} "
            f"metadata={metadata}",
        )

        reader = self.__sampler.sample_from_nodes(
            index.node,
            batch_size=self.__batch_size,
            input_id=index.input_id,
            input_time=index.time,
            metadata=metadata,
            **kwargs,
        )

        return self._choose_reader(reader)

    def sample_from_edges(
        self,
        index: "torch_geometric.sampler.EdgeSamplerInput",
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        **kwargs,
    ) -> Iterator[
        Union[
            "torch_geometric.sampler.HeteroSamplerOutput",
            "torch_geometric.sampler.SamplerOutput",
        ]
    ]:
        from walpurgis.sampler.sampler_utils import neg_sample, neg_cat

        src = index.row
        dst = index.col
        input_id = index.input_id
        input_time = index.time

        node_time = self.__graph_store._get_ntime_func()

        neg_batch_size = 0
        if neg_sampling:
            src_neg, dst_neg = neg_sample(
                self.__graph_store,
                index.row,
                index.col,
                index.input_type,
                self.__batch_size,
                neg_sampling,
                index.time,
                node_time,
            )
            if neg_sampling.is_binary():
                src, _ = neg_cat(src.cuda(), src_neg, self.__batch_size)
            else:
                scu = src.cuda()
                per = torch.randint(
                    0, scu.numel(), (dst_neg.numel(),), device=scu.device
                )
                src, _ = neg_cat(scu, scu[per], self.__batch_size)
            dst, neg_batch_size = neg_cat(dst.cuda(), dst_neg, self.__batch_size)

            if node_time is not None and input_time is not None:
                input_time, _ = neg_cat(
                    input_time.repeat_interleave(int(ceil(neg_sampling.amount))).cuda(),
                    input_time.cuda(),
                    self.__batch_size,
                )

            input_id, _ = neg_cat(
                input_id,
                torch.full(
                    (dst_neg.numel(),), -1, dtype=torch.int64, device=input_id.device
                ),
                self.__batch_size,
            )

        metadata = (
            {"input_type": index.input_type}
            if index.input_type is not None
            else None
        )

        _dbg(
            "BaseSampler.sample_from_edges",
            f"input_type={index.input_type!r} "
            f"src.shape={tuple(src.shape)} dst.shape={tuple(dst.shape)} "
            f"neg_batch_size={neg_batch_size} "
            f"metadata={metadata}",
        )

        reader = self.__sampler.sample_from_edges(
            torch.stack([src, dst]),  # 注意：上游约定 src/dst 顺序与标准相反
            input_id=input_id,
            input_time=input_time,
            input_label=index.label,
            batch_size=self.__batch_size + neg_batch_size,
            metadata=metadata,
            **kwargs,
        )

        return self._choose_reader(reader)
