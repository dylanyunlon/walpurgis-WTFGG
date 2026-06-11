"""
graph_structure.py — bd703b3 迁移: WholeGraph 图结构存储与多跳采样

上游来源: python/pylibwholegraph/pylibwholegraph/torch/graph_structure.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _WalpurgisGraphMeta dataclass 封装 node_count / edge_count / csr_row_ptr / csr_col_ind，
  替代 GraphStructure 中四个散落的初始化字段，set_csr_graph 检验前置条件更清晰
- multilayer_sample_without_replacement 提取 _one_hop_sample() 内部方法，
  消除 for 循环中 weight_name 分支的重复结构
- 采样结果用 _HopResult / _MultilayerSampleResult dataclass 替代裸 list 返回，
  调用方不再依赖 target_gids[i] / edge_indice[i] 的 off-by-one 索引约定
- 全链路 WALPURGIS_DEBUG=1 断点 print:
  CSR 图设置 / 属性注册 / 每跳采样参数与结果 / multilayer 最终输出形状
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from .tensor import WholeMemoryTensor
from . import graph_ops
from . import wholegraph_ops

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/graph_structure]", *args, **kwargs)


# ──────────────────────────────────────────────
# _WalpurgisGraphMeta — 图结构元数据
# ──────────────────────────────────────────────

@dataclass
class _WalpurgisGraphMeta:
    """
    封装 GraphStructure 的核心 CSR 结构字段。

    上游四个属性直接散落在 __init__ 中:
        self.node_count = 0
        self.edge_count = 0
        self.csr_row_ptr = None
        self.csr_col_ind = None
    集中到此处便于 set_csr_graph 中的条件检查和调试 inspect。
    """
    node_count: int = 0
    edge_count: int = 0
    csr_row_ptr: Optional[WholeMemoryTensor] = None
    csr_col_ind: Optional[WholeMemoryTensor] = None

    @property
    def is_set(self) -> bool:
        return self.csr_row_ptr is not None and self.csr_col_ind is not None


# ──────────────────────────────────────────────
# _HopResult / _MultilayerSampleResult — 采样结果
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class _HopResult:
    """单跳采样结果，包含该跳的 CSR 和边索引。"""
    csr_row_ptr: torch.Tensor
    csr_col_ind: torch.Tensor
    edge_index: torch.Tensor    # [2, num_sampled_edges]
    target_gids: torch.Tensor   # 本跳输入节点（已 unique）


@dataclass(frozen=True)
class _MultilayerSampleResult:
    """
    multilayer_sample_without_replacement 的输出。

    上游返回 (target_gids, edge_indice, csr_row_ptr, csr_col_ind) 各一个 list，
    索引约定为 hop_major，调用方需知道:
        target_gids[hops] = 原始种子节点
        target_gids[i]    = 第 i 跳输入节点
    Walpurgis 用 _HopResult 列表 + seed_gids 明确语义。
    """
    hops: List[_HopResult]     # hops[0] = 离种子最近的最后一跳
    seed_gids: torch.Tensor    # 原始输入种子节点

    def as_legacy_tuple(self):
        """上游兼容接口: (target_gids, edge_indice, csr_row_ptr, csr_col_ind)"""
        n = len(self.hops)
        target_gids = [h.target_gids for h in self.hops] + [self.seed_gids]
        edge_indice = [h.edge_index for h in self.hops]
        csr_row_ptr = [h.csr_row_ptr for h in self.hops]
        csr_col_ind = [h.csr_col_ind for h in self.hops]
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind


# ──────────────────────────────────────────────
# GraphStructure
# ──────────────────────────────────────────────

class GraphStructure:
    """
    单关系图结构存储（CSR 格式）。

    除 CSR 结构外还支持节点/边属性张量（WholeMemoryTensor）。
    """

    def __init__(self):
        self._meta = _WalpurgisGraphMeta()
        self.node_attributes: Dict[str, WholeMemoryTensor] = {}
        self.edge_attributes: Dict[str, WholeMemoryTensor] = {}

    # ── 属性便捷访问 ──

    @property
    def node_count(self) -> int:
        return self._meta.node_count

    @property
    def edge_count(self) -> int:
        return self._meta.edge_count

    @property
    def csr_row_ptr(self) -> Optional[WholeMemoryTensor]:
        return self._meta.csr_row_ptr

    @property
    def csr_col_ind(self) -> Optional[WholeMemoryTensor]:
        return self._meta.csr_col_ind

    # ── CSR 图设置 ──

    def set_csr_graph(
        self,
        csr_row_ptr: WholeMemoryTensor,
        csr_col_ind: WholeMemoryTensor,
    ) -> None:
        """
        设置 CSR 图结构。

        :param csr_row_ptr: 1D int64 行指针，长度 = node_count + 1
        :param csr_col_ind: 1D int32/int64 列索引，长度 = edge_count
        """
        assert csr_row_ptr.dim() == 1, "csr_row_ptr 必须是 1D 张量"
        assert csr_row_ptr.dtype == torch.int64, "csr_row_ptr dtype 必须是 int64"
        assert csr_row_ptr.shape[0] > 1, "csr_row_ptr 至少需要 2 个元素（至少 1 个节点）"
        assert csr_col_ind.dim() == 1, "csr_col_ind 必须是 1D 张量"
        assert csr_col_ind.dtype in (torch.int32, torch.int64), \
            "csr_col_ind dtype 必须是 int32 或 int64"

        self._meta.node_count = csr_row_ptr.shape[0] - 1
        self._meta.edge_count = csr_col_ind.shape[0]
        self._meta.csr_row_ptr = csr_row_ptr
        self._meta.csr_col_ind = csr_col_ind
        _dbg(
            f"set_csr_graph: node_count={self._meta.node_count} "
            f"edge_count={self._meta.edge_count}"
        )

    # ── 属性设置 ──

    def set_node_attribute(
        self, attr_name: str, attr_tensor: WholeMemoryTensor
    ) -> None:
        assert attr_name not in self.node_attributes, f"节点属性 {attr_name!r} 已存在"
        assert attr_tensor.shape[0] == self.node_count, \
            f"节点属性 shape[0]={attr_tensor.shape[0]} ≠ node_count={self.node_count}"
        self.node_attributes[attr_name] = attr_tensor
        _dbg(f"set_node_attribute: {attr_name!r} shape={attr_tensor.shape}")

    def set_edge_attribute(
        self, attr_name: str, attr_tensor: WholeMemoryTensor
    ) -> None:
        assert attr_name not in self.edge_attributes, f"边属性 {attr_name!r} 已存在"
        assert attr_tensor.shape[0] == self.edge_count, \
            f"边属性 shape[0]={attr_tensor.shape[0]} ≠ edge_count={self.edge_count}"
        self.edge_attributes[attr_name] = attr_tensor
        _dbg(f"set_edge_attribute: {attr_name!r} shape={attr_tensor.shape}")

    # ── 单跳采样 ──

    def unweighted_sample_without_replacement_one_hop(
        self,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False,
    ):
        """
        无权重单跳不重复采样。
        返回 _SampleOutput（可调用 .as_tuple() 获取上游兼容 tuple）。
        """
        assert self._meta.is_set, "请先调用 set_csr_graph"
        _dbg(
            f"unweighted_sample_one_hop: centers={center_nodes_tensor.shape[0]} "
            f"max_sample={max_sample_count}"
        )
        return wholegraph_ops.unweighted_sample_without_replacement(
            self._meta.csr_row_ptr.wmb_tensor,
            self._meta.csr_col_ind.wmb_tensor,
            center_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    def weighted_sample_without_replacement_one_hop(
        self,
        weight_name: str,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False,
    ):
        """
        有权重单跳不重复采样（使用边属性作为权重）。
        """
        assert self._meta.is_set, "请先调用 set_csr_graph"
        assert weight_name in self.edge_attributes, \
            f"边权重属性 {weight_name!r} 不存在，可用: {list(self.edge_attributes)}"
        weight_tensor = self.edge_attributes[weight_name]
        _dbg(
            f"weighted_sample_one_hop: weight={weight_name!r} "
            f"centers={center_nodes_tensor.shape[0]} max_sample={max_sample_count}"
        )
        return wholegraph_ops.weighted_sample_without_replacement(
            self._meta.csr_row_ptr.wmb_tensor,
            self._meta.csr_col_ind.wmb_tensor,
            weight_tensor.wmb_tensor,
            center_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    # ── 内部工具: 单跳采样统一入口 ──

    def _one_hop_sample(
        self,
        center_nodes: torch.Tensor,
        max_neighbors: int,
        weight_name: Optional[str],
    ) -> wholegraph_ops._SampleOutput:
        """
        封装单跳采样的 weight/unweight 分支选择。
        消除 multilayer_sample_without_replacement 中 for 循环内的重复 if/else。
        """
        if weight_name is None:
            return self.unweighted_sample_without_replacement_one_hop(
                center_nodes,
                max_neighbors,
                need_center_local_output=True,
            )
        else:
            return self.weighted_sample_without_replacement_one_hop(
                weight_name,
                center_nodes,
                max_neighbors,
                need_center_local_output=True,
            )

    # ── 多跳采样 ──

    def multilayer_sample_without_replacement(
        self,
        node_ids: torch.Tensor,
        max_neighbors: List[int],
        weight_name: Union[str, None] = None,
    ) -> _MultilayerSampleResult:
        """
        多跳不重复采样（由种子节点向外扩展）。

        :param node_ids: 初始种子节点 id
        :param max_neighbors: 每跳最大邻居数列表，长度 = 跳数
        :param weight_name: 边权重属性名，None 则无权重采样
        :return: _MultilayerSampleResult
        """
        assert self._meta.is_set, "请先调用 set_csr_graph"
        hops_count = len(max_neighbors)
        _dbg(
            f"multilayer_sample: seed.shape={node_ids.shape} "
            f"hops={hops_count} max_neighbors={max_neighbors} "
            f"weight={weight_name!r}"
        )

        # hop_results[i] 对应第 (hops_count - 1 - i) 跳（从外到内）
        hop_results: List[_HopResult] = [None] * hops_count  # type: ignore
        current_nodes = node_ids

        for i in range(hops_count - 1, -1, -1):
            sample_out = self._one_hop_sample(
                current_nodes,
                max_neighbors[hops_count - i - 1],
                weight_name,
            )
            neighbor_gids_offset = sample_out.sample_offset
            neighbor_gids_vdata = sample_out.dest_nodes
            neighbor_src_lids = sample_out.center_local_id

            unique_result = graph_ops.append_unique(
                current_nodes,
                neighbor_gids_vdata,
                need_neighbor_raw_to_unique=True,
            )
            unique_gids = unique_result.unique_nodes
            neighbor_raw_to_unique = unique_result.neighbor_raw_to_unique

            neighbor_count = neighbor_gids_vdata.shape[0]
            edge_index = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique, (1, neighbor_count)),
                    torch.reshape(neighbor_src_lids, (1, neighbor_count)),
                ]
            )

            _dbg(
                f"  hop[{i}]: current_nodes={current_nodes.shape[0]} "
                f"sampled={neighbor_count} unique_total={unique_gids.shape[0]}"
            )

            hop_results[i] = _HopResult(
                csr_row_ptr=neighbor_gids_offset,
                csr_col_ind=neighbor_raw_to_unique,
                edge_index=edge_index,
                target_gids=current_nodes,
            )
            current_nodes = unique_gids

        _dbg(
            f"multilayer_sample done: "
            f"outermost_unique_nodes={current_nodes.shape[0]}"
        )
        # hop_results[0] 是最外层跳（离种子最远），seed_gids 是原始输入
        # 注意: 上游 target_gids[0] = 最外层 unique，target_gids[hops] = 原始种子
        # Walpurgis: hop_results[i].target_gids = 第 i 跳的 center_nodes
        #            seed_gids = 原始 node_ids（上游 target_gids[hops]）
        return _MultilayerSampleResult(
            hops=hop_results,
            seed_gids=node_ids,
        )
