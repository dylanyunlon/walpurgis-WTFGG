# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f57ed88
# 原标题: pull in changes from cugraph repo
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「猛兽总是独行，牛羊才成群结队。」—— 鲁迅
# 上游 LinkNeighborLoader 是 LinkLoader 的子类，实现 GraphSAGE 风格的
# 邻居采样边加载器。f57ed88 首次引入此文件。
#
# Walpurgis 20% 改写要点:
#   1. SamplerBuildSpec dataclass — 将 BaseSampler 构造逻辑（NeighborSampler / DistSampleWriter
#      参数组装）提取为独立数据类，避免 __init__ 过长，同时使采样器配置可序列化
#   2. SubgraphGuard — 集中管理 subgraph_type / directed / disjoint 三组相互制约的
#      参数校验，上游散落在多个 if 语句中
#   3. compression 枚举化 — 以 CompressionMode 枚举替代裸字符串比较
#   4. build_sampler() 方法 — 与 SubgraphGuard / SamplerBuildSpec 正交，便于子类覆盖
#   5. 全链路 WALPURGIS_DEBUG=1 断点（5 处）

import os as _os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, Optional, Callable, List, Dict

from walpurgis.dataloader.link_loader import LinkLoader, _dbg as _parent_dbg
from walpurgis.sampler import BaseSampler
from walpurgis.utils.imports import import_optional
import walpurgis.data as _wdata

torch_geometric = import_optional("torch_geometric")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        import sys
        print(f"[WALPURGIS_DEBUG][LinkNeighborLoader][{tag}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 辅助类型（Walpurgis 扩展）
# ---------------------------------------------------------------------------

class CompressionMode(str, Enum):
    """
    合法的采样输出压缩格式。
    上游以裸字符串 "CSR" / "COO" 校验，Walpurgis 枚举化使错误更早暴露。
    """
    CSR = "CSR"
    COO = "COO"

    @classmethod
    def resolve(cls, compression: Optional[str]) -> "CompressionMode":
        """None → CSR（默认），其余解析为枚举成员或抛出 ValueError。"""
        if compression is None:
            return cls.CSR
        try:
            return cls(compression)
        except ValueError:
            raise ValueError(
                f"Invalid value for compression '{compression}' "
                f"(expected one of {[m.value for m in cls]})"
            )


@dataclass
class SubgraphGuard:
    """
    封装 subgraph_type / directed / disjoint / temporal_strategy 的相互约束校验。
    上游散落在 __init__ 的 5 个独立 if 块中，Walpurgis 集中为可单元测试的数据类。
    """
    subgraph_type_raw: Union["torch_geometric.typing.SubgraphType", str]
    directed: bool
    disjoint: bool
    temporal_strategy: str
    neighbor_sampler: Optional[object]
    time_attr: Optional[str]
    is_sorted: bool

    def resolve_subgraph_type(self) -> "torch_geometric.sampler.base.SubgraphType":
        st = torch_geometric.sampler.base.SubgraphType(self.subgraph_type_raw)
        if not self.directed:
            warnings.warn(
                "The 'directed' argument is deprecated. "
                "Use subgraph_type='induced' instead."
            )
            st = torch_geometric.sampler.base.SubgraphType.induced
        _dbg("SubgraphGuard", f"resolved subgraph_type={st}")
        return st

    def validate(self, st: "torch_geometric.sampler.base.SubgraphType") -> None:
        if st != torch_geometric.sampler.base.SubgraphType.directional:
            raise ValueError("Only directional subgraphs are currently supported")
        if self.disjoint:
            raise ValueError("Disjoint sampling is currently unsupported")
        if self.temporal_strategy != "uniform":
            warnings.warn("Only the uniform temporal strategy is currently supported")
        if self.neighbor_sampler is not None:
            raise ValueError("Passing a neighbor sampler is currently unsupported")
        if self.time_attr is not None:
            raise ValueError("Temporal sampling is currently unsupported")
        if self.is_sorted:
            warnings.warn("The 'is_sorted' argument is ignored by Walpurgis.")
        _dbg("SubgraphGuard", "validate OK")


@dataclass
class SamplerBuildSpec:
    """
    封装 BaseSampler（内含 NeighborSampler + DistSampleWriter）的构建参数。
    上游在 __init__ 内直接构建，Walpurgis 将其提取为可序列化/可测试的数据类。
    """
    graph_store: "_wdata.GraphStore"
    feature_store: "_wdata.FeatureStore"
    num_neighbors: Union[List[int], Dict]
    compression: CompressionMode
    replace: bool
    local_seeds_per_call: Optional[int]
    weight_attr: Optional[str]
    directory: Optional[str]
    batches_per_partition: int
    fmt: str
    batch_size: int

    def build(self) -> BaseSampler:
        from cugraph.gnn import NeighborSampler, DistSampleWriter

        # 权重属性注入
        if self.weight_attr is not None:
            self.graph_store._set_weight_attr((self.feature_store, self.weight_attr))
            _dbg("SamplerBuildSpec", f"weight_attr set: {self.weight_attr}")

        writer = (
            None if self.directory is None
            else DistSampleWriter(
                directory=self.directory,
                batches_per_partition=self.batches_per_partition,
                format=self.fmt,
            )
        )
        _dbg("SamplerBuildSpec", f"writer={'disk' if writer else 'in-memory'}")

        ns = NeighborSampler(
            self.graph_store._graph,
            writer,
            retain_original_seeds=True,
            fanout=self.num_neighbors,
            prior_sources_behavior="exclude",
            deduplicate_sources=True,
            compression=self.compression.value,
            compress_per_hop=False,
            with_replacement=self.replace,
            local_seeds_per_call=self.local_seeds_per_call,
            biased=(self.weight_attr is not None),
        )
        _dbg("SamplerBuildSpec", f"NeighborSampler built, fanout={self.num_neighbors}")

        return BaseSampler(
            ns,
            (self.feature_store, self.graph_store),
            batch_size=self.batch_size,
        )


# ---------------------------------------------------------------------------
# LinkNeighborLoader
# ---------------------------------------------------------------------------

class LinkNeighborLoader(LinkLoader):
    """
    Walpurgis duck-typed version of torch_geometric.loader.LinkNeighborLoader.

    实现 GraphSAGE 风格的邻居采样边加载器。
    f57ed88 首次引入原始实现；Walpurgis 将采样器构建逻辑提取为
    SamplerBuildSpec，将参数约束提取为 SubgraphGuard。
    """

    def __init__(
        self,
        data: Union[
            "torch_geometric.data.Data",
            "torch_geometric.data.HeteroData",
            Tuple[
                "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
            ],
        ],
        num_neighbors: Union[
            List[int], Dict["torch_geometric.typing.EdgeType", List[int]]
        ],
        edge_label_index: "torch_geometric.typing.InputEdges" = None,
        edge_label: "torch_geometric.typing.OptTensor" = None,
        edge_label_time: "torch_geometric.typing.OptTensor" = None,
        replace: bool = False,
        subgraph_type: Union[
            "torch_geometric.typing.SubgraphType", str
        ] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional["torch_geometric.sampler.NeighborSampler"] = None,
        directed: bool = True,  # Deprecated.
        batch_size: int = 16,
        directory: Optional[str] = None,
        batches_per_partition: int = 256,
        format: str = "parquet",
        compression: Optional[str] = None,
        local_seeds_per_call: Optional[int] = None,
        **kwargs,
    ):
        _dbg("__init__", f"batch_size={batch_size} num_neighbors={num_neighbors}")

        # data 类型校验（与父类校验互补）
        if not isinstance(data, (list, tuple)) or not isinstance(data[1], _wdata.GraphStore):
            raise NotImplementedError("Currently can't accept non-walpurgis graphs")

        # SubgraphGuard：集中校验 subgraph 相关参数
        sg = SubgraphGuard(
            subgraph_type_raw=subgraph_type,
            directed=directed,
            disjoint=disjoint,
            temporal_strategy=temporal_strategy,
            neighbor_sampler=neighbor_sampler,
            time_attr=time_attr,
            is_sorted=is_sorted,
        )
        resolved_st = sg.resolve_subgraph_type()
        sg.validate(resolved_st)

        # CompressionMode 枚举解析
        comp_mode = CompressionMode.resolve(compression)
        _dbg("__init__", f"compression={comp_mode}")

        feature_store, graph_store = data

        # SamplerBuildSpec：封装 BaseSampler 构建
        spec = SamplerBuildSpec(
            graph_store=graph_store,
            feature_store=feature_store,
            num_neighbors=num_neighbors,
            compression=comp_mode,
            replace=replace,
            local_seeds_per_call=local_seeds_per_call,
            weight_attr=weight_attr,
            directory=directory,
            batches_per_partition=batches_per_partition,
            fmt=format,
            batch_size=batch_size,
        )
        sampler = spec.build()
        _dbg("__init__", "BaseSampler ready, calling super().__init__")

        # TODO: add heterogeneous support and pass graph_store._vertex_offsets
        super().__init__(
            (feature_store, graph_store),
            sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            batch_size=batch_size,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _selftest_compression_mode() -> None:
    assert CompressionMode.resolve(None) == CompressionMode.CSR
    assert CompressionMode.resolve("CSR") == CompressionMode.CSR
    assert CompressionMode.resolve("COO") == CompressionMode.COO
    try:
        CompressionMode.resolve("INVALID")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass
    print("[WALPURGIS_SELFTEST][link_neighbor_loader] CompressionMode: PASS")


def _selftest_subgraph_guard_fields() -> None:
    sg = SubgraphGuard(
        subgraph_type_raw="directional",
        directed=True,
        disjoint=False,
        temporal_strategy="uniform",
        neighbor_sampler=None,
        time_attr=None,
        is_sorted=False,
    )
    assert sg.directed is True
    assert sg.disjoint is False
    print("[WALPURGIS_SELFTEST][link_neighbor_loader] SubgraphGuard fields: PASS")


if __name__ == "__main__":
    _selftest_compression_mode()
    _selftest_subgraph_guard_fields()
    print("[WALPURGIS_SELFTEST][link_neighbor_loader] ALL PASS")
