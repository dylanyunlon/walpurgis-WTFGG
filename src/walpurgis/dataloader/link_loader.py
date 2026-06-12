# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f57ed88
# 原标题: pull in changes from cugraph repo
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「横眉冷对千夫指，俯首甘为孺子牛。」—— 鲁迅
# 上游 LinkLoader 是对 torch_geometric.loader.LinkLoader 的鸭子类型仿制，
# 将边采样入口标准化。f57ed88 首次引入此文件。
# Walpurgis 在保持 API 兼容的前提下，将若干隐式状态提升为可审计的数据类，
# 并增加全链路 WALPURGIS_DEBUG 断点，使 __iter__ 的执行路径可追踪。
#
# Walpurgis 20% 改写要点:
#   1. EdgeSamplerSpec dataclass — 封装 EdgeSamplerInput 构建逻辑，使构造函数更内聚
#   2. NegSamplingSpec dataclass — 将 neg_sampling / neg_sampling_ratio 校验集中
#   3. _validate_link_loader_args() — 独立校验函数，便于单元测试注入
#   4. __iter__ 中新增 _IterState 枚举，区分 perm/drop/slice 三阶段
#   5. 全链路 WALPURGIS_DEBUG=1 断点（6 处）

import os as _os
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union, Tuple, Callable, Optional

from walpurgis.utils.imports import import_optional
import walpurgis.sampler as _wsampler
import walpurgis.data as _wdata

torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        import sys
        print(f"[WALPURGIS_DEBUG][LinkLoader][{tag}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 辅助数据类（Walpurgis 扩展，上游无此抽象）
# ---------------------------------------------------------------------------

class _IterPhase(Enum):
    """__iter__ 执行阶段，用于断点追踪。"""
    PERM_BUILD = auto()
    DROP_LAST  = auto()
    SLICE      = auto()
    SAMPLE     = auto()


@dataclass
class NegSamplingSpec:
    """
    封装负采样参数校验逻辑。
    上游将 neg_sampling_ratio 废弃警告与 cast 散落在 __init__ 中，
    Walpurgis 集中在此处管理。
    """
    neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"]
    neg_sampling_ratio: Optional[Union[int, float]]

    def validate_and_cast(self) -> "torch_geometric.sampler.NegativeSampling":
        """校验并返回规范化后的 NegativeSampling 对象（可为 None）。"""
        if self.neg_sampling_ratio is not None:
            warnings.warn(
                "The 'neg_sampling_ratio' argument is deprecated in PyG"
                " and is not supported in Walpurgis LinkLoader."
            )
        casted = torch_geometric.sampler.NegativeSampling.cast(self.neg_sampling)
        _dbg("NegSamplingSpec", f"cast result={casted}")
        return casted

    def check_edge_label_consistency(
        self,
        edge_label: Optional["torch.Tensor"],
        ns: Optional["torch_geometric.sampler.NegativeSampling"],
    ) -> Optional["torch.Tensor"]:
        """
        执行 PyG LinkLoader 的 edge_label 一致性检查并在必要时修改 label。
        返回可能被 +1 偏移后的 edge_label。
        """
        if ns is None:
            return edge_label
        if ns.is_binary() and edge_label is not None and edge_label.min() == 0:
            _dbg("NegSamplingSpec", "binary neg_sampling: edge_label += 1")
            edge_label = edge_label + 1
        if ns.is_triplet() and edge_label is not None:
            raise ValueError(
                "'edge_label' needs to be undefined for "
                "'triplet'-based negative sampling. Please use "
                "`src_index`, `dst_pos_index` and "
                "`neg_pos_index` of the returned mini-batch "
                "instead to differentiate between positive and negative samples."
            )
        return edge_label


@dataclass
class EdgeSamplerSpec:
    """
    封装 EdgeSamplerInput 的构建参数。
    上游在 __init__ 中直接构造 EdgeSamplerInput，Walpurgis 将其提取为可
    单独测试的数据类，同时使 input_id 默认值策略更透明。
    """
    edge_label_index: Tuple["torch.Tensor", "torch.Tensor"]
    edge_label: Optional["torch.Tensor"]
    edge_label_time: Optional["torch.Tensor"]
    input_type: Optional[str]
    input_id: Optional["torch.Tensor"] = field(default=None)

    def build(self) -> "torch_geometric.sampler.EdgeSamplerInput":
        n_edges = self.edge_label_index[0].numel()
        resolved_id = (
            torch.arange(n_edges, dtype=torch.int64, device="cuda")
            if self.input_id is None
            else self.input_id
        )
        _dbg("EdgeSamplerSpec", f"build: n_edges={n_edges}, id_auto={self.input_id is None}")
        return torch_geometric.sampler.EdgeSamplerInput(
            input_id=resolved_id,
            row=self.edge_label_index[0],
            col=self.edge_label_index[1],
            label=self.edge_label,
            time=self.edge_label_time,
            input_type=self.input_type,
        )


# ---------------------------------------------------------------------------
# 参数校验（独立函数，便于子类复用）
# ---------------------------------------------------------------------------

def _validate_link_loader_args(
    data,
    link_sampler,
    edge_label_time,
    filter_per_worker: bool,
    custom_cls,
    transform,
    transform_sampler_output,
) -> None:
    """集中执行 LinkLoader 构造参数的合法性校验，抛出或发出警告。"""
    if not isinstance(data, (list, tuple)) or not isinstance(data[1], _wdata.GraphStore):
        raise NotImplementedError("Currently can't accept non-walpurgis graphs")
    if not isinstance(link_sampler, _wsampler.BaseSampler):
        raise NotImplementedError("Must provide a Walpurgis BaseSampler")
    if edge_label_time is not None:
        raise ValueError("Temporal sampling is currently unsupported")
    if filter_per_worker:
        warnings.warn("filter_per_worker is currently ignored")
    if custom_cls is not None:
        warnings.warn("custom_cls is currently ignored")
    if transform is not None:
        warnings.warn("transform is currently ignored.")
    if transform_sampler_output is not None:
        warnings.warn("transform_sampler_output is currently ignored.")
    _dbg("validate", "args OK")


# ---------------------------------------------------------------------------
# LinkLoader
# ---------------------------------------------------------------------------

class LinkLoader:
    """
    Walpurgis duck-typed version of torch_geometric.loader.LinkLoader.

    从输入边批次中采样子图，调用
    `~walpurgis.sampler.BaseSampler.sample_from_edges`。

    f57ed88 引入原始实现；Walpurgis 将内部状态管理提升为可审计的数据类，
    并在关键路径添加 WALPURGIS_DEBUG 断点。
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
        link_sampler: "_wsampler.BaseSampler",
        edge_label_index: "torch_geometric.typing.InputEdges" = None,
        edge_label: "torch_geometric.typing.OptTensor" = None,
        edge_label_time: "torch_geometric.typing.OptTensor" = None,
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional["torch_geometric.data.HeteroData"] = None,
        input_id: "torch_geometric.typing.OptTensor" = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        _dbg("__init__", f"batch_size={batch_size} shuffle={shuffle} drop_last={drop_last}")

        # 参数校验（集中在独立函数，便于子类 override）
        _validate_link_loader_args(
            data, link_sampler, edge_label_time,
            bool(filter_per_worker), custom_cls, transform, transform_sampler_output,
        )

        # 负采样规范化
        ns_spec = NegSamplingSpec(neg_sampling, neg_sampling_ratio)
        resolved_neg = ns_spec.validate_and_cast()

        # 解析 edge_label_index
        input_type, eli = torch_geometric.loader.utils.get_edge_label_index(
            data, (None, edge_label_index)
        )
        _dbg("__init__", f"input_type={input_type}, eli shape={eli[0].shape}")

        # edge_label 一致性修正
        edge_label = ns_spec.check_edge_label_consistency(edge_label, resolved_neg)

        # 构建 EdgeSamplerInput
        esp = EdgeSamplerSpec(
            edge_label_index=eli,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            input_type=input_type,
            input_id=input_id,
        )
        self.__input_data: "torch_geometric.sampler.EdgeSamplerInput" = esp.build()

        self.__data = data
        self.__link_sampler = link_sampler
        self.__neg_sampling = resolved_neg
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__drop_last = drop_last

    # ------------------------------------------------------------------
    # __iter__
    # ------------------------------------------------------------------

    def __iter__(self):
        # 阶段 1: 构建 perm
        _dbg("__iter__", f"phase={_IterPhase.PERM_BUILD.name} shuffle={self.__shuffle}")
        if self.__shuffle:
            perm = torch.randperm(self.__input_data.row.numel())
        else:
            perm = torch.arange(self.__input_data.row.numel())

        # 阶段 2: drop_last 裁剪
        if self.__drop_last:
            _dbg("__iter__", f"phase={_IterPhase.DROP_LAST.name}")
            d = perm.numel() % self.__batch_size
            if d != 0:
                perm = perm[:-d]

        # 阶段 3: 构建重排后的 EdgeSamplerInput
        _dbg("__iter__", f"phase={_IterPhase.SLICE.name} perm.numel={perm.numel()}")
        input_data = torch_geometric.sampler.EdgeSamplerInput(
            input_id=self.__input_data.input_id[perm],
            row=self.__input_data.row[perm],
            col=self.__input_data.col[perm],
            label=(
                None if self.__input_data.label is None
                else self.__input_data.label[perm]
            ),
            time=(
                None if self.__input_data.time is None
                else self.__input_data.time[perm]
            ),
            input_type=self.__input_data.input_type,
        )

        # 阶段 4: 触发采样
        _dbg("__iter__", f"phase={_IterPhase.SAMPLE.name}")
        return _wsampler.SampleIterator(
            self.__data,
            self.__link_sampler.sample_from_edges(
                input_data,
                neg_sampling=self.__neg_sampling,
            ),
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _selftest_neg_sampling_spec() -> None:
    """快速单元测试 NegSamplingSpec（无 GPU 依赖）。"""
    import traceback

    # T1: None neg_sampling 返回 None
    spec = NegSamplingSpec(None, None)
    # 不调用 validate_and_cast（需要 torch_geometric），仅测试数据类构造
    assert spec.neg_sampling is None
    assert spec.neg_sampling_ratio is None

    # T2: edge_label None → 原样返回
    result = spec.check_edge_label_consistency(None, None)
    assert result is None

    print("[WALPURGIS_SELFTEST][link_loader] NegSamplingSpec: PASS")


def _selftest_iter_phase() -> None:
    """测试 _IterPhase 枚举完整性。"""
    phases = list(_IterPhase)
    assert len(phases) == 4, f"期待 4 个阶段，得到 {len(phases)}"
    print("[WALPURGIS_SELFTEST][link_loader] _IterPhase: PASS")


if __name__ == "__main__":
    _selftest_neg_sampling_spec()
    _selftest_iter_phase()
    print("[WALPURGIS_SELFTEST][link_loader] ALL PASS")
