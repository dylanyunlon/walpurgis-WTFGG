# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/dataloading/sampler.py
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「时间就是性命。无端的空耗别人的时间，其实是无异于谋财害命的。」
# —— 鲁迅《且介亭杂文·门外文谈》
#
# f4ca484 引入了 cugraph_dgl/dataloading/sampler.py，定义了：
# - SampleReader：cuGraph 分布式采样器输出的迭代器基类
# - HomogeneousSampleReader：同构图输出处理
# - Sampler：所有 cugraph-DGL 采样器的基类
#
# 与现有 walpurgis/sampler/sampler.py（PyG 架构）不同，
# 此文件是 DGL 架构下的对应实现，故命名为 dgl_sampler.py。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. _next_batch() 私有方法 — 把 SampleReader.__next__ 中
#      "加载新分区 → 解码 → 返回单样本"的三段式逻辑独立，
#      让 __next__ 只做调度
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - SampleReader.__next__ 何时触发新分区加载
#      - HomogeneousSampleReader._decode_all CSC/COO 分支选择
#      - Sampler.__init__ sparse_format/output_format 参数
#      - Sampler.sample 未实现时的调用栈提示

import os as _os
import sys as _sys
import time as _time
from typing import Iterator, Dict, Tuple, List, Union

from walpurgis.utils.imports import import_optional
from walpurgis.tensor.sparse_graph import SparseGraph
from walpurgis.graph.typing import DGLSamplerOutput
from walpurgis.sampler.sampling_csc_helpers import (
    create_homogeneous_sampled_graphs_from_dataframe_csc,
)

torch = import_optional("torch")
dgl = import_optional("dgl")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DGL-SAMPLER:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 延迟导入：避免无 GPU 环境崩溃
# ---------------------------------------------------------------------------

def _get_dist_sample_reader():
    """延迟获取 cugraph.gnn.DistSampleReader。"""
    from cugraph.gnn import DistSampleReader
    return DistSampleReader


# ---------------------------------------------------------------------------
# SampleReader — cuGraph 分布式采样器输出迭代器基类
# ---------------------------------------------------------------------------

class SampleReader:
    """
    cuGraph 分布式采样器输出迭代器基类。

    f4ca484 新增：为 DGL 架构提供通用的分区读取→解码→输出框架。
    Walpurgis 改写：_next_batch() 私有方法分离加载与解码职责。
    """

    def __init__(
        self,
        base_reader: "DistSampleReader",
        output_format: str = "dgl.Block",
    ):
        """
        Parameters
        ----------
        base_reader : DistSampleReader
            cuGraph 分布式采样器的底层读取器。
        output_format : str (default='dgl.Block')
            输出块格式：'dgl.Block' 或 'cugraph_dgl.nn.SparseGraph'。
        """
        _dbg("SampleReader.__init__", f"output_format={output_format!r}")
        self.__output_format = output_format
        self.__base_reader = base_reader
        self.__num_samples_remaining = 0
        self.__index = 0

    @property
    def output_format(self) -> str:
        return self.__output_format

    def _next_batch(self) -> None:
        """
        Walpurgis 改写：从底层 reader 加载下一个分区并解码全部样本。
        原版将此逻辑内联于 __next__，难以单独 DEBUG。
        """
        raw_sample_data, start_inclusive, end_inclusive = next(self.__base_reader)

        _dbg(
            "SampleReader._next_batch",
            f"加载新分区 samples=[{start_inclusive}, {end_inclusive}] "
            f"keys={list(raw_sample_data.keys()) if isinstance(raw_sample_data, dict) else type(raw_sample_data).__name__}",
        )

        self.__decoded_samples = self._decode_all(raw_sample_data)
        self.__num_samples_remaining = end_inclusive - start_inclusive + 1
        self.__index = 0

    def __next__(self) -> DGLSamplerOutput:
        if self.__num_samples_remaining == 0:
            self._next_batch()

        out = self.__decoded_samples[self.__index]
        self.__index += 1
        self.__num_samples_remaining -= 1
        return out

    def _decode_all(self, raw_sample_data) -> List[DGLSamplerOutput]:
        raise NotImplementedError(
            "[Walpurgis:SampleReader] _decode_all 必须由子类实现。"
        )

    def __iter__(self) -> "SampleReader":
        return self


# ---------------------------------------------------------------------------
# HomogeneousSampleReader — 同构图输出处理
# ---------------------------------------------------------------------------

class HomogeneousSampleReader(SampleReader):
    """
    处理 cuGraph 分布式采样器同构图输出的 SampleReader 子类。

    f4ca484 新增：支持 CSC 和 COO 两种压缩格式的输出解码。
    Walpurgis 改写：_decode_all 统一入口替代原版两个独立 __decode_* 方法。
    """

    def __init__(
        self,
        base_reader: "DistSampleReader",
        output_format: str = "dgl.Block",
        edge_dir: str = "in",
    ):
        """
        Parameters
        ----------
        base_reader : DistSampleReader
            底层读取器。
        output_format : str (default='dgl.Block')
            输出块格式。
        edge_dir : str (default='in')
            采样方向（'in' 或 'out'）。
        """
        _dbg(
            "HomogeneousSampleReader.__init__",
            f"output_format={output_format!r} edge_dir={edge_dir!r}",
        )
        self.__edge_dir = edge_dir
        super().__init__(base_reader, output_format=output_format)

    def __decode_csc(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        """解码 CSC 压缩格式的采样输出。"""
        # 复用 sampling_csc_helpers 中的工具函数
        from walpurgis.sampler.sampling_csc_helpers import (
            _process_sampled_tensors_csc,
            _create_homogeneous_sparse_graphs_from_csc,
            _create_homogeneous_blocks_from_csc,
        )

        _dbg("HomogeneousSampleReader.__decode_csc", "使用 CSC 格式解码")

        processed = _process_sampled_tensors_csc(raw_sample_data)

        if self.output_format == "cugraph_dgl.nn.SparseGraph":
            return _create_homogeneous_sparse_graphs_from_csc(*processed)
        else:
            return _create_homogeneous_blocks_from_csc(*processed)

    def __decode_coo(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        """COO 格式在非 dask API 中暂不支持。"""
        raise NotImplementedError(
            "[Walpurgis:HomogeneousSampleReader] "
            "COO 格式在非 dask API 中暂不支持，请使用 CSC 格式。"
        )

    def _decode_all(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        """
        Walpurgis 改写：统一 CSC/COO 分支入口，加断点打印路径选择。
        """
        if "major_offsets" in raw_sample_data:
            _dbg("HomogeneousSampleReader._decode_all", "检测到 major_offsets → CSC 分支")
            return self.__decode_csc(raw_sample_data)
        else:
            _dbg("HomogeneousSampleReader._decode_all", "未检测到 major_offsets → COO 分支")
            return self.__decode_coo(raw_sample_data)


# ---------------------------------------------------------------------------
# Sampler — 所有 cugraph-DGL 采样器的基类
# ---------------------------------------------------------------------------

class Sampler:
    """
    cugraph-DGL 采样器基类。

    f4ca484 新增：为 NeighborSampler 等提供统一的 sparse_format/output_format
    参数管理，以及强制子类实现 sample() 的契约。

    Walpurgis 改写：__init__ 加断点 + sample() 提供明确的错误信息含调用栈提示。
    """

    def __init__(
        self,
        sparse_format: str = "csc",
        output_format: str = "dgl.Block",
    ):
        """
        Parameters
        ----------
        sparse_format : str (default='csc')
            采样输出的稀疏格式，目前仅支持 'csc'。
        output_format : str (default='dgl.Block')
            输出块格式：'dgl.Block' 或 'cugraph_dgl.nn.SparseGraph'。
        """
        if sparse_format != "csc":
            raise ValueError(
                f"[Walpurgis:Sampler] 当前仅支持 CSC 格式，实际收到：{sparse_format!r}"
            )

        _dbg(
            "Sampler.__init__",
            f"sparse_format={sparse_format!r} output_format={output_format!r}",
        )

        self.__output_format = output_format
        self.__sparse_format = sparse_format

    @property
    def output_format(self) -> str:
        return self.__output_format

    @property
    def sparse_format(self) -> str:
        return self.__sparse_format

    def sample(
        self,
        g: "walpurgis.graph.Graph",
        indices: Iterator["torch.Tensor"],
        batch_size: int = 1,
    ) -> Iterator[
        Tuple["torch.Tensor", "torch.Tensor", List[Union[SparseGraph, "dgl.Block"]]]
    ]:
        """
        对图进行采样（子类必须实现）。

        Parameters
        ----------
        g : walpurgis.graph.Graph
            待采样的图。
        indices : Iterator[torch.Tensor]
            种子节点 ID。
        batch_size : int (default=1)
            每批种子节点数量。

        Returns
        -------
        Iterator[DGLSamplerOutput]
            (input_nodes, output_nodes, blocks) 迭代器。
        """
        raise NotImplementedError(
            "[Walpurgis:Sampler] sample() 必须由子类实现。\n"
            "请检查是否正确实例化了 NeighborSampler 或其他具体采样器，"
            "而非直接实例化 Sampler 基类。"
        )
