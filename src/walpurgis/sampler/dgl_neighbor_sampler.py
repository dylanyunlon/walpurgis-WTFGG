# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — cugraph_dgl/dataloading/neighbor_sampler.py 重构
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「穷人的孩子，蚤熟世事，也许倒不是幸福。」 —— 鲁迅《故乡》
#
# f4ca484 对 NeighborSampler 做了大幅扩展：
# - 继承自新增的 Sampler 基类
# - 新增大量可选参数（prob/mask/prefetch_*/output_device/fused/sparse_format/output_format）
# - 新增 sample() 方法，将采样逻辑移出 DataLoader，
#   使用 UniformNeighborSampler + DistSampleWriter 完成实际采样
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. _build_sampler_kwargs() 私有方法 — 把 sample() 中
#      kwargs pop/build 逻辑独立，便于 DEBUG 打印最终传入 UniformNeighborSampler 的参数
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - __init__ 参数摘要（fanouts/edge_dir/replace/sparse_format/output_format）
#      - sample() 入口：图类型/indices 维度/batch_size
#      - _build_sampler_kwargs：实际传入 UniformNeighborSampler 的 kwargs
#      - 同构路径 sample_from_nodes 后的 reader 类型

import os as _os
import sys as _sys
import time as _time
import warnings
import tempfile
from typing import Sequence, Optional, Union, List, Tuple, Iterator

from walpurgis.utils.imports import import_optional
from walpurgis.sampler.dgl_sampler import Sampler, HomogeneousSampleReader
from walpurgis.graph.typing import DGLSamplerOutput

torch = import_optional("torch")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DGL-NEIGHBOR-SAMPLER:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


class NeighborSampler(Sampler):
    """
    多层 GNN 邻居采样器（DGL 架构）。

    f4ca484 重构：
    - 继承自 Sampler 基类
    - 新增 sample() 方法，直接驱动 UniformNeighborSampler + DistSampleWriter
    - 新增多个可选参数以保持与 DGL NeighborSampler API 兼容

    Walpurgis 改写：
    - _build_sampler_kwargs() 私有方法分离 kwargs 处理逻辑
    - 全链路断点覆盖
    """

    def __init__(
        self,
        fanouts_per_layer: Sequence[int],
        edge_dir: str = "in",
        replace: bool = False,
        prob: Optional[str] = None,
        mask: Optional[str] = None,
        prefetch_node_feats: Optional[Union[List[str], dict]] = None,
        prefetch_edge_feats: Optional[Union[List[str], dict]] = None,
        prefetch_labels: Optional[Union[List[str], dict]] = None,
        output_device: Optional[Union["torch.device", int, str]] = None,
        fused: Optional[bool] = None,
        sparse_format: str = "csc",
        output_format: str = "dgl.Block",
        **kwargs,
    ):
        """
        Parameters
        ----------
        fanouts_per_layer : Sequence[int]
            每层采样的邻居数量。
        edge_dir : str (default='in')
            边遍历方向（'in' 或 'out'）。
        replace : bool (default=False)
            是否有放回采样。
        prob : str, optional
            概率采样的边特征名（当前不支持）。
        mask : str, optional
            边掩码的特征名（当前不支持）。
        prefetch_node_feats : optional
            预取节点特征（当前被 cuGraph-DGL 忽略）。
        prefetch_edge_feats : optional
            预取边特征（当前被 cuGraph-DGL 忽略）。
        prefetch_labels : optional
            预取标签（当前被 cuGraph-DGL 忽略）。
        output_device : optional
            输出设备（当前不影响行为）。
        fused : bool, optional
            融合采样（当前被 cuGraph-DGL 忽略）。
        sparse_format : str (default='csc')
            输出稀疏格式（当前仅支持 'csc'）。
        output_format : str (default='dgl.Block')
            输出块格式（'dgl.Block' 或 'cugraph_dgl.nn.SparseGraph'）。
        **kwargs
            传递给底层 UniformNeighborSampler 和 DistSampleWriter 的参数
            （directory/batches_per_partition/format/local_seeds_per_call）。
        """
        if mask:
            raise NotImplementedError(
                "[Walpurgis:NeighborSampler] 边掩码（mask）当前不支持。"
            )
        if prob:
            raise NotImplementedError(
                "[Walpurgis:NeighborSampler] 概率采样（prob）当前不支持。"
            )
        if prefetch_edge_feats:
            warnings.warn("[Walpurgis:NeighborSampler] 'prefetch_edge_feats' 被忽略。")
        if prefetch_node_feats:
            warnings.warn("[Walpurgis:NeighborSampler] 'prefetch_node_feats' 被忽略。")
        if prefetch_labels:
            warnings.warn("[Walpurgis:NeighborSampler] 'prefetch_labels' 被忽略。")
        if fused:
            warnings.warn("[Walpurgis:NeighborSampler] 'fused' 被忽略。")

        self.fanouts = fanouts_per_layer
        reverse_fanouts = list(fanouts_per_layer)
        reverse_fanouts.reverse()
        self._reversed_fanout_vals = reverse_fanouts

        self.edge_dir = edge_dir
        self.replace = replace
        self.__kwargs = kwargs

        _dbg(
            "__init__",
            f"fanouts={fanouts_per_layer} edge_dir={edge_dir!r} "
            f"replace={replace} sparse_format={sparse_format!r} "
            f"output_format={output_format!r}",
        )

        super().__init__(
            sparse_format=sparse_format,
            output_format=output_format,
        )

    def _build_sampler_kwargs(self, kwargs: dict) -> Tuple[str, dict]:
        """
        Walpurgis 改写：从 kwargs 提取 directory/batches_per_partition/format，
        返回 (directory, uniform_sampler_kwargs)。
        原版在 sample() 中内联处理，此处独立以便 DEBUG 打印实际传入参数。
        """
        directory = kwargs.pop("directory", None)
        if directory is None:
            warnings.warn(
                "[Walpurgis:NeighborSampler] 建议设置 directory 参数以存储采样结果。"
            )
            self._tempdir = tempfile.TemporaryDirectory()
            directory = self._tempdir.name

        writer_kwargs = {
            "batches_per_partition": kwargs.pop("batches_per_partition", 256),
            "format": kwargs.pop("format", "parquet"),
        }

        _dbg(
            "_build_sampler_kwargs",
            f"directory={directory!r} "
            f"batches_per_partition={writer_kwargs['batches_per_partition']} "
            f"format={writer_kwargs['format']!r} "
            f"remaining_kwargs={list(kwargs.keys())}",
        )

        return directory, writer_kwargs, kwargs

    def sample(
        self,
        g: "walpurgis.graph.Graph",
        indices: Iterator["torch.Tensor"],
        batch_size: int = 1,
    ) -> Iterator[DGLSamplerOutput]:
        """
        对图进行邻居采样。

        Parameters
        ----------
        g : walpurgis.graph.Graph
            待采样的图。
        indices : Iterator[torch.Tensor]
            种子节点 ID 批次迭代器。
        batch_size : int (default=1)
            每批种子节点数量。

        Returns
        -------
        Iterator[DGLSamplerOutput]
            (input_nodes, output_nodes, blocks) 迭代器。
        """
        from cugraph.gnn import UniformNeighborSampler, DistSampleWriter

        kwargs = dict(**self.__kwargs)
        directory, writer_kwargs, remaining_kwargs = self._build_sampler_kwargs(kwargs)

        _dbg(
            "sample",
            f"is_homogeneous={g.is_homogeneous} "
            f"batch_size={batch_size} edge_dir={self.edge_dir!r}",
        )

        writer = DistSampleWriter(
            directory=directory,
            batches_per_partition=writer_kwargs["batches_per_partition"],
            format=writer_kwargs["format"],
        )

        ds = UniformNeighborSampler(
            g._graph(self.edge_dir),
            writer,
            compression="CSR",
            fanout=self._reversed_fanout_vals,
            prior_sources_behavior="carryover",
            deduplicate_sources=True,
            compress_per_hop=True,
            with_replacement=self.replace,
            **remaining_kwargs,
        )

        if g.is_homogeneous:
            indices_t = torch.concat(list(indices))
            _dbg(
                "sample",
                f"同构路径 indices.shape={tuple(indices_t.shape)}",
            )
            ds.sample_from_nodes(indices_t, batch_size=batch_size)
            return HomogeneousSampleReader(
                ds.get_reader(),
                self.output_format,
                self.edge_dir,
            )

        raise ValueError(
            "[Walpurgis:NeighborSampler] 非 dask API 当前不支持异构图采样。\n"
            "请使用 DaskDataLoader + CuGraphStorage 以支持异构图。"
        )
