# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — cugraph_dgl/dataloading/dataloader.py 重构
#         （旧 DataLoader/Dask 路径拆出为 DaskDataLoader，此处为全新 DataLoader 鸭子类型）
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「贪安稳就没有自由，要自由就要历些危险。只有这两条路。」
# —— 鲁迅《两地书》
#
# f4ca484 将 DataLoader 彻底重构为鸭子类型（不再继承 torch.utils.data.DataLoader），
# 直接委托给 graph_sampler.sample() 进行采样，不依赖 Dask/BulkSampler。
# 这条路径配合 NeighborSampler.sample() 使用，是"未来"的非 dask API。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. _warn_ignored_args() 私有方法 — 把 __init__ 中
#      4 个独立的 if warnings.warn() 合并，减少重复代码
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - __init__：各忽略参数列表 / batch_size / device
#      - __iter__：委托给 sampler.sample() 前的参数摘要

import os as _os
import sys as _sys
import time as _time
import warnings
from typing import Union, Optional, Dict, Iterator

from walpurgis.utils.imports import import_optional
from walpurgis.graph.typing import TensorType

torch = import_optional("torch")
dgl = import_optional("dgl")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DGL-DL:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


def _cast_to_torch_tensor(t: TensorType) -> "torch.Tensor":
    """将各类数组类型统一转为 torch.Tensor。"""
    if isinstance(t, torch.Tensor):
        return t
    try:
        import cupy as cp, cudf
        if isinstance(t, (cp.ndarray, cudf.Series)):
            return torch.as_tensor(t, device="cuda")
    except ImportError:
        pass
    try:
        import pandas as pd, numpy as np
        if isinstance(t, (pd.Series, np.ndarray)):
            return torch.as_tensor(t, device="cpu")
    except ImportError:
        pass
    return torch.as_tensor(t)


class DataLoader:
    """
    DGL DataLoader 的鸭子类型版本（非 dask，非继承 torch DataLoader）。

    f4ca484 重构：替代原来继承 torch.utils.data.DataLoader 的实现，
    通过委托 graph_sampler.sample() 直接产出 mini-batch，
    不再依赖 BulkSampler/Dask。

    适用于配合 walpurgis.sampler.dgl_neighbor_sampler.NeighborSampler 使用。

    Walpurgis 改写：
    - _warn_ignored_args() 私有方法合并 4 个独立 warnings.warn
    - 全链路断点覆盖
    """

    def __init__(
        self,
        graph: "walpurgis.graph.Graph",
        indices: TensorType,
        graph_sampler: "walpurgis.sampler.dgl_sampler.Sampler",
        device: Union[int, str, "torch.device"] = None,
        use_ddp: bool = False,
        ddp_seed: int = 0,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        use_prefetch_thread: Optional[bool] = None,
        use_alternate_streams: Optional[bool] = None,
        pin_prefetcher: Optional[bool] = None,
        use_uva: bool = False,
        gpu_cache: Optional[Dict[str, Dict[str, int]]] = None,
        output_format: str = "dgl.Block",
        **kwargs,
    ):
        """
        Parameters
        ----------
        graph : walpurgis.graph.Graph
            待采样的图。
        indices : TensorType
            种子节点 ID。
        graph_sampler : Sampler
            采样器。
        device : optional
            输出设备（当前被忽略）。
        use_ddp : bool (default=False)
            DDP 模式下是否自动分割种子节点。
        ddp_seed : int (default=0)
            DDP 模式下的 shuffle 种子。
        batch_size : int (default=1)
            批大小。
        drop_last : bool (default=False)
            是否丢弃最后不完整批次。
        shuffle : bool (default=False)
            是否随机打乱。
        use_prefetch_thread : bool, optional
            忽略（cuGraph-DGL 不支持）。
        use_alternate_streams : bool, optional
            忽略（cuGraph-DGL 不支持）。
        pin_prefetcher : bool, optional
            忽略（cuGraph-DGL 不支持）。
        use_uva : bool (default=False)
            忽略（cuGraph-DGL 不支持）。
        gpu_cache : dict, optional
            HugeCTR GPU 缓存配置（不支持，传入时报错）。
        output_format : str (default='dgl.Block')
            输出格式（'dgl.Block' 或 'cugraph_dgl.nn.SparseGraph'）。
        """
        self._warn_ignored_args(
            use_uva=use_uva,
            use_prefetch_thread=use_prefetch_thread,
            use_alternate_streams=use_alternate_streams,
            pin_prefetcher=pin_prefetcher,
        )

        if gpu_cache:
            raise ValueError(
                "[Walpurgis:DataLoader] HugeCTR GPU 缓存（gpu_cache）不被 cuGraph-DGL 支持。\n"
                "如需 GPU 特征缓存，请考虑使用 walpurgis.graph.Graph 的 WholeGraph 存储。"
            )

        indices_t = _cast_to_torch_tensor(indices)

        self.__dataset = dgl.dataloading.create_tensorized_dataset(
            indices_t,
            batch_size,
            drop_last,
            use_ddp,
            ddp_seed,
            shuffle,
            kwargs.get("persistent_workers", False),
        )

        _dbg(
            "__init__",
            f"batch_size={batch_size} device={device!r} "
            f"use_ddp={use_ddp} output_format={output_format!r} "
            f"indices.shape={tuple(indices_t.shape)}",
        )

        self.__output_format = output_format
        self.__sampler = graph_sampler
        self.__batch_size = batch_size
        self.__graph = graph
        self.__device = device

    @staticmethod
    def _warn_ignored_args(**kwargs) -> None:
        """
        Walpurgis 改写：统一发出被忽略参数的警告。
        原版对每个参数各自 if/warnings.warn，此处合并。
        """
        ignored_names = {
            "use_uva": "'use_uva' 参数被 cuGraph-DGL 忽略。",
            "use_prefetch_thread": "'use_prefetch_thread' 参数被 cuGraph-DGL 忽略。",
            "use_alternate_streams": "'use_alternate_streams' 参数被 cuGraph-DGL 忽略。",
            "pin_prefetcher": "'pin_prefetcher' 参数被 cuGraph-DGL 忽略。",
        }
        for name, msg in ignored_names.items():
            if kwargs.get(name):
                warnings.warn(f"[Walpurgis:DataLoader] {msg}")

    @property
    def dataset(
        self,
    ) -> Union[
        "dgl.dataloading.dataloader.TensorizedDataset",
        "dgl.dataloading.dataloader.DDPTensorizedDataset",
    ]:
        return self.__dataset

    def __iter__(self) -> Iterator:
        _dbg(
            "__iter__",
            f"委托 {type(self.__sampler).__name__}.sample() "
            f"batch_size={self.__batch_size} "
            f"output_format={self.__output_format!r}",
        )
        # TODO: 移至正确的设备 (rapidsai/cugraph-gnn#11)
        return self.__sampler.sample(
            self.__graph,
            self.__dataset,
            batch_size=self.__batch_size,
        )
