# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/dataloading/dask_dataloader.py
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「楼下一个男人病得要死，那间壁的一家唱着留声机；
#   对面是弄孩子。楼上有两个人狂笑；还有打牌声。
#   河中船上有女人哭着她死去的母亲。
#   人类的悲欢并不相通，我只觉得他们吵闹。」
# —— 鲁迅《而已集·小杂感》
#
# f4ca484 将原来的 DataLoader（Dask/BulkSampler 路径）重命名为 DaskDataLoader，
# 同时引入一个新的鸭子类型 DataLoader（非 dask，直接驱动 UniformNeighborSampler）。
# 此文件迁移 DaskDataLoader——基于 Dask 的大规模批量采样路径。
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. _setup_dataset() 私有方法 — 把 __init__ 中
#      "根据图类型创建 HomogenousBulkSamplerDataset / HeterogenousBulkSamplerDataset"
#      的分支逻辑独立，便于 DEBUG 打印数据集类型选择
#   2. _setup_cugraph() 私有方法 — 把 DDP / 单 GPU 两条构建 cugraph 图的路径独立
#   3. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - __init__：sparse_format / batch_size / 图类型
#      - _setup_dataset：选择的 Dataset 类型
#      - _setup_cugraph：DDP/单 GPU 路径 + G 的类型
#      - __iter__：每个 epoch 的输出目录 + BulkSampler 参数
#      - __del__：清理路径

import os as _os
import sys as _sys
import time as _time
import os
import shutil

from walpurgis.utils.imports import import_optional

torch = import_optional("torch")
dgl = import_optional("dgl")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DASK-DL:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


def _dgl_idx_to_cugraph_idx(idx, cugraph_gs):
    """将 DGL 节点 ID 转换为 cuGraph Storage 内部 ID。"""
    if not isinstance(idx, dict):
        if len(cugraph_gs.ntypes) > 1:
            raise dgl.DGLError(
                "[Walpurgis:DaskDataLoader] 异构图必须使用字典形式的索引。"
            )
        return idx
    else:
        return {k: cugraph_gs.dgl_n_id_to_cugraph_id(n, k) for k, n in idx.items()}


def _clean_directory(path: str) -> None:
    """删除文件或目录（含子目录）。"""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def get_batch_id_series(n_output_rows: int, batch_size: int):
    """生成与行数匹配的批次 ID 序列（cudf.Series）。"""
    import cupy as cp
    import cudf

    num_batches = (n_output_rows + batch_size - 1) // batch_size
    _dbg("get_batch_id_series", f"n_output_rows={n_output_rows} num_batches={num_batches}")
    batch_ar = cp.arange(0, num_batches).repeat(batch_size)
    batch_ar = batch_ar[0:n_output_rows].astype(cp.int32)
    return cudf.Series(batch_ar)


def create_batch_df(dataset: "torch.Tensor"):
    """将种子节点数据集转换为 BulkSampler 所需的 cudf.DataFrame。"""
    import cupy as cp
    import cudf

    batch_id_ls = []
    indices_ls = []
    for batch_id, b_indices in enumerate(dataset):
        if isinstance(b_indices, dict):
            b_indices = torch.cat(list(b_indices.values()))
        batch_id_ar = cp.full(shape=len(b_indices), fill_value=batch_id, dtype=cp.int32)
        batch_id_ls.append(batch_id_ar)
        indices_ls.append(b_indices)

    batch_id_ar = cp.concatenate(batch_id_ls)
    indices_ar = cp.asarray(torch.concat(indices_ls))
    return cudf.DataFrame({"start": indices_ar, "batch_id": batch_id_ar})


class DaskDataLoader(torch.utils.data.DataLoader):
    """
    基于 Dask/BulkSampler 的 DGL 图采样 DataLoader。

    f4ca484 重命名：原 DataLoader（Dask 路径）→ DaskDataLoader。
    适用于大规模、基于 CuGraphStorage 的批量采样场景。

    Walpurgis 改写：
    - _setup_dataset() / _setup_cugraph() 私有方法分离初始化逻辑
    - 全链路断点覆盖
    """

    def __init__(
        self,
        graph,
        indices: "torch.Tensor",
        graph_sampler,
        sampling_output_dir: str,
        batches_per_partition: int = 50,
        seeds_per_call: int = 200_000,
        device: "torch.device" = None,
        use_ddp: bool = False,
        ddp_seed: int = 0,
        batch_size: int = 1024,
        drop_last: bool = False,
        shuffle: bool = False,
        sparse_format: str = "coo",
        **kwargs,
    ):
        """
        Parameters
        ----------
        graph : CuGraphStorage
            图存储对象。
        indices : Tensor or dict[ntype, Tensor]
            种子节点索引。
        graph_sampler : NeighborSampler
            子图采样器。
        sampling_output_dir : str
            采样结果输出目录。
        batches_per_partition : int (default=50)
            每分区批次数。
        seeds_per_call : int (default=200_000)
            每次采样的种子数。
        device : torch.device, optional
            结果张量设备。
        use_ddp : bool (default=False)
            是否使用 DistributedDataParallel。
        ddp_seed : int (default=0)
            DDP shuffle 种子。
        batch_size : int (default=1024)
            批大小。
        drop_last : bool (default=False)
            是否丢弃最后不完整批次。
        shuffle : bool (default=False)
            是否随机打乱种子节点顺序。
        sparse_format : str (default='coo')
            输出稀疏格式（'coo' 或 'csc'）。
        """
        if sparse_format not in ("coo", "csc"):
            raise ValueError(
                f"[Walpurgis:DaskDataLoader] sparse_format 须为 'coo' 或 'csc'，"
                f"实际收到：{sparse_format!r}"
            )

        _dbg(
            "__init__",
            f"sparse_format={sparse_format!r} batch_size={batch_size} "
            f"use_ddp={use_ddp} ntypes={graph.ntypes}",
        )

        self.sparse_format = sparse_format
        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        worker_init_fn = dgl.dataloading.WorkerInitWrapper(
            kwargs.get("worker_init_fn", None)
        )
        self.other_storages = {}
        self.epoch_number = 0
        self._batch_size = batch_size
        self._sampling_output_dir = sampling_output_dir
        self._batches_per_partition = batches_per_partition
        self._seeds_per_call = seeds_per_call
        self._rank = None

        indices = _dgl_idx_to_cugraph_idx(indices, graph)

        self.tensorized_indices_ds = dgl.dataloading.create_tensorized_dataset(
            indices,
            batch_size,
            drop_last,
            use_ddp,
            ddp_seed,
            shuffle,
            kwargs.get("persistent_workers", False),
        )

        self.cugraph_dgl_dataset = self._setup_dataset(graph)
        G = self._setup_cugraph(graph, graph_sampler, use_ddp)

        self._cugraph_graph = G
        super().__init__(
            self.cugraph_dgl_dataset,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: x,
            **kwargs,
        )

    def _setup_dataset(self, graph):
        """
        Walpurgis 改写：根据图类型选择合适的 BulkSamplerDataset。
        原版内联于 __init__，此处独立以便断点观察。
        """
        from cugraph_dgl.dataloading import (
            HomogenousBulkSamplerDataset,
            HeterogenousBulkSamplerDataset,
        )

        if len(graph.ntypes) <= 1:
            _dbg("_setup_dataset", "选择 HomogenousBulkSamplerDataset")
            return HomogenousBulkSamplerDataset(
                total_number_of_nodes=graph.total_number_of_nodes,
                edge_dir=self.graph_sampler.edge_dir,
                sparse_format=self.sparse_format,
            )
        else:
            _dbg("_setup_dataset", f"选择 HeterogenousBulkSamplerDataset ntypes={graph.ntypes}")
            etype_id_to_etype_str_dict = {v: k for k, v in graph._etype_id_dict.items()}
            return HeterogenousBulkSamplerDataset(
                num_nodes_dict=graph.num_nodes_dict,
                etype_id_dict=etype_id_to_etype_str_dict,
                etype_offset_dict=graph._etype_offset_d,
                ntype_offset_dict=graph._ntype_offset_d,
                edge_dir=self.graph_sampler.edge_dir,
            )

    def _setup_cugraph(self, graph, graph_sampler, use_ddp: bool):
        """
        Walpurgis 改写：DDP/单 GPU 两条路径的 cuGraph 图构建独立为方法。
        原版内联于 __init__，此处便于断点观察构建过程。
        """
        from cugraph_dgl.dataloading.utils.extract_graph_helpers import (
            create_cugraph_graph_from_edges_dict,
        )

        if use_ddp:
            from dask.distributed import default_client, Event

            rank = torch.distributed.get_rank()
            client = default_client()
            self._graph_creation_event = Event("cugraph_dgl_load_mg_graph_event")
            if rank == 0:
                G = create_cugraph_graph_from_edges_dict(
                    edges_dict=graph._edges_dict,
                    etype_id_dict=graph._etype_id_dict,
                    edge_dir=graph_sampler.edge_dir,
                )
                client.publish_dataset(cugraph_dgl_mg_graph_ds=G)
                self._graph_creation_event.set()
                _dbg("_setup_cugraph", f"DDP rank=0 发布图数据集")
            else:
                if self._graph_creation_event.wait(timeout=1000):
                    G = client.get_dataset("cugraph_dgl_mg_graph_ds")
                    _dbg("_setup_cugraph", f"DDP rank={rank} 获取图数据集成功")
                else:
                    raise RuntimeError(
                        f"[Walpurgis:DaskDataLoader] rank={rank} 等待图数据集超时。"
                    )
            self._rank = rank
        else:
            rank = 0
            G = create_cugraph_graph_from_edges_dict(
                edges_dict=graph._edges_dict,
                etype_id_dict=graph._etype_id_dict,
                edge_dir=graph_sampler.edge_dir,
            )
            _dbg("_setup_cugraph", "单 GPU 模式，直接构建图")
            self._rank = rank

        return G

    def __iter__(self):
        output_dir = os.path.join(
            self._sampling_output_dir, "epoch_" + str(self.epoch_number)
        )

        _dbg("__iter__", f"epoch={self.epoch_number} output_dir={output_dir!r}")

        from cugraph.gnn import BulkSampler

        kwargs = {}
        if isinstance(self.cugraph_dgl_dataset,
                      __import__("cugraph_dgl.dataloading",
                                 fromlist=["HomogenousBulkSamplerDataset"]).HomogenousBulkSamplerDataset):
            kwargs["deduplicate_sources"] = True
            kwargs["prior_sources_behavior"] = "carryover"
            kwargs["renumber"] = True

            if self.sparse_format == "csc":
                kwargs["compression"] = "CSR"
                kwargs["compress_per_hop"] = True
                kwargs["use_legacy_names"] = False
                kwargs["include_hop_column"] = False
        else:
            kwargs["deduplicate_sources"] = False
            kwargs["prior_sources_behavior"] = None
            kwargs["renumber"] = False

        _dbg("__iter__", f"BulkSampler kwargs={kwargs}")

        bs = BulkSampler(
            output_path=output_dir,
            batch_size=self._batch_size,
            graph=self._cugraph_graph,
            batches_per_partition=self._batches_per_partition,
            seeds_per_call=self._seeds_per_call,
            fanout_vals=self.graph_sampler._reversed_fanout_vals,
            with_replacement=self.graph_sampler.replace,
            **kwargs,
        )

        if self.shuffle:
            self.tensorized_indices_ds.shuffle()

        batch_df = create_batch_df(self.tensorized_indices_ds)
        bs.add_batches(batch_df, start_col_name="start", batch_col_name="batch_id")
        bs.flush()
        self.cugraph_dgl_dataset.set_input_files(input_directory=output_dir)
        self.epoch_number += 1
        return super().__iter__()

    def __del__(self):
        if self.use_ddp:
            torch.distributed.barrier()
        if self._rank == 0:
            if self.use_ddp:
                from dask.distributed import default_client
                client = default_client()
                client.unpublish_dataset("cugraph_dgl_mg_graph_ds")
                self._graph_creation_event.clear()
            _dbg("__del__", f"清理采样目录 {self._sampling_output_dir!r}")
            _clean_directory(self._sampling_output_dir)
