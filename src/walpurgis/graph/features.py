# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/features.py (WholeFeatureStore)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「捣鬼有术，也有效，然而有限，所以以此成大事者，古来无有。」
# —— 鲁迅《华盖集续编·捣鬼心传》
#
# 上游 WholeFeatureStore 把 all_gather、scatter、barrier 全部埋进 __init__，
# 无法观察分布式初始化状态，出错只能靠猜。
# Walpurgis 改写：
#   1. _init_distributed_tensor() 私有方法 — 把 all_gather→scatter→barrier
#      的三段式拆出来，每步可独立断点
#   2. 全链路 WALPURGIS_DEBUG=1 断点，覆盖：
#      - __init__ 参数校验完成后（tensor.shape / memory_type / location）
#      - _init_distributed_tensor rank/world_size/local_size/global_shape
#      - fetch() 入口（indices.shape / device）以及 gather 后 shape

import os as _os
import sys as _sys
import time as _time
import warnings

from walpurgis.utils.imports import import_optional, MissingModule

torch = import_optional("torch")
dgl = import_optional("dgl")
wgth = import_optional("pylibwholegraph.torch")

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-FEATURES:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


class WholeFeatureStore(
    object if isinstance(dgl, MissingModule) else dgl.storages.base.FeatureStorage
):
    """
    分布式特征存储：基于 WholeGraph wholememory 的分布式张量。

    f4ca484 新增：支持将 PyTorch tensor 的各 rank 切片合并为跨进程可访问
    的 wholememory 张量，供 cugraph-DGL Graph 图对象使用。

    Walpurgis 改写：
    - _init_distributed_tensor() 私有方法将分布式初始化逻辑独立出来
    - 全链路 WALPURGIS_DEBUG=1 断点覆盖初始化与 fetch 路径
    """

    def __init__(
        self,
        tensor: "torch.Tensor",
        memory_type: str = "distributed",
        location: str = "cpu",
    ):
        """
        Parameters
        ----------
        tensor : torch.Tensor
            本 rank 持有的张量切片（各 rank 按 rank id 顺序排列）。
        memory_type : str (default='distributed')
            wholememory 内存类型：'distributed'/'chunked'/'continuous'。
        location : str (default='cpu')
            存储位置：'cpu' 或 'cuda'。
        """
        if len(tensor.shape) > 2:
            raise ValueError(
                "[Walpurgis:WholeFeatureStore] 仅支持 1-D 或 2-D 张量。"
            )

        _dbg(
            "__init__",
            f"tensor.shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"memory_type={memory_type!r} location={location!r}",
        )

        self.__wg_comm = wgth.get_global_communicator()
        self.__td = -1 if len(tensor.shape) == 1 else tensor.shape[1]
        self.__wg_tensor = self._init_distributed_tensor(
            tensor, memory_type, location
        )

    # ------------------------------------------------------------------
    # Walpurgis 改写：_init_distributed_tensor — 分布式初始化独立方法
    # ------------------------------------------------------------------

    def _init_distributed_tensor(
        self,
        tensor: "torch.Tensor",
        memory_type: str,
        location: str,
    ):
        """
        执行 all_gather → create_wholememory_tensor → scatter → barrier。
        将原版内联于 __init__ 的三段式逻辑拆出，便于断点观察。
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        ld = torch.tensor(tensor.shape[0], device="cuda", dtype=torch.int64)
        sizes = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(sizes, ld)
        sizes = sizes.cpu()
        total_len = int(sizes.sum())

        global_shape = [
            total_len,
            self.__td if self.__td > 0 else 1,
        ]

        _dbg(
            "_init_distributed_tensor",
            f"rank={rank} world_size={world_size} "
            f"local_size={int(tensor.shape[0])} global_shape={global_shape}",
        )

        # 1D 张量需要临时 reshape 为 2D 才能传给 wholememory
        if self.__td < 0:
            tensor = tensor.reshape((tensor.shape[0], 1))

        wg_tensor = wgth.create_wholememory_tensor(
            self.__wg_comm,
            memory_type,
            location,
            global_shape,
            tensor.dtype,
            [global_shape[1], 1],
        )

        offset = int(sizes[:rank].sum()) if rank > 0 else 0

        _dbg(
            "_init_distributed_tensor",
            f"scatter offset={offset} tensor_shape={tuple(tensor.shape)}",
        )

        wg_tensor.scatter(
            tensor.clone(memory_format=torch.contiguous_format).cuda(),
            torch.arange(
                offset, offset + tensor.shape[0], dtype=torch.int64, device="cuda"
            ).contiguous(),
        )

        self.__wg_comm.barrier()
        return wg_tensor

    def requires_ddp(self) -> bool:
        return True

    def fetch(
        self,
        indices: "torch.Tensor",
        device: "torch.cuda.Device",
        pin_memory: bool = False,
        **kwargs,
    ) -> "torch.Tensor":
        if pin_memory:
            warnings.warn(
                "[Walpurgis:WholeFeatureStore] pin_memory 对 WholeFeatureStore 无效。"
            )

        _dbg(
            "fetch",
            f"indices.shape={tuple(indices.shape)} device={device}",
        )

        t = self.__wg_tensor.gather(
            indices.cuda(),
            force_dtype=self.__wg_tensor.dtype,
        )

        if self.__td < 0:
            t = t.reshape((t.shape[0],))

        _dbg("fetch", f"gathered shape={tuple(t.shape)}")

        return t.to(torch.device(device))
