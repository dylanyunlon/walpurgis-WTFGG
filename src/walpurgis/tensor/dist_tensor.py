# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 539d0ad: tensor/dist_tensor.py
# 鲁迅拿法20%改写笔记:
#   原文 DistTensor 像一面镜子——只照出 WholeGraph API 的形状, 自己没有声音。
#   我们给它加了喉咙:
#     1. __init__ 入口断点: 打印 src 类型/shape、backend、device;
#     2. load_from_global_tensor 打印 barrier 前后状态;
#     3. load_from_local_tensor 加 shape/dtype 不匹配的 early-exit 诊断;
#     4. __getitem__/__setitem__ 打印 idx 范围, 抓越界隐患;
#     5. DistEmbedding.__init__ 打印 cache_policy 是否生效;
#     6. _init_from_single_source 对未知扩展名给出具体建议, 而非泛化报错。

import os
import sys
import time
from typing import List, Optional, Union, Literal

import numpy as np

from walpurgis.tensor.utils import (
    copy_host_global_tensor_to_local,
    create_wg_dist_tensor,
    create_wg_dist_tensor_from_files,
)
from walpurgis.utils.imports import import_optional

torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")
pylibwholegraph = import_optional("pylibwholegraph")

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(cls_name: str, tag: str, msg: str) -> None:
    """断点调试打印 (WALPURGIS_DEBUG=1)"""
    if _DEBUG:
        print(
            f"[WALPURGIS-TENSOR:{cls_name}][{time.strftime('%H:%M:%S')}][{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ─── DistTensor ──────────────────────────────────────────────────────────────


class DistTensor:
    """WholeGraph 分布式张量接口 (Walpurgis 版)。

    WholeGraph 把内存看作公地——各进程各取所需, 无须传递副本。
    这个类是那片公地的门卫: 你告诉它"我要多大、什么类型、放哪里",
    它替你和 WholeGraph 谈判。

    Parameters
    ----------
    src : Optional[Union[torch.Tensor, str, List[str]]]
        张量来源: host Tensor / 单文件路径 (.pt/.npy) / 文件列表 (binary)。
        为 None 时创建空张量 (需同时传 shape 和 dtype)。
    shape : Optional[list | tuple]
        张量形状 (1D 或 2D)。当 src 为 None 时必填。
    dtype : Optional[torch.dtype]
        数据类型。当 src 为 None 时必填。
    device : Optional[Literal["cpu", "cuda"]]
        存储位置。"cpu" = host-pinned (UVA), "cuda" = GPU HBM。
    partition_book : Union[List[int], None]
        按 dim-0 的 range partition: partition_book[i] = rank i 的行数。
        None 表示均匀切分。
    backend : Optional[str]
        通信后端。"nccl" (默认) | "vmm"。
    """

    def __init__(
        self,
        src: Optional[Union["torch.Tensor", str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Optional[Union[List[int], None]] = None,
        backend: Optional[str] = "nccl",
        *args,
        **kwargs,
    ):
        self._tensor = None
        self.__device = device

        # 断点: 构造入口
        _dbg(
            "DistTensor",
            "__init__",
            f"src_type={type(src).__name__} "
            f"shape={shape} dtype={dtype} device={device} backend={backend}",
        )

        if src is None:
            if shape is None:
                raise ValueError("src=None 时必须指定 shape。")
            if dtype is None:
                raise ValueError("src=None 时必须指定 dtype。")
            if len(shape) not in [1, 2]:
                raise ValueError(
                    f"shape 必须是 1D 或 2D, 但传入了 {len(shape)}D shape={shape}。"
                )

            self._tensor = create_wg_dist_tensor(
                list(shape), dtype, device, partition_book, backend, *args, **kwargs
            )
            self.__dtype = dtype
            _dbg("DistTensor", "__init__", "空张量创建成功")

        else:
            if isinstance(src, list):
                # 文件列表: 只支持 binary (需要显式 shape/dtype)
                if shape is None or dtype is None:
                    raise ValueError(
                        "从文件列表加载时必须同时提供 shape 和 dtype "
                        "(目前仅支持 binary 格式)。"
                    )
                self._tensor = create_wg_dist_tensor_from_files(
                    src, list(shape), dtype, device, partition_book, backend,
                    *args, **kwargs
                )
                self.__dtype = dtype
                _dbg(
                    "DistTensor",
                    "__init__",
                    f"从 {len(src)} 个文件创建完毕",
                )
            else:
                self._init_from_single_source(
                    src, device, partition_book, backend, *args, **kwargs
                )

    def _init_from_single_source(
        self, src, device, partition_book, backend, *args, **kwargs
    ):
        """从单一来源 (Tensor 或文件) 初始化。

        鲁迅改写: 上游对未知扩展名给 "Unsupported source type",
        此处列出已支持的格式, 告诉用户往哪走。
        """
        _dbg(
            "DistTensor",
            "_init_from_single_source",
            f"src_type={type(src).__name__} "
            f"src_repr={repr(src)[:80]}",
        )

        if isinstance(src, torch.Tensor):
            _dbg("DistTensor", "_init_from_single_source", "路径A: host Tensor")
            self._tensor = create_wg_dist_tensor(
                list(src.shape), src.dtype, device,
                partition_book, backend, *args, **kwargs,
            )
            self.__dtype = src.dtype
            host_tensor = src

        elif isinstance(src, str) and src.endswith(".pt"):
            _dbg("DistTensor", "_init_from_single_source", f"路径B: .pt 文件 {src}")
            host_tensor = torch.load(src, mmap=True)
            self._tensor = create_wg_dist_tensor(
                list(host_tensor.shape), host_tensor.dtype, device,
                partition_book, backend, *args, **kwargs,
            )
            self.__dtype = host_tensor.dtype

        elif isinstance(src, str) and src.endswith(".npy"):
            _dbg("DistTensor", "_init_from_single_source", f"路径C: .npy 文件 {src}")
            host_tensor = torch.from_numpy(np.load(src, mmap_mode="c"))
            self.__dtype = host_tensor.dtype
            self._tensor = create_wg_dist_tensor(
                list(host_tensor.shape), host_tensor.dtype, device,
                partition_book, backend, *args, **kwargs,
            )

        else:
            ext = os.path.splitext(src)[1] if isinstance(src, str) else ""
            raise ValueError(
                f"不支持的 src 类型/格式: {type(src).__name__!r} ext={ext!r}。\n"
                "支持的来源:\n"
                "  • torch.Tensor (host memory)\n"
                "  • str 路径 (.pt — PyTorch tensor)\n"
                "  • str 路径 (.npy — NumPy array)\n"
                "  • List[str] (binary 文件列表, 需同时传 shape/dtype)"
            )

        self.load_from_global_tensor(host_tensor)

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def load_from_global_tensor(self, tensor):
        """将 host 全局张量写入 WholeGraph 分布式张量 (各 rank 各取其片段)。"""
        if self._tensor is None:
            raise ValueError("请先创建 WholeGraph 张量。")

        _dbg(
            "DistTensor",
            "load_from_global_tensor",
            f"host_shape={list(tensor.shape)} dtype={tensor.dtype}",
        )
        self.__dtype = tensor.dtype

        if isinstance(self._tensor, wgth.WholeMemoryEmbedding):
            _wm = self._tensor.get_embedding_tensor()
        else:
            _wm = self._tensor

        copy_host_global_tensor_to_local(_wm, tensor, _wm.get_comm())
        _dbg("DistTensor", "load_from_global_tensor", "写入完毕 ✓")

    def load_from_local_tensor(self, tensor):
        """将本 rank 的本地张量直接复制进 WholeGraph 分布式张量的本地分片。"""
        if self._tensor is None:
            raise ValueError("请先创建 WholeGraph 张量。")

        _dbg(
            "DistTensor",
            "load_from_local_tensor",
            f"local_shape={list(tensor.shape)} dtype={tensor.dtype} "
            f"wm_local_shape={list(self._tensor.local_shape)}",
        )

        if self._tensor.local_shape != tensor.shape:
            raise ValueError(
                f"shape 不匹配: WholeGraph 本地分片={list(self._tensor.local_shape)}, "
                f"传入 tensor={list(tensor.shape)}。"
            )
        if self.dtype != tensor.dtype:
            raise ValueError(
                f"dtype 不匹配: DistTensor.dtype={self.dtype}, "
                f"传入 tensor.dtype={tensor.dtype}。"
            )

        if isinstance(self._tensor, wgth.WholeMemoryEmbedding):
            self._tensor.get_embedding_tensor().get_local_tensor().copy_(tensor)
        else:
            self._tensor.get_local_tensor().copy_(tensor)

        _dbg("DistTensor", "load_from_local_tensor", "本地拷贝完毕 ✓")

    # ── 工厂方法 ──────────────────────────────────────────────────────────────

    @classmethod
    def from_tensor(
        cls,
        tensor: "torch.Tensor",
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
    ) -> "DistTensor":
        """从 PyTorch Tensor 创建 DistTensor。"""
        return cls(src=tensor, device=device,
                   partition_book=partition_book, backend=backend)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
    ) -> "DistTensor":
        """从 .pt 或 .npy 文件创建 DistTensor。"""
        return cls(src=file_path, device=device,
                   partition_book=partition_book, backend=backend)

    # ── 访问接口 ──────────────────────────────────────────────────────────────

    def __setitem__(self, idx: "torch.Tensor", val: "torch.Tensor"):
        """通过全局索引写入 embedding。所有进程必须同时调用。

        migrate 2776772: 支持 1D wholememory tensor 的 scatter。
        上游 pylibwholegraph scatter_op 内部始终以 2D 矩阵操作；对 1D 张量，
        输入 val 需 unsqueeze(1) 变成 [N,1] 再传入，否则 dim 断言失败。
        Walpurgis 改写: 在我们这一层统一处理维度转换，而不是让上游 assert 报错。
        WALPURGIS_DEBUG=1 时打印 1D/2D 路径选择，便于排查多GPU scatter不对齐。
        """
        assert self._tensor is not None, "请先创建 WholeGraph 张量。"

        tensor_dim = self._tensor.dim()
        _dbg(
            "DistTensor",
            "__setitem__",
            f"idx.shape={list(idx.shape)} val.shape={list(val.shape)} "
            f"idx_range=[{idx.min().item()}, {idx.max().item()}] "
            f"tensor_dim={tensor_dim}",
        )

        idx = idx.cuda()
        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        if not val.is_cuda:
            val = val.pin_memory()

        # --- 1D tensor scatter fix (migrate 2776772) ---
        # pylibwholegraph scatter_op 内部把 wholememory_tensor unsqueeze 成 2D，
        # 因此输入 val 也必须是 2D [N,1]。上游在 tensor.py 中 assert input_tensor.dim() == 2，
        # 对 1D 张量 (val.dim()==1) 会直接 AssertionError。
        # 解法: 1D wholememory tensor 时，在调用前 unsqueeze(1)。
        if tensor_dim == 1:
            _dbg("DistTensor", "__setitem__", "1D tensor: val.unsqueeze(1) 再 scatter")
            val = val.unsqueeze(1)

        self._tensor.scatter(val, idx)

    def __getitem__(self, idx: "torch.Tensor") -> "torch.Tensor":
        """通过全局索引读取 embedding。所有进程必须同时调用。

        migrate 2776772: 支持 1D wholememory tensor 的 gather。
        上游 pylibwholegraph gather 输出始终是 [N, embedding_dim] 的 2D 张量；
        对 1D wholememory tensor，embedding_dim=1，输出为 [N,1]，需要 view(-1) 还原为 [N]。
        Walpurgis 改写: 在我们这一层统一处理，不依赖上游 pylibwholegraph 是否已修复。
        """
        assert self._tensor is not None, "请先创建 WholeGraph 张量。"

        tensor_dim = self._tensor.dim()
        _dbg(
            "DistTensor",
            "__getitem__",
            f"idx.shape={list(idx.shape)} "
            f"idx_range=[{idx.min().item()}, {idx.max().item()}] "
            f"tensor_dim={tensor_dim}",
        )

        idx = idx.cuda()
        output_tensor = self._tensor.gather(idx)

        # --- 1D tensor gather fix (migrate 2776772) ---
        # pylibwholegraph gather 输出 [N, embedding_dim]；
        # 若底层是 1D wholememory tensor，embedding_dim=1，用户期待 [N] 而非 [N,1]。
        # view(-1) 将 [N,1] → [N]，与原生 torch 索引行为一致。
        if tensor_dim == 1:
            _dbg(
                "DistTensor",
                "__getitem__",
                f"1D tensor: view(-1) 还原 [N,1]→[N], before={list(output_tensor.shape)}",
            )
            output_tensor = output_tensor.view(-1)

        _dbg(
            "DistTensor",
            "__getitem__",
            f"gather 完毕 output.shape={list(output_tensor.shape)}",
        )
        return output_tensor

    def get_local_tensor(self, host_view=False):
        """获取本 rank 的本地张量。"""
        local_tensor, _ = self._tensor.get_local_tensor(host_view=host_view)
        return local_tensor

    def get_local_offset(self) -> int:
        """获取本 rank 本地分片的全局起始偏移量。"""
        _, offset = self._tensor.get_local_tensor()
        return offset

    def get_comm(self):
        """获取 WholeGraph 通信器。"""
        assert self._tensor is not None, "请先创建 WholeGraph 张量。"
        return self._tensor.get_comm()

    # ── 属性 ─────────────────────────────────────────────────────────────────

    @property
    def dim(self):
        return self._tensor.dim()

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def __repr__(self):
        if self._tensor is None:
            return "<DistTensor: 未加载张量>"
        return (
            f"DistTensor("
            f"shape={self._tensor.shape}, "
            f"dtype={self.dtype}, "
            f"device='{self.device}')"
        )


# ─── DistEmbedding ───────────────────────────────────────────────────────────


class DistEmbedding(DistTensor):
    """WholeGraph 分布式 Embedding 接口 (Walpurgis 版)。

    在 DistTensor 的基础上包一层 WholeMemoryEmbedding:
    支持 cache_policy、gather_sms、round_robin_size, 并钩入 PyTorch 梯度追踪。

    鲁迅: DistTensor 是仓库, DistEmbedding 是柜台——
    前者只管存取, 后者还要记账(梯度)。
    """

    def __init__(
        self,
        src: Optional[Union["torch.Tensor", str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
        cache_policy: Optional["pylibwholegraph.WholeMemoryCachePolicy"] = None,
        gather_sms: Optional[int] = -1,
        round_robin_size: int = 0,
        name: Optional[str] = None,
    ):
        self._name = name

        _dbg(
            "DistEmbedding",
            "__init__",
            f"name={name} cache_policy={'set' if cache_policy else 'None'} "
            f"gather_sms={gather_sms} round_robin_size={round_robin_size}",
        )

        super().__init__(
            src,
            shape,
            dtype,
            device,
            partition_book,
            backend,
            cache_policy=cache_policy,
            gather_sms=gather_sms,
            round_robin_size=round_robin_size,
        )

        # _tensor 此时是 WmEmbedding; 将其存为 _embedding, 再取底层 tensor
        self._embedding = self._tensor
        self._tensor = self._embedding.get_embedding_tensor()

        _dbg(
            "DistEmbedding",
            "__init__",
            f"_embedding={type(self._embedding).__name__} 绑定完毕",
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: "torch.Tensor",
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy=None,
        *args,
        **kwargs,
    ) -> "DistEmbedding":
        return cls(
            src=tensor,
            device=device,
            partition_book=partition_book,
            name=name,
            cache_policy=cache_policy,
            *args,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy=None,
        *args,
        **kwargs,
    ) -> "DistEmbedding":
        return cls(
            src=file_path,
            device=device,
            partition_book=partition_book,
            name=name,
            cache_policy=cache_policy,
            *args,
            **kwargs,
        )

    def __setitem__(self, idx: "torch.Tensor", val: "torch.Tensor"):
        """写入 embedding (经由 WmEmbedding 接口)。"""
        assert self._tensor is not None, "请先创建 WholeGraph embedding 张量。"

        _dbg(
            "DistEmbedding",
            "__setitem__",
            f"idx.shape={list(idx.shape)} val.shape={list(val.shape)}",
        )

        idx = idx.cuda()
        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        if not val.is_cuda:
            val = val.pin_memory()
        self._embedding.get_embedding_tensor().scatter(val, idx)

    def __getitem__(self, idx: "torch.Tensor") -> "torch.Tensor":
        """读取 embedding (经由 WmEmbedding.gather, 支持梯度)。"""
        assert self._tensor is not None, "请先创建 WholeGraph embedding 张量。"

        _dbg(
            "DistEmbedding",
            "__getitem__",
            f"idx.shape={list(idx.shape)} "
            f"idx_range=[{idx.min().item()}, {idx.max().item()}]",
        )

        idx = idx.cuda()
        output_tensor = self._embedding.gather(idx)

        _dbg(
            "DistEmbedding",
            "__getitem__",
            f"gather 完毕 output.shape={list(output_tensor.shape)}",
        )
        return output_tensor

    @property
    def name(self) -> Optional[str]:
        return self._name

    def __repr__(self):
        if self._embedding is None:
            return f"<DistEmbedding: 未加载 embedding, name={self._name!r}>"
        parts = ["DistEmbedding("]
        if self._name:
            parts.append(f"name={self._name!r}, ")
        parts.append(
            f"shape={self.shape}, dtype={self.dtype}, device='{self.device}')"
        )
        return "".join(parts)
