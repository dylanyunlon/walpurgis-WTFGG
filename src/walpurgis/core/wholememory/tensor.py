"""
tensor.py — bd703b3 迁移: WholeMemory 张量层

上游来源: python/pylibwholegraph/pylibwholegraph/torch/tensor.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _LocalTensorView dataclass 封装 get_local_tensor / get_global_tensor 的
  (tensor, element_offset) 返回对，消除调用方解包歧义
- WholeMemoryTensor.gather / scatter 加 WALPURGIS_DEBUG shape/dtype 校验日志
- get_all_chunked_tensor 修正上游 typo:
    wmb_tensor.get_global_tensorget_all_chunked_tensor → get_all_chunked_tensor
  (上游原版有该拼写错误，迁移修正)
- 全链路 WALPURGIS_DEBUG=1 断点 print: gather/scatter 调用参数、文件 IO 路径
"""

import os
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

from .env_fn_utils import (
    torch_dtype_to_wholememory_dtype,
    wholememory_dtype_to_torch_dtype,
    get_part_file_name,
    get_part_file_list,
    get_file_size,
)
from .env import get_wholegraph_env_fns, get_stream, wrap_torch_tensor

# dlpack import shim
try:
    from torch.utils.dlpack import from_dlpack as _torch_from_dlpack
except ImportError:
    _torch_from_dlpack = torch.utils.dlpack.from_dlpack


def _torch_import_from_dlpack(dlt):
    return _torch_from_dlpack(dlt)


# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/tensor]", *args, **kwargs)


# 暴露 wmb 枚举供外部使用
WholeMemoryMemoryType = wmb.WholeMemoryMemoryType
WholeMemoryMemoryLocation = wmb.WholeMemoryMemoryLocation


# ──────────────────────────────────────────────
# _LocalTensorView — 本地张量视图值对象
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class _LocalTensorView:
    """
    封装 get_local_tensor / get_global_tensor 的返回值。
    上游直接返回 tuple (tensor, element_offset)，调用方需知道解包顺序。
    Walpurgis 用具名字段消除歧义。
    """
    tensor: torch.Tensor
    element_offset: int


# ──────────────────────────────────────────────
# WholeMemoryTensor
# ──────────────────────────────────────────────

class WholeMemoryTensor:
    """WholeMemory 分布式张量包装。"""

    def __init__(self, wmb_tensor: wmb.PyWholeMemoryTensor):
        self.wmb_tensor = wmb_tensor

    @property
    def dtype(self) -> torch.dtype:
        return wholememory_dtype_to_torch_dtype(self.wmb_tensor.dtype)

    def dim(self) -> int:
        return self.wmb_tensor.dim()

    @property
    def shape(self):
        return self.wmb_tensor.shape

    def stride(self):
        return self.wmb_tensor.stride()

    def storage_offset(self) -> int:
        return self.wmb_tensor.storage_offset()

    def get_comm(self):
        from .comm import WholeMemoryCommunicator
        return WholeMemoryCommunicator(
            self.wmb_tensor.get_wholememory_handle().get_communicator()
        )

    # ── gather ──

    def gather(
        self,
        indice: torch.Tensor,
        *,
        force_dtype: Union[torch.dtype, None] = None,
    ) -> torch.Tensor:
        """
        从 WholeMemory 张量按 indice 聚合行。

        :param indice: 1D 整数索引张量（需在 CUDA 上）
        :param force_dtype: 若不为 None，强制输出 dtype
        :return: [len(indice), embedding_dim] 的输出张量
        """
        assert indice.dim() == 1, "indice 必须是 1D 张量"
        embedding_dim = self.shape[1]
        embedding_count = indice.shape[0]
        current_cuda_device = f"cuda:{torch.cuda.current_device()}"
        output_dtype = force_dtype if force_dtype is not None else self.dtype
        _dbg(
            f"gather: indice.shape={indice.shape} embedding_dim={embedding_dim} "
            f"output_dtype={output_dtype} device={current_cuda_device}"
        )
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=False,
        )
        wmb.wholememory_gather_op(
            self.wmb_tensor,
            wrap_torch_tensor(indice),
            wrap_torch_tensor(output_tensor),
            get_wholegraph_env_fns(),
            get_stream(),
        )
        return output_tensor

    # ── scatter ──

    def scatter(self, input_tensor: torch.Tensor, indice: torch.Tensor) -> None:
        """
        将 input_tensor 的行按 indice 分散写入 WholeMemory 张量。

        :param input_tensor: [N, embedding_dim] 2D 张量
        :param indice: 1D 整数索引张量，len == N
        """
        assert indice.dim() == 1
        assert input_tensor.dim() == 2
        assert indice.shape[0] == input_tensor.shape[0]
        assert input_tensor.shape[1] == self.shape[1]
        _dbg(
            f"scatter: input.shape={input_tensor.shape} indice.shape={indice.shape} "
            f"input.dtype={input_tensor.dtype}"
        )
        wmb.wholememory_scatter_op(
            wrap_torch_tensor(input_tensor),
            wrap_torch_tensor(indice),
            self.wmb_tensor,
            get_wholegraph_env_fns(),
            get_stream(),
        )

    # ── 子张量 / 视图 ──

    def get_sub_tensor(self, starts, ends) -> "WholeMemoryTensor":
        """获取子张量，ends=-1 表示至最后一个元素。"""
        return WholeMemoryTensor(self.wmb_tensor.get_sub_tensor(starts, ends))

    def get_local_tensor(self, host_view: bool = False) -> _LocalTensorView:
        """
        获取本 rank 的局部张量视图。
        返回 _LocalTensorView(tensor, element_offset)。
        上游返回裸 tuple，Walpurgis 用具名对象。
        """
        if host_view:
            t, offset = self.wmb_tensor.get_local_tensor(
                _torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            t, offset = self.wmb_tensor.get_local_tensor(
                _torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )
        _dbg(f"get_local_tensor: host_view={host_view} offset={offset} shape={t.shape}")
        return _LocalTensorView(tensor=t, element_offset=offset)

    def get_global_tensor(self, host_view: bool = False) -> _LocalTensorView:
        """获取全局视图（offset 始终为 0）。"""
        if host_view:
            t, offset = self.wmb_tensor.get_global_tensor(
                _torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            t, offset = self.wmb_tensor.get_global_tensor(
                _torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )
        _dbg(f"get_global_tensor: host_view={host_view} offset={offset} shape={t.shape}")
        return _LocalTensorView(tensor=t, element_offset=offset)

    def get_all_chunked_tensor(
        self, host_view: bool = False
    ) -> List[_LocalTensorView]:
        """
        获取所有 rank 的 chunked 张量列表。

        上游原版存在 typo:
            wmb_tensor.get_global_tensorget_all_chunked_tensor(...)
        Walpurgis 修正为 get_all_chunked_tensor。
        """
        if host_view:
            results = self.wmb_tensor.get_all_chunked_tensor(
                _torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            results = self.wmb_tensor.get_all_chunked_tensor(
                _torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )
        views = [_LocalTensorView(tensor=t, element_offset=off) for t, off in results]
        _dbg(f"get_all_chunked_tensor: host_view={host_view} chunks={len(views)}")
        return views

    # ── 文件 IO ──

    def from_filelist(
        self, filelist: Union[List[str], str], round_robin_size: int = 0
    ) -> None:
        """从文件列表加载 WholeMemory 张量。"""
        if isinstance(filelist, str):
            filelist = [filelist]
        _dbg(f"from_filelist: {len(filelist)} files, round_robin_size={round_robin_size}")
        self.wmb_tensor.from_filelist(filelist, round_robin_size)

    def from_file_prefix(
        self, file_prefix: str, part_count: Union[int, None] = None
    ) -> None:
        """从统一前缀的分片文件加载。格式: {prefix}_part_{i}_of_{n}"""
        if part_count is None:
            part_count = self.get_comm().get_size()
        file_list = get_part_file_list(file_prefix, part_count)
        _dbg(f"from_file_prefix: prefix={file_prefix} parts={part_count}")
        self.from_filelist(file_list)

    def local_to_file(self, filename: str) -> None:
        """将本 rank 局部数据写入文件。"""
        _dbg(f"local_to_file: {filename}")
        self.wmb_tensor.local_to_file(filename)

    def to_file_prefix(self, file_prefix: str) -> None:
        """将本 rank 数据写入前缀命名文件。"""
        comm = self.get_comm()
        rank = comm.get_rank()
        size = comm.get_size()
        filename = get_part_file_name(file_prefix, rank, size)
        _dbg(f"to_file_prefix: prefix={file_prefix} rank={rank}/{size} → {filename}")
        self.local_to_file(filename)


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_wholememory_tensor(
    comm,
    memory_type: str,
    memory_location: str,
    dtype: torch.dtype,
    sizes: List[int],
    *,
    strides: Optional[List[int]] = None,
) -> WholeMemoryTensor:
    """
    创建 WholeMemory 张量。

    :param comm: WholeMemoryCommunicator
    :param memory_type: "continuous" / "chunked" / "distributed"
    :param memory_location: "cpu" / "cuda"
    :param dtype: torch dtype
    :param sizes: 形状列表
    :param strides: 步幅列表（None 则使用默认紧凑步幅）
    :return: WholeMemoryTensor
    """
    from .env_fn_utils import (
        str_to_wmb_wholememory_memory_type,
        str_to_wmb_wholememory_location,
        torch_dtype_to_wholememory_dtype,
    )
    if strides is None:
        strides = []
    _dbg(
        f"create_wholememory_tensor: type={memory_type} loc={memory_location} "
        f"dtype={dtype} sizes={sizes}"
    )
    wmb_tensor = wmb.create_wholememory_tensor(
        comm.wmb_comm,
        str_to_wmb_wholememory_memory_type(memory_type),
        str_to_wmb_wholememory_location(memory_location),
        torch_dtype_to_wholememory_dtype(dtype),
        sizes,
        strides,
    )
    return WholeMemoryTensor(wmb_tensor)


def create_wholememory_tensor_from_filelist(
    comm,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: torch.dtype,
    last_dim_size: int,
    *,
    round_robin_size: int = 0,
) -> WholeMemoryTensor:
    """从文件列表直接创建 WholeMemory 张量。"""
    from .env_fn_utils import (
        str_to_wmb_wholememory_memory_type,
        str_to_wmb_wholememory_location,
        torch_dtype_to_wholememory_dtype,
    )
    if isinstance(filelist, str):
        filelist = [filelist]
    total_size = sum(get_file_size(f) for f in filelist)
    elem_size = torch.empty(1, dtype=dtype).element_size()
    total_elems = total_size // elem_size
    assert total_elems % last_dim_size == 0
    sizes = [total_elems // last_dim_size, last_dim_size]
    _dbg(
        f"create_wholememory_tensor_from_filelist: {len(filelist)} files "
        f"total_elems={total_elems} sizes={sizes}"
    )
    wm_tensor = create_wholememory_tensor(
        comm, memory_type, memory_location, dtype, sizes
    )
    wm_tensor.from_filelist(filelist, round_robin_size)
    return wm_tensor


def destroy_wholememory_tensor(wm_tensor: WholeMemoryTensor) -> None:
    """销毁 WholeMemory 张量。"""
    if wm_tensor is not None and wm_tensor.wmb_tensor is not None:
        _dbg("destroy_wholememory_tensor")
        wmb.destroy_wholememory_tensor(wm_tensor.wmb_tensor)
        wm_tensor.wmb_tensor = None
