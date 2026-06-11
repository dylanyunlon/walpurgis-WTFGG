"""
env_fn_utils.py — bd703b3 迁移: dtype/location/type 转换工具

上游来源: python/pylibwholegraph/pylibwholegraph/torch/utils.py
commit: bd703b3 (add wholegraph to repo)

独立为工具模块避免循环导入（env.py 在 comm.py 之前加载）。
"""

import os
import torch
import pylibwholegraph.binding.wholememory_binding as wmb

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/env_fn_utils]", *args, **kwargs)


# ── dtype 双向映射 ──

_WM_TO_TORCH = {
    wmb.WholeMemoryDataType.DtFloat: torch.float32,
    wmb.WholeMemoryDataType.DtHalf: torch.float16,
    wmb.WholeMemoryDataType.DtBF16: torch.bfloat16,
    wmb.WholeMemoryDataType.DtDouble: torch.float64,
    wmb.WholeMemoryDataType.DtInt: torch.int32,
    wmb.WholeMemoryDataType.DtInt64: torch.int64,
    wmb.WholeMemoryDataType.DtInt16: torch.int16,
    wmb.WholeMemoryDataType.DtInt8: torch.int8,
    wmb.WholeMemoryDataType.DtUInt8: torch.uint8,
}

_TORCH_TO_WM = {v: k for k, v in _WM_TO_TORCH.items()}


def wholememory_dtype_to_torch_dtype(wm_dtype) -> torch.dtype:
    """将 wmb dtype（枚举或整数）转为 torch.dtype。"""
    if isinstance(wm_dtype, int):
        wm_dtype = wmb.WholeMemoryDataType(wm_dtype)
    dtype = _WM_TO_TORCH.get(wm_dtype)
    if dtype is None:
        raise ValueError(f"未知 WholeMemory dtype: {wm_dtype}")
    _dbg(f"wholememory_dtype_to_torch_dtype: {wm_dtype} → {dtype}")
    return dtype


def torch_dtype_to_wholememory_dtype(torch_dtype: torch.dtype) -> wmb.WholeMemoryDataType:
    """将 torch.dtype 转为 wmb dtype 枚举。"""
    wm_dtype = _TORCH_TO_WM.get(torch_dtype)
    if wm_dtype is None:
        raise ValueError(f"未知 torch dtype: {torch_dtype}")
    _dbg(f"torch_dtype_to_wholememory_dtype: {torch_dtype} → {wm_dtype}")
    return wm_dtype


# ── memory type / location / backend str 转换 ──

def str_to_wmb_wholememory_memory_type(s: str) -> wmb.WholeMemoryMemoryType:
    _map = {
        "continuous": wmb.WholeMemoryMemoryType.MtContinuous,
        "chunked": wmb.WholeMemoryMemoryType.MtChunked,
        "distributed": wmb.WholeMemoryMemoryType.MtDistributed,
    }
    v = _map.get(s)
    if v is None:
        raise ValueError(f"未知 memory_type: {s}, 支持: {list(_map)}")
    return v


def str_to_wmb_wholememory_location(s: str) -> wmb.WholeMemoryMemoryLocation:
    _map = {
        "cpu": wmb.WholeMemoryMemoryLocation.MlHost,
        "cuda": wmb.WholeMemoryMemoryLocation.MlDevice,
    }
    v = _map.get(s)
    if v is None:
        raise ValueError(f"未知 memory_location: {s}, 支持: {list(_map)}")
    return v


def str_to_wmb_wholememory_distributed_backend_type(s: str):
    _map = {
        "nccl": wmb.WholeMemoryDistributedBackendType.DbNccl,
        "nvshmem": wmb.WholeMemoryDistributedBackendType.DbNvshmem,
    }
    v = _map.get(s)
    if v is None:
        raise ValueError(f"未知 distributed_backend: {s}")
    return v


def wholememory_distributed_backend_type_to_str(t) -> str:
    _map = {
        wmb.WholeMemoryDistributedBackendType.DbNccl: "nccl",
        wmb.WholeMemoryDistributedBackendType.DbNvshmem: "nvshmem",
    }
    return _map.get(t, str(t))


def str_to_wmb_wholememory_access_type(s: str):
    _map = {
        "readonly": wmb.WholeMemoryAccessType.AtReadOnly,
        "readwrite": wmb.WholeMemoryAccessType.AtReadWrite,
    }
    v = _map.get(s)
    if v is None:
        raise ValueError(f"未知 access_type: {s}")
    return v


def str_to_wmb_wholememory_optimizer_type(s: str):
    _map = {
        "sgd": wmb.WholeMemoryOptimizerType.OtSgd,
        "adam": wmb.WholeMemoryOptimizerType.OtAdam,
        "rmsprop": wmb.WholeMemoryOptimizerType.OtRmsprop,
        "adagrad": wmb.WholeMemoryOptimizerType.OtAdagrad,
    }
    v = _map.get(s)
    if v is None:
        raise ValueError(f"未知 optimizer_type: {s}")
    return v


def str_to_wmb_wholememory_log_level(s: str):
    _map = {
        "fatal": wmb.WholeMemoryLogLevel.LlFatal,
        "error": wmb.WholeMemoryLogLevel.LlError,
        "warn": wmb.WholeMemoryLogLevel.LlWarn,
        "info": wmb.WholeMemoryLogLevel.LlInfo,
        "debug": wmb.WholeMemoryLogLevel.LlDebug,
        "trace": wmb.WholeMemoryLogLevel.LlTrace,
    }
    v = _map.get(s.lower())
    if v is None:
        raise ValueError(f"未知 log_level: {s}")
    return v


# ── 文件工具 ──

def get_part_file_name(file_prefix: str, part_id: int, part_count: int) -> str:
    return f"{file_prefix}_part_{part_id}_of_{part_count}"


def get_part_file_list(file_prefix: str, part_count: int):
    return [get_part_file_name(file_prefix, i, part_count) for i in range(part_count)]


def get_file_size(filename: str) -> int:
    return os.path.getsize(filename)
