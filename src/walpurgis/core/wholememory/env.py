"""
env.py — bd703b3 迁移: WholeGraph 运行时环境层

上游来源: python/pylibwholegraph/pylibwholegraph/torch/wholegraph_env.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _WalpurgisEnvCtx dataclass 封装全局 env context 状态，替代模块级散落变量
- _OutputBuffer 替换 TorchMemoryContext: 统一 cpp_ext / fallback 两条路径，
  消除 get_c_context() / get_tensor() 之间的隐性状态耦合
- compile_cpp_extension 改为惰性单例，首次调用后缓存，避免重入
- 全链路 WALPURGIS_DEBUG=1 断点 print: env 初始化 / malloc 决策 / buffer 分配 / stream 查询
"""

import os
import importlib
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS wholememory/env]", *args, **kwargs)


# ──────────────────────────────────────────────
# _WalpurgisEnvCtx — 运行时环境单例
# ──────────────────────────────────────────────

@dataclass
class _WalpurgisEnvCtx:
    """
    封装 WalpurGIS wholegraph 环境的全局状态。

    上游用三个模块级变量管理:
        default_wholegraph_env_context = None
        torch_cpp_ext_loaded = False
        torch_cpp_ext_lib = None
    集中到此处便于调试 inspect 和单元测试替换。
    """
    env_context: Optional[object] = None
    cpp_ext_loaded: bool = False
    cpp_ext_lib: Optional[object] = None


_env: _WalpurgisEnvCtx = _WalpurgisEnvCtx()


# ──────────────────────────────────────────────
# CUDA stream 工具
# ──────────────────────────────────────────────

def get_stream() -> int:
    """返回当前 CUDA stream 的整数指针，供 wmb C++ 层使用。"""
    cuda_stream = torch.cuda.current_stream()._as_parameter_
    ptr = cuda_stream.value if cuda_stream.value is not None else 0
    _dbg(f"get_stream → {ptr:#x}")
    return ptr


# ──────────────────────────────────────────────
# _OutputBuffer — 替代上游 TorchMemoryContext
# ──────────────────────────────────────────────

class _OutputBuffer:
    """
    管理一次 wholegraph 输出分配的生命周期。

    上游 TorchMemoryContext 在 cpp_ext_loaded / fallback 两条路径下
    get_c_context() 语义不同（真实 handle vs id(self)），调用方
    需知道内部状态才能正确解读返回值。
    _OutputBuffer 统一接口，隐藏路径差异。
    """

    def __init__(self):
        self._tensor: Optional[torch.Tensor] = None
        if _env.cpp_ext_loaded:
            self._handle = _env.cpp_ext_lib.create_output_context()
            _dbg(f"_OutputBuffer init: cpp_ext handle={self._handle}")
        else:
            self._handle = 0
            _dbg(f"_OutputBuffer init: fallback mode, id={id(self):#x}")

    def __del__(self):
        self._release()

    # ── 供 wmb C++ 层使用的不透明句柄 ──

    def c_handle(self) -> int:
        """返回传递给 C++ 层的句柄。cpp_ext 模式返回真实 handle；fallback 返回 id。"""
        return self._handle if _env.cpp_ext_loaded else id(self)

    # ── 供 Python 层取回张量 ──

    def tensor(self) -> Optional[torch.Tensor]:
        """从 C++ 侧取回分配的张量（cpp_ext 模式），或返回 Python 侧设置的张量。"""
        if _env.cpp_ext_loaded:
            self._tensor = _env.cpp_ext_lib.get_tensor_from_context(self._handle)
            _dbg(f"_OutputBuffer.tensor(): cpp_ext → shape={self._tensor.shape if self._tensor is not None else None}")
        else:
            _dbg(f"_OutputBuffer.tensor(): fallback → shape={self._tensor.shape if self._tensor is not None else None}")
        return self._tensor

    def set_tensor(self, t: torch.Tensor) -> None:
        """fallback 路径下由 Python env_fn 回调写入张量。"""
        _dbg(f"_OutputBuffer.set_tensor(): shape={t.shape}, dtype={t.dtype}")
        self._tensor = t

    def _release(self) -> None:
        self._tensor = None
        if _env.cpp_ext_loaded and self._handle != 0:
            _env.cpp_ext_lib.destroy_output_context(self._handle)
            self._handle = 0


# ──────────────────────────────────────────────
# wrap_torch_tensor
# ──────────────────────────────────────────────

def wrap_torch_tensor(t: Optional[torch.Tensor]) -> Optional[wmb.PyWholeMemoryTensor]:
    """将 torch.Tensor 包装为 wmb 可识别的引用。None 安全。"""
    if t is None:
        return None
    if _env.cpp_ext_loaded:
        return _env.cpp_ext_lib.wrap_torch_tensor(t)
    return wmb.wrap_torch_tensor(t)


# ──────────────────────────────────────────────
# wholegraph env_fn 回调（fallback 路径）
# ──────────────────────────────────────────────

from .env_fn_utils import (
    wholememory_dtype_to_torch_dtype,
    torch_dtype_to_wholememory_dtype,
)


class _TorchEmptyGlobalCtx:
    """全局上下文占位对象（无状态）。"""
    pass


def _torch_create_memory_context_env_fn(
    global_context: _TorchEmptyGlobalCtx,
) -> _OutputBuffer:
    ctx = _OutputBuffer()
    _dbg("env_fn: create_memory_context")
    return ctx


def _torch_destroy_memory_context_env_fn(
    memory_context: _OutputBuffer,
    global_context: _TorchEmptyGlobalCtx,
) -> None:
    _dbg("env_fn: destroy_memory_context")
    memory_context._release()


def _torch_malloc_env_fn(
    shape,
    dtype_int: int,
    malloc_type_int: int,
    memory_context: _OutputBuffer,
    global_context: _TorchEmptyGlobalCtx,
) -> int:
    """
    fallback malloc 回调。

    上游 fbea7cb 改变了签名: 不再传 PyWholeMemoryTensorDescription 对象，
    改传 (shape: tuple, dtype_int: int, malloc_type_int: int)。
    Walpurgis 迁移与 fbea7cb 保持一致。
    """
    dtype = wholememory_dtype_to_torch_dtype(dtype_int)
    alloc_type = wmb.WholeMemoryMemoryAllocType(malloc_type_int)
    pinned = False
    if alloc_type == wmb.WholeMemoryMemoryAllocType.MatDevice:
        device = torch.device("cuda")
    elif alloc_type == wmb.WholeMemoryMemoryAllocType.MatHost:
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        pinned = True

    _dbg(f"env_fn: malloc shape={shape} dtype={dtype} device={device} pinned={pinned}")

    t = torch.empty(list(shape), dtype=dtype, device=device)
    if pinned:
        t = t.pin_memory()
    memory_context.set_tensor(t)
    return t.data_ptr()


def _torch_free_env_fn(
    memory_context: _OutputBuffer,
    global_context: _TorchEmptyGlobalCtx,
) -> None:
    _dbg("env_fn: free")
    memory_context._tensor = None


# ──────────────────────────────────────────────
# get_wholegraph_env_fns — 构建 env 函数表
# ──────────────────────────────────────────────

_env_global_ctx: Optional[_TorchEmptyGlobalCtx] = None


def get_wholegraph_env_fns():
    """
    返回 wmb 所需的环境函数表。
    上游用 default_wholegraph_env_context 缓存；Walpurgis 用 _env_global_ctx。
    """
    global _env_global_ctx
    if _env_global_ctx is None:
        _env_global_ctx = _TorchEmptyGlobalCtx()
        _dbg("get_wholegraph_env_fns: 首次初始化 env_global_ctx")
    if _env.cpp_ext_loaded:
        return _env.cpp_ext_lib.get_wholegraph_env_fns(_env_global_ctx)
    return wmb.create_wholememory_env(
        _torch_create_memory_context_env_fn,
        _torch_destroy_memory_context_env_fn,
        _torch_malloc_env_fn,
        _torch_free_env_fn,
        _torch_malloc_env_fn,   # output malloc 与 temp malloc 共享实现
        _torch_free_env_fn,
        _env_global_ctx,
    )


# ──────────────────────────────────────────────
# compile_cpp_extension — 惰性单例
# ──────────────────────────────────────────────

def compile_cpp_extension() -> None:
    """
    编译并加载 wholegraph torch cpp extension。
    Walpurgis 改为惰性单例模式：首次调用后缓存，重入时直接返回。
    上游每次调用均尝试 importlib.import_module，无幂等保护。
    """
    if _env.cpp_ext_loaded:
        _dbg("compile_cpp_extension: 已加载，跳过重入")
        return
    _dbg("compile_cpp_extension: 开始加载 torch_cpp_ext")
    try:
        lib = importlib.import_module("pylibwholegraph.torch_cpp_ext")
        lib.load_wholegraph_op()
        _env.cpp_ext_lib = lib
        _env.cpp_ext_loaded = True
        _dbg("compile_cpp_extension: 加载成功")
    except ImportError as e:
        _dbg(f"compile_cpp_extension: 加载失败，降级为 fallback 模式 ({e})")
