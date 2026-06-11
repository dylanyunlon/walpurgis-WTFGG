"""
graph_ops.py — bd703b3 迁移: 图操作（append_unique / add_csr_self_loop）

上游来源: python/pylibwholegraph/pylibwholegraph/torch/graph_ops.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _AppendUniqueResult dataclass 封装 append_unique 的返回值:
  (unique_nodes, neighbor_raw_to_unique_mapping?) — 消除调用方 len/unpack 不对称
- _CsrSelfLoopResult dataclass 封装 add_csr_self_loop 返回的 (row_ptr, col_ptr)
- 前置断言加 device 检查 error message（上游 assert 无消息体）
- 全链路 WALPURGIS_DEBUG=1 断点 print: 输入 shape / unique 结果 / CSR 尺寸变化
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

from .env import get_stream, wrap_torch_tensor, get_wholegraph_env_fns, _OutputBuffer

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/graph_ops]", *args, **kwargs)


# ──────────────────────────────────────────────
# _AppendUniqueResult — append_unique 结果对象
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class _AppendUniqueResult:
    """
    封装 append_unique 的输出。

    上游: 返回 1 或 2 元素 tuple，调用方需根据 need_neighbor_raw_to_unique 判断。
    Walpurgis: 固定字段，Optional 表达可选项。
    """
    unique_nodes: torch.Tensor
    neighbor_raw_to_unique: Optional[torch.Tensor] = None

    def as_tuple(self):
        """上游兼容接口。"""
        if self.neighbor_raw_to_unique is not None:
            return self.unique_nodes, self.neighbor_raw_to_unique
        return (self.unique_nodes,)


# ──────────────────────────────────────────────
# _CsrSelfLoopResult — add_csr_self_loop 结果对象
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class _CsrSelfLoopResult:
    """封装加自环后的 CSR 张量对。"""
    csr_row_ptr: torch.Tensor
    csr_col_ptr: torch.Tensor


# ──────────────────────────────────────────────
# append_unique
# ──────────────────────────────────────────────

def append_unique(
    target_node_tensor: torch.Tensor,
    neighbor_node_tensor: torch.Tensor,
    need_neighbor_raw_to_unique: bool = False,
) -> _AppendUniqueResult:
    """
    将 neighbor_node_tensor 追加到 target_node_tensor 后去重。

    target_node_tensor 中的元素顺序保持不变；
    新出现的 neighbor 节点以任意顺序追加到末尾。

    例：target=[3,11,2,10], neighbor=[4,5,2,11,6,9,10,5]
        → unique_nodes=[3,11,2,10,6,4,9,5]（新节点顺序不定）
        → neighbor_raw_to_unique=[5,7,2,1,4,6,3,7]

    :param target_node_tensor: 目标节点 id（1D，CUDA）
    :param neighbor_node_tensor: 邻居节点 id（1D，CUDA）
    :param need_neighbor_raw_to_unique: 是否输出 neighbor→unique 映射
    :return: _AppendUniqueResult
    """
    assert target_node_tensor.dim() == 1, "target_node_tensor 必须是 1D 张量"
    assert neighbor_node_tensor.dim() == 1, "neighbor_node_tensor 必须是 1D 张量"
    assert target_node_tensor.is_cuda, "target_node_tensor 必须在 CUDA 设备上"
    assert neighbor_node_tensor.is_cuda, "neighbor_node_tensor 必须在 CUDA 设备上"

    _dbg(
        f"append_unique: target.shape={target_node_tensor.shape} "
        f"neighbor.shape={neighbor_node_tensor.shape} "
        f"need_mapping={need_neighbor_raw_to_unique}"
    )

    unique_node_buf = _OutputBuffer()
    raw_to_unique: Optional[torch.Tensor] = None
    if need_neighbor_raw_to_unique:
        raw_to_unique = torch.empty(
            neighbor_node_tensor.shape[0], device="cuda", dtype=torch.int
        )

    wmb.append_unique(
        wrap_torch_tensor(target_node_tensor),
        wrap_torch_tensor(neighbor_node_tensor),
        unique_node_buf.c_handle(),
        wrap_torch_tensor(raw_to_unique),
        get_wholegraph_env_fns(),
        get_stream(),
    )

    unique_nodes = unique_node_buf.tensor()
    _dbg(
        f"append_unique → unique.shape={unique_nodes.shape} "
        f"mapping={'yes' if raw_to_unique is not None else 'no'}"
    )
    return _AppendUniqueResult(
        unique_nodes=unique_nodes,
        neighbor_raw_to_unique=raw_to_unique,
    )


# ──────────────────────────────────────────────
# add_csr_self_loop
# ──────────────────────────────────────────────

def add_csr_self_loop(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
) -> _CsrSelfLoopResult:
    """
    向采样后的 CSR 图添加自环。

    注意: 不检查原图是否已有自环，若已有则会重复添加。

    :param csr_row_ptr_tensor: CSR 行指针（1D，CUDA）
    :param csr_col_ptr_tensor: CSR 列索引（1D，CUDA）
    :return: _CsrSelfLoopResult，包含加自环后的 row_ptr 和 col_ptr
    """
    assert csr_row_ptr_tensor.dim() == 1, "csr_row_ptr_tensor 必须是 1D 张量"
    assert csr_col_ptr_tensor.dim() == 1, "csr_col_ptr_tensor 必须是 1D 张量"
    assert csr_row_ptr_tensor.is_cuda, "csr_row_ptr_tensor 必须在 CUDA 设备上"
    assert csr_col_ptr_tensor.is_cuda, "csr_col_ptr_tensor 必须在 CUDA 设备上"

    num_nodes = csr_row_ptr_tensor.shape[0] - 1
    num_edges = csr_col_ptr_tensor.shape[0]
    _dbg(
        f"add_csr_self_loop: num_nodes={num_nodes} "
        f"num_edges_before={num_edges} num_edges_after={num_edges + num_nodes}"
    )

    out_row_ptr = torch.empty(
        (csr_row_ptr_tensor.shape[0],),
        device="cuda",
        dtype=csr_row_ptr_tensor.dtype,
    )
    out_col_ptr = torch.empty(
        (csr_col_ptr_tensor.shape[0] + csr_row_ptr_tensor.shape[0] - 1,),
        device="cuda",
        dtype=csr_col_ptr_tensor.dtype,
    )

    wmb.add_csr_self_loop(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(out_row_ptr),
        wrap_torch_tensor(out_col_ptr),
        get_stream(),
    )

    _dbg(f"add_csr_self_loop → out_col.shape={out_col_ptr.shape}")
    return _CsrSelfLoopResult(csr_row_ptr=out_row_ptr, csr_col_ptr=out_col_ptr)
