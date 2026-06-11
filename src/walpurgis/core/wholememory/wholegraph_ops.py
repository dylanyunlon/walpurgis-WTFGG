"""
wholegraph_ops.py — bd703b3 迁移: WholeGraph 采样操作

上游来源: python/pylibwholegraph/pylibwholegraph/torch/wholegraph_ops.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _SampleOutput dataclass 替代 unweighted/weighted 两个函数末尾
  if/elif/else 四路返回分支，统一通过一个具名结构体返回，
  调用方无需关心返回 tuple 长度
- _prepare_sample_output() 内部工厂函数消除两个函数间的重复输出分配逻辑
- 全链路 WALPURGIS_DEBUG=1 断点 print:
  采样参数 / random_seed / 输出 offset/dest shape / center_localid/edge_gid 是否启用
"""

import os
import random
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

from .env import get_stream, wrap_torch_tensor, get_wholegraph_env_fns
from .env import _OutputBuffer

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/wholegraph_ops]", *args, **kwargs)


# ──────────────────────────────────────────────
# _SampleOutput — 采样结果值对象
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class _SampleOutput:
    """
    封装单跳采样的输出张量。

    上游两个函数均用 4 路 if/elif/else 返回不同长度的 tuple，
    调用方需依赖参数顺序解包。Walpurgis 用具名字段消除歧义。
    """
    sample_offset: torch.Tensor          # [num_center+1]  采样偏移
    dest_nodes: torch.Tensor             # [total_sampled]  目标节点 id
    center_local_id: Optional[torch.Tensor] = None   # 每采样节点对应的 center 局部 id
    edge_gid: Optional[torch.Tensor] = None          # 每采样节点对应的边全局 id

    def as_tuple(self):
        """按原始上游顺序返回 tuple（兼容旧调用方）。"""
        result = [self.sample_offset, self.dest_nodes]
        if self.center_local_id is not None:
            result.append(self.center_local_id)
        if self.edge_gid is not None:
            result.append(self.edge_gid)
        return tuple(result)


# ──────────────────────────────────────────────
# 内部工具: 分配输出 buffer
# ──────────────────────────────────────────────

def _prepare_output_buffers(
    center_nodes_tensor: torch.Tensor,
    need_center_local_output: bool,
    need_edge_output: bool,
) -> Tuple[torch.Tensor, _OutputBuffer, Optional[_OutputBuffer], Optional[_OutputBuffer]]:
    """
    分配采样输出所需的 buffer（消除 unweighted/weighted 之间重复的 8 行分配代码）。
    """
    output_sample_offset = torch.empty(
        center_nodes_tensor.shape[0] + 1, device="cuda", dtype=torch.int
    )
    _dbg(
        f"_prepare_output_buffers: num_center={center_nodes_tensor.shape[0]} "
        f"need_center_local={need_center_local_output} need_edge={need_edge_output}"
    )
    dest_buf = _OutputBuffer()
    center_buf: Optional[_OutputBuffer] = None
    edge_buf: Optional[_OutputBuffer] = None
    if need_center_local_output:
        center_buf = _OutputBuffer()
    if need_edge_output:
        edge_buf = _OutputBuffer()
    return output_sample_offset, dest_buf, center_buf, edge_buf


def _collect_output(
    sample_offset: torch.Tensor,
    dest_buf: _OutputBuffer,
    center_buf: Optional[_OutputBuffer],
    edge_buf: Optional[_OutputBuffer],
) -> _SampleOutput:
    dest = dest_buf.tensor()
    center = center_buf.tensor() if center_buf is not None else None
    edge = edge_buf.tensor() if edge_buf is not None else None
    _dbg(
        f"_collect_output: dest.shape={dest.shape} "
        f"center={'yes' if center is not None else 'no'} "
        f"edge={'yes' if edge is not None else 'no'}"
    )
    return _SampleOutput(
        sample_offset=sample_offset,
        dest_nodes=dest,
        center_local_id=center,
        edge_gid=edge,
    )


# ──────────────────────────────────────────────
# unweighted_sample_without_replacement
# ──────────────────────────────────────────────

def unweighted_sample_without_replacement(
    wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
    center_nodes_tensor: torch.Tensor,
    max_sample_count: int,
    random_seed: Union[int, None] = None,
    need_center_local_output: bool = False,
    need_edge_output: bool = False,
) -> _SampleOutput:
    """
    CSR WholeGraph 无权重不重复邻居采样。

    :param wm_csr_row_ptr_tensor: CSR 行指针（WholeMemory 张量）
    :param wm_csr_col_ptr_tensor: CSR 列索引（WholeMemory 张量）
    :param center_nodes_tensor: 中心节点 id，1D，CUDA
    :param max_sample_count: 每个中心节点最多采样邻居数
    :param random_seed: 随机种子，None 则自动生成
    :param need_center_local_output: 是否输出 center_local_id 张量
    :param need_edge_output: 是否输出边全局 id 张量
    :return: _SampleOutput（可调用 .as_tuple() 获取上游兼容 tuple）
    """
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert center_nodes_tensor.dim() == 1

    if random_seed is None:
        random_seed = random.getrandbits(64)
    _dbg(
        f"unweighted_sample: center={center_nodes_tensor.shape[0]} "
        f"max_sample={max_sample_count} seed={random_seed}"
    )

    sample_offset, dest_buf, center_buf, edge_buf = _prepare_output_buffers(
        center_nodes_tensor, need_center_local_output, need_edge_output
    )

    wmb.csr_unweighted_sample_without_replacement(
        wm_csr_row_ptr_tensor,
        wm_csr_col_ptr_tensor,
        wrap_torch_tensor(center_nodes_tensor),
        max_sample_count,
        wrap_torch_tensor(sample_offset),
        dest_buf.c_handle(),
        center_buf.c_handle() if center_buf is not None else 0,
        edge_buf.c_handle() if edge_buf is not None else 0,
        random_seed,
        get_wholegraph_env_fns(),
        get_stream(),
    )
    return _collect_output(sample_offset, dest_buf, center_buf, edge_buf)


# ──────────────────────────────────────────────
# weighted_sample_without_replacement
# ──────────────────────────────────────────────

def weighted_sample_without_replacement(
    wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_weight_ptr_tensor: wmb.PyWholeMemoryTensor,
    center_nodes_tensor: torch.Tensor,
    max_sample_count: int,
    random_seed: Union[int, None] = None,
    need_center_local_output: bool = False,
    need_edge_output: bool = False,
) -> _SampleOutput:
    """
    CSR WholeGraph 有权重不重复邻居采样。

    :param wm_csr_weight_ptr_tensor: 边权重（WholeMemory 张量，shape 同 col_ptr）
    """
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.shape[0] == wm_csr_col_ptr_tensor.shape[0]
    assert center_nodes_tensor.dim() == 1

    if random_seed is None:
        random_seed = random.getrandbits(64)
    _dbg(
        f"weighted_sample: center={center_nodes_tensor.shape[0]} "
        f"max_sample={max_sample_count} seed={random_seed}"
    )

    sample_offset, dest_buf, center_buf, edge_buf = _prepare_output_buffers(
        center_nodes_tensor, need_center_local_output, need_edge_output
    )

    wmb.csr_weighted_sample_without_replacement(
        wm_csr_row_ptr_tensor,
        wm_csr_col_ptr_tensor,
        wm_csr_weight_ptr_tensor,
        wrap_torch_tensor(center_nodes_tensor),
        max_sample_count,
        wrap_torch_tensor(sample_offset),
        dest_buf.c_handle(),
        center_buf.c_handle() if center_buf is not None else 0,
        edge_buf.c_handle() if edge_buf is not None else 0,
        random_seed,
        get_wholegraph_env_fns(),
        get_stream(),
    )
    return _collect_output(sample_offset, dest_buf, center_buf, edge_buf)


# ──────────────────────────────────────────────
# CPU 随机数生成工具
# ──────────────────────────────────────────────

def generate_random_positive_int_cpu(
    random_seed: int, sub_sequence: int, output_random_value_count: int
) -> torch.Tensor:
    """在 CPU 上生成指定数量的正整数随机数。"""
    output = torch.empty((output_random_value_count,), dtype=torch.int)
    wmb.host_generate_random_positive_int(
        random_seed, sub_sequence, wrap_torch_tensor(output)
    )
    return output


def generate_exponential_distribution_negative_float_cpu(
    random_seed: int, sub_sequence: int, output_random_value_count: int
) -> torch.Tensor:
    """在 CPU 上生成指定数量的负指数分布浮点随机数。"""
    output = torch.empty((output_random_value_count,), dtype=torch.float)
    wmb.host_generate_exponential_distribution_negative_float(
        random_seed, sub_sequence, wrap_torch_tensor(output)
    )
    return output
