"""
comm.py — bd703b3 迁移: WholeMemory 通信器层

上游来源: python/pylibwholegraph/pylibwholegraph/torch/comm.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _WalpurgisCommRegistry dataclass 替代五个模块级散落变量
  (global_communicators / local_node_communicator / ... / all_comm_*)
  统一注册表，rank 信息与 comm 对象聚合管理
- _CommKey: str → NamedTuple，防止 distributed_backend 字符串拼写错误进入字典
- WholeMemoryCommunicator.destroy() 加防重入保护（上游可能 double-free）
- 全链路 WALPURGIS_DEBUG=1 断点 print: 注册表初始化 / comm 创建 / split / destroy
"""

import os
import torch
import torch.distributed as dist
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb
from dataclasses import dataclass, field
from typing import Dict, Optional, NamedTuple

from .env_fn_utils import (
    str_to_wmb_wholememory_distributed_backend_type,
    wholememory_distributed_backend_type_to_str,
    str_to_wmb_wholememory_memory_type,
    str_to_wmb_wholememory_location,
)

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/comm]", *args, **kwargs)


# ──────────────────────────────────────────────
# _CommKey — 防拼写错误的通信器键
# ──────────────────────────────────────────────

class _CommKey(NamedTuple):
    backend: str   # e.g. "nccl", "nvshmem"


# ──────────────────────────────────────────────
# _WalpurgisCommRegistry — 集中管理全局通信器状态
# ──────────────────────────────────────────────

@dataclass
class _WalpurgisCommRegistry:
    """
    封装上游五个模块级变量:
        global_communicators: Dict[str, WholeMemoryCommunicator]
        local_node_communicator: Optional[...]
        local_device_communicator: Optional[...]
        local_mnnvl_communicator: Optional[...]
        all_comm_{world_rank,world_size,local_rank,local_size}: int
    """
    global_comms: Dict[_CommKey, "WholeMemoryCommunicator"] = field(default_factory=dict)
    local_node_comm: Optional["WholeMemoryCommunicator"] = None
    local_device_comm: Optional["WholeMemoryCommunicator"] = None
    local_mnnvl_comm: Optional["WholeMemoryCommunicator"] = None
    world_rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_size: int = 1

    def reset(self) -> None:
        _dbg("_WalpurgisCommRegistry.reset()")
        self.global_comms.clear()
        self.local_node_comm = None
        self.local_device_comm = None
        self.local_mnnvl_comm = None
        self.world_rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_size = 1


_registry: _WalpurgisCommRegistry = _WalpurgisCommRegistry()


# ──────────────────────────────────────────────
# 公共 API: 设置 world 信息
# ──────────────────────────────────────────────

def set_world_info(
    world_rank: int, world_size: int, local_rank: int, local_size: int
) -> None:
    """设置全局 world 信息，用于构建各类通信器。"""
    _registry.world_rank = world_rank
    _registry.world_size = world_size
    _registry.local_rank = local_rank
    _registry.local_size = local_size
    _dbg(
        f"set_world_info: world={world_rank}/{world_size} local={local_rank}/{local_size}"
    )


def reset_communicators() -> None:
    """重置所有通信器状态，通常在 finalize() 时调用。"""
    _registry.reset()


# ──────────────────────────────────────────────
# WholeMemoryCommunicator
# ──────────────────────────────────────────────

class WholeMemoryCommunicator:
    """
    WholeMemory 通信器包装。

    不应直接构造；请使用:
        create_group_communicator / get_global_communicator /
        get_local_node_communicator / get_local_device_communicator
    """

    def __init__(self, wmb_comm: wmb.PyWholeMemoryComm):
        self.wmb_comm: Optional[wmb.PyWholeMemoryComm] = wmb_comm
        _dbg(f"WholeMemoryCommunicator created: rank={wmb_comm.get_rank()} size={wmb_comm.get_size()}")

    def get_rank(self) -> int:
        return self.wmb_comm.get_rank()

    def get_size(self) -> int:
        return self.wmb_comm.get_size()

    def get_clique_info(self):
        """返回当前进程所在 mnnvl clique 信息。"""
        return self.wmb_comm.get_clique_info()

    def barrier(self) -> None:
        """
        在通信器上执行 barrier，使用内部 CUDA stream 并同步主机。
        若有其他 stream 上的工作需在 barrier 前完成，请先 sync 该 stream。
        """
        _dbg(f"barrier: rank={self.get_rank()}")
        return self.wmb_comm.barrier()

    def support_type_location(self, memory_type: str, memory_location: str) -> bool:
        """检查该通信器是否支持指定的内存类型/位置组合。"""
        wm_memory_type = str_to_wmb_wholememory_memory_type(memory_type)
        wm_location = str_to_wmb_wholememory_location(memory_location)
        return self.wmb_comm.support_type_location(wm_memory_type, wm_location)

    def destroy(self) -> None:
        """销毁通信器，防重入保护（上游无此保护）。"""
        if self.wmb_comm is not None:
            _dbg(f"WholeMemoryCommunicator.destroy: rank={self.wmb_comm.get_rank()}")
            wmb.destroy_communicator(self.wmb_comm)
            self.wmb_comm = None
        else:
            _dbg("WholeMemoryCommunicator.destroy: 已销毁，跳过重入")

    @property
    def distributed_backend(self) -> str:
        return wholememory_distributed_backend_type_to_str(
            self.wmb_comm.get_distributed_backend()
        )

    @distributed_backend.setter
    def distributed_backend(self, value: str) -> None:
        self.wmb_comm.set_distributed_backend(
            str_to_wmb_wholememory_distributed_backend_type(value)
        )


# ──────────────────────────────────────────────
# 公共 API: 创建通信器
# ──────────────────────────────────────────────

def create_group_communicator(
    group_size: int = -1, comm_stride: int = 1
) -> WholeMemoryCommunicator:
    """
    创建 WholeMemory 通信器组。

    例: 24 ranks, group_size=4, comm_stride=2 →
        [0,2,4,6], [1,3,5,7], [8,10,12,14], ...
    """
    world_size = dist.get_world_size()
    if group_size == -1:
        group_size = world_size
    strided_group_size = group_size * comm_stride
    assert world_size % strided_group_size == 0, (
        f"world_size={world_size} 不能被 group_size*comm_stride={strided_group_size} 整除"
    )
    strided_group_count = world_size // strided_group_size
    world_rank = dist.get_rank()
    strided_group_idx = world_rank // strided_group_size
    idx_in_strided_group = world_rank % strided_group_size
    inner_group_idx = idx_in_strided_group % comm_stride
    idx_in_group = idx_in_strided_group // comm_stride

    _dbg(
        f"create_group_communicator: world_size={world_size} group_size={group_size} "
        f"comm_stride={comm_stride} world_rank={world_rank} idx_in_group={idx_in_group}"
    )

    wm_uid = wmb.PyWholeMemoryUniqueID()
    for strided_group in range(strided_group_count):
        for inner_group in range(comm_stride):
            group_root_rank = strided_group * strided_group_size + inner_group
            if world_rank == group_root_rank:
                tmp_wm_uid = wmb.create_unique_id()
            else:
                tmp_wm_uid = wmb.PyWholeMemoryUniqueID()
            uid_th = torch.utils.dlpack.from_dlpack(tmp_wm_uid.__dlpack__())
            uid_th_cuda = uid_th.cuda()
            dist.broadcast(uid_th_cuda, group_root_rank)
            uid_th.copy_(uid_th_cuda.cpu())
            if strided_group_idx == strided_group and inner_group_idx == inner_group:
                wm_uid_th = torch.utils.dlpack.from_dlpack(wm_uid.__dlpack__())
                wm_uid_th.copy_(uid_th)

    wm_comm = wmb.create_communicator(wm_uid, idx_in_group, group_size)
    _dbg(f"create_group_communicator: 创建成功 rank={idx_in_group} size={group_size}")
    return WholeMemoryCommunicator(wm_comm)


def split_communicator(
    comm: WholeMemoryCommunicator, color: int, key: int = 0
) -> Optional[WholeMemoryCommunicator]:
    """
    按 color 分割通信器，color 相同的 ranks 组成新通信器。
    key 决定新通信器中的 rank 顺序（小 key → 小 rank）。
    color < 0 返回 None（该 rank 不参与任何新通信器）。
    """
    if not isinstance(color, int) or not isinstance(key, int):
        raise TypeError("color 和 key 必须为 int")
    if color < 0:
        _dbg(f"split_communicator: color={color} < 0，返回 None")
        return None
    _dbg(f"split_communicator: color={color} key={key}")
    new_wm_comm = wmb.split_communicator(comm.wmb_comm, color, key)
    return WholeMemoryCommunicator(new_wm_comm)


def destroy_communicator(wm_comm: Optional[WholeMemoryCommunicator]) -> None:
    """销毁 WholeMemoryCommunicator（None 安全）。"""
    if wm_comm is not None:
        wm_comm.destroy()


# ──────────────────────────────────────────────
# 内置通信器获取（惰性创建，缓存于 registry）
# ──────────────────────────────────────────────

def get_global_communicator(
    distributed_backend: str = "nccl",
) -> WholeMemoryCommunicator:
    """获取包含所有 GPU 的全局通信器（惰性创建）。"""
    key = _CommKey(backend=distributed_backend)
    if key not in _registry.global_comms:
        _dbg(f"get_global_communicator: 创建 backend={distributed_backend}")
        wm_comm = create_group_communicator()
        wm_comm.distributed_backend = distributed_backend
        _registry.global_comms[key] = wm_comm
    return _registry.global_comms[key]


def get_local_node_communicator() -> WholeMemoryCommunicator:
    """获取同一物理节点内所有 GPU 的通信器（惰性创建）。"""
    if _registry.local_node_comm is None:
        _dbg("get_local_node_communicator: 创建")
        global_comm = get_global_communicator()
        node_id = _registry.world_rank // _registry.local_size
        _registry.local_node_comm = split_communicator(
            global_comm, node_id, _registry.local_rank
        )
    return _registry.local_node_comm


def get_local_device_communicator() -> WholeMemoryCommunicator:
    """获取每个 GPU 自身的单设备通信器（intra-GPU，惰性创建）。"""
    if _registry.local_device_comm is None:
        _dbg("get_local_device_communicator: 创建")
        global_comm = get_global_communicator()
        _registry.local_device_comm = split_communicator(
            global_comm, _registry.local_rank, _registry.world_rank
        )
    return _registry.local_device_comm


def get_local_mnnvl_communicator() -> Optional[WholeMemoryCommunicator]:
    """获取 mnnvl 域通信器（若设备不在 mnnvl 域则返回 None）。"""
    if _registry.local_mnnvl_comm is None:
        _dbg("get_local_mnnvl_communicator: 创建")
        global_comm = get_global_communicator()
        clique_info = global_comm.get_clique_info()
        is_in_clique = clique_info[0]
        clique_id = clique_info[4]
        if is_in_clique <= 0:
            _dbg("get_local_mnnvl_communicator: 不在 mnnvl 域，返回 None")
            return None
        _registry.local_mnnvl_comm = split_communicator(
            global_comm, clique_id, _registry.world_rank
        )
    return _registry.local_mnnvl_comm
