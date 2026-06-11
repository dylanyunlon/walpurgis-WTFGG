"""
initialize.py — bd703b3 迁移: WholeGraph 环境初始化与析构

上游来源: python/pylibwholegraph/pylibwholegraph/torch/initialize.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _WalpurgisInitState dataclass 追踪初始化阶段，防止重复 init / finalize 不对称
- init_torch_env 的 MASTER_ADDR/PORT fallback 改为 _dbg 而非 print("[WARNING] ...")
- finalize() 加 distributed.is_initialized() 保护（上游已有，Walpurgis 保留并加日志）
- 全链路 WALPURGIS_DEBUG=1 断点 print:
  init 参数 / 环境变量设置 / comm 创建 / finalize 阶段
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

from .comm import (
    set_world_info,
    get_global_communicator,
    get_local_node_communicator,
    reset_communicators,
    WholeMemoryCommunicator,
)
from .env_fn_utils import str_to_wmb_wholememory_log_level

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/initialize]", *args, **kwargs)


# ──────────────────────────────────────────────
# _WalpurgisInitState — 初始化阶段追踪
# ──────────────────────────────────────────────

@dataclass
class _WalpurgisInitState:
    """
    追踪 WholeGraph 初始化状态。
    上游无此追踪，多次 init() 会导致 wmb.init 重入；
    finalize() 后再 init 亦无保护。
    """
    wmb_initialized: bool = False
    torch_env_initialized: bool = False
    world_rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_size: int = 1

    def reset(self) -> None:
        self.wmb_initialized = False
        self.torch_env_initialized = False


_init_state: _WalpurgisInitState = _WalpurgisInitState()


# ──────────────────────────────────────────────
# init — 最小化初始化（无 torch.distributed）
# ──────────────────────────────────────────────

def init(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    wm_log_level: str = "info",
) -> None:
    """
    最小化 WholeGraph 初始化（不初始化 torch.distributed）。
    适用于已外部初始化 distributed 的场景。
    """
    _dbg(
        f"init: world={world_rank}/{world_size} local={local_rank}/{local_size} "
        f"log_level={wm_log_level}"
    )
    if _init_state.wmb_initialized:
        _dbg("init: wmb 已初始化，跳过重入")
    else:
        wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
        _init_state.wmb_initialized = True

    set_world_info(world_rank, world_size, local_rank, local_size)
    _init_state.world_rank = world_rank
    _init_state.world_size = world_size
    _init_state.local_rank = local_rank
    _init_state.local_size = local_size


# ──────────────────────────────────────────────
# init_torch_env — 完整 torch 分布式 + WholeGraph 初始化
# ──────────────────────────────────────────────

def init_torch_env(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    wm_log_level: str = "info",
) -> None:
    """
    初始化 torch.distributed 和 WholeGraph 环境。

    :param world_rank: 当前进程全局 rank
    :param world_size: 全局进程总数
    :param local_rank: 当前机器上的本地 rank
    :param local_size: 当前机器上的本地进程数
    :param wm_log_level: WholeGraph 日志级别
    """
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if "MASTER_ADDR" not in os.environ:
        # 上游 print("[WARNING] ...")，Walpurgis 降级为 _dbg
        _dbg(f"MASTER_ADDR 未设置，使用 localhost (rank={world_rank})")
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        _dbg(f"MASTER_PORT 未设置，使用 12335 (rank={world_rank})")
        os.environ["MASTER_PORT"] = "12335"

    _dbg(
        f"init_torch_env: world={world_rank}/{world_size} "
        f"local={local_rank}/{local_size} log_level={wm_log_level}"
    )

    if not _init_state.wmb_initialized:
        wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
        _init_state.wmb_initialized = True

    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        _dbg("init_torch_env: torch.distributed 初始化完成")

    set_world_info(world_rank, world_size, local_rank, local_size)
    _init_state.torch_env_initialized = True
    _init_state.world_rank = world_rank
    _init_state.world_size = world_size
    _init_state.local_rank = local_rank
    _init_state.local_size = local_size


# ──────────────────────────────────────────────
# init_torch_env_and_create_wm_comm
# ──────────────────────────────────────────────

def init_torch_env_and_create_wm_comm(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    distributed_backend_type: str = "nccl",
    wm_log_level: str = "info",
) -> Tuple[WholeMemoryCommunicator, WholeMemoryCommunicator]:
    """
    初始化 torch + WholeGraph 并创建全局/本地通信器。

    :return: (global_comm, local_node_comm)
    """
    _dbg(
        f"init_torch_env_and_create_wm_comm: backend={distributed_backend_type}"
    )
    init_torch_env(world_rank, world_size, local_rank, local_size, wm_log_level)
    global_comm = get_global_communicator(distributed_backend_type)
    local_comm = get_local_node_communicator()
    _dbg(
        f"init_torch_env_and_create_wm_comm: "
        f"global_comm size={global_comm.get_size()} "
        f"local_comm size={local_comm.get_size()}"
    )
    return global_comm, local_comm


# ──────────────────────────────────────────────
# finalize
# ──────────────────────────────────────────────

def finalize() -> None:
    """
    析构 WholeGraph 环境。

    上游实现：
        wmb.finalize()
        reset_communicators()
        torch.distributed.destroy_process_group() if initialized
    Walpurgis 加 _init_state 追踪，防止多次 finalize。
    """
    _dbg("finalize: 开始")
    if _init_state.wmb_initialized:
        wmb.finalize()
        _init_state.wmb_initialized = False
        _dbg("finalize: wmb.finalize() 完成")
    else:
        _dbg("finalize: wmb 未初始化，跳过")

    reset_communicators()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        _dbg("finalize: torch.distributed 进程组已销毁")
    else:
        _dbg("finalize: torch.distributed 未初始化，跳过")

    _init_state.reset()
    _dbg("finalize: 完成")
