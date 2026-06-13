"""
异构GPU网格规划 — 从Neuron_SP/deepspeed/compile/custom_ops/hetero_mesh.py鲁迅拿法
改写点 (~20%):
  1. 去除deepspeed.comm依赖, 改用torch.distributed或单机多卡直接probe
  2. 新增NUMA拓扑感知: 同NUMA node的GPU优先分组(walpurgis服务器GPU全在NUMA node1)
  3. 新增图卷积特有的chunk_split: 按节点数而非序列长度切分
  4. 全链路_dbg() + dump_struct_state()断点调试 (原代码仅有logger)
  5. 新增memory_pressure_score: 结合显存剩余率与带宽做负载评估
鲁迅: 拿来主义——只要送来的, 都可以拿来, 但要挑选。
"""
import os
import math
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch

from .. import _dbg, _is_debug, dump_struct_state

_MODULE = "hetero_mesh"


# ═══ GPU层级信息 (从Neuron_SP移植, 改写: 新增numa_node + memory_pressure_score) ═══
@dataclass
class GPUTierInfo:
    """单个GPU的硬件能力画像"""
    rank: int
    device_name: str = ""
    compute_capability: Tuple[int, int] = (0, 0)
    memory_total_gb: float = 0.0
    memory_free_gb: float = 0.0        # 改写: 运行时动态剩余
    memory_bandwidth_gbps: float = 0.0
    nvlink_available: bool = False
    pcie_bandwidth_gbps: float = 0.0
    tier: int = 0
    numa_node: int = -1                 # 改写: NUMA拓扑

    def compute_score(self) -> float:
        """综合性能评分 (0~100)"""
        mem_bw_score = self.memory_bandwidth_gbps / 100.0
        compute_score = (self.compute_capability[0] * 10
                         + self.compute_capability[1])
        link_bonus = 2.0 if self.nvlink_available else 0.0
        return 0.6 * mem_bw_score + 0.3 * compute_score + 0.1 * link_bonus

    def memory_pressure_score(self) -> float:
        """显存压力评分 (0=空闲, 1=满载) — 改写: 新增指标"""
        if self.memory_total_gb <= 0:
            return 0.5
        used = self.memory_total_gb - self.memory_free_gb
        return min(1.0, max(0.0, used / self.memory_total_gb))

    def __repr__(self):
        return (f"GPU[{self.rank}] {self.device_name} "
                f"cc={self.compute_capability} "
                f"mem={self.memory_free_gb:.1f}/{self.memory_total_gb:.1f}GB "
                f"bw={self.memory_bandwidth_gbps}GB/s "
                f"tier={self.tier} numa={self.numa_node}")


# ═══ 带宽查找表 (从Neuron_SP移植, 改写: 新增B系列+PCIe带宽) ═══
_BW_TABLE = {
    "H100": (3350, 64),    # (HBM带宽, PCIe Gen5 x16)
    "H200": (4800, 64),
    "B100": (8000, 64),
    "B200": (8000, 128),
    "A100": (2039, 64),
    "A6000": (768, 32),     # PCIe Gen4 x16
    "L40": (864, 64),
    "L40S": (864, 64),
    "A40": (696, 32),
    "V100": (900, 32),
    "RTX 4090": (1008, 32),
    "RTX 3090": (936, 32),
    "RTX 6000": (960, 32),
}


def probe_local_gpu(rank: int = 0) -> GPUTierInfo:
    """探测本地GPU硬件信息 (改写: 新增NUMA探测 + 动态显存)"""
    info = GPUTierInfo(rank=rank)

    if not torch.cuda.is_available():
        info.device_name = "cpu"
        _dbg(f"{_MODULE}.probe", "No CUDA, returning CPU info", _MODULE)
        return info

    dev = rank if rank < torch.cuda.device_count() else 0
    props = torch.cuda.get_device_properties(dev)
    info.device_name = props.name
    info.compute_capability = (props.major, props.minor)
    info.memory_total_gb = props.total_mem / (1024 ** 3)

    # 动态显存探测 (改写: 实时free mem)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        info.memory_free_gb = free_bytes / (1024 ** 3)
    except Exception:
        info.memory_free_gb = info.memory_total_gb * 0.8  # fallback估计

    # 带宽查表
    for name_frag, (hbm_bw, pcie_bw) in _BW_TABLE.items():
        if name_frag.lower() in props.name.lower():
            info.memory_bandwidth_gbps = hbm_bw
            info.pcie_bandwidth_gbps = pcie_bw
            break

    # tier分级
    if props.major >= 9:
        info.tier = 3  # Hopper/Blackwell
    elif props.major >= 8:
        info.tier = 2  # Ampere
    else:
        info.tier = 1  # Volta or older

    info.nvlink_available = (torch.cuda.device_count() > 1
                             and info.tier >= 2)

    # NUMA探测 (改写: 从/sys读取)
    try:
        numa_path = f"/sys/bus/pci/devices/0000:{props.pci_bus_id}/numa_node"
        if os.path.exists(numa_path):
            info.numa_node = int(open(numa_path).read().strip())
        else:
            # 备用方案: nvidia-smi拓扑推断
            info.numa_node = dev % 2  # 简化推断
    except Exception:
        info.numa_node = -1

    _dbg(f"{_MODULE}.probe.gpu{rank}", str(info), _MODULE)
    return info


# ═══ 异构网格规划 (从Neuron_SP移植, 改写: 新增NUMA感知分组 + 图节点chunk) ═══
@dataclass
class HeteroMeshPlan:
    """异构GPU分组方案"""
    num_gpus: int
    groups: List[List[int]]               # GPU分组 (每组协同做图卷积)
    tier_infos: Dict[int, GPUTierInfo] = field(default_factory=dict)
    node_chunk_weights: Dict[int, List[float]] = field(default_factory=dict)
    # 改写: 图卷积特有 — 每个GPU处理多少个节点
    node_assignments: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"HeteroMesh: {self.num_gpus} GPUs, {len(self.groups)} groups"]
        for gid, ranks in enumerate(self.groups):
            tiers = [self.tier_infos.get(r, GPUTierInfo(rank=r)).tier for r in ranks]
            scores = [f"{self.tier_infos.get(r, GPUTierInfo(rank=r)).compute_score():.1f}" for r in ranks]
            lines.append(f"  group[{gid}]: ranks={ranks} tiers={tiers} scores={scores}")
        return "\n".join(lines)


def plan_hetero_mesh(
    num_gpus: int,
    tier_infos: Optional[Dict[int, GPUTierInfo]] = None,
    strategy: str = "numa_aware",
    num_nodes: int = 0,
) -> HeteroMeshPlan:
    """规划异构GPU网格分组

    改写点 (vs Neuron_SP):
      1. 新增 numa_aware 策略: 同NUMA node的GPU优先分组
      2. 新增 node_assignments: 按GPU性能分配图节点
      3. 去除sp_size/dp_size概念, 改为num_gpus直接分组
    """
    if tier_infos is None:
        tier_infos = {}
        for r in range(num_gpus):
            tier_infos[r] = probe_local_gpu(r)

    _dbg(f"{_MODULE}.plan.strategy", strategy, _MODULE)
    dump_struct_state(
        "hetero_mesh_plan_input",
        num_gpus=num_gpus,
        strategy=strategy,
        num_nodes=num_nodes,
        tier_count=len(tier_infos))

    if strategy == "single":
        # 所有GPU一组
        groups = [list(range(num_gpus))]

    elif strategy == "numa_aware":
        # 改写: 按NUMA node分组, 同NUMA的GPU协同
        numa_groups: Dict[int, List[int]] = {}
        for r, info in tier_infos.items():
            node = info.numa_node if info.numa_node >= 0 else 0
            numa_groups.setdefault(node, []).append(r)

        groups = list(numa_groups.values())
        # 每组内按compute_score排序
        for g in groups:
            g.sort(key=lambda r: tier_infos[r].compute_score(), reverse=True)

        _dbg(f"{_MODULE}.plan.numa_groups",
             f"{len(groups)} groups: {[len(g) for g in groups]}", _MODULE)

    elif strategy == "capability_sort":
        # 从Neuron_SP直接搬: 按性能交错分组
        sorted_ranks = sorted(
            tier_infos.keys(),
            key=lambda r: tier_infos[r].compute_score(),
            reverse=True)
        # 分成2组 (或更多)
        num_groups = max(1, num_gpus // 2)
        groups = [[] for _ in range(num_groups)]
        for i, rank in enumerate(sorted_ranks):
            groups[i % num_groups].append(rank)
        for g in groups:
            g.sort()

    elif strategy == "contiguous":
        groups = [[r] for r in range(num_gpus)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 计算每个GPU应分配的图节点数 (改写: 图特有)
    node_chunk_weights = {}
    node_assignments = {}
    if num_nodes > 0:
        for gid, ranks in enumerate(groups):
            scores = [tier_infos.get(r, GPUTierInfo(rank=r)).compute_score()
                      for r in ranks]
            total = sum(scores) or 1.0
            weights = [s / total for s in scores]
            node_chunk_weights[gid] = weights

            # 按权重分配节点
            offset = 0
            for i, (rank, w) in enumerate(zip(ranks, weights)):
                chunk_size = int(num_nodes * w)
                if i == len(ranks) - 1:
                    chunk_size = num_nodes - offset  # 最后一个GPU取剩余
                node_assignments[rank] = (offset, offset + chunk_size)
                offset += chunk_size

    plan = HeteroMeshPlan(
        num_gpus=num_gpus,
        groups=groups,
        tier_infos=tier_infos,
        node_chunk_weights=node_chunk_weights,
        node_assignments=node_assignments)

    _dbg(f"{_MODULE}.plan.result", plan.summary(), _MODULE)
    dump_struct_state(
        "hetero_mesh_plan_result",
        num_groups=len(groups),
        group_sizes=[len(g) for g in groups],
        node_assignments=str(node_assignments)[:200])

    return plan


# ═══ 全局单例 (从Neuron_SP移植) ═══
_MESH_PLAN: Optional[HeteroMeshPlan] = None
_MESH_LOCK = threading.Lock()


def get_mesh_plan() -> Optional[HeteroMeshPlan]:
    return _MESH_PLAN


def init_mesh(strategy: str = "numa_aware", num_nodes: int = 0) -> HeteroMeshPlan:
    """初始化异构网格 (线程安全)"""
    global _MESH_PLAN
    with _MESH_LOCK:
        if _MESH_PLAN is not None:
            return _MESH_PLAN

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        tier_infos = {}
        for r in range(num_gpus):
            tier_infos[r] = probe_local_gpu(r)

        _MESH_PLAN = plan_hetero_mesh(
            num_gpus, tier_infos, strategy, num_nodes)

        if _is_debug():
            print(f"\n[HETERO-MESH] Initialized:", flush=True)
            print(_MESH_PLAN.summary(), flush=True)

        return _MESH_PLAN


def reset_mesh():
    global _MESH_PLAN
    with _MESH_LOCK:
        _MESH_PLAN = None


def get_node_range(rank: int) -> Tuple[int, int]:
    """获取指定GPU负责的节点范围 [start, end)"""
    plan = _MESH_PLAN
    if plan is None or rank not in plan.node_assignments:
        return (0, 0)
    return plan.node_assignments[rank]


# ═══ 自检 (改写: 新增, 用于验证分组正确性) ═══
def self_check():
    """验证mesh规划的一致性"""
    plan = _MESH_PLAN
    if plan is None:
        return True

    all_ranks = set()
    for g in plan.groups:
        for r in g:
            assert r not in all_ranks, f"Rank {r} appears in multiple groups"
            all_ranks.add(r)

    if plan.node_assignments:
        ranges = sorted(plan.node_assignments.values())
        for i in range(1, len(ranges)):
            assert ranges[i][0] >= ranges[i-1][1], \
                f"Overlapping node assignments: {ranges[i-1]} and {ranges[i]}"

    _dbg(f"{_MODULE}.self_check", "PASSED", _MODULE)
    return True
