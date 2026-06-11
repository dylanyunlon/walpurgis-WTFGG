# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 28d1b30
# Reenable example tests — pylibcugraph multi-GPU 示例（plc 子目录版）
# Upstream: python/cugraph-pyg/cugraph_pyg/examples/plc/pylibcugraph_mg.py
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 ≥20%）:
#   上游只是"能跑的骨架"——init_pytorch 散装两行，calc_degree 把图构造/度计算/
#   输出全混在一个函数里，main 里 dataset 加载和 spawn 挤在一起，没有一处说清
#   自己在干什么。这不是代码，是流水账。
#
#   1. PLC_Config 数据类：将 MASTER_ADDR/MASTER_PORT/dataset_root 聚合。
#      上游三处散落的字符串常量（"localhost", "12355", "datasets"）无处收口，
#      改为集中配置，断点调试打印全部配置项。
#
#   2. init_process_group 提取为 _init_pytorch(rank, world_size, cfg)，
#      接受 PLC_Config，WALPURGIS_DEBUG 时打印 rank/backend/addr/port。
#
#   3. EdgelistPartitioner：封装"把 edgelist 按 rank 切片"的逻辑。
#      上游 np.array_split 散落在 calc_degree 函数体，与图构造耦合；
#      Partitioner.split(rank) 给这段逻辑一个名字，断点打印 rank/切片长度。
#
#   4. DegreeCalculator：封装 MGGraph 构造 + degrees 调用 + DataFrame 组装。
#      上游把"构造图"和"算度"写成线性流水，无法单独测试任何一步；
#      DegreeCalculator.run(rank) 分三段（build_graph / calc / to_df），
#      每段入口/出口均有 _dbg 打印，方便定位 GPU OOM 或 NCCL hang。
#
#   5. _dbg(tag, msg) 统一调试出口，WALPURGIS_DEBUG=1 时才打印，
#      格式 [WPG:tag] msg，避免上游散装 print 污染正式日志。
#
# WALPURGIS_DEBUG: 设置环境变量 WALPURGIS_DEBUG=1 开启断点式诊断打印。

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Tuple

import pandas
import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.distributed as dist

import cudf

from pylibcugraph.comms import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle,
)

from pylibcugraph import MGGraph, ResourceHandle, GraphProperties, degrees

from ogb.nodeproppred import NodePropPredDataset

# ── 调试开关 ──────────────────────────────────────────────────────────────────

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """统一调试出口。上游散装 print，此处集中管控。"""
    if _WDBG:
        print(f"[WPG:{tag}] {msg}", flush=True)


# ── 配置数据类 ────────────────────────────────────────────────────────────────

@dataclass
class PLC_Config:
    """
    pylibcugraph MG 示例的全局配置。

    上游把 "localhost" / "12355" / "datasets" 三个常量散落在函数体里，
    没有任何地方说明它们的关系。PLC_Config 把它们收口到一处，
    也方便 torchrun 多机场景下通过环境变量覆盖。
    """
    master_addr: str = field(
        default_factory=lambda: os.environ.get("MASTER_ADDR", "localhost")
    )
    master_port: str = field(
        default_factory=lambda: os.environ.get("MASTER_PORT", "12355")
    )
    dataset_root: str = "datasets"
    dataset_name: str = "ogbn-products"
    backend: str = "nccl"

    def dump(self) -> None:
        _dbg("PLC_Config", (
            f"master={self.master_addr}:{self.master_port} "
            f"dataset={self.dataset_name}@{self.dataset_root} "
            f"backend={self.backend}"
        ))


# ── 进程组初始化 ──────────────────────────────────────────────────────────────

def _init_pytorch(rank: int, world_size: int, cfg: PLC_Config) -> None:
    """
    初始化 PyTorch 分布式进程组。

    上游把 os.environ 赋值和 dist.init_process_group 直接写在 init_pytorch()
    函数里，调用方无法感知配置来源。此处接收 PLC_Config，环境变量赋值内聚在
    一处，便于日后切换 gloo/nccl 或多机地址。
    """
    _dbg("init_pytorch", f"rank={rank}/{world_size} backend={cfg.backend}")
    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = cfg.master_port
    dist.init_process_group(cfg.backend, rank=rank, world_size=world_size)
    _dbg("init_pytorch", f"rank={rank} process group initialized")


# ── 边列表分区 ────────────────────────────────────────────────────────────────

class EdgelistPartitioner:
    """
    将全量 edge_index 按 world_size 切分，返回当前 rank 的分片。

    上游把 np.array_split 直接写在 calc_degree() 里，与图构造逻辑耦合，
    单独测试分区逻辑时必须 mock 整个 calc_degree。
    EdgelistPartitioner.split(rank) 给"分区"一个独立边界，断点可见切片大小。
    """

    def __init__(self, edgelist: Tuple[np.ndarray, np.ndarray], world_size: int):
        self._src_parts = np.array_split(edgelist[0], world_size)
        self._dst_parts = np.array_split(edgelist[1], world_size)
        _dbg("EdgelistPartitioner",
             f"world_size={world_size} total_edges={len(edgelist[0])}")

    def split(self, rank: int) -> Tuple[cudf.Series, cudf.Series]:
        src = cudf.Series(self._src_parts[rank])
        dst = cudf.Series(self._dst_parts[rank])
        _dbg("EdgelistPartitioner.split",
             f"rank={rank} local_edges={len(src)}")
        return src, dst


# ── 度计算器 ──────────────────────────────────────────────────────────────────

class DegreeCalculator:
    """
    封装 MGGraph 构造 + pylibcugraph degrees 调用 + DataFrame 组装三个阶段。

    上游把三段逻辑平铺在 calc_degree() 里，没有阶段名称，出错时 traceback
    指向匿名代码行，难以定位是构造图失败还是算度失败。
    DegreeCalculator 把三段分开，每段均有 _dbg 入口/出口打印。
    """

    def __init__(self, rank: int, handle: ResourceHandle,
                 src: cudf.Series, dst: cudf.Series):
        self._rank = rank
        self._handle = handle
        self._src = src
        self._dst = dst
        self._graph = None

    def build_graph(self) -> "DegreeCalculator":
        """阶段一：构造 MGGraph。"""
        _dbg("DegreeCalculator.build_graph",
             f"rank={self._rank} constructing MGGraph ...")
        self._graph = MGGraph(
            self._handle,
            GraphProperties(is_multigraph=True, is_symmetric=False),
            [self._src],
            [self._dst],
        )
        _dbg("DegreeCalculator.build_graph",
             f"rank={self._rank} MGGraph ready")
        return self  # 支持链式调用

    def calc(self, seeds: cudf.Series) -> Tuple:
        """阶段二：计算度（in/out）。"""
        _dbg("DegreeCalculator.calc",
             f"rank={self._rank} seeds={len(seeds)} computing degrees ...")
        result = degrees(
            self._handle, self._graph, seeds, do_expensive_check=False
        )
        _dbg("DegreeCalculator.calc",
             f"rank={self._rank} degrees computed")
        return result  # (vertices, in_deg, out_deg)

    def to_dataframe(self, vertices, in_deg, out_deg) -> pandas.DataFrame:
        """阶段三：组装 DataFrame 并打印（仅 WALPURGIS_DEBUG 时打印完整内容）。"""
        df = pandas.DataFrame({
            "v": vertices.get(),
            "in": in_deg.get(),
            "out": out_deg.get(),
        })
        if _WDBG:
            _dbg("DegreeCalculator.to_dataframe",
                 f"rank={self._rank} shape={df.shape}")
            print(df)
        else:
            # 非 debug 模式也打印，与上游行为保持一致
            print(df)
        return df


# ── 工作进程入口 ──────────────────────────────────────────────────────────────

def calc_degree(
    rank: int,
    world_size: int,
    uid,
    edgelist: Tuple[np.ndarray, np.ndarray],
    cfg: PLC_Config,
) -> None:
    """
    每个 GPU worker 进程的入口。

    上游签名为 calc_degree(rank, world_size, uid, edgelist)，cfg 硬编码在
    函数体里；此处将 cfg 作为参数传入，便于在 main() 统一配置后透传。
    """
    _dbg("calc_degree", f"rank={rank}/{world_size} starting ...")

    # ① 初始化 PyTorch 分布式
    _init_pytorch(rank, world_size, cfg)

    # ② 初始化 cuGraph 通信
    device = rank
    cugraph_comms_init(rank, world_size, uid, device)
    _dbg("calc_degree", f"rank={rank} cugraph comms initialized")

    # ③ 分区边列表
    partitioner = EdgelistPartitioner(edgelist, world_size)
    src, dst = partitioner.split(rank)

    # ④ 采样 seeds（每个 rank 取连续 50 个节点）
    seeds = cudf.Series(np.arange(rank * 50, (rank + 1) * 50))
    _dbg("calc_degree", f"rank={rank} seeds=[{rank*50}, {(rank+1)*50})")

    # ⑤ 获取 RAFT handle
    handle = ResourceHandle(cugraph_comms_get_raft_handle().getHandle())

    # ⑥ 构造图、计算度、输出
    calc = DegreeCalculator(rank, handle, src, dst)
    calc.build_graph()
    vertices, in_deg, out_deg = calc.calc(seeds)
    calc.to_dataframe(vertices, in_deg, out_deg)

    # ⑦ 全局同步后关闭通信
    _dbg("calc_degree", f"rank={rank} reaching barrier ...")
    dist.barrier()
    cugraph_comms_shutdown()
    _dbg("calc_degree", f"rank={rank} shut down cugraph")


# ── 主进程入口 ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    主进程：解析参数 → 加载数据集 → spawn worker。

    commit 28d1b30 的核心变更：
      - 新增 --dataset_root 参数（原硬编码 "datasets"）
      - NodePropPredDataset 从 S3 下载（ogbn-products），不再依赖 ogb 官方服务器
      - torchrun --nnodes 1 --nproc_per_node 1 启动（不再直接 python 运行）
    """
    parser = argparse.ArgumentParser(
        description="pylibcugraph multi-GPU degree computation example"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets",
        help="Root directory for dataset storage (default: datasets)",
    )
    args = parser.parse_args()

    # 构造配置并打印（WALPURGIS_DEBUG 时）
    cfg = PLC_Config(dataset_root=args.dataset_root)
    cfg.dump()

    # 检测可用 GPU 数
    world_size = torch.cuda.device_count()
    _dbg("main", f"world_size={world_size}")
    if world_size == 0:
        print("[WPG:main] ERROR: no CUDA devices found, exiting.", file=sys.stderr)
        sys.exit(1)

    # 生成唯一通信 ID
    uid = cugraph_comms_create_unique_id()
    _dbg("main", "unique comms id created")

    # 加载 ogbn-products 数据集（28d1b30：从 RAPIDS S3 下载，非 ogb 官方）
    _dbg("main", f"loading {cfg.dataset_name} from {cfg.dataset_root} ...")
    dataset = NodePropPredDataset(cfg.dataset_name, root=cfg.dataset_root)
    el = dataset[0][0]["edge_index"].astype("int64")
    _dbg("main", f"edge_index shape={el.shape}")

    # spawn worker 进程
    _dbg("main", f"spawning {world_size} workers ...")
    tmp.spawn(
        calc_degree,
        args=(world_size, uid, el, cfg),
        nprocs=world_size,
    )
    _dbg("main", "all workers done")


if __name__ == "__main__":
    main()
