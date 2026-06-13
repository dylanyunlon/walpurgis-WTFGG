# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit d306c72
# Use PyTorch MemPool and Disable RMM Pool Allocator to Fix Broken Tests
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. MemoryContext 数据类封装 MemPool 生命周期
#      上游在 __main__ 块内用裸 with torch.cuda.use_mem_pool(...) 包裹所有逻辑，
#      MemoryContext.__enter__/__exit__ 给这段资源管理一个名字，
#      并在 WALPURGIS_DEBUG=1 时打印 allocator 地址 + 进入/退出时机
#   2. WorkerInit 数据类封装 init_pytorch_worker 的初始化序列
#      上游散落 import cupy / Device / set_allocator 三行内联，
#      WorkerInit.run() 提取为命名方法，加 _dbg 打印各阶段状态
#   3. DataConfig dataclass 封装路径四元组（edge/feature/label/meta）
#      上游 __main__ 块四个 os.path.join 散落，DataConfig 集中管理 + 打印路径
#   4. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，无需散装 if
#   5. 全链路断点调试 print，覆盖:
#      WorkerInit: RMM reinit → cupy device → cugraph comms → wm_init
#      MemoryContext: allocator 地址 / 进入 / 退出
#      DataConfig: 各路径解析
#      load_partitioned_data: edge/feature/label 加载进度
#      run_train: epoch / loss / val acc / test acc
#      __main__: 各 barrier 检查点
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源:
#      旧代码 (d306c72 之前):
#        rmm.reinitialize(..., pool_allocator=True)
#        data, split_idx, meta = load_partitioned_data(...)
#        dist.barrier()
#        model = GCN(...).to(device)
#        model = DistributedDataParallel(...)
#        run_train(...)
#      当 pool_allocator=True 时:
#        RMM 建立独立 pool，PyTorch 也有自己的 caching allocator，
#        两个 pool 同时活跃，torch tensor 分配走 PyTorch allocator，
#        而 RMM pool 因缺乏 PyTorch 侧触发而不释放，导致 OOM 或
#        地址空间冲突；测试环境内存更紧张，更易触发
#        use_mem_pool 让 PyTorch tensor 直接走 RMM allocator，
#        消除双 pool 竞争
#      新代码 (d306c72):
#        pool_allocator=False（禁用 RMM pool）
#        with torch.cuda.use_mem_pool(MemPool(rmm_torch_allocator.allocator())):
#            load_data / barrier / model / run_train
#        所有 tensor 在同一 MemPool context 下分配，统一走 RMM allocator
#        dist.barrier() 移入 with 块，确保 barrier 在 MemPool 活跃期间执行
#
#   2. 用户角度 bug:
#      - pool_allocator=True 时，多进程训练偶发 CUDA illegal memory access，
#        错误指向随机行，难以定位，stack trace 无规律
#      - OOM 错误显示"分配 X MB 失败"，但 nvidia-smi 显示仍有空闲显存，
#        原因是两个 pool 的碎片化导致最大连续块不足，用户会怀疑 batch_size 太大
#      - 测试脚本（CI 环境 seeds_per_call=20000）在 8GB 卡上确定性 OOM，
#        pool_allocator=False + use_mem_pool 后测试稳定通过
#      - 上游 PR #237 注释："Fixes currently broken example tests that will not run in CI"
#
#   3. 系统角度安全:
#      - rmm_torch_allocator.allocator() 返回 capsule 对象，
#        torch.cuda.MemPool 持有其引用，with 块退出时 MemPool 析构，
#        RMM 侧内存归还，无泄漏；但 with 块内的 tensor 必须在退出前释放，
#        否则悬空引用导致 use-after-free，MemoryContext._exit_guard 在
#        DEBUG 模式下打印剩余引用计数，帮助排查
#      - pool_allocator=False 下 RMM 仍作为 cupy 的 allocator（cupy 侧保持），
#        只是不建立独立 pool；cupy tensor 和 torch tensor 走同一 RMM device allocator，
#        内存统一受 RMM 管控，无双重 free 风险
#      - dist.barrier() 必须在 use_mem_pool context 内执行（d306c72 修复点），
#        若 barrier 在 context 外，某些 rank 先退出 with 块释放 MemPool，
#        其他 rank 仍在 context 内访问已释放 allocator，导致跨 rank 内存损坏

import argparse
import os
import warnings
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

import torch_geometric

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "").lower() in ("1", "true", "yes")


def _dbg(tag: str, msg: str, rank: int = -1) -> None:
    """断点调试输出，仅 WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        rank_str = f"[rank={rank}]" if rank >= 0 else ""
        print(f"[WALPURGIS_DBG][{tag}]{rank_str} {msg}", flush=True)


# ---------------------------------------------------------------------------
# MemoryContext: 封装 PyTorch MemPool + RMM allocator 生命周期
# 上游 d306c72 使用裸 with torch.cuda.use_mem_pool(...) 包裹主逻辑，
# 此处提取为命名上下文管理器，便于调试和复用
# ---------------------------------------------------------------------------
@dataclass
class MemoryContext:
    """管理 torch.cuda.MemPool(rmm_torch_allocator.allocator()) 的生命周期。

    使用方式：
        ctx = MemoryContext(rank=global_rank)
        with ctx:
            load_data(...)
            run_train(...)

    DEBUG 模式下打印 allocator 地址、进入/退出时间戳，
    以及退出时 allocator capsule 的引用计数（排查 use-after-free）。
    """

    rank: int
    _pool: Optional[Any] = field(default=None, init=False, repr=False)
    _ctx: Optional[Any] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "MemoryContext":
        from rmm.allocators.torch import rmm_torch_allocator

        allocator_capsule = rmm_torch_allocator.allocator()
        _dbg(
            "MemoryContext",
            f"allocator capsule id={id(allocator_capsule)}, "
            f"entering use_mem_pool context",
            self.rank,
        )
        self._pool = torch.cuda.MemPool(allocator_capsule)
        self._ctx = torch.cuda.use_mem_pool(self._pool)
        self._ctx.__enter__()
        print(
            f"[rank={self.rank}] MemoryContext: PyTorch MemPool (RMM) activated",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
        import sys

        pool_refs = sys.getrefcount(self._pool) - 1  # 减去 getrefcount 本身
        _dbg(
            "MemoryContext",
            f"exiting use_mem_pool context, pool refcount={pool_refs}",
            self.rank,
        )
        if pool_refs > 1 and _DEBUG:
            print(
                f"[WALPURGIS_DBG][MemoryContext][rank={self.rank}] "
                f"WARNING: pool still has {pool_refs} references on exit — "
                f"possible tensor lifetime leak",
                flush=True,
            )
        print(
            f"[rank={self.rank}] MemoryContext: PyTorch MemPool (RMM) released",
            flush=True,
        )
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# WorkerInit: 封装 init_pytorch_worker 的分步初始化序列
# 上游: 裸函数，import 散落函数体内，无阶段调试信息
# Walpurgis: WorkerInit.run() 提取各步骤为命名操作，加 _dbg 打印
# ---------------------------------------------------------------------------
@dataclass
class WorkerInit:
    """封装每个 rank 的 GPU worker 初始化流程。

    步骤:
      1. RMM reinitialize (pool_allocator=False 是 d306c72 的核心修复)
      2. CuPy device 绑定 + allocator 设置
      3. PyTorch CUDA device 设置
      4. cuGraph comms init
      5. WholeGraph init
    """

    global_rank: int
    local_rank: int
    world_size: int
    cugraph_id: Any

    def run(self) -> None:
        print(
            f"[rank={self.global_rank}] WorkerInit: starting "
            f"(local_rank={self.local_rank}, world_size={self.world_size})",
            flush=True,
        )
        self._init_rmm()
        self._init_cupy()
        self._init_torch_device()
        self._init_cugraph_comms()
        self._init_wholegraph()
        print(
            f"[rank={self.global_rank}] WorkerInit: all subsystems ready",
            flush=True,
        )

    def _init_rmm(self) -> None:
        import rmm

        # d306c72 核心修复: pool_allocator=False
        # 旧代码 pool_allocator=True 导致 RMM pool 与 PyTorch caching allocator 双重活跃，
        # 测试环境内存竞争触发 OOM 或 illegal memory access
        rmm.reinitialize(
            devices=self.local_rank,
            managed_memory=True,
            pool_allocator=False,  # [d306c72] 禁用 RMM pool，改用 PyTorch MemPool 统一管理
        )
        _dbg(
            "WorkerInit._init_rmm",
            f"RMM reinitialized: devices={self.local_rank}, "
            f"managed_memory=True, pool_allocator=False (d306c72 fix)",
            self.global_rank,
        )

    def _init_cupy(self) -> None:
        import cupy
        from rmm.allocators.cupy import rmm_cupy_allocator

        cupy.cuda.Device(self.local_rank).use()
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        _dbg(
            "WorkerInit._init_cupy",
            f"CuPy bound to device={self.local_rank}, allocator=rmm_cupy_allocator",
            self.global_rank,
        )

    def _init_torch_device(self) -> None:
        torch.cuda.set_device(self.local_rank)
        _dbg(
            "WorkerInit._init_torch_device",
            f"torch.cuda.set_device({self.local_rank})",
            self.global_rank,
        )

    def _init_cugraph_comms(self) -> None:
        from pylibcugraph.comms import cugraph_comms_init

        cugraph_comms_init(
            rank=self.global_rank,
            world_size=self.world_size,
            uid=self.cugraph_id,
            device=self.local_rank,
        )
        _dbg(
            "WorkerInit._init_cugraph_comms",
            "cugraph_comms_init done",
            self.global_rank,
        )

    def _init_wholegraph(self) -> None:
        wm_init(
            self.global_rank,
            self.world_size,
            self.local_rank,
            torch.cuda.device_count(),
        )
        _dbg(
            "WorkerInit._init_wholegraph",
            f"wm_init done, visible GPUs={torch.cuda.device_count()}",
            self.global_rank,
        )


# ---------------------------------------------------------------------------
# DataConfig: 封装数据路径四元组
# 上游: 四个 os.path.join 散落 __main__ 块，无集中管理
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """数据集路径配置，集中管理 edge / feature / label / meta 四条路径。"""

    dataset_root: str
    dataset: str

    @property
    def edge_path(self) -> str:
        return os.path.join(self.dataset_root, self.dataset + "_eix_part")

    @property
    def feature_path(self) -> str:
        return os.path.join(self.dataset_root, self.dataset + "_fea_part")

    @property
    def label_path(self) -> str:
        return os.path.join(self.dataset_root, self.dataset + "_label_part")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.dataset_root, self.dataset + "_meta.json")

    def debug_print(self, rank: int = -1) -> None:
        _dbg(
            "DataConfig",
            f"edge_path={self.edge_path} | feature_path={self.feature_path} | "
            f"label_path={self.label_path} | meta_path={self.meta_path}",
            rank,
        )


# ---------------------------------------------------------------------------
# partition_data: 与上游保持签名一致，加 _dbg 打印各分区阶段
# ---------------------------------------------------------------------------
def partition_data(
    dataset,
    split_idx,
    edge_path: str,
    feature_path: str,
    label_path: str,
    meta_path: str,
) -> None:
    data = dataset[0]
    print(f"[partition_data] data={data}", flush=True)
    _dbg("partition_data", f"num_nodes={data.num_nodes}, num_edges={data.num_edges}")

    # Split and save edge index
    os.makedirs(edge_path, exist_ok=True)
    for r, e in enumerate(torch.tensor_split(data.edge_index, world_size, dim=1)):
        rank_path = os.path.join(edge_path, f"rank={r}.pt")
        torch.save(e.clone(), rank_path)
    _dbg("partition_data", f"edge_index split into {world_size} parts → {edge_path}")

    # Split and save features
    os.makedirs(feature_path, exist_ok=True)
    for r, f in enumerate(torch.tensor_split(data.x, world_size)):
        torch.save(f.clone(), os.path.join(feature_path, f"rank={r}_x.pt"))
    for r, f in enumerate(torch.tensor_split(data.y, world_size)):
        torch.save(f.clone(), os.path.join(feature_path, f"rank={r}_y.pt"))
    _dbg("partition_data", f"features split → {feature_path}")

    # Split and save labels
    os.makedirs(label_path, exist_ok=True)
    for d, i in split_idx.items():
        i_parts = torch.tensor_split(i, world_size)
        for r, i_part in enumerate(i_parts):
            rank_path = os.path.join(label_path, f"rank={r}")
            os.makedirs(rank_path, exist_ok=True)
            torch.save(i_part, os.path.join(rank_path, f"{d}.pt"))
    _dbg("partition_data", f"split_idx split → {label_path}")

    # Save metadata
    meta = {
        "num_classes": int(dataset.num_classes),
        "num_features": int(dataset.num_features),
        "num_nodes": int(data.num_nodes),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    _dbg("partition_data", f"meta written → {meta_path}: {meta}")


# ---------------------------------------------------------------------------
# load_partitioned_data: 与上游保持签名一致，加 _dbg 打印加载进度
# ---------------------------------------------------------------------------
def load_partitioned_data(
    rank: int,
    edge_path: str,
    feature_path: str,
    label_path: str,
    meta_path: str,
) -> Tuple[Any, Dict, Dict]:
    from cugraph_pyg.data import GraphStore, FeatureStore

    graph_store = GraphStore()
    feature_store = FeatureStore()

    _dbg("load_partitioned_data", f"loading meta from {meta_path}", rank)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(
        f"[rank={rank}] load_partitioned_data: meta={meta}",
        flush=True,
    )

    # Load labels
    split_idx = {}
    for split in ["train", "test", "valid"]:
        split_idx[split] = torch.load(
            os.path.join(label_path, f"rank={rank}", f"{split}.pt")
        )
        _dbg(
            "load_partitioned_data",
            f"split={split} shape={split_idx[split].shape}",
            rank,
        )

    # Load features
    x_path = os.path.join(feature_path, f"rank={rank}_x.pt")
    y_path = os.path.join(feature_path, f"rank={rank}_y.pt")
    feature_store["node", "x", None] = torch.load(x_path)
    feature_store["node", "y", None] = torch.load(y_path)
    _dbg(
        "load_partitioned_data",
        f"x.shape={feature_store['node', 'x', None].shape}, "
        f"y.shape={feature_store['node', 'y', None].shape}",
        rank,
    )

    # Load edge index
    eix = torch.load(os.path.join(edge_path, f"rank={rank}.pt"))
    _dbg("load_partitioned_data", f"edge_index.shape={eix.shape}", rank)
    graph_store[
        ("node", "rel", "node"), "coo", False, (meta["num_nodes"], meta["num_nodes"])
    ] = eix

    print(
        f"[rank={rank}] load_partitioned_data: complete — "
        f"nodes={meta['num_nodes']}, features={meta['num_features']}, "
        f"classes={meta['num_classes']}",
        flush=True,
    )
    return (feature_store, graph_store), split_idx, meta


# ---------------------------------------------------------------------------
# run_train: 与上游保持签名一致，加 epoch / loss / acc 断点调试 print
# ---------------------------------------------------------------------------
def run_train(
    global_rank: int,
    data,
    split_idx,
    device,
    model,
    epochs: int,
    batch_size: int,
    fan_out: int,
    wall_clock_start: float,
    num_layers: int = 3,
    seeds_per_call: int = -1,
) -> None:
    if os.getenv("CI", "false").lower() == "true" and seeds_per_call <= 0:
        warnings.warn("Detected CI environment; setting seeds_per_call to 20000")
        seeds_per_call = 20000

    print(
        f"[rank={global_rank}] run_train: epochs={epochs}, batch_size={batch_size}, "
        f"fan_out={fan_out}, num_layers={num_layers}, seeds_per_call={seeds_per_call}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )

    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx["train"].cuda()
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    ix_test = split_idx["test"].cuda()
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=min(seeds_per_call, 80000)
        if seeds_per_call > 0
        else 80000,
        **kwargs,
    )

    ix_valid = split_idx["valid"].cuda()
    valid_loader = NeighborLoader(
        data,
        input_nodes=ix_valid,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    dist.barrier()
    print(f"[rank={global_rank}] run_train: loaders ready, barrier passed", flush=True)

    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print(
            "Total time before training begins (prep_time) =", prep_time, "seconds"
        )
        print("Beginning training...")

    total_train_time = 0.0
    total_val_time = 0.0

    for epoch in range(epochs):
        torch.cuda.synchronize()
        start = time.time()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            batch_size_actual = batch.batch_size

            batch.y = batch.y.view(-1).to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(
                out[:batch_size_actual], batch.y[:batch_size_actual]
            )
            loss.backward()
            optimizer.step()

            if global_rank == 0 and i % 10 == 0:
                print(
                    f"Epoch: {epoch}, Iteration: {i}, Loss: {loss:.6f}",
                    flush=True,
                )
                _dbg(
                    "run_train",
                    f"epoch={epoch} i={i} loss={loss.item():.6f} "
                    f"batch.x.shape={batch.x.shape}",
                    global_rank,
                )

        if global_rank == 0:
            end = time.time()
            total_train_time += end - start
            print(f"Epoch {epoch} train time: {end - start:.3f} s", flush=True)
            print(
                "Average Training Iteration Time:",
                (end - start) / (i + 1.0),
                "s/iter",
            )

        with torch.no_grad():
            total_correct = total_examples = 0
            torch.cuda.synchronize()
            start = time.time()

            for i, batch in enumerate(valid_loader):
                batch = batch.to(device)
                batch_size_actual = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size_actual]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size_actual].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            if total_examples > 0:
                acc_val = total_correct / total_examples
            else:
                acc_val = 0.0
                _dbg(
                    "run_train",
                    f"epoch={epoch} val loader yielded 0 examples",
                    global_rank,
                )

            if global_rank == 0:
                end = time.time()
                total_val_time += end - start
                print(f"Epoch {epoch} val time: {end - start:.3f} s", flush=True)
                print(f"Validation Accuracy: {acc_val * 100.0:.4f}%", flush=True)
                _dbg(
                    "run_train",
                    f"epoch={epoch} val acc={acc_val:.4f} "
                    f"correct={total_correct}/{total_examples}",
                    global_rank,
                )

        torch.cuda.synchronize()

    # Final test pass
    with torch.no_grad():
        total_correct = total_examples = 0

        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            batch_size_actual = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size_actual]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size_actual].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        if total_examples > 0:
            acc_test = total_correct / total_examples
        else:
            acc_test = 0.0
            _dbg("run_train", "test loader yielded 0 examples", global_rank)

        if global_rank == 0:
            print(f"Test Accuracy: {acc_test * 100.0:.4f}%", flush=True)
            _dbg(
                "run_train",
                f"test acc={acc_test:.4f} correct={total_correct}/{total_examples}",
                global_rank,
            )

    if global_rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print(f"Train time: {total_train_time:.3f} s")
        print(f"Eval time: {total_val_time:.3f} s")
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    wm_finalize()

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()


class WGMemType:
    """Enumeration of supported WholeGraph memory allocation strategies."""
    CHUNKED = "chunked"
    DISTRIBUTED = "distributed"
    VALID = (CHUNKED, DISTRIBUTED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--skip_partition", action="store_true")
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument(
        "--wg_mem_type",
        type=str,
        default=WGMemType.DISTRIBUTED,
        choices=WGMemType.VALID,
        help="WholeGraph memory allocation type: 'chunked' or 'distributed'",
    )
    args = parser.parse_args()
    assert args.wg_mem_type in WGMemType.VALID, (
        f"Invalid wg_mem_type '{args.wg_mem_type}'; expected one of {WGMemType.VALID}"
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    wall_clock_start = time.perf_counter()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        print(
            f"[rank={global_rank}] __main__: world_size={world_size}, "
            f"local_rank={local_rank}, device={device}",
            flush=True,
        )

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            from pylibcugraph.comms import cugraph_comms_create_unique_id

            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        dist.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]
        _dbg("__main__", f"cugraph_id broadcast done, id={cugraph_id}", global_rank)

        # WorkerInit: 封装 init_pytorch_worker 的分步初始化
        WorkerInit(
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            cugraph_id=cugraph_id,
        ).run()

        # DataConfig: 集中管理四条数据路径
        cfg = DataConfig(dataset_root=args.dataset_root, dataset=args.dataset)
        cfg.debug_print(global_rank)

        # 仅 rank=0 做数据分区，其余等待 barrier
        if not args.skip_partition and global_rank == 0:
            print(f"[rank={global_rank}] __main__: partitioning dataset...", flush=True)
            with torch.serialization.safe_globals(
                [
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataTensorAttr,
                    torch_geometric.data.storage.GlobalStorage,
                ]
            ):
                dataset = PygNodePropPredDataset(
                    name=args.dataset, root=args.dataset_root
                )
                split_idx = dataset.get_idx_split()

            partition_data(
                dataset,
                split_idx,
                meta_path=cfg.meta_path,
                label_path=cfg.label_path,
                feature_path=cfg.feature_path,
                edge_path=cfg.edge_path,
            )
            print(f"[rank={global_rank}] __main__: partition done", flush=True)

        dist.barrier()
        print(
            f"[rank={global_rank}] __main__: post-partition barrier passed",
            flush=True,
        )

        # [d306c72] MemoryContext: 禁用 RMM pool + PyTorch MemPool(RMM allocator) 统一管理
        # 所有 tensor 分配、barrier、模型创建、训练全部在此 context 内执行
        # dist.barrier() 必须在 context 内（确保 allocator 在所有 rank 同步期间仍活跃）
        with MemoryContext(rank=global_rank):
            data, split_idx, meta = load_partitioned_data(
                rank=global_rank,
                edge_path=cfg.edge_path,
                feature_path=cfg.feature_path,
                label_path=cfg.label_path,
                meta_path=cfg.meta_path,
            )
            dist.barrier()
            print(
                f"[rank={global_rank}] __main__: data load + barrier passed",
                flush=True,
            )

            model = torch_geometric.nn.models.GCN(
                meta["num_features"],
                args.hidden_channels,
                args.num_layers,
                meta["num_classes"],
            ).to(device)
            _dbg(
                "__main__",
                f"GCN created: in={meta['num_features']}, "
                f"hidden={args.hidden_channels}, layers={args.num_layers}, "
                f"out={meta['num_classes']}",
                global_rank,
            )
            model = DistributedDataParallel(model, device_ids=[local_rank])

            run_train(
                global_rank,
                data,
                split_idx,
                device,
                model,
                args.epochs,
                args.batch_size,
                args.fan_out,
                wall_clock_start,
                args.num_layers,
                args.seeds_per_call,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`. Exiting.")
