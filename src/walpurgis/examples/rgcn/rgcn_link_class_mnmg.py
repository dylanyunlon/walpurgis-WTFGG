# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit d306c72
# Use PyTorch MemPool and Disable RMM Pool Allocator to Fix Broken Tests
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. MemoryContext 数据类封装 MemPool 生命周期（同 gcn_dist_mnmg.py）
#      上游裸 with torch.cuda.use_mem_pool(...) 包裹 100+ 行主逻辑，
#      MemoryContext.__enter__/__exit__ 给资源管理命名，加调试打印
#   2. GraphBroadcaster 数据类封装图数据广播序列
#      上游 d306c72 中 broadcast edge_rel_type / edge_index / splits / neg_splits
#      四段散落 print + 赋值混合，GraphBroadcaster 将其提取为命名步骤：
#      _broadcast_edge_rel_type / _broadcast_edge_index /
#      _broadcast_splits / _broadcast_neg_splits
#      每步加 _dbg 打印 tensor shape + rank 信息
#   3. WorkerInit 数据类封装 init_pytorch_worker（同 gcn 侧，保持风格一致）
#   4. SplitsAccessor: 封装 splits_storage 的 lazy accessor
#      run_train 内频繁从 splits_storage[stage, key, None] 取数据，
#      SplitsAccessor.get() 集中 + 加 shape 断点打印
#   5. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印
#   6. 全链路断点调试 print，覆盖:
#      WorkerInit: RMM/cupy/comms/wm 各子步
#      MemoryContext: allocator 地址 / 进入 / 退出
#      GraphBroadcaster: 各 broadcast 步骤的 tensor shape
#      train: epoch / iteration / loss
#      test: epoch / stage / mrr
#      __main__: 各 barrier 检查点
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源:
#      旧代码 (d306c72 之前):
#        rmm.reinitialize(..., pool_allocator=True, ...)
#        torch.distributed.barrier()
#        if global_rank == 0: ... load dataset ... nr = [num_nodes, num_rels]
#        else: nr = [0, 0]
#        torch.distributed.broadcast_object_list(nr, ...)
#        ...100+ 行图构建 / split 广播 / 模型创建 / run_train...
#      问题:
#        pool_allocator=True 双 pool 竞争（同 gcn_dist_mnmg.py 分析）
#        所有图构建 tensor 在 MemPool context 外分配，
#        退出 context 后这些 tensor 若仍被持有，allocator 已销毁 → use-after-free
#      新代码 (d306c72):
#        pool_allocator=False
#        with torch.cuda.use_mem_pool(MemPool(rmm_torch_allocator.allocator())):
#            barrier / load_dataset / broadcast_object_list / graph_build / run_train
#        所有 tensor 生命周期与 MemPool context 对齐
#
#   2. 用户角度 bug:
#      - ogbl-wikikg2 数据集有 2.5M 节点 / 17M 边，pool_allocator=True 时
#        node_emb Parameter(2.5M × 32 × float32 = ~320MB) + 图构建 tensor 累积，
#        RMM pool 碎片化导致 CUDAMemoryError 指向无关行（embedding lookup / conv）
#      - 多 rank 时 rank=0 加载数据后 broadcast，pool 竞争在 broadcast 完成前触发，
#        非 rank=0 的进程报 illegal memory access，用户误以为通信配置问题
#      - 测试环境（CI 4-8GB 卡）确定性崩溃，本地 A100 可能偶发，难以稳定复现
#      - nr = [0, 0] 在非 rank=0 进程初始化，broadcast 后才有效值，
#        旧代码 barrier() 在 with 块外，极端情况下 barrier 完成但 allocator
#        已被另一进程释放（进程调度差异），新代码将 barrier 移入 with 块修复此竞争
#
#   3. 系统角度安全:
#      - RGCNEncoder 的 node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
#        在 with 块内分配，MemPool 管理其生命周期；
#        run_train 内 model.to(device) 也在 with 块内，
#        退出 with 块后 run_train 已完成，无悬空引用
#      - splits_storage 是 FeatureStore（cugraph_pyg 自有类型），
#        其内部 tensor 在 with 块内分配，run_train 调用也在 with 块内，
#        get_local_split() 的 lambda 延迟求值，但 lambda 在 run_train 内调用，
#        仍在 with 块内，安全
#      - global_rank=0 的 `del data` 在 with 块内（d306c72 保持），
#        确保 PygLinkPropPredDataset 对象在 MemPool 活跃期间释放，
#        其持有的 CUDA tensor 归还给 MemPool 而非挂起
#      - empty(dim=1) / empty(dim=2) 是 cugraph_pyg.tensor.empty，
#        返回零维 placeholder，在 MemPool context 内分配，安全

import os
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy

import torch
import torch_geometric

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import FastRGCNConv, GAE
from torch.nn.parallel import DistributedDataParallel

from ogb.linkproppred import PygLinkPropPredDataset

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
# 与 gcn_dist_mnmg.py 保持完全一致的接口，方便跨示例复用
# ---------------------------------------------------------------------------
@dataclass
class MemoryContext:
    """管理 torch.cuda.MemPool(rmm_torch_allocator.allocator()) 的生命周期。"""

    rank: int
    _pool: Optional[Any] = field(default=None, init=False, repr=False)
    _ctx: Optional[Any] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "MemoryContext":
        from rmm.allocators.torch import rmm_torch_allocator

        allocator_capsule = rmm_torch_allocator.allocator()
        _dbg(
            "MemoryContext",
            f"allocator capsule id={id(allocator_capsule)}, entering use_mem_pool",
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

        pool_refs = sys.getrefcount(self._pool) - 1
        _dbg(
            "MemoryContext",
            f"exiting use_mem_pool, pool refcount={pool_refs}",
            self.rank,
        )
        if pool_refs > 1 and _DEBUG:
            print(
                f"[WALPURGIS_DBG][MemoryContext][rank={self.rank}] "
                f"WARNING: pool still has {pool_refs} references — possible leak",
                flush=True,
            )
        print(
            f"[rank={self.rank}] MemoryContext: PyTorch MemPool (RMM) released",
            flush=True,
        )
        return False


# ---------------------------------------------------------------------------
# WorkerInit: 封装 init_pytorch_worker 的分步初始化序列
# ---------------------------------------------------------------------------
@dataclass
class WorkerInit:
    """每个 rank 的 GPU worker 初始化（RMM / CuPy / PyTorch / cuGraph / WholeGraph）。"""

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

        # [d306c72] pool_allocator=False: 禁用 RMM pool，消除双 pool 竞争
        rmm.reinitialize(
            devices=[self.local_rank],
            pool_allocator=False,  # [d306c72] fix: was True
            managed_memory=True,
        )
        _dbg(
            "WorkerInit._init_rmm",
            f"RMM reinit: devices=[{self.local_rank}], pool_allocator=False",
            self.global_rank,
        )

    def _init_cupy(self) -> None:
        import cupy
        from rmm.allocators.cupy import rmm_cupy_allocator

        cupy.cuda.Device(self.local_rank).use()
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        _dbg(
            "WorkerInit._init_cupy",
            f"CuPy device={self.local_rank}, allocator=rmm_cupy_allocator",
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
            self.global_rank,
            self.world_size,
            self.cugraph_id,
            self.local_rank,
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
# GraphBroadcaster: 封装图数据广播序列
# 上游 d306c72 中 4 段 print + broadcast 混合散落，GraphBroadcaster 提取为命名步骤
# ---------------------------------------------------------------------------
@dataclass
class GraphBroadcaster:
    """封装 rank=0 → all ranks 的图数据广播序列。

    步骤:
      1. broadcast edge reltype
      2. broadcast edge index
      3. broadcast train/test/valid splits (head/tail)
      4. broadcast negative splits (head_neg/tail_neg/relation) for test/valid

    每步加 _dbg 打印 tensor shape，方便排查 broadcast 前后数据对齐问题。
    """

    global_rank: int
    device: torch.device
    edge_feature_store: Any
    graph_store: Any
    splits_storage: Any
    num_nodes: int

    # rank=0 提供的原始数据（非 rank=0 时为 None）
    dataset: Optional[Any] = None
    splits: Optional[Any] = None

    def broadcast_all(self) -> None:
        """按顺序广播所有图数据。"""
        self._broadcast_edge_rel_type()
        self._broadcast_edge_index()
        self._broadcast_splits()
        self._broadcast_neg_splits()
        print(
            f"[rank={self.global_rank}] GraphBroadcaster: all broadcasts complete",
            flush=True,
        )

    def _broadcast_edge_rel_type(self) -> None:
        from cugraph_pyg.tensor import empty

        print(
            f"broadcasting edge rel type (rank {self.global_rank})", flush=True
        )
        self.edge_feature_store[("n", "e", "n"), "rel", None] = (
            self.dataset.edge_reltype.to(torch.int32)
            if self.global_rank == 0
            else empty(dim=2)
        )
        shape = self.edge_feature_store[("n", "e", "n"), "rel", None].shape
        _dbg(
            "GraphBroadcaster._broadcast_edge_rel_type",
            f"edge_reltype.shape={shape}",
            self.global_rank,
        )

    def _broadcast_edge_index(self) -> None:
        from cugraph_pyg.tensor import empty

        print(
            f"broadcasting edge index (rank {self.global_rank})", flush=True
        )
        self.graph_store[
            ("n", "e", "n"), "coo", False, (self.num_nodes, self.num_nodes)
        ] = (
            self.dataset.edge_index if self.global_rank == 0 else empty(dim=2)
        )
        _dbg(
            "GraphBroadcaster._broadcast_edge_index",
            f"num_nodes={self.num_nodes}",
            self.global_rank,
        )

    def _broadcast_splits(self) -> None:
        from cugraph_pyg.tensor import empty

        print("broadcasting splits", flush=True)
        for stage in ["train", "test", "valid"]:
            self.splits_storage[stage, "head", None] = (
                self.splits[stage]["head"].to(torch.int64)
                if self.global_rank == 0
                else empty(dim=1)
            )
            self.splits_storage[stage, "tail", None] = (
                self.splits[stage]["tail"].to(torch.int64)
                if self.global_rank == 0
                else empty(dim=1)
            )
            if self.global_rank == 0:
                _dbg(
                    "GraphBroadcaster._broadcast_splits",
                    f"stage={stage} "
                    f"head.shape={self.splits[stage]['head'].shape} "
                    f"tail.shape={self.splits[stage]['tail'].shape}",
                    self.global_rank,
                )

    def _broadcast_neg_splits(self) -> None:
        from cugraph_pyg.tensor import empty

        print("broadcasting negative splits", flush=True)
        for stage in ["test", "valid"]:
            self.splits_storage[stage, "head_neg", None] = (
                self.splits[stage]["head_neg"].to(torch.int64)
                if self.global_rank == 0
                else empty(dim=2)
            )
            self.splits_storage[stage, "tail_neg", None] = (
                self.splits[stage]["tail_neg"].to(torch.int64)
                if self.global_rank == 0
                else empty(dim=2)
            )
            self.splits_storage[stage, "relation", None] = (
                self.splits[stage]["relation"].to(torch.int32)
                if self.global_rank == 0
                else empty(dim=1)
            )
            if self.global_rank == 0:
                _dbg(
                    "GraphBroadcaster._broadcast_neg_splits",
                    f"stage={stage} "
                    f"head_neg.shape={self.splits[stage]['head_neg'].shape} "
                    f"tail_neg.shape={self.splits[stage]['tail_neg'].shape}",
                    self.global_rank,
                )


# ---------------------------------------------------------------------------
# RGCNEncoder: 与上游保持签名一致
# ---------------------------------------------------------------------------
class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations, num_bases=30):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.conv2 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


def get_local_split(t):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    u = t()
    return u[torch.tensor_split(torch.arange(u.shape[0]), world_size, dim=0)[rank]]


# ---------------------------------------------------------------------------
# train / test: 与上游保持签名一致，加断点调试 print
# ---------------------------------------------------------------------------
def train(epoch, model, optimizer, train_loader, edge_feature_store, num_steps=None):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        r = (
            edge_feature_store[("n", "e", "n"), "rel", None][batch.e_id]
            .flatten()
            .cuda()
        ).to(torch.int64)
        z = model.encode(batch.edge_index, r)

        loss = model.recon_loss(z, batch.edge_index)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch: {epoch:02d}, Iteration: {i:02d}, Loss: {loss:.4f}",
                flush=True,
            )
            _dbg(
                "train",
                f"epoch={epoch} i={i} loss={loss.item():.6f} "
                f"batch.edge_index.shape={batch.edge_index.shape}",
            )
        if num_steps and i == num_steps:
            break


def test(stage, epoch, model, loader, num_steps=None):
    model.eval()

    rr = 0.0
    for i, (h, h_neg, t, t_neg, r) in enumerate(loader):
        if num_steps and i >= num_steps:
            break

        ei = torch.concatenate(
            [
                torch.stack([h, t]).cuda(),
                torch.stack([h_neg.flatten(), t_neg.flatten()]).cuda(),
            ],
            dim=-1,
        )

        r = (
            torch.concatenate([r, torch.repeat_interleave(r, h_neg.shape[-1])])
            .cuda()
            .to(torch.int64)
        )

        z = model.encode(ei, r)
        q = model.decode(z, ei)

        _, ix = torch.sort(q, descending=True)
        rr += 1.0 / (1.0 + ix[0])

    mrr = rr / i if i > 0 else 0.0
    print(f"epoch {epoch:02d} {stage} mrr: {mrr:.6f}", flush=True)
    _dbg("test", f"epoch={epoch} stage={stage} mrr={mrr:.6f} steps={i}")


# ---------------------------------------------------------------------------
# run_train: 与上游保持签名一致，加 loader 创建 / eval 断点 print
# ---------------------------------------------------------------------------
def run_train(global_rank, local_rank, model, data, edge_feature_store, splits, args):
    model = model.to(torch.device(local_rank))
    model = GAE(DistributedDataParallel(model, device_ids=[local_rank]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eli = torch.stack(
        [
            get_local_split(splits["train", "head", None]).cpu(),
            get_local_split(splits["train", "tail", None]).cpu(),
        ]
    )
    _dbg("run_train", f"eli.shape={eli.shape}", global_rank)

    from cugraph_pyg.loader import LinkNeighborLoader

    print("creating train loader...", flush=True)
    train_loader = LinkNeighborLoader(
        data,
        [args.fan_out] * args.num_layers,
        edge_label_index=eli,
        local_seeds_per_call=args.seeds_per_call if args.seeds_per_call > 0 else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(f"[rank={global_rank}] run_train: train_loader ready", flush=True)

    def get_eval_loader(stage: str):
        head = get_local_split(splits[stage, "head", None]).cpu()
        tail = get_local_split(splits[stage, "tail", None]).cpu()
        head_neg = (
            get_local_split(splits[stage, "head_neg", None])[:, : args.num_neg].cpu()
        )
        tail_neg = (
            get_local_split(splits[stage, "tail_neg", None])[:, : args.num_neg].cpu()
        )
        rel = get_local_split(splits[stage, "relation", None]).cpu()

        print(
            head.shape,
            head_neg.shape,
            tail.shape,
            tail_neg.shape,
            rel.shape,
            flush=True,
        )
        _dbg(
            "run_train.get_eval_loader",
            f"stage={stage} head={head.shape} tail={tail.shape} "
            f"head_neg={head_neg.shape}",
            global_rank,
        )

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(head, head_neg, tail, tail_neg, rel),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )

    test_loader = get_eval_loader("test")
    valid_loader = get_eval_loader("valid")

    num_train_steps = (args.num_pos // args.batch_size) if args.num_pos > 0 else 100

    for epoch in range(1, 1 + args.epochs):
        train(
            epoch,
            model,
            optimizer,
            train_loader,
            edge_feature_store,
            num_steps=num_train_steps,
        )
        test("validation", epoch, model, valid_loader, num_steps=1024)

    test("test", epoch, model, test_loader, num_steps=1024)

    wm_finalize()

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_neg", type=int, default=500)
    parser.add_argument("--num_pos", type=int, default=-1)
    parser.add_argument("--fan_out", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ogbl-wikikg2")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument("--n_devices", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        print(
            f"[rank={global_rank}] __main__: world_size={world_size}, "
            f"local_rank={local_rank}",
            flush=True,
        )

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            from pylibcugraph.comms import cugraph_comms_create_unique_id

            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        # WorkerInit: 封装 init_pytorch_worker
        WorkerInit(
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            cugraph_id=cugraph_id,
        ).run()

        torch.distributed.barrier()
        print(
            f"[rank={global_rank}] __main__: post-worker-init barrier passed",
            flush=True,
        )

        # [d306c72] MemoryContext: pool_allocator=False + PyTorch MemPool(RMM) 统一管理
        # barrier 在 context 内（确保 allocator 在 barrier 期间仍活跃）
        with MemoryContext(rank=global_rank):

            # rank=0 加载数据集并广播节点/关系数
            dataset_obj = None
            splits_raw = None
            if global_rank == 0:
                with torch.serialization.safe_globals(
                    [
                        torch_geometric.data.data.DataEdgeAttr,
                        torch_geometric.data.data.DataTensorAttr,
                        torch_geometric.data.storage.GlobalStorage,
                        numpy.core.multiarray._reconstruct,
                        numpy.ndarray,
                        numpy.dtype,
                        numpy.dtypes.Int64DType,
                    ]
                ):
                    dataset_obj = PygLinkPropPredDataset(
                        args.dataset, root=args.dataset_root
                    )
                    dataset_inner = dataset_obj[0]
                    print(dataset_inner, flush=True)
                    splits_raw = dataset_obj.get_edge_split()

                nr = [dataset_inner.num_nodes, int(dataset_inner.edge_reltype.max()) + 1]
                _dbg(
                    "__main__",
                    f"dataset loaded: num_nodes={nr[0]}, num_rels={nr[1]}",
                    global_rank,
                )
            else:
                nr = [0, 0]
                dataset_inner = None

            torch.distributed.barrier()
            torch.distributed.broadcast_object_list(nr, src=0, device=device)
            num_nodes, num_rels = nr

            print(
                f"num_nodes: {num_nodes}, num_rels: {num_rels}, rank: {global_rank}",
                flush=True,
            )
            torch.distributed.barrier()
            print(
                f"[rank={global_rank}] __main__: nr broadcast barrier passed",
                flush=True,
            )

            from cugraph_pyg.data import FeatureStore, GraphStore

            edge_feature_store = FeatureStore()
            splits_storage = FeatureStore()
            feature_store = torch_geometric.data.HeteroData()
            graph_store = GraphStore()
            torch.distributed.barrier()
            print(
                f"[rank={global_rank}] __main__: stores initialized, barrier passed",
                flush=True,
            )

            # GraphBroadcaster: 封装四段广播序列
            broadcaster = GraphBroadcaster(
                global_rank=global_rank,
                device=device,
                edge_feature_store=edge_feature_store,
                graph_store=graph_store,
                splits_storage=splits_storage,
                num_nodes=num_nodes,
                dataset=dataset_inner if global_rank == 0 else None,
                splits=splits_raw if global_rank == 0 else None,
            )
            broadcaster.broadcast_all()

            print("reached barrier", flush=True)
            torch.distributed.barrier()
            print(
                f"[rank={global_rank}] __main__: post-broadcast barrier passed",
                flush=True,
            )

            model = RGCNEncoder(
                num_nodes,
                hidden_channels=args.hidden_channels,
                num_relations=num_rels,
            )
            _dbg(
                "__main__",
                f"RGCNEncoder created: num_nodes={num_nodes}, "
                f"hidden={args.hidden_channels}, num_rels={num_rels}",
                global_rank,
            )

            # rank=0 释放原始数据集，节省显存（与上游 d306c72 保持一致）
            if global_rank == 0:
                del dataset_obj
                _dbg("__main__", "dataset_obj deleted (rank=0)", global_rank)

            run_train(
                global_rank,
                local_rank,
                model,
                (feature_store, graph_store),
                edge_feature_store,
                splits_storage,
                args,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`. Exiting.")
