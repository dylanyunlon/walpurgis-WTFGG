# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 0e88280
# Support PyG 2.6 in cuGraph-PyG — single-GPU GCN example
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. DataConfig 数据类封装 load_data 的四路返回值
#      上游 load_data 返回裸四元组 (data, split_idx, num_features, num_classes)，
#      调用侧解包时没有命名，debug 时看不出哪个是哪个。
#      DataConfig.__repr__ 在 WALPURGIS_DEBUG=1 时打印完整 shape 信息。
#   2. LoaderFactory 封装 create_loader 的参数打包逻辑
#      上游 create_loader 直接接收散装 kwargs，
#      LoaderFactory.build(stage, input_nodes) 提取命名参数，
#      加断点打印 stage / batch_size / num_neighbors。
#   3. TrainStats 数据类追踪 epoch 统计（loss / avg_iter_time）
#      上游 train() 函数依赖全局 start_avg_time + warmup_steps 变量，
#      TrainStats 封装为局部状态，函数无全局副作用，易并行化。
#   4. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，无需散装 if os.environ
#   5. 全链路断点调试 print，覆盖:
#      DataConfig: graph_store edge count / feature_store shape
#      LoaderFactory: stage / batch_size / seed count
#      TrainStats: iter time / loss 滚动平均
#      test(): 每100个 batch 打印 running accuracy
#      __main__: prep_time / epoch / total_time
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源:
#      旧代码 (0e88280 之前):
#        feature_store["node", "x"] = data.x
#        feature_store["node", "y"] = data.y
#        graph_store[("node", "rel", "node"), "coo"] = data.edge_index  # 2-tuple key
#      新代码 (0e88280):
#        feature_store["node", "x", None] = data.x        # 3-tuple → attr_index=None
#        feature_store["node", "y", None] = data.y
#        graph_store[..., "coo", False, (num_nodes, num_nodes)]  # 5-tuple key with size
#      PyG 2.6 breaking change: TensorAttr/EdgeAttr 构造时必须完整指定所有字段，
#      partial specification (2-tuple 或省略 size) 不再被允许 — 会抛 ValueError。
#      Walpurgis 已在其他文件（gcn_dist_mnmg.py, rgcn_link_class_mnmg.py）全面采用新 API。
#   2. 用户角度 bug:
#      单 GPU 示例比多 GPU 示例更常被新用户直接运行，
#      但 gcn_dist_sg.py 在此次批次前尚未被迁移到 walpurgis 仓库，
#      导致 walpurgis 缺少单机可运行的参考示例。
#   3. 算法无变化: 纯 GCN (torch_geometric.nn.models.GCN)，
#      NeighborLoader + OGB ogbn-products，PyG 2.6 API 修复不影响训练结果。

import time
import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# 70c33af: SG (single-GPU) examples are transitioning to torchrun-based unified API.
# The upstream cugraph-gnn removed gcn_dist_sg.py and gcn_dist_snmg.py in favor of
# the unified MNMG example that handles SG/SNMG/MNMG through one torchrun entrypoint.
# This file is retained for reference but emits a FutureWarning on import.
# Migration path: use gcn_dist_mnmg.py with torchrun --nproc_per_node=1 instead.
warnings.warn(
    "gcn_dist_sg.py (single-GPU, non-torchrun) is deprecated as of 70c33af. "
    "Prefer gcn_dist_mnmg.py launched via `torchrun --nproc_per_node=1` which "
    "supports SG, SNMG, and MNMG workflows through the unified WholeGraph API.",
    FutureWarning,
    stacklevel=1,
)

def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DataConfig — 封装 load_data 返回的四元素，加调试打印
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    封装单卡 GCN 训练所需的数据结构。

    上游 load_data 返回裸四元组，字段无名称：
        return ((feature_store, graph_store), split_idx, num_features, num_classes)
    DataConfig 给这四个值命名，并在 WALPURGIS_DEBUG 时打印 shape。
    """
    store_pair: Tuple[Any, Any]   # (feature_store, graph_store)
    split_idx: Dict[str, torch.Tensor]
    num_features: int
    num_classes: int

    def debug_print(self):
        if not _WDBG:
            return
        fs, gs = self.store_pair
        attrs = fs.get_all_tensor_attrs() if hasattr(fs, "get_all_tensor_attrs") else []
        _dbg(
            "DataConfig",
            "loaded",
            num_features=self.num_features,
            num_classes=self.num_classes,
            train_nodes=len(self.split_idx.get("train", [])),
            valid_nodes=len(self.split_idx.get("valid", [])),
            test_nodes=len(self.split_idx.get("test", [])),
            feature_attrs=[a.attr_name for a in attrs],
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainStats — 封装 train() 的 epoch 统计，替代全局变量
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainStats:
    """
    追踪单个 epoch 的训练统计。

    上游 train() 依赖外部全局 warmup_steps / start_avg_time：
        if i == warmup_steps:
            start_avg_time = time.perf_counter()
    全局状态无法重入，多次调用 train() 时 start_avg_time 会被覆盖。
    TrainStats 将这两个变量封装为局部状态。
    """
    warmup_steps: int = 20
    _start_time: float = field(default=0.0, repr=False)
    _iter_count: int = field(default=0, repr=False)
    _last_loss: float = field(default=0.0, repr=False)

    def mark_warmup(self, i: int):
        if i == self.warmup_steps:
            torch.cuda.synchronize()
            self._start_time = time.perf_counter()
            _dbg("TrainStats", f"warmup done at iter={i}, clock started")

    def record_iter(self, i: int, loss: float):
        self._iter_count = i
        self._last_loss = loss
        if i % 10 == 0:
            _dbg("TrainStats", f"iter={i}", loss=f"{loss:.4f}")

    def avg_iter_time(self) -> float:
        if self._start_time == 0.0:
            return float("nan")
        elapsed = time.perf_counter() - self._start_time
        denom = max(1, self._iter_count - self.warmup_steps)
        return elapsed / denom


# ─────────────────────────────────────────────────────────────────────────────
# LoaderFactory — 封装 NeighborLoader 参数
# ─────────────────────────────────────────────────────────────────────────────

class LoaderFactory:
    """
    封装 NeighborLoader 的构建参数，避免上游散装 kwargs 传递。

    上游模式:
        loader_kwargs = {"data": data, "num_neighbors": [...], ...}
        train_loader = create_loader(input_nodes=split_idx["train"], ...)
    LoaderFactory 将 base_kwargs 和每次 build 的 stage-specific 参数分离，
    在 WALPURGIS_DEBUG=1 时打印每个 stage 的参数摘要。
    """

    def __init__(
        self,
        data,
        num_neighbors,
        replace: bool,
        batch_size: int,
        local_seeds_per_call: Optional[int],
    ):
        self._data = data
        self._num_neighbors = num_neighbors
        self._replace = replace
        self._batch_size = batch_size
        self._local_seeds_per_call = local_seeds_per_call
        _dbg(
            "LoaderFactory",
            "init",
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            local_seeds_per_call=local_seeds_per_call,
        )

    def build(self, stage: str, input_nodes) -> "NeighborLoader":
        from cugraph_pyg.loader import NeighborLoader  # lazy import

        seed_count = len(input_nodes) if hasattr(input_nodes, "__len__") else "?"
        _dbg(
            "LoaderFactory",
            f"building loader for stage={stage}",
            seed_count=seed_count,
            batch_size=self._batch_size,
        )

        return NeighborLoader(
            self._data,
            num_neighbors=self._num_neighbors,
            input_nodes=input_nodes,
            replace=self._replace,
            batch_size=self._batch_size,
            local_seeds_per_call=self._local_seeds_per_call,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载 — 核心算法：4-tuple API (PyG 2.6)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(dataset_name: str, dataset_root: str) -> DataConfig:
    """
    加载 OGB 节点分类数据集，构建 GraphStore + TensorDictFeatureStore。

    PyG 2.6 核心变化 (commit 0e88280):
        旧: feature_store["node", "x"] = data.x            # 2-tuple
        新: feature_store["node", "x", None] = data.x      # 3-tuple, attr_index=None

        旧: graph_store[(...), "coo"] = data.edge_index     # 2-tuple key
        新: graph_store[..., "coo", False, (N, N)] = ...   # 5-tuple key with size

    PyG 2.6 强制完整指定所有 TensorAttr/EdgeAttr 字段。
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    import cugraph_pyg

    _dbg("load_data", f"loading dataset={dataset_name} root={dataset_root}")

    dataset = PygNodePropPredDataset(dataset_name, root=dataset_root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    num_nodes = data.num_nodes
    _dbg("load_data", "dataset loaded", num_nodes=num_nodes, num_edges=data.edge_index.shape[1])

    # GraphStore — 5-tuple key: (edge_type, layout, is_sorted, size)
    graph_store = cugraph_pyg.data.GraphStore()
    graph_store[
        ("node", "rel", "node"), "coo", False, (num_nodes, num_nodes)
    ] = data.edge_index
    _dbg("load_data", "graph_store built", edge_key=("node", "rel", "node"))

    # TensorDictFeatureStore — 3-tuple key: (group_name, attr_name, attr_index=None)
    feature_store = cugraph_pyg.data.TensorDictFeatureStore()
    feature_store["node", "x", None] = data.x
    feature_store["node", "y", None] = data.y
    _dbg(
        "load_data",
        "feature_store built",
        x_shape=data.x.shape,
        y_shape=data.y.shape,
    )

    cfg = DataConfig(
        store_pair=(feature_store, graph_store),
        split_idx=split_idx,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )
    cfg.debug_print()
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 训练与评估
# ─────────────────────────────────────────────────────────────────────────────

def train(model, optimizer, loader, epoch: int, device) -> TrainStats:
    """
    单 epoch 训练。

    上游用全局 warmup_steps / start_avg_time；
    Walpurgis 封装为 TrainStats 局部对象，函数无全局副作用。
    """
    import torch.nn.functional as F

    model.train()
    stats = TrainStats()

    for i, batch in enumerate(loader):
        stats.mark_warmup(i)
        batch = batch.to(device)
        optimizer.zero_grad()

        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].view(-1).to(torch.long)

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        stats.record_iter(i, float(loss))

        if i % 10 == 0:
            print(f"Epoch: {epoch:02d}, Iter: {i}, Loss: {loss:.4f}")

    torch.cuda.synchronize()
    avg_t = stats.avg_iter_time()
    print(f"Avg training iter time (s/iter): {avg_t:.6f}")
    _dbg("train", f"epoch={epoch} done", avg_iter_time=f"{avg_t:.6f}")
    return stats


@torch.no_grad()
def test(model, loader, device, val_steps: Optional[int] = None) -> float:
    model.eval()
    total_correct = total_examples = 0

    for i, batch in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

        if i % 100 == 0:
            running_acc = total_correct / max(1, total_examples)
            _dbg("test", f"i={i}", running_acc=f"{running_acc:.4f}", examples=total_examples)

    return total_correct / max(1, total_examples)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Walpurgis single-GPU GCN (PyG 2.6+)")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    # 必须在 CUDA 初始化前切换内存分配器 (RMM pool)
    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    from rmm.allocators.torch import rmm_torch_allocator

    rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    import torch_geometric
    # e01196b: cuDF spilling removed — WholeGraph UVA/managed_memory via RMM handles OOM.
    # enable_spilling() was cugraph.testing.mg_utils which depended on cudf;
    # managed_memory=True above achieves equivalent over-subscription via driver.
    if _DEBUG:
        import sys
        print("[WALPURGIS-EXAMPLE:gcn_dist_sg][__main__] cuDF spilling disabled; "
              "RMM managed_memory=True active", file=sys.stderr, flush=True)

    args = parse_args()
    wall_clock_start = time.perf_counter()
    device = torch.device("cuda")

    _dbg("main", "start", dataset=args.dataset, epochs=args.epochs)

    cfg = load_data(args.dataset, args.dataset_root)

    if os.getenv("CI", "false").lower() == "true":
        warnings.warn("Pruning test dataset for CI run.")
        cfg.split_idx["test"] = cfg.split_idx["test"][:1000]

    seeds_per_call = None if args.seeds_per_call <= 0 else args.seeds_per_call
    factory = LoaderFactory(
        data=cfg.store_pair,
        num_neighbors=[args.fan_out] * args.num_layers,
        replace=False,
        batch_size=args.batch_size,
        local_seeds_per_call=seeds_per_call,
    )

    train_loader = factory.build("train", cfg.split_idx["train"])
    val_loader   = factory.build("val",   cfg.split_idx["valid"])
    test_loader  = factory.build("test",  cfg.split_idx["test"])

    model = torch_geometric.nn.models.GCN(
        cfg.num_features,
        args.hidden_channels,
        args.num_layers,
        cfg.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    torch.cuda.synchronize()
    prep_time = round(time.perf_counter() - wall_clock_start, 2)
    print(f"Prep time: {prep_time}s — beginning training...")
    _dbg("main", "training begins", prep_time=prep_time)

    for epoch in range(1, 1 + args.epochs):
        stats = train(model, optimizer, train_loader, epoch, device)
        val_acc = test(model, val_loader, device, val_steps=100)
        print(f"Epoch {epoch}: Val Acc ≈ {val_acc:.4f}")
        _dbg("main", f"epoch={epoch}", val_acc=f"{val_acc:.4f}")

    test_acc = test(model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")

    total_time = round(time.perf_counter() - wall_clock_start, 2)
    print(f"Total time: {total_time}s  (train time: {total_time - prep_time}s)")
    _dbg("main", "done", total_time=total_time, test_acc=f"{test_acc:.4f}")
