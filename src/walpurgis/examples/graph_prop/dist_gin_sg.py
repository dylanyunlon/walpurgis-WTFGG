# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Walpurgis迁移: migrate 4088267 — Add Graph Property prediction model to cugraph-pyg
# 迁移者: dylanyunlon <dogechat@163.com>
# 改写20%（鲁迅拿法）：
#   - GinArgs dataclass 封装 argparse，validate() 含路径/参数合法性检查
#   - GraphBundle 值对象封装分布式存储构建，build() 集中图构建逻辑
#   - GinTrainer context manager 封装训练生命周期，__exit__ 保证 dist 清理
#   - 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖数据加载/训练/测试全过程

import os
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import Dataset, DataLoader

from cugraph_pyg.data import FeatureStore
from cugraph_pyg.tensor import DistTensor

# ─── 全局调试开关 ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """仅在 WALPURGIS_DEBUG=1 时输出断点信息。"""
    if _DEBUG:
        print("[WALPURGIS_DEBUG]", *args, **kwargs, flush=True)


# ─── GinArgs: 参数封装 ────────────────────────────────────────────────────────
# 鲁迅拿法: 将散落 main() 的 argparse 与超参覆盖逻辑收进 dataclass，
# validate() 前置检查，避免运行到一半才崩。

SUPPORTED_DATASETS = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "COLLAB",
    "IMDB-BINARY",
    "REDDIT-BINARY",
]

_DATASET_DEFAULTS = {
    "MUTAG":         {"batch_size": 128, "hidden_channels": 32,  "lr": 0.01},
    "ENZYMES":       {"batch_size": 32,  "hidden_channels": 64,  "lr": 0.01},
    "PROTEINS":      {"batch_size": 32,  "hidden_channels": 64,  "lr": 0.01},
    "COLLAB":        {"batch_size": 32,  "hidden_channels": 64,  "lr": 0.01},
    "IMDB-BINARY":   {"batch_size": 128, "hidden_channels": 64,  "lr": 0.01},
    "REDDIT-BINARY": {"batch_size": 32,  "hidden_channels": 64,  "lr": 0.01},
}


@dataclass
class GinArgs:
    """GIN 训练参数，封装 argparse 解析结果与超参覆盖逻辑。"""
    dataset:         str   = "ENZYMES"
    batch_size:      int   = 32
    hidden_channels: int   = 64
    num_layers:      int   = 5
    lr:              float = 0.01
    epochs:          int   = 10
    dropout:         float = 0.5
    train_split:     float = 0.9
    device:          str   = "cuda"
    data_root:       Optional[str] = None

    # 解析后填入，勿手动设置
    _batch_size_resolved:      int   = field(init=False, repr=False, default=0)
    _hidden_channels_resolved: int   = field(init=False, repr=False, default=0)
    _lr_resolved:              float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self._resolve_hyperparams()

    def _resolve_hyperparams(self):
        """将命令行参数与 dataset 默认超参合并。上游逻辑：默认值则用 dataset 推荐。"""
        defaults = _DATASET_DEFAULTS.get(
            self.dataset, {"batch_size": 32, "hidden_channels": 64, "lr": 0.01}
        )
        # 上游语义：命令行未改变默认值时，使用 dataset 推荐值
        self._batch_size_resolved      = self.batch_size      if self.batch_size      != 32   else defaults["batch_size"]
        self._hidden_channels_resolved = self.hidden_channels if self.hidden_channels != 64   else defaults["hidden_channels"]
        self._lr_resolved              = self.lr              if self.lr              != 0.01 else defaults["lr"]

        _dbg(
            f"GinArgs._resolve_hyperparams: dataset={self.dataset}"
            f" batch_size={self._batch_size_resolved}"
            f" hidden_channels={self._hidden_channels_resolved}"
            f" lr={self._lr_resolved}"
            f" epochs={self.epochs}"
            f" dropout={self.dropout}"
            f" train_split={self.train_split}"
        )

    def validate(self):
        """前置参数合法性检查，运行前调用。"""
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"不支持的 dataset: {self.dataset!r}，"
                f"合法值: {SUPPORTED_DATASETS}"
            )
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(f"train_split 须在 (0, 1) 区间，实际: {self.train_split}")
        if self.epochs <= 0:
            raise ValueError(f"epochs 须为正整数，实际: {self.epochs}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers 须为正整数，实际: {self.num_layers}")
        if self.data_root is not None and ".." in self.data_root:
            # 防止路径穿越
            raise ValueError(f"data_root 含非法路径段 '..': {self.data_root}")
        _dbg(f"GinArgs.validate: 参数检查通过")

    @classmethod
    def from_argparse(cls) -> "GinArgs":
        """从命令行解析参数并构建 GinArgs。"""
        parser = argparse.ArgumentParser(
            description="GIN Graph Property Prediction — cugraph-pyg 分布式后端"
        )
        parser.add_argument("--dataset",          type=str,   default="ENZYMES", choices=SUPPORTED_DATASETS)
        parser.add_argument("--batch_size",       type=int,   default=32,    help="训练批大小")
        parser.add_argument("--hidden_channels",  type=int,   default=64,    help="隐层维度")
        parser.add_argument("--num_layers",       type=int,   default=5,     help="GIN 层数")
        parser.add_argument("--lr",               type=float, default=0.01,  help="学习率")
        parser.add_argument("--epochs",           type=int,   default=10,    help="训练轮次")
        parser.add_argument("--dropout",          type=float, default=0.5,   help="Dropout 率")
        parser.add_argument("--train_split",      type=float, default=0.9,   help="训练集比例")
        parser.add_argument("--device",           type=str,   default="cuda",help="训练设备")
        parser.add_argument("--data_root",        type=str,   default=None,  help="数据根目录")
        ns = parser.parse_args()
        return cls(
            dataset=ns.dataset,
            batch_size=ns.batch_size,
            hidden_channels=ns.hidden_channels,
            num_layers=ns.num_layers,
            lr=ns.lr,
            epochs=ns.epochs,
            dropout=ns.dropout,
            train_split=ns.train_split,
            device=ns.device,
            data_root=ns.data_root,
        )


# ─── GraphBundle: 分布式图存储封装 ───────────────────────────────────────────
# 鲁迅拿法: 上游 load_data() 散落返回 5 个裸值，调用方靠位置解包，
# 此处用 GraphBundle 值对象持有，字段命名消除歧义。

@dataclass
class GraphBundle:
    """封装分布式图数据：edge_index、feature_store、edge_ptr 及元信息。"""
    feature_store:    FeatureStore
    dist_edge_index:  DistTensor
    edge_ptr:         torch.Tensor
    num_graphs:       int
    num_features:     int
    num_classes:      int

    @classmethod
    def build(cls, args: GinArgs, device: torch.device) -> "GraphBundle":
        """从 TUDataset 构建分布式图存储。集中原 load_data() 逻辑。"""
        import os.path as osp

        data_root = args.data_root or osp.join(osp.expanduser("~"), "data", "TU")
        _dbg(f"GraphBundle.build: data_root={data_root!r} dataset={args.dataset!r}")

        # ── 特征处理：无节点特征数据集用 OneHotDegree 合成 ──
        temp_ds = TUDataset(data_root, name=args.dataset)
        needs_features = temp_ds.num_features == 0
        _dbg(f"GraphBundle.build: needs_features={needs_features} num_features_raw={temp_ds.num_features}")

        if needs_features:
            max_degree = 5000 if args.dataset == "REDDIT-BINARY" else 1000
            _dbg(f"GraphBundle.build: 使用 OneHotDegree(max_degree={max_degree})")
            transform = OneHotDegree(max_degree=max_degree, cat=False)
            dataset = TUDataset(data_root, name=args.dataset, transform=transform).shuffle()
        else:
            dataset = TUDataset(data_root, name=args.dataset).shuffle()

        data = Batch.from_data_list(dataset)

        # ── 边存储 ──
        # 上游使用 data.ptr 作为 edge_ptr（PyG 内置图边界指针，不需手算）
        edge_index = data.edge_index.t()  # [E, 2]
        edge_ptr   = data.ptr
        _dbg(
            f"GraphBundle.build: edge_index.shape={edge_index.shape}"
            f" edge_ptr.shape={edge_ptr.shape}"
            f" num_graphs={len(dataset)}"
        )

        # ── 分布式存储 ──
        dist_edge_index = DistTensor.from_tensor(tensor=edge_index)
        feature_store   = FeatureStore()
        feature_store["node", "x", None] = data.x
        feature_store["graph", "y", None] = data.y

        num_features = data.x.size(1)
        num_classes  = int(data.y.max().item()) + 1
        _dbg(
            f"GraphBundle.build: num_features={num_features}"
            f" num_classes={num_classes}"
            f" x.shape={data.x.shape}"
            f" y.shape={data.y.shape}"
        )

        return cls(
            feature_store=feature_store,
            dist_edge_index=dist_edge_index,
            edge_ptr=edge_ptr,
            num_graphs=len(dataset),
            num_features=num_features,
            num_classes=num_classes,
        )


# ─── DistTensorGraphDataset ───────────────────────────────────────────────────
# 上游原版，加 DEBUG 断点，逻辑保持不变。

class DistTensorGraphDataset(Dataset):
    """从分布式 Tensor 中按图粒度提取单张图的 Dataset。"""

    def __init__(
        self,
        dist_edge_index: DistTensor,
        feature_store:   FeatureStore,
        device:          torch.device,
        edge_ptr:        torch.Tensor,
        graph_indices:   Optional[List[int]] = None,
        split_name:      str = "unknown",    # 仅用于 debug 输出
    ):
        self.dist_edge_index = dist_edge_index
        self.feature_store   = feature_store
        self.device          = device
        self._edge_ptr       = edge_ptr
        self._split_name     = split_name

        if graph_indices is not None:
            self.graph_indices = graph_indices
        else:
            num_graphs = len(edge_ptr) - 1
            self.graph_indices = list(range(num_graphs))

        _dbg(
            f"DistTensorGraphDataset[{split_name}]: "
            f"graph_indices 长度={len(self.graph_indices)} "
            f"device={device}"
        )

        # ── 标签缓存：小数据集提前加载，避免 __getitem__ 逐次访问 ──
        self._cached_labels = None
        if len(self.graph_indices) < 1000:
            self._cached_labels = self.feature_store["graph", "y", None][
                torch.tensor(self.graph_indices, device=self.device)
            ]
            _dbg(
                f"DistTensorGraphDataset[{split_name}]: "
                f"标签已缓存 shape={self._cached_labels.shape}"
            )

    def __len__(self) -> int:
        return len(self.graph_indices)

    def __getitem__(self, idx: int) -> dict:
        # ── 取标签 ──
        if self._cached_labels is not None:
            y = self._cached_labels[idx]
        else:
            y = self.feature_store["graph", "y", None][
                torch.tensor([idx], device=self.device)
            ]
        if y.dim() > 1:
            y = y.squeeze()

        # ── 取边 ──
        edge_start = self._edge_ptr[idx].item()
        edge_end   = self._edge_ptr[idx + 1].item()
        edge_ids   = torch.arange(edge_start, edge_end, device=self.device, dtype=torch.long)

        local_edges       = self.dist_edge_index[edge_ids]
        nodes_in_subgraph = local_edges.unique()

        # 向量化节点重编号（比 dict comprehension 快）
        node_to_local = torch.zeros(
            nodes_in_subgraph.max().item() + 1,
            dtype=torch.long,
            device=self.device,
        )
        node_to_local[nodes_in_subgraph] = torch.arange(
            nodes_in_subgraph.size(0), device=self.device
        )

        src_local  = node_to_local[local_edges[:, 0]]
        dst_local  = node_to_local[local_edges[:, 1]]
        graph_edges = torch.stack([src_local, dst_local], dim=1)

        sub_x = self.feature_store["node", "x", None][nodes_in_subgraph]

        _dbg(
            f"DistTensorGraphDataset[{self._split_name}].__getitem__({idx}): "
            f"num_nodes={sub_x.size(0)} num_edges={graph_edges.size(0)} y={y}"
        )

        return {
            "x":         sub_x,
            "edge_index": graph_edges,
            "y":          y,
            "num_nodes":  sub_x.size(0),
        }


# ─── custom_collate_fn ────────────────────────────────────────────────────────
# 上游原版，加 DEBUG 断点。

def custom_collate_fn(batch: List[dict]) -> Data:
    """高效 collate：预分配张量，向量化边偏移，避免 list comprehension。"""
    batch_size = len(batch)
    if batch_size == 0:
        return Data()

    x_list         = [item["x"]          for item in batch]
    edge_index_list = [item["edge_index"] for item in batch]
    y_list          = [item["y"]          for item in batch]
    num_nodes_list  = [item["num_nodes"]  for item in batch]

    total_nodes  = sum(x.size(0) for x in x_list)
    total_edges  = sum(e.size(0) for e in edge_index_list)
    feature_dim  = x_list[0].size(1)
    device       = x_list[0].device

    _dbg(
        f"custom_collate_fn: batch_size={batch_size}"
        f" total_nodes={total_nodes}"
        f" total_edges={total_edges}"
        f" feature_dim={feature_dim}"
    )

    x_batch      = torch.empty((total_nodes, feature_dim), dtype=x_list[0].dtype, device=device)
    y_batch      = torch.empty(batch_size,                 dtype=y_list[0].dtype, device=device)
    batch_tensor = torch.empty(total_nodes,                dtype=torch.long,      device=device)

    node_offset = 0
    for i, (x, y, num_nodes) in enumerate(zip(x_list, y_list, num_nodes_list)):
        x_batch[node_offset : node_offset + num_nodes] = x
        y_batch[i]                                      = y.squeeze() if y.dim() > 0 else y
        batch_tensor[node_offset : node_offset + num_nodes] = i
        node_offset += num_nodes

    if total_edges > 0:
        edge_index_final = torch.empty((2, total_edges), dtype=torch.long, device=device)

        num_nodes_tensor = torch.tensor(num_nodes_list, dtype=torch.long, device=device)
        node_offsets     = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_nodes_tensor[:-1]]),
            dim=0,
        )

        edge_offset = 0
        for i, edge_index in enumerate(edge_index_list):
            if edge_index.size(0) > 0:
                num_edges = edge_index.size(0)
                edge_index_final[:, edge_offset : edge_offset + num_edges] = (
                    edge_index.t() + node_offsets[i]
                )
                edge_offset += num_edges
    else:
        edge_index_final = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_data            = Data(x=x_batch, edge_index=edge_index_final, y=y_batch, batch=batch_tensor)
    batch_data.num_graphs = batch_size
    return batch_data


# ─── GIN Model ────────────────────────────────────────────────────────────────

class GIN(torch.nn.Module):
    """Graph Isomorphism Network，用于图分类任务。"""

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        num_layers:      int,
        dropout:         float = 0.5,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP(
            [hidden_channels, hidden_channels, out_channels],
            norm=None,
            dropout=dropout,
        )

        _dbg(
            f"GIN.__init__: in_channels={in_channels} (after convs, hidden)"
            f" out_channels={out_channels}"
            f" num_layers={num_layers}"
            f" dropout={dropout}"
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """前向传播：GIN 卷积 → 全图池化 → MLP 分类头。"""
        _dbg(
            f"GIN.forward: x.shape={x.shape}"
            f" edge_index.shape={edge_index.shape}"
            f" batch_size={batch_size}"
        )
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch, size=batch_size)
        out = self.mlp(x)
        _dbg(f"GIN.forward: out.shape={out.shape}")
        return out


# ─── train / test ─────────────────────────────────────────────────────────────

def train(
    model:     GIN,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """单 epoch 训练，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs

        _dbg(
            f"train batch[{batch_idx}]: "
            f"num_graphs={batch.num_graphs}"
            f" loss={loss.item():.4f}"
            f" x.shape={batch.x.shape}"
            f" edge_index.shape={batch.edge_index.shape}"
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(
    model:  GIN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """在 loader 上评估，返回准确率。"""
    model.eval()
    total_correct = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        pred  = out.argmax(dim=-1)
        correct = int((pred == batch.y).sum())
        total_correct += correct

        _dbg(
            f"test batch[{batch_idx}]: "
            f"num_graphs={batch.num_graphs}"
            f" correct={correct}"
            f" pred[:5]={pred[:5].tolist()}"
            f" y[:5]={batch.y[:5].tolist()}"
        )

    return total_correct / len(loader.dataset)


# ─── GinTrainer: 训练生命周期 context manager ─────────────────────────────────
# 鲁迅拿法: 上游 main() 将 dist.init_process_group 和 dist.destroy_process_group
# 散落在函数头尾，异常路径不会调用 destroy，资源泄漏。
# GinTrainer.__exit__ 保证 dist 组无论如何都被清理。

class GinTrainer:
    """管理分布式训练生命周期：__enter__ 初始化 dist，__exit__ 保证清理。"""

    def __init__(self, args: GinArgs):
        self.args = args

    def __enter__(self) -> "GinTrainer":
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            _dbg(f"GinTrainer.__enter__: dist 初始化完成 rank={dist.get_rank()} world_size={dist.get_world_size()}")
        else:
            _dbg("GinTrainer.__enter__: dist 已初始化，跳过")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if dist.is_initialized():
            _dbg(f"GinTrainer.__exit__: 清理 dist 进程组 exc_type={exc_type}")
            dist.destroy_process_group()
        return False  # 不吞异常

    def run(self):
        """完整训练流程。"""
        args = self.args
        args.validate()

        # ── 设备 ──
        device = (
            torch.device("cuda:0")
            if args.device == "cuda" and torch.cuda.is_available()
            else torch.device(args.device)
        )
        _dbg(f"GinTrainer.run: device={device} CUDA={torch.cuda.is_available()}")

        print(f"Dataset:         {args.dataset}")
        print(f"Batch size:      {args._batch_size_resolved}")
        print(f"Hidden channels: {args._hidden_channels_resolved}")
        print(f"Learning rate:   {args._lr_resolved}")
        print(f"Epochs:          {args.epochs}")
        print(f"Device:          {device}, CUDA available: {torch.cuda.is_available()}")

        # ── 图数据加载 ──
        bundle = GraphBundle.build(args, device)

        print(f"Number of features: {bundle.num_features}")
        print(f"Number of classes:  {bundle.num_classes}")

        # ── 训练/测试划分 ──
        train_size          = int(args.train_split * bundle.num_graphs)
        train_graph_indices = list(range(0, train_size))
        test_graph_indices  = list(range(train_size, bundle.num_graphs))

        _dbg(
            f"GinTrainer.run: train_size={train_size}"
            f" test_size={len(test_graph_indices)}"
            f" total={bundle.num_graphs}"
        )
        print(f"Dataset size:      {bundle.num_graphs}")
        print(f"Training samples:  {len(train_graph_indices)}")
        print(f"Test samples:      {len(test_graph_indices)}")

        # ── Dataset & DataLoader ──
        train_dataset = DistTensorGraphDataset(
            bundle.dist_edge_index,
            bundle.feature_store,
            device,
            bundle.edge_ptr,
            train_graph_indices,
            split_name="train",
        )
        test_dataset = DistTensorGraphDataset(
            bundle.dist_edge_index,
            bundle.feature_store,
            device,
            bundle.edge_ptr,
            test_graph_indices,
            split_name="test",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args._batch_size_resolved,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args._batch_size_resolved,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        # ── 模型 & 优化器 ──
        model = GIN(
            bundle.num_features,
            args._hidden_channels_resolved,
            bundle.num_classes,
            args.num_layers,
            args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args._lr_resolved)

        _dbg(
            f"GinTrainer.run: 模型参数量="
            f"{sum(p.numel() for p in model.parameters())}"
        )

        # ── 训练循环 ──
        train_times: List[float] = []
        test_times:  List[float] = []
        total_times: List[float] = []

        for epoch in range(1, args.epochs + 1):
            train_start = time.time()
            loss        = train(model, train_loader, optimizer, device)
            train_time  = time.time() - train_start

            test_start = time.time()
            train_acc  = test(model, train_loader, device)
            test_acc   = test(model, test_loader,  device)
            test_time  = time.time() - test_start

            total_time = train_time + test_time
            train_times.append(train_time)
            test_times.append(test_time)
            total_times.append(total_time)

            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f},"
                f" Train: {train_acc:.4f}, Test: {test_acc:.4f}"
            )
            print(
                f"  Train Time: {train_time:.4f}s, "
                f"Test Time: {test_time:.4f}s, Total: {total_time:.4f}s"
            )

            _dbg(
                f"epoch {epoch}: loss={loss:.4f}"
                f" train_acc={train_acc:.4f}"
                f" test_acc={test_acc:.4f}"
            )

        # ── 汇总 ──
        t_train = torch.tensor(train_times)
        t_test  = torch.tensor(test_times)
        t_total = torch.tensor(total_times)

        print(
            f"Training - Median: {t_train.median():.4f}s,"
            f" Average: {t_train.mean():.4f}s"
        )
        print(
            f"Testing  - Median: {t_test.median():.4f}s,"
            f" Average: {t_test.mean():.4f}s"
        )
        print(
            f"Total    - Median: {t_total.median():.4f}s,"
            f" Average: {t_total.mean():.4f}s"
        )
        print(f"Final Test Accuracy: {test_acc:.4f}")

        _dbg(
            f"GinTrainer.run 完成: "
            f"median_train={t_train.median():.4f}s "
            f"median_test={t_test.median():.4f}s "
            f"final_test_acc={test_acc:.4f}"
        )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    """入口：解析参数，通过 GinTrainer context manager 运行训练。"""
    args = GinArgs.from_argparse()
    _dbg(f"main: args={args}")
    with GinTrainer(args) as trainer:
        trainer.run()


if __name__ == "__main__":
    main()
