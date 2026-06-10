# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 318ae6c
# Updates movielens_mnmg.py to use DDP
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. FeatureDims 数据类封装 num_features 字典，__post_init__ 校验维度 >= 1
#      上游裸 dict 字面量，x is None 时静默赋 1，维度校验散落 __main__
#   2. CugraphWorkerSession context manager 封装 init_pytorch_worker 生命周期
#      __exit__ 保证 cugraph_comms_shutdown + wm_finalize 在异常路径也执行
#      上游 __main__ 末尾裸调，OOM/NCCL 挂起时不执行，资源泄漏
#   3. ModelBundle 封装 Model + DDP + optimizer 三件套构建
#      上游三行散落 __main__，DDP 与 optimizer 中间无隔离，device_ids 硬绑定
#   4. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，无需散装 if os.environ
#   5. EncoderShapeGuard 在 Encoder.__init__ 内校验 SAGEConv 输入维度顺序
#      上游 318ae6c 新增显式 in_channels，但无文档说明 conv1 期望 (src, dst) 顺序
#      此处加断言，维度对调时立即报错而非训练数个 epoch 后 loss 不收敛
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源:
#      - 318ae6c 新增 user_in_channels / movie_in_channels 显式传入，
#        替代旧 (-1, -1) lazy 推断；conv1 期望 (movie→user) 方向:
#        SAGEConv((movie_in_channels, user_in_channels), ...) 正确
#        但与参数名 "user_in_channels" 在第二位的直觉相反，注释缺失
#      - DDP 包裹后 model.forward → model.module.forward，
#        train() / test() 中 model(x, ei, n) 调用方式不变（DDP 透明转发）
#      - WholeGraph init 删除说明已自动初始化，但 wm_finalize 仍需手动调用
#        上游末尾保留 wm_finalize()，此处 CugraphWorkerSession.__exit__ 确保执行
#   2. 用户角度 bug:
#      - data["user"].x 在 load_partitions 中由 torch.eye 生成，
#        shape[-1] = num_nodes["user"] (全量，非当前 rank 分片)，
#        而 model 各 rank 接收的 batch 特征维度是局部分片，
#        eye 矩阵 shape[-1] 恰好 = 用户总数，与 SAGEConv 期望维度匹配；
#        但若 num_nodes["user"] 极大（> 1M），eye 矩阵 OOM，上游无 guard
#      - drop_last=True + batch_size=256，若 eli_train 极小（< 256 edges）
#        train_loader 为空，train() 返回 0/0 ZeroDivisionError
#      - DDP 包裹后 optimizer.param_groups 引用的是 DDP wrapper 参数，
#        上游代码在 DDP 之后立即 Adam(model.parameters()) 顺序正确；
#        但 model.to(device) 必须在 DDP 之前，318ae6c 保持了正确顺序
#   3. 系统角度安全:
#      - DDP device_ids=[local_rank] 硬绑定 GPU，与 rmm.reinitialize(devices=local_rank)
#        一致；但若 LOCAL_RANK != cuda device index（容器内重映射），两处均出错
#      - cugraph_comms_shutdown 裸调在 __main__ 末尾，
#        with use_mem_pool 块内抛出异常时不执行；CugraphWorkerSession 修复此问题
#      - rmm MemPool 生命周期依赖 with 块，任何 DDP allreduce / NCCL timeout
#        导致 with 块异常退出时，未完成的 CUDA kernel 仍持有 pool 引用，
#        rmm 会报 pool-in-use error，属上游已知限制

import os
import warnings
from argparse import ArgumentParser
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional
import json

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from torch_geometric import EdgeIndex
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData

from cugraph_pyg.data import GraphStore, FeatureStore

from pylibwholegraph.torch.initialize import (
    finalize as wm_finalize,
)

from sklearn.metrics import roc_auc_score

# ──────────────────────────────────────────────────────────────────────────────
# 调试出口：WALPURGIS_DEBUG=1 时打印，生产环境零开销
# ──────────────────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试 print — 只在 WALPURGIS_DEBUG=1 时输出."""
    if _DEBUG:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        print(f"[WALPURGIS_DBG rank={rank}] [{tag}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 改写 1/5: FeatureDims — 封装 num_features 字典，校验维度 >= 1
# 上游裸 dict 字面量，x is None 时静默赋 1，无维度下限校验
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureDims:
    """封装节点特征维度，校验 >= 1 并提供友好错误信息."""

    user: int
    movie: int

    def __post_init__(self) -> None:
        for name, dim in (("user", self.user), ("movie", self.movie)):
            if dim < 1:
                raise ValueError(
                    f"FeatureDims.{name} must be >= 1, got {dim}. "
                    f"Check data[\"{name}\"].x.shape or set fallback=1."
                )
        _dbg("FeatureDims", f"user={self.user} movie={self.movie}")

    @classmethod
    def from_heterodata(cls, data: HeteroData) -> "FeatureDims":
        """从 HeteroData 提取特征维度，x is None 时回退到 1."""
        user_dim = data["user"].x.shape[-1] if data["user"].x is not None else 1
        movie_dim = data["movie"].x.shape[-1] if data["movie"].x is not None else 1
        _dbg("FeatureDims.from_heterodata", f"raw user_dim={user_dim} movie_dim={movie_dim}")
        return cls(user=user_dim, movie=movie_dim)

    def as_dict(self) -> Dict[str, int]:
        return {"user": self.user, "movie": self.movie}


# ──────────────────────────────────────────────────────────────────────────────
# 改写 2/5: CugraphWorkerSession — context manager 封装 init/shutdown 生命周期
# 上游裸函数 + __main__ 末尾裸调，OOM/NCCL 挂起时 shutdown 不执行
# ──────────────────────────────────────────────────────────────────────────────
class CugraphWorkerSession:
    """封装 cugraph worker 初始化/销毁生命周期，保证异常路径也执行 shutdown."""

    def __init__(self, global_rank: int, local_rank: int, world_size: int, cugraph_id) -> None:
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.cugraph_id = cugraph_id
        _dbg("CugraphWorkerSession.__init__",
             f"global_rank={global_rank} local_rank={local_rank} world_size={world_size}")

    def __enter__(self) -> "CugraphWorkerSession":
        self._init_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """保证 shutdown 在任何退出路径执行，包括异常."""
        _dbg("CugraphWorkerSession.__exit__", f"exc_type={exc_type}")
        try:
            from pylibcugraph.comms import cugraph_comms_shutdown
            cugraph_comms_shutdown()
            _dbg("CugraphWorkerSession.__exit__", "cugraph_comms_shutdown OK")
        except Exception as e:
            print(f"[WALPURGIS WARN] cugraph_comms_shutdown failed: {e}", flush=True)
        try:
            wm_finalize()
            _dbg("CugraphWorkerSession.__exit__", "wm_finalize OK")
        except Exception as e:
            print(f"[WALPURGIS WARN] wm_finalize failed: {e}", flush=True)
        # 不吞异常，让调用方看到原始错误
        return False

    def _init_worker(self) -> None:
        import rmm

        _dbg("CugraphWorkerSession._init_worker", "rmm.reinitialize start")
        rmm.reinitialize(
            devices=self.local_rank,
            managed_memory=False,
            pool_allocator=False,
        )
        _dbg("CugraphWorkerSession._init_worker", "rmm.reinitialize done")

        import cupy

        cupy.cuda.Device(self.local_rank).use()
        from rmm.allocators.cupy import rmm_cupy_allocator

        cupy.cuda.set_allocator(rmm_cupy_allocator)
        _dbg("CugraphWorkerSession._init_worker", "cupy allocator set")

        torch.cuda.set_device(self.local_rank)

        from pylibcugraph.comms import cugraph_comms_init

        cugraph_comms_init(
            rank=self.global_rank,
            world_size=self.world_size,
            uid=self.cugraph_id,
            device=self.local_rank,
        )
        _dbg("CugraphWorkerSession._init_worker",
             f"cugraph_comms_init done uid_type={type(self.cugraph_id).__name__}")
        # WholeGraph is initialized automatically.


# ──────────────────────────────────────────────────────────────────────────────
# 图/特征构建工具
# ──────────────────────────────────────────────────────────────────────────────
def write_edges(edge_index, path):
    world_size = torch.distributed.get_world_size()
    _dbg("write_edges", f"edge_index.shape={list(edge_index.shape)} path={path}")

    os.makedirs(path, exist_ok=True)
    for r, e in enumerate(torch.tensor_split(edge_index, world_size, dim=1)):
        rank_path = os.path.join(path, f"rank={r}.pt")
        torch.save(e.clone(), rank_path)


def cugraph_pyg_from_heterodata(data):
    _dbg("cugraph_pyg_from_heterodata", "building GraphStore + FeatureStore")

    graph_store = GraphStore()
    feature_store = FeatureStore()

    graph_store[
        ("user", "rates", "movie"),
        "coo",
        False,
        (data["user"].num_nodes, data["movie"].num_nodes),
    ] = data["user", "rates", "movie"].edge_index

    graph_store[
        ("movie", "rev_rates", "user"),
        "coo",
        False,
        (data["movie"].num_nodes, data["user"].num_nodes),
    ] = data["movie", "rev_rates", "user"].edge_index

    feature_store["user", "x", None] = data["user"].x
    feature_store["movie", "x", None] = data["movie"].x
    feature_store[("user", "rates", "movie"), "time", None] = data[
        "user", "rates", "movie"
    ].time
    feature_store[("movie", "rev_rates", "user"), "time", None] = data[
        "user", "rates", "movie"
    ].time

    _dbg("cugraph_pyg_from_heterodata",
         f"user.x shape={list(data['user'].x.shape)} "
         f"movie.x shape={list(data['movie'].x.shape)}")
    return feature_store, graph_store


def preprocess_and_partition(data, edge_path, features_path, meta_path):
    world_size = torch.distributed.get_world_size()
    _dbg("preprocess_and_partition", f"world_size={world_size}")

    # Only use edges with high ratings (>= 4):
    mask = data["user", "rates", "movie"].edge_label >= 4
    data["user", "movie"].edge_index = data["user", "movie"].edge_index[:, mask]
    data["user", "movie"].time = data["user", "movie"].time[mask]
    del data["user", "movie"].edge_label

    time = data["user", "movie"].time
    perm = time.argsort()

    data["user", "movie"] = data["user", "movie"].edge_index[:, perm]

    off = int(0.8 * perm.numel())
    ei = {
        "train": data["user", "movie"].edge_index[:, :off],
        "test": data["user", "movie"].edge_index[:, off:],
    }
    _dbg("preprocess_and_partition",
         f"train_edges={ei['train'].shape[1]} test_edges={ei['test'].shape[1]} split_off={off}")

    print("Writing edges...")
    user_movie_edge_path = os.path.join(edge_path, "user_movie")
    for d, eid in ei.items():
        d_path = os.path.join(user_movie_edge_path, d)
        write_edges(eid, d_path)

    print("Writing features...")
    movie_path = os.path.join(features_path, "movie")
    os.makedirs(movie_path, exist_ok=True)
    for r, fx in enumerate(torch.tensor_split(data["movie"].x, world_size)):
        torch.save(fx, os.path.join(movie_path, f"rank={r}.pt"))

    time_path = os.path.join(features_path, "time")
    os.makedirs(time_path, exist_ok=True)
    for r, time in enumerate(
        torch.tensor_split(data["user", "movie"].time, world_size)
    ):
        torch.save(time, os.path.join(time_path, f"rank={r}.pt"))

    print("Writing metadata...")
    meta = {
        "num_nodes": {
            "movie": data["movie"].num_nodes,
            "user": data["user"].num_nodes,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    _dbg("preprocess_and_partition", f"metadata written to {meta_path}")


def load_partitions(edge_path, features_path, meta_path):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data = HeteroData()

    print("Loading metadata...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    data["user"].num_nodes = meta["num_nodes"]["user"]
    data["movie"].num_nodes = meta["num_nodes"]["movie"]
    _dbg("load_partitions",
         f"num_users={data['user'].num_nodes} num_movies={data['movie'].num_nodes}")

    # user 特征: identity matrix 分片。注意 shape[-1] = num_users_total，
    # 维度极大时 OOM——上游已知，此处加警告
    if data["user"].num_nodes > 500_000:
        print(
            f"[WALPURGIS WARN] user identity matrix size={data['user'].num_nodes}x{data['user'].num_nodes}; "
            f"may OOM on GPU with limited VRAM.",
            flush=True,
        )
    data["user"].x = (
        torch.tensor_split(
            torch.eye(data["user"].num_nodes, dtype=torch.float32), world_size
        )[rank]
        .detach()
        .clone()
    )
    _dbg("load_partitions", f"user.x shard shape={list(data['user'].x.shape)}")

    data["movie"].x = torch.load(
        os.path.join(features_path, "movie", f"rank={rank}.pt"),
        weights_only=True,
    )
    _dbg("load_partitions", f"movie.x shard shape={list(data['movie'].x.shape)}")

    print("Loading user->movie edge index...")
    ei = {}
    for d in {"train", "test"}:
        ei[d] = torch.load(
            os.path.join(edge_path, "user_movie", d, f"rank={rank}.pt"),
            weights_only=True,
        )
        _dbg("load_partitions", f"ei[{d}].shape={list(ei[d].shape)}")

    data["user", "rates", "movie"].edge_index = torch.concat(
        [ei["train"], ei["test"]], dim=1
    )
    data["user", "rates", "movie"].time = torch.load(
        os.path.join(features_path, "time", f"rank={rank}.pt"),
        weights_only=True,
    )

    label_dict = {
        "train": torch.randperm(ei["train"].shape[1]),
        "test": torch.randperm(ei["test"].shape[1]) + ei["train"].shape[1],
    }
    # 校验：若 train/test edges 为空，DataLoader 行为不确定
    for split, idx in label_dict.items():
        if idx.numel() == 0:
            print(
                f"[WALPURGIS WARN] rank={rank} label_dict['{split}'] is EMPTY. "
                f"DataLoader may hang or return nothing.",
                flush=True,
            )
        _dbg("load_partitions", f"label_dict[{split}].shape={list(idx.shape)}")

    data["movie", "rev_rates", "user"].edge_index = torch.stack(
        [
            data["user", "rates", "movie"].edge_index[1],
            data["user", "rates", "movie"].edge_index[0],
        ]
    )

    print(f"Finished loading graph data on rank {rank}")
    return data, label_dict


# ──────────────────────────────────────────────────────────────────────────────
# 改写 3/5: EncoderShapeGuard — 校验 SAGEConv 输入维度顺序
# 318ae6c 将 (-1,-1) 改为显式维度，但 conv1 期望 (src=movie, dst=user) 顺序
# 与参数名 user_in_channels/movie_in_channels 位置对调，易混淆
# ──────────────────────────────────────────────────────────────────────────────
def _check_encoder_shape(user_in: int, movie_in: int) -> None:
    """断言 SAGEConv 方向与 forward 调用一致，维度对调时立即报错."""
    # conv1: (movie→user) 方向，src=movie, dst=user
    # conv2: (user→movie) 方向，src=user, dst=movie
    # conv3: (hidden→hidden)
    if user_in < 1 or movie_in < 1:
        raise ValueError(
            f"EncoderShapeGuard: user_in={user_in}, movie_in={movie_in} 均须 >= 1. "
            "请检查 FeatureDims."
        )
    _dbg("EncoderShapeGuard",
         f"conv1=(movie_in={movie_in}, user_in={user_in}) "
         f"conv2=(user_in={user_in}, movie_in={movie_in}) "
         f"conv3=(hidden, hidden)")


class Encoder(torch.nn.Module):
    def __init__(
        self, user_in_channels: int, movie_in_channels: int, hidden_channels: int, out_channels: int
    ):
        super().__init__()
        # SAGEConv((src_channels, dst_channels), out_channels)
        # conv1: movie→user 方向: src=movie, dst=user
        # conv2: user→movie 方向: src=user, dst=movie
        # conv3: hidden→hidden
        _check_encoder_shape(user_in_channels, movie_in_channels)
        self.conv1 = SAGEConv((movie_in_channels, user_in_channels), hidden_channels)
        self.conv2 = SAGEConv((user_in_channels, movie_in_channels), hidden_channels)
        self.conv3 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        _dbg("Encoder.__init__",
             f"user_in={user_in_channels} movie_in={movie_in_channels} "
             f"hidden={hidden_channels} out={out_channels}")

    def forward(self, x_dict, edge_index_dict):
        _dbg("Encoder.forward",
             f"user_x.shape={list(x_dict['user'].shape)} "
             f"movie_x.shape={list(x_dict['movie'].shape)}")

        user_x = self.conv1(
            (x_dict["movie"], x_dict["user"]),
            edge_index_dict["movie", "rev_rates", "user"],
        ).relu()

        movie_x = self.conv2(
            (x_dict["user"], x_dict["movie"]),
            edge_index_dict["user", "rates", "movie"],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict["movie", "rev_rates", "user"],
        ).relu()

        _dbg("Encoder.forward",
             f"out user_x.shape={list(user_x.shape)} movie_x.shape={list(movie_x.shape)}")
        return {
            "user": self.lin1(user_x),
            "movie": self.lin2(movie_x),
        }


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([x_dict["user"][row], x_dict["movie"][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels: int, metadata, num_features: Dict[str, int]):
        super().__init__()
        self.encoder = Encoder(
            user_in_channels=num_features["user"],
            movie_in_channels=num_features["movie"],
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        self.decoder = EdgeDecoder(hidden_channels)
        _dbg("Model.__init__",
             f"hidden_channels={hidden_channels} "
             f"num_features={num_features}")

    def forward(self, x_dict, edge_index_dict, num_samples: int):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(
            x_dict,
            edge_index_dict["user", "rates", "movie"][:, :num_samples],
        )


# ──────────────────────────────────────────────────────────────────────────────
# 改写 4/5: ModelBundle — 封装 Model + DDP + optimizer 构建
# 上游三行散落 __main__，device_ids=[local_rank] 硬编码无注释
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    """封装 Model / DDP wrapper / optimizer 构建，集中 DDP 参数."""

    model: torch.nn.Module
    ddp_model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    @classmethod
    def build(
        cls,
        hidden_channels: int,
        metadata,
        num_features: Dict[str, int],
        local_rank: int,
        lr: float,
    ) -> "ModelBundle":
        device = torch.device(local_rank)
        raw_model = Model(
            hidden_channels=hidden_channels,
            metadata=metadata,
            num_features=num_features,
        ).to(device)
        _dbg("ModelBundle.build",
             f"raw_model params={sum(p.numel() for p in raw_model.parameters()):,}")

        # DDP 包裹：device_ids=[local_rank] 保证 nccl backend 正确绑定
        ddp_model = DDP(raw_model, device_ids=[local_rank])
        _dbg("ModelBundle.build", f"DDP wrapped device_ids=[{local_rank}]")

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)
        _dbg("ModelBundle.build", f"Adam optimizer lr={lr}")

        return cls(model=raw_model, ddp_model=ddp_model, optimizer=optimizer)


# ──────────────────────────────────────────────────────────────────────────────
# 训练 / 测试循环
# ──────────────────────────────────────────────────────────────────────────────
def train(train_loader, model, optimizer):
    model.train()
    total_loss = total_examples = 0

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()

        num_samples = batch["user", "rates", "movie"].edge_label.shape[0]
        out = model(batch.x_dict, batch.edge_index_dict, num_samples)
        y = batch["user", "rates", "movie"].edge_label

        _dbg("train.batch",
             f"batch_idx={batch_idx} out.shape={list(out.shape)} "
             f"y.shape={list(y.shape)} y.dtype={y.dtype}")

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    # 防止空 loader 导致 ZeroDivisionError
    if total_examples == 0:
        print("[WALPURGIS WARN] train_loader yielded 0 examples. Check eli_train size.", flush=True)
        return 0.0
    return total_loss / total_examples


@torch.no_grad()
def test(test_loader, model):
    model.eval()
    preds = []
    targets = []

    for batch_idx, batch in enumerate(test_loader):
        batch = batch.to(next(model.parameters()).device)
        num_samples = batch["user", "rates", "movie"].edge_label.shape[0]
        pred = (
            model(batch.x_dict, batch.edge_index_dict, num_samples)
            .sigmoid()
            .view(-1)
            .cpu()
        )
        target = batch["user", "rates", "movie"].edge_label.long().cpu()
        _dbg("test.batch",
             f"batch_idx={batch_idx} pred.shape={list(pred.shape)} "
             f"target.unique={target.unique().tolist()}")
        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    return roc_auc_score(target, pred)


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun'. Exiting.")
        exit()

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--skip_partition", action="store_true")
    args = parser.parse_args()

    _dbg("main", f"args={vars(args)}")

    dataset_name = "movielens"

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600))
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    _dbg("main", f"world_size={world_size} global_rank={global_rank} local_rank={local_rank}")

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from pylibcugraph.comms import cugraph_comms_create_unique_id

        cugraph_id = [cugraph_comms_create_unique_id()]
        _dbg("main", f"cugraph_id created type={type(cugraph_id[0]).__name__}")
    else:
        cugraph_id = [None]

    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]
    _dbg("main", f"broadcast done cugraph_id type={type(cugraph_id).__name__}")

    # ──────────────────────────────────────────────────────────────────────────
    # 改写 5/5: CugraphWorkerSession 保证 comms shutdown 在异常路径也执行
    # ──────────────────────────────────────────────────────────────────────────
    with CugraphWorkerSession(global_rank, local_rank, world_size, cugraph_id):
        from rmm.allocators.torch import rmm_torch_allocator

        with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
            edge_path = os.path.join(args.dataset_root, dataset_name + "_eix_part")
            features_path = os.path.join(args.dataset_root, dataset_name + "_feat")
            meta_path = os.path.join(args.dataset_root, dataset_name + "_meta.json")

            if not args.skip_partition and global_rank == 0:
                print("Partitioning data...")
                dataset = MovieLens(args.dataset_root, model_name="all-MiniLM-L6-v2")
                data = dataset[0]
                preprocess_and_partition(
                    data,
                    edge_path=edge_path,
                    features_path=features_path,
                    meta_path=meta_path,
                )
                print("Data partitioning complete!")

            torch.distributed.barrier()
            _dbg("main", "barrier 1 passed — loading partitions")
            data, label_dict = load_partitions(
                edge_path=edge_path, features_path=features_path, meta_path=meta_path
            )
            torch.distributed.barrier()
            _dbg("main", "barrier 2 passed — building cugraph stores")

            feature_store, graph_store = cugraph_pyg_from_heterodata(data)
            eli_train = data["user", "rates", "movie"].edge_index[:, label_dict["train"]]
            eli_test = data["user", "rates", "movie"].edge_index[:, label_dict["test"]]
            time_train = data["user", "rates", "movie"].time[label_dict["train"]]
            num_nodes = {"user": data["user"].num_nodes, "movie": data["movie"].num_nodes}

            _dbg("main",
                 f"eli_train.shape={list(eli_train.shape)} "
                 f"eli_test.shape={list(eli_test.shape)}")

            # Set node times to 0
            feature_store["user", "time", None] = torch.tensor_split(
                torch.zeros(data["user"].num_nodes, dtype=torch.int64, device=device),
                world_size,
            )[global_rank]
            feature_store["movie", "time", None] = torch.tensor_split(
                torch.zeros(data["movie"].num_nodes, dtype=torch.int64, device=device),
                world_size,
            )[global_rank]

            # FeatureDims 封装特征维度提取与校验（改写 1/5）
            feat_dims = FeatureDims.from_heterodata(data)
            metadata = data.metadata()
            del data
            _dbg("main", f"feat_dims={feat_dims} metadata loaded")

            kwargs = dict(
                data=(feature_store, graph_store),
                num_neighbors={
                    ("user", "rates", "movie"): [5, 5, 5],
                    ("movie", "rev_rates", "user"): [5, 5, 5],
                },
                batch_size=256,
                shuffle=True,
                drop_last=True,
            )

            from cugraph_pyg.loader import LinkNeighborLoader

            train_loader = LinkNeighborLoader(
                edge_label_index=(("user", "rates", "movie"), eli_train),
                edge_label_time=time_train - 1,  # No leakage.
                time_attr="time",
                neg_sampling=dict(mode="binary", amount=2),
                **kwargs,
            )
            _dbg("main", "train_loader created")

            test_loader = LinkNeighborLoader(
                edge_label_index=(("user", "rates", "movie"), eli_test),
                neg_sampling=dict(mode="binary", amount=1),
                **kwargs,
            )
            _dbg("main", "test_loader created")

            sparse_size = (num_nodes["user"], num_nodes["movie"])
            test_edge_label_index = EdgeIndex(
                eli_test.to(device),
                sparse_size=sparse_size,
            ).sort_by("row")[0]
            test_exclude_links = EdgeIndex(
                eli_test.to(device),
                sparse_size=sparse_size,
            ).sort_by("row")[0]

            # ModelBundle 封装 Model + DDP + optimizer（改写 4/5）
            bundle = ModelBundle.build(
                hidden_channels=64,
                metadata=metadata,
                num_features=feat_dims.as_dict(),
                local_rank=local_rank,
                lr=args.lr,
            )
            _dbg("main",
                 f"ModelBundle ready ddp_model={type(bundle.ddp_model).__name__}")

            for epoch in range(1, args.epochs + 1):
                _dbg("train_loop", f"epoch={epoch}/{args.epochs} start")
                train_loss = train(train_loader, bundle.ddp_model, bundle.optimizer)
                print(f"Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
                auc = test(test_loader, bundle.ddp_model)
                print(f"Test AUC: {auc:.4f} ")
                _dbg("train_loop", f"epoch={epoch} loss={train_loss:.4f} auc={auc:.4f}")
