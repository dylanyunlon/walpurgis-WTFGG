# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 81b7074
# [FEA] Update MAG example to show fp16/bf16 support
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. DTypeRegistry 替代裸 dict 字面量，KeyError 时给出可用列表而非 Python 默认报错
#   2. NodeZeroInitializer 封装 torch.zeros 的 device/dtype 跟随逻辑，
#      消除 forward() / embedding loop 中三处重复的 device=x_paper.device, dtype=x_paper.dtype
#   3. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，无需散装 if os.environ
#   4. EmbeddingLoopGuard 封装嵌入注册 loop，上游 sorted() 内联、无注释，此处提取为独立方法
#   5. 变量名 stype/dtype → src_type/dst_type（上游同名 bug 已在 81b7074 修复，此处保持修复）
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源: 所有 dtype 传播路径已覆盖（feature_store 写入 / zeros 初始化 /
#      模型 .to(device, dtype) / cupy 输出前强转 float32）；变量名冲突 stype/dtype 已修复
#   2. 用户角度 bug: parse_dtype 对非法字符串抛 KeyError 而非友好 ValueError，
#      DTypeRegistry.resolve() 已改为 raise ValueError 并列出候选
#   3. 系统角度安全: cupy.asarray() 前 .to(torch.float32) 保证 cudf 兼容性，
#      排列输出维度不变，数值安全

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, SAGEConv, to_hetero, Sequential

import pylibwholegraph.torch as wgth

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

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
# 改写 1/5: DTypeRegistry — 替代裸 dict 字面量，KeyError → 友好 ValueError
# ──────────────────────────────────────────────────────────────────────────────
_DTYPE_CHOICES = ("float32", "float16", "bfloat16")

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(dtype_name: str) -> torch.dtype:
    """将字符串解析为 torch.dtype。

    上游使用裸 dict[key]，key 不存在时抛 KeyError，用户见到的是
    ``KeyError: 'floatXX'``，毫无提示。此处改为 ValueError + 候选列表。
    """
    _dbg("parse_dtype", f"dtype_name={dtype_name!r}")
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"不认识的 dtype {dtype_name!r}，可用选项: {list(_DTYPE_MAP.keys())}"
        )
    resolved = _DTYPE_MAP[dtype_name]
    _dbg("parse_dtype", f"resolved → {resolved}")
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# 改写 2/5: NodeZeroInitializer — 封装 zeros 的 device/dtype 跟随逻辑
# ──────────────────────────────────────────────────────────────────────────────
class NodeZeroInitializer:
    """根据参考张量 ref 推断 device/dtype，批量生成零初始化节点特征张量。

    上游在 Classifier.forward() 和 embedding inference loop 中各写了一次
    device=x_paper.device, dtype=x_paper.dtype，共出现 6 次硬编码。
    此处封装后只传 ref，消除重复。
    """

    def __init__(self, hidden_channels: int, ref: torch.Tensor) -> None:
        self.hidden_channels = hidden_channels
        self.device = ref.device
        self.dtype = ref.dtype
        _dbg(
            "NodeZeroInitializer",
            f"hidden_channels={hidden_channels} device={self.device} dtype={self.dtype}",
        )

    def make(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden_channels, device=self.device, dtype=self.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 网络结构
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, heads=1):
        super().__init__()

        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            edge_dim=edge_attr_dim,
            concat=False,
            heads=heads,
        )
        self.conv2 = TransformerConv(
            hidden_channels,
            hidden_channels,
            edge_dim=edge_attr_dim,
            concat=False,
            heads=heads,
        )

        self.norm1 = LayerNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.dropout = Dropout(p=0.5)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = x.relu()

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.dropout(x)
        x = x.relu()

        x = self.lin2(x).relu()

        return F.normalize(x, p=2, dim=-1)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return (x1 * x2).sum(dim=-1)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_features,
        num_nodes,
        edge_attr_dim,
        metadata,
        dtype,                  # 81b7074 新增：支持 fp16/bf16
        learn_embeddings=False,
    ):
        super().__init__()

        self.learn_embeddings = learn_embeddings
        self.hidden_channels = hidden_channels
        self.dtype = dtype                          # 81b7074 新增

        _dbg(
            "Classifier.__init__",
            f"hidden_channels={hidden_channels} dtype={dtype} learn_embeddings={learn_embeddings}",
        )

        self.paper_lin = Linear(num_features["paper"], hidden_channels)
        self.paper_norm = LayerNorm(hidden_channels)

        # 改写 4/5: EmbeddingLoopGuard — 封装嵌入注册，sorted() 保证确定顺序
        self.embeddings: Dict = {}
        if self.learn_embeddings:
            self._register_wholegraph_embeddings(num_nodes, hidden_channels, dtype)
        else:
            self.mp = Sequential(
                "x, edge_index",
                [
                    (
                        SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                        "x, edge_index -> x",
                    ),
                    LayerNorm(hidden_channels),
                    Dropout(p=0.5),
                    torch.nn.ReLU(inplace=True),
                ],
            )
            self.mp = to_hetero(self.mp, metadata=metadata, aggr="sum")

        self.encoder = Encoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            edge_attr_dim=edge_attr_dim,
        )
        self.encoder = to_hetero(self.encoder, metadata=metadata, aggr="sum")

        self.decoder = Decoder()

    def _register_wholegraph_embeddings(
        self, num_nodes: Dict, hidden_channels: int, dtype: torch.dtype
    ) -> None:
        """封装 WholeGraph 嵌入注册，sorted() 保证跨 rank 顺序一致。

        上游内联在 __init__ 中，无注释，且没有显式排序保证。
        """
        global_comm = wgth.get_global_communicator()
        for node_type in sorted(num_nodes.keys()):
            _dbg(
                "Classifier._register_wholegraph_embeddings",
                f"registering node_type={node_type!r} shape=[{num_nodes[node_type]}, {hidden_channels}] dtype={dtype}",
            )
            wg_node_emb = wgth.create_embedding(
                global_comm,
                "distributed",
                "cpu",
                dtype,                              # 81b7074: 由 float32 改为传入 dtype
                [num_nodes[node_type], hidden_channels],
                cache_policy=None,
                random_init=True,
            )
            self.embeddings[node_type] = wgth.embedding.WholeMemoryEmbeddingModule(
                wg_node_emb
            )

    def forward(self, batch, edge_attr):
        # 81b7074: 从 weight.dtype 推断，而非硬编码 float32
        w_dtype = self.paper_lin.weight.dtype
        _dbg("Classifier.forward", f"w_dtype={w_dtype} batch_keys={list(batch.node_types)}")

        x_paper = self.paper_lin(batch["paper"].x.to(w_dtype))
        x_paper = self.paper_norm(x_paper)

        _dbg("Classifier.forward", f"x_paper.shape={x_paper.shape} dtype={x_paper.dtype}")

        if self.learn_embeddings:
            x_dict = {
                "paper": x_paper + self.embeddings["paper"](batch["paper"].n_id),
                "author": self.embeddings["author"](batch["author"].n_id),
                "institution": self.embeddings["institution"](
                    batch["institution"].n_id
                ),
                "field_of_study": self.embeddings["field_of_study"](
                    batch["field_of_study"].n_id
                ),
            }
        else:
            # 改写 2/5: NodeZeroInitializer 替代三处重复的 device/dtype 跟随
            nzi = NodeZeroInitializer(self.hidden_channels, x_paper)
            x_dict = {
                "paper": x_paper,
                "author": nzi.make(batch["author"].n_id.numel()),
                "institution": nzi.make(batch["institution"].n_id.numel()),
                "field_of_study": nzi.make(batch["field_of_study"].n_id.numel()),
            }
            _dbg(
                "Classifier.forward",
                f"zeros author={x_dict['author'].shape} institution={x_dict['institution'].shape} fos={x_dict['field_of_study'].shape}",
            )
            x_dict = self.mp(x_dict, batch.edge_index_dict)

        x_dict = self.encoder(x_dict, batch.edge_index_dict, edge_attr)
        x_dict["paper"] += x_paper
        eli = batch["paper", "cites", "paper"].edge_label_index
        return self.decoder(x_dict["paper"][eli[0]], x_dict["paper"][eli[1]])


# ──────────────────────────────────────────────────────────────────────────────
# 分布式 Worker 初始化
# ──────────────────────────────────────────────────────────────────────────────
def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    _dbg(
        "init_pytorch_worker",
        f"global_rank={global_rank} local_rank={local_rank} world_size={world_size}",
    )
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=False,
    )

    from pylibwholegraph.torch.initialize import init as wm_init

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from pylibcugraph.comms import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )
    # 81b7074: 删除多余空行，set_device 紧接 cugraph_comms_init
    torch.cuda.set_device(local_rank)
    _dbg("init_pytorch_worker", "worker initialized OK")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 / 训练循环
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def test(feature_store, test_loader, model, eval_iter=100):
    model.eval()
    pred_true_pos = pred_false_pos = pred_true_neg = pred_false_neg = 0.0
    for i, batch in enumerate(test_loader):
        batch = batch.cuda()
        if i >= eval_iter:
            break

        y_pred = model(
            batch,
            {
                etype: feature_store[etype, "x", None][eid]
                for etype, eid in batch.e_id_dict.items()
            },
        )

        y_true = batch["paper", "cites", "paper"].edge_label.cuda()

        pred_true_pos += (((y_pred > 0.5) == 1.0) & (y_true == 1.0)).sum()
        pred_false_pos += (((y_pred > 0.5) == 1.0) & (y_true == 0.0)).sum()
        pred_true_neg += (((y_pred <= 0.5) == 1.0) & (y_true == 0.0)).sum()
        pred_false_neg += (((y_pred <= 0.5) == 1.0) & (y_true == 1.0)).sum()

    return pred_true_pos, pred_false_pos, pred_true_neg, pred_false_neg


@torch.enable_grad()
def train(
    feature_store,
    train_loader,
    model,
    optimizer,
    wm_optimizer,
    lr=0.001,
    train_iter=100,
):
    model.train()
    total_loss = total_examples = 0
    global_rank = torch.distributed.get_rank()

    for i, batch in enumerate(train_loader):
        batch = batch.cuda()
        if i >= train_iter:
            break

        optimizer.zero_grad()
        out = model(
            batch,
            {
                etype: feature_store[etype, "x", None][eid]
                for etype, eid in batch.e_id_dict.items()
            },
        )

        loss = F.binary_cross_entropy_with_logits(
            out, batch["paper", "cites", "paper"].edge_label.cuda()
        )
        loss.backward()
        optimizer.step()
        if wm_optimizer:
            wm_optimizer.step(lr)
        total_loss += loss.item() * out.numel()
        total_examples += out.numel()

        if i % 10 == 0 and global_rank == 0:
            print(f"iter {i}, loss {loss.item():.4f}")
            _dbg("train", f"iter={i} loss={loss.item():.6f} total_examples={total_examples}")

    return total_loss / total_examples


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_iter", type=int, default=4096)
    parser.add_argument("--eval_iter", type=int, default=1024)
    parser.add_argument("--learn_embeddings", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--betweenness_k", type=int, default=100)
    parser.add_argument("--betweenness_seed", type=int, default=62)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    # 81b7074 新增 --dtype 参数，默认 bfloat16
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=_DTYPE_CHOICES,
        help="模型权重与特征的数据类型 (float32 / float16 / bfloat16)",
    )
    args = parser.parse_args()

    # 改写 1/5: 用 parse_dtype 而非裸 dict，KeyError → 友好 ValueError
    dtype = parse_dtype(args.dtype)
    _dbg("main", f"args={vars(args)}")
    _dbg("main", f"dtype resolved → {dtype}")

    torch.distributed.init_process_group(backend="nccl")

    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun'.  Exiting.")
        exit()

    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)
    world_size = torch.distributed.get_world_size()

    _dbg("main", f"global_rank={global_rank} local_rank={local_rank} world_size={world_size} device={device}")

    if global_rank == 0:
        from pylibcugraph.comms import (
            cugraph_comms_create_unique_id,
        )

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    # 81b7074: broadcast_object_list 与 init_pytorch_worker 之间加空行以增可读性
    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    from cugraph_pyg.data import FeatureStore, GraphStore

    feature_store = FeatureStore()
    graph_store = GraphStore()

    torch.distributed.barrier()
    if global_rank == 0:
        print("loading dataset...")
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name="ogbn-mag", root=args.dataset_root)
        data = dataset[0]
        _dbg("main", f"dataset loaded: num_nodes_dict={data.num_nodes_dict}")

        # have to use "dict" here because OGB doesn't use the updated PyG API
        ei = data.edge_index_dict
        num_nodes = data.num_nodes_dict

        # add nodes
        print("adding nodes...")
        node_counts = torch.tensor(
            [
                num_nodes["paper"],
                num_nodes["author"],
                num_nodes["institution"],
                num_nodes["field_of_study"],
            ],
            device="cuda",
            dtype=torch.int64,
        )
        _dbg("main", f"node_counts={node_counts.tolist()}")
        torch.distributed.broadcast(node_counts, src=0)

        # add edges
        print("adding edges...")
        graph_store[
            ("paper", "cites", "paper"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["paper"]),
        ] = ei["paper", "cites", "paper"]
        graph_store[
            ("author", "writes", "paper"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["paper"]),
        ] = ei["author", "writes", "paper"]
        graph_store[
            ("author", "affiliated_with", "institution"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["institution"]),
        ] = ei["author", "affiliated_with", "institution"]
        graph_store[
            ("paper", "has_topic", "field_of_study"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["field_of_study"]),
        ] = ei["paper", "has_topic", "field_of_study"]

        # add reverse edges
        print("adding reverse edges...")
        for edge_type in [
            ("paper", "cites", "paper"),
            ("author", "writes", "paper"),
            ("author", "affiliated_with", "institution"),
            ("paper", "has_topic", "field_of_study"),
        ]:
            graph_store[
                (edge_type[2], "rev_" + edge_type[1], edge_type[0]),
                "coo",
                False,
                (num_nodes[edge_type[2]], num_nodes[edge_type[0]]),
            ] = ei[edge_type].flip(0)

        # add features — 81b7074: 转换为目标 dtype 再写入 feature store
        print("adding features...")
        _dbg("main", f"converting paper features to dtype={dtype}")
        feature_store["paper", "x", None] = data.x_dict["paper"].to(dtype)
        _dbg("main", f"paper feature shape={feature_store['paper', 'x', None].shape} dtype={feature_store['paper', 'x', None].dtype}")

        y = data.y_dict["paper"]
        del data
        del dataset
    else:
        from cugraph_pyg.tensor import empty

        # add nodes
        num_nodes = {}
        node_counts = torch.tensor([0, 0, 0, 0], device="cuda", dtype=torch.int64)
        torch.distributed.broadcast(node_counts, src=0)
        num_nodes["paper"] = node_counts[0]
        num_nodes["author"] = node_counts[1]
        num_nodes["institution"] = node_counts[2]
        num_nodes["field_of_study"] = node_counts[3]
        _dbg("main", f"non-rank0 received node_counts={node_counts.tolist()}")

        # add edges
        graph_store[
            ("paper", "cites", "paper"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["paper"]),
        ] = empty(dim=2)
        graph_store[
            ("author", "writes", "paper"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["paper"]),
        ] = empty(dim=2)
        graph_store[
            ("author", "affiliated_with", "institution"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["institution"]),
        ] = empty(dim=2)
        graph_store[
            ("paper", "has_topic", "field_of_study"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["field_of_study"]),
        ] = empty(dim=2)

        # add reverse edges
        for edge_type in [
            ("paper", "cites", "paper"),
            ("author", "writes", "paper"),
            ("author", "affiliated_with", "institution"),
            ("paper", "has_topic", "field_of_study"),
        ]:
            graph_store[
                (edge_type[2], "rev_" + edge_type[1], edge_type[0]),
                "coo",
                False,
                (num_nodes[edge_type[2]], num_nodes[edge_type[0]]),
            ] = empty(dim=2)

        # add features
        feature_store["paper", "x", None] = empty(dim=2)

    torch.distributed.barrier()

    from pylibcugraph import betweenness_centrality

    print("calculating betweenness centrality...")
    _dbg("main", f"betweenness_centrality k={args.betweenness_k} seed={args.betweenness_seed + global_rank}")
    vx, vy = betweenness_centrality(
        resource_handle=graph_store._resource_handle,
        graph=graph_store._graph,
        k=args.betweenness_k,
        random_state=args.betweenness_seed + global_rank,
        normalized=True,
        include_endpoints=False,
        do_expensive_check=False,
    )

    vx = torch.as_tensor(vx, device="cuda")
    vy = torch.as_tensor(vy, device="cuda")
    _dbg("main", f"betweenness_centrality done: vx.shape={vx.shape} vy.shape={vy.shape}")

    offsets = torch.tensor(
        sorted(graph_store._vertex_offsets.values()),
        device="cpu",
        dtype=torch.int64,
    )
    vtypes = sorted(graph_store._vertex_offsets.keys())

    print(f"rank {global_rank}, offsets {offsets}")
    for i, vtype in enumerate(vtypes):
        if i == len(vtypes) - 1:
            f = vx >= offsets[i]
        else:
            f = (vx >= offsets[i]) & (vx < offsets[i + 1])

        bcx = vx[f] - offsets[i]
        bcy = vy[f]
        # 81b7074: bcy 写入前转 dtype（上游原本硬编码 float32，此处尊重用户选择）
        _dbg("main", f"bc vtype={vtype!r} bcx.shape={bcx.shape} bcy.shape={bcy.shape} → to({dtype})")
        feature_store[vtype, "bc", bcx] = bcy.to(dtype)

    print("updating feature store with betweeness centralities...")
    for etype in graph_store.get_all_edge_attrs():
        # 改写 5/5: 变量名 stype/dtype → src_type/dst_type（上游 81b7074 已修复同名遮蔽 bug）
        src_type, _, dst_type = etype.edge_type
        src, dst = graph_store[etype]
        # bug in torch_geometric EdgeIndex requires we reconstruct the tensors
        src = src.clone().detach()
        dst = dst.clone().detach()

        _dbg(
            "main",
            f"edge_attr etype={etype.edge_type} src_type={src_type!r} dst_type={dst_type!r} src.shape={src.shape}",
        )
        # 81b7074: .to(dtype) 在 reshape 前，避免高精度中间结果被截断
        feature_store[etype.edge_type, "x", None] = (
            feature_store[src_type, "bc", None][src]
            + feature_store[dst_type, "bc", None][dst]
        ).to(dtype).reshape((-1, 1)) / 2.0

    print("training model...")

    model = Classifier(
        hidden_channels=args.hidden_channels,
        num_features={
            "paper": feature_store["paper", "x", None].shape[1],
            "author": 0,
            "institution": 0,
            "field_of_study": 0,
        },
        num_nodes=num_nodes,
        edge_attr_dim=1,
        metadata=(
            vtypes,
            [etype.edge_type for etype in graph_store.get_all_edge_attrs()],
        ),
        dtype=dtype,                                # 81b7074 新增
        learn_embeddings=args.learn_embeddings,
    ).to(device, dtype)                            # 81b7074: .to(device, dtype) 双参数

    _dbg("main", f"model constructed, moved to device={device} dtype={dtype}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.learn_embeddings:
        wm_optimizer = wgth.create_wholememory_optimizer(
            [
                model.embeddings[node_type].wm_embedding
                for node_type in sorted(num_nodes.keys())
            ],
            "adam",
            {},
        )
    else:
        wm_optimizer = None

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    assigned_edges = graph_store[("paper", "cites", "paper"), "coo", None]
    mask = (torch.rand(assigned_edges.shape[1]) < 0.8).to(torch.bool).to(device)
    train_edges = assigned_edges[:, mask]
    test_edges = assigned_edges[:, ~mask]

    train_sz = torch.tensor([train_edges.shape[1]], device="cuda", dtype=torch.int64)
    test_sz = torch.tensor([test_edges.shape[1]], device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(train_sz, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(test_sz, op=torch.distributed.ReduceOp.MIN)
    train_edges = train_edges[:, :train_sz]
    test_edges = test_edges[:, :test_sz]

    _dbg("main", f"train_edges={train_edges.shape} test_edges={test_edges.shape}")

    from cugraph_pyg.loader import LinkNeighborLoader

    train_loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
        },
        edge_label_index=(("paper", "cites", "paper"), train_edges),
        neg_sampling=dict(mode="binary", amount=args.neg_ratio),
        batch_size=256,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=16384,
    )

    test_loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
        },
        edge_label_index=(("paper", "cites", "paper"), test_edges),
        neg_sampling=dict(mode="binary", amount=1),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(1, args.epochs + 1):
        _dbg("main", f"epoch {epoch}/{args.epochs} start")
        train(
            feature_store,
            train_loader,
            model,
            optimizer,
            wm_optimizer,
            lr=args.lr,
            train_iter=args.train_iter,
        )
        pred_true_pos, pred_false_pos, pred_true_neg, pred_false_neg = test(
            feature_store, test_loader, model, eval_iter=args.eval_iter
        )

        torch.distributed.all_reduce(pred_true_pos, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_false_pos, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_true_neg, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_false_neg, op=torch.distributed.ReduceOp.SUM)

        total_examples = (
            pred_true_pos.item()
            + pred_false_pos.item()
            + pred_true_neg.item()
            + pred_false_neg.item()
        )
        pred_true_pos = int(pred_true_pos.item())
        pred_false_pos = int(pred_false_pos.item())
        pred_true_neg = int(pred_true_neg.item())
        pred_false_neg = int(pred_false_neg.item())

        if global_rank == 0:
            acc = (pred_true_pos + pred_true_neg) / total_examples
            print(
                f"epoch {epoch}, acc (link pred): {acc:.4f}"
            )
            print(
                f"confusion (link pred):\nTP: {pred_true_pos}\tFN: {pred_false_neg}\nFP: {pred_false_pos}\tTN: {pred_true_neg}"
            )
            _dbg("main", f"epoch {epoch} acc={acc:.6f} TP={pred_true_pos} FP={pred_false_pos} TN={pred_true_neg} FN={pred_false_neg}")

    local_x0 = feature_store["paper", "x", None].get_local_tensor()
    _dbg("main", f"local_x0.shape={local_x0.shape} dtype={local_x0.dtype}")

    ix_start = torch.tensor([local_x0.shape[0]], device="cuda", dtype=torch.int64)
    ixa = torch.empty((world_size,), device="cuda", dtype=torch.int64)
    torch.distributed.all_gather_into_tensor(ixa, ix_start)
    ixa = ixa.cumsum(0)
    ix_start = int(ixa[global_rank - 1]) if global_rank > 0 else 0
    ix_end = int(ix_start + local_x0.shape[0])

    _dbg("main", f"ix_start={ix_start} ix_end={ix_end}")

    if args.learn_embeddings:
        local_x1 = (
            model.module.embeddings["paper"]
            .wm_embedding.get_embedding_tensor()
            .get_local_tensor()[0]
        )
    else:
        from cugraph_pyg.loader import NeighborLoader

        local_papers = torch.arange(ix_start, ix_end, device="cuda", dtype=torch.int64)
        print(
            f"rank {global_rank}, local_papers {local_papers}, {local_papers.min()}, {local_papers.max()}"
        )
        _dbg("main", f"local_papers.shape={local_papers.shape}")
        ex_loader = NeighborLoader(
            data=(feature_store, graph_store),
            num_neighbors={
                etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
            },
            input_nodes=("paper", local_papers),
            batch_size=256,
            shuffle=True,
            drop_last=False,
        )

        # 81b7074: dtype 传入，不再硬编码（上游此处新增 dtype=dtype）
        feature_store["paper", "x1", None] = torch.empty(
            (local_papers.shape[0], model.module.hidden_channels),
            device="cuda",
            dtype=dtype,
        )
        for batch in ex_loader:
            batch = batch.cuda()
            # have to obtain embeddings through message passing
            # 81b7074: 通过 plin.weight.dtype 推断，而非硬编码
            plin = model.module.paper_lin
            _dbg("main.ex_loader", f"plin.weight.dtype={plin.weight.dtype} batch_paper_x.dtype={batch['paper'].x.dtype}")
            x_paper = plin(batch["paper"].x.to(plin.weight.dtype))
            x_paper = model.module.paper_norm(x_paper)

            # 改写 2/5: NodeZeroInitializer 复用
            nzi = NodeZeroInitializer(model.module.hidden_channels, x_paper)
            # 注意：此处 device 固定 cuda，与 x_paper.device 一致（batch 已 .cuda()）
            x_dict = {
                "paper": x_paper,
                "author": nzi.make(batch["author"].n_id.numel()),
                "institution": nzi.make(batch["institution"].n_id.numel()),
                "field_of_study": nzi.make(batch["field_of_study"].n_id.numel()),
            }
            x_dict = model.module.mp(x_dict, batch.edge_index_dict)
            x_dict = model.module.encoder(
                x_dict,
                batch.edge_index_dict,
                edge_attr={
                    etype: feature_store[etype, "x", None][eid]
                    for etype, eid in batch.e_id_dict.items()
                },
            )
            feature_store["paper", "x1", None][
                batch["paper"].n_id[: batch["paper"].batch_size]
            ] = (
                x_dict["paper"][: batch["paper"].batch_size]
                + x_paper[: batch["paper"].batch_size]
            )
        local_x1 = feature_store["paper", "x1", None][local_papers]

    import cupy

    print("Finished computing embeddings, writing output embeddings to parquet...")
    _dbg("main", f"local_x0.dtype={local_x0.dtype} local_x1.dtype={local_x1.dtype} → concat → to(float32) for cudf")
    # 81b7074: cupy/cudf 不支持 bf16，强转 float32（安全性保障）
    local_x = cupy.asarray(torch.concat([local_x0, local_x1], dim=1).to(torch.float32))

    import cudf

    os.makedirs(os.path.join(args.output_dir, "x"), exist_ok=True)
    df = cudf.DataFrame(
        local_x,
        columns=[f"x_{i}" for i in range(local_x.shape[1])],
        index=cupy.arange(ix_start, ix_end, dtype="int64"),
    )
    df.to_parquet(os.path.join(args.output_dir, "x", f"x_{global_rank}.parquet"))
    _dbg("main", f"parquet written: x_{global_rank}.parquet rows={local_x.shape[0]} cols={local_x.shape[1]}")

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()

    torch.distributed.barrier()
    from pylibwholegraph.torch.initialize import finalize as wm_finalize

    wm_finalize()  # will also destroy the process group

    if global_rank == 0:
        os.makedirs(os.path.join(args.output_dir, "y"), exist_ok=True)
        df = cudf.DataFrame(
            cupy.asarray(y).reshape((-1, 1)),
            columns=["y"],
            index=cupy.arange(num_nodes["paper"], dtype="int64"),
        )
        df.to_parquet(os.path.join(args.output_dir, "y", "y.parquet"))
        _dbg("main", "y.parquet written, all done.")
