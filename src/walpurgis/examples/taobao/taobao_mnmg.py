# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit f2b7f50
# [BUG] Fix shuffle on single GPU in Taobao Example
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. EdgeShuffler 数据类封装 balance_shuffle_edge_split 逻辑
#      上游函数散落两段分支，单 GPU / 多 GPU 逻辑混在 if-elif-else 链中
#      EdgeShuffler.split() 将单卡路径提前 early-return，多卡路径清晰分离
#   2. _compute_local_slice() 静态方法封装 start/end 计算
#      上游 f2b7f50 修复后 start/end 两行仍散落 else 块，无名称，无注释
#      _compute_local_slice 给这段逻辑一个名字，断点调试打印 rank/slice 信息
#   3. DataPreprocessor 封装 preprocess_and_partition 的 del 链
#      上游 7 行 del + print(data) 散落函数顶，DataPreprocessor.__call__ 集中管理
#      _dbg() 在 WALPURGIS_DEBUG=1 时逐步打印 data 结构变化
#   4. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，无需散装 if os.environ
#   5. 全链路断点调试 print，覆盖:
#      DataPreprocessor: data 结构 → del 后变化
#      EdgeShuffler: world_size / rank / edge_offsets / local_slice
#      balance_shuffle_edge_split: broadcast 前后 dst_rank 分布
#      __main__: 各阶段 barrier 检查点
#
# Knuth 审查结论（迁移前三问）:
#   1. diff 对比源:
#      旧代码 (f2b7f50 之前):
#        if rank > 0 and rank < world_size - 1:
#            local_rank_t = dst_rank[edge_offsets[rank - 1] : edge_offsets[rank]]
#        elif rank == 0:
#            local_rank_t = dst_rank[0 : edge_offsets[0]]
#        else:       # rank == world_size - 1（最后一个 rank）
#            local_rank_t = dst_rank[edge_offsets[-1] :]
#      当 world_size == 1 时:
#        rank == 0 且 rank == world_size - 1 同时成立
#        elif rank == 0 分支被触发 → local_rank_t = dst_rank[0 : edge_offsets[0]]
#        但 edge_offsets = num_edges.cumsum(0).cpu()[:-1]
#        world_size == 1 时 num_edges.shape = (1,)，cumsum = (N,)，[:-1] = empty tensor ()
#        edge_offsets[0] 触发 IndexError（单卡时直接崩溃）
#        即使不崩溃：edge_offsets 为空，edge_offsets[0] 是未定义行为
#      新代码 (f2b7f50):
#        if world_size == 1:
#            local_rank_t = dst_rank           # 全量，直接用
#        else:
#            start = 0 if rank == 0 else edge_offsets[rank - 1]
#            end = edge_offsets[rank] if rank < world_size - 1 else None
#            local_rank_t = dst_rank[start:end]
#      多卡路径重写更清晰：start/end 二元组替代三段 if-elif-else
#      None 作为 end 切片等价于"到末尾"，避免了旧代码 else 分支的隐含语义
#
#   2. 用户角度 bug:
#      - 单卡训练（最常见的调试场景）：运行到 balance_shuffle_edge_split 直接
#        IndexError: index 0 is out of bounds for dimension 0 with size 0
#        错误信息指向 edge_offsets[0]，与业务逻辑毫无关联，难以定位
#      - 错误在 balance_shuffle_edge_split 内部，而非用户代码，stack trace 深，
#        初学者会怀疑环境问题而非代码 bug
#      - world_size==1 时 torch.distributed.broadcast 仍被调用（合法但无意义），
#        旧代码在 broadcast 之后才崩溃，浪费了 broadcast 的开销
#      - 多卡路径旧代码三段式逻辑：rank==0, 0<rank<N-1, rank==N-1
#        f2b7f50 将其统一为 start/end 二元组，消除了边界条件穷举的认知负担
#
#   3. 系统角度安全:
#      - edge_offsets 是 cumsum[:-1]，长度 = world_size - 1
#        旧代码在 elif rank==0 中访问 edge_offsets[0]，
#        当 world_size==1 时 edge_offsets 长度为 0，IndexError 确定性触发
#      - dst_rank = torch.randperm(total_num_edges) % world_size
#        world_size==1 时 dst_rank 全为 0（合法），广播正常，
#        但后续切片逻辑是 bug 的所在
#      - all_gather_into_tensor + broadcast 是集体操作，
#        单卡环境下它们退化为 no-op（或单进程完成），不会挂起
#      - 新代码 world_size==1 直接 local_rank_t = dst_rank（全量），
#        跳过了 edge_offsets 的访问，从根本上规避了 IndexError
#      - all_to_all 在 world_size==1 时等价于 rx[0] = s[0]（单槽自环），
#        此路径在新旧代码中均存在，f2b7f50 未修改，属已知 benign path

import os
import warnings
import argparse
import json
import gc
from datetime import timedelta
from time import perf_counter
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch.nn.parallel import DistributedDataParallel
import torch_geometric.transforms as T
from torch_geometric.datasets import Taobao
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.data import HeteroData

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)
from sklearn.metrics import roc_auc_score

# e01196b: CUDF_SPILL removed — RMM managed_memory handles over-subscription now.
# cudf spilling via CUDF_SPILL=1 depended on cudf which is no longer required.
os.environ["RAPIDS_NO_INITIALIZE"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# 调试工具（鲁迅拿法：不用 logging，用会说话的 print）
# ──────────────────────────────────────────────────────────────────────────────

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, tag: str = "WALPURGIS") -> None:
    """断点调试 print，仅 WALPURGIS_DEBUG=1 时输出。"""
    if _DEBUG:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[{tag}][rank={rank}]", *args, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────────────────────────────────────────

def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    _dbg(f"init_pytorch_worker: global_rank={global_rank} local_rank={local_rank} "
         f"world_size={world_size}", tag="INIT")

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
    )
    _dbg("rmm.reinitialize done", tag="INIT")

    import cupy
    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    _dbg("cupy allocator set", tag="INIT")
    # e01196b: enable_spilling() removed — cudf dependency eliminated.
    # Memory over-subscription handled by RMM managed_memory + WholeGraph UVA.
    _dbg("cudf spilling skipped (e01196b: use RMM managed_memory instead)", tag="INIT")

    torch.cuda.set_device(local_rank)

    from pylibcugraph.comms import cugraph_comms_init
    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )
    _dbg("cugraph_comms_init done", tag="INIT")

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())
    _dbg("wm_init done", tag="INIT")


# ──────────────────────────────────────────────────────────────────────────────
# 模型定义（与上游保持一致，无改写）
# ──────────────────────────────────────────────────────────────────────────────

class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict["item"],
            edge_index_dict[("item", "to", "item")],
        ).relu()
        user_x = self.conv2(
            (x_dict["item"], x_dict["user"]),
            edge_index_dict[("item", "rev_to", "user")],
        ).relu()
        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[("item", "rev_to", "user")],
        ).relu()
        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.item_emb = Embedding(num_items, hidden_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict["user"] = self.user_emb(x_dict["user"])
        x_dict["item"] = self.item_emb(x_dict["item"])
        z_dict["item"] = self.item_encoder(
            x_dict["item"],
            edge_index_dict[("item", "to", "item")],
        )
        z_dict["user"] = self.user_encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict["user"], z_dict["item"], edge_label_index)


# ──────────────────────────────────────────────────────────────────────────────
# 数据预处理（鲁迅拿法改写 #1：DataPreprocessor 封装 del 链）
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataPreprocessor:
    """
    封装 preprocess_and_partition 前段的数据清洗逻辑。
    上游散落 del + 裸 print(data)；此处集中管理，DEBUG 模式逐步可见。

    改写点（vs 上游 f2b7f50 preprocess_and_partition 前段）:
      - print(data)  →  _dbg("data before del", ...) 仅 DEBUG 时输出
      - 三行 del 散落  →  _clean() 方法，名称说明意图
    """

    def _clean(self, data) -> None:
        _dbg(f"DataPreprocessor._clean: data keys before = {list(data.node_types)}", tag="PREPROCESS")
        _dbg(f"  edge_types before = {list(data.edge_types)}", tag="PREPROCESS")

        # 上游 f2b7f50: print(data) — 断点调试，保留为 _dbg
        _dbg(f"data repr:\n{data}", tag="PREPROCESS")

        del data["category"]
        del data["item", "category"]
        del data["user", "item"].time
        del data["user", "item"].behavior

        _dbg(f"DataPreprocessor._clean: data keys after = {list(data.node_types)}", tag="PREPROCESS")
        _dbg(f"  edge_types after = {list(data.edge_types)}", tag="PREPROCESS")

    def __call__(self, data, edge_path: str, meta_path: str) -> None:
        """对应上游 preprocess_and_partition，含清洗 + 写出。"""
        self._clean(data)

        print("Writing item->item edge partitions...")
        item_item_edge_path = os.path.join(edge_path, "item_item")
        write_edges(data["item", "item"].edge_index, item_item_edge_path)

        print("Writing user->item edge partitions...")
        user_item_edge_path = os.path.join(edge_path, "user_item")
        write_edges(data["user", "item"].edge_index, user_item_edge_path)

        print("Writing metadata...")
        meta = {
            "num_nodes": {
                "item": data["item"].num_nodes,
                "user": data["user"].num_nodes,
            }
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        _dbg(f"preprocess_and_partition done: meta={meta}", tag="PREPROCESS")


# ──────────────────────────────────────────────────────────────────────────────
# 边写出工具（与上游一致）
# ──────────────────────────────────────────────────────────────────────────────

def write_edges(edge_index, path):
    world_size = torch.distributed.get_world_size()
    os.makedirs(path, exist_ok=True)
    for r, e in enumerate(torch.tensor_split(edge_index, world_size, dim=1)):
        rank_path = os.path.join(path, f"rank={r}.pt")
        torch.save(e.clone(), rank_path)


def pre_transform(data):
    print("Computing item->item relationships (this may take a very long time)...")
    mat = to_scipy_sparse_matrix(data["user", "item"].edge_index).tocsr()
    mat = mat[: data["user"].num_nodes, : data["item"].num_nodes]
    comat = mat.T @ mat
    comat.setdiag(0)
    comat = comat >= 3.0
    comat = comat.tocoo()
    row = torch.from_numpy(comat.row).to(torch.long)
    col = torch.from_numpy(comat.col).to(torch.long)
    data["item", "item"].edge_index = torch.stack([row, col], dim=0)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 分区加载（与上游一致，加 _dbg 断点）
# ──────────────────────────────────────────────────────────────────────────────

def cugraph_pyg_from_heterodata(data, return_edge_label=True):
    from cugraph_pyg.data import GraphStore, FeatureStore

    _dbg(f"cugraph_pyg_from_heterodata: item.x={data['item'].x.shape} "
         f"user.x={data['user'].x.shape}", tag="STORE")

    graph_store = GraphStore()
    feature_store = FeatureStore()

    graph_store[
        ("user", "to", "item"), "coo", False,
        (data["user"].num_nodes, data["item"].num_nodes),
    ] = data["user", "to", "item"].edge_index
    graph_store[
        ("item", "rev_to", "user"), "coo", False,
        (data["item"].num_nodes, data["user"].num_nodes),
    ] = data["item", "rev_to", "user"].edge_index
    graph_store[
        ("item", "to", "item"), "coo", False,
        (data["item"].num_nodes, data["item"].num_nodes),
    ] = data["item", "to", "item"].edge_index
    graph_store[
        ("item", "rev_to", "item"), "coo", False,
        (data["item"].num_nodes, data["item"].num_nodes),
    ] = data["item", "rev_to", "item"].edge_index

    feature_store["item", "x", None] = data["item"].x
    feature_store["user", "x", None] = data["user"].x

    out = (
        (feature_store, graph_store),
        data["user", "to", "item"].edge_label_index,
        (data["user", "to", "item"].edge_label if return_edge_label else None),
    )
    return out


def load_partitions(edge_path, meta_path):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data = HeteroData()

    print("Loading metadata...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    _dbg(f"load_partitions: meta={meta} rank={rank} world_size={world_size}", tag="LOAD")

    data["user"].num_nodes = meta["num_nodes"]["user"]
    data["item"].num_nodes = meta["num_nodes"]["item"]

    data["user"].x = torch.tensor_split(
        torch.arange(data["user"].num_nodes), world_size
    )[rank]
    data["item"].x = torch.tensor_split(
        torch.arange(data["item"].num_nodes), world_size
    )[rank]

    print("Loading item->item edge index...")
    data["item", "to", "item"].edge_index = torch.load(
        os.path.join(edge_path, "item_item", f"rank={rank}.pt"),
        weights_only=True,
    )
    data["item", "rev_to", "item"].edge_index = torch.stack([
        data["item", "to", "item"].edge_index[1],
        data["item", "to", "item"].edge_index[0],
    ])

    print("Loading user->item edge index...")
    data["user", "to", "item"].edge_index = torch.load(
        os.path.join(edge_path, "user_item", f"rank={rank}.pt"),
        weights_only=True,
    )
    data["item", "rev_to", "user"].edge_index = torch.stack([
        data["user", "to", "item"].edge_index[1],
        data["user", "to", "item"].edge_index[0],
    ])

    print("Splitting data...")
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=[("user", "to", "item")],
        rev_edge_types=[("item", "rev_to", "user")],
    )(data)

    _dbg(f"train eli shape={train_data['user','to','item'].edge_label_index.shape}", tag="LOAD")
    print(train_data, test_data, val_data)
    print(f"Finished loading graph data on rank {rank}")

    return {
        "train": cugraph_pyg_from_heterodata(train_data, return_edge_label=False),
        "test":  cugraph_pyg_from_heterodata(test_data),
        "val":   cugraph_pyg_from_heterodata(val_data),
    }, meta


# ──────────────────────────────────────────────────────────────────────────────
# 核心修复（鲁迅拿法改写 #2 + #3：EdgeShuffler 封装 BUG fix）
#
# 上游 f2b7f50 修复了单卡路径 IndexError，Walpurgis 迁移进一步将:
#   - world_size==1 的 early-return 提取为明确的 _single_gpu_path()
#   - start/end 计算提取为 _compute_local_slice()，加断点调试 print
#   - 三段 all_to_all 操作提取为 _scatter_gather_tensor()
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeShuffler:
    """
    封装 balance_shuffle_edge_split 的分布式 shuffle 逻辑。

    BUG 根因（f2b7f50 修复）:
        旧代码 else 分支（rank == world_size-1）隐含了 world_size >= 2 的前提。
        单卡时 world_size==1，rank==0 触发 elif rank==0 分支，
        访问 edge_offsets[0]，但 edge_offsets = cumsum[:-1] 为空 tensor，
        IndexError 确定性触发。
        新代码提前检查 world_size==1，直接使用全量 dst_rank，跳过 edge_offsets 访问。

    改写点（vs 上游 f2b7f50）:
        - world_size==1 early-return  →  _single_gpu_path() 方法，名称说明意图
        - start/end 二行  →  _compute_local_slice() 静态方法，加断点调试
        - 三段 all_to_all 散落  →  _scatter_gather_tensor() 方法
    """

    @staticmethod
    def _compute_local_slice(
        rank: int,
        world_size: int,
        edge_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算当前 rank 在全量 dst_rank tensor 中对应的切片索引。

        上游 f2b7f50 修复后的逻辑:
            start = 0 if rank == 0 else edge_offsets[rank - 1]
            end   = edge_offsets[rank] if rank < world_size - 1 else None
            local_rank_t = dst_rank[start:end]

        edge_offsets = cumsum(num_edges)[:-1]，长度 = world_size - 1，
        含义是每个 rank 的边在全局拼接后的累积偏移量（不含最后一个 rank 的尾部）。

        断点调试: 打印 rank / start / end，确认各 rank 切片不重叠、不遗漏。
        """
        start: int = 0 if rank == 0 else int(edge_offsets[rank - 1].item())
        end: Optional[int] = (
            int(edge_offsets[rank].item()) if rank < world_size - 1 else None
        )
        _dbg(
            f"_compute_local_slice: rank={rank} world_size={world_size} "
            f"start={start} end={end} "
            f"edge_offsets={edge_offsets.tolist()}",
            tag="SHUFFLE",
        )
        return start, end

    @staticmethod
    def _scatter_gather_tensor(
        tensor: torch.Tensor,
        local_rank_t: torch.Tensor,
        r_counts: torch.Tensor,
        world_size: int,
    ) -> torch.Tensor:
        """
        按 local_rank_t 分桶后执行 all_to_all，返回本 rank 收到的拼接 tensor。
        上游三段重复的 all_to_all 代码块，此处封装为可复用方法。
        """
        s = [tensor.cuda()[local_rank_t == r] for r in range(world_size)]
        rx = [
            torch.empty((int(ln.item()),), device="cuda", dtype=tensor.dtype)
            for ln in r_counts
        ]
        torch.distributed.all_to_all(rx, s)
        return torch.concat(rx).cpu()

    def split(
        self,
        edge_label_index: torch.Tensor,
        edge_label: Optional[torch.Tensor],
    ):
        """
        对应上游 balance_shuffle_edge_split。
        按 world_size 将 edge_label_index / edge_label 均衡随机 shuffle 到各 rank。

        返回: (edge_label_index, edge_label)  — 本 rank 分得的部分
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        _dbg(
            f"EdgeShuffler.split: rank={rank} world_size={world_size} "
            f"local_num_edges={edge_label_index.shape[1]}",
            tag="SHUFFLE",
        )

        # ── 收集各 rank 边数 ──────────────────────────────────────────────────
        local_num_edges = torch.tensor(
            [int(edge_label_index.shape[1])], device="cuda", dtype=torch.int64
        )
        num_edges = torch.empty((world_size,), dtype=torch.int64, device="cuda")
        torch.distributed.all_gather_into_tensor(num_edges, local_num_edges)
        total_num_edges = num_edges.sum()

        _dbg(
            f"num_edges per rank = {num_edges.tolist()} "
            f"total = {total_num_edges.item()}",
            tag="SHUFFLE",
        )

        # ── 只有 rank 0 生成随机置换，然后广播 ──────────────────────────────
        if rank == 0:
            dst_rank = (
                torch.randperm(total_num_edges, device="cuda", dtype=torch.int64)
                % world_size
            )
            _dbg(
                f"dst_rank generated (rank 0): "
                f"unique counts = { {r: int((dst_rank==r).sum()) for r in range(world_size)} }",
                tag="SHUFFLE",
            )
        else:
            dst_rank = torch.empty(
                (total_num_edges,), device="cuda", dtype=torch.int64
            )

        torch.distributed.broadcast(dst_rank, src=0)
        _dbg("broadcast dst_rank done", tag="SHUFFLE")

        # ── f2b7f50 核心修复：单卡 early-return ─────────────────────────────
        #
        #   旧代码直接进入三段 if-elif-else，单卡时触发:
        #     edge_offsets = cumsum(num_edges)[:-1]  →  空 tensor（world_size==1）
        #     elif rank==0: dst_rank[0 : edge_offsets[0]]  →  IndexError
        #
        #   新代码: world_size==1 时 dst_rank 就是全量，无需切片
        #
        if world_size == 1:
            # 单卡路径：全量即为本 rank 的份额，all_to_all 退化为自环
            _dbg("world_size==1: single GPU path, skip edge_offsets slicing", tag="SHUFFLE")
            local_rank_t = dst_rank
        else:
            # 多卡路径：计算本 rank 对应的 dst_rank 切片
            edge_offsets = num_edges.cumsum(0).cpu()[:-1]
            start, end = self._compute_local_slice(rank, world_size, edge_offsets)
            local_rank_t = dst_rank[start:end]

        _dbg(f"local_rank_t shape = {local_rank_t.shape}", tag="SHUFFLE")

        # ── 收集各 rank 发送计数（s_counts），转置得接收计数（r_counts）──────
        s_counts = torch.tensor(
            [(local_rank_t == r).sum().item() for r in range(world_size)],
            device="cuda",
            dtype=torch.int64,
        )
        s_counts_global = torch.empty(
            (world_size, world_size), device="cuda", dtype=torch.int64
        )
        torch.distributed.all_gather_into_tensor(s_counts_global, s_counts)
        r_counts = s_counts_global[:, rank]

        _dbg(
            f"s_counts = {s_counts.tolist()} r_counts = {r_counts.tolist()}",
            tag="SHUFFLE",
        )

        # ── all_to_all: edge_label_index[0], [1], edge_label ────────────────
        edge_label_index[0] = self._scatter_gather_tensor(
            edge_label_index[0], local_rank_t, r_counts, world_size
        )
        edge_label_index[1] = self._scatter_gather_tensor(
            edge_label_index[1], local_rank_t, r_counts, world_size
        )

        if edge_label is not None:
            edge_label = self._scatter_gather_tensor(
                edge_label, local_rank_t, r_counts, world_size
            )

        _dbg(
            f"EdgeShuffler.split done: "
            f"edge_label_index.shape = {edge_label_index.shape}",
            tag="SHUFFLE",
        )
        return edge_label_index, edge_label


# 模块级 shuffler 单例（避免每次创建 loader 都构造新对象）
_edge_shuffler = EdgeShuffler()


def balance_shuffle_edge_split(edge_label_index, edge_label):
    """
    公共接口，与上游函数签名兼容。
    内部委托给 EdgeShuffler.split()。
    """
    return _edge_shuffler.split(edge_label_index, edge_label)


# ──────────────────────────────────────────────────────────────────────────────
# 训练 / 评估（与上游一致，加 _dbg 断点）
# ──────────────────────────────────────────────────────────────────────────────

def train(model, optimizer, loader, max_iter=None):
    start_time = perf_counter()
    rank = torch.distributed.get_rank()
    model.train()

    total_loss = total_examples = 0
    for i, batch in enumerate(loader):
        if max_iter is not None and i >= max_iter:
            break

        batch = batch.cuda()
        optimizer.zero_grad()

        if i % 10 == 0 and rank == 0:
            curr_time = perf_counter()
            print(f"iter {i}, {curr_time - start_time:.4f} sec elapsed.")
            _dbg(
                f"train iter {i}: "
                f"edge_label shape = {batch['user','item'].edge_label.shape}",
                tag="TRAIN",
            )

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "item"].edge_label_index,
        )
        loss = F.binary_cross_entropy_with_logits(
            pred, batch["user", "item"].edge_label
        )

        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        total_examples += pred.numel()

    if total_examples == 0:
        # 防御：loader 为空时避免 ZeroDivisionError
        _dbg("WARNING: train loader was empty, returning loss=0.0", tag="TRAIN")
        return 0.0

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
    model.eval()
    preds, targets = [], []
    for i, batch in enumerate(loader):
        batch = batch.cuda()
        pred = (
            model(
                batch.x_dict,
                batch.edge_index_dict,
                batch["user", "item"].edge_label_index,
            )
            .sigmoid()
            .view(-1)
            .cpu()
        )
        target = batch["user", "item"].edge_label.long().cpu()
        _dbg(
            f"test batch {i}: pred.shape={pred.shape} "
            f"target.unique={target.unique().tolist()}",
            tag="TEST",
        )
        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    return roc_auc_score(target, pred)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn(
            f"This script ({__file__}) should be run with 'torchrun'.  Exiting."
        )
        exit()
    if os.getenv("CI", "false").lower() == "true":
        warnings.warn(f"Skipping example {__file__} in CI due to memory limit")
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--skip_partition", action="store_true")
    args = parser.parse_args()

    dataset_name = "taobao"

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600))
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    _dbg(
        f"__main__ start: world_size={world_size} "
        f"global_rank={global_rank} local_rank={local_rank}",
        tag="MAIN",
    )

    if global_rank == 0:
        from rmm.allocators.torch import rmm_torch_allocator
        torch.cuda.change_current_allocator(rmm_torch_allocator)

    if global_rank == 0:
        from pylibcugraph.comms import cugraph_comms_create_unique_id
        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    edge_path  = os.path.join(args.dataset_root, dataset_name + "_eix_part")
    meta_path  = os.path.join(args.dataset_root, dataset_name + "_meta.json")

    if not args.skip_partition and global_rank == 0:
        print("Partitioning data...")
        dataset = Taobao(args.dataset_root, pre_transform=pre_transform)
        data = dataset[0]
        preprocessor = DataPreprocessor()
        preprocessor(data, edge_path=edge_path, meta_path=meta_path)
        print("Data partitioning complete!")

    _dbg("barrier: after partition", tag="MAIN")
    torch.distributed.barrier()

    data_dict, meta = load_partitions(edge_path, meta_path)

    _dbg("barrier: after load_partitions", tag="MAIN")
    torch.distributed.barrier()

    from cugraph_pyg.loader import LinkNeighborLoader

    def create_loader(data_l):
        edge_label_index = data_l[1]
        edge_label = data_l[2]

        _dbg(
            f"create_loader: eli.shape={edge_label_index.shape} "
            f"edge_label={'None' if edge_label is None else edge_label.shape}",
            tag="MAIN",
        )

        edge_label_index, edge_label = balance_shuffle_edge_split(
            edge_label_index, edge_label
        )

        return LinkNeighborLoader(
            data=data_l[0],
            edge_label_index=(("user", "to", "item"), edge_label_index),
            edge_label=edge_label,
            neg_sampling="binary" if edge_label is None else None,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_neighbors={
                ("user", "to", "item"):   [8, 4],
                ("item", "rev_to", "user"): [8, 4],
                ("item", "to", "item"):   [8, 4],
                ("item", "rev_to", "item"): [8, 4],
            },
            local_seeds_per_call=16384,
        )

    print("Creating train loader...")
    train_loader = create_loader(data_dict["train"])
    print(f"Created train loader on rank {global_rank}")

    torch.distributed.barrier()

    print("Creating validation loader...")
    val_loader = create_loader(data_dict["val"])
    print(f"Created validation loader on rank {global_rank}")

    torch.distributed.barrier()

    model = Model(
        num_users=meta["num_nodes"]["user"],
        num_items=meta["num_nodes"]["item"],
        hidden_channels=64,
        out_channels=64,
    ).to(local_rank)
    print(f"Created model on rank {global_rank}")

    init_start = perf_counter()
    # Initialize lazy modules
    # FIXME DO NOT DO THIS!!!!  Use set parameters
    for batch in train_loader:
        batch = batch.to(local_rank)
        _ = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "item"].edge_label_index,
        )
        break
    init_end = perf_counter()
    print(
        f"Initialized model on rank {global_rank}, "
        f"took {init_end - init_start:.4f} seconds."
    )

    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_start = perf_counter()
    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_start = perf_counter()
        print("Train")
        loss = train(model, optimizer, train_loader, args.max_iter)
        train_end = perf_counter()

        if global_rank == 0:
            print("Val")

        torch.cuda.synchronize()

        val_start = perf_counter()
        val_auc = test(model, val_loader)
        best_val_auc = max(best_val_auc, val_auc)
        val_end = perf_counter()

        if global_rank == 0:
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f},"
                f" Train time: {train_end - train_start:.4f} s, "
                f"Val time: {val_end - val_start:.4f} s"
            )
        _dbg(
            f"epoch {epoch} done: loss={loss:.4f} val_auc={val_auc:.4f}",
            tag="MAIN",
        )

    training_end = perf_counter()
    print(f"Training complete in {training_end - training_start:.4f} seconds.")

    del train_loader
    del val_loader
    gc.collect()
    print("Creating test loader...")
    test_loader = create_loader(data_dict["test"])

    if global_rank == 0:
        print("Test")
    test_auc = test(model, test_loader)
    print(
        f"Total {args.epochs:02d} epochs: Final Loss: {loss:.4f}, "
        f"Best Val AUC: {best_val_auc:.4f}, "
        f"Test AUC: {test_auc:.4f}"
    )

    wm_finalize()
    from pylibcugraph.comms import cugraph_comms_shutdown
    cugraph_comms_shutdown()
