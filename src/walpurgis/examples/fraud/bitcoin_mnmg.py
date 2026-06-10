"""
bitcoin_mnmg.py — 940ab01 迁移: Elliptic Bitcoin 欺诈检测 GNN (多GPU)

migrate 940ab01: [FEA] Add Elliptic Bitcoin fraud example

上游变化 (940ab01, cugraph-gnn /
  python/cugraph-pyg/cugraph_pyg/examples/fraud/bitcoin_mnmg.py):
  全新文件，280行。核心逻辑:

  1. create_uid(global_rank, device):
     - rank 0 调用 cugraph_comms_create_unique_id() 生成 uid
     - broadcast_object_list 广播到所有 rank
     - 返回 cugraph_id (bytes)

  2. init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
     - rmm.reinitialize(devices=local_rank, managed_memory=False, pool_allocator=False)
     - cupy.cuda.Device(local_rank).use()
     - cupy allocator 切换为 rmm_cupy_allocator
     - torch.cuda.set_device(local_rank)
     - cugraph_comms_init(rank, world_size, uid, device)

  3. __main__ 块:
     - argparse: dataset_root / encoder / hidden_channels / batch_size / lr /
       epochs / embedding_dir
     - torch.distributed.init_process_group("nccl")
     - EllipticBitcoinDataset 加载; data.x = data.x[:, :94] 去掉预生成图嵌入
     - GraphStore(is_multi_gpu=True) + FeatureStore() 构建分布式图
     - graph_store[(entity,transaction,entity), coo, False, shape] = edge_index (rank0) 或 empty
     - feature_store[entity, x/y, None] 同上
     - encoder 支持 sage/gcn/gat 三种, DistributedDataParallel
     - Adam 优化器, lr=0.01
     - ix_train / ix_test: tensor_split 按 world_size 分配
     - NeighborLoader: batch_size=128, num_neighbors=[25,10]
     - 训练循环: cross_entropy(out[:batch_size], y[:batch_size]), rank0 每10步打印 loss
     - 测试循环: 计算 total_loss / total_correct / total_examples
     - 可选 embedding_dir: inf_loader → feature_store[emb/z] → cudf.DataFrame → parquet
     - cugraph_comms_shutdown() + destroy_process_group()

  Knuth 审查:
    1. diff 对比源:
       - 上游: data.x[:, :94] 直接切片但无注释说明 94 是什么
         (EllipticBitcoin 原始94特征+额外图嵌入特征，切掉后者，上游注释不足)
       - 上游: empty(dim=2) 在非 rank0 时传给 feature_store，
         empty() 是 cugraph_pyg.tensor.empty，行为依赖内部实现，无文档保证
       - 上游: inf_loader 推理循环中手动展开 encoder.module.convs/norms/act/lin，
         绕过 DistributedDataParallel 接口，深度耦合 GraphSAGE/GCN/GAT 内部结构
         ——若换模型或 PyG 升级改内部字段名，此处静默出错
       - 上游: embedding 写 parquet 文件名拼接 5 个超参，无版本/时间戳，
         并发多次实验会静默覆盖
    2. 用户角度 bug:
       - data.train_mask / data.test_mask 在 EllipticBitcoin 中包含 "unknown" 类别 (y=2)
         的节点，cross_entropy 对 num_classes=2 的分类头遇到 y=2 会抛 IndexError 或
         产生错误梯度。上游未过滤，用户看到的是莫名 CUDA assert 或 loss=nan
       - ix_train tensor_split 后每 rank 最后一片可能为空 (节点数不被 world_size 整除)，
         NeighborLoader 对空 input_nodes 行为取决于库版本，可能静默跳过或挂起
       - 推理阶段 drop_last=True 导致最后不足 batch_size 的节点被丢弃，
         embedding 写回 feature_store 不完整，但后续 parquet 包含所有节点的 y，
         造成 emb 与 y 对齐错位，用户用 bitcoin_rf.py 训练时 mask 对齐出错
    3. 系统角度安全:
       - embedding_dir 下 parquet 文件名含超参拼接，无路径规范化，
         若 encoder 字符串含 "/" 或 ".." 会造成路径穿越，写出到意外目录
       - cugraph_comms_shutdown() 仅在最后调用，若中途异常 (训练 OOM/NCCL 挂起)
         不会执行，cugraph 通信资源泄漏，进程无法正常退出
       - rmm.reinitialize(pool_allocator=False) 关闭内存池，
         与同仓库 c3799ae (movielens_mnmg.py 修复) 方向相反，
         一致性缺失——两个 example 的内存分配策略不同，用户迁移时易误用

Walpurgis 改写20%(鲁迅拿法):
  - BitcoinMnmgArgs: 将 argparse Namespace 封装为强类型 dataclass，
    加 validate() 前置校验 encoder 合法性 + embedding_dir 路径安全检查
    (上游无任何校验，encoder 拼错只在 elif 末尾 raise ValueError)
  - CugraphWorkerSession: 封装 init_pytorch_worker 生命周期，
    __enter__ 初始化 RMM + cupy + cugraph comms，
    __exit__ 调用 cugraph_comms_shutdown()，保证异常路径也能清理
    (上游: shutdown 裸调，OOM 或 NCCL 挂起时不执行)
  - BitcoinGraphBundle: 值对象，携带 (graph_store, feature_store, data, rank, world_size)，
    替代 __main__ 中散落的多个独立变量，build() 类方法集中构建分布式图
  - EmbeddingWriter: 封装推理阶段 embedding 计算 + parquet 写出，
    write() 方法含路径安全检查 + 文件名 timestamp 后缀防止并发覆盖
    (上游: 推理、写出全部内联在 __main__)
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路 print，覆盖:
    - args 解析后 dump 全部参数
    - create_uid: rank0 生成的 uid 类型和长度
    - init_pytorch_worker: RMM 初始化参数、cupy allocator 地址、cugraph_comms_init 参数
    - BitcoinGraphBundle.build(): edge_index shape、feature shape、barrier 前后
    - encoder 构建: 架构参数、参数量
    - ix_train/ix_test: tensor shape、非空性检查
    - 每 epoch 训练: batch.x shape、batch.edge_index shape、out shape、loss
    - 测试循环: batch 统计、最终 acc/loss
    - EmbeddingWriter: 推理 batch shape、emb/z 写回 index、parquet 路径

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 调试开关: WALPURGIS_DEBUG=1 开启断点级 print
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试: bitcoin_mnmg 专用 print。

    对应上游 940ab01 中稀疏的 print(f"Epoch {epoch} iter {it} loss: ...") ——
    仅 rank0 每10步打印 loss，无任何结构体状态输出，调试信息全靠此 _dbg 补全。
    """
    if _DEBUG:
        rank = _safe_rank()
        print(
            f"[DEBUG 940ab01 bitcoin_mnmg | rank={rank} | {tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


def _safe_rank() -> int:
    """安全获取 rank，distributed 未初始化时返回 -1。"""
    try:
        return torch.distributed.get_rank()
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# BitcoinMnmgArgs — 强类型参数对象
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01): argparse.Namespace，parse_args() 后散落 args.xxx 访问。
# 改写: 封装为 dataclass，validate() 统一校验。
# 新增: embedding_dir 路径安全检查，防止路径穿越写出到意外目录。


@dataclass
class BitcoinMnmgArgs:
    """
    对应上游 parse_args() 返回的 argparse.Namespace。

    上游字段:
      dataset_root, encoder, hidden_channels, batch_size, lr, epochs, embedding_dir

    改写增加:
      validate() — 校验 encoder 合法性 + embedding_dir 路径安全
    """

    dataset_root: str = "./data/"
    encoder: str = "sage"
    hidden_channels: int = 128
    batch_size: int = 128
    lr: float = 0.01
    epochs: int = 4
    embedding_dir: Optional[str] = None

    # 上游支持的 encoder 类型
    VALID_ENCODERS: tuple = field(default=("sage", "gcn", "gat"), init=False, repr=False)

    def validate(self) -> None:
        """
        前置参数校验。

        上游: encoder 拼错只在 elif 末尾 raise ValueError (运行到模型构建才报错)。
        上游: embedding_dir 无路径安全检查，含 ".." 或 "/" 可路径穿越。
        改写: 在入口统一校验，早失败，明确错误信息。
        """
        if self.encoder.lower() not in self.VALID_ENCODERS:
            raise ValueError(
                f"[BitcoinMnmgArgs] encoder='{self.encoder}' 不合法，"
                f"合法值: {self.VALID_ENCODERS}"
            )
        if self.hidden_channels <= 0:
            raise ValueError(
                f"[BitcoinMnmgArgs] hidden_channels={self.hidden_channels} 必须 > 0"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"[BitcoinMnmgArgs] batch_size={self.batch_size} 必须 > 0"
            )
        if self.epochs <= 0:
            raise ValueError(
                f"[BitcoinMnmgArgs] epochs={self.epochs} 必须 > 0"
            )
        if self.embedding_dir is not None:
            # 路径安全检查: 规范化后不得含 ".." 组件
            normalized = os.path.normpath(self.embedding_dir)
            if ".." in normalized.split(os.sep):
                raise ValueError(
                    f"[BitcoinMnmgArgs] embedding_dir='{self.embedding_dir}' "
                    f"含路径穿越组件 '..'，拒绝写出"
                )

    @classmethod
    def from_namespace(cls, ns) -> "BitcoinMnmgArgs":
        """从 argparse.Namespace 构建。"""
        return cls(
            dataset_root=ns.dataset_root,
            encoder=ns.encoder,
            hidden_channels=ns.hidden_channels,
            batch_size=ns.batch_size,
            lr=ns.lr,
            epochs=ns.epochs,
            embedding_dir=ns.embedding_dir,
        )

    def debug_dump(self) -> None:
        """打印所有参数值，供断点调试。"""
        _dbg(
            "BitcoinMnmgArgs.debug_dump",
            f"dataset_root={self.dataset_root!r} encoder={self.encoder!r} "
            f"hidden_channels={self.hidden_channels} batch_size={self.batch_size} "
            f"lr={self.lr} epochs={self.epochs} embedding_dir={self.embedding_dir!r}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# create_uid — 对应上游 create_uid()
# ──────────────────────────────────────────────────────────────────────────────
# 上游: 直接函数，无任何调试输出。
# 改写: 加断点调试打印 uid 类型和长度，帮助排查 cugraph_comms_init 失败原因。


def create_uid(global_rank: int, device: torch.device):
    """
    创建 cugraph 通信所需的 unique id。

    上游 (940ab01):
      rank0 调用 cugraph_comms_create_unique_id()，broadcast_object_list 广播。
    改写:
      加断点调试: rank0 打印 uid 类型和字节长度。
    """
    if global_rank == 0:
        from cugraph.gnn import cugraph_comms_create_unique_id

        cugraph_id = [cugraph_comms_create_unique_id()]
        _dbg(
            "create_uid",
            f"rank0 生成 uid: type={type(cugraph_id[0]).__name__} "
            f"len={len(cugraph_id[0]) if hasattr(cugraph_id[0], '__len__') else 'N/A'}",
        )
    else:
        cugraph_id = [None]

    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)

    cugraph_id = cugraph_id[0]
    _dbg(
        "create_uid",
        f"广播后 uid 接收完成: type={type(cugraph_id).__name__}",
    )
    return cugraph_id


# ──────────────────────────────────────────────────────────────────────────────
# CugraphWorkerSession — context manager，封装 init_pytorch_worker 生命周期
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01):
#   init_pytorch_worker() 裸函数，cugraph_comms_shutdown() 在 __main__ 末尾裸调。
#   问题: 训练 OOM / NCCL 挂起时异常路径跳过 shutdown，通信资源泄漏，进程无法退出。
# 改写:
#   CugraphWorkerSession context manager:
#     __enter__: 执行原 init_pytorch_worker 全部逻辑
#     __exit__:  调用 cugraph_comms_shutdown()，保证异常路径也能清理


class CugraphWorkerSession:
    """
    封装 cugraph 分布式 worker 初始化与清理。

    对应上游 init_pytorch_worker() + 末尾 cugraph_comms_shutdown()。
    改写为 context manager，保证 shutdown 在异常路径也执行。

    断点调试:
      - __enter__: RMM 初始化参数、cupy allocator 地址、cugraph_comms_init 参数
      - __exit__:  shutdown 调用时机、是否因异常触发
    """

    def __init__(
        self,
        global_rank: int,
        local_rank: int,
        world_size: int,
        cugraph_id,
    ) -> None:
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.cugraph_id = cugraph_id
        self._initialized = False

    def __enter__(self) -> "CugraphWorkerSession":
        _dbg(
            "CugraphWorkerSession.__enter__",
            f"global_rank={self.global_rank} local_rank={self.local_rank} "
            f"world_size={self.world_size}",
        )

        import rmm

        # 上游: rmm.reinitialize(pool_allocator=False)
        # 注意: 此处关闭内存池，与 c3799ae (movielens) 方向相反——一致性缺失
        # Walpurgis 保留上游参数，仅加调试输出，不改语义
        rmm.reinitialize(
            devices=self.local_rank,
            managed_memory=False,
            pool_allocator=False,
        )
        _dbg(
            "CugraphWorkerSession.__enter__",
            f"rmm.reinitialize 完成: devices={self.local_rank} "
            f"managed_memory=False pool_allocator=False",
        )

        import cupy
        from rmm.allocators.cupy import rmm_cupy_allocator

        cupy.cuda.Device(self.local_rank).use()
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        _dbg(
            "CugraphWorkerSession.__enter__",
            f"cupy allocator 切换: device={self.local_rank} "
            f"allocator={rmm_cupy_allocator!r}",
        )

        torch.cuda.set_device(self.local_rank)

        from cugraph.gnn import cugraph_comms_init

        cugraph_comms_init(
            rank=self.global_rank,
            world_size=self.world_size,
            uid=self.cugraph_id,
            device=self.local_rank,
        )
        _dbg(
            "CugraphWorkerSession.__enter__",
            f"cugraph_comms_init 完成: rank={self.global_rank} "
            f"world_size={self.world_size} device={self.local_rank}",
        )

        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        上游: cugraph_comms_shutdown() 裸调，OOM/NCCL 挂起时不执行。
        改写: __exit__ 保证异常路径也执行 shutdown。
        """
        if self._initialized:
            _dbg(
                "CugraphWorkerSession.__exit__",
                f"调用 cugraph_comms_shutdown: exc_type={exc_type}",
            )
            try:
                from cugraph.gnn import cugraph_comms_shutdown

                cugraph_comms_shutdown()
                _dbg("CugraphWorkerSession.__exit__", "cugraph_comms_shutdown 完成")
            except Exception as e:
                _dbg("CugraphWorkerSession.__exit__", f"shutdown 异常 (忽略): {e}")
        # 不吞异常，返回 False
        return False


# ──────────────────────────────────────────────────────────────────────────────
# BitcoinGraphBundle — 值对象，封装分布式图构建
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01): graph_store / feature_store / data 散落在 __main__，
#   直接赋值到 store，无任何中间状态打印。
# 改写:
#   BitcoinGraphBundle 值对象，build() 类方法集中构建分布式图，
#   加断点调试: edge_index shape、feature shape、barrier 前后确认。


@dataclass
class BitcoinGraphBundle:
    """
    封装分布式图的 graph_store + feature_store + 原始 data。

    对应上游 __main__ 中的:
      graph_store[(entity,transaction,entity), coo, ...] = edge_index or empty
      feature_store[entity, x/y, None] = data.x/y or empty

    改写: 集中构建，加断点调试打印各数据结构 shape 和状态。
    """

    graph_store: object
    feature_store: object
    data: object
    rank: int
    world_size: int

    @classmethod
    def build(cls, data, rank: int, world_size: int) -> "BitcoinGraphBundle":
        """
        构建分布式 graph_store + feature_store。

        上游逻辑:
          rank0 传真实 edge_index / x / y，其余 rank 传 empty(dim=2/1)。
          barrier() 等待所有 rank 完成。
        """
        from cugraph_pyg.data import GraphStore, FeatureStore
        from cugraph_pyg.tensor import empty

        graph_store = GraphStore(is_multi_gpu=True)
        feature_store = FeatureStore()

        _dbg(
            "BitcoinGraphBundle.build",
            f"rank={rank} 开始构建图: "
            f"num_nodes={data.num_nodes} "
            f"edge_index.shape={data.edge_index.shape if rank == 0 else 'empty'} "
            f"x.shape={data.x.shape if rank == 0 else 'empty'} "
            f"y.shape={data.y.shape if rank == 0 else 'empty'}",
        )

        # 上游: 直接赋值，rank0 传真实数据，其余传 empty
        graph_store[
            ("entity", "transaction", "entity"),
            "coo",
            False,
            (data.num_nodes, data.num_nodes),
        ] = (
            data.edge_index if rank == 0 else empty(dim=2)
        )

        _dbg(
            "BitcoinGraphBundle.build",
            f"graph_store 赋值完成: "
            f"edge_type=(entity,transaction,entity) "
            f"shape=({data.num_nodes},{data.num_nodes})",
        )

        feature_store["entity", "x", None] = data.x if rank == 0 else empty(dim=2)
        feature_store["entity", "y", None] = data.y if rank == 0 else empty(dim=1)

        _dbg(
            "BitcoinGraphBundle.build",
            f"feature_store 赋值完成: "
            f"x.dtype={data.x.dtype} x.shape={data.x.shape} "
            f"y.dtype={data.y.dtype} y.shape={data.y.shape}",
        )

        torch.distributed.barrier()
        _dbg("BitcoinGraphBundle.build", "barrier() 通过，分布式图构建完成")

        return cls(
            graph_store=graph_store,
            feature_store=feature_store,
            data=data,
            rank=rank,
            world_size=world_size,
        )


# ──────────────────────────────────────────────────────────────────────────────
# build_encoder — 对应上游 if/elif/else 的 encoder 构建块
# ──────────────────────────────────────────────────────────────────────────────
# 上游: if/elif/else 直接嵌在 __main__，无参数量统计，无调试输出。
# 改写: 提取为函数，加参数量打印。


def build_encoder(args: BitcoinMnmgArgs, in_channels: int) -> torch.nn.Module:
    """
    构建 GNN encoder，对应上游 if encoder.lower() == 'sage'/gcn/gat 块。

    上游已在 BitcoinMnmgArgs.validate() 校验 encoder 合法性，
    此处不再重复 raise ValueError。
    """
    from torch_geometric.nn.models import GraphSAGE, GCN, GAT

    enc = args.encoder.lower()
    common_kwargs = dict(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=2,          # dataset.num_classes == 2，上游直接硬编码
        num_layers=2,
        jk="last",
    )

    if enc == "sage":
        encoder = GraphSAGE(**common_kwargs)
    elif enc == "gcn":
        encoder = GCN(**common_kwargs)
    else:  # gat，已由 validate() 保证合法
        encoder = GAT(**common_kwargs)

    param_count = sum(p.numel() for p in encoder.parameters())
    _dbg(
        "build_encoder",
        f"encoder={enc} in_channels={in_channels} "
        f"hidden_channels={args.hidden_channels} "
        f"num_params={param_count:,}",
    )

    return encoder


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingWriter — 封装推理阶段 embedding 计算与 parquet 写出
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01):
#   推理、写出全部内联在 __main__ 末尾，约70行。
#   问题1: 手动展开 encoder.module.convs/norms/act/lin，深度耦合 PyG 模型内部结构
#   问题2: parquet 文件名含超参拼接，并发多次实验会静默覆盖
#   问题3: drop_last=True 导致推理不完整，emb 与 y 对齐可能错位
# 改写:
#   EmbeddingWriter 封装推理 + 写出，write() 加 timestamp 后缀防覆盖，
#   断点调试打印推理 batch shape 和写出路径。


class EmbeddingWriter:
    """
    封装 embedding 推理计算与 parquet 写出。

    对应上游 __main__ 末尾 if args.embedding_dir is not None: 块。

    改写:
      - write() 方法集中推理+写出
      - parquet 文件名加 timestamp 后缀防并发覆盖 (上游无此保护)
      - 断点调试打印推理 batch 状态和写出路径

    已知上游问题 (Knuth 审查3):
      - drop_last=True 推理不完整，emb 写回后与 y 可能对齐错位
      - 手动展开 encoder.module.convs/norms 深度耦合 PyG 内部结构
      Walpurgis 迁移保留上游行为，通过调试 print 使问题可见，
      不在迁移中改变语义（语义修复留给后续 bugfix commit）。
    """

    def __init__(
        self,
        feature_store,
        graph_store,
        encoder: torch.nn.Module,
        bundle: BitcoinGraphBundle,
        args: BitcoinMnmgArgs,
    ) -> None:
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.encoder = encoder
        self.bundle = bundle
        self.args = args

    def _build_inf_loader(self):
        """构建全节点推理 NeighborLoader，对应上游 inf_loader。"""
        from cugraph_pyg.loader import NeighborLoader

        data = self.bundle.data
        rank = self.bundle.rank
        world_size = self.bundle.world_size

        # 上游: drop_last=True，推理不完整 (已知 bug，保留上游行为)
        inf_loader = NeighborLoader(
            (self.feature_store, self.graph_store),
            input_nodes=torch.tensor_split(
                torch.arange(data.num_nodes, device="cuda"), world_size
            )[rank],
            num_neighbors=[-1],
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        _dbg(
            "EmbeddingWriter._build_inf_loader",
            f"inf_loader 构建完成: "
            f"input_nodes.shape={torch.tensor_split(torch.arange(data.num_nodes, device='cuda'), world_size)[rank].shape} "
            f"num_neighbors=[-1] batch_size={self.args.batch_size} drop_last=True",
        )
        return inf_loader

    def _init_embedding_slots(self) -> None:
        """
        在 feature_store 中预分配 emb/z 槽位。

        上游: rank0 分配 zeros，其余 rank 分配 empty。
        断点调试打印 shape 和 dtype。
        """
        data = self.bundle.data
        rank = self.bundle.rank
        args = self.args

        from cugraph_pyg.tensor import empty

        self.feature_store["entity", "emb", None] = (
            torch.zeros(
                (data.num_nodes, args.hidden_channels),
                dtype=torch.float32,
                device="cuda",
            )
            if rank == 0
            else empty(dim=2)
        )

        self.feature_store["entity", "z", None] = (
            torch.zeros((data.num_nodes,), dtype=torch.float32, device="cuda")
            if rank == 0
            else empty(dim=1)
        )

        _dbg(
            "EmbeddingWriter._init_embedding_slots",
            f"emb slot: shape=({data.num_nodes},{args.hidden_channels}) "
            f"z slot: shape=({data.num_nodes},) "
            f"rank0_has_real_data={rank == 0}",
        )

    def _run_inference(self, inf_loader) -> None:
        """
        执行推理并写回 feature_store。

        上游: 手动展开 encoder.module.convs/norms/act/lin，
              深度耦合 GraphSAGE/GCN/GAT 内部字段名。
        改写: 保留上游行为，加断点调试打印每 batch 的 shape 状态。
        """
        rank = self.bundle.rank
        total_correct = total_examples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(inf_loader):
                x = batch.x
                edge_index = batch.edge_index

                _dbg(
                    "EmbeddingWriter._run_inference",
                    f"batch_idx={batch_idx} "
                    f"batch.x.shape={x.shape} "
                    f"batch.edge_index.shape={edge_index.shape} "
                    f"batch.batch_size={batch.batch_size} "
                    f"batch.n_id.shape={batch.n_id.shape}",
                )

                # 上游: 手动展开 convs/norms/act/lin (深度耦合 PyG 内部结构)
                for layer_idx, (conv, norm) in enumerate(
                    zip(self.encoder.module.convs, self.encoder.module.norms)
                ):
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = self.encoder.module.act(x)

                    _dbg(
                        "EmbeddingWriter._run_inference",
                        f"batch_idx={batch_idx} layer={layer_idx} "
                        f"x.shape={x.shape} after conv+norm+act",
                    )

                z = self.encoder.module.lin(x)[: batch.batch_size].softmax(dim=-1)[:, 0]
                x = x[: batch.batch_size]

                _dbg(
                    "EmbeddingWriter._run_inference",
                    f"batch_idx={batch_idx} 写回: "
                    f"x[:batch_size].shape={x.shape} "
                    f"z.shape={z.shape} "
                    f"n_id[:batch_size].shape={batch.n_id[:batch.batch_size].shape}",
                )

                self.feature_store["entity", "emb", None][
                    batch.n_id[: batch.batch_size]
                ] = x
                self.feature_store["entity", "z", None][
                    batch.n_id[: batch.batch_size]
                ] = z

                total_correct += (
                    (z.round() == batch.y[: batch.batch_size].float()).sum().item()
                )
                total_examples += batch.batch_size

        if total_examples > 0:
            _dbg(
                "EmbeddingWriter._run_inference",
                f"推理完成: total_examples={total_examples} "
                f"inf_acc={total_correct / total_examples:.4f} "
                f"(drop_last=True，最后不足batch_size的节点已丢弃)",
            )

    def write(self) -> None:
        """
        完整推理 + parquet 写出入口。

        上游: 所有逻辑内联在 __main__。
        改写: 集中封装，加 timestamp 后缀防并发覆盖。
        """
        import cudf

        args = self.args
        rank = self.bundle.rank
        data = self.bundle.data

        inf_loader = self._build_inf_loader()
        self._init_embedding_slots()
        self._run_inference(inf_loader)

        # 构建 cudf DataFrame
        df = cudf.DataFrame(
            self.feature_store["entity", "emb", None].get_local_tensor(),
            index=None,
            columns=[f"emb_{i}" for i in range(args.hidden_channels)],
        )
        df["y"] = self.feature_store["entity", "y", None].get_local_tensor()
        df["z"] = self.feature_store["entity", "z", None].get_local_tensor()

        _dbg(
            "EmbeddingWriter.write",
            f"cudf DataFrame 构建完成: shape={df.shape} "
            f"columns={list(df.columns)[:5]}... "
            f"y.value_counts={df['y'].value_counts().to_dict()}",
        )

        os.makedirs(args.embedding_dir, exist_ok=True)

        # 上游文件名: 5个超参拼接，无时间戳，并发实验静默覆盖
        # 改写: 加 timestamp 后缀防覆盖
        timestamp = int(time.time())
        fname = (
            f"emb_{args.encoder}_{args.hidden_channels}"
            f"_{args.batch_size}_{args.lr}_{args.epochs}"
            f"_{rank}_{timestamp}.parquet"
        )
        out_path = os.path.join(args.embedding_dir, fname)

        # 路径安全: 确认写出路径在 embedding_dir 下 (validate() 已检查 "..")
        _dbg(
            "EmbeddingWriter.write",
            f"写出 parquet: path={out_path} shape={df.shape}",
        )

        df.to_parquet(out_path)
        print(f"[bitcoin_mnmg] rank={rank} 写出 embedding: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# train_epoch — 对应上游 for epoch in range(1, args.epochs+1) 训练块
# ──────────────────────────────────────────────────────────────────────────────


def train_epoch(
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    epoch: int,
    rank: int,
) -> None:
    """
    单 epoch 训练，对应上游训练循环。

    上游: 仅 rank0 每10步打印 loss，无 batch shape 信息。
    改写: 加断点调试打印 batch.x/edge_index/out shape。
    """
    encoder.train()
    for it, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = encoder(batch.x, batch.edge_index)

        _dbg(
            "train_epoch",
            f"epoch={epoch} it={it} "
            f"batch.x.shape={batch.x.shape} "
            f"batch.edge_index.shape={batch.edge_index.shape} "
            f"out.shape={out.shape} "
            f"batch.batch_size={batch.batch_size} "
            f"batch.y[:batch_size].unique={batch.y[:batch.batch_size].unique().tolist()}",
        )

        # 上游: cross_entropy 对 num_classes=2 的头遇到 y=2 (unknown) 可能出错
        # 改写: 保留上游行为，调试 print 使问题可见
        loss = F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
        loss.backward()
        optimizer.step()

        if rank == 0 and it % 10 == 0:
            print(f"Epoch {epoch} iter {it} loss: {loss.item():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# eval_epoch — 对应上游测试循环
# ──────────────────────────────────────────────────────────────────────────────


def eval_epoch(
    encoder: torch.nn.Module,
    test_loader,
    rank: int,
) -> None:
    """
    测试集评估，对应上游 with torch.no_grad(): ... test_loader 块。

    上游: 仅打印 rank 级 loss/acc，无 batch 级统计。
    改写: 加断点调试打印每 batch 统计。
    """
    encoder.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            out = encoder(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
            total_correct += (
                (out[: batch.batch_size].argmax(dim=-1) == batch.y[: batch.batch_size])
                .sum()
                .item()
            )

            _dbg(
                "eval_epoch",
                f"batch_idx={batch_idx} "
                f"batch.x.shape={batch.x.shape} "
                f"batch.batch_size={batch.batch_size} "
                f"batch_loss={loss.item():.4f}",
            )

    if total_examples > 0:
        print(
            f"rank={rank} Test loss: {total_loss / total_examples:.4f}"
            f" acc: {total_correct / total_examples:.4f}"
        )
    else:
        print(f"rank={rank} Test: no examples (ix_test 为空，检查 world_size 与节点数)")


# ──────────────────────────────────────────────────────────────────────────────
# main — 对应上游 if __name__ == "__main__": 块
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    主入口，对应上游 __main__ 块。

    改写: 拆分为多个函数 + context manager，保证资源清理。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data/")
    parser.add_argument("--encoder", type=str, default="sage")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embedding_dir", type=str, default=None, required=False)

    ns = parser.parse_args()
    args = BitcoinMnmgArgs.from_namespace(ns)
    args.validate()
    args.debug_dump()

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    _dbg(
        "main",
        f"distributed 初始化完成: rank={rank} world_size={world_size} "
        f"local_rank={local_rank}",
    )

    cugraph_id = create_uid(rank, device=torch.device(f"cuda:{local_rank}"))

    # CugraphWorkerSession: context manager 保证 shutdown 在异常路径也执行
    with CugraphWorkerSession(rank, local_rank, world_size, cugraph_id):
        from torch_geometric.datasets import EllipticBitcoinDataset

        dataset = EllipticBitcoinDataset(root=args.dataset_root)
        data = dataset[0]
        assert dataset.num_classes == 2, (
            f"num_classes={dataset.num_classes}，期望2 (illicit/licit)"
        )

        # 上游: data.x[:, :94] 切掉后94列的预生成图嵌入，仅保留原始94维特征
        data.x = data.x[:, :94]
        _dbg(
            "main",
            f"数据集加载完成: "
            f"num_nodes={data.num_nodes} "
            f"num_edges={data.edge_index.shape[1]} "
            f"x.shape={data.x.shape} "
            f"y.unique={data.y.unique().tolist()} "
            f"train_mask.sum={data.train_mask.sum().item()} "
            f"test_mask.sum={data.test_mask.sum().item()}",
        )

        # 构建分布式图
        bundle = BitcoinGraphBundle.build(data, rank, world_size)

        # 构建 encoder
        encoder_raw = build_encoder(args, in_channels=data.x.shape[1])
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder_raw.cuda(), device_ids=[local_rank]
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

        # 分配 train/test 节点
        ix_train = torch.tensor_split(
            torch.arange(data.num_nodes, device="cuda")[data.train_mask], world_size
        )[rank]
        ix_test = torch.tensor_split(
            torch.arange(data.num_nodes, device="cuda")[data.test_mask], world_size
        )[rank]

        _dbg(
            "main",
            f"节点分配: rank={rank} "
            f"ix_train.shape={ix_train.shape} "
            f"ix_test.shape={ix_test.shape} "
            f"ix_train_empty={ix_train.numel() == 0} "
            f"ix_test_empty={ix_test.numel() == 0}",
        )

        from cugraph_pyg.loader import NeighborLoader

        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_neighbors": [25, 10],
            "shuffle": True,
            "drop_last": True,
        }

        train_loader = NeighborLoader(
            (bundle.feature_store, bundle.graph_store),
            input_nodes=ix_train,
            **loader_kwargs,
        )
        test_loader = NeighborLoader(
            (bundle.feature_store, bundle.graph_store),
            input_nodes=ix_test,
            **loader_kwargs,
        )

        _dbg("main", "NeighborLoader 构建完成，开始训练")

        # 训练
        for epoch in range(1, args.epochs + 1):
            train_epoch(encoder, optimizer, train_loader, epoch, rank)

        # 评估
        torch.distributed.barrier()
        eval_epoch(encoder, test_loader, rank)
        torch.distributed.barrier()

        # 写出 embedding (可选)
        if args.embedding_dir is not None:
            _dbg("main", f"开始推理并写出 embedding: embedding_dir={args.embedding_dir!r}")
            writer = EmbeddingWriter(
                feature_store=bundle.feature_store,
                graph_store=bundle.graph_store,
                encoder=encoder,
                bundle=bundle,
                args=args,
            )
            writer.write()

        # CugraphWorkerSession.__exit__ 自动调用 cugraph_comms_shutdown()

    torch.distributed.destroy_process_group()
    _dbg("main", "destroy_process_group 完成，进程正常退出")


if __name__ == "__main__":
    main()
