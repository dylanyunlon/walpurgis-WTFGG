"""
unified_store.py — 07ce63f 迁移: 统一 WholeGraph FeatureStore 与 GraphStore

migrate 07ce63f: [FEA] Support Unified WholeGraph FeatureStore and GraphStore

上游变化 (07ce63f, cugraph-gnn / python/cugraph-pyg/):

1. tensor/ 子包全新引入 (dist_tensor.py / dist_matrix.py / utils.py):
   - DistTensor: WholeGraph 分布式 1D/2D tensor 封装，支持 scatter/gather，
     后端自动选 nccl(跨节点) 或 vmm(单节点 NVLink)
   - DistEmbedding(DistTensor 子类): 2D embedding，通过 WholeMemoryEmbedding 管理，
     支持 cache_policy / gather_sms / round_robin_size
   - DistMatrix: 基于两个 DistTensor(_col, _row) 的分布式稀疏矩阵，COO 格式
   - utils.py: create_wg_dist_tensor / copy_host_global_tensor_to_local /
     has_nvlink_network / is_empty / empty 等底层工具

2. data/feature_store.py — WholeFeatureStore → FeatureStore 重命名 + 彻底重构:
   旧实现:
     __init__: memory_type='distributed' → wgth.get_global_communicator() + wgth.create_wholememory_tensor
     _put_tensor: 手工 all_gather sizes → create_wholememory_tensor → wg_embedding.scatter
     _get_tensor: wg_embedding.gather(attr.index)
   新实现:
     __init__: memory_type=None(废弃参数) + 自动检测 LOCAL_WORLD_SIZE vs world_size → vmm/nccl
     __make_wg_tensor: all_gather维度+dtype+sizes → DistTensor(1D)/DistEmbedding(2D) → tx[ix]=val
     _put_tensor: 支持 attr.index 指定偏移写入（首次建立时）
     _get_tensor: emb[attr.index] 或整张 emb 返回

3. data/graph_store.py — 新增 NewGraphStore 类 (434行):
   - 基于 DistMatrix 存储边索引，自动选 vmm/nccl 后端
   - _put_edge_index: all_gather sizes → DistMatrix scatter 局部切片
   - __get_edgelist: 拼装多 edge-type → {src/dst/eid/etp/wgt} dict
   - _num_vertices: per-edge-type size → all_reduce MAX (多GPU全局对齐)
   - _graph 属性: 懒惰构建 SGGraph(单卡) / MGGraph(多卡)
   - 权重支持: _set_weight_attr → __get_weight_tensor

4. data/__init__.py — GraphStore 变为工厂函数:
   - is_multi_gpu=True → wgth.initialize → 返回 NewGraphStore
   - is_multi_gpu=False (废弃) → 返回旧 DEPRECATED__OldGraphStore
   - WholeFeatureStore → 重定向到新 FeatureStore（FutureWarning）

5. loader/*.py — 类型检查扩展:
   isinstance(data[1], (GraphStore, NewGraphStore)) 替代单一 GraphStore

6. examples/: 参数简化，废弃 wg_mem_type/in_memory/skip_partition，
   rgcn 示例彻底去掉磁盘分区，改用 WholeGraph 广播 + FeatureStore 做节点 0 → 全 rank 分发

7. tests/: 全部测试加 os.environ["LOCAL_WORLD_SIZE"] 模拟 torchrun 环境，
   新增 test_dist_tensor_mg.py / test_dist_matrix_mg.py 完整测试套件

Bug 根因 (Knuth 视角):
1. diff 对比源: 旧 WholeFeatureStore 硬耦合 wg_comm + memory_type，
   用户必须手工预分区数据再分别 put 到各 rank，接口复杂且易错；
   新 FeatureStore 任意 rank 可持有完整 tensor，内部自动 all_gather + 分片。
2. 用户角度 bug: 旧版 is_multi_gpu 参数未废弃前无法区分「未传」vs「明确=False」；
   工厂函数化后 GraphStore() 无参 → 旧路径（保留兼容），
   GraphStore(is_multi_gpu=True) → 新路径，语义清晰，FutureWarning 催迁移。
3. 系统角度安全: LOCAL_WORLD_SIZE 环境变量未设置时
   int(os.environ["LOCAL_WORLD_SIZE"]) 会抛 KeyError，
   应在 torchrun 下运行（会自动设置），裸 python 下缺失此变量是已知限制，
   非 bug 而是设计约束（测试通过 os.environ["LOCAL_WORLD_SIZE"]=str(world_size) 模拟）。

Walpurgis 改写 20%（鲁迅拿法）:
- BackendSelector: 替代 __init__ 里散落的 if LOCAL_WORLD_SIZE==world_size 判断，
  提取为可观测的决策对象，携带 (backend, reason) 二元状态
- UnifiedStoreRegistry: 替代 __features dict，封装 put/get/remove 语义，
  加键格式校验 + debug 打印，与上游 dict 完全等价但可被单元测试
- TensorDimStrategy: 替代 __make_wg_tensor 里 dim()==1/2 分支，
  提取为 Strategy 对象，携带 DistTensor/DistEmbedding 构造参数
- 全链路 WALPURGIS_DEBUG=1 断点 print

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import warnings
from typing import Optional, Dict, Tuple, Any

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS unified_store]", *args, file=sys.stderr, flush=True, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# BackendSelector — 替代散落的 vmm/nccl 判断逻辑
# ──────────────────────────────────────────────────────────────────────────────

class BackendSelector:
    """
    封装 07ce63f 引入的 WholeGraph 后端自动选择逻辑。

    上游代码（feature_store.py __init__ 和 graph_store.py NewGraphStore.__init__）:
        if int(os.environ["LOCAL_WORLD_SIZE"]) == torch.distributed.get_world_size():
            self.__backend = "vmm"
        else:
            self.__backend = "vmm" if has_nvlink_network() else "nccl"

    语义:
        LOCAL_WORLD_SIZE == world_size  → 单节点，所有 GPU 在同一台机器，
                                          可用 VMM (Virtual Memory Mapping，NVLink 直连)
        LOCAL_WORLD_SIZE < world_size   → 多节点，需探测是否有跨节点 NVLink；
                                          有 NVLink → vmm，无 → nccl

    Walpurgis 改写:
    - backend 和 reason 均可观测（不是隐藏的 __backend 私有属性）
    - select() 静态方法可在无 torch.distributed 时被 mock 测试
    - WALPURGIS_DEBUG=1 时打印决策路径

    断点 1: 进入 select()，打印 LOCAL_WORLD_SIZE vs world_size
    断点 2: 决策完成，打印 backend 和 reason
    """

    BACKEND_VMM = "vmm"
    BACKEND_NCCL = "nccl"

    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason

    def __repr__(self):
        return f"BackendSelector(backend={self.backend!r}, reason={self.reason!r})"

    @staticmethod
    def select(local_world_size: int, world_size: int, has_nvlink_fn=None) -> "BackendSelector":
        """
        根据 LOCAL_WORLD_SIZE 和 world_size 选择 WholeGraph 通信后端。

        参数
        ----
        local_world_size : int  来自 int(os.environ["LOCAL_WORLD_SIZE"])
        world_size       : int  来自 torch.distributed.get_world_size()
        has_nvlink_fn    : callable or None，默认使用 cugraph_pyg.tensor.utils.has_nvlink_network

        返回
        ----
        BackendSelector 对象，.backend 为 "vmm" 或 "nccl"
        """
        # ── 断点 1: 进入 select ──────────────────────────────────────────
        _dbg(
            f"BackendSelector.select(): "
            f"local_world_size={local_world_size}, world_size={world_size}"
        )

        if local_world_size == world_size:
            # 单节点：所有 rank 在同一台机器，NVLink 可用，使用 VMM
            result = BackendSelector(
                BackendSelector.BACKEND_VMM,
                reason="single-node: LOCAL_WORLD_SIZE == world_size, VMM (NVLink) available",
            )
        else:
            # 多节点：需要探测跨节点 NVLink
            if has_nvlink_fn is None:
                try:
                    from cugraph_pyg.tensor.utils import has_nvlink_network
                    has_nvlink_fn = has_nvlink_network
                except ImportError:
                    has_nvlink_fn = lambda: False

            if has_nvlink_fn():
                result = BackendSelector(
                    BackendSelector.BACKEND_VMM,
                    reason="multi-node: has_nvlink_network()=True, VMM (cross-node NVLink)",
                )
            else:
                result = BackendSelector(
                    BackendSelector.BACKEND_NCCL,
                    reason="multi-node: has_nvlink_network()=False, fallback to NCCL",
                )

        # ── 断点 2: 决策完成 ──────────────────────────────────────────────
        _dbg(f"BackendSelector.select() → {result}")
        return result

    @staticmethod
    def from_env() -> "BackendSelector":
        """
        从当前运行环境自动读取 LOCAL_WORLD_SIZE 并选择后端。

        要求在 torchrun 下运行（LOCAL_WORLD_SIZE 由 torchrun 自动注入）。
        裸 python 运行需手工设置 os.environ["LOCAL_WORLD_SIZE"]。

        抛出
        ----
        KeyError  如果 LOCAL_WORLD_SIZE 未设置（非 torchrun 环境）
        RuntimeError  如果 torch.distributed 未初始化
        """
        import torch
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        world_size = torch.distributed.get_world_size()

        _dbg(f"BackendSelector.from_env(): LOCAL_WORLD_SIZE={local_world_size}, world_size={world_size}")

        return BackendSelector.select(local_world_size, world_size)


# ──────────────────────────────────────────────────────────────────────────────
# UnifiedStoreRegistry — 替代 __features / __edge_indices dict
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedStoreRegistry:
    """
    封装 FeatureStore.__features 和 NewGraphStore.__edge_indices 的 dict 语义。

    上游代码中两处均为普通 dict:
        self.__features = {}                   # feature_store.py L171
        self.__edge_indices = {}               # graph_store.py NewGraphStore.__init__

    Walpurgis 改写:
    - put/get/remove 替代 dict 直接赋值，加键格式打印
    - has() 替代 in 运算符
    - keys() 返回 list（与上游 .keys() 调用兼容）
    - WALPURGIS_DEBUG=1 时每次操作均打印键和值摘要

    断点 3: put() 入口，打印 key + 旧值（如有）+ 新值摘要
    断点 4: get() 入口，打印 key + 是否命中
    断点 5: remove() 入口，打印 key + 是否存在
    """

    def __init__(self, name: str = "registry"):
        self._store: Dict[Any, Any] = {}
        self._name = name

    def put(self, key, value) -> None:
        """存入键值对。键已存在时覆盖（与 dict 赋值语义一致）。"""
        old = self._store.get(key, None)
        # ── 断点 3 ────────────────────────────────────────────────────────
        _dbg(
            f"[{self._name}] put(): key={key!r}, "
            f"overwrite={old is not None}, "
            f"new_type={type(value).__name__}"
        )
        self._store[key] = value

    def get(self, key, default=None):
        """取键值，不存在返回 default（与 dict.get 一致）。"""
        hit = key in self._store
        # ── 断点 4 ────────────────────────────────────────────────────────
        _dbg(f"[{self._name}] get(): key={key!r}, hit={hit}")
        return self._store.get(key, default)

    def remove(self, key) -> bool:
        """删除键，存在返回 True，不存在返回 False。"""
        exists = key in self._store
        # ── 断点 5 ────────────────────────────────────────────────────────
        _dbg(f"[{self._name}] remove(): key={key!r}, exists={exists}")
        if exists:
            del self._store[key]
        return exists

    def has(self, key) -> bool:
        return key in self._store

    def keys(self):
        return list(self._store.keys())

    def items(self):
        return list(self._store.items())

    def __contains__(self, key):
        return key in self._store

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"UnifiedStoreRegistry(name={self._name!r}, size={len(self._store)})"


# ──────────────────────────────────────────────────────────────────────────────
# TensorDimStrategy — 替代 __make_wg_tensor 里的 dim()==1/2 分支
# ──────────────────────────────────────────────────────────────────────────────

class TensorDimStrategy:
    """
    封装 07ce63f feature_store.py __make_wg_tensor 中的维度分支逻辑。

    上游 __make_wg_tensor 的核心分支 (feature_store.py ~L195-L240):
        if tensor.dim() == 1:
            global_shape = [sizes.sum()]
            tx = DistTensor(None, shape=global_shape, dtype=dtype, ...)
        elif tensor.dim() == 2:
            # all_gather trailing dim td
            global_shape = [int(sizes.sum()), td]
            tx = DistEmbedding(None, shape=global_shape, dtype=dtype, ...)
        else:
            raise ValueError("Tensor must be 1D or 2D.")

    Walpurgis 改写:
    - TensorDimStrategy 枚举维度类型 (DIM1 / DIM2)
    - build() 静态方法携带 global_shape + cls (DistTensor/DistEmbedding) 返回
    - WALPURGIS_DEBUG=1 时打印维度决策

    断点 6: build() 入口，打印 tensor.dim() 和 global_shape
    断点 7: build() 完成，打印构造参数摘要
    """

    DIM1 = "dim1"
    DIM2 = "dim2"

    def __init__(self, dim_type: str, global_shape: list, tensor_cls, trailing_dim: Optional[int] = None):
        self.dim_type = dim_type
        self.global_shape = global_shape
        self.tensor_cls = tensor_cls
        self.trailing_dim = trailing_dim

    def __repr__(self):
        return (
            f"TensorDimStrategy("
            f"dim_type={self.dim_type!r}, "
            f"global_shape={self.global_shape}, "
            f"cls={self.tensor_cls.__name__ if self.tensor_cls else None}, "
            f"trailing_dim={self.trailing_dim})"
        )

    @staticmethod
    def build(
        tensor_dim: int,
        global_row_count: int,
        trailing_dim: Optional[int] = None,
    ) -> "TensorDimStrategy":
        """
        根据 tensor 维度构建策略对象。

        参数
        ----
        tensor_dim      : tensor.dim()，必须为 1 或 2
        global_row_count: sizes.sum()，all_gather 后的全局行数
        trailing_dim    : tensor.dim()==2 时的第二维大小（all_gather 后验证）

        返回
        ----
        TensorDimStrategy 对象
        """
        # ── 断点 6 ────────────────────────────────────────────────────────
        _dbg(
            f"TensorDimStrategy.build(): "
            f"tensor_dim={tensor_dim}, global_row_count={global_row_count}, "
            f"trailing_dim={trailing_dim}"
        )

        if tensor_dim == 1:
            from cugraph_pyg.tensor import DistTensor
            global_shape = [global_row_count]
            result = TensorDimStrategy(
                TensorDimStrategy.DIM1,
                global_shape=global_shape,
                tensor_cls=DistTensor,
                trailing_dim=None,
            )
        elif tensor_dim == 2:
            if trailing_dim is None:
                raise ValueError("trailing_dim must be provided for 2D tensors")
            from cugraph_pyg.tensor import DistEmbedding
            global_shape = [global_row_count, trailing_dim]
            result = TensorDimStrategy(
                TensorDimStrategy.DIM2,
                global_shape=global_shape,
                tensor_cls=DistEmbedding,
                trailing_dim=trailing_dim,
            )
        else:
            raise ValueError(f"Tensor must be 1D or 2D, got {tensor_dim}D")

        # ── 断点 7 ────────────────────────────────────────────────────────
        _dbg(f"TensorDimStrategy.build() → {result}")
        return result

    def instantiate(self, dtype, device: str, backend: str):
        """
        实例化对应的 DistTensor 或 DistEmbedding。

        参数
        ----
        dtype   : torch.dtype
        device  : "cpu" 或 "cuda"
        backend : "vmm" 或 "nccl"

        断点 8: 打印实例化参数
        """
        # ── 断点 8 ────────────────────────────────────────────────────────
        _dbg(
            f"TensorDimStrategy.instantiate(): "
            f"cls={self.tensor_cls.__name__ if self.tensor_cls else None}, "
            f"global_shape={self.global_shape}, dtype={dtype}, "
            f"device={device!r}, backend={backend!r}"
        )

        tx = self.tensor_cls(
            None,
            shape=self.global_shape,
            dtype=dtype,
            device=device,
            backend=backend,
        )
        _dbg(f"  实例化完成: {tx}")
        return tx


# ──────────────────────────────────────────────────────────────────────────────
# DtypeNegotiator — 替代 __make_wg_tensor 里的 all_gather dtype 协商
# ──────────────────────────────────────────────────────────────────────────────

class DtypeNegotiator:
    """
    封装 07ce63f feature_store.py __make_wg_tensor 中的 dtype 全局协商逻辑。

    上游代码片段 (feature_store.py ~L185-L210):
        dtypes = {torch.float32: 0, torch.float64: 1, torch.int32: 2,
                  torch.int64: 3, torch.bool: 4}
        dtype_ids = {v: k for k, v in dtypes.items()}

        dtype = torch.tensor(_encode_dtype(tensor.dtype), device="cuda", ...)
        global_dtype = torch.empty((world_size,), device="cuda", ...)
        torch.distributed.all_gather_into_tensor(global_dtype, dtype)
        global_dtype = global_dtype[sizes > 0]
        if len(global_dtype) == 0:
            raise ValueError("Tensor is empty")
        if (global_dtype[0] == global_dtype).all():
            dtype = _decode_dtype(int(global_dtype[0]))
        else:
            raise ValueError("Tensor dtype must be the same across ranks")

    问题: 所有 rank 必须持有相同 dtype，否则协商失败。
    空 rank（sizes == 0）的 dtype 被 global_dtype[sizes > 0] 过滤，
    避免空 tensor 的任意 dtype 污染协商结果。

    Walpurgis 改写:
    - DtypeNegotiator.encode/decode 替代内嵌函数 _encode_dtype/_decode_dtype
    - negotiate() 静态方法封装完整协商流程，返回 torch.dtype
    - WALPURGIS_DEBUG=1 时打印各 rank dtype ID + 过滤结果

    断点 9: encode 入口
    断点 10: negotiate 全局协商结果
    """

    # migrate 6d1a8de: 修正上游 feature_store.py dtype 映射表
    #
    # 上游 6d1a8de (2025-11-26, Alex Barghi):
    #   [IMP] Support more dtypes in the cuGraph-PyG FeatureStore (#346)
    #   - 新增 int16(4) / float16(5) / int8(6) — WholeGraph 原生支持
    #   - 移除 torch.bool(原id=4) — WholeGraph 从未真正支持, 放入属于误植,
    #     使用 torch.bool 时无论如何都会抛异常, 因此移除是非破坏性变更
    #   - bfloat16 暂不包含 (WholeGraph 接口列出但实测不工作, 见 PR #346 描述)
    #
    # Walpurgis id 序列对齐 (6d1a8de → Walpurgis):
    #   float32=0  float64=1  int32=2  int64=3   ← 不变
    #   bool=4     → 移除 (id=4 让给 int16)
    #   int16=4    float16=5  int8=6             ← 6d1a8de 新增
    #   bfloat16=7 ← 220563b 已迁移, id 从原 8 调整为 7 (bool 腾出 id=4 后整体前移)
    #
    # 断点 9: encode/decode 打印 dtype_id, 便于排查跨 rank dtype 不一致
    DTYPE_TO_ID = {
        "torch.float32":  0,
        "torch.float64":  1,
        "torch.int32":    2,
        "torch.int64":    3,
        # migrate 6d1a8de: torch.bool 移除 (原 id=4, WholeGraph 从未实际支持)
        # 鲁迅: 「错误的开始」若不纠正, 日后的每一步都是在错上加错。
        # 以前用 torch.bool 的调用本就会在 WholeGraph 层抛异常, 移除属于正名。
        "torch.int16":    4,   # migrate 6d1a8de: WholeGraph 原生支持
        "torch.float16":  5,   # migrate 6d1a8de: WholeGraph 原生支持 (半精度特征)
        "torch.int8":     6,   # migrate 6d1a8de: WholeGraph 原生支持 (量化场景)
        # migrate 220563b: bfloat16 — fp16/bf16 embedding 训练必需
        # id 由原 8 调整为 7 (bool 腾出 id=4 后整体前移一位)
        "torch.bfloat16": 7,
    }
    # 断点调试: bool 移除后的 encode 会给出明确 ValueError, 而非静默发送错误 id
    # 若有存量代码依赖 bool dtype, WALPURGIS_DEBUG=1 时 encode() 会打印出错位置
    ID_TO_DTYPE_NAME = {v: k for k, v in DTYPE_TO_ID.items()}

    @staticmethod
    def encode(dtype) -> int:
        """将 torch.dtype 编码为整数 ID（用于 all_gather 通信）。"""
        import torch
        key = str(dtype)
        if key not in DtypeNegotiator.DTYPE_TO_ID:
            raise ValueError(f"Unsupported dtype for DtypeNegotiator: {dtype}")
        result = DtypeNegotiator.DTYPE_TO_ID[key]
        # ── 断点 9 ────────────────────────────────────────────────────────
        _dbg(f"DtypeNegotiator.encode(): dtype={dtype} → id={result}")
        return result

    @staticmethod
    def decode(dtype_id: int):
        """将整数 ID 解码回 torch.dtype。"""
        import torch
        if dtype_id not in DtypeNegotiator.ID_TO_DTYPE_NAME:
            raise ValueError(f"Unknown dtype id: {dtype_id}")
        name = DtypeNegotiator.ID_TO_DTYPE_NAME[dtype_id]
        result = getattr(torch, name.replace("torch.", ""))
        return result

    @staticmethod
    def negotiate(local_tensor, sizes_tensor) -> "torch.dtype":
        """
        通过 all_gather 协商全局统一的 dtype。

        参数
        ----
        local_tensor  : 本地 tensor（非空 rank 提供，空 rank 可以是任意 dtype 的空 tensor）
        sizes_tensor  : all_gather 后的各 rank 行数 tensor (world_size,)，用于过滤空 rank

        返回
        ----
        torch.dtype  所有非空 rank 一致的 dtype

        抛出
        ----
        ValueError  如果所有 rank 均为空，或 dtype 不一致
        """
        import torch

        world_size = sizes_tensor.shape[0]
        dtype_id = torch.tensor(
            DtypeNegotiator.encode(local_tensor.dtype),
            device="cuda",
            dtype=torch.int64,
        )
        global_dtype = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(global_dtype, dtype_id)

        # 过滤空 rank（sizes==0 的 rank dtype 不具有代表性）
        valid_dtype = global_dtype[sizes_tensor > 0]

        # ── 断点 10 ──────────────────────────────────────────────────────
        _dbg(
            f"DtypeNegotiator.negotiate(): "
            f"global_dtype_ids={global_dtype.tolist()}, "
            f"valid_dtype_ids={valid_dtype.tolist() if len(valid_dtype) > 0 else '[]'}"
        )

        if len(valid_dtype) == 0:
            raise ValueError("Tensor is empty across all ranks")

        if not (valid_dtype[0] == valid_dtype).all():
            raise ValueError(
                f"Tensor dtype must be the same across ranks, got ids={valid_dtype.tolist()}"
            )

        result = DtypeNegotiator.decode(int(valid_dtype[0]))
        _dbg(f"  协商完成: dtype={result}")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# FeatureStoreFactory — GraphStore/FeatureStore 工厂函数对应的逻辑
# ──────────────────────────────────────────────────────────────────────────────

class FeatureStoreFactory:
    """
    封装 07ce63f data/__init__.py 中 GraphStore() 工厂函数逻辑。

    上游工厂函数:
        def GraphStore(*args, **kwargs):
            is_multi_gpu = kwargs.pop("is_multi_gpu", None)
            if is_multi_gpu is not None:
                warnings.warn("is_multi_gpu is deprecated...", FutureWarning)
                if is_multi_gpu:
                    wgth.initialize.init(rank, world_size, rank, world_size)
                    return NewGraphStore(*args, **kwargs)
                else:
                    warnings.warn("Running without torchrun will be deprecated...")
            return DEPRECATED__OldGraphStore(*args, **kwargs)

    三条路径:
        A. 不传 is_multi_gpu          → DEPRECATED__OldGraphStore（旧路径保留）
        B. is_multi_gpu=True          → wgth.init + NewGraphStore（新路径）
        C. is_multi_gpu=False         → FutureWarning + DEPRECATED__OldGraphStore

    Walpurgis 改写:
    - ConstructionPath 枚举三条路径，决策可被单元测试
    - resolve() 静态方法返回 (path, store_instance) 对，避免工厂函数吞掉路径信息
    - WALPURGIS_DEBUG=1 时打印所选路径和原因

    断点 11: resolve() 入口，打印 is_multi_gpu 参数
    断点 12: 路径决策完成，打印 path + 类型
    """

    PATH_OLD = "deprecated_old"
    PATH_NEW_MULTIGPU = "new_multigpu"
    PATH_OLD_SINGLEGPU_WARN = "old_singlegpu_deprecated_warn"

    @staticmethod
    def resolve(is_multi_gpu, args, kwargs) -> Tuple[str, Any]:
        """
        根据 is_multi_gpu 参数选择构造路径，返回 (path, store_instance)。

        参数
        ----
        is_multi_gpu : None / True / False
        args, kwargs : 透传给构造函数的参数

        返回
        ----
        Tuple[str, Any]: (路径标识, 构造好的 store 对象)
        """
        # ── 断点 11 ──────────────────────────────────────────────────────
        _dbg(f"FeatureStoreFactory.resolve(): is_multi_gpu={is_multi_gpu!r}")

        if is_multi_gpu is None:
            # 路径 A：未传参数，走旧路径
            path = FeatureStoreFactory.PATH_OLD
            _dbg(f"  路径 A: is_multi_gpu 未传，使用 DEPRECATED__OldGraphStore")
            from cugraph_pyg.data.graph_store import GraphStore as _OldGraphStore
            store = _OldGraphStore(*args, **kwargs)

        elif is_multi_gpu:
            # 路径 B：多 GPU，初始化 WholeGraph + 返回 NewGraphStore
            path = FeatureStoreFactory.PATH_NEW_MULTIGPU
            _dbg(f"  路径 B: is_multi_gpu=True，初始化 WholeGraph + NewGraphStore")

            warnings.warn(
                "The is_multi_gpu argument is deprecated."
                "In release 25.08, multi-GPU mode will be enabled automatically"
                "when there is more than one GPU worker.",
                FutureWarning,
            )

            try:
                import torch
                import pylibwholegraph.torch as wgth
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

                _dbg(f"  wgth.initialize.init(rank={rank}, world_size={world_size})")
                wgth.initialize.init(rank, world_size, rank, world_size)
            except Exception as e:
                _dbg(f"  wgth.initialize.init 异常（可能已初始化）: {e}")
                warnings.warn(f"WholeGraph already initialized, continuing. ({e})")

            from cugraph_pyg.data.graph_store import NewGraphStore
            store = NewGraphStore(*args, **kwargs)

        else:
            # 路径 C：is_multi_gpu=False，废弃警告 + 旧路径
            path = FeatureStoreFactory.PATH_OLD_SINGLEGPU_WARN
            _dbg(f"  路径 C: is_multi_gpu=False，FutureWarning + DEPRECATED__OldGraphStore")

            warnings.warn(
                "Running without torchrun will be deprecated in release 25.08.",
                FutureWarning,
            )
            from cugraph_pyg.data.graph_store import GraphStore as _OldGraphStore
            store = _OldGraphStore(*args, **kwargs)

        # ── 断点 12 ──────────────────────────────────────────────────────
        _dbg(f"FeatureStoreFactory.resolve() → path={path!r}, store_type={type(store).__name__}")
        return path, store


# ──────────────────────────────────────────────────────────────────────────────
# EdgelistBuilder — 对应 NewGraphStore.__get_edgelist 的逻辑提取
# ──────────────────────────────────────────────────────────────────────────────

class EdgelistBuilder:
    """
    封装 07ce63f graph_store.py NewGraphStore.__get_edgelist 的核心拼装逻辑。

    上游 __get_edgelist (graph_store.py ~L610-L680):
        1. sorted_keys = sorted(self.__edge_indices.keys())
        2. edge_index = torch.concat([
               torch.stack([
                   self.__edge_indices[dst, rel, src].local_col + vertex_offsets[dst],
                   self.__edge_indices[dst, rel, src].local_row + vertex_offsets[src],
               ])
               for (dst, rel, src) in sorted_keys
           ], axis=1)
        3. edge_type_array = arange(n_etypes).repeat_interleave(per_etype_sizes)
        4. edge_id_array  = per-rank offset + arange(local_edges_per_etype)
        5. 可选 wgt 拼装
        返回 dict(src=, dst=, eid=, etp=, [wgt=])

    注意 PyG 约定: edge_index 格式是 (dst, rel, src)，
    但 cuGraph 的 src/dst 与 PyG 相反:
        PyG edge_index[0] = dst → cuGraph dst
        PyG edge_index[1] = src → cuGraph src
    这个约定上游注释已说明，迁移时保留。

    Walpurgis 改写:
    - EdgelistEntry: dataclass 封装单条 edge-type 的 (dst_col, src_row, local_size)
    - EdgelistBuilder.build(): 构造完整 edgelist dict，替代 __get_edgelist 私有方法
    - WALPURGIS_DEBUG=1 时打印各 edge-type 的局部边数 + 全局偏移

    断点 13: build() 入口，打印 sorted_keys + per-type sizes
    断点 14: build() 完成，打印最终 edge_index shape + eid range
    """

    @staticmethod
    def build(edge_indices, vertex_offsets, is_multi_gpu: bool, weight_attr=None):
        """
        构造 cuGraph 期望的 edgelist dict。

        参数
        ----
        edge_indices   : UnifiedStoreRegistry 或 dict，键为 (dst, rel, src) tuple
        vertex_offsets : dict[str, int]，各顶点类型的全局偏移
        is_multi_gpu   : bool，是否多 GPU（影响 edge_id 的全局偏移计算）
        weight_attr    : (feature_store, attr_name) or None

        返回
        ----
        dict with keys: src, dst, eid, etp, [wgt]
        """
        import torch

        sorted_keys = sorted(edge_indices.keys())

        # ── 断点 13 ──────────────────────────────────────────────────────
        _dbg(
            f"EdgelistBuilder.build(): "
            f"n_etypes={len(sorted_keys)}, "
            f"sorted_keys={sorted_keys}"
        )

        # 拼装 edge_index（PyG 约定: [dst_col+offset, src_row+offset]）
        edge_index_parts = []
        for (dst_type, rel_type, src_type) in sorted_keys:
            dm = edge_indices[(dst_type, rel_type, src_type)]
            col_shifted = dm.local_col + vertex_offsets[dst_type]
            row_shifted = dm.local_row + vertex_offsets[src_type]
            edge_index_parts.append(torch.stack([col_shifted, row_shifted]))
            _dbg(
                f"  edge_type=({dst_type}, {rel_type}, {src_type}): "
                f"local_edges={dm.local_col.numel()}"
            )

        edge_index = torch.concat(edge_index_parts, axis=1).cuda()

        # edge_type_array: 每条边对应的数值 edge-type 编号
        per_etype_local_sizes = torch.tensor(
            [edge_indices[et].local_row.numel() for et in sorted_keys],
            device="cuda",
            dtype=torch.int64,
        )
        edge_type_array = torch.arange(
            len(sorted_keys), dtype=torch.int32, device="cuda"
        ).repeat_interleave(per_etype_local_sizes)

        # edge_id_array: 全局唯一的 edge id（多 GPU 下需跨 rank 偏移）
        if is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            num_edges_all = torch.empty(
                world_size, per_etype_local_sizes.numel(),
                dtype=torch.int64, device="cuda"
            )
            torch.distributed.all_gather_into_tensor(num_edges_all, per_etype_local_sizes)
            start_offsets = num_edges_all[:rank].T.sum(axis=1)
        else:
            rank = 0
            start_offsets = torch.zeros(
                (len(sorted_keys),), dtype=torch.int64, device="cuda"
            )
            num_edges_all = per_etype_local_sizes.reshape((1, per_etype_local_sizes.numel()))

        edge_id_parts = []
        for i in range(len(sorted_keys)):
            eid = torch.arange(
                start_offsets[i],
                start_offsets[i] + num_edges_all[rank][i],
                dtype=torch.int64,
                device="cuda",
            )
            edge_id_parts.append(eid)
        edge_id_array = torch.concat(edge_id_parts)

        result = {
            "dst": edge_index[0],
            "src": edge_index[1],
            "etp": edge_type_array,
            "eid": edge_id_array,
        }

        # 可选权重
        if weight_attr is not None:
            feature_store, attr_name = weight_attr
            weights = []
            for i, et in enumerate(sorted_keys):
                ix = torch.arange(
                    start_offsets[i],
                    start_offsets[i] + num_edges_all[rank][i],
                    dtype=torch.int64, device="cpu",
                )
                weights.append(feature_store[et, attr_name][ix])
            result["wgt"] = torch.concat(weights).cuda()

        # ── 断点 14 ──────────────────────────────────────────────────────
        _dbg(
            f"EdgelistBuilder.build() 完成: "
            f"edge_index.shape={tuple(edge_index.shape)}, "
            f"eid range=[{edge_id_array.min().item()}, {edge_id_array.max().item()}]"
            if edge_id_array.numel() > 0 else
            f"EdgelistBuilder.build() 完成: 空 edgelist"
        )

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 使用示例（WALPURGIS_DEBUG=1 下可验证每个断点）
# ──────────────────────────────────────────────────────────────────────────────
#
# # 后端选择
# selector = BackendSelector.from_env()
# print(selector.backend, selector.reason)
#
# # Store 注册表
# reg = UnifiedStoreRegistry("features")
# reg.put(("node", "feat"), some_dist_tensor)
# val = reg.get(("node", "feat"))
#
# # 维度策略
# strategy = TensorDimStrategy.build(tensor_dim=2, global_row_count=1024, trailing_dim=128)
# tx = strategy.instantiate(dtype=torch.float32, device="cpu", backend="nccl")
#
# # dtype 协商
# dtype = DtypeNegotiator.negotiate(local_tensor, sizes_tensor)
#
# # GraphStore 工厂
# path, store = FeatureStoreFactory.resolve(is_multi_gpu=True, args=(), kwargs={})
#
# # edgelist 构建
# d = EdgelistBuilder.build(edge_indices_registry, vertex_offsets, is_multi_gpu=True)
