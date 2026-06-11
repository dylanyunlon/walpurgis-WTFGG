"""
embedding.py — bd703b3 迁移: WholeMemory 嵌入层

上游来源: python/pylibwholegraph/pylibwholegraph/torch/embedding.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

Walpurgis 改写20%(鲁迅拿法):
- _GradAccumulator dataclass 替代 WholeMemoryEmbedding 中散落的
  sparse_indices / sparse_grads / need_apply 三个列表字段，
  封装梯度累积生命周期，apply/reset 方法职责清晰
- WholeMemoryOptimizer.step() 加 WALPURGIS_DEBUG 输出每个 embedding 的 apply 状态
- create_builtin_cache_policy 的 WARNING print 升级为 _dbg（噪音控制）
- 全链路 WALPURGIS_DEBUG=1 断点 print: 缓存策略构建 / gather 参数 / 梯度 accumulate/apply
"""

import os
from dataclasses import dataclass, field
from typing import Union, List, Optional

import torch
import pylibwholegraph.binding.wholememory_binding as wmb

from .env_fn_utils import (
    torch_dtype_to_wholememory_dtype,
    str_to_wmb_wholememory_location,
    str_to_wmb_wholememory_memory_type,
    str_to_wmb_wholememory_optimizer_type,
    str_to_wmb_wholememory_access_type,
    get_part_file_list,
    get_part_file_name,
)
from .comm import WholeMemoryCommunicator, get_global_communicator, get_local_node_communicator, get_local_device_communicator
from .tensor import WholeMemoryTensor
from .env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream

# ──────────────────────────────────────────────
# 调试开关
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[WALPURGIS wholememory/embedding]", *args, **kwargs)


# ──────────────────────────────────────────────
# _GradAccumulator — 梯度累积生命周期管理
# ──────────────────────────────────────────────

@dataclass
class _GradAccumulator:
    """
    封装一次前向到反向之间的稀疏梯度累积。

    上游 WholeMemoryEmbedding 用三个散落属性管理:
        self.need_apply = False
        self.sparse_indices = []
        self.sparse_grads = []
    Walpurgis 集中到此处，apply() / reset() 职责清晰。
    """
    indices: List[torch.Tensor] = field(default_factory=list)
    grads: List[torch.Tensor] = field(default_factory=list)
    has_pending: bool = False

    def accumulate(self, indice: torch.Tensor, grad: torch.Tensor) -> None:
        _dbg(f"_GradAccumulator.accumulate: indice.shape={indice.shape} grad.shape={grad.shape}")
        self.indices.append(indice)
        self.grads.append(grad)
        self.has_pending = True

    def flush_tensors(self):
        """返回合并后的 (indices, grads) 并清空缓冲。"""
        assert self.has_pending, "apply 前必须有待处理梯度"
        merged_indices = torch.cat(self.indices)
        merged_grads = torch.cat(self.grads)
        _dbg(
            f"_GradAccumulator.flush_tensors: "
            f"merged_indices={merged_indices.shape} merged_grads={merged_grads.shape}"
        )
        self.reset()
        return merged_indices, merged_grads

    def reset(self) -> None:
        self.indices.clear()
        self.grads.clear()
        self.has_pending = False


# ──────────────────────────────────────────────
# WholeMemoryOptimizer
# ──────────────────────────────────────────────

class WholeMemoryOptimizer:
    """
    WholeMemoryEmbedding 的稀疏优化器。
    多个 embedding 可共享同一 optimizer。
    请使用 create_wholememory_optimizer 创建，而非直接构造。
    """

    def __init__(self, global_comm: WholeMemoryCommunicator):
        self.wmb_opt = wmb.WholeMemoryOptimizer()
        self.embeddings: List["WholeMemoryEmbedding"] = []
        self.global_comm = global_comm
        _dbg("WholeMemoryOptimizer: 创建")

    def add_embedding(self, wm_embedding: "WholeMemoryEmbedding") -> None:
        """将 embedding 注册到此 optimizer（通常由 create_wholememory_optimizer 自动调用）。"""
        assert isinstance(wm_embedding, WholeMemoryEmbedding)
        if wm_embedding.wmb_optimizer is not None:
            raise ValueError("optimizer 只能设置一次")
        wm_embedding.wmb_optimizer = self.wmb_opt
        wm_embedding.dummy_input.requires_grad_(True)
        self.wmb_opt.add_embedding(wm_embedding.wmb_embedding)
        self.embeddings.append(wm_embedding)
        _dbg(f"add_embedding: 已注册 embedding，当前数量={len(self.embeddings)}")

    def step(self, lr: float) -> None:
        """将梯度应用到所有已注册 embedding。"""
        _dbg(f"WholeMemoryOptimizer.step: lr={lr} embeddings={len(self.embeddings)}")
        for i, wm_embedding in enumerate(self.embeddings):
            _dbg(f"  embedding[{i}]: has_pending={wm_embedding._grad_accum.has_pending}")
            if wm_embedding._grad_accum.has_pending:
                wm_embedding.apply_gradients(lr)
        self.global_comm.barrier()


# ──────────────────────────────────────────────
# WholeMemoryCachePolicy
# ──────────────────────────────────────────────

class WholeMemoryCachePolicy:
    """缓存策略对象。请使用工厂函数而非直接构造。"""

    def __init__(self, wmb_cache_policy: wmb.WholeMemoryCachePolicy):
        self.wmb_cache_policy = wmb_cache_policy


def create_wholememory_cache_policy(
    cache_comm: WholeMemoryCommunicator,
    *,
    memory_type: str = "chunked",
    memory_location: str = "cuda",
    access_type: str = "readonly",
    ratio: float = 0.5,
) -> WholeMemoryCachePolicy:
    """创建自定义缓存策略。大多数情况下 create_builtin_cache_policy 已够用。"""
    _dbg(
        f"create_wholememory_cache_policy: type={memory_type} loc={memory_location} "
        f"access={access_type} ratio={ratio}"
    )
    wmb_cp = wmb.WholeMemoryCachePolicy()
    wmb_cp.create_policy(
        cache_comm.wmb_comm,
        str_to_wmb_wholememory_memory_type(memory_type),
        str_to_wmb_wholememory_location(memory_location),
        str_to_wmb_wholememory_access_type(access_type),
        ratio,
    )
    return WholeMemoryCachePolicy(wmb_cp)


def destroy_wholememory_cache_policy(cache_policy: WholeMemoryCachePolicy) -> None:
    if cache_policy is not None and cache_policy.wmb_cache_policy is not None:
        _dbg("destroy_wholememory_cache_policy")
        cache_policy.wmb_cache_policy.destroy_policy()
        cache_policy.wmb_cache_policy = None


def create_builtin_cache_policy(
    builtin_cache_type: str,
    embedding_memory_type: str,
    embedding_memory_location: str,
    access_type: str,
    cache_ratio: float,
    *,
    cache_memory_type: str = "",
    cache_memory_location: str = "",
) -> Optional[WholeMemoryCachePolicy]:
    """
    创建内置缓存策略。

    builtin_cache_type 支持:
        "none"         — 不使用缓存，返回 None
        "all_devices"  — 所有设备均缓存
        "local_node"   — 仅节点内 GPU 缓存
        "local_device" — 每 GPU 独立缓存（continuous memory type）
    """
    if embedding_memory_type not in ("continuous", "chunked", "distributed"):
        raise ValueError(f"embedding_memory_type={embedding_memory_type} 无效")
    if embedding_memory_location not in ("cpu", "cuda"):
        raise ValueError(f"embedding_memory_location={embedding_memory_location} 无效")
    if builtin_cache_type == "none":
        _dbg("create_builtin_cache_policy: type=none，返回 None")
        return None
    if cache_memory_location not in ("", "cpu", "cuda"):
        raise ValueError(f"cache_memory_location={cache_memory_location} 应为空或 cpu/cuda")
    cache_memory_location = "cuda" if cache_memory_location == "" else cache_memory_location

    _dbg(
        f"create_builtin_cache_policy: builtin={builtin_cache_type} "
        f"emb_type={embedding_memory_type} emb_loc={embedding_memory_location} "
        f"ratio={cache_ratio}"
    )

    if builtin_cache_type == "all_devices":
        if embedding_memory_location == "cuda":
            # 上游用 print("[WARNING] ...")，Walpurgis 降级为 _dbg
            _dbg(
                "[WARN] device cache on device memory: "
                "可能消耗更多显存且性能低于 no-cache"
            )
        cache_memory_type = (
            embedding_memory_type if cache_memory_type == "" else cache_memory_type
        )
        return create_wholememory_cache_policy(
            get_global_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    if builtin_cache_type == "local_node":
        cache_memory_type = "chunked" if cache_memory_type == "" else cache_memory_type
        return create_wholememory_cache_policy(
            get_local_node_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    if builtin_cache_type == "local_device":
        cache_memory_type = "continuous"
        return create_wholememory_cache_policy(
            get_local_device_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    raise ValueError(
        f"builtin_cache_type={builtin_cache_type} 不支持，"
        f"应为 none / local_device / local_node / all_devices"
    )


# ──────────────────────────────────────────────
# EmbeddingLookupFn — autograd Function
# ──────────────────────────────────────────────

class EmbeddingLookupFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indice: torch.Tensor,
        dummy_input: torch.Tensor,
        wm_embedding: "WholeMemoryEmbedding",
        is_training: bool = False,
        force_dtype: Union[torch.dtype, None] = None,
    ) -> torch.Tensor:
        output_tensor = wm_embedding.gather(
            indice, is_training=is_training, force_dtype=force_dtype
        )
        if is_training and wm_embedding.need_grad():
            ctx.save_for_backward(indice, output_tensor, dummy_input)
            ctx.wm_embedding = wm_embedding
        return output_tensor

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        indice, output_tensor, dummy_input = ctx.saved_tensors
        wm_embedding = ctx.wm_embedding
        wm_embedding.add_gradients(indice, grad_outputs)
        ctx.wm_embedding = None
        return None, torch.zeros_like(dummy_input), None, None, None


# ──────────────────────────────────────────────
# WholeMemoryEmbedding
# ──────────────────────────────────────────────

class WholeMemoryEmbedding:
    """WholeMemory 嵌入表。请使用 create_embedding / create_embedding_from_filelist 创建。"""

    def __init__(
        self,
        wmb_embedding: wmb.PyWholeMemoryEmbedding,
        wmb_cache_policy: Optional[WholeMemoryCachePolicy],
    ):
        self.wmb_embedding = wmb_embedding
        self.embedding_tensor: Optional[WholeMemoryTensor] = None
        self.optimizer_states: dict = {}
        self.wmb_cache_policy = wmb_cache_policy
        self.adjust_cache = wmb_cache_policy is not None
        self.wmb_optimizer = None
        self.dummy_input = torch.nn.Parameter(
            torch.zeros(1), requires_grad=False
        )
        # Walpurgis: 用 _GradAccumulator 替代三个散落属性
        self._grad_accum = _GradAccumulator()
        _dbg(f"WholeMemoryEmbedding: 创建 adjust_cache={self.adjust_cache}")

    def dim(self) -> int:
        return self.get_embedding_tensor().dim()

    @property
    def shape(self):
        return self.get_embedding_tensor().shape

    def set_adjust_cache(self, adjust_cache: bool) -> None:
        self.adjust_cache = adjust_cache if self.wmb_cache_policy is not None else False

    def need_grad(self) -> bool:
        return self.wmb_embedding is not None

    def gather(
        self,
        indice: torch.Tensor,
        *,
        is_training: bool = False,
        force_dtype: Union[torch.dtype, None] = None,
    ) -> torch.Tensor:
        """按 indice 从 embedding 表中聚合行。"""
        assert indice.dim() == 1
        embedding_dim = self.get_embedding_tensor().shape[1]
        embedding_count = indice.shape[0]
        current_cuda_device = f"cuda:{torch.cuda.current_device()}"
        output_dtype = (
            force_dtype if force_dtype is not None else self.embedding_tensor.dtype
        )
        need_grad = self.need_grad() and is_training
        _dbg(
            f"gather: indice.shape={indice.shape} emb_dim={embedding_dim} "
            f"dtype={output_dtype} is_training={is_training} need_grad={need_grad}"
        )
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=need_grad,
        )
        if need_grad:
            self._grad_accum.has_pending = True
        wmb.EmbeddingGatherForward(
            self.wmb_embedding,
            wrap_torch_tensor(indice),
            wrap_torch_tensor(output_tensor),
            self.adjust_cache,
            get_wholegraph_env_fns(),
            get_stream(),
        )
        return output_tensor

    def add_gradients(self, indice: torch.Tensor, grad_outputs: torch.Tensor) -> None:
        """向梯度累积器添加一次反向传播的稀疏梯度。"""
        self._grad_accum.accumulate(indice, grad_outputs)

    def apply_gradients(self, lr: float) -> None:
        """将累积梯度应用到 embedding 表。"""
        sparse_indices, sparse_grads = self._grad_accum.flush_tensors()
        _dbg(f"apply_gradients: lr={lr}")
        wmb.EmbeddingGatherGradientApply(
            self.wmb_embedding,
            wrap_torch_tensor(sparse_indices),
            wrap_torch_tensor(sparse_grads),
            self.adjust_cache,
            lr,
            get_wholegraph_env_fns(),
            get_stream(),
        )

    def writeback_all_cache(self) -> None:
        self.wmb_embedding.writeback_all_cache(get_stream(False))

    def drop_all_cache(self) -> None:
        self.wmb_embedding.drop_all_cache(get_stream(False))

    def get_embedding_tensor(self) -> WholeMemoryTensor:
        if self.embedding_tensor is None:
            self.embedding_tensor = WholeMemoryTensor(
                self.wmb_embedding.get_embedding_tensor()
            )
        return self.embedding_tensor

    def get_optimizer_state_names(self) -> List[str]:
        return self.wmb_embedding.get_optimizer_state_names()

    def get_optimizer_state(self, state_name: str) -> WholeMemoryTensor:
        if state_name not in self.optimizer_states:
            self.optimizer_states[state_name] = WholeMemoryTensor(
                self.wmb_embedding.get_optimizer_state(state_name)
            )
        return self.optimizer_states[state_name]

    def save(self, file_prefix: str) -> None:
        self.get_embedding_tensor().to_file_prefix(file_prefix + "_embedding_tensor")
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.to_file_prefix(file_prefix + "_" + state_name)

    def load(
        self,
        file_prefix: str,
        *,
        ignore_embedding: bool = False,
        part_count: Union[int, None] = None,
    ) -> None:
        if not ignore_embedding:
            self.get_embedding_tensor().from_file_prefix(
                file_prefix + "_embedding_tensor", part_count
            )
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.from_file_prefix(file_prefix + "_" + state_name, part_count)


# ──────────────────────────────────────────────
# WholeMemoryEmbeddingModule — nn.Module 包装
# ──────────────────────────────────────────────

class WholeMemoryEmbeddingModule(torch.nn.Module):
    """将 WholeMemoryEmbedding 包装为 PyTorch Module，使其参与 optimizer 的 step/backward。"""

    def __init__(self, wm_embedding: WholeMemoryEmbedding):
        super().__init__()
        self.wm_embedding = wm_embedding
        self.register_parameter("dummy_input", wm_embedding.dummy_input)

    def forward(
        self,
        indice: torch.Tensor,
        is_training: bool = False,
        force_dtype: Union[torch.dtype, None] = None,
    ) -> torch.Tensor:
        return EmbeddingLookupFn.apply(
            indice, self.dummy_input, self.wm_embedding, is_training, force_dtype
        )


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_wholememory_optimizer(
    optimizer_type: str,
    param_dict: dict,
    global_comm: WholeMemoryCommunicator,
) -> WholeMemoryOptimizer:
    opt = WholeMemoryOptimizer(global_comm)
    wmb_opt_type = str_to_wmb_wholememory_optimizer_type(optimizer_type)
    opt.wmb_opt.set_parameter(wmb_opt_type, param_dict)
    _dbg(f"create_wholememory_optimizer: type={optimizer_type}")
    return opt


def destroy_wholememory_optimizer(wm_optimizer: WholeMemoryOptimizer) -> None:
    if wm_optimizer is not None:
        _dbg("destroy_wholememory_optimizer")
        # wmb_opt 由 C++ 侧管理，Python 侧清空引用即可
        wm_optimizer.wmb_opt = None


def create_embedding(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    dtype: torch.dtype,
    sizes: List[int],
    *,
    cache_policy: Optional[WholeMemoryCachePolicy] = None,
    random_init: bool = False,
    gather_sms: int = -1,
    optimizer: Optional[WholeMemoryOptimizer] = None,
) -> WholeMemoryEmbedding:
    _dbg(
        f"create_embedding: type={memory_type} loc={memory_location} "
        f"dtype={dtype} sizes={sizes} random_init={random_init}"
    )
    wmb_cache_policy = cache_policy.wmb_cache_policy if cache_policy is not None else None
    wmb_opt = optimizer.wmb_opt if optimizer is not None else None
    wmb_embedding = wmb.create_embedding(
        comm.wmb_comm,
        str_to_wmb_wholememory_memory_type(memory_type),
        str_to_wmb_wholememory_location(memory_location),
        torch_dtype_to_wholememory_dtype(dtype),
        sizes,
        wmb_cache_policy,
        random_init,
        gather_sms,
        wmb_opt,
    )
    wm_embedding = WholeMemoryEmbedding(wmb_embedding, cache_policy)
    if optimizer is not None:
        optimizer.add_embedding(wm_embedding)
    return wm_embedding


def create_embedding_from_filelist(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: torch.dtype,
    last_dim_size: int,
    *,
    cache_policy: Optional[WholeMemoryCachePolicy] = None,
    gather_sms: int = -1,
    optimizer: Optional[WholeMemoryOptimizer] = None,
    round_robin_size: int = 0,
) -> WholeMemoryEmbedding:
    from .env_fn_utils import get_file_size
    if isinstance(filelist, str):
        filelist = [filelist]
    total_size = sum(get_file_size(f) for f in filelist)
    elem_size = torch.empty(1, dtype=dtype).element_size()
    total_elems = total_size // elem_size
    sizes = [total_elems // last_dim_size, last_dim_size]
    _dbg(f"create_embedding_from_filelist: {len(filelist)} files sizes={sizes}")
    wm_embedding = create_embedding(
        comm, memory_type, memory_location, dtype, sizes,
        cache_policy=cache_policy, gather_sms=gather_sms, optimizer=optimizer,
    )
    wm_embedding.get_embedding_tensor().from_filelist(filelist, round_robin_size)
    return wm_embedding


def destroy_embedding(wm_embedding: WholeMemoryEmbedding) -> None:
    if wm_embedding is not None and wm_embedding.wmb_embedding is not None:
        _dbg("destroy_embedding")
        wmb.destroy_embedding(wm_embedding.wmb_embedding)
        wm_embedding.wmb_embedding = None
