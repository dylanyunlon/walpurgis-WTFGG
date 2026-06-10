"""
混合精度Embedding训练支持 — 迁移自 cugraph-gnn b58ea19
  upstream: support embedding training with bf16 and fp16

迁移改写 (~20%):
  - 上游: C++ CUDA kernel 模板化 (EmbeddingT = float/half/bf16),
    optimizer states 固定 float32
  - Walpurgis改写: Python层 MixedPrecisionEmbedding 包装器,
    embedding权重以 fp16/bf16 存储, SGD/Adam/AdaGrad/RMSProp
    optimizer states 保持 float32 高精度主副本
  - 与上游一致: 前向gather用低精度, backward/update用float32计算,
    写回前cast回低精度
  - 融入Cascade诊断体系: dump_struct_state + _dbg 断点

上游关键设计点 (逐行读diff所得):
  1. embedding.cpp: set_optimizer() 从"仅float"改为 float|half|bf16
  2. 新增 cachable_state_desc.dtype = WHOLEMEMORY_DT_FLOAT —— optimizer
     states 始终用 float32, 与embedding dtype解耦 (关键!)
  3. 所有4个optimizer kernel (SGD/LazyAdam/AdaGrad/RMSProp) 变成双类型模板:
     IndiceT x EmbeddingT; embedding读写加 static_cast<float>/static_cast<EmbeddingT>
  4. align_count从硬编码4改为 16/sizeof(EmbeddingT) (fp16→8, bf16→8, fp32→4)
  5. gather/scatter dispatch: HALF_FLOAT_DOUBLE → ALLFLOAT (加入bf16)
  6. DlPack binding bug修复: DtHalf分支写了两次, bf16分支永远不可达 (已修复)

作者: dylanyunlon <dogechat@163.com>
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from walpurgis import _dbg, _is_debug, dump_struct_state

# ─── 支持的embedding dtype ─────────────────────────────────────
SUPPORTED_EMB_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# 上游对应: wholememory_dtype_get_element_size + align_count = 16/element_size
# fp32: 4字节 → align=4; fp16/bf16: 2字节 → align=8
_DTYPE_ALIGN = {
    torch.float32: 4,
    torch.float16: 8,
    torch.bfloat16: 8,
}

# 精度容差 (对应upstream test中的atol/rtol设置)
_DTYPE_TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def _check_emb_dtype(dtype: torch.dtype):
    """dtype合法性校验 — 对应 embedding.cpp set_optimizer() 中的dtype check"""
    if dtype not in _DTYPE_ALIGN:
        raise ValueError(
            f"[Walpurgis] Only float32, float16, bfloat16 embeddings support training. "
            f"Got: {dtype}. "
            f"(upstream: 'Only float, half and bf16 embeddings support training.')"
        )


def _pad_dim(embedding_dim: int, dtype: torch.dtype) -> int:
    """
    计算padded embedding dim — 对应upstream align_count逻辑:
      align_count = 16 / emb_element_size
      padded_dim  = round_up(embedding_dim, align_count)
    """
    align = _DTYPE_ALIGN[dtype]
    return ((embedding_dim + align - 1) // align) * align


class MixedPrecisionEmbedding(nn.Module):
    """
    混合精度Embedding: 权重以 fp16/bf16 存储节省显存,
    optimizer states(m, v, sum等)保持 float32 高精度.

    设计对应关系 (upstream → Walpurgis):
      local_embedding_ptr (EmbeddingT*)  → self.weight (fp16/bf16)
      per_element_local_embedding_ptr (float*)  → self._state_* (float32)
      static_cast<float>(embedding_ptr[i])  → weight.float() 用于计算
      static_cast<EmbeddingT>(embedding_value)  → .to(emb_dtype) 写回

    断点: 设置 WALPURGIS_DEBUG=1 查看每次optimizer step的精度信息
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        emb_dtype: torch.dtype = torch.float32,
        padding_idx: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()
        _check_emb_dtype(emb_dtype)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_dtype = emb_dtype
        self.padding_idx = padding_idx

        # 上游: padded_embedding_dim = round_up(embedding_dim, align_count)
        self._padded_dim = _pad_dim(embedding_dim, emb_dtype)

        # Walpurgis改写: 使用 nn.Parameter 而非原始指针;
        # 以低精度存储, float32主副本在optimizer states中
        # 对应: local_embedding_ptr (EmbeddingT*)
        weight_data = torch.empty(num_embeddings, self._padded_dim,
                                  dtype=emb_dtype)
        nn.init.xavier_uniform_(weight_data.float().view(num_embeddings, -1)
                                 ).to(emb_dtype)  # 以fp32初始化后cast, 避免精度损失
        self.weight = nn.Parameter(weight_data)

        # ── 断点调试: 初始化完成 ──
        print(
            f"[MixedPrecisionEmbedding] init: "
            f"num_embeddings={num_embeddings} embedding_dim={embedding_dim} "
            f"padded_dim={self._padded_dim} emb_dtype={emb_dtype} "
            f"weight_bytes={weight_data.nbytes / 1024:.1f}KB",
            file=sys.stderr, flush=True
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather: 对应upstream的 gather_floating_int32/64_func
        低精度lookup后转float32用于后续计算 (训练时)
        推理时直接返回低精度节省带宽
        """
        # ── 断点调试: gather前 ──
        _dbg("mp_emb_gather_indices", indices, "mp_emb")

        # 只取有效dim (去掉padding)
        out = F.embedding(
            indices,
            self.weight[:, :self.embedding_dim],
            padding_idx=self.padding_idx,
        )

        # 训练时: 转float32用于梯度计算 (上游的 static_cast<float> 对应)
        if self.training and self.emb_dtype != torch.float32:
            out = out.float()

        # ── 断点调试: gather后 ──
        if _is_debug("mp_emb"):
            dump_struct_state(
                "mp_emb_gather_out",
                indices_shape=indices.shape,
                out_dtype=str(out.dtype),
                out_shape=out.shape,
                out_mean=out.float().mean().item(),
                out_std=out.float().std().item(),
            )

        return out

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padded_dim={self._padded_dim}, "
            f"emb_dtype={self.emb_dtype}"
        )


# ─── Optimizer States (float32主副本) ─────────────────────────────────────────
# 对应upstream: cachable_state_desc.dtype = WHOLEMEMORY_DT_FLOAT (float32)
# 无论embedding本身是何dtype, states始终是float32

class _MixedPrecisionOptimizerBase:
    """
    混合精度Embedding Optimizer基类.
    对应upstream:
      optimizer states: float* (always float32)
      embedding write: static_cast<EmbeddingT>(computed_float_value)
    """

    def __init__(self, embedding: MixedPrecisionEmbedding,
                 lr: float = 0.01, weight_decay: float = 0.0):
        _check_emb_dtype(embedding.emb_dtype)
        self.embedding = embedding
        self.lr = lr
        self.weight_decay = weight_decay
        self._step_count = 0

        # float32 optimizer states — 对应 per_element_local_embedding_ptr (float*)
        # 子类按需初始化
        self._float32_master: Optional[torch.Tensor] = None

        atol, rtol = _DTYPE_TOLERANCE[embedding.emb_dtype]
        self._atol = atol
        self._rtol = rtol

        print(
            f"[{self.__class__.__name__}] init: "
            f"emb_dtype={embedding.emb_dtype} lr={lr} wd={weight_decay} "
            f"tolerance=(atol={atol}, rtol={rtol})",
            file=sys.stderr, flush=True
        )

    def _get_float32_weight(self) -> torch.Tensor:
        """获取float32主副本用于计算 — 对应 static_cast<float>(embedding_ptr[i])"""
        return self.embedding.weight.data.float()[:, :self.embedding.embedding_dim]

    def _write_back(self, updated_float: torch.Tensor, indices: torch.Tensor):
        """
        将float32计算结果写回低精度embedding — 对应:
          embedding_ptr[embedding_idx] = static_cast<EmbeddingT>(embedding_value)
        """
        dtype = self.embedding.emb_dtype
        # cast回低精度
        updated_low = updated_float.to(dtype)

        # ── 断点调试: 写回精度损失 ──
        if _is_debug("mp_opt"):
            delta = (updated_float - updated_low.float()).abs().max().item()
            dump_struct_state(
                f"mp_opt_writeback_step{self._step_count}",
                dtype=str(dtype),
                indices_count=indices.numel(),
                max_cast_delta=delta,
                updated_float_mean=updated_float.mean().item(),
                updated_float_std=updated_float.std().item(),
            )

        # scatter写回: 只更新被索引的行
        unique_indices = indices.unique()
        self.embedding.weight.data[unique_indices, :self.embedding.embedding_dim] = \
            updated_low[unique_indices]

    def step(self, indices: torch.Tensor, grads: torch.Tensor):
        """
        Optimizer step — 子类实现.
        对应upstream kernel launch: block_count=indice_count
        """
        raise NotImplementedError

    def zero_grad(self):
        """清除embedding weight的梯度"""
        if self.embedding.weight.grad is not None:
            self.embedding.weight.grad.zero_()


class SGDEmbeddingOptimizer(_MixedPrecisionOptimizerBase):
    """
    SGD Embedding Optimizer — 对应 sgd_optimizer_step_kernel<IndiceT, EmbeddingT>

    上游公式 (embedding.cpp):
      grad_value += weight_decay * embedding_value
      embedding_value -= lr * grad_value
      embedding_ptr[i] = static_cast<EmbeddingT>(embedding_value)

    Walpurgis改写: 用 float32 计算全程, 写回时cast
    """

    def step(self, indices: torch.Tensor, grads: torch.Tensor):
        """
        indices: [B] 或 [B, L] 的整数索引
        grads: [B, embedding_dim] float32 梯度
        """
        if grads.dtype != torch.float32:
            grads = grads.float()

        # 上游: float embedding_value = static_cast<float>(embedding_ptr[i])
        emb_float = self._get_float32_weight()

        flat_indices = indices.view(-1)
        flat_grads = grads.view(-1, self.embedding.embedding_dim)

        # ── 断点调试: step入口 ──
        print(
            f"[SGDEmbeddingOptimizer] step={self._step_count} "
            f"indices={flat_indices.shape} grads={flat_grads.shape} "
            f"lr={self.lr} wd={self.weight_decay}",
            file=sys.stderr, flush=True
        )

        # 上游SGD kernel逐index处理:
        # grad_value += weight_decay * embedding_value
        # embedding_value -= lr * grad_value
        current = emb_float[flat_indices]  # [B, D]
        grad_with_wd = flat_grads + self.weight_decay * current
        updated = current - self.lr * grad_with_wd  # [B, D]

        # 写回低精度 — static_cast<EmbeddingT>(embedding_value)
        self._write_back(updated, flat_indices)
        self._step_count += 1


class AdamEmbeddingOptimizer(_MixedPrecisionOptimizerBase):
    """
    Lazy Adam Embedding Optimizer — 对应 lazy_adam_optimizer_step_kernel

    上游公式:
      m = beta1*m + (1-beta1)*grad
      v = beta2*v + (1-beta2)*grad^2
      mhat = m / (1 - beta1^t)
      vhat = v / (1 - beta2^t)
      embedding_value -= lr * mhat / (sqrt(vhat) + epsilon)
      embedding_ptr[i] = static_cast<EmbeddingT>(embedding_value)

    关键: m/v states 始终 float32 (对应 per_element_local_embedding_ptr float*)
    Walpurgis改写: per-entry beta power 用全局beta^t近似 (lazy adam语义),
                   per-entry state存在 _state_m / _state_v 中
    """

    def __init__(self, embedding: MixedPrecisionEmbedding,
                 lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0.0,
                 adam_w: bool = False):
        super().__init__(embedding, lr=lr, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.adam_w = adam_w

        N = embedding.num_embeddings
        D = embedding.embedding_dim
        device = embedding.weight.device

        # float32 optimizer states — 对应 per_element_local_embedding_ptr (float*)
        # upstream存储m和v交叉 [N, 2*D], 这里分开存储更清晰
        self._state_m = torch.zeros(N, D, dtype=torch.float32, device=device)
        self._state_v = torch.zeros(N, D, dtype=torch.float32, device=device)
        # per-entry beta power (lazy adam: 只更新被访问的entry)
        self._beta1t = torch.ones(N, dtype=torch.float32, device=device)
        self._beta2t = torch.ones(N, dtype=torch.float32, device=device)

        print(
            f"[AdamEmbeddingOptimizer] states: "
            f"m={self._state_m.shape} v={self._state_v.shape} "
            f"dtype=float32 (fixed, per upstream design) "
            f"adam_w={adam_w}",
            file=sys.stderr, flush=True
        )

    def step(self, indices: torch.Tensor, grads: torch.Tensor):
        if grads.dtype != torch.float32:
            grads = grads.float()

        flat_indices = indices.view(-1)
        flat_grads = grads.view(-1, self.embedding.embedding_dim)

        # ── 断点调试: step入口 ──
        print(
            f"[AdamEmbeddingOptimizer] step={self._step_count} "
            f"adam_w={self.adam_w} indices={flat_indices.shape} "
            f"lr={self.lr} b1={self.beta1} b2={self.beta2}",
            file=sys.stderr, flush=True
        )

        # 上游: float embedding_value = static_cast<float>(embedding_ptr[i])
        emb_float = self._get_float32_weight()
        current = emb_float[flat_indices]  # [B, D] float32

        # AdamW: 权重衰减先于梯度更新 (decoupled)
        if self.adam_w:
            current = current - self.lr * self.weight_decay * current
        else:
            flat_grads = flat_grads + self.weight_decay * current

        # lazy adam: 只更新被访问的indices的beta power
        self._beta1t[flat_indices] *= self.beta1
        self._beta2t[flat_indices] *= self.beta2

        # m/v states 更新 (float32)
        m = self._state_m[flat_indices]
        v = self._state_v[flat_indices]
        m = self.beta1 * m + (1 - self.beta1) * flat_grads
        v = self.beta2 * v + (1 - self.beta2) * flat_grads ** 2
        self._state_m[flat_indices] = m
        self._state_v[flat_indices] = v

        # bias correction
        beta1t = self._beta1t[flat_indices].unsqueeze(1)
        beta2t = self._beta2t[flat_indices].unsqueeze(1)
        mhat = m / (1 - beta1t)
        vhat = v / (1 - beta2t)

        # embedding更新 (float32计算)
        updated = current - self.lr * mhat / (vhat.sqrt() + self.epsilon)

        # ── 断点调试: m/v统计 ──
        if _is_debug("mp_opt"):
            dump_struct_state(
                f"adam_step{self._step_count}",
                m_mean=m.mean().item(),
                v_mean=v.mean().item(),
                mhat_mean=mhat.mean().item(),
                vhat_mean=vhat.mean().item(),
                updated_mean=updated.mean().item(),
                updated_std=updated.std().item(),
            )

        # 写回低精度 — static_cast<EmbeddingT>(embedding_value)
        self._write_back(updated, flat_indices)
        self._step_count += 1


class AdaGradEmbeddingOptimizer(_MixedPrecisionOptimizerBase):
    """
    AdaGrad Embedding Optimizer — 对应 ada_grad_optimizer_step_kernel

    上游公式:
      grad_value += weight_decay * embedding_value
      state_sum += grad_value * grad_value
      embedding_value -= lr * grad_value / (sqrt(state_sum) + epsilon)
      embedding_ptr[i] = static_cast<EmbeddingT>(embedding_value)

    state_sum 始终 float32
    """

    def __init__(self, embedding: MixedPrecisionEmbedding,
                 lr: float = 0.01, epsilon: float = 1e-10,
                 weight_decay: float = 0.0):
        super().__init__(embedding, lr=lr, weight_decay=weight_decay)
        self.epsilon = epsilon
        N = embedding.num_embeddings
        D = embedding.embedding_dim
        device = embedding.weight.device
        # float32 optimizer state — 对应 per_element_local_embedding_ptr (float*)
        self._state_sum = torch.zeros(N, D, dtype=torch.float32, device=device)

        print(
            f"[AdaGradEmbeddingOptimizer] state_sum: "
            f"shape={self._state_sum.shape} dtype=float32",
            file=sys.stderr, flush=True
        )

    def step(self, indices: torch.Tensor, grads: torch.Tensor):
        if grads.dtype != torch.float32:
            grads = grads.float()

        flat_indices = indices.view(-1)
        flat_grads = grads.view(-1, self.embedding.embedding_dim)

        print(
            f"[AdaGradEmbeddingOptimizer] step={self._step_count} "
            f"indices={flat_indices.shape} lr={self.lr}",
            file=sys.stderr, flush=True
        )

        emb_float = self._get_float32_weight()
        current = emb_float[flat_indices]

        grad_wd = flat_grads + self.weight_decay * current
        state_sum = self._state_sum[flat_indices]
        state_sum = state_sum + grad_wd ** 2
        self._state_sum[flat_indices] = state_sum

        updated = current - self.lr * grad_wd / (state_sum.sqrt() + self.epsilon)

        if _is_debug("mp_opt"):
            dump_struct_state(
                f"adagrad_step{self._step_count}",
                state_sum_mean=state_sum.mean().item(),
                grad_wd_norm=grad_wd.norm().item(),
                updated_mean=updated.mean().item(),
            )

        self._write_back(updated, flat_indices)
        self._step_count += 1


class RMSPropEmbeddingOptimizer(_MixedPrecisionOptimizerBase):
    """
    RMSProp Embedding Optimizer — 对应 rms_prop_optimizer_step_kernel

    上游公式:
      grad_value += weight_decay * embedding_value
      v = alpha*v + (1-alpha)*grad_value^2
      embedding_value -= lr * grad_value / (sqrt(v) + epsilon)
      embedding_ptr[i] = static_cast<EmbeddingT>(embedding_value)

    v 始终 float32
    """

    def __init__(self, embedding: MixedPrecisionEmbedding,
                 lr: float = 0.01, alpha: float = 0.99,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(embedding, lr=lr, weight_decay=weight_decay)
        self.alpha = alpha
        self.epsilon = epsilon
        N = embedding.num_embeddings
        D = embedding.embedding_dim
        device = embedding.weight.device
        # float32 optimizer state
        self._state_v = torch.zeros(N, D, dtype=torch.float32, device=device)

        print(
            f"[RMSPropEmbeddingOptimizer] state_v: "
            f"shape={self._state_v.shape} dtype=float32 alpha={alpha}",
            file=sys.stderr, flush=True
        )

    def step(self, indices: torch.Tensor, grads: torch.Tensor):
        if grads.dtype != torch.float32:
            grads = grads.float()

        flat_indices = indices.view(-1)
        flat_grads = grads.view(-1, self.embedding.embedding_dim)

        print(
            f"[RMSPropEmbeddingOptimizer] step={self._step_count} "
            f"indices={flat_indices.shape} lr={self.lr} alpha={self.alpha}",
            file=sys.stderr, flush=True
        )

        emb_float = self._get_float32_weight()
        current = emb_float[flat_indices]

        grad_wd = flat_grads + self.weight_decay * current
        v = self._state_v[flat_indices]
        v = self.alpha * v + (1 - self.alpha) * grad_wd ** 2
        self._state_v[flat_indices] = v

        updated = current - self.lr * grad_wd / (v.sqrt() + self.epsilon)

        if _is_debug("mp_opt"):
            dump_struct_state(
                f"rmsprop_step{self._step_count}",
                v_mean=v.mean().item(),
                v_max=v.max().item(),
                updated_mean=updated.mean().item(),
            )

        self._write_back(updated, flat_indices)
        self._step_count += 1


# ─── 工厂函数 ──────────────────────────────────────────────────────────────────

def make_mixed_precision_embedding(
    num_embeddings: int,
    embedding_dim: int,
    emb_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    optimizer: Literal["sgd", "adam", "adagrad", "rmsprop"] = "adam",
    **opt_kwargs,
) -> tuple:
    """
    创建混合精度Embedding + 对应的Optimizer.

    Walpurgis用法示例 (对应model.py中的adaptive_embedding):
      >>> emb, opt = make_mixed_precision_embedding(
      ...     num_embeddings=207,  # METR-LA节点数
      ...     embedding_dim=64,
      ...     emb_dtype="bfloat16",
      ...     optimizer="adam",
      ...     lr=0.001,
      ... )

    对应upstream: REGISTER_DISPATCH_TWO_TYPES(..., SINT3264, BF16_HALF_FLOAT)
    返回: (MixedPrecisionEmbedding, optimizer_instance)
    """
    dtype_torch = SUPPORTED_EMB_DTYPES.get(emb_dtype)
    if dtype_torch is None:
        raise ValueError(
            f"emb_dtype must be one of {list(SUPPORTED_EMB_DTYPES.keys())}, "
            f"got '{emb_dtype}'"
        )

    print(
        f"[make_mixed_precision_embedding] "
        f"num_embeddings={num_embeddings} dim={embedding_dim} "
        f"dtype={emb_dtype} optimizer={optimizer}",
        file=sys.stderr, flush=True
    )

    emb = MixedPrecisionEmbedding(num_embeddings, embedding_dim, dtype_torch)

    opt_map = {
        "sgd": SGDEmbeddingOptimizer,
        "adam": AdamEmbeddingOptimizer,
        "adagrad": AdaGradEmbeddingOptimizer,
        "rmsprop": RMSPropEmbeddingOptimizer,
    }
    if optimizer not in opt_map:
        raise ValueError(
            f"optimizer must be one of {list(opt_map.keys())}, got '{optimizer}'"
        )

    opt_instance = opt_map[optimizer](emb, **opt_kwargs)
    return emb, opt_instance


# ─── DlPack bug注: ───────────────────────────────────────────────────────────
# upstream wholememory_binding.pyx 修复了一个 dead code bug:
#   old:
#     elif self.data_type == DtFloat or DtDouble or DtHalf:
#         dtype.code = kDLFloat
#     elif self.data_type == DtHalf:   # ← 永远不可达! DtHalf已在上方处理
#         dtype.code = kDLBfloat
#   fixed:
#     elif ... DtHalf:
#         dtype.code = kDLFloat
#     elif self.data_type == DtBF16:   # ← 正确: bf16走bfloat分支
#         dtype.code = kDLBfloat
# Walpurgis Python层: 下面确保类似逻辑的正确性 (torch.Tensor.numpy() dtype映射)

def tensor_to_dlpack_dtype_code(t: torch.Tensor) -> str:
    """
    返回张量对应的DlPack dtype code字符串.
    修复了上游 DtHalf 分支被 DtFloat/DtDouble/DtHalf 合并判断覆盖的bug
    (参见 wholememory_binding.pyx diff).

    ── 断点调试: bf16映射验证 ──
    """
    dtype = t.dtype
    # 注意顺序: bf16必须在half之前单独判断, 否则会fallthrough (upstream bug的教训)
    if dtype == torch.bfloat16:
        code = "kDLBfloat"  # 对应 upstream 修复后的 DtBF16 → kDLBfloat
    elif dtype in (torch.float16, torch.float32, torch.float64):
        code = "kDLFloat"
    elif dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
        code = "kDLInt"
    else:
        code = "kDLOpaque"

    print(
        f"[tensor_to_dlpack_dtype_code] dtype={dtype} → {code}",
        file=sys.stderr, flush=True
    )
    return code
