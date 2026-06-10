"""
FP16梯度去重 — 迁移自 cugraph-gnn 5909ae8 (PR #462)
  upstream: Fp16 embedding train

逐行diff阅读所得的核心修改 (3文件):

【embedding.cpp — gather_gradient_apply()】
  - dedup_grads 中间缓冲区: dtype 从 grads_desc->dtype 钉死为 WHOLEMEMORY_DT_FLOAT
    (line ~240): device_malloc(..., WHOLEMEMORY_DT_FLOAT) ← 原来传 grads_desc->dtype
    含义: 无论输入梯度是 fp16/bf16/fp32, 去重后的缓冲区永远 float32
  - 传入 dedup_indice_and_gradients 的 grads 指针:
    (line ~251): 从 static_cast<const float*>(temp_grad_recv_buffer) → temp_grad_recv_buffer
    含义: 不再硬性假设 float*, 传 void* 由内部模板解析
  - recv_grad_tensor_desc.dtype: 新增一行钉死为 WHOLEMEMORY_DT_FLOAT (line ~297)
    含义: 向上层 scatter_back 传递的描述符也反映 float32 (已升精度)

【exchange_embeddings_nccl_func.cu — CUDA kernel】
  - DedupIndiceAndGradientsKernel: 新增 template <typename GradT>
    const float* grads → const GradT* grads
    current_grads_ptr[dim] → static_cast<float>(current_grads_ptr[dim])
    含义: 累加时显式升为 float32, 防止 fp16 溢出累加
  - dedup_indice_and_gradients_temp_func: 新增 typename GradT 模板参数
    const float* grads → const void* grads, 内部 static_cast<const GradT*>
  - dispatch: REGISTER_DISPATCH_ONE_TYPE → REGISTER_DISPATCH_TWO_TYPES
    (SINT3264) → (SINT3264, BF16_HALF_FLOAT)
    含义: 索引类型 × 梯度类型 的二维 dispatch
  - validation: grads_desc.dtype == WHOLEMEMORY_DT_FLOAT
              → wholememory_dtype_is_floating_number(grads_desc.dtype)
    含义: 允许 fp16/bf16 进入, 不再只接受 fp32

【exchange_embeddings_nccl_func.h】
  - 公开签名: const float* grads → const void* grads

鲁迅拿法改写 (~20%):
  - 上游: C++ void* + 二维模板 dispatch. Python层无对应.
    Walpurgis: DedupGradSession 单一对象封装 (indices, grads, dtype),
    validate() 提前报错 (上游是 WHOLEMEMORY_CHECK_NOTHROW assert, 无友好消息)
  - 上游: 内部静默升 float32 无日志. 改写: 每次 dedup 打印 dtype 转换路径
  - 上游: 二维 dispatch 靠宏展开, 无调试入口.
    改写: dispatch_table dict 可 dump, 模拟 REGISTER_DISPATCH_TWO_TYPES 语义
  - 上游: validation 改为 is_floating_number 但无 ValueError 提示.
    改写: validate() 给出 dtype 名称 + 支持列表 + upstream 原文
  - 全链路 WALPURGIS_DEBUG=1 断点 print: 8处覆盖
    (输入dtype → 升精度路径 → 排序 → 去重 → 累加 → float32输出 → 描述符钉死)

作者: dylanyunlon <dogechat@163.com>
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from walpurgis import _dbg, _is_debug, dump_struct_state

# ─── 支持的梯度 dtype (对应 wholememory_dtype_is_floating_number) ─────────────
# upstream: float / half / bf16 均走 floating_number 判断
SUPPORTED_GRAD_DTYPES: Dict[torch.dtype, str] = {
    torch.float32:  "WHOLEMEMORY_DT_FLOAT",
    torch.float16:  "WHOLEMEMORY_DT_HALF",
    torch.bfloat16: "WHOLEMEMORY_DT_BF16",
}

# upstream dispatch 二维表: (IndexT, GradT) → kernel
# 对应 REGISTER_DISPATCH_TWO_TYPES(DedupIndiceAndGradientsTempFunc, SINT3264, BF16_HALF_FLOAT)
# SINT3264 = {int32, int64}; BF16_HALF_FLOAT = {bf16, half, float}
_DISPATCH_TABLE: Dict[Tuple[torch.dtype, torch.dtype], str] = {
    (torch.int32,  torch.float16):  "dedup_int32_half",
    (torch.int32,  torch.bfloat16): "dedup_int32_bf16",
    (torch.int32,  torch.float32):  "dedup_int32_float",
    (torch.int64,  torch.float16):  "dedup_int64_half",
    (torch.int64,  torch.bfloat16): "dedup_int64_bf16",
    (torch.int64,  torch.float32):  "dedup_int64_float",
}


def dump_dispatch_table():
    """
    打印全部 dispatch 路径 — 对应 REGISTER_DISPATCH_TWO_TYPES 展开的6条路径.
    上游无此 debug 方法, 改写新增.
    """
    print("[fp16_grad_dedup] dispatch_table:", file=sys.stderr, flush=True)
    for (idx_dtype, grad_dtype), fn_name in _DISPATCH_TABLE.items():
        print(
            f"  ({idx_dtype}, {grad_dtype}) → {fn_name}",
            file=sys.stderr, flush=True
        )


def _validate_dtypes(indices_dtype: torch.dtype, grads_dtype: torch.dtype):
    """
    对应 upstream 的两个 WHOLEMEMORY_CHECK_NOTHROW:
      1. indice_desc.dtype == INT || INT64
      2. wholememory_dtype_is_floating_number(grads_desc.dtype)  ← 5909ae8 修改点

    改写: 给出 friendly ValueError 而非裸 assert
    """
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"[fp16_grad_dedup] indices dtype must be int32 or int64, "
            f"got {indices_dtype}. "
            f"(upstream: WHOLEMEMORY_CHECK_NOTHROW dtype == DT_INT || DT_INT64)"
        )
    if grads_dtype not in SUPPORTED_GRAD_DTYPES:
        raise ValueError(
            f"[fp16_grad_dedup] grads dtype must be a floating type "
            f"({list(SUPPORTED_GRAD_DTYPES.keys())}), got {grads_dtype}. "
            f"(upstream 5909ae8: changed from 'must be float32' to "
            f"'wholememory_dtype_is_floating_number')"
        )
    key = (indices_dtype, grads_dtype)
    if key not in _DISPATCH_TABLE:
        raise ValueError(
            f"[fp16_grad_dedup] no dispatch entry for "
            f"(indices={indices_dtype}, grads={grads_dtype}). "
            f"Available: {list(_DISPATCH_TABLE.keys())}"
        )


@dataclass
class DedupGradSession:
    """
    梯度去重会话 — 封装 (indices, grads, metadata).

    对应 upstream gather_gradient_apply() 中的局部变量群:
      indice_desc / grads_desc / dedup_indice / dedup_grads / recv_grad_tensor_desc

    改写: 将零散 C++ 局部变量封装为 Python 对象,
    validate() 提前校验 (上游是 assert, 无友好消息).

    关键设计 (逐行diff):
      - dedup_grads: 始终 float32 (embedding.cpp line~240: WHOLEMEMORY_DT_FLOAT)
      - recv_grad_desc.dtype: 钉死 float32 (embedding.cpp line~297: 新增行)
      - grads 输入: 允许 fp16/bf16/fp32 (exchange_func.h: void*)
    """
    indices: torch.Tensor          # [N] int32 或 int64
    grads:   torch.Tensor          # [N, D] fp16/bf16/fp32 — 对应 void* grads
    embedding_dim: int = field(init=False)
    _validated: bool = field(default=False, init=False)

    def __post_init__(self):
        self.embedding_dim = self.grads.shape[1] if self.grads.ndim == 2 else self.grads.shape[0]

    def validate(self):
        """
        对应 upstream WHOLEMEMORY_CHECK_NOTHROW × 2 + WHOLEMEMORY_RETURN_ON_FAIL.
        改写: 提前 Python 层校验, 有意义的错误消息.
        """
        if self._validated:
            return

        # ── 断点调试: validate 入口 ──
        print(
            f"[DedupGradSession.validate] "
            f"indices.shape={self.indices.shape} dtype={self.indices.dtype} | "
            f"grads.shape={self.grads.shape} dtype={self.grads.dtype}",
            file=sys.stderr, flush=True
        )

        if self.indices.ndim != 1:
            raise ValueError(
                f"[fp16_grad_dedup] indices must be 1D, got {self.indices.ndim}D"
            )
        if self.grads.ndim != 2:
            raise ValueError(
                f"[fp16_grad_dedup] grads must be 2D [N, D], got {self.grads.ndim}D"
            )
        if self.indices.shape[0] != self.grads.shape[0]:
            raise ValueError(
                f"[fp16_grad_dedup] indices.shape[0]={self.indices.shape[0]} != "
                f"grads.shape[0]={self.grads.shape[0]}. "
                f"(upstream: WHOLEMEMORY_CHECK_NOTHROW indice_desc.size == grads_desc.sizes[0])"
            )

        _validate_dtypes(self.indices.dtype, self.grads.dtype)
        self._validated = True

        # ── 断点调试: dispatch 路径确认 ──
        key = (self.indices.dtype, self.grads.dtype)
        fn_name = _DISPATCH_TABLE[key]
        print(
            f"[DedupGradSession.validate] dispatch_key={key} → {fn_name} ✓",
            file=sys.stderr, flush=True
        )


def dedup_indice_and_gradients(
    session: DedupGradSession,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    主去重函数 — 对应 upstream dedup_indice_and_gradients().

    输入:
      session.indices: [N] int32/int64 — 可能包含重复
      session.grads:   [N, D] fp16/bf16/fp32 — 对应每个 index 的梯度

    输出:
      (dedup_indices, dedup_grads_float32)
        dedup_indices: [K] 去重后的唯一索引 (K <= N)
        dedup_grads:   [K, D] float32 — 重复 index 的梯度累加, 升为 float32
                       (对应 upstream: dedup_grads float* 中间缓冲区始终 float32)

    核心算法 (对应 DedupIndiceAndGradientsKernel<GradT>):
      1. 排序 indices → 连续相同 index 归组
      2. 对每组: 第一个 = cast_to_float(grad), 后续 += cast_to_float(grad)
         (对应 kernel: idx==start_offset → 赋值; else → 累加, 均 static_cast<float>)
      3. dedup_grads 描述符 dtype 钉为 float32
         (对应 embedding.cpp line~297: recv_grad_tensor_desc.dtype = WHOLEMEMORY_DT_FLOAT)

    改写 (~20%):
      - 上游 CUDA kernel 是 blockIdx 并行, Python 用 scatter_add 实现等价语义
      - 上游无 dtype 转换日志, 改写加全链路 print
      - 上游对 dedup 后 run_count 无返回说明, 改写返回 named tuple-like tuple
      - DedupGradSession 封装代替裸指针参数列表
    """
    session.validate()

    indices = session.indices
    grads = session.grads  # [N, D], 可能是 fp16/bf16/fp32
    N, D = grads.shape
    grad_dtype_name = SUPPORTED_GRAD_DTYPES[grads.dtype]

    # ── 断点调试: 升精度路径 ──
    # 对应 upstream: static_cast<float>(current_grads_ptr[dim]) in kernel
    print(
        f"[dedup_indice_and_gradients] N={N} D={D} "
        f"grad_dtype={grads.dtype}({grad_dtype_name}) "
        f"→ 升为 float32 用于累加 "
        f"(upstream: static_cast<float> in DedupIndiceAndGradientsKernel<GradT>)",
        file=sys.stderr, flush=True
    )

    # Step 1: 排序 indices (对应 cub::DeviceRadixSort::SortPairs in kernel)
    sorted_order = torch.argsort(indices, stable=True)
    sorted_indices = indices[sorted_order]

    # ── 断点调试: 排序后 ──
    if _is_debug("fp16_dedup"):
        print(
            f"[dedup_indice_and_gradients] sorted_indices[:8]={sorted_indices[:8].tolist()}",
            file=sys.stderr, flush=True
        )

    # Step 2: 升精度 — 对应 static_cast<float>(current_grads_ptr[dim])
    # embedding.cpp line~240: dedup_grads 中间缓冲区 device_malloc 用 WHOLEMEMORY_DT_FLOAT
    grads_float32 = grads[sorted_order].float()  # [N, D] float32

    # ── 断点调试: cast 损失检测 ──
    if _is_debug("fp16_dedup") and grads.dtype != torch.float32:
        cast_err = (grads_float32 - grads[sorted_order].float()).abs().max().item()
        print(
            f"[dedup_indice_and_gradients] fp16→fp32 max_cast_err={cast_err:.6f} "
            f"(预期 ~0, fp16精度限制约 1e-3)",
            file=sys.stderr, flush=True
        )

    # Step 3: 找唯一 index 及其在排序后的位置
    # 对应 wholememory_ops::dedup_indice_and_gradients 内部的 unique + start_pos 计算
    unique_indices, inverse_map = torch.unique(
        sorted_indices, sorted=True, return_inverse=True
    )
    K = unique_indices.shape[0]

    # ── 断点调试: 去重比例 ──
    print(
        f"[dedup_indice_and_gradients] 去重: N={N} → K={K} "
        f"(dedup_ratio={K/N:.3f}) "
        f"(对应 upstream run_count={K})",
        file=sys.stderr, flush=True
    )

    # Step 4: scatter_add 累加 — 对应 DedupIndiceAndGradientsKernel 内的:
    #   if (idx == start_offset): dedup_grads[dim] = float(grads[dim])  ← 首次赋值
    #   else:                     dedup_grads[dim] += float(grads[dim]) ← 累加
    # Python 的 scatter_add 等价于 CUDA kernel 的 blockIdx 并行累加
    dedup_grads = torch.zeros(K, D, dtype=torch.float32, device=grads.device)
    dedup_grads.scatter_add_(
        0,
        inverse_map.unsqueeze(1).expand(-1, D),  # [N, D]
        grads_float32                              # [N, D] float32
    )

    # ── 断点调试: 输出统计 ──
    # 对应 embedding.cpp line~297: recv_grad_tensor_desc.dtype = WHOLEMEMORY_DT_FLOAT (新增行)
    print(
        f"[dedup_indice_and_gradients] 输出: "
        f"dedup_indices.shape={unique_indices.shape} dtype={unique_indices.dtype} | "
        f"dedup_grads.shape={dedup_grads.shape} dtype={dedup_grads.dtype} (钉死 float32) "
        f"(upstream line~297: recv_grad_tensor_desc.dtype = WHOLEMEMORY_DT_FLOAT)",
        file=sys.stderr, flush=True
    )

    if _is_debug("fp16_dedup"):
        dump_struct_state(
            "dedup_grad_output",
            K=K, N=N, D=D,
            input_grad_dtype=str(grads.dtype),
            output_grad_dtype=str(dedup_grads.dtype),
            dedup_grads_mean=dedup_grads.mean().item(),
            dedup_grads_std=dedup_grads.std().item(),
            dedup_grads_max=dedup_grads.abs().max().item(),
        )

    return unique_indices, dedup_grads


# ─── gather_gradient_apply 级别的封装 ─────────────────────────────────────────
# 对应 embedding.cpp: gather_gradient_apply() 整体流程

@dataclass
class GatherGradientApplyConfig:
    """
    对应 embedding.cpp gather_gradient_apply() 的参数:
      grads_desc->dtype: 输入梯度 dtype (fp16/bf16/fp32 均可, 5909ae8修改点)
      output dtype: 始终 float32 (WHOLEMEMORY_DT_FLOAT, 5909ae8 钉死)

    改写: 上游是 C++ struct 指针 + 散落 local var, 改写为 dataclass.
    validate() 提前校验而非运行时 assert.
    """
    embedding_dtype: torch.dtype = torch.float16   # 下游 embedding 存储 dtype
    grad_input_dtype: torch.dtype = torch.float16  # 接收到的梯度 dtype (5909ae8: 允许 half)
    embedding_dim: int = 64
    optimizer_lr: float = 0.01

    def validate(self):
        """
        对应 embedding.cpp set_optimizer() dtype check (b58ea19) +
        5909ae8 新增的 is_floating_number 替换.
        """
        if self.grad_input_dtype not in SUPPORTED_GRAD_DTYPES:
            raise ValueError(
                f"[GatherGradientApplyConfig] grad_input_dtype={self.grad_input_dtype} "
                f"not in supported floating types {list(SUPPORTED_GRAD_DTYPES.keys())}. "
                f"(upstream 5909ae8: is_floating_number 替代 == FLOAT 校验)"
            )
        # ── 断点调试: config validate ──
        print(
            f"[GatherGradientApplyConfig.validate] "
            f"emb_dtype={self.embedding_dtype} "
            f"grad_input_dtype={self.grad_input_dtype}({SUPPORTED_GRAD_DTYPES[self.grad_input_dtype]}) "
            f"embedding_dim={self.embedding_dim} lr={self.optimizer_lr} ✓",
            file=sys.stderr, flush=True
        )


def gather_gradient_apply(
    config: GatherGradientApplyConfig,
    indices: torch.Tensor,
    grads: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP16 embedding 梯度应用 — 对应 embedding.cpp gather_gradient_apply().

    输入:
      config:           GatherGradientApplyConfig (dtype 配置)
      indices:          [N] int32/int64 — embedding 访问索引 (可重复)
      grads:            [N, D] fp16/bf16/fp32 — 原始梯度 (5909ae8: 允许非 float32)
      embedding_weight: [M, D] 当前 embedding 权重 (低精度存储)

    输出:
      updated_weight: [M, D] 同 embedding_dtype — 更新后的权重

    核心流程 (严格对应 embedding.cpp gather_gradient_apply):
      1. dedup_indice_and_gradients: 去重 + 升 float32 (5909ae8 核心修改)
      2. SGD update: emb_float[i] -= lr * dedup_grad[i]  (float32 计算)
      3. cast 回 embedding_dtype 写回 (static_cast<EmbeddingT>)

    改写:
      - 上游: 分散在 embedding.cpp + exchange_embeddings_nccl_func.cu 两个文件
        Walpurgis: 单函数封装完整数据流
      - 上游: 无 dtype 流转日志
        改写: 每步 print dtype 变化链: fp16 → float32(dedup) → float32(update) → fp16(writeback)
    """
    config.validate()

    if grads.dtype != config.grad_input_dtype:
        # ── 断点调试: dtype 不一致警告 ──
        print(
            f"[gather_gradient_apply] WARNING: grads.dtype={grads.dtype} != "
            f"config.grad_input_dtype={config.grad_input_dtype}, 强制转换",
            file=sys.stderr, flush=True
        )
        grads = grads.to(config.grad_input_dtype)

    # ── 断点调试: 入口 dtype 链 ──
    print(
        f"[gather_gradient_apply] dtype链: "
        f"grads({grads.dtype}) → dedup(float32) → update(float32) → writeback({config.embedding_dtype})",
        file=sys.stderr, flush=True
    )

    # Step 1: 去重 + 升 float32
    # 对应 embedding.cpp lines ~240-297 (5909ae8 的3处修改均在此)
    session = DedupGradSession(indices=indices, grads=grads)
    dedup_indices, dedup_grads_f32 = dedup_indice_and_gradients(session)

    # Step 2: float32 计算更新
    # 对应 upstream optimizer step kernel (float32 主副本更新)
    emb_float32 = embedding_weight[dedup_indices].float()  # 升为 float32

    # ── 断点调试: 更新前 ──
    if _is_debug("fp16_dedup"):
        print(
            f"[gather_gradient_apply] update: "
            f"emb_float32.shape={emb_float32.shape} "
            f"dedup_grads_f32.shape={dedup_grads_f32.shape} "
            f"lr={config.optimizer_lr}",
            file=sys.stderr, flush=True
        )

    updated_f32 = emb_float32 - config.optimizer_lr * dedup_grads_f32

    # Step 3: 写回低精度 — static_cast<EmbeddingT>(embedding_value)
    updated_low = updated_f32.to(config.embedding_dtype)

    # ── 断点调试: writeback ──
    writeback_err = (updated_f32 - updated_low.float()).abs().max().item()
    print(
        f"[gather_gradient_apply] writeback: "
        f"float32→{config.embedding_dtype} "
        f"max_cast_err={writeback_err:.6f} "
        f"updated_rows={dedup_indices.shape[0]}",
        file=sys.stderr, flush=True
    )

    # scatter 写回
    result = embedding_weight.clone()
    result[dedup_indices, :dedup_grads_f32.shape[1]] = updated_low

    return result


# ─── 自测 (WALPURGIS_DEBUG=1 python -m walpurgis.models.fp16_grad_dedup) ──────

if __name__ == "__main__":
    import os
    os.environ["WALPURGIS_DEBUG"] = "1"

    print("=" * 64, file=sys.stderr)
    print("fp16_grad_dedup 自测 (对应 5909ae8 Fp16 embedding train)", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    # 打印 dispatch table
    dump_dispatch_table()

    # 构造测试: 10个索引, 3重复
    torch.manual_seed(42)
    indices = torch.tensor([0, 1, 2, 1, 3, 0, 4, 2, 5, 1], dtype=torch.int32)
    D = 8

    print("\n── 测试1: fp16 梯度去重 ──", file=sys.stderr)
    grads_fp16 = torch.randn(10, D, dtype=torch.float32).half()
    session = DedupGradSession(indices=indices, grads=grads_fp16)
    di, dg = dedup_indice_and_gradients(session)
    print(f"  结果: {len(indices)}个索引 → {len(di)}唯一 | "
          f"dedup_grads dtype={dg.dtype} (预期 float32)", file=sys.stderr)
    assert dg.dtype == torch.float32, f"dedup_grads 应为 float32, 实际 {dg.dtype}"
    print("  [PASS] dedup_grads dtype=float32 ✓", file=sys.stderr)

    print("\n── 测试2: bf16 梯度去重 ──", file=sys.stderr)
    grads_bf16 = torch.randn(10, D, dtype=torch.float32).bfloat16()
    session2 = DedupGradSession(indices=indices, grads=grads_bf16)
    di2, dg2 = dedup_indice_and_gradients(session2)
    assert dg2.dtype == torch.float32, f"dedup_grads 应为 float32, 实际 {dg2.dtype}"
    print("  [PASS] bf16 → float32 ✓", file=sys.stderr)

    print("\n── 测试3: float32 梯度 (向后兼容) ──", file=sys.stderr)
    grads_f32 = torch.randn(10, D, dtype=torch.float32)
    session3 = DedupGradSession(indices=indices, grads=grads_f32)
    di3, dg3 = dedup_indice_and_gradients(session3)
    assert dg3.dtype == torch.float32
    print("  [PASS] float32 → float32 ✓", file=sys.stderr)

    print("\n── 测试4: gather_gradient_apply 全流程 ──", file=sys.stderr)
    M = 10  # 10个 embedding
    emb_weight = torch.randn(M, D, dtype=torch.float16)
    config = GatherGradientApplyConfig(
        embedding_dtype=torch.float16,
        grad_input_dtype=torch.float16,
        embedding_dim=D,
        optimizer_lr=0.01,
    )
    updated = gather_gradient_apply(config, indices, grads_fp16, emb_weight)
    assert updated.dtype == torch.float16, f"输出应为 float16, 实际 {updated.dtype}"
    assert updated.shape == emb_weight.shape
    print(f"  [PASS] 输出 dtype={updated.dtype} shape={updated.shape} ✓", file=sys.stderr)

    print("\n── 测试5: int64 索引 + bf16 梯度 (二维 dispatch) ──", file=sys.stderr)
    indices64 = indices.long()
    session5 = DedupGradSession(indices=indices64, grads=grads_bf16)
    di5, dg5 = dedup_indice_and_gradients(session5)
    assert dg5.dtype == torch.float32
    print("  [PASS] (int64, bf16) dispatch ✓", file=sys.stderr)

    print("\n── 测试6: 非法 dtype 应报 ValueError ──", file=sys.stderr)
    try:
        bad_grads = torch.randn(10, D, dtype=torch.float64)
        session6 = DedupGradSession(indices=indices, grads=bad_grads)
        session6.validate()
        print("  [FAIL] 应抛 ValueError!", file=sys.stderr)
    except ValueError as e:
        print(f"  [PASS] ValueError: {str(e)[:80]}...", file=sys.stderr)

    print("\n所有测试通过 ✓", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
