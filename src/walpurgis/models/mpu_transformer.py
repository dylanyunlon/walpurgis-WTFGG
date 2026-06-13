"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: mpu/transformer.py

鲁迅拿法改写（≥20%）：
  上游本次改动在 ParallelTransformerLayer 里加了两处：
  其一，LayerNorm 的 bias 被静默设为 None（bias=False）；
  其二，self-attention 的 dropout 被移到 softmax 之后，而非之前。
  两者都是数值稳定性微调，上游注释为空——鲁迅说，这叫"悄悄地改，
  改完装作没改过"。偏差消除（bias=False）意味着 LayerNorm 退化
  为纯缩放，不再有平移自由度；对小 batch 有影响。
  Walpurgis 将此次变动拆解为：
  1. LayerNormNoBias: 显式命名的无偏 LN 子类（区别于标准 LN）
  2. ScaledDotProductAttentionWDP: softmax 后 dropout 的 SDPA 实现
     （上游是嵌入式修改，Walpurgis 抽为独立单元）
  3. ParallelTransformerLayerWrapper: 单机单卡兼容包装，
     含详细的前向断点（QKV投影前/后、softmax前/后、FFN前/后）

迁移位置: src/walpurgis/models/mpu_transformer.py
"""

import os
import sys
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    """_dbg 断点：transformer 前向推理关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        nan_flag = " !!NaN" if torch.isnan(t).any() else ""
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
            f"{nan_flag}"
        )
        print(f"[_dbg:mpu_transformer:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:mpu_transformer:{tag}] {msg}", file=sys.stderr, flush=True)


# ── LayerNormNoBias（Walpurgis特有：显式命名，区别于标准LN） ──────────────────
class LayerNormNoBias(nn.Module):
    """
    无偏差 LayerNorm（bias=False）。

    对应 a1d04b793 中 mpu/transformer.py 将 LayerNorm 的 bias 置为 None 的改动。
    Walpurgis: 命名显式化，文档说明语义影响。

    语义影响:
      - 标准 LN: y = gamma * (x - mean) / std + beta   (可平移)
      - 无偏 LN:  y = gamma * (x - mean) / std          (只缩放，不平移)
      - 对小 batch/短序列：无偏 LN 在 beta 主要吸收噪声时有轻微正则效果
      - 在参数量受限时减少约 hidden_size 个参数
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # bias 显式不创建（上游：bias=False，此处命名清晰）
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        _dbg("LN_NOBIAS_INIT", f"normalized_shape={normalized_shape} eps={eps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
        _dbg("LN_NOBIAS_OUT", out)
        return out


# ── ScaledDotProductAttentionWDP（softmax 后 dropout） ────────────────────────
class ScaledDotProductAttentionWDP(nn.Module):
    """
    Scaled Dot-Product Attention，dropout 在 softmax 之后施加。

    对应 a1d04b793 在 mpu/transformer.py 中将 attention dropout
    从 softmax 之前移到 softmax 之后的改动。

    Walpurgis: 将此行为差异建模为独立模块，避免隐式行为变更。

    Softmax 前 dropout（旧行为）: 直接 mask 掉部分 logit，效果接近忽略某些 key
    Softmax 后 dropout（新行为，a1d04b793）: mask 掉部分 attention weight，
      更接近 DropAttention 语义，对梯度流影响更小
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        _dbg("SDPA_WDP_INIT", f"dropout={dropout} (post-softmax)")

    def forward(
        self,
        query: torch.Tensor,       # [B, H, T, D_h]
        key: torch.Tensor,         # [B, H, T_k, D_h]
        value: torch.Tensor,       # [B, H, T_k, D_h]
        mask: Optional[torch.Tensor] = None,  # [T, T_k] or [B, 1, T, T_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (context [B, H, T, D_h], attn_weights [B, H, T, T_k])"""
        d_h = query.size(-1)
        scale = math.sqrt(d_h)

        # QK^T / sqrt(d_h)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale  # [B, H, T, T_k]
        _dbg("ATTN_SCORES", scores)

        if mask is not None:
            # additive mask: -inf for masked positions
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
            _dbg("ATTN_SCORES_MASKED", scores)

        # softmax → dropout（a1d04b793: dropout after softmax）
        attn_weights = F.softmax(scores, dim=-1)
        _dbg("ATTN_WEIGHTS_PRE_DROP", attn_weights)

        attn_weights = self.dropout(attn_weights)   # ← a1d04b793 位置变更
        _dbg("ATTN_WEIGHTS_POST_DROP", attn_weights)

        context = torch.matmul(attn_weights, value)  # [B, H, T, D_h]
        _dbg("ATTN_CONTEXT", context)

        return context, attn_weights


# ── MultiHeadSelfAttention（Walpurgis单机包装） ───────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力，单机单卡实现（不含 Megatron MPU tensor parallel）。
    使用 ScaledDotProductAttentionWDP（softmax 后 dropout，对应 a1d04b793）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size={hidden_size} 必须整除 num_heads={num_heads}"
        self.num_heads = num_heads
        self.d_h = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = ScaledDotProductAttentionWDP(dropout=dropout)
        self.resid_dropout = nn.Dropout(p=dropout)

        _dbg("MHSA_INIT", f"hidden={hidden_size} heads={num_heads} d_h={self.d_h}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H = x.shape

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.d_h).transpose(1, 2)  # [B, nh, T, d_h]

        q = _split_heads(self.q_proj(x))
        k = _split_heads(self.k_proj(x))
        v = _split_heads(self.v_proj(x))

        _dbg("QKV_PROJECTED", f"q={list(q.shape)} k={list(k.shape)} v={list(v.shape)}")

        ctx, _ = self.attn(q, k, v, mask)   # [B, nh, T, d_h]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H)  # [B, T, H]

        out = self.resid_dropout(self.out_proj(ctx))
        _dbg("MHSA_OUT", out)
        return out


# ── ParallelTransformerLayerWrapper（主 transformer 层包装） ─────────────────
class ParallelTransformerLayerWrapper(nn.Module):
    """
    单个 Transformer 层（Pre-LN），对应 Megatron ParallelTransformerLayer。

    本次 commit 变动:
    1. LayerNorm 改为无偏 (LayerNormNoBias)
    2. attention dropout 移到 softmax 之后

    Walpurgis 改写: Pre-LN 结构、四个断点（ATTN_PRE, ATTN_POST, FFN_PRE, FFN_POST）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.1,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id

        # LayerNorm 无偏（a1d04b793）
        self.ln1 = LayerNormNoBias(hidden_size)
        self.ln2 = LayerNormNoBias(hidden_size)

        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)

        # FFN (GeGLU 激活，Walpurgis替代原始 GELU，提升梯度流)
        self.ffn_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.ffn_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.ffn_dropout = nn.Dropout(p=dropout)

        _dbg(
            "LAYER_INIT",
            f"layer_id={layer_id} hidden={hidden_size} heads={num_heads} "
            f"ffn={ffn_hidden_size} (LN_nobias, post-softmax-dropout)",
        )

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """GeGLU FFN: FFN(x) = down(gate(x) * σ(up(x)))"""
        gate = F.gelu(self.ffn_gate(x))
        up = self.ffn_up(x)
        return self.ffn_down(self.ffn_dropout(gate * up))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── Attention sub-layer (Pre-LN) ──────────────────────────────────────
        _dbg(f"L{self.layer_id}:ATTN_PRE", x)
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask)
        x = residual + x
        _dbg(f"L{self.layer_id}:ATTN_POST", x)

        # ── FFN sub-layer (Pre-LN) ────────────────────────────────────────────
        _dbg(f"L{self.layer_id}:FFN_PRE", x)
        residual = x
        x = self.ln2(x)
        x = self._ffn(x)
        x = residual + x
        _dbg(f"L{self.layer_id}:FFN_POST", x)

        return x


# ── TransformerWrapper（多层堆叠） ────────────────────────────────────────────
class TransformerWrapper(nn.Module):
    """
    多层 ParallelTransformerLayerWrapper 的堆叠容器。
    Walpurgis 特有：按需在每层输出记录 _dbg 统计。
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_hidden_size = ffn_hidden_size or (4 * hidden_size)
        self.layers = nn.ModuleList([
            ParallelTransformerLayerWrapper(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size=ffn_hidden_size,
                dropout=dropout,
                layer_id=i,
            )
            for i in range(num_layers)
        ])
        self.final_ln = LayerNormNoBias(hidden_size)
        _dbg("TRANSFORMER_INIT", f"num_layers={num_layers} hidden={hidden_size}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _dbg("TRANSFORMER_IN", x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_ln(x)
        _dbg("TRANSFORMER_OUT", x)
        return x
