"""
Walpurgis v2 Temporal Models — LSTM + ALiBi Transformer
==========================================================
Deltas vs prior:
  - BiGRU → *LSTM* with forget-gate bias init (+1.0).  LSTM's explicit
    cell state provides better long-range gradient flow than GRU's
    hidden-only architecture.  Forward-only (not bidirectional) to
    match the causal nature of traffic forecasting.
  - RoPE → *ALiBi* (Attention with Linear Biases): instead of rotating
    Q/K, ALiBi adds a linear distance penalty directly to attention
    scores.  This is simpler (no trig), extrapolates better to unseen
    lengths, and has zero learnable parameters for position encoding.
  - Per-head slope follows the geometric series m_i = 2^{-8/H * i}.

Breakpoint helpers:
    rnn._diag_last              # last forward stats
    transformer._attn_entropy   # attention entropy from last forward
"""
import torch
import torch.nn as nn
import math
from collections import deque


class RNNLayer(nn.Module):
    """LSTM temporal encoder with forget-gate bias initialization."""

    _n = 0

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            batch_first=True, dropout=dropout,
        )
        # Initialize forget gate bias to +1 (better gradient flow)
        for name, p in self.rnn.named_parameters():
            if "bias" in name:
                n = p.shape[0]
                p.data[n // 4: n // 2].fill_(1.0)
        self._debug = True
        self._diag_last = {}

    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        RNNLayer._n += 1
        out, (h_n, c_n) = self.rnn(x)

        if self._debug:
            with torch.no_grad():
                self._diag_last = {
                    "step": RNNLayer._n,
                    "out_norm": round(out.norm().item(), 4),
                    "h_norm": round(h_n.norm().item(), 4),
                    "c_norm": round(c_n.norm().item(), 4),
                    "c_mean": round(c_n.mean().item(), 5),
                }
            if RNNLayer._n % 500 == 1:
                d = self._diag_last
                print(
                    f"        [LSTM #{RNNLayer._n}] "
                    f"out_‖={d['out_norm']:.4f} h_‖={d['h_norm']:.4f} "
                    f"cell: ‖={d['c_norm']:.4f} μ={d['c_mean']:.5f}"
                )
        return out


def _alibi_slopes(n_heads):
    """Geometric slopes for ALiBi: m_i = 2^{-8/H · (i+1)}."""
    ratio = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([ratio ** (i + 1) for i in range(n_heads)])


def _alibi_bias(n_heads, seq_len, device):
    """Compute ALiBi position bias matrix: slopes × distance."""
    slopes = _alibi_slopes(n_heads).to(device)  # [H]
    # Distance matrix: |i - j|
    pos = torch.arange(seq_len, device=device).float()
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # [L, L]
    # Bias = -slope * distance (closer = higher attention)
    bias = -slopes.unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)  # [H, L, L]
    return bias


class TransformerLayer(nn.Module):
    """Pre-norm Transformer with ALiBi positional encoding.

    ALiBi adds a linear distance penalty to attention logits,
    encoding position without any learnable parameters.
    """

    _n = 0

    def __init__(self, hidden_dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self._debug = True
        self._attn_entropy = None
        self._cached_bias = None
        self._cached_len = -1

    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        TransformerLayer._n += 1
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Pre-norm attention
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, L, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product + ALiBi bias
        scale = math.sqrt(dh)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale

        # ALiBi: cache bias for fixed seq_len
        if self._cached_len != L:
            self._cached_bias = _alibi_bias(H, L, x.device)
            self._cached_len = L
        attn_logits = attn_logits + self._cached_bias.unsqueeze(0)  # broadcast over batch

        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        x = x + out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        # Track attention entropy for diagnostics
        if self._debug:
            with torch.no_grad():
                # Shannon entropy of attention distribution
                ent = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
                self._attn_entropy = round(ent.item(), 4)
            if TransformerLayer._n % 500 == 1:
                print(
                    f"        [ALiBi-Transformer #{TransformerLayer._n}] "
                    f"out_‖={x.norm().item():.4f} "
                    f"attn_entropy={self._attn_entropy:.4f} "
                    f"(max_ent={math.log(L):.4f})"
                )
        return x
