"""
Walpurgis v2 Temporal Models — BiGRU + RoPE Transformer
=========================================================
Deltas:
  - RNN: unidirectional GRU → *bidirectional GRU*, output = mean(fwd, bwd).
    Short traffic sequences (L≤12) benefit from seeing both directions.
  - Transformer: learnable PE → *Rotary Position Embedding (RoPE)* applied
    inside the attention.  RoPE encodes relative position directly in the
    dot-product, which is more parameter-efficient than learnable PE.
"""
import torch
import torch.nn as nn
import math


class RNNLayer(nn.Module):
    """Bidirectional GRU temporal encoder.

    Output = (forward_hidden + backward_hidden) / 2
    so the downstream shape is unchanged from the unidirectional case.
    """

    _n = 0

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)  # merge directions
        self._debug = True

    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        RNNLayer._n += 1
        out, _ = self.rnn(x)   # [B*N, L, 2D]
        out = self.proj(out)    # [B*N, L, D]

        if self._debug and RNNLayer._n % 500 == 1:
            print(f"        [BiGRU #{RNNLayer._n}] in={list(x.shape)} out_norm={out.norm().item():.4f}")
        return out


def _rope_freqs(dim, seq_len, device, base=10000.0):
    """Precompute rotary position frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # [L, D/2]
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos, sin):
    """Apply rotary embeddings to x: [B, H, L, D_head]."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TransformerLayer(nn.Module):
    """Pre-norm Transformer with Rotary Position Embedding.

    RoPE is applied inside the attention computation so that the
    dot-product implicitly encodes relative position.
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

    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        TransformerLayer._n += 1
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Pre-norm attention
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, L, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, dh]

        # RoPE
        cos, sin = _rope_freqs(dh, L, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        x = x + out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        if self._debug and TransformerLayer._n % 500 == 1:
            print(f"        [RoPE-Transformer #{TransformerLayer._n}] out_norm={x.norm().item():.4f}")
        return x
