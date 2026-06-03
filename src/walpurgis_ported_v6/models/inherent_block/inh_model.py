"""Inherent model components — LayerNorm-GRU + Pre-Norm Transformer.

Changes
-------
1. ``RNNLayer`` — wraps GRUCell output with LayerNorm before passing to
   next step.  Standard GRU has no normalisation between steps, which
   causes hidden state magnitude drift in deep temporal unrolls.
2. ``TransformerLayer`` — pre-norm architecture: LayerNorm is applied
   *before* multi-head attention, not after.  Pre-norm is more stable
   for training and converges faster (see Xiong et al., 2020).
3. Attention entropy diagnostic: when debug is on, computes the entropy
   of the attention weight matrix to detect if heads are collapsing
   (entropy → 0) or becoming uniform (entropy → log(N)).
"""

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from walpurgis_ported_v6 import _dbg


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)          # ← new
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, T, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, T, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(T):
            hx = self.gru_cell(X[:, t, :], hx)
            hx = self.ln(hx)                        # ← normalise per step
            output.append(hx)
        output = torch.stack(output, dim=0)          # (T, B*N, D)
        output = self.dropout(output)
        _dbg("RNNLayer", output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.pre_ln = nn.LayerNorm(hidden_dim)       # ← pre-norm
        self.mha = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        # pre-norm: normalise queries before attention
        X_n = self.pre_ln(X)
        attn_out, attn_weights = self.mha(X_n, K, V)
        attn_out = self.dropout(attn_out)

        # attention entropy diagnostic
        if attn_weights is not None:
            _entropy_diagnostic(attn_weights)

        return attn_out


def _entropy_diagnostic(weights):
    """Print attention entropy (higher = more diffuse attention)."""
    # weights shape: (B*N, T, T) or (T, T)
    eps = 1e-12
    p = weights.detach().clamp(min=eps)
    entropy = -(p * p.log()).sum(dim=-1).mean().item()
    _dbg("AttnEntropy", entropy=f"{entropy:.4f}")
