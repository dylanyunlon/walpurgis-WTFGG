"""
Inherent temporal modelling primitives:
  - RNNLayer  : GRU-based sequential encoder
  - TransformerLayer : single-head / multi-head self-attention wrapper
"""

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys

_DBG_INH_M = ("--debug-inhm" in sys.argv) or False


class RNNLayer(nn.Module):
    """
    Per-node GRU that encodes the temporal dimension.
    Input : [B, L, N, D]
    Output: [L, B*N, D]  — stacked hidden states (seq-first for transformer)
    """

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.d_hidden = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        B, L, N, D = X.shape
        # merge batch & node dims so each node runs an independent GRU
        X_flat = X.transpose(1, 2).reshape(B * N, L, D)
        h = torch.zeros_like(X_flat[:, 0, :])      # initial hidden state

        states = []
        for t in range(L):
            h = self.gru_cell(X_flat[:, t, :], h)
            states.append(h)

        out = torch.stack(states, dim=0)            # [L, B*N, D]
        out = self.drop(out)

        if _DBG_INH_M:
            print(f"[DBG:inhm] RNNLayer  input=[{B},{L},{N},{D}]  "
                  f"out={tuple(out.shape)}  h_final_norm={h.norm().item():.4f}")
        return out


class TransformerLayer(nn.Module):
    """Thin wrapper around PyTorch MultiheadAttention."""

    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mha = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        """
        Q, K, V : [L, B*N, D]  (seq-first layout)
        Returns : [L, B*N, D]
        """
        attn_out, _ = self.mha(Q, K, V)
        attn_out = self.drop(attn_out)
        if _DBG_INH_M:
            print(f"[DBG:inhm] TransformerLayer  Q={tuple(Q.shape)}  "
                  f"out_norm={attn_out.norm().item():.4f}")
        return attn_out
