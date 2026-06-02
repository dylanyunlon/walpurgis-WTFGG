"""
Inherent-pattern sub-models: GRU temporal encoder + Transformer MSA.
"""
import sys
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

_DBG = ("--debug-inhmod" in sys.argv)


class RNNLayer(nn.Module):
    """Single-layer GRU that processes (B*N, L, D) sequences step by step."""

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        B, L, N, D = X.shape
        flat = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(flat[:, 0, :])
        steps = []
        for t in range(L):
            hx = self.gru_cell(flat[:, t, :], hx)
            steps.append(hx)
        out = torch.stack(steps, dim=0)       # (L, B*N, D)
        out = self.drop(out)

        if _DBG:
            print(f"[DBG:inhmod] RNN  in=({B},{L},{N},{D})  "
                  f"out=(L={out.shape[0]}, BN={out.shape[1]}, D={out.shape[2]})  "
                  f"hx_norm={hx.norm().item():.4f}")
        return out


class TransformerLayer(nn.Module):
    """Single multi-head self-attention layer."""

    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mha  = MultiheadAttention(hidden_dim, num_heads,
                                       dropout=dropout, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        attn_out, _ = self.mha(Q, K, V)
        out = self.drop(attn_out)
        if _DBG:
            print(f"[DBG:inhmod] Transformer  Q={tuple(Q.shape)}  "
                  f"out_mean={out.mean().item():.4f}")
        return out
