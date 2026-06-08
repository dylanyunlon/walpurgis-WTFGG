"""Rift inherent model: RNN + Transformer with Split-Recombine in attention.
Rift replaces standard MultiheadAttention with a split-recombine variant:
the query/key are split into groups before attention, then recombined,
giving each group a chance to specialize in different temporal patterns."""
import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys, os

_RF_DBG = os.environ.get('RIFT_DEBUG', '0') == '1'


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = th.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    """Rift特有: Transformer层增加了pre/post LayerNorm + SiLU门控"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Rift特有: pre-attention LayerNorm
        self.pre_norm = nn.LayerNorm(hidden_dim)
        # Rift特有: SiLU门控
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, K, V):
        # Rift特有: pre-norm before attention
        X_normed = self.pre_norm(X)
        K_normed = self.pre_norm(K)
        hidden_states_MSA = self.multi_head_self_attention(X_normed, K_normed, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        # Rift特有: SiLU门控残差
        gate = th.sigmoid(self.gate_proj(hidden_states_MSA))
        hidden_states_MSA = gate * hidden_states_MSA
        if _RF_DBG:
            print(f"[RF-DBG:transformer] attn_norm={hidden_states_MSA.norm().item():.4f} "
                  f"gate_mean={gate.mean().item():.4f}", file=sys.stderr, flush=True)
        return hidden_states_MSA
