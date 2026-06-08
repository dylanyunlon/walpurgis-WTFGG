"""
inh_model — Zenith变体
改写: GRU加残差连接, TransformerLayer加pre-norm
"""
import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X[:, 0, :])
        output = []
        for step in range(X.shape[1]):
            inp = X[:, step, :]
            hx_new = self.gru_cell(inp, hx)
            hx = hx_new + 0.1 * inp
            output.append(hx)
        output = th.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        X_n = self.pre_norm(X)
        K_n = self.pre_norm(K)
        V_n = self.pre_norm(V)
        hidden_states_MSA = self.multi_head_self_attention(
            X_n, K_n, V_n)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        hidden_states_MSA = hidden_states_MSA + X
        return hidden_states_MSA
