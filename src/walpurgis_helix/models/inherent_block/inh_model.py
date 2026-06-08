import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention


class RNNLayer(nn.Module):
    """Helix改写: GRU层加入残差连接,
    每步GRU输出与输入做残差混合, 增强梯度流"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        # Helix特有: 残差混合比例
        self.residual_alpha = nn.Parameter(th.tensor(0.1))

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim]    = X.shape
        X   = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx  = th.zeros_like(X[:, 0, :])
        output  = []
        alpha = th.sigmoid(self.residual_alpha)
        for _ in range(X.shape[1]):
            hx_new  = self.gru_cell(X[:, _, :], hx)
            # Helix: 残差连接 — 将输入混入GRU输出
            hx  = (1 - alpha) * hx_new + alpha * X[:, _, :]
            output.append(hx)
        output  = th.stack(output, dim=0)
        output  = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention  = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout                    = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA   = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA   = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
