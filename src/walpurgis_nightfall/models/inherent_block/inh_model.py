"""
RNNLayer & TransformerLayer — Nightfall变体
算法改写:
  1. RNNLayer: GRU输出后接LayerNorm (稳定隐状态尺度)
  2. TransformerLayer: 加残差连接 out = X + attn(X,K,V)
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from ... import _dbg


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # GRU后接LayerNorm
        self.post_ln = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            hx = self.gru_cell(X[:, t, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        # LayerNorm on GRU output
        output = self.post_ln(output)
        output = self.dropout(output)
        _dbg("rnn.output", output, "model")
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # 残差连接用的LayerNorm
        self.res_ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        attn_out = self.multi_head_self_attention(X, K, V)[0]
        attn_out = self.dropout(attn_out)
        # 残差连接: out = X + attn
        out = self.res_ln(X + attn_out)
        _dbg("transformer.residual_out", out, "model")
        return out
