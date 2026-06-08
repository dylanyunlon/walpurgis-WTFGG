"""
InhModel — Penumbra变体
算法改动: MinGRU + 跨模态交叉注意力
  原版: 标准GRUCell + MultiheadSelfAttention
  Penumbra:
    - MinGRU: 简化GRU, 去掉reset gate, 只保留update gate
      h_t = (1-z_t) * h_{t-1} + z_t * tanh(W_h * x_t)
      参数量减少1/3, 训练更快
    - Cross-Attention: Q来自RNN输出, K/V来自原始输入
      让时序信号能"回看"原始空间特征
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from ... import _dbg


class MinGRULayer(nn.Module):
    """Minimal GRU: 去掉reset gate的轻量GRU"""

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # update gate
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)
        # candidate: 不再需要reset gate
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            # update gate: z = σ(W_z * [h, x])
            z = torch.sigmoid(
                self.W_z(torch.cat([hx, x_t], dim=-1)))
            # candidate: h̃ = tanh(W_h * x)  (无reset gate)
            h_tilde = torch.tanh(self.W_h(x_t))
            # 更新: h = (1-z)*h + z*h̃
            hx = (1 - z) * hx + z * h_tilde
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)

        _dbg("mingru.output_norm",
             output.norm(), "inherent")
        return output


class CrossAttentionLayer(nn.Module):
    """跨模态交叉注意力: Q=RNN输出, K/V=原始信号"""

    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.cross_attn = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        self.self_attn = MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, bias=bias)
        # 门控: 控制cross vs self attention的混合比
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Q, K, V, cross_K=None, cross_V=None):
        # Self-attention
        self_out = self.self_attn(Q, K, V)[0]
        self_out = self.dropout(self_out)

        if cross_K is not None and cross_V is not None:
            # Cross-attention: Q=RNN output, K/V=original
            cross_out = self.cross_attn(
                Q, cross_K, cross_V)[0]
            cross_out = self.dropout(cross_out)
            # 门控混合
            g = torch.sigmoid(self.gate)
            combined = g * self_out + (1 - g) * cross_out
            _dbg("cross_attn.gate", g, "inherent")
        else:
            combined = self_out

        combined = self.norm(combined)
        return combined
