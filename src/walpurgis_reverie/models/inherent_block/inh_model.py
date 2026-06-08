import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from walpurgis_reverie import _dbg

_TAG = "inh_model"


class RNNLayer(nn.Module):
    """upstream: 标准GRU cell
    改动: Minimal GRU — 只有更新门, 没有重置门
    参数量减少~30%, 训练更快
    min_gru: h_t = (1-z_t) * h_{t-1} + z_t * candidate
    where z_t = sigmoid(W_z * [x_t, h_{t-1}])
    candidate = tanh(W_h * x_t)  (注意没有h的重置)
    """

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 改动: MinGRU components
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros(
            X.shape[0], hidden_dim, device=X.device, dtype=X.dtype)
        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            # update gate (minimal: only one gate)
            z_t = torch.sigmoid(self.W_z(torch.cat([x_t, hx], dim=-1)))
            # candidate (no reset gate — this is the "minimal" part)
            h_candidate = torch.tanh(self.W_h(x_t))
            # update
            hx = (1 - z_t) * hx + z_t * h_candidate
            output.append(hx)

        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _dbg(f"{_TAG}/mingru_out", output, _TAG)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
