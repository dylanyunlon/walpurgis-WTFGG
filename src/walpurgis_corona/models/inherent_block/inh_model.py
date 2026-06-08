"""
Corona InhModel — 算法改写:
  upstream: GRUCell + MultiheadAttention
  corona: LSTMCell + RoPE (旋转位置编码), 替代GRU的遗忘门结构,
          LSTM的细胞状态提供更稳定的长程记忆
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from ... import lstm_hidden_state_dump


def _apply_rope(x, seq_len, dim):
    """旋转位置编码 (Rotary Position Embedding) — 支持任意batch维度"""
    device = x.device
    half_dim = dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [seq, half_dim]
    cos_vals = torch.cos(angles)  # [seq, half_dim]
    sin_vals = torch.sin(angles)
    # 需要将cos/sin广播到x的形状: x可能是[seq, batch*nodes, dim]
    # 只在第0维(seq)上对齐
    shape = [1] * len(x.shape)
    shape[0] = seq_len
    shape[-1] = half_dim
    cos_vals = cos_vals.view(*shape)
    sin_vals = sin_vals.view(*shape)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:2*half_dim]
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x1 * sin_vals + x2 * cos_vals
    if x.shape[-1] > 2 * half_dim:
        return torch.cat([out1, out2, x[..., 2*half_dim:]], dim=-1)
    return torch.cat([out1, out2], dim=-1)


class RNNLayer(nn.Module):
    """Corona: LSTMCell替代GRUCell"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)  # Corona: LSTM
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        cx = torch.zeros_like(X[:, 0, :])  # Corona: LSTM细胞状态
        output = []
        for t in range(X.shape[1]):
            hx, cx = self.lstm_cell(X[:, t, :], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=0)
        # Corona: LSTM隐藏状态诊断
        lstm_hidden_state_dump("inh_rnn", hx.detach(), cx.detach())
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.multi_head_self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        # Corona: 在attention前加RoPE
        seq_len = X.shape[0]
        X = _apply_rope(X, seq_len, self.hidden_dim)
        K = _apply_rope(K, K.shape[0], self.hidden_dim)
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
