import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:inhmod:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class RNNLayer(nn.Module):
    """upstream: 裸GRU
    equinox: Highway GRU — 每步增加highway gate: out = gate*gru + (1-gate)*input"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # equinox: Highway gate — 可学习跳过连接
        self.highway_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            hx_new = self.gru_cell(x_t, hx)
            # equinox: Highway gate
            gate_input = torch.cat([x_t, hx_new], dim=-1)
            gate = torch.sigmoid(self.highway_gate(gate_input))
            hx = gate * hx_new + (1 - gate) * x_t
            hx = self.ln(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _edbg("highway_gru_hidden", output)
        return output


class TransformerLayer(nn.Module):
    """upstream: 裸attention
    equinox: 残差连接 output = X + attention(X,K,V)"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # equinox: LayerNorm for pre-norm
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        # equinox: pre-norm
        X_normed = self.ln(X)
        attn_out = self.multi_head_self_attention(X_normed, K, V)[0]
        attn_out = self.dropout(attn_out)
        # equinox: 残差连接
        out = X + attn_out
        _edbg("transformer_out", out)
        return out
