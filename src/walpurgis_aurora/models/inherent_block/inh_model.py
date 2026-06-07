import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:inhmod:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class RNNLayer(nn.Module):
    """upstream: 裸GRU
    aurora: GRU后接LayerNorm稳定隐状态"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # aurora: LayerNorm稳定GRU隐状态
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            hx = self.gru_cell(X[:, t, :], hx)
            # aurora: LayerNorm
            hx = self.ln(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        _adbg("gru_hidden", output)
        return output


class TransformerLayer(nn.Module):
    """upstream: 裸attention
    aurora: 残差连接 output = X + attention(X,K,V)"""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # aurora: LayerNorm for pre-norm
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        # aurora: pre-norm
        X_normed = self.ln(X)
        attn_out = self.multi_head_self_attention(X_normed, K, V)[0]
        attn_out = self.dropout(attn_out)
        # aurora: 残差连接
        out = X + attn_out
        _adbg("transformer_out", out)
        return out
