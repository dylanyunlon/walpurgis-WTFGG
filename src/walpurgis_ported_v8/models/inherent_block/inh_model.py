import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys

_DBG = ("--dbg" in sys.argv)


class RNNLayer(nn.Module):
    """算法改动: bidirectional GRU
    原版: 单向 GRUCell 逐步展开
    改为: 正向 + 反向两个 GRUCell, 最终取均值
    双向能看到未来上下文, 对 backcast 分支有帮助 (inherent signal 是全局的)
    """

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_cell_rev = nn.GRUCell(hidden_dim, hidden_dim)  # 反向
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)

        # forward pass
        hx_fwd = torch.zeros_like(X[:, 0, :])
        out_fwd = []
        for t in range(seq_len):
            hx_fwd = self.gru_cell(X[:, t, :], hx_fwd)
            out_fwd.append(hx_fwd)
        out_fwd = torch.stack(out_fwd, dim=0)

        # backward pass
        hx_bwd = torch.zeros_like(X[:, 0, :])
        out_bwd = []
        for t in range(seq_len - 1, -1, -1):
            hx_bwd = self.gru_cell_rev(X[:, t, :], hx_bwd)
            out_bwd.append(hx_bwd)
        out_bwd.reverse()
        out_bwd = torch.stack(out_bwd, dim=0)

        # 取均值而非拼接, 保持维度不变
        output = (out_fwd + out_bwd) * 0.5
        output = self.dropout(output)

        if _DBG:
            with torch.no_grad():
                print(f"[DBG][RNNLayer] bidir output "
                      f"shape={list(output.shape)}  "
                      f"fwd_last={out_fwd[-1].mean().item():.5f}  "
                      f"bwd_last={out_bwd[0].mean().item():.5f}", flush=True)
        return output


class TransformerLayer(nn.Module):
    """算法改动: pre-LayerNorm + residual connection
    原版: MSA(X, K, V) -> dropout -> 输出
    改为: X_norm = LN(X) -> MSA(X_norm, K, V) -> dropout -> X + residual
    pre-norm 架构训练更稳定, 不需要 warmup
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        X_norm = self.layer_norm(X)
        attn_out = self.multi_head_self_attention(X_norm, K, V)[0]
        attn_out = self.dropout(attn_out)
        # residual
        out = X + attn_out
        return out
