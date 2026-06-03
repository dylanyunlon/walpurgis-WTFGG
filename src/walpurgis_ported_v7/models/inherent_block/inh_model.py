import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys

_DBG_INH = ("--dbg-inh" in sys.argv)


class RNNLayer(nn.Module):
    """算法改动:
    1. GRU hidden state 初始化用 xavier 而非 zeros —— 零初始化在前几步
       会让 reset gate 和 update gate 对称退化, xavier 打破对称性
    2. 加 LayerNorm 在 GRU 输出之后
    """
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(hidden_dim)

        # 用于 xavier init 的可学习初始隐状态
        self.h0 = nn.Parameter(th.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.h0)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)

        # 算法改动: 用可学习的 h0 expand 成 batch 大小, 而非 zeros
        hx = self.h0.expand(X.shape[0], -1).contiguous()

        output = []
        for t in range(X.shape[1]):
            hx = self.gru_cell(X[:, t, :], hx)
            output.append(hx)
        output = th.stack(output, dim=0)

        # LayerNorm on output
        output = self.out_ln(output)
        output = self.dropout(output)

        if _DBG_INH:
            with th.no_grad():
                print(f"[DBG-INH-RNN] seq_len={seq_len}  "
                      f"h0_norm={self.h0.norm().item():.4f}  "
                      f"output_norm={output.norm().item():.4f}  "
                      f"output_range=[{output.min().item():.4f}, "
                      f"{output.max().item():.4f}]")
        return output


class TransformerLayer(nn.Module):
    """算法改动: Pre-LayerNorm 模式
    原版是 post-norm (MSA -> dropout -> add -> norm)
    改为 pre-norm (norm -> MSA -> dropout), 训练更稳定
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        # Pre-norm
        X_n = self.pre_norm(X)
        K_n = self.pre_norm(K)
        V_n = self.pre_norm(V)

        hidden_states_MSA = self.multi_head_self_attention(X_n, K_n, V_n)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)

        if _DBG_INH:
            with th.no_grad():
                attn_energy = (X_n * hidden_states_MSA).sum(-1).mean().item()
                print(f"[DBG-INH-TF] attn_energy_proxy={attn_energy:.5f}  "
                      f"out_norm={hidden_states_MSA.norm().item():.4f}")

        return hidden_states_MSA
