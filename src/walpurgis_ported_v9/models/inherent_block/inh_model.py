"""
inh_model.py — v9 port
Algo delta:
  1. RNNLayer: GRU 每步输出后加 LayerNorm, 防止长序列隐状态幅度漂移
  2. TransformerLayer: post-norm → pre-norm 架构 (先 LN 再 attention),
     训练更稳定 (参见 "On Layer Normalization in the Transformer Architecture")
  3. 注意力权重计算后打印 entropy 诊断:
     entropy 低 = 注意力集中在少数位置; 高 = 均匀分散 (可能没学到东西)
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from walpurgis_ported_v9 import _dbg

_TAG = "inh_model"


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # v9: inter-step LayerNorm
        self.step_ln = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        B, S, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, S, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(S):
            hx = self.gru_cell(X[:, t, :], hx)
            # v9: LN after each step
            hx = self.step_ln(hx)
            output.append(hx)
        output = torch.stack(output, dim=0)   # [S, B*N, D]
        output = self.dropout(output)
        _dbg(_TAG, f"GRU  steps={S}  hx_norm={hx.norm(2, dim=-1).mean().item():.4g}")
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.msa = MultiheadAttention(hidden_dim, num_heads,
                                      dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # v9: pre-norm
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        # v9: pre-norm architecture
        X_n = self.ln_q(X)
        K_n = self.ln_kv(K)
        V_n = self.ln_kv(V)
        attn_out, attn_weights = self.msa(X_n, K_n, V_n)
        attn_out = self.dropout(attn_out)

        # v9: attention entropy diagnostic
        if attn_weights is not None:
            # attn_weights: [B*N, S, S] or similar
            eps = 1e-8
            ent = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
            _dbg(_TAG, f"attn_entropy={ent.item():.4f}  "
                        f"attn_max={attn_weights.max().item():.4f}")

        return attn_out
