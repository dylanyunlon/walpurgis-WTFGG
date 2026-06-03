import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.checkpoint import checkpoint
from walpurgis import _dbg

_TAG = "inhmod"


class RMSNorm(nn.Module):
    """RMSNorm — 比 LayerNorm 更轻量, 不减均值只除 RMS."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 改动1: 步间 RMSNorm — upstream GRU 输出无归一化
        self.step_norm = RMSNorm(hidden_dim)

    def _step_fn(self, x_t, hx):
        """单步 GRU, 用于 checkpoint."""
        return self.gru_cell(x_t, hx)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            # 改动2: gradient checkpoint 减显存
            # upstream 直接 self.gru_cell(X[:,t,:], hx)
            if self.training and t > 0:
                hx = checkpoint(self._step_fn, X[:, t, :], hx,
                                use_reentrant=False)
            else:
                hx = self.gru_cell(X[:, t, :], hx)

            # 改动1: 步间 RMSNorm
            hx = self.step_norm(hx)
            output.append(hx)

        output = torch.stack(output, dim=0)
        output = self.dropout(output)

        _dbg(_TAG, "gru_out", output=output, final_h=hx)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mha = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # 改动3: pre-norm — upstream 是 post-norm(实际无 norm)
        # pre-norm: 先 LN 再 attention, 训练更稳定
        self.pre_ln = nn.LayerNorm(hidden_dim)

    def forward(self, X, K, V):
        # 改动3: pre-norm 架构
        X_normed = self.pre_ln(X)
        K_normed = self.pre_ln(K)
        V_normed = self.pre_ln(V)

        out, attn_weights = self.mha(X_normed, K_normed, V_normed)
        out = self.dropout(out)

        # 改动4: 注意力熵诊断 — upstream 完全不监控 attention 分布
        if attn_weights is not None:
            # entropy = -sum(p * log(p))
            eps = 1e-8
            entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            _dbg(_TAG, "attn_entropy",
                 mean_entropy=entropy.mean(),
                 min_entropy=entropy.min(),
                 max_entropy=entropy.max())

        _dbg(_TAG, "transformer_out", out=out)
        return out
