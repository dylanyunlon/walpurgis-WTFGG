import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.checkpoint import checkpoint
from walpurgis_walking import _dbg

_TAG = "inhmod"


class RMSNorm(nn.Module):
    """RMSNorm with optional learnable bias and running RMS tracking.

    比 LayerNorm 更轻量(不减均值), 但加了:
    - 可选 bias 参数, 让模型能学到非零中心
    - 指数移动平均跟踪 running_rms, 推理时可用于异常检测
    """
    def __init__(self, dim, eps=1e-6, affine_bias=True, momentum=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if affine_bias else None
        self.eps = eps
        self.register_buffer('running_rms', torch.ones(1))
        self.momentum = momentum

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # 更新 running statistics (仅训练时)
        if self.training:
            with torch.no_grad():
                batch_rms = rms.mean()
                self.running_rms.mul_(1 - self.momentum).add_(
                    self.momentum * batch_rms)
        out = x / rms * self.scale
        if self.bias is not None:
            out = out + self.bias
        return out


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

        # 注意力熵诊断 + head diversity 检测
        if attn_weights is not None:
            eps = 1e-8
            entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
            _dbg(_TAG, "attn_entropy",
                 mean_entropy=entropy.mean(),
                 min_entropy=entropy.min(),
                 max_entropy=entropy.max())
            # head diversity: 各 head 的 attention pattern 之间的余弦相似度
            # 如果所有 head 关注同样的位置 → 冗余, 需要更多正则
            if attn_weights.dim() >= 3 and attn_weights.shape[0] > 1:
                # attn_weights: (num_heads, tgt_len, src_len) or (B*H, T, T)
                flat = attn_weights.reshape(attn_weights.shape[0], -1)
                normed = flat / (flat.norm(dim=1, keepdim=True) + eps)
                sim_matrix = torch.mm(normed, normed.t())
                # 去掉对角线自身相似度，取上三角均值
                n_heads = sim_matrix.shape[0]
                if n_heads > 1:
                    mask = torch.triu(torch.ones(n_heads, n_heads,
                                                  device=sim_matrix.device), diagonal=1)
                    mean_sim = (sim_matrix * mask).sum() / mask.sum().clamp(min=1)
                    _dbg(_TAG, "head_diversity",
                         mean_cosine_sim=mean_sim,
                         n_heads=torch.tensor(float(n_heads)))

        _dbg(_TAG, "transformer_out", out=out)
        return out
