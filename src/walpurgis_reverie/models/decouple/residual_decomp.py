import torch
import torch.nn as nn
from walpurgis_reverie import _dbg

_TAG = "decomp"


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    改动: InstanceNorm1d + learnable affine + GELU
    InstanceNorm对每个样本独立归一化, 比LayerNorm更适应不同traffic pattern的尺度差异
    GELU比ReLU平滑, 避免dead neuron
    """

    def __init__(self, input_shape):
        super().__init__()
        feat_dim = input_shape[-1]
        self.norm = nn.InstanceNorm1d(feat_dim, affine=True)
        self.ac = nn.GELU()
        # 可学习残差缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, x, y):
        residual = x - self.ac(y)
        # reshape for InstanceNorm1d: (B*N, D, L)
        B, L, N, D = residual.shape
        r = residual.permute(0, 2, 3, 1).reshape(B * N, D, L)
        r = self.norm(r)
        r = r.reshape(B, N, D, L).permute(0, 3, 1, 2)
        out = r * self.scale
        _dbg(f"{_TAG}/residual_norm", out, _TAG)
        return out
