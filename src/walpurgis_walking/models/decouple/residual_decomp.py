import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_walking import _dbg

_TAG = "resid"


def _mish(x):
    """Mish: x * tanh(softplus(x)), 比ReLU更平滑, 允许少量负值通过."""
    return x * torch.tanh(F.softplus(x))


class ResidualDecomp(nn.Module):
    """upstream: u = LayerNorm(x - ReLU(y))
    改动: u = LayerNorm(x - α * Mish(y)), α可学习 init=0.9
    """
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # 改动1: Mish 替代 ReLU
        # 改动2: 可学习残差缩放系数 α
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        a = self.alpha.clamp(0.0, 1.5)
        u = x - a * _mish(y)
        u = self.ln(u)
        _dbg(_TAG, "decomp", alpha=a.item(), u_mean=u.mean(), u_std=u.std(),
             x_norm=x.norm(), y_norm=y.norm())
        return u
