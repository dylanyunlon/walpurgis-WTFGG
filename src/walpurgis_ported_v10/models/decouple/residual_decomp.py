import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

_TAG = "resdecomp"


def _mish(x):
    """Mish激活: x * tanh(softplus(x)).
    upstream用ReLU; Mish保持负值梯度流且更光滑."""
    return x * torch.tanh(F.softplus(x))


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        feat_dim = input_shape[-1]
        self.ln = nn.LayerNorm(feat_dim)

        # 改动2: 可学习残差缩放 α ∈ (0,1)
        # upstream 直接 u = x - ReLU(y), 无缩放
        # 这里 u = x - α * Mish(y), α 由 sigmoid(raw_alpha) 控制
        self._raw_alpha = nn.Parameter(torch.tensor(2.2))  # sigmoid(2.2) ≈ 0.9

        # 改动3: LN 前加轻量 dropout
        self.drop = nn.Dropout(0.05)

    def forward(self, x, y):
        alpha = torch.sigmoid(self._raw_alpha)

        # 改动1: Mish 替代 ReLU
        activated_y = _mish(y)

        # 改动2: 缩放残差
        u = x - alpha * activated_y

        # 改动3: dropout → LN
        u = self.drop(u)
        u = self.ln(u)

        _dbg(_TAG, "decomp",
             alpha=alpha, x_norm=x.norm(), y_norm=y.norm(),
             residual_norm=u.norm())
        return u
