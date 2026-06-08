import torch.nn as nn
import torch.nn.functional as F


class ResidualDecomp(nn.Module):
    """Residual decomposition.
    Helix改写: 用GELU替代ReLU作为分解激活函数,
    GELU更平滑, 对残差信号的分解更柔和"""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # Helix: GELU替代ReLU
        self.ac = nn.GELU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u
