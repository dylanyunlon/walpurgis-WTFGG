"""
ResidualDecomp — Zenith变体
改写: ReLU → ELU, 增加可学习的衰减因子alpha
"""
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """残差分解: 从信号中移除已学到的模式"""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ELU(alpha=0.8)
        self.decay = nn.Parameter(
            __import__('torch').tensor(1.0))

    def forward(self, x, y):
        scale = __import__('torch').sigmoid(self.decay)
        u = x - scale * self.ac(y)
        u = self.ln(u)
        return u
