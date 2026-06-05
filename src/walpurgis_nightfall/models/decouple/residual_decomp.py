"""
ResidualDecomp — Nightfall变体
算法改写:
  1. ReLU → LeakyReLU(0.1) (保留负值梯度流)
  2. 加可学习残差缩放因子 alpha (初始0.9, 控制残差比例)
"""
import torch
import torch.nn as nn
from walpurgis_nightfall import _dbg


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.LeakyReLU(negative_slope=0.1)
        # 可学习残差缩放: 初始接近1.0
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        residual = x - self.ac(y)
        scaled = self.alpha * residual
        out = self.ln(scaled)
        _dbg("resid_decomp.alpha", self.alpha, "model")
        _dbg("resid_decomp.out", out, "model")
        return out
