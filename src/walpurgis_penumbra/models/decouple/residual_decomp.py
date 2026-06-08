"""
ResidualDecomp — Penumbra变体
算法改动: PowerNorm + Swish 替代 LayerNorm + ReLU
  PowerNorm: 使用可学习的幂次参数 p 做归一化, x / (mean(|x|^p))^(1/p)
  Swish: x * sigmoid(beta * x), beta可学习
  加入残差缩放因子控制分解强度
"""
import torch
import torch.nn as nn
from ... import _dbg


class PowerNorm(nn.Module):
    """可学习幂次归一化 — 比LayerNorm更灵活"""

    def __init__(self, dim, init_power=2.0, eps=1e-6):
        super().__init__()
        self.power = nn.Parameter(torch.tensor(init_power))
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        p = torch.clamp(self.power, min=1.0, max=4.0)
        # |x|^p 的均值, 取 1/p 次方
        norm_factor = (x.abs().pow(p).mean(dim=-1,
                       keepdim=True) + self.eps).pow(1.0 / p)
        x_normed = x / norm_factor
        return x_normed * self.scale + self.bias


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        self.pnorm = PowerNorm(dim)
        # 可学习Swish的beta参数
        self.swish_beta = nn.Parameter(torch.tensor(1.0))
        # 残差缩放: 控制分解的"攻击性"
        self.decomp_strength = nn.Parameter(torch.tensor(0.8))

    def _swish(self, x):
        beta = torch.clamp(self.swish_beta, min=0.1, max=5.0)
        return x * torch.sigmoid(beta * x)

    def forward(self, x, y):
        strength = torch.sigmoid(self.decomp_strength)
        # 用Swish替代ReLU做残差提取
        residual = self._swish(y)
        u = x - strength * residual
        u = self.pnorm(u)

        _dbg("residual_decomp.strength",
             strength, "decouple")
        _dbg("residual_decomp.swish_beta",
             self.swish_beta, "decouple")

        return u
