"""
ResidualDecomp — Aurora变体
算法改写: ReLU → SiLU(Swish), 增加可学习的momentum参数
SiLU: x * sigmoid(x) 提供平滑非线性和自门控特性
momentum参数控制残差分解中新旧信号的指数加权移动平均
"""
import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """残差分解: 从信号中移除已学到的模式, 使用SiLU和动量加权"""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # Aurora: SiLU(Swish) 替代ReLU
        # SiLU = x * sigmoid(x), 既有非线性又保持负值信息
        self.ac = nn.SiLU()
        # Aurora: 可学习的momentum参数, 控制残差保留比例
        # 经sigmoid后范围(0,1), 作为指数移动平均的衰减系数
        self.momentum = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y):
        # Aurora: momentum控制新旧信号的混合比例
        # alpha接近1时保留更多原始信号, 接近0时更多减去学到的模式
        alpha = torch.sigmoid(self.momentum)
        u = alpha * x - (1.0 - alpha) * self.ac(y)
        u = self.ln(u)
        return u
