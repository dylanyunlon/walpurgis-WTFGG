"""
ResidualDecomp — Umbra变体
算法改动: Haar小波分解 替代 LayerNorm
  Haar小波: 将信号沿最后一维相邻元素配对
    低频(近似): (x[2k] + x[2k+1]) / sqrt(2)
    高频(细节): (x[2k] - x[2k+1]) / sqrt(2)
  分别对低频/高频施加可学习增益, 然后逆变换重建
  效果: 保留主趋势的同时自适应地衰减或增强高频噪声
  对比LayerNorm: 不强制零均值单位方差, 而是按频率分量调节
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, _haar_tracker


class HaarWaveletNorm(nn.Module):
    """Haar小波归一化: 分解→缩放→重建"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        # 低频增益和偏差
        half_dim = max(dim // 2, 1)
        self.low_gain = nn.Parameter(torch.ones(half_dim))
        self.low_bias = nn.Parameter(torch.zeros(half_dim))
        # 高频增益和偏差 (初始化为较小值, 默认抑制噪声)
        self.high_gain = nn.Parameter(torch.ones(half_dim) * 0.5)
        self.high_bias = nn.Parameter(torch.zeros(half_dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        orig_shape = x.shape
        d = x.shape[-1]

        # 如果维度为奇数, 补零使之为偶数
        if d % 2 == 1:
            x = F.pad(x, (0, 1))
            d = d + 1

        # 重塑为配对: (..., d//2, 2)
        x_pairs = x.reshape(*orig_shape[:-1], d // 2, 2)

        # Haar正变换
        inv_sqrt2 = 0.7071067811865476
        low_freq = (x_pairs[..., 0] + x_pairs[..., 1]) * inv_sqrt2
        high_freq = (x_pairs[..., 0] - x_pairs[..., 1]) * inv_sqrt2

        # 对低频和高频分别施加可学习增益+偏差
        low_scaled = low_freq * self.low_gain + self.low_bias
        high_scaled = high_freq * self.high_gain + self.high_bias

        # 记录能量比
        low_energy = low_scaled.detach().pow(2).mean().item()
        high_energy = high_scaled.detach().pow(2).mean().item()
        _haar_tracker.record(low_energy, high_energy)

        # Haar逆变换: 重建信号
        x_rebuilt_0 = (low_scaled + high_scaled) * inv_sqrt2
        x_rebuilt_1 = (low_scaled - high_scaled) * inv_sqrt2
        x_rebuilt = torch.stack([x_rebuilt_0, x_rebuilt_1], dim=-1)
        x_rebuilt = x_rebuilt.reshape(*orig_shape[:-1], d)

        # 截掉可能的补零
        if d != orig_shape[-1]:
            x_rebuilt = x_rebuilt[..., :orig_shape[-1]]

        return x_rebuilt


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        self.haar_norm = HaarWaveletNorm(dim)
        # 残差缩放: 控制分解的强度
        self.decomp_strength = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, y):
        strength = torch.sigmoid(self.decomp_strength)
        residual = F.silu(y)
        u = x - strength * residual
        u = self.haar_norm(u)

        _dbg("residual_decomp.strength", strength, "decouple")
        _dbg("residual_decomp.haar_high_gain",
             self.haar_norm.high_gain.mean(), "decouple")

        return u
