"""
ResidualDecomp — Parallax变体 (M054)
算法改动: STL/LOESS分解 替代 LayerNorm + ReLU
  原版: LayerNorm(x - ReLU(y))
  Parallax: 将输入分解为趋势(低通滤波) + 季节(周期提取) + 残差
           趋势: 可学习窗口的加权滑动平均 (模拟LOESS局部回归)
           季节: 用FFT提取主频率分量 (模拟STL的季节提取)
           残差: 原始 - 趋势 - 季节
           只保留残差分量, 趋势和季节被分离出去

  这让模型显式地做时间序列分解, 而不是隐式的归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg


class LearnableLoessFilter(nn.Module):
    """可学习的LOESS风格局部回归滤波器"""

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        # 可学习的滤波核权重 — 每个特征通道独立
        self.raw_weights = nn.Parameter(
            torch.randn(1, 1, dim, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """x: [B, L, N, D] → 趋势分量 [B, L, N, D]"""
        B, L, N, D = x.shape
        # 归一化滤波核
        weights = F.softmax(self.raw_weights, dim=-1)
        # 对时间维度做加权滑动平均
        pad = self.kernel_size // 2
        # 重排为 [B*N, D, L] 方便1d卷积
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        x_padded = F.pad(x_flat, (pad, pad), mode='replicate')
        # 逐通道滑窗加权
        trend = torch.zeros_like(x_flat)
        for k in range(self.kernel_size):
            w = weights[0, 0, :, k].unsqueeze(0).unsqueeze(-1)
            trend += w * x_padded[:, :, k:k + L]
        trend = trend + self.bias.unsqueeze(0).unsqueeze(-1)
        trend = trend.reshape(B, N, D, L).permute(0, 3, 1, 2)
        return trend


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape, num_freq=3):
        super().__init__()
        dim = input_shape[-1]
        self.num_freq = num_freq

        # 趋势提取: LOESS风格局部回归
        self.trend_filter = LearnableLoessFilter(dim, kernel_size=5)

        # 季节分量权重: 控制保留多少个FFT频率分量
        self.seasonal_gate = nn.Parameter(torch.ones(num_freq) * 0.5)

        # 残差归一化: 对分解后的残差做LayerNorm稳定
        self.residual_norm = nn.LayerNorm(dim)

        # 分解强度: 控制STL分解的aggressiveness
        self.decomp_alpha = nn.Parameter(torch.tensor(0.7))

    def _seasonal_extract(self, x, trend):
        """FFT提取季节分量 — 取前num_freq个最强频率"""
        detrended = x - trend  # [B, L, N, D]
        B, L, N, D = detrended.shape
        # 对时间维度做FFT
        freq = torch.fft.rfft(detrended, dim=1)
        # 取前num_freq个非零频率的幅度
        magnitudes = freq.abs()[:, 1:, :, :]  # 去掉DC分量
        n_freq = min(self.num_freq, magnitudes.shape[1])
        # 门控: 决定每个频率保留多少
        gate = torch.sigmoid(self.seasonal_gate[:n_freq])
        # 构造滤波后的频谱
        filtered_freq = torch.zeros_like(freq)
        for i in range(n_freq):
            filtered_freq[:, i + 1, :, :] = (
                freq[:, i + 1, :, :] * gate[i])
        # IFFT回时域
        seasonal = torch.fft.irfft(filtered_freq, n=L, dim=1)
        return seasonal

    def forward(self, x, y):
        alpha = torch.sigmoid(self.decomp_alpha)

        # STL分解: x = trend + seasonal + residual
        trend = self.trend_filter(x)
        seasonal = self._seasonal_extract(x, trend)
        stl_residual = x - alpha * (trend + seasonal)

        # 减去预测分量y (对应原始的residual_decomp)
        output = stl_residual - (1 - alpha) * y
        output = self.residual_norm(output)

        _dbg("residual_decomp.alpha",
             alpha, "decouple")
        _dbg("residual_decomp.trend_energy",
             trend.detach().norm(), "decouple")
        _dbg("residual_decomp.seasonal_gate",
             torch.sigmoid(self.seasonal_gate), "decouple")

        return output
