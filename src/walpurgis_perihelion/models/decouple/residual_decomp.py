"""
ResidualDecomp — Perihelion变体
算法改动: FFT Band-Pass分解(傅里叶变换→带通滤波→IFFT)
  原版: LayerNorm + ReLU 做残差分解
  Perihelion: 对残差信号做FFT, 用可学习的带通滤波器
             保留中频成分(去除直流偏移和高频噪声)
             IFFT回时域后与原始信号做残差
             带通边界可学习, 滤波器形状用sigmoid软窗
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg


class FFTBandPassFilter(nn.Module):
    """可学习的傅里叶域带通滤波器"""

    def __init__(self, max_freq_bins, init_low=0.1, init_high=0.8):
        super().__init__()
        # 可学习的截止频率(归一化到0-1)
        self.low_cutoff = nn.Parameter(torch.tensor(init_low))
        self.high_cutoff = nn.Parameter(torch.tensor(init_high))
        # 滤波器过渡带宽度
        self.transition_width = nn.Parameter(torch.tensor(0.15))
        self.max_freq_bins = max_freq_bins

    def forward(self, fft_signal):
        """
        fft_signal: [..., F] 复数傅里叶系数
        返回: [..., F] 滤波后的傅里叶系数
        """
        num_freqs = fft_signal.shape[-1]
        # 归一化频率轴 [0, 1]
        freq_axis = torch.linspace(0, 1, num_freqs,
                                   device=fft_signal.device)

        low = torch.sigmoid(self.low_cutoff)
        high = torch.sigmoid(self.high_cutoff)
        # 确保 low < high
        high = low + F.softplus(high - low + 0.1)
        high = torch.clamp(high, max=0.95)
        width = torch.clamp(self.transition_width.abs(), min=0.02, max=0.3)

        # sigmoid软窗带通: 上升沿和下降沿
        rising = torch.sigmoid((freq_axis - low) / width)
        falling = torch.sigmoid((high - freq_axis) / width)
        bandpass_mask = rising * falling

        # 保持DC分量一定比例(避免完全去除均值)
        bandpass_mask[0] = 0.3

        _dbg("fft_bandpass.low", low, "decouple")
        _dbg("fft_bandpass.high", high, "decouple")
        _dbg("fft_bandpass.pass_ratio",
             f"{bandpass_mask.mean().item():.4f}", "decouple")

        return fft_signal * bandpass_mask


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        # FFT带通滤波器
        self.bandpass = FFTBandPassFilter(max_freq_bins=64)
        # 分解后的LayerNorm稳定输出
        self.post_norm = nn.LayerNorm(dim)
        # 分解强度: 控制滤波残差的贡献权重
        self.decomp_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, y):
        # y是backcast信号, 在特征维度做FFT分解
        # 将最后一维视为"频率序列"做FFT
        y_fft = torch.fft.rfft(y.float(), dim=-1)

        # 带通滤波: 保留中频, 去除DC漂移和高频噪声
        y_filtered = self.bandpass(y_fft)

        # IFFT回时域
        y_reconstructed = torch.fft.irfft(y_filtered, n=y.shape[-1], dim=-1)
        y_reconstructed = y_reconstructed.to(x.dtype)

        # 加权残差分解
        w = torch.sigmoid(self.decomp_weight)
        residual = x - w * y_reconstructed

        # LayerNorm稳定
        residual = self.post_norm(residual)

        _dbg("residual_decomp.fft_energy",
             y_fft.abs().mean(), "decouple")
        _dbg("residual_decomp.weight", w, "decouple")

        return residual
