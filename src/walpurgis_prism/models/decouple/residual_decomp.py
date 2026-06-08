"""Prism residual decomposition: spectral-aware residual with frequency-domain smoothing.
Unlike upstream (LayerNorm + ReLU) and vortex (GroupNorm + Mish),
Prism applies a frequency-domain low-pass filter to the residual before normalization,
smoothing out high-frequency noise in the decoupled signal."""
import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """Prism: Spectral-aware residual decomposition."""
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()
        # Prism特有: 频域平滑系数
        self.freq_smooth_ratio = nn.Parameter(
            torch.tensor(0.8))

    def _spectral_smooth(self, x):
        """对残差做频域低通滤波, 平滑高频噪声"""
        B, L, N, D = x.shape
        if L < 4:
            return x
        x_freq = torch.fft.rfft(x, dim=1)
        # 构造低通滤波器: 保留前ratio比例的频率
        n_freq = x_freq.shape[1]
        ratio = torch.sigmoid(self.freq_smooth_ratio)
        cutoff = max(1, int(n_freq * ratio.item()))
        mask = torch.zeros(n_freq, device=x.device)
        mask[:cutoff] = 1.0
        # 平滑过渡
        if cutoff < n_freq:
            transition = min(3, n_freq - cutoff)
            for i in range(transition):
                if cutoff + i < n_freq:
                    mask[cutoff + i] = 1.0 - (i + 1) / (
                        transition + 1)
        mask = mask.reshape(1, -1, 1, 1)
        x_filtered = torch.fft.irfft(
            x_freq * mask, n=L, dim=1)
        return x_filtered

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self._spectral_smooth(u)
        u = self.ln(u)
        return u
