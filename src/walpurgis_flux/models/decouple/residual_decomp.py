"""Flux ResidualDecomp: 带动量的残差分解.
与upstream(ReLU + LayerNorm)和vortex(同upstream)不同,
Flux加入残差动量: 当前残差与前一步残差做EMA混合,
平滑流式推理中的残差抖动."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class ResidualDecomp(nn.Module):
    """Residual decomposition with momentum smoothing."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()
        # Flux: 残差动量参数
        self.residual_momentum = nn.Parameter(
            torch.tensor(0.85))
        self._prev_residual = None

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        # Flux: 动量平滑残差 — 减少流式推理抖动
        momentum = torch.sigmoid(self.residual_momentum)
        if self._prev_residual is not None and \
                self._prev_residual.shape == u.shape:
            u = momentum * u + \
                (1 - momentum) * self._prev_residual.detach()
        if self.training:
            self._prev_residual = u.detach()
        if _FX_DBG:
            print(f"[FX:residual_decomp] u_range="
                  f"[{u.min().item():.4f},"
                  f"{u.max().item():.4f}] "
                  f"momentum={momentum.item():.4f}",
                  file=sys.stderr)
        return u
