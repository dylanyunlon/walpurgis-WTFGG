"""
Corona ResidualDecomp — 算法改写:
  upstream: LayerNorm(x - ReLU(y))
  corona: EMA-based decomposition — 用指数移动平均平滑残差信号,
          可学习的momentum参数控制平滑程度
"""
import torch
import torch.nn as nn
from ... import _dbg, EMAStateMonitor

_ema_monitor = EMAStateMonitor()


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        # Corona改写: 可学习EMA momentum (替代upstream的ReLU)
        self.ema_momentum = nn.Parameter(torch.tensor(0.9))
        self._ema_state = None

    def forward(self, x, y):
        # Corona: EMA平滑代替直接ReLU激活
        momentum = torch.sigmoid(self.ema_momentum)  # 限制到(0,1)
        if self._ema_state is not None and self._ema_state.shape == y.shape:
            # detach ema_state to prevent backward through old graph
            ema_detached = self._ema_state.detach()
            smoothed_y = momentum * ema_detached + (1 - momentum) * y
        else:
            smoothed_y = y * momentum
        self._ema_state = smoothed_y.detach()

        u = x - smoothed_y
        u = self.ln(u)
        return u
