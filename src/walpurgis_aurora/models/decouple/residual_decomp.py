import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:resdecomp:{tag}] mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class RMSNorm(nn.Module):
    """aurora: RMSNorm替代LayerNorm, 无偏置更轻量"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    aurora: RMSNorm(x - GELU(y) * α), α可学习标量初始化0.9"""
    def __init__(self, input_shape):
        super().__init__()
        self.norm = RMSNorm(input_shape[-1])
        self.act = nn.GELU()
        # aurora: 可学习残差缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        u = x - self.act(y) * alpha
        u = self.norm(u)
        _adbg("residual_norm", u)
        _adbg("alpha", alpha)
        return u
