import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:resdecomp:{tag}] mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)


class ScaleNorm(nn.Module):
    """solstice: ScaleNorm替代LayerNorm/RMSNorm — 仅可学习scale, 更轻量"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return self.g * x / norm


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    solstice: ScaleNorm(x - Mish(y) * α), α可学习标量初始化0.9"""
    def __init__(self, input_shape):
        super().__init__()
        self.norm = ScaleNorm(input_shape[-1])
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def _mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x, y):
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        u = x - self._mish(y) * alpha
        u = self.norm(u)
        _sdbg("residual_norm", u)
        _sdbg("alpha", alpha)
        return u
