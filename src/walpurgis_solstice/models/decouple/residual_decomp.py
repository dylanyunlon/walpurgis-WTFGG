import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:resdecomp:{tag}] mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)

class PowerNorm(nn.Module):
    """solstice: PowerNorm — 用幂均值代替方差, 对非零偏数据更鲁棒
    norm(x) = scale * x / (E[|x|^p])^(1/p), p=2默认"""
    def __init__(self, dim, p=2.0, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.p = p
        self.eps = eps

    def forward(self, x):
        power_mean = torch.mean(torch.abs(x) ** self.p, dim=-1, keepdim=True)
        denom = torch.pow(power_mean + self.eps, 1.0 / self.p)
        return self.scale * x / denom


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    solstice: PowerNorm(x - SiLU(y) * α), α可学习标量初始化0.85"""
    def __init__(self, input_shape):
        super().__init__()
        self.norm = PowerNorm(input_shape[-1])
        self.act = nn.SiLU()
        # solstice: 可学习残差缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.85))

    def forward(self, x, y):
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        u = x - self.act(y) * alpha
        u = self.norm(u)
        _adbg("residual_norm", u)
        _adbg("alpha", alpha)
        return u
