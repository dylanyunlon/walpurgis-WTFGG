import torch
import torch.nn as nn
import torch.nn.utils as utils
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:resdecomp:{tag}] mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)


class WeightNormLinear(nn.Module):
    """equinox: WeightNorm替代BatchNorm/LayerNorm
    对权重矩阵做方向/幅度分解: w = g * v/||v||
    无需batch统计量, 训练更稳定, 适合小batch场景"""
    def __init__(self, in_features, out_features):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        self.layer = utils.weight_norm(linear, name='weight')

    def forward(self, x):
        return self.layer(x)


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    equinox: WeightNorm投影(x - Mish(y) * α), α可学习标量"""
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        # equinox: WeightNorm替代LayerNorm/RMSNorm
        self.wn_proj = WeightNormLinear(dim, dim)
        # equinox: Mish激活 = x * tanh(softplus(x))
        self.act = nn.Mish()
        # equinox: 可学习残差缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        u = x - self.act(y) * alpha
        u = self.wn_proj(u)
        _edbg("residual_wn", u)
        _edbg("alpha", alpha)
        return u
