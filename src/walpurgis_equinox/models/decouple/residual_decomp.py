import torch
import torch.nn as nn
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:resdecomp:{tag}] mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)


class DenseNorm(nn.Module):
    """equinox: DenseNet-style归一化 — 拼接原始+变换后特征再投影,
    替代RMSNorm, 保留更多原始信息"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dense_proj = nn.Linear(dim * 2, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x_orig, x_trans):
        # DenseNet: concat原始和变换特征
        concat = torch.cat([x_orig, x_trans], dim=-1)
        fused = self.dense_proj(concat)
        # 然后做简单的均值归一化
        rms = torch.sqrt(torch.mean(fused ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * fused / rms


class ResidualDecomp(nn.Module):
    """upstream: LayerNorm(x - ReLU(y))
    equinox: DenseNorm(x, y) — DenseNet残差融合, α可学习标量初始化0.9"""
    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        self.norm = DenseNorm(dim)
        self.act = nn.GELU()
        # equinox: 可学习残差缩放系数
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        residual = x - self.act(y) * alpha
        u = self.norm(x, residual)
        _edbg("residual_norm", u)
        _edbg("alpha", alpha)
        return u
