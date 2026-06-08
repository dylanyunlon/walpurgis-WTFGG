"""
Cathexis ResidualDecomp — 算法改写 #2
upstream: LayerNorm(x - ReLU(y))
cathexis: AdaIN-style: normalize(x - Mish(y)) * adaptive_scale + adaptive_shift
Uses GroupNorm instead of InstanceNorm for numerical stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        feat_dim = input_shape[-1]
        # Use GroupNorm (num_groups=1 = equivalent to LayerNorm on last dim) 
        # but with adaptive modulation on top
        self.norm = nn.LayerNorm(feat_dim)
        self.style_scale = nn.Linear(feat_dim, feat_dim)
        self.style_shift = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, y):
        # Cathexis改写: Mish activation instead of ReLU
        residual = x - (y * torch.tanh(F.softplus(y)))
        normed = self.norm(residual)
        # Adaptive Instance Normalization: modulate with global statistics
        style_in = residual.mean(dim=(1, 2))  # [B, D]
        gamma = self.style_scale(style_in).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        beta = self.style_shift(style_in).unsqueeze(1).unsqueeze(2)
        return normed * (1.0 + gamma) + beta
