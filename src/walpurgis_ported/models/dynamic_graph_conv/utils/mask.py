"""
Walpurgis v2 Mask — Per-Row Adaptive Soft Thresholding
========================================================
Delta: global temperature → *per-row* adaptive threshold derived from
each row's statistics, giving heterogeneous sparsity across nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    """Per-row adaptive soft-threshold mask.

    Each row's threshold = sigmoid(α) × row_mean.
    Nodes with inherently sparser neighbourhoods get a lower threshold
    than hub nodes.  α is learnable.
    """

    _n = 0

    def __init__(self, **kw):
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(0.0))   # controls threshold level
        self._sharpness = nn.Parameter(torch.tensor(8.0))  # soft-mask steepness
        self._debug = True

    def forward(self, dist_mx):
        Mask._n += 1
        # Per-row adaptive threshold
        row_mean = dist_mx.mean(dim=-1, keepdim=True)  # [B, N, 1]
        thresh = torch.sigmoid(self._alpha) * row_mean

        tau = F.softplus(self._sharpness)
        soft_mask = torch.sigmoid(tau * (dist_mx - thresh))
        masked = dist_mx * soft_mask

        if self._debug and Mask._n % 200 == 1:
            dens = (masked.abs() > 1e-6).float().mean().item()
            print(
                f"        [Mask #{Mask._n}] α={torch.sigmoid(self._alpha).item():.4f} "
                f"τ={tau.item():.2f} density={dens:.4f} "
                f"thresh_μ={thresh.mean().item():.5f}"
            )

        return masked
