"""
Walpurgis Mask — Adaptive Sparsification
==========================================
Derived from D2STGNN mask.py.

Change: uses a learnable temperature parameter for the soft-threshold
instead of fixed top-k masking. The network learns how aggressive
the sparsification should be.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    """Sparsify distance matrix using soft thresholding.
    
    Upstream D2STGNN uses hard top-k masking. Walpurgis uses a learnable
    temperature-controlled sigmoid: entries below threshold get suppressed
    but not hard-zeroed, preserving gradient flow.
    
    mask(x) = x * sigmoid(τ * (x - threshold))
    where τ (temperature) is learnable.
    """
    
    _call_count = 0
    
    def __init__(self, **model_args):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.threshold_ratio = nn.Parameter(torch.tensor(0.5))  # fraction of max to keep
        self._debug_on = True
    
    def forward(self, dist_mx):
        """
        Args:
            dist_mx: [B, N, N] pairwise distances
        Returns:
            masked: [B, N, N] sparsified distances
        """
        Mask._call_count += 1
        
        # Compute adaptive threshold as fraction of per-row max
        row_max = dist_mx.max(dim=-1, keepdim=True)[0]
        threshold = torch.sigmoid(self.threshold_ratio) * row_max
        
        # Soft mask: sigmoid with learnable temperature
        tau = F.softplus(self.temperature)  # ensure positive
        mask = torch.sigmoid(tau * (dist_mx - threshold))
        masked = dist_mx * mask
        
        if self._debug_on and Mask._call_count % 200 == 1:
            density = (masked.abs() > 1e-6).float().mean().item()
            print(f"        [Mask #{Mask._call_count}] "
                  f"τ={tau.item():.2f} thresh_ratio={torch.sigmoid(self.threshold_ratio).item():.4f} "
                  f"density={density:.4f}")
        
        return masked
