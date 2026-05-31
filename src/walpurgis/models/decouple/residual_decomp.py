"""
Walpurgis Residual Decomposition — Learnable Residual Scaling
==============================================================
Derived from D2STGNN residual_decomp.py.

Change: uses learnable scaling factor (initialized to 0.5) instead of
simple subtraction. The network can learn to adjust how much of the
backcast signal is removed from the residual.
"""

import torch
import torch.nn as nn


class ResidualDecomp(nn.Module):
    """Decompose residual signal: residual = input - scale * backcast.
    
    Upstream D2STGNN uses: residual = LayerNorm(input - backcast).
    Walpurgis adds a learnable scale factor initialized to 0.5, which
    lets the network control decomposition aggressiveness. Early in
    training when backcast is noisy, scale < 1 preserves more of the
    input; as backcast improves, scale can approach 1.
    """
    
    _call_count = 0
    
    def __init__(self, input_channel):
        super().__init__()
        self.ln = nn.LayerNorm(input_channel)
        self.scale = nn.Parameter(torch.tensor(0.5))  # learnable decomposition strength
        self._debug_on = True
    
    def forward(self, input_data, backcast):
        """
        Args:
            input_data: [B, L, N, D]
            backcast:   [B, L, N, D] (same shape, predicted component to remove)
        Returns:
            residual:   [B, L, N, D]
        """
        ResidualDecomp._call_count += 1
        
        # Learnable scaling: how much of backcast to subtract
        s = torch.sigmoid(self.scale)  # bound to (0, 1)
        residual = self.ln(input_data - s * backcast)
        
        if self._debug_on and ResidualDecomp._call_count % 500 == 1:
            with torch.no_grad():
                ratio = backcast.norm().item() / (input_data.norm().item() + 1e-8)
                print(f"      [ResDecomp #{ResidualDecomp._call_count}] "
                      f"scale={s.item():.4f} | backcast/input ratio={ratio:.4f}")
        
        return residual
