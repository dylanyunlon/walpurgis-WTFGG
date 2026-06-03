"""Residual decomposition — with scaling factor and pre-LN option.

Changes
-------
1. Learnable ``scale`` parameter (init 1.0) multiplies the residual before
   LayerNorm.  In deep stacks the network can learn to attenuate residuals
   from early layers, preventing them from overwhelming later signals.
2. ``pre_ln`` flag: when True, LayerNorm is applied *before* the
   subtraction (Pre-LN style).  Default False keeps Post-LN behaviour.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class ResidualDecomp(nn.Module):

    def __init__(self, input_shape, pre_ln=False):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))     # ← new
        self.pre_ln = pre_ln

    def forward(self, x, y):
        if self.pre_ln:
            x = self.ln(x)
        residual = x - self.ac(y)
        residual = residual * self.scale
        if not self.pre_ln:
            residual = self.ln(residual)

        _dbg("ResDecomp", residual, scale=self.scale.item())
        return residual
