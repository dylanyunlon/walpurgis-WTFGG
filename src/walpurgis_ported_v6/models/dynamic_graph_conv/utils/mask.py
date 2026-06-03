"""Graph mask — soft-threshold variant.

Changes
-------
Instead of hard-multiplying by the binary adjacency mask (which forces
zero on all non-edges), this version uses a soft threshold:
``sigmoid(alpha * mask_val) * adj`` where alpha is a learnable sharpness
parameter.  At init alpha is large (10.0) so behaviour ≈ upstream, but
the optimiser can soften the boundary if cross-edge diffusion helps.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.register_buffer(
            '_raw_mask',
            torch.stack(model_args['adjs']))       # (n_adj, N, N)
        # learnable sharpness — init high for near-hard behaviour
        self.alpha = nn.Parameter(torch.tensor(10.0))

    def _soft_mask(self, idx, adj):
        raw = self._raw_mask[idx] + 1e-7
        soft = torch.sigmoid(self.alpha * raw)
        return soft.to(adj.device) * adj

    def forward(self, adj_list):
        out = []
        for i, a in enumerate(adj_list):
            out.append(self._soft_mask(i, a))
        _dbg("Mask", out[0], alpha=self.alpha.item())
        return out
