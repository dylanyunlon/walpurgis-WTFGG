"""
Mask — walpurgis_ported_v4
Modifications:
  - Mask gating: added learnable scalar alpha per adjacency matrix that
    modulates mask strength via sigmoid(alpha), allowing the model to
    learn how aggressively to mask (original: hard binary mask)
  - forward() prints mask coverage statistics
"""
import torch
import torch.nn as nn
import sys

_V4_DEBUG = True


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # v4: learnable mask strength per adjacency matrix
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(3.0))  # sigmoid(3)≈0.95, starts near hard mask
            for _ in self.mask
        ])

    def _mask(self, index, adj):
        base_mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        # v4: soft gating — sigmoid(alpha) controls mask strength
        gate = torch.sigmoid(self.alpha[index])
        soft_mask = gate * base_mask + (1.0 - gate) * 1e-7
        result = soft_mask.to(adj.device) * adj

        if _V4_DEBUG:
            coverage = (base_mask > 1e-6).float().mean().item()
            print(f"[v4-DBG][Mask._mask] idx={index} "
                  f"alpha={self.alpha[index].item():.4f} "
                  f"gate={gate.item():.4f} "
                  f"mask_coverage={coverage:.4f} "
                  f"adj_shape={tuple(adj.shape)}",
                  file=sys.stderr)
        return result

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._mask(index, a))
        return result
