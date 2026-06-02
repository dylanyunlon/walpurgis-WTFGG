"""
Structural mask: zero out dynamic-graph edges where predefined adj is absent.
"""
import sys
import torch
import torch.nn as nn

_DBG = ("--debug-mask" in sys.argv)


class Mask(nn.Module):

    def __init__(self, **kw):
        super().__init__()
        self.structural_mask = kw['adjs']

    def _apply_mask(self, idx, adj):
        template = self.structural_mask[idx] + torch.ones_like(self.structural_mask[idx]) * 1e-7
        masked = template.to(adj.device) * adj
        if _DBG:
            nnz = (masked > 1e-6).float().mean().item()
            print(f"[DBG:mask] _apply_mask idx={idx}  "
                  f"density={nnz:.4f}")
        return masked

    def forward(self, adj_list):
        return [self._apply_mask(i, a) for i, a in enumerate(adj_list)]
