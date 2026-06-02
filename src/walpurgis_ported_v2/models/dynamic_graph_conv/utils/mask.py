"""
Topology mask: enforce dynamic graphs to only have edges where
the predefined adjacency has non-zero entries (+ epsilon smoothing).
"""

import torch
import torch.nn as nn
import sys

_DBG_MASK = ("--debug-mask" in sys.argv) or False


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.predefined_adj = model_args['adjs']

    def _apply_mask(self, idx, learned_adj):
        """Multiply learned adjacency by the predefined topology + eps."""
        topo = self.predefined_adj[idx] + torch.ones_like(self.predefined_adj[idx]) * 1e-7
        masked = topo.to(learned_adj.device) * learned_adj
        if _DBG_MASK:
            nnz_ratio = (topo > 1e-6).float().mean().item()
            print(f"[DBG:mask] _apply_mask  idx={idx}  "
                  f"topo_nnz_ratio={nnz_ratio:.4f}  "
                  f"out_norm={masked.norm().item():.4f}")
        return masked

    def forward(self, adj_list):
        return [self._apply_mask(i, a) for i, a in enumerate(adj_list)]
