"""
Dynamic graph constructor: distance → mask → normalize → multi-order → ST localize.
"""
import sys
import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_DBG = ("--debug-dygraph" in sys.argv)


class DynamicGraphConstructor(nn.Module):

    def __init__(self, **kw):
        super().__init__()
        self.k_s = kw['k_s']
        self.k_t = kw['k_t']
        self.hidden_dim = kw['num_hidden']
        self.node_dim   = kw['node_hidden']

        self.dist_fn    = DistanceFunction(**kw)
        self.mask_fn    = Mask(**kw)
        self.norm_fn    = Normalizer()
        self.multiord   = MultiOrder(order=self.k_s)

    def _st_localize(self, ordered_graphs):
        """Reshape (N, N) multi-order graphs into (B, N, k_t*N) for ST conv."""
        localized = []
        for modality in ordered_graphs:
            for g_k in modality:
                g_exp = g_k.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                flat = g_exp.reshape(
                    g_exp.shape[0], g_exp.shape[1],
                    g_exp.shape[2] * g_exp.shape[3])
                localized.append(flat)
        if _DBG:
            print(f"[DBG:dygraph] st_localize  "
                  f"n_localized={len(localized)}  "
                  f"shape_0={tuple(localized[0].shape) if localized else 'empty'}")
        return localized

    def forward(self, **inputs):
        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist = self.dist_fn(X, E_d, E_u, T_D, D_W)
        masked  = self.mask_fn(dist)
        normed  = self.norm_fn(masked)
        powered = self.multiord(normed)
        graphs  = self._st_localize(powered)

        if _DBG:
            print(f"[DBG:dygraph] forward  "
                  f"n_dist={len(dist)}  n_output_graphs={len(graphs)}")

        return graphs
