"""
Dynamic graph constructor: learns time-varying adjacency matrices
from traffic signals, node embeddings, and temporal features,
then applies spatial-temporal localization for the diffusion conv.
"""

import torch.nn as nn
import sys

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_DBG_DGC = ("--debug-dgc" in sys.argv) or False


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.d_hidden = model_args['num_hidden']
        self.d_node   = model_args['node_hidden']

        self.dist_fn    = DistanceFunction(**model_args)
        self.topo_mask  = Mask(**model_args)
        self.row_norm   = Normalizer()
        self.order_exp  = MultiOrder(order=self.k_s)

    def _st_localize(self, graph_orders):
        """
        Reshape each k-order graph for the ST-localized convolution kernel.
        Input : list[list[Tensor]]  — [modality][order] each [B, N, N]
        Output: list[Tensor]        — each [B, N, k_t * N]
        """
        localized = []
        for modality_graphs in graph_orders:
            for g_k in modality_graphs:
                # expand temporal kernel dim: [B, N, k_t, N]
                expanded = g_k.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                # flatten last two dims: [B, N, k_t * N]
                flat = expanded.reshape(
                    expanded.shape[0], expanded.shape[1],
                    expanded.shape[2] * expanded.shape[3]
                )
                localized.append(flat)
        if _DBG_DGC:
            print(f"[DBG:dgc] st_localize  n_graphs={len(localized)}  "
                  f"shapes={[tuple(g.shape) for g in localized[:3]]}...")
        return localized

    def forward(self, **inputs):
        """
        Build dynamic graphs from current-batch context.

        Expected keys in *inputs*:
            history_data, node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat
        """
        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        # step 1: compute raw pairwise distance / similarity
        raw_adj = self.dist_fn(X, E_d, E_u, T_D, D_W)
        # step 2: mask by predefined topology
        masked  = self.topo_mask(raw_adj)
        # step 3: row-normalize
        normed  = self.row_norm(masked)
        # step 4: multi-order expansion
        orders  = self.order_exp(normed)
        # step 5: spatial-temporal localization
        dyn_graphs = self._st_localize(orders)

        if _DBG_DGC:
            print(f"[DBG:dgc] forward done  n_dynamic_graphs={len(dyn_graphs)}  "
                  f"batch={X.shape[0]}  nodes={X.shape[2]}")
        return dyn_graphs
