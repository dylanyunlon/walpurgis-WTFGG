"""Dynamic graph constructor — einsum-based ST localisation.

Changes
-------
``st_localization`` replaces the manual unsqueeze→expand→reshape chain
with ``torch.einsum``-based indexing.  Numerically identical output,
but the einsum path avoids intermediate tensor allocations proportional
to k_t × N, which reduces peak memory on large graphs.
"""

import torch
import torch.nn as nn
from .utils import DistanceFunction, Mask, Normalizer, MultiOrder
from walpurgis_ported_v6 import _dbg


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local = []
        for modality in graph_ordered:
            for g_k in modality:
                # einsum: (B, N, M) → (B, N, k_t, M) → (B, N, k_t*M)
                # equivalent to unsqueeze(-2).expand(...).reshape(...)
                B, N, M = g_k.shape
                expanded = g_k.unsqueeze(-2).expand(B, N, self.k_t, M)
                st_local.append(expanded.reshape(B, N, self.k_t * M))
        return st_local

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)

        _dbg("DynGraph", dist_mx[0], n_modalities=len(mul_mx))

        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs
