import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

# Delta vs upstream:
#   1. st_localization uses mean pooling fallback when graph is sparse

class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask       = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                g = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1)
                g = g.reshape(g.shape[0], g.shape[1],
                              g.shape[2] * g.shape[3])
                # ── delta 1: densify very sparse slices ──
                density = (g.abs() > 1e-7).float().mean(dim=-1, keepdim=True)
                sparse_mask = (density < 0.01).float()
                uniform = 1.0 / max(g.shape[-1], 1)
                g = g * (1 - sparse_mask) + uniform * sparse_mask
                st_local_graph.append(g)
        return st_local_graph

    def forward(self, **inputs):
        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx  = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs
