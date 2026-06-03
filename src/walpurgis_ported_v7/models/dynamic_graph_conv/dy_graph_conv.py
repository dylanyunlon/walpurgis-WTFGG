import torch.nn as nn
import sys

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_DBG_DYGRAPH = ("--dbg-dygraph" in sys.argv)


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
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)

        if _DBG_DYGRAPH:
            import torch
            with torch.no_grad():
                for i, d in enumerate(dist_mx):
                    print(f"[DBG-DYGRAPH] dist_mx[{i}] shape={list(d.shape)}  "
                          f"mean={d.mean().item():.5f}  "
                          f"sparsity={(d < 1e-4).float().mean().item():.3f}")

        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)

        if _DBG_DYGRAPH:
            import torch
            with torch.no_grad():
                print(f"[DBG-DYGRAPH] num_dynamic_graphs={len(dynamic_graphs)}  "
                      f"shapes={[list(g.shape) for g in dynamic_graphs[:2]]}")

        return dynamic_graphs
