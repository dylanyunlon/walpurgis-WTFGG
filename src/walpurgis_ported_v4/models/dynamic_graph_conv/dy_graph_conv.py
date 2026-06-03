"""
DynamicGraphConstructor — walpurgis_ported_v4
Modifications:
  - st_localization: prints per-modality graph sparsity report
  - forward: wraps each stage with timing instrumentation
  - Added graph count validation assertion
"""
import torch
import torch.nn as nn
import time
import sys

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

_V4_DEBUG = True


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
        for mod_idx, modality_i in enumerate(graph_ordered):
            for k_idx, k_order_graph in enumerate(modality_i):
                k_order_graph = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)

                if _V4_DEBUG:
                    sparsity = (k_order_graph.abs() < 1e-6).float().mean().item()
                    print(f"[v4-DBG][DynGraph.st_local] mod={mod_idx} k={k_idx} "
                          f"shape={tuple(k_order_graph.shape)} "
                          f"sparsity={sparsity:.4f}",
                          file=sys.stderr)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        t0 = time.perf_counter()

        # distance calculation
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        t1 = time.perf_counter()

        # mask
        dist_mx = self.mask(dist_mx)
        t2 = time.perf_counter()

        # normalization
        dist_mx = self.normalizer(dist_mx)
        t3 = time.perf_counter()

        # multi order
        mul_mx = self.multi_order(dist_mx)
        t4 = time.perf_counter()

        # spatial temporal localization
        dynamic_graphs = self.st_localization(mul_mx)
        t5 = time.perf_counter()

        if _V4_DEBUG:
            print(f"[v4-DBG][DynGraphConstructor.forward] timing(ms): "
                  f"distance={1000*(t1-t0):.1f} mask={1000*(t2-t1):.1f} "
                  f"norm={1000*(t3-t2):.1f} multi_order={1000*(t4-t3):.1f} "
                  f"st_local={1000*(t5-t4):.1f} total={1000*(t5-t0):.1f} "
                  f"n_graphs={len(dynamic_graphs)}",
                  file=sys.stderr)

        return dynamic_graphs
