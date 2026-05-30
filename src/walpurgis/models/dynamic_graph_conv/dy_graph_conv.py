import time

import torch.nn as nn

from .utils import DistanceFunction, Mask, Normalizer, MultiOrder

# ── Walpurgis tier thresholds for graph construction ──
_GRAPH_HBM_MS  = 5.0
_GRAPH_GDDR_MS = 2.0


class DynamicGraphConstructor(nn.Module):
    """Dynamic graph learning module — constructs adaptive spatial graphs
    from node embeddings, time features, and historical data.

    Walpurgis adaptation:
    - Full pipeline timing: distance → mask → normalize → multi-order → ST-localize
    - Each stage individually profiled for tier placement decisions
    - Graph sparsity tracked (ratio of near-zero entries) to estimate
      effective memory footprint for heterogeneous placement
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        # model args
        self.k_s = model_args['k_s']  # spatial order
        self.k_t = model_args['k_t']  # temporal kernel size
        # hidden dimension
        self.hidden_dim = model_args['num_hidden']
        # trainable node embedding dimension
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

        print(f"[Walpurgis::DynamicGraphConstructor] init k_s={self.k_s} k_t={self.k_t} "
              f"hidden_dim={self.hidden_dim} node_dim={self.node_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Walpurgis::DynamicGraphConstructor] total params={total_params:,}")

    def st_localization(self, graph_ordered):
        """Spatial-temporal localization: expand graph adjacency across
        temporal kernel dimension for localized ST convolution."""
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
        """Dynamic graph learning module.

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            list: dynamic graphs for ST convolution
        """
        DynamicGraphConstructor._call_count += 1
        _verbose = (DynamicGraphConstructor._call_count <= 3 or
                     DynamicGraphConstructor._call_count % 200 == 0)
        timings = {}

        X   = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']

        if _verbose:
            print(f"[Walpurgis::DynGraphCtor::forward] call#{DynamicGraphConstructor._call_count} "
                  f"X shape={list(X.shape)} E_d={list(E_d.shape)} E_u={list(E_u.shape)}")

        # distance calculation
        t0 = time.perf_counter()
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        timings['distance'] = (time.perf_counter() - t0) * 1000

        # mask
        t0 = time.perf_counter()
        dist_mx = self.mask(dist_mx)
        timings['mask'] = (time.perf_counter() - t0) * 1000

        # normalization
        t0 = time.perf_counter()
        dist_mx = self.normalizer(dist_mx)
        timings['normalize'] = (time.perf_counter() - t0) * 1000

        # multi order
        t0 = time.perf_counter()
        mul_mx = self.multi_order(dist_mx)
        timings['multi_order'] = (time.perf_counter() - t0) * 1000

        # spatial temporal localization
        t0 = time.perf_counter()
        dynamic_graphs = self.st_localization(mul_mx)
        timings['st_local'] = (time.perf_counter() - t0) * 1000

        total_ms = sum(timings.values())

        if _verbose:
            tier = ("HBM" if total_ms >= _GRAPH_HBM_MS else
                    ("GDDR" if total_ms >= _GRAPH_GDDR_MS else "DRAM"))
            print(f"  [DynGraph timing] "
                  f"dist={timings['distance']:.3f}ms mask={timings['mask']:.3f}ms "
                  f"norm={timings['normalize']:.3f}ms multi={timings['multi_order']:.3f}ms "
                  f"st_loc={timings['st_local']:.3f}ms TOTAL={total_ms:.3f}ms → tier={tier}")
            print(f"  [DynGraph output] num_graphs={len(dynamic_graphs)} "
                  f"graph[0] shape={list(dynamic_graphs[0].shape) if dynamic_graphs else 'EMPTY'}")
            # Sparsity analysis on first graph
            if dynamic_graphs:
                g0 = dynamic_graphs[0]
                sparsity = (g0.abs() < 1e-6).float().mean().item()
                print(f"  [DynGraph sparsity] graph[0] zero_ratio={sparsity:.4f} "
                      f"(effective density={1.0 - sparsity:.4f})")

        return dynamic_graphs
