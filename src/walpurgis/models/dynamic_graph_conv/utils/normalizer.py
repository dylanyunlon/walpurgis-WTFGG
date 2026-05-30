import time

import torch
import torch.nn as nn

from utils.cal_adj import remove_nan_inf


class Normalizer(nn.Module):
    """Row-normalize adjacency matrices (D^{-1} A).

    Walpurgis notes:
    - remove_nan_inf is called on the degree inverse to handle isolated
      nodes (degree=0 → 1/0=inf → replaced with 0).
    - This is a lightweight operation; DRAM tier is sufficient.
    """

    _call_count = 0

    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        degree  = remove_nan_inf(1 / degree)
        degree  = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        Normalizer._call_count += 1
        _verbose = (Normalizer._call_count <= 3 or Normalizer._call_count % 500 == 0)

        t0 = time.perf_counter()
        result = [self._norm(_) for _ in adj]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"[Walpurgis::Normalizer::forward] call#{Normalizer._call_count} "
                  f"num_adj={len(adj)} elapsed={elapsed_ms:.3f}ms")
            for i, r in enumerate(result):
                row_sums = r.sum(dim=-1)
                print(f"  normed[{i}] row_sum mean={row_sums.mean().item():.4f} "
                      f"std={row_sums.std().item():.4f} "
                      f"(ideal=1.0 for non-isolated nodes)")

        return result


class MultiOrder(nn.Module):
    """Compute multi-order graph powers (A, A^2, ..., A^k) with self-loop removal.

    Walpurgis notes:
    - Higher-order powers amplify numerical errors; NaN/Inf checks are
      performed on each power.
    - Memory scales as O(k * N^2 * batch); for large graphs, k>3 may
      require GDDR+ tier placement.
    """

    _call_count = 0

    def __init__(self, order=2):
        super().__init__()
        self.order = order
        print(f"[Walpurgis::MultiOrder] init order={order}")

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph               # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        MultiOrder._call_count += 1
        _verbose = (MultiOrder._call_count <= 3 or MultiOrder._call_count % 500 == 0)

        t0 = time.perf_counter()
        result = [self._multi_order(_) for _ in adj]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"[Walpurgis::MultiOrder::forward] call#{MultiOrder._call_count} "
                  f"order={self.order} num_adj={len(adj)} elapsed={elapsed_ms:.3f}ms")
            # Check for NaN in higher-order powers
            for mod_i, modality in enumerate(result):
                for k_i, kg in enumerate(modality):
                    if torch.isnan(kg).any().item():
                        print(f"  ⚠ MultiOrder modality[{mod_i}] order[{k_i}] has NaN!")

        return result
