"""
Walpurgis Graph Normalizer & Multi-Order Expansion
=====================================================
Adapted from D2STGNN Normalizer + MultiOrder.

Algorithm changes:
  1. Normalizer: symmetric normalization D^{-1/2} A D^{-1/2} instead of
     left-only D^{-1}A. Symmetric normalization preserves eigenvalue
     symmetry and is more stable for deep GCN stacks.
  2. MultiOrder: exponential decay on higher-order powers to prevent
     over-smoothing. Each k-th power is weighted by gamma^(k-1).
  3. Both include NaN/Inf guards with diagnostic output.
"""

import time
import torch
import torch.nn as nn

from utils.cal_adj import remove_nan_inf


class Normalizer(nn.Module):
    """Row-normalize graph adjacency matrices.

    Walpurgis change: uses symmetric normalization D^{-1/2} A D^{-1/2}
    instead of D2STGNN's left normalization D^{-1}A.
    Symmetric form has better spectral properties for gradient flow.
    """

    _call_count = 0

    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        """Symmetric normalization: D^{-1/2} A D^{-1/2}.

        Falls back to left-normalization if symmetric causes issues
        (e.g., when the graph has isolated nodes with degree=0).
        """
        degree = torch.sum(graph, dim=2)
        # D^{-1/2}
        deg_inv_sqrt = remove_nan_inf(1.0 / torch.sqrt(degree.clamp(min=1e-10)))
        deg_inv_sqrt_mat = torch.diag_embed(deg_inv_sqrt)

        # Symmetric: D^{-1/2} A D^{-1/2}
        normed_graph = torch.bmm(torch.bmm(deg_inv_sqrt_mat, graph), deg_inv_sqrt_mat)

        # Safety: if symmetric norm produced NaN (rare edge case with
        # zero-degree nodes), fall back to left-norm
        if torch.isnan(normed_graph).any():
            degree_inv = remove_nan_inf(1.0 / degree)
            degree_mat = torch.diag_embed(degree_inv)
            normed_graph = torch.bmm(degree_mat, graph)
            if Normalizer._call_count <= 5:
                print(f"  [Normalizer] ⚠ symmetric norm produced NaN, fell back to left-norm")

        return normed_graph

    def forward(self, adj):
        Normalizer._call_count += 1
        _verbose = (Normalizer._call_count <= 3 or Normalizer._call_count % 500 == 0)

        results = []
        for i, a in enumerate(adj):
            t0 = time.perf_counter()
            normed = self._norm(a)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if _verbose:
                row_sums = normed.sum(dim=-1)
                print(f"  [Normalizer] adj[{i}] row_sum: mean={row_sums.mean().item():.6f} "
                      f"std={row_sums.std().item():.6f} elapsed={elapsed_ms:.3f}ms")

            results.append(normed)
        return results


class MultiOrder(nn.Module):
    """Compute multi-order graph powers for k-hop neighborhoods.

    Walpurgis change: exponential decay weighting on higher orders.
    D2STGNN applies raw matrix powers A^k which grow exponentially,
    causing over-smoothing at k >= 3. We weight each order by
    gamma^(k-1) where gamma ∈ (0, 1).
    """

    _call_count = 0

    def __init__(self, order=2, gamma=0.7):
        super().__init__()
        self.order = order
        self.gamma = gamma  # decay factor per hop
        print(f"[Walpurgis::MultiOrder] init order={order} gamma={gamma}")

    def _multi_order(self, graph):
        """Compute graph powers A^1, A^2, ..., A^k with decay weighting."""
        graph_ordered = []
        k_1_order = graph
        mask = 1 - torch.eye(graph.shape[1]).to(graph.device)

        # Order 1: full weight
        graph_ordered.append(k_1_order * mask)

        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            # Walpurgis: decay weight for higher orders
            decay = self.gamma ** (k - 1)
            graph_ordered.append(k_1_order * mask * decay)

        return graph_ordered

    def forward(self, adj):
        MultiOrder._call_count += 1
        _verbose = (MultiOrder._call_count <= 3 or MultiOrder._call_count % 500 == 0)

        results = []
        for i, a in enumerate(adj):
            orders = self._multi_order(a)

            if _verbose:
                norms = [o.norm().item() for o in orders]
                print(f"  [MultiOrder] adj[{i}] order norms: "
                      + " ".join(f"k{k+1}={n:.4f}" for k, n in enumerate(norms))
                      + f" (decay γ={self.gamma})")

            results.append(orders)
        return results
