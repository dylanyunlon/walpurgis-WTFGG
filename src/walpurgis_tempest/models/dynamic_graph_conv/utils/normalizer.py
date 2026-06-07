"""Tempest normalizer: doubly stochastic via Sinkhorn normalization + MultiOrder.
Unlike upstream (row-normalization D^{-1}A) and eclipse (symmetric D^{-1/2}AD^{-1/2}),
Tempest uses Sinkhorn iterations to produce doubly stochastic matrices where both
rows and columns sum to 1. This preserves more structural information and provides
a balanced message-passing scheme across the graph."""
import torch, torch.nn as nn, sys, os
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

def _remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

class Normalizer(nn.Module):
    """Sinkhorn normalization: iteratively normalize rows and columns to produce
    a doubly stochastic matrix. num_iters controls convergence."""
    def __init__(self, num_iters=5):
        super().__init__()
        self.num_iters = num_iters

    def _sinkhorn_norm(self, graph):
        """Apply Sinkhorn iterations to make graph doubly stochastic.
        Each iteration: normalize rows, then normalize columns."""
        # Ensure non-negative
        graph = graph.clamp(min=0)
        eps = 1e-8
        for _ in range(self.num_iters):
            # Row normalization
            row_sum = graph.sum(dim=-1, keepdim=True).clamp(min=eps)
            graph = graph / row_sum
            # Column normalization
            col_sum = graph.sum(dim=-2, keepdim=True).clamp(min=eps)
            graph = graph / col_sum
        graph = _remove_nan_inf(graph)
        if _TEM_DBG:
            row_dev = (graph.sum(dim=-1) - 1.0).abs().mean().item()
            col_dev = (graph.sum(dim=-2) - 1.0).abs().mean().item()
            print(f"[TEM:sinkhorn@normalizer] row_dev={row_dev:.6f} col_dev={col_dev:.6f} "
                  f"iters={self.num_iters}", file=sys.stderr)
        return graph

    def forward(self, adj):
        return [self._sinkhorn_norm(a) for a in adj]

class MultiOrder(nn.Module):
    """Multi-order graph expansion using Chebyshev-inspired recurrence."""
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        ordered = []
        mask = 1 - torch.eye(graph.shape[1]).to(graph.device)
        t_prev = torch.eye(graph.shape[1]).to(graph.device)
        t_curr = graph
        ordered.append(t_curr * mask)
        for k in range(2, self.order + 1):
            # Chebyshev recurrence for consistent multi-order expansion
            t_next = 2.0 * torch.matmul(graph, t_curr) - t_prev
            ordered.append(t_next * mask)
            t_prev = t_curr; t_curr = t_next
        return ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
