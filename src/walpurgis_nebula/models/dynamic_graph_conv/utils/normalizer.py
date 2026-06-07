"""Nebula normalizer: degree-aware D^{-alpha} A D^{-(1-alpha)} + MultiOrder."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

def _remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


class Normalizer(nn.Module):
    """Degree-aware asymmetric normalization: D^{-alpha} A D^{-(1-alpha)}.
    Learnable alpha interpolates between left-norm (alpha=1) and right-norm (alpha=0).
    Generalizes symmetric (alpha=0.5) and random-walk (alpha=1) normalizations."""
    def __init__(self):
        super().__init__()
        # Learnable interpolation parameter, initialized to 0.5 (symmetric)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def _norm(self, graph):
        alpha = torch.sigmoid(self.log_alpha)  # alpha in (0, 1)
        degree = torch.sum(graph, dim=2)  # [B, N]
        degree = _remove_nan_inf(degree)
        # D^{-alpha}
        d_left = _remove_nan_inf(degree.pow(-alpha))
        d_left = torch.diag_embed(d_left)
        # D^{-(1-alpha)}
        d_right = _remove_nan_inf(degree.pow(-(1.0 - alpha)))
        d_right = torch.diag_embed(d_right)
        # D^{-alpha} A D^{-(1-alpha)}
        normed = torch.bmm(torch.bmm(d_left, graph), d_right)
        if _NEB_DBG:
            print(f"[NEB:degree_norm@normalizer] alpha={alpha.item():.4f} "
                  f"normed_range=[{normed.min().item():.4f},{normed.max().item():.4f}]", file=sys.stderr)
        return normed

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
