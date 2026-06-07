"""Eclipse normalizer: symmetric D^{-1/2}AD^{-1/2} + MultiOrder."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def _remove_nan_inf(tensor):
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

class Normalizer(nn.Module):
    def __init__(self): super().__init__()
    def _norm(self, graph):
        # Symmetric normalization D^{-1/2} A D^{-1/2} (vs upstream D^{-1}A)
        degree = torch.sum(graph, dim=2)
        d_inv_sqrt = _remove_nan_inf(1.0 / torch.sqrt(degree + 1e-8))
        d_left = torch.diag_embed(d_inv_sqrt)
        d_right = torch.diag_embed(d_inv_sqrt)
        normed = torch.bmm(torch.bmm(d_left, graph), d_right)
        if _ECL_DBG: print(f"[ECL:normalizer] row_sum_mean={normed.sum(dim=2).mean().item():.4f}", file=sys.stderr)
        return normed
    def forward(self, adj):
        return [self._norm(a) for a in adj]

class MultiOrder(nn.Module):
    def __init__(self, order=2): super().__init__(); self.order = order
    def _multi_order(self, graph):
        ordered = []
        k1 = graph; mask = 1 - torch.eye(graph.shape[1]).to(graph.device)
        ordered.append(k1 * mask)
        for k in range(2, self.order + 1):
            k1 = torch.matmul(k1, graph); ordered.append(k1 * mask)
        return ordered
    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
