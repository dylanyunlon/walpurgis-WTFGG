"""
Row-normalizer and multi-order graph power series.
"""
import sys
import torch
import torch.nn as nn

from utils.cal_adj import remove_nan_inf

_DBG = ("--debug-norm" in sys.argv)


class Normalizer(nn.Module):
    """Row-normalize a batch of adjacency matrices: D^{-1} A."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _row_norm(graph):
        degree = graph.sum(dim=2)
        inv_deg = remove_nan_inf(1.0 / degree)
        D_inv = torch.diag_embed(inv_deg)
        normed = torch.bmm(D_inv, graph)
        return normed

    def forward(self, adj_list):
        out = [self._row_norm(a) for a in adj_list]
        if _DBG:
            for i, a in enumerate(out):
                print(f"[DBG:norm] matrix {i}  row_sum_mean="
                      f"{a.sum(-1).mean().item():.4f}")
        return out


class MultiOrder(nn.Module):
    """Compute [A, A^2, ..., A^order] with self-loop mask removed."""

    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _powers(self, graph):
        N = graph.shape[1]
        eye = torch.eye(N, device=graph.device)
        anti_eye = 1.0 - eye

        powers = []
        cur = graph
        powers.append(cur * anti_eye)
        for _k in range(2, self.order + 1):
            cur = torch.matmul(cur, graph)
            powers.append(cur * anti_eye)

        if _DBG:
            print(f"[DBG:norm] MultiOrder  N={N}  "
                  f"order={self.order}  powers_len={len(powers)}")
        return powers

    def forward(self, adj_list):
        return [self._powers(a) for a in adj_list]
