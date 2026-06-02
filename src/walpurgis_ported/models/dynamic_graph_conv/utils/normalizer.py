"""
Walpurgis v4 Normalizer & MultiOrder — Spectral-Aware Mixed-Norm
==========================================================================
Delta vs v3:
  - Mixed norm → *spectral-aware mixed norm* that additionally monitors
    between symmetric (D^{-0.5} A D^{-0.5}) and random-walk (D^{-1} A)
    via  α·sym + (1-α)·rw  with α learned.  This is strictly more
    expressive than the single-exponent approach since the two norms
    have qualitatively different spectral properties.
  - MultiOrder: added geometric decay weighting on higher-order powers
    with learnable decay rate β — prevents A^k from dominating when k
    is large and the graph has high spectral radius.
  - Spectral radius proxy printed every 200 calls for trend monitoring.

Breakpoint helpers:
    norm._diag_last            # last forward diagnostics
    multi._power_norms         # list of per-order norms from last call
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalizer(nn.Module):
    """Mixed symmetric/random-walk normalisation with learned interpolation."""

    _n = 0

    def __init__(self):
        super().__init__()
        # Interpolation weight: sigmoid → [0, 1], init ≈ 0.5
        self._alpha_logit = nn.Parameter(torch.tensor(0.0))
        self._debug = True
        self._diag_last = {}

    def _sym_norm(self, adj):
        """D^{-1/2} A D^{-1/2}"""
        deg = adj.abs().sum(dim=-1).clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        return adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)

    def _rw_norm(self, adj):
        """D^{-1} A (random walk)"""
        deg = adj.abs().sum(dim=-1).clamp(min=1e-8)
        d_inv = deg.pow(-1.0)
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros_like(d_inv), d_inv)
        return adj * d_inv.unsqueeze(-1)

    def forward(self, adj):
        Normalizer._n += 1
        alpha = torch.sigmoid(self._alpha_logit)  # weight for symmetric

        sym = self._sym_norm(adj)
        rw = self._rw_norm(adj)
        normed = alpha * sym + (1.0 - alpha) * rw

        if self._debug and Normalizer._n % 200 == 1:
            with torch.no_grad():
                deg = adj.abs().sum(dim=-1)
                self._diag_last = {
                    "step": Normalizer._n,
                    "alpha": round(alpha.item(), 4),
                    "sym_range": (round(sym.min().item(), 5), round(sym.max().item(), 5)),
                    "rw_range": (round(rw.min().item(), 5), round(rw.max().item(), 5)),
                    "out_range": (round(normed.min().item(), 5), round(normed.max().item(), 5)),
                    "degree_mean": round(deg.mean().item(), 4),
                }
                d = self._diag_last
                print(
                    f"        [MixedNorm #{Normalizer._n}] α={d['alpha']:.4f} "
                    f"(sym={d['alpha']:.1%} rw={1-d['alpha']:.1%}) | "
                    f"out∈[{d['out_range'][0]:.4f},{d['out_range'][1]:.4f}] "
                    f"deg_μ={d['degree_mean']:.4f}"
                )
        return normed


class MultiOrder(nn.Module):
    """Multi-hop graph powers with geometric decay weighting.

    A^k is weighted by β^(k-1) where β ∈ (0.3, 0.95) is learned.
    This prevents spectral explosion in higher orders.
    """

    _n = 0

    def __init__(self, order=2):
        super().__init__()
        self.order = order
        self._decay_logit = nn.Parameter(torch.tensor(1.0))  # sigmoid → ~0.73
        self._debug = True
        self._power_norms = []

    def forward(self, adj):
        MultiOrder._n += 1
        # Decay rate β ∈ [0.3, 0.95]
        beta = torch.sigmoid(self._decay_logit) * 0.65 + 0.3

        powers = [adj]
        cur = adj
        weights = [1.0]
        self._power_norms = [adj.norm().item()]

        for k in range(2, self.order + 1):
            cur = torch.bmm(adj, cur)
            w_k = beta.pow(k - 1)
            powers.append(cur * w_k)
            weights.append(w_k.item())
            self._power_norms.append(cur.norm().item())

        if self._debug and MultiOrder._n % 200 == 1:
            raw_norms = [f"{n:.3f}" for n in self._power_norms]
            weighted = [f"{n * w:.3f}" for n, w in zip(self._power_norms, weights)]
            rs = [p.abs().sum(-1).max().item() for p in powers]
            print(
                f"        [MultiOrder #{MultiOrder._n}] order={self.order} "
                f"β={beta.item():.3f} | "
                f"raw_norms={raw_norms} weighted={weighted} | "
                f"spectral_proxy={['%.2f' % r for r in rs]}"
            )
        return [powers]
