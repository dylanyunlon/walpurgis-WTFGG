"""
Walpurgis v2 Normalizer & MultiOrder
======================================
Delta: symmetric norm → *degree-scalable* norm with learnable exponent γ.
D^{-γ} A D^{-(1-γ)}.  γ=0.5 recovers symmetric; γ=1.0 gives row-norm.
The network can interpolate between the two.
"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Degree-scalable normalisation: D^{-γ} A D^{-(1-γ)}.

    γ is a learnable scalar (sigmoid-bounded to [0.1, 0.9]).
    """

    _n = 0

    def __init__(self):
        super().__init__()
        self._gamma_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5 initially
        self._debug = True

    def forward(self, adj):
        Normalizer._n += 1
        gamma = torch.sigmoid(self._gamma_logit) * 0.8 + 0.1  # clamp to [0.1, 0.9]

        degree = adj.abs().sum(dim=-1).clamp(min=1e-8)
        d_left = degree.pow(-gamma)
        d_right = degree.pow(-(1.0 - gamma))
        d_left = torch.where(torch.isinf(d_left), torch.zeros_like(d_left), d_left)
        d_right = torch.where(torch.isinf(d_right), torch.zeros_like(d_right), d_right)

        normed = adj * d_left.unsqueeze(-1) * d_right.unsqueeze(-2)

        if self._debug and Normalizer._n % 200 == 1:
            print(
                f"        [DegNorm #{Normalizer._n}] γ={gamma.item():.4f} "
                f"in=[{adj.min().item():.4f},{adj.max().item():.4f}] "
                f"out=[{normed.min().item():.4f},{normed.max().item():.4f}] "
                f"deg_μ={degree.mean().item():.4f}"
            )

        return normed


class MultiOrder(nn.Module):
    """Multi-hop graph powers with spectral radius monitoring."""

    _n = 0

    def __init__(self, order=2):
        super().__init__()
        self.order = order
        self._debug = True

    def forward(self, adj):
        MultiOrder._n += 1
        powers = [adj]
        cur = adj
        for k in range(2, self.order + 1):
            cur = torch.bmm(adj, cur)
            powers.append(cur)

        if self._debug and MultiOrder._n % 200 == 1:
            norms = [f"{p.norm().item():.3f}" for p in powers]
            # Spectral radius proxy: max eigenvalue ≈ max row-sum
            rs = [p.abs().sum(-1).max().item() for p in powers]
            print(
                f"        [MultiOrder #{MultiOrder._n}] order={self.order} "
                f"norms={norms} spectral_proxy={['%.2f'%r for r in rs]}"
            )

        return [powers]
