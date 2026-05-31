"""
Walpurgis Normalizer and Multi-Order — Graph Normalization Utilities
=====================================================================
Derived from D2STGNN normalizer.py.

Changes:
  - Normalizer: uses symmetric normalization D^{-1/2} A D^{-1/2} instead
    of row normalization D^{-1}A for better spectral properties
  - MultiOrder: adds spectral radius logging for debugging
"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Symmetric graph normalization: D^{-1/2} A D^{-1/2}.
    
    Upstream D2STGNN uses row normalization (D^{-1}A), which creates
    asymmetric transition matrices. Symmetric normalization preserves
    the spectral structure of the graph and typically gives better
    gradient properties in GCN layers.
    """
    
    _call_count = 0
    
    def __init__(self):
        super().__init__()
        self._debug_on = True
    
    def forward(self, adj):
        """
        Args:
            adj: [B, N, N] adjacency matrix
        Returns:
            normalized: [B, N, N] symmetrically normalized
        """
        Normalizer._call_count += 1
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degree = adj.abs().sum(dim=-1).clamp(min=1e-8)  # [B, N]
        d_inv_sqrt = degree.pow(-0.5)  # [B, N]
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), 
                                 torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        
        # D^{-1/2} A D^{-1/2} via element-wise: (d_i^{-1/2} * a_ij * d_j^{-1/2})
        normalized = adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        
        if self._debug_on and Normalizer._call_count % 200 == 1:
            print(f"        [SymNorm #{Normalizer._call_count}] "
                  f"in_range=[{adj.min().item():.4f},{adj.max().item():.4f}] "
                  f"out_range=[{normalized.min().item():.4f},{normalized.max().item():.4f}] "
                  f"degree_μ={degree.mean().item():.4f}")
        
        return normalized


class MultiOrder(nn.Module):
    """Compute multi-order graph powers: A, A², ..., Aᵏ.
    
    Each order captures k-hop neighborhood information.
    """
    
    _call_count = 0
    
    def __init__(self, order=2):
        super().__init__()
        self.order = order
        self._debug_on = True
    
    def forward(self, adj):
        """
        Args:
            adj: [B, N, N] normalized adjacency
        Returns:
            list of [list of [B, N, N]] — [[A¹, A², ..., Aᵏ]]
        """
        MultiOrder._call_count += 1
        
        powers = [adj]
        current = adj
        for k in range(2, self.order + 1):
            current = torch.bmm(adj, current)
            powers.append(current)
        
        if self._debug_on and MultiOrder._call_count % 200 == 1:
            norms = [p.norm().item() for p in powers]
            print(f"        [MultiOrder #{MultiOrder._call_count}] "
                  f"order={self.order} power_norms={['%.3f'%n for n in norms]}")
        
        return [powers]  # wrap in list for compatibility with downstream
