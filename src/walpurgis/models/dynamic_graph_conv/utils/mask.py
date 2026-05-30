"""
Walpurgis Graph Mask — Topology-Constrained Graph Filtering
=============================================================
Adapted from D2STGNN Mask.

Algorithm changes:
  1. Adaptive epsilon: instead of fixed 1e-7 floor, use a fraction of the
     mask's mean value. This prevents the epsilon from dominating when
     the predefined graph has very small edge weights.
  2. Soft thresholding: after masking, entries below a dynamic threshold
     (based on the masked matrix's statistics) are zeroed to promote
     sparsity. This reduces effective graph density without hard cutoffs.
  3. Per-mask statistics tracking for debug.
"""

import torch
import torch.nn as nn


class Mask(nn.Module):
    """Apply predefined adjacency masks to learned dynamic graphs.

    Walpurgis: adaptive epsilon + soft sparsification post-mask.
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        self._mask_stats = []  # track per-call sparsity changes

        # Precompute mask statistics for adaptive epsilon
        for i, m in enumerate(self.mask):
            m_abs_mean = m.abs().mean().item()
            nnz_ratio = (m.abs() > 1e-10).float().mean().item()
            print(f"[Walpurgis::Mask] predefined mask[{i}] shape={list(m.shape)} "
                  f"abs_mean={m_abs_mean:.6f} nnz_ratio={nnz_ratio:.4f}")

    def _mask(self, index, adj):
        """Apply mask with adaptive epsilon.

        D2STGNN uses fixed eps=1e-7 which can dominate for small-weight graphs.
        Walpurgis: eps = max(1e-8, 0.01 * mask_mean) — scales with the mask.
        """
        mask_template = self.mask[index]
        mask_mean = mask_template.abs().mean().item()
        eps = max(1e-8, 0.01 * mask_mean)
        mask = mask_template + torch.ones_like(mask_template) * eps
        return mask.to(adj.device) * adj

    def _soft_threshold(self, adj, percentile=0.1):
        """Zero out the bottom `percentile` fraction of entries.

        This promotes sparsity in the masked graph without requiring
        a fixed threshold. Adaptive to the actual value distribution.
        """
        with torch.no_grad():
            flat = adj.abs().view(-1)
            if flat.numel() == 0:
                return adj
            k = max(1, int(flat.numel() * percentile))
            threshold = flat.kthvalue(k).values.item()
        # Soft zero: entries below threshold are multiplied by a very small factor
        # rather than hard-zeroed, preserving gradient flow
        below_mask = (adj.abs() < threshold).float()
        adj = adj * (1.0 - below_mask * 0.99)  # keep 1% of sub-threshold signal
        return adj

    def forward(self, adj):
        Mask._call_count += 1
        _verbose = (Mask._call_count <= 3 or Mask._call_count % 500 == 0)

        result = []
        for index, a in enumerate(adj):
            # Pre-mask density
            pre_density = (a.abs() > 1e-8).float().mean().item() if _verbose else 0

            masked = self._mask(index, a)

            # Walpurgis: soft sparsification
            masked = self._soft_threshold(masked, percentile=0.1)

            # Post-mask density
            post_density = (masked.abs() > 1e-8).float().mean().item() if _verbose else 0

            if _verbose:
                print(f"  [Mask] adj[{index}] density: {pre_density:.4f} → {post_density:.4f} "
                      f"(sparsified {(pre_density - post_density)*100:.1f}%)")

            result.append(masked)

        return result
