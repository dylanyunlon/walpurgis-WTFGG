"""Meridian Mask — adaptive threshold instead of fixed predefined mask.
Changes vs upstream: applies soft threshold that removes bottom-p edges per sample."""
import torch
import torch.nn as nn
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # adaptive threshold: learn the pruning percentile
        self.threshold_logit = nn.Parameter(torch.tensor(0.0))

    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        masked = mask.to(adj.device) * adj
        # adaptive soft threshold: prune weakest edges
        thresh = torch.sigmoid(self.threshold_logit) * 0.3  # max 30% threshold
        flat = masked.reshape(masked.shape[0], -1)
        cutoff = torch.quantile(flat, thresh.item(), dim=-1, keepdim=True)
        cutoff = cutoff.unsqueeze(-1) if masked.dim() == 3 else cutoff
        soft_mask = torch.sigmoid((masked - cutoff.expand_as(masked)) * 10.0)
        result = masked * soft_mask
        if _DBG:
            sparsity = (result < 1e-6).float().mean().item()
            print(f"[MER:mask] idx={index} thresh={thresh.item():.4f} "
                  f"sparsity={sparsity:.3f}", file=sys.stderr)
        return result

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
