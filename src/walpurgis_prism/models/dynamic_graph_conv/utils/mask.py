"""Prism mask: with soft thresholding for contrastive-aware edges."""
import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Prism特有: 可学习的soft threshold
        self.soft_threshold = nn.Parameter(
            torch.tensor(1e-7))

    def _mask(self, index, adj):
        threshold = torch.abs(self.soft_threshold)
        mask = self.mask[index] + torch.ones_like(
            self.mask[index]) * threshold
        return mask.to(adj.device) * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result
