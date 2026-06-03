import torch
import torch.nn as nn

# Delta vs upstream:
#   1. mask epsilon: 1e-7 → 1e-5 (avoids vanishing on fp16 runs)

class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']

    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-5  # delta 1
        return mask.to(adj.device) * adj

    def forward(self, adj):
        return [self._mask(i, a) for i, a in enumerate(adj)]
