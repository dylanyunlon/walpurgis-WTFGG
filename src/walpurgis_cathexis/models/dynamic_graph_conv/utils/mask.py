"""
Cathexis Mask — 算法改写 #5
upstream: element-wise multiply with predefined adj
cathexis: Bernoulli sampling with learned temperature per edge bucket
"""
import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # Cathexis改写: learned sampling temperature
        self.log_temperature = nn.Parameter(torch.tensor(1.0))

    def _mask(self, index, adj):
        # Use modulo to handle case where distance returns more adjs than predefined
        mask_idx = index % len(self.mask)
        base_mask = self.mask[mask_idx] + torch.ones_like(self.mask[mask_idx]) * 1e-7
        base_mask = base_mask.to(adj.device)
        # Broadcast: base_mask is [N,N], adj may be [B,N,N]
        if adj.dim() == 3 and base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0)
        # Cathexis改写: soft Bernoulli gating via sigmoid(mask * temperature)
        temperature = torch.exp(self.log_temperature).clamp(min=0.5, max=5.0)
        soft_gate = torch.sigmoid((base_mask - 0.5) * temperature)
        return soft_gate * adj

    def forward(self, adj):
        return [self._mask(i, a) for i, a in enumerate(adj)]
