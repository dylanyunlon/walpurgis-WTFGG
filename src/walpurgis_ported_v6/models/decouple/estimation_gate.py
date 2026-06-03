"""Estimation gate — GELU + temperature-scaled sigmoid.

Changes
-------
1. Hidden activation: ReLU → GELU.  GELU's smooth curvature near zero
   produces less gradient noise when the gate is near its decision boundary.
2. Final sigmoid gets a learnable temperature parameter ``tau`` (init=1.0).
   When tau < 1 the gate sharpens toward binary; when tau > 1 it softens.
   This gives the optimiser one more degree of freedom to control the
   diffusion/inherent split ratio.
3. Diagnostic: prints gate statistics (mean, std, % > 0.5) every call
   when WALPURGIS_DEBUG=1.
"""

import torch
import torch.nn as nn
from walpurgis_ported_v6 import _dbg


class EstimationGate(nn.Module):

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + 2 * time_emb_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()                         # ← was ReLU
        self.fc2 = nn.Linear(hidden_dim, 1)
        # learnable temperature for sigmoid sharpness
        self.tau = nn.Parameter(torch.ones(1))        # ← new

    def forward(self, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_emb_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_emb_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
        ], dim=-1)

        h = self.act(self.fc1(feat))
        # temperature-scaled sigmoid
        gate = torch.sigmoid(self.fc2(h) / (self.tau.abs() + 1e-6))
        gate = gate[:, -history_data.shape[1]:, :, :]

        _dbg("EstGate", gate, tau=self.tau.item(),
             frac_above_half=f"{(gate > 0.5).float().mean().item():.3f}")

        return history_data * gate
