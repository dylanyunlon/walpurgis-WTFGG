"""
Estimation gate: learns a sigmoid gate that modulates the proportion
of diffusion vs. inherent signals in the decoupled representation.
"""

import torch
import torch.nn as nn
import sys

_DBG_GATE = ("--debug-gate" in sys.argv) or False


class EstimationGate(nn.Module):
    """Produce a (0,1) gate from node embeddings + temporal embeddings."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        input_dim = 2 * node_emb_dim + 2 * time_emb_dim
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb_u, node_emb_d, tod_feat, dow_feat, history):
        """
        Parameters
        ----------
        node_emb_u, node_emb_d : [N, d_node]
        tod_feat, dow_feat     : [B, L, N, d_time]
        history                : [B, L', N, D]

        Returns
        -------
        gated_history : [B, L', N, D]  — element-wise gated input
        """
        B, L, _, _ = tod_feat.shape

        # broadcast node embeddings to (B, L, N, d_node)
        eu = node_emb_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = node_emb_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        combined = torch.cat([tod_feat, dow_feat, eu, ed], dim=-1)
        h = self.act(self.linear_1(combined))
        gate = torch.sigmoid(self.linear_2(h))

        # align temporal dimension with history
        gate = gate[:, -history.shape[1]:, :, :]
        gated = history * gate

        if _DBG_GATE:
            print(f"[DBG:gate] EstimationGate  gate_mean={gate.mean().item():.4f}  "
                  f"gate_std={gate.std().item():.4f}  "
                  f"history_shape={tuple(history.shape)}  "
                  f"gated_norm={gated.norm().item():.4f}")
        return gated
