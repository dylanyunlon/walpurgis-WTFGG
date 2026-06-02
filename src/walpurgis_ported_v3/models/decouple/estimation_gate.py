"""
Estimation gate: learns a scalar gate in (0,1) for each node-time pair
to blend the diffusion and inherent signals.
"""
import sys
import torch
import torch.nn as nn

_DBG = ("--debug-gate" in sys.argv)


class EstimationGate(nn.Module):

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        input_dim = 2 * node_emb_dim + 2 * time_emb_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, emb_u, emb_d, t_day, t_week, history):
        B, L, _, _ = t_day.shape
        # broadcast node embeddings to (B, L, N, d)
        eu = emb_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = emb_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        cat = torch.cat([t_day, t_week, eu, ed], dim=-1)
        h = self.act(self.fc1(cat))
        gate = torch.sigmoid(self.fc2(h))
        # align temporal dim with history
        gate = gate[:, -history.shape[1]:, :, :]

        if _DBG:
            print(f"[DBG:gate] EstimationGate  gate_mean={gate.mean().item():.4f}  "
                  f"gate_std={gate.std().item():.4f}  "
                  f"history.shape={tuple(history.shape)}")

        return history * gate
