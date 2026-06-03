import torch
import torch.nn as nn

# Delta vs upstream:
#   1. Gate activation: sigmoid → hard-sigmoid (cheaper, same expressiveness)
#   2. Hidden projection adds LayerNorm before ReLU

class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln  = nn.LayerNorm(hidden_dim)          # delta 2
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb_u, node_emb_d, t_feat, d_feat, history_data):
        B, L, _, _ = t_feat.shape
        gate_in = torch.cat([
            t_feat, d_feat,
            node_emb_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_emb_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
        ], dim=-1)

        h = self.fc1(gate_in)
        h = self.ln(h)                                # delta 2
        h = self.act(h)

        # ── delta 1: hard-sigmoid gate ──
        raw = self.fc2(h)
        gate = torch.clamp(raw / 6.0 + 0.5, 0.0, 1.0)
        gate = gate[:, -history_data.shape[1]:, :, :]

        return history_data * gate
