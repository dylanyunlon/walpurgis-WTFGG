"""Meridian EstimationGate — bilinear node-time interaction + Mish activation.
Changes vs upstream:
  - Bilinear layer for node×time cross-feature (upstream: simple concat)
  - Mish activation (upstream: ReLU)
  - Residual skip from input to gated output
  - Debug: prints gate statistics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'

def mish(x):
    return x * torch.tanh(F.softplus(x))


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        # bilinear cross between node embedding and time embedding
        self.bilinear = nn.Bilinear(2 * node_emb_dim, 2 * time_emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        # residual scaling factor
        self.res_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        # expand node embeddings
        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        # concat pairs
        node_feat = torch.cat([eu, ed], dim=-1)  # [B, L, N, 2*node_dim]
        time_feat = torch.cat([time_in_day_feat, day_in_week_feat], dim=-1)  # [B, L, N, 2*time_dim]
        # bilinear interaction
        cross = self.bilinear(node_feat, time_feat)  # [B, L, N, hidden]
        cross = mish(cross)
        gate = torch.sigmoid(self.fc_out(cross))  # [B, L, N, 1]
        gate = gate[:, -history_data.shape[1]:, :, :]
        # gated output with residual
        gated = history_data * gate + self.res_alpha * history_data
        if _DBG:
            gv = gate.detach()
            print(f"[MER:est_gate] gate mean={gv.mean().item():.4f} "
                  f"std={gv.std().item():.4f} res_alpha={self.res_alpha.item():.4f}",
                  file=sys.stderr)
        return gated
