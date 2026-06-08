"""
Cathexis EstimationGate — 算法改写 #1
upstream: FC(concat) → ReLU → FC → sigmoid
cathexis: Bilinear(u,d) + SiLU + time modulation → sigmoid
"""
import torch
import torch.nn as nn
from ... import _dbg, dataflow_checkpoint

class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(node_emb_dim, node_emb_dim, hidden_dim, bias=False)
        self.time_proj = nn.Linear(time_emb_dim * 2, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, N, _ = time_in_day_feat.shape
        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        bilinear_out = self.bilinear(eu.reshape(-1, eu.shape[-1]),
                                     ed.reshape(-1, ed.shape[-1])).reshape(B, L, N, -1)
        time_cat = torch.cat([time_in_day_feat, day_in_week_feat], dim=-1)
        time_mod = self.time_proj(time_cat)
        combined = self.activation(bilinear_out + time_mod)
        gate_val = torch.sigmoid(self.gate_fc(combined))[:, -history_data.shape[1]:, :, :]
        _dbg("egate.gate", gate_val)
        return history_data * gate_val
