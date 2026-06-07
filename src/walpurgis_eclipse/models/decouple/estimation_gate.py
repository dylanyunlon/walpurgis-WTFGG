"""Eclipse gate: 3-layer FC + Swish + ChannelSE + learnable tau."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class ChannelSE(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc1 = nn.Linear(ch, mid); self.fc2 = nn.Linear(mid, ch)
    def forward(self, x):
        se = x.mean(dim=-2, keepdim=True)
        return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(se))))

class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = ChannelSE(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B,L,-1,-1)], dim=-1)
        h = self.act(self.fc1(feat)); h = self.act(self.fc2(h)); h = self.se(h)
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        if _ECL_DBG: print(f"[ECL:gate] mean={gate.mean().item():.4f} std={gate.std().item():.4f} tau={tau.item():.4f}", file=sys.stderr)
        return history_data * gate
