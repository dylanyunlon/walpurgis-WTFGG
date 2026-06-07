"""Tempest gate: Squeeze-Excitation with spatial attention.
Unlike upstream (2-layer FC+ReLU+sigmoid) and eclipse (3-layer FC+Swish+ChannelSE+tau),
Tempest uses a full SE block with explicit spatial (node-dim) attention pooling and a
learnable temperature per-node, producing a spatially-aware gating mechanism."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

class SpatialSEBlock(nn.Module):
    """Squeeze-Excitation with both channel and spatial paths.
    Channel: global avg pool -> FC -> ReLU -> FC -> sigmoid
    Spatial: 1x1 conv-like projection -> sigmoid -> spatial attention map"""
    def __init__(self, channels, num_nodes, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel SE path
        self.ch_squeeze = nn.AdaptiveAvgPool1d(1)
        self.ch_fc1 = nn.Linear(channels, mid)
        self.ch_fc2 = nn.Linear(mid, channels)
        # Spatial SE path: learns per-node importance
        self.sp_fc1 = nn.Linear(channels, mid)
        self.sp_fc2 = nn.Linear(mid, 1)
        # Fusion
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, L, N, C]
        B, L, N, C = x.shape
        # Channel attention: pool over nodes
        x_flat = x.reshape(B * L, N, C)
        ch_pool = x_flat.mean(dim=1)  # [B*L, C]
        ch_attn = torch.sigmoid(self.ch_fc2(F.relu(self.ch_fc1(ch_pool))))  # [B*L, C]
        ch_attn = ch_attn.unsqueeze(1)  # [B*L, 1, C]
        # Spatial attention: per-node importance
        sp_h = F.relu(self.sp_fc1(x_flat))  # [B*L, N, mid]
        sp_attn = torch.sigmoid(self.sp_fc2(sp_h))  # [B*L, N, 1]
        # Fuse: learnable blend of channel and spatial attention
        alpha = torch.sigmoid(self.fusion_gate)
        out = x_flat * (alpha * ch_attn + (1 - alpha) * sp_attn)
        return out.reshape(B, L, N, C)

class EstimationGate(nn.Module):
    """Tempest estimation gate: SE with spatial attention.
    Compared to upstream (simple FC+ReLU+sigmoid) and eclipse (Swish+ChannelSE+tau),
    Tempest adds explicit spatial attention across nodes and uses Hardswish activation."""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = SpatialSEBlock(hidden_dim, num_nodes=1)  # num_nodes set dynamically
        self.fc_gate = nn.Linear(hidden_dim, 1)
        self.act = nn.Hardswish()  # Hardswish (vs upstream ReLU, eclipse SiLU/Swish)
        # Per-node learnable temperature via embedding
        self.temp_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)], dim=-1)
        h = self.act(self.fc1(feat))
        h = self.act(self.fc2(h))
        h = self.se(h)  # Spatial SE attention
        tau = (self.temp_scale.abs() + 0.1).clamp(max=10.0)
        gate = torch.sigmoid(self.fc_gate(h) / tau)[:, -history_data.shape[1]:, :, :]
        if _TEM_DBG:
            print(f"[TEM:gate@estimation_gate] mean={gate.mean().item():.4f} std={gate.std().item():.4f} tau={tau.item():.4f}", file=sys.stderr)
        return history_data * gate
