"""
estimation_gate.py — v9 port
Algo delta:
  1. ReLU → GELU (更平滑的梯度, 近零区域有非零梯度)
  2. sigmoid 前加可学习标量温度 τ (初始化=1.0):
     gate = sigmoid(logit / τ), τ 越小 gate 越 sharp
  3. FC2 之前插入 LayerNorm 稳定 logit 幅度
"""
import torch
import torch.nn as nn
from walpurgis_ported_v9 import _dbg

_TAG = "est_gate"


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()                            # v9: GELU
        self.ln  = nn.LayerNorm(hidden_dim)             # v9: pre-output LN
        self.fc2 = nn.Linear(hidden_dim, 1)
        # v9: learnable temperature for sigmoid sharpness
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):

        B, L, _, _ = time_in_day_feat.shape
        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        cat = torch.cat([time_in_day_feat, day_in_week_feat, eu, ed], dim=-1)

        h = self.fc1(cat)
        h = self.act(h)
        h = self.ln(h)                                  # v9: LN
        logit = self.fc2(h)
        # v9: temperature-scaled sigmoid
        tau_clamped = torch.clamp(self.tau, min=0.01)
        gate = torch.sigmoid(logit / tau_clamped)[:, -history_data.shape[1]:, :, :]

        _dbg(_TAG, f"τ={tau_clamped.item():.4f}  gate∈[{gate.min().item():.4f},{gate.max().item():.4f}]  "
                    f"gate_mean={gate.mean().item():.4f}")

        return history_data * gate
