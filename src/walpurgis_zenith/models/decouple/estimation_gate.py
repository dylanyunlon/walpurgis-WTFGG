"""
EstimationGate — Zenith变体
改写: ReLU → Mish激活, 增加可学习的bias偏移
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimationGate(nn.Module):
    """比例门控: 估算扩散/固有信号的混合比例"""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fc_1 = nn.Linear(
            2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1)
        # Zenith: 可学习的gate偏移量
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        hidden = self.fc_1(gate_feat)
        # Mish替代ReLU
        hidden = hidden * torch.tanh(F.softplus(hidden))
        gate = torch.sigmoid(
            self.fc_2(hidden) + self.gate_bias
        )[:, -history_data.shape[1]:, :, :]
        history_data = history_data * gate
        return history_data
