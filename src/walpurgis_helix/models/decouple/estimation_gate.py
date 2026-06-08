import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimationGate(nn.Module):
    """The estimation gate module.
    Helix改写: 在gate计算后加入螺旋相位调制,
    用cos/sin gate对时间维度做旋转增强"""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)
        # Helix特有: 螺旋相位偏置, 为gate加入旋转成分
        self.helix_phase_bias = nn.Parameter(torch.zeros(1))

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        raw_gate = self.fully_connected_layer_2(hidden)
        # Helix特有: 螺旋相位调制 — 在sigmoid前加入周期性偏置
        seq_pos = torch.arange(seq_length, device=raw_gate.device).float()
        phase_mod = torch.sin(seq_pos * 0.1 + self.helix_phase_bias)
        phase_mod = phase_mod.view(1, -1, 1, 1)
        raw_gate = raw_gate + 0.05 * phase_mod
        estimation_gate = torch.sigmoid(raw_gate)[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data
