"""
EstimationGate — Aurora变体
算法改写: ReLU → GELU激活, 增加可学习的scale因子
GELU提供更平滑的梯度流, scale因子允许网络自适应调整gate的幅度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimationGate(nn.Module):
    """比例门控: 估算扩散/固有信号的混合比例"""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(
            2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        # Aurora: GELU替代ReLU, 提供非零的负半轴梯度
        self.activation = nn.GELU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)
        # Aurora: 可学习的输出缩放因子, 初始化为1.0
        # 允许网络学习gate的全局幅度, 比固定sigmoid范围更灵活
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value with GELU activation and learnable scaling."""
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        # Aurora: GELU activation — 保持负值区间的微弱梯度
        hidden = self.activation(hidden)
        # Aurora: 用scale_factor缩放sigmoid输入, 控制gate的锐度
        # scale > 1 使gate更二值化, scale < 1 使gate更平滑
        raw_gate = self.fully_connected_layer_2(hidden)
        estimation_gate = torch.sigmoid(
            raw_gate * torch.abs(self.scale_factor)
        )[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data
