"""
EstimationGate — Penumbra变体
算法改动: Squeeze-Excitation通道注意力门控
  原版: 2层FC → sigmoid 生成逐时间步门控
  Penumbra: 先squeeze(全局平均池化) → excitation(2层FC+Swish) → 通道重标定
           加入skip connection保持梯度流
"""
import torch
import torch.nn as nn
from ... import _dbg, dataflow_checkpoint


def _swish(x):
    """Swish激活: x * sigmoid(x) — 比ReLU更平滑"""
    return x * torch.sigmoid(x)


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        input_dim = 2 * node_emb_dim + time_emb_dim * 2
        # 主通道: 保留原始FC但用Swish替代ReLU
        self.fc_main_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_main_2 = nn.Linear(hidden_dim, 1)

        # Squeeze-Excitation分支: 通道注意力
        # squeeze: 对时间维度做全局平均池化 → hidden_dim
        # excitation: 两层FC瓶颈结构
        se_bottleneck = max(hidden_dim // 4, 4)
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_fc1 = nn.Linear(input_dim, se_bottleneck)
        self.se_fc2 = nn.Linear(se_bottleneck, input_dim)

        # skip权重: 控制SE分支和主通道的混合
        self.skip_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        # 拼接特征
        gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)

        dataflow_checkpoint("est_gate.feat", gate_feat)

        # Squeeze-Excitation路径
        # squeeze: [B,L,N,D] → 对L维平均 → [B,1,N,D]
        se_input = gate_feat.mean(dim=1, keepdim=True)
        se_hidden = _swish(self.se_fc1(se_input))
        se_weight = torch.sigmoid(self.se_fc2(se_hidden))
        # excitation: 重标定原始特征
        se_feat = gate_feat * se_weight

        # 主路径: Swish替代ReLU
        alpha = torch.sigmoid(self.skip_alpha)
        blended_feat = alpha * se_feat + (1 - alpha) * gate_feat
        hidden = _swish(self.fc_main_1(blended_feat))
        estimation_gate = torch.sigmoid(
            self.fc_main_2(hidden)
        )[:, -history_data.shape[1]:, :, :]

        _dbg("est_gate.se_weight_mean",
             se_weight.mean(), "decouple")
        _dbg("est_gate.alpha", alpha, "decouple")
        _dbg("est_gate.output_gate",
             estimation_gate, "decouple")

        return history_data * estimation_gate
