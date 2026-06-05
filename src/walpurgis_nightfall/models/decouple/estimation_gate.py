"""
EstimationGate — Nightfall变体
算法改写:
  1. 2层FC → 3层FC + 瓶颈LayerNorm (bottleneck regularization)
  2. ReLU → GELU (更平滑的梯度流)
  3. sigmoid gate加可学习temperature τ (控制门值锐度)
"""
import torch
import torch.nn as nn
from ... import _dbg


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        bottleneck = hidden_dim // 2
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.ln_bottleneck = nn.LayerNorm(hidden_dim)
        self.fc_mid = nn.Linear(hidden_dim, bottleneck)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(bottleneck, 1)
        # 可学习温度参数: 初始化为1.0, 控制sigmoid陡峭程度
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        # 拼接node和time embedding
        gate_input = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ], dim=-1)
        _dbg("est_gate.input", gate_input, "model")
        # 3层FC + LayerNorm瓶颈
        h = self.fc_in(gate_input)
        h = self.ln_bottleneck(h)
        h = self.act(h)
        h = self.fc_mid(h)
        h = self.act(h)
        logits = self.fc_out(h)
        # temperature-scaled sigmoid
        tau = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        gate_val = torch.sigmoid(logits / tau)[:, -history_data.shape[1]:, :, :]
        _dbg("est_gate.tau", tau, "model")
        _dbg("est_gate.gate_val", gate_val, "model")
        return history_data * gate_val
