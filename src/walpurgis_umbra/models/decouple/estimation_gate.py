"""
EstimationGate — Umbra变体
算法改动: MoE Gating (2-expert mixture, 学习路由概率)
  原版: 2层FC → sigmoid 生成逐时间步门控
  Umbra: 将门控分为两个"专家"网络, 每个专家有独立的FC参数
        一个可学习路由网络根据输入特征分配权重给两个专家
        路由概率通过softmax产出, 最终门控 = Σ(route_i * expert_i(x))
        添加负载均衡辅助损失防止路由坍缩到单个专家
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint, _moe_tracker


class ExpertGate(nn.Module):
    """单个专家: 独立的两层FC"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.silu(self.fc1(x))))


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        input_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.num_experts = 2

        # 两个独立专家网络
        self.expert_a = ExpertGate(input_dim, hidden_dim)
        self.expert_b = ExpertGate(input_dim, hidden_dim)

        # 路由网络: 根据输入特征产出2维路由概率
        self.router_fc1 = nn.Linear(input_dim, hidden_dim // 2)
        self.router_fc2 = nn.Linear(hidden_dim // 2, self.num_experts)

        # 路由温度: 控制softmax的锐利度
        self.route_temperature = nn.Parameter(torch.tensor(1.0))

        # 负载均衡系数
        self.balance_coeff = 0.01

    def _compute_balance_loss(self, route_probs):
        """负载均衡辅助损失: 鼓励两个专家被均匀使用
        loss = num_experts * Σ(f_i * P_i), f_i=实际选中比例, P_i=平均概率"""
        avg_probs = route_probs.mean(dim=tuple(range(route_probs.dim() - 1)))
        # 理想情况avg_probs应约等于[0.5, 0.5]
        balance_loss = self.num_experts * (avg_probs * avg_probs).sum()
        return balance_loss * self.balance_coeff

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
                batch_size, seq_length, -1, -1),
        ], dim=-1)

        dataflow_checkpoint("est_gate.feat", gate_feat)

        # 路由网络: 产出[B, L, N, 2]的路由概率
        temp = torch.clamp(self.route_temperature, min=0.1, max=5.0)
        route_logits = self.router_fc2(F.silu(self.router_fc1(gate_feat)))
        route_probs = F.softmax(route_logits / temp, dim=-1)

        # 两个专家分别计算门控
        gate_a = self.expert_a(gate_feat)   # [B, L, N, 1]
        gate_b = self.expert_b(gate_feat)   # [B, L, N, 1]

        # 加权混合: route_probs[..., 0] * expert_a + route_probs[..., 1] * expert_b
        estimation_gate = (
            route_probs[..., 0:1] * gate_a
            + route_probs[..., 1:2] * gate_b
        )
        estimation_gate = estimation_gate[:, -history_data.shape[1]:, :, :]

        # 记录路由统计 + 负载均衡损失
        _moe_tracker.record(route_probs)
        self._last_balance_loss = self._compute_balance_loss(route_probs)

        _dbg("est_gate.route_probs_mean",
             route_probs.mean(dim=(0, 1, 2)), "decouple")
        _dbg("est_gate.temperature", temp, "decouple")
        _dbg("est_gate.output_gate", estimation_gate, "decouple")
        _dbg("est_gate.balance_loss",
             f"{self._last_balance_loss.item():.6f}", "decouple")

        return history_data * estimation_gate
