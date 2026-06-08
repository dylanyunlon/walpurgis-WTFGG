"""
EstimationGate — Transit变体
算法改动: Capsule Routing Gate (胶囊网络动态路由做门控)
  原版: 2层FC → sigmoid 生成逐时间步门控
  Transit: 将特征打包为capsule向量, 通过routing-by-agreement
           迭代计算coupling系数, squash非线性压缩幅值
           路由协议决定哪些capsule(时空特征)应被门控传递

  squash(s) = ||s||^2 / (1+||s||^2) * s/||s||
  routing: b_ij += u_hat_j|i · v_j (agreement)
           c_ij = softmax(b_ij)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint, _capsule_tracker


def _squash(tensor, dim=-1, eps=1e-8):
    """Squash非线性: 将向量压缩到[0,1)范围但保持方向
    ||s||^2/(1+||s||^2) * s/||s||"""
    sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq_norm + eps)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * tensor / norm


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim,
                 capsule_dim=8, num_routing=3):
        super().__init__()
        input_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
        # 特征投影到capsule空间: 每个capsule是一个capsule_dim向量
        num_capsules = max(hidden_dim // capsule_dim, 2)
        self.num_capsules_in = num_capsules
        self.num_capsules_out = 1  # 最终输出一个标量门
        self.feat_to_capsules = nn.Linear(
            input_dim, num_capsules * capsule_dim)
        # 路由权重矩阵: 将输入capsule映射到输出capsule预测
        self.W_route = nn.Parameter(
            torch.randn(self.num_capsules_in,
                        self.num_capsules_out,
                        capsule_dim, capsule_dim) * 0.01)
        # 最终从capsule空间映射到gate标量
        self.gate_proj = nn.Linear(capsule_dim, 1)
        # 可学习的路由温度
        self.route_temp = nn.Parameter(torch.tensor(1.0))

    def _route_capsules(self, u_hat, batch_shape):
        """动态路由: routing-by-agreement迭代"""
        B = u_hat.shape[0]
        # b_ij: routing logits, 初始化为0
        b_ij = torch.zeros(
            B, self.num_capsules_in, self.num_capsules_out,
            device=u_hat.device)
        temp = torch.clamp(self.route_temp, min=0.1, max=5.0)
        final_agreement = 0.0
        for r_iter in range(self.num_routing):
            # coupling系数: softmax over output capsules
            c_ij = F.softmax(b_ij / temp, dim=2)  # [B, in, out]
            # 加权求和得到输出capsule
            # u_hat: [B, in, out, cap_dim]
            # c_ij:  [B, in, out]
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  # [B, out, cap_dim]
            v_j = _squash(s_j, dim=-1)  # [B, out, cap_dim]
            if r_iter < self.num_routing - 1:
                # agreement: u_hat · v_j
                agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
                b_ij = b_ij + agreement
                final_agreement = agreement.mean().item()
        _capsule_tracker.record(self.num_routing, final_agreement)
        return v_j  # [B, num_capsules_out, capsule_dim]

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, num_nodes, _ = time_in_day_feat.shape
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

        orig_shape = gate_feat.shape  # [B, L, N, D_in]
        flat = gate_feat.reshape(-1, gate_feat.shape[-1])  # [B*L*N, D_in]

        # 投影为输入capsules
        u_raw = self.feat_to_capsules(flat)  # [B*L*N, num_in * cap_dim]
        u_raw = u_raw.view(-1, self.num_capsules_in, self.capsule_dim)
        u_raw = _squash(u_raw, dim=-1)

        # 计算 u_hat: 输入capsule对每个输出capsule的"预测"
        # u_raw: [BLN, in, cap_dim] -> [BLN, in, out, cap_dim]
        u_expanded = u_raw.unsqueeze(2).expand(
            -1, -1, self.num_capsules_out, -1)  # [BLN, in, out, cap_dim]
        u_hat = torch.einsum(
            'biod,iodj->bioj', u_expanded, self.W_route)  # [BLN, in, out, cap_dim]

        # 路由
        v_j = self._route_capsules(u_hat, orig_shape)  # [BLN, out, cap_dim]

        # 从capsule空间提取gate值
        gate_val = torch.sigmoid(
            self.gate_proj(v_j.squeeze(1)))  # [BLN, 1]
        estimation_gate = gate_val.view(
            batch_size, seq_length, num_nodes, 1)
        estimation_gate = estimation_gate[
            :, -history_data.shape[1]:, :, :]

        _dbg("est_gate.routing_temp",
             self.route_temp, "decouple")
        _dbg("est_gate.output_gate",
             estimation_gate, "decouple")

        return history_data * estimation_gate
