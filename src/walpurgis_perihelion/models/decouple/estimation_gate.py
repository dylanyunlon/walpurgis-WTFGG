"""
EstimationGate — Perihelion变体
算法改动: Cross-Attention Gate(Q=时间特征, K/V=节点嵌入, 跨模态门控)
  原版: 2层FC → sigmoid 生成逐时间步门控
  Perihelion: 将时间特征作为Query, 节点嵌入作为Key/Value
             通过缩放点积注意力计算跨模态门控权重
             加入温度缩放和残差旁路保持梯度畅通
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


class CrossAttentionGating(nn.Module):
    """跨模态注意力门控: Q=时间, K/V=空间(节点嵌入)"""

    def __init__(self, q_dim, kv_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(q_dim // num_heads, 4)
        self.inner_dim = self.head_dim * num_heads
        self.W_q = nn.Linear(q_dim, self.inner_dim, bias=False)
        self.W_k = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.W_v = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, q_dim)
        # 可学习温度: 控制注意力分布锐度
        self.attn_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        """
        query: [B, L, N, D_q] 时间特征
        key/value: [B, 1, N, D_kv] 节点嵌入(广播到L)
        返回: [B, L, N, D_q] 门控后的特征
        """
        B, L, N, _ = query.shape
        # 展平N到batch维度进行多头注意力
        q = self.W_q(query).reshape(B * N, L, self.num_heads, self.head_dim).transpose(1, 2)
        # key/value在时间维只有1步,扩展
        K_len = key.shape[1]
        k = self.W_k(key).reshape(B * N, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).reshape(B * N, K_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力 + 温度
        temp = torch.clamp(self.attn_temperature, min=0.1, max=5.0)
        scale = math.sqrt(self.head_dim) * temp
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).reshape(B * N, L, self.inner_dim)
        attended = self.out_proj(attended).reshape(B, N, L, -1).transpose(1, 2)
        return attended


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        time_feat_dim = time_emb_dim * 2
        node_feat_dim = 2 * node_emb_dim

        # Cross-Attention门控: Q=时间, K/V=节点嵌入
        self.cross_gate = CrossAttentionGating(
            q_dim=time_feat_dim,
            kv_dim=node_feat_dim,
            num_heads=2)

        # 输出投影: 将cross-attention结果压缩到门控标量
        self.gate_proj = nn.Sequential(
            nn.Linear(time_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 残差旁路权重: 控制门控 vs 直通的比例
        self.bypass_alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape

        # 构造时间Query: [B, L, N, time_dim*2]
        time_query = torch.cat([time_in_day_feat,
                                day_in_week_feat], dim=-1)

        # 构造节点Key/Value: [B, 1, N, node_dim*2]
        node_kv = torch.cat([
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, 1, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, 1, -1, -1)
        ], dim=-1)

        dataflow_checkpoint("est_gate.time_query", time_query)
        dataflow_checkpoint("est_gate.node_kv", node_kv)

        # Cross-Attention: 时间特征关注节点嵌入
        cross_out = self.cross_gate(time_query, node_kv, node_kv)

        # 投影到门控值
        gate_values = self.gate_proj(cross_out)
        gate_values = gate_values[:, -history_data.shape[1]:, :, :]

        # 残差旁路: 防止门控过于激进
        alpha = torch.sigmoid(self.bypass_alpha)
        effective_gate = alpha * gate_values + (1 - alpha) * torch.ones_like(gate_values)

        _dbg("est_gate.cross_attn_energy",
             cross_out.detach().norm(), "decouple")
        _dbg("est_gate.bypass_alpha", alpha, "decouple")
        _dbg("est_gate.gate_mean",
             effective_gate.mean(), "decouple")

        return history_data * effective_gate
