"""
Normalizer — Perihelion变体
算法改动: Attention-based归一化(学习注意力权重做归一化)
  原版: D^{-1} * A (行归一化)
  Perihelion: 用可学习的注意力网络计算行归一化权重
             不是简单的度倒数, 而是根据图结构自适应的权重
             类似GAT的注意力系数但作为归一化使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg


def _remove_nan_inf(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor), tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor), tensor)
    return tensor


class AttentionNormalizer(nn.Module):
    """注意力归一化: 用MLP学习边的归一化权重"""

    def __init__(self, num_nodes, hidden_dim=16):
        super().__init__()
        # 边注意力: 根据邻接值计算权重
        self.edge_attn = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 可学习的归一化温度
        self.norm_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, adj):
        """adj: [B, N, N] → 归一化后的 [B, N, N]"""
        # 将邻接值作为边特征输入
        adj_feat = adj.unsqueeze(-1)  # [B, N, N, 1]
        # 计算边注意力权重
        attn_logits = self.edge_attn(adj_feat).squeeze(-1)  # [B, N, N]

        # 温度缩放
        temp = torch.clamp(self.norm_temperature, min=0.1, max=5.0)
        attn_logits = attn_logits / temp

        # 掩码: 对零边不分配权重
        mask = (adj > 1e-8).float()
        attn_logits = attn_logits * mask + (1 - mask) * (-1e9)

        # softmax归一化(按行)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights * mask

        # 最终归一化: 原始邻接值 × 注意力权重
        normed = adj * attn_weights

        normed = _remove_nan_inf(normed)

        _dbg("attn_norm.temp", temp, "graph")
        _dbg("attn_norm.weight_entropy",
             f"{-(attn_weights * torch.log(attn_weights + 1e-8)).sum(-1).mean():.4f}",
             "graph")

        return normed


class Normalizer(nn.Module):
    def __init__(self, num_nodes=10, hidden_dim=16):
        super().__init__()
        self.attn_norm = AttentionNormalizer(
            num_nodes, hidden_dim)

    def forward(self, adj):
        normed = [self.attn_norm(a) for a in adj]
        return normed


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(a) for a in adj]
