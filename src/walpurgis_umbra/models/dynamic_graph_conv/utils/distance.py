"""
DistanceFunction — Umbra变体
算法改动: Hyperbolic Poincaré距离 替代 dot-product attention距离
  原版: Q*K^T / sqrt(d) → softmax → 邻接矩阵
  Umbra: 将节点嵌入映射到Poincaré球模型的双曲空间
        d_P(u,v) = arccosh(1 + 2*||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        双曲空间天然适合树状/层次结构数据, 低维就能编码复杂层次
        通过exponential map将欧氏嵌入投影到球内
  保留时间序列特征提取部分, 用SiLU替代ReLU
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg, dataflow_checkpoint


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        # 时间序列特征提取: SiLU替代ReLU
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(
            self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(
            self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        # 时间槽嵌入
        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)
        # 全特征维度
        self.all_feat_dim = (self.ts_feat_dim + self.node_dim
                             + model_args['time_emb_dim'] * 2)
        # Poincaré投影维度
        self.poincare_dim = self.hidden_dim
        self.proj = nn.Linear(
            self.all_feat_dim, self.poincare_dim, bias=False)
        # 双曲空间的曲率参数 (可学习, 负曲率)
        self.curvature_logit = nn.Parameter(torch.tensor(0.5))
        # BN: 用于时间序列特征
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def _project_to_poincare(self, x, eps=1e-5):
        """将欧氏向量投影到Poincaré球内 (范数<1)
        使用tanh缩放确保||x|| < 1"""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        # tanh(||x||) * x/||x|| 保证在球内
        scale = torch.tanh(norm) / norm
        return x * scale * 0.95  # 留小margin远离边界

    def _poincare_distance(self, u, v, eps=1e-5):
        """Poincaré球模型距离
        d(u,v) = arccosh(1 + 2*||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        返回: [B, N, N] 距离矩阵"""
        c = torch.sigmoid(self.curvature_logit) + 0.1  # 曲率 ∈ (0.1, 1.1)
        # ||u||^2 和 ||v||^2
        u_sq = (u * u).sum(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        v_sq = (v * v).sum(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        # ||u_i - v_j||^2 — 批量距离矩阵
        diff_sq = (u.unsqueeze(-2) - v.unsqueeze(-3)).pow(2).sum(dim=-1)
        # arccosh参数
        numer = 2.0 * c * diff_sq
        denom = (1.0 - c * u_sq) * (1.0 - c * v_sq.transpose(-1, -2))
        denom = denom.clamp(min=eps)
        arg = 1.0 + numer / denom
        arg = arg.clamp(min=1.0 + eps)
        dist = torch.acosh(arg) / (c.sqrt() + eps)
        return dist

    def _poincare_adj(self, X1, X2):
        """Poincaré距离 → 邻接矩阵"""
        Z1 = self._project_to_poincare(self.proj(X1))
        Z2 = self._project_to_poincare(self.proj(X2))
        dist = self._poincare_distance(Z1, Z2)
        # 距离越小相似度越高: softmax(-dist)
        adj = torch.softmax(-dist / math.sqrt(self.poincare_dim), dim=-1)
        return adj

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.silu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        dataflow_checkpoint("distance.features", X1)

        adj1 = self._poincare_adj(X1, X2)
        adj2 = self._poincare_adj(X2, X1)

        curvature = torch.sigmoid(self.curvature_logit) + 0.1
        _dbg("distance.curvature",
             f"c={curvature.item():.4f}", "graph")
        _dbg("distance.adj1_sparsity",
             f"{(adj1 < 0.01).float().mean():.3f}", "graph")

        return [adj1, adj2]
