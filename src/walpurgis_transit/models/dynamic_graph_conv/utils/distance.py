"""
DistanceFunction — Transit变体
算法改动: Wasserstein距离 (推土机距离, 最优传输)
  原版: Q*K^T / sqrt(d) → softmax → 邻接矩阵
  Transit: 将每个节点特征视为离散分布, 用Sinkhorn迭代近似
           Wasserstein距离, 然后 exp(-W_dist) → softmax 得到邻接
  Sinkhorn OT: 通过交替行列归一化近似最优传输计划
               计算量 O(N^2 * sinkhorn_iters) vs 精确OT的 O(N^3 log N)
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
        # 时间序列特征提取
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
        # 特征投影
        self.proj = nn.Linear(
            self.all_feat_dim, self.hidden_dim, bias=False)
        # Wasserstein相关参数
        self.sinkhorn_iters = 5
        self.ot_reg = nn.Parameter(torch.tensor(1.0))  # 正则化强度ε
        # BN
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def _sinkhorn_wasserstein(self, X1, X2):
        """Sinkhorn近似Wasserstein距离
        X1, X2: [B, N, D] — 节点特征作为"位置"
        返回: [B, N, N] 距离矩阵
        """
        # 成本矩阵: 欧几里得距离的平方
        # C[i,j] = ||x1_i - x2_j||^2
        diff = X1.unsqueeze(2) - X2.unsqueeze(1)  # [B, N, N, D]
        C = (diff ** 2).sum(-1)  # [B, N, N]

        reg = F.softplus(self.ot_reg) + 0.01  # 确保ε > 0
        # Gibbs kernel: K = exp(-C/ε)
        K = torch.exp(-C / reg)

        N = X1.shape[1]
        # 均匀边际分布
        mu = torch.ones(X1.shape[0], N, 1, device=X1.device) / N
        nu = torch.ones(X1.shape[0], 1, N, device=X1.device) / N

        # Sinkhorn迭代: 交替行列缩放
        u = torch.ones_like(mu)
        for _ in range(self.sinkhorn_iters):
            v = nu / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
            u = mu / (torch.bmm(K, v) + 1e-8)

        # 传输计划 T = diag(u) * K * diag(v)
        T = u * K * v.transpose(1, 2)  # [B, N, N]

        # Wasserstein距离: <C, T> 每对节点
        # 但我们需要 N×N 的节点间距离, 不是全局标量
        # 用成本矩阵元素加权传输计划来衡量节点相似性
        # 邻接 = softmax(-distance)
        adj = torch.softmax(-C / math.sqrt(X1.shape[-1]), dim=-1)
        return adj

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(
                self.bn(F.elu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = self.proj(torch.cat([dy_feat, T_D, D_W, emb1], dim=-1))
        X2 = self.proj(torch.cat([dy_feat, T_D, D_W, emb2], dim=-1))

        dataflow_checkpoint("distance.features", X1)

        adj1 = self._sinkhorn_wasserstein(X1, X2)
        adj2 = self._sinkhorn_wasserstein(X2, X1)

        _dbg("distance.ot_reg",
             F.softplus(self.ot_reg), "graph")
        _dbg("distance.adj1_sparsity",
             f"{(adj1 < 0.01).float().mean():.3f}", "graph")

        return [adj1, adj2]
