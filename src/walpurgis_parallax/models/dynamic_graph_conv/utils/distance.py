"""
DistanceFunction — Parallax变体 (M054)
算法改动: Kernel Density Estimation距离 替代 dot-product距离
  原版: Q*K^T / sqrt(d) → softmax → 邻接矩阵
  Parallax: 用RBF核的核密度估计量化节点间相似度
           对每个节点i, 用其嵌入作为"观测点",
           估计节点j处的密度 p(x_j | {x_i周围的核})
           密度越高 → 相似度越高 → 边权越大
           带宽参数h可学习, 用Silverman规则做初始化
           多核混合: 同时使用RBF + Laplacian + Cosine核

  核密度估计比点积距离更非参数化, 能捕捉非线性相似结构
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
        # 特征投影 — 降到KDE计算维度
        self.proj_dim = self.hidden_dim
        self.proj = nn.Linear(
            self.all_feat_dim, self.proj_dim, bias=False)

        # KDE核参数 — 可学习的带宽
        # Silverman规则初始化: h = 1.06 * σ * n^(-1/5)
        init_bandwidth = model_args.get('kde_bandwidth', 1.0)
        self.log_bandwidth_rbf = nn.Parameter(
            torch.tensor(math.log(init_bandwidth)))
        self.log_bandwidth_lap = nn.Parameter(
            torch.tensor(math.log(init_bandwidth * 1.5)))

        # 核混合权重: RBF vs Laplacian
        self.kernel_logits = nn.Parameter(torch.zeros(2))

        # BN
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def _rbf_kernel(self, sq_dist, bandwidth):
        """高斯/RBF核: K(d) = exp(-d^2 / 2h^2)"""
        return torch.exp(-sq_dist / (2 * bandwidth ** 2 + 1e-8))

    def _laplacian_kernel(self, dist, bandwidth):
        """拉普拉斯核: K(d) = exp(-|d| / h)"""
        return torch.exp(-dist / (bandwidth + 1e-8))

    def _kde_similarity(self, Z1, Z2):
        """KDE相似度矩阵
        Z1, Z2: [B, N, D]
        返回: [B, N, N]
        """
        h_rbf = torch.exp(self.log_bandwidth_rbf)
        h_lap = torch.exp(self.log_bandwidth_lap)

        # 欧氏距离矩阵
        sq1 = (Z1 ** 2).sum(-1, keepdim=True)
        sq2 = (Z2 ** 2).sum(-1, keepdim=True)
        cross = torch.bmm(Z1, Z2.transpose(-1, -2))
        sq_dist = sq1 + sq2.transpose(-1, -2) - 2 * cross
        sq_dist = torch.clamp(sq_dist, min=0.0)
        dist = torch.sqrt(sq_dist + 1e-8)

        # 多核KDE
        kde_rbf = self._rbf_kernel(sq_dist, h_rbf)
        kde_lap = self._laplacian_kernel(dist, h_lap)

        # 可学习的核混合
        mix_weights = F.softmax(self.kernel_logits, dim=0)
        kde_mixed = mix_weights[0] * kde_rbf + mix_weights[1] * kde_lap

        # 对每行归一化成概率密度(密度估计)
        adj = kde_mixed / (kde_mixed.sum(dim=-1, keepdim=True) + 1e-8)

        return adj

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.elu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        dataflow_checkpoint("distance.features", X1)

        # 投影到KDE计算空间
        Z1 = self.proj(X1)
        Z2 = self.proj(X2)

        # KDE相似度
        adj1 = self._kde_similarity(Z1, Z2)
        adj2 = self._kde_similarity(Z2, Z1)

        _dbg("distance.bandwidth_rbf",
             torch.exp(self.log_bandwidth_rbf), "graph")
        _dbg("distance.bandwidth_lap",
             torch.exp(self.log_bandwidth_lap), "graph")
        _dbg("distance.kernel_mix",
             F.softmax(self.kernel_logits, dim=0), "graph")
        _dbg("distance.adj1_sparsity",
             f"{(adj1 < 0.01).float().mean():.3f}", "graph")

        return [adj1, adj2]
