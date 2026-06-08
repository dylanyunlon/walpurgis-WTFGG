"""
DistanceFunction — Penumbra变体
算法改动: Mahalanobis距离 替代 dot-product attention距离
  原版: Q*K^T / sqrt(d) → softmax → 邻接矩阵
  Penumbra: 学习协方差矩阵L (Cholesky分解保正定)
           d_M(x,y) = (x-y)^T * L^T * L * (x-y)
           再通过 exp(-d_M) → softmax 得到邻接矩阵
  保留时间序列特征提取部分, 用Swish替代ReLU
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg, dataflow_checkpoint


def _swish(x):
    return x * torch.sigmoid(x)


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        # 时间序列特征提取: Swish替代ReLU
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
        # Mahalanobis: 学习Cholesky因子L, 协方差Σ^{-1} = L^T L
        # 先投影到较低维度再算距离
        self.proj_dim = self.hidden_dim
        self.proj = nn.Linear(
            self.all_feat_dim, self.proj_dim, bias=False)
        # L矩阵: 下三角, 保证正定
        self.L_diag = nn.Parameter(
            torch.ones(self.proj_dim) * 0.5)
        self.L_lower = nn.Parameter(
            torch.zeros(self.proj_dim * (self.proj_dim - 1) // 2)
            * 0.01)
        # BN: 用于时间序列特征
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def _build_L(self):
        """构建Cholesky因子L (下三角+正对角)"""
        L = torch.zeros(
            self.proj_dim, self.proj_dim,
            device=self.L_diag.device)
        # 正对角线: softplus保正
        L[range(self.proj_dim),
          range(self.proj_dim)] = F.softplus(self.L_diag)
        # 下三角
        idx = torch.tril_indices(
            self.proj_dim, self.proj_dim, offset=-1)
        L[idx[0], idx[1]] = self.L_lower
        return L

    def _mahalanobis_adj(self, X1, X2):
        """Mahalanobis距离 → 邻接矩阵
        X1, X2: [B, N, D]
        返回: [B, N, N] 邻接矩阵
        """
        L = self._build_L()
        # 投影
        Z1 = self.proj(X1)  # [B, N, proj_dim]
        Z2 = self.proj(X2)
        # L * Z: [B, N, proj_dim]
        LZ1 = torch.matmul(Z1, L.T)
        LZ2 = torch.matmul(Z2, L.T)
        # 距离矩阵: ||LZ1_i - LZ2_j||^2
        # = ||LZ1||^2 + ||LZ2||^2 - 2*(LZ1 @ LZ2^T)
        sq1 = (LZ1 ** 2).sum(-1, keepdim=True)  # [B,N,1]
        sq2 = (LZ2 ** 2).sum(-1, keepdim=True)  # [B,N,1]
        cross = torch.bmm(LZ1, LZ2.transpose(-1, -2))
        dist_sq = sq1 + sq2.transpose(-1, -2) - 2 * cross
        dist_sq = torch.clamp(dist_sq, min=0.0)
        # exp(-dist) → softmax
        adj = torch.softmax(-dist_sq / math.sqrt(
            self.proj_dim), dim=-1)
        return adj

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(_swish(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        dataflow_checkpoint("distance.features", X1)

        adj1 = self._mahalanobis_adj(X1, X2)
        adj2 = self._mahalanobis_adj(X2, X1)

        _dbg("distance.L_cond",
             f"diag_range=[{F.softplus(self.L_diag).min():.4f},"
             f"{F.softplus(self.L_diag).max():.4f}]",
             "graph")
        _dbg("distance.adj1_sparsity",
             f"{(adj1 < 0.01).float().mean():.3f}", "graph")

        return [adj1, adj2]
