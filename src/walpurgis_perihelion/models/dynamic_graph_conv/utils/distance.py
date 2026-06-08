"""
DistanceFunction — Perihelion变体
算法改动: MINE互信息距离(Mutual Information Neural Estimation)
  原版: Q*K^T / sqrt(d) → softmax → 邻接矩阵
  Perihelion: 用神经网络估计节点对之间的互信息
             MINE: MI(X,Y) ≈ E[T(x,y)] - log(E[e^{T(x',y)}])
             T是参数化的统计网络(2层MLP)
             高互信息 → 强连接, 低互信息 → 弱连接
  保留时间序列特征提取, 用SiLU替代ReLU
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .... import _dbg, _mine_tracker, dataflow_checkpoint


class MINEStatisticsNet(nn.Module):
    """MINE的统计网络T(x,y): 估计联合分布 vs 边际乘积分布的比值"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # T(x, y) = MLP([x; y])
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, y):
        """x, y: [B, N, D] → T(x,y): [B, N, N]"""
        B, N, D = x.shape
        # 构造所有(i,j)对: x_i与y_j
        x_exp = x.unsqueeze(2).expand(-1, -1, N, -1)  # [B,N,N,D]
        y_exp = y.unsqueeze(1).expand(-1, N, -1, -1)  # [B,N,N,D]
        pairs = torch.cat([x_exp, y_exp], dim=-1)  # [B,N,N,2D]
        scores = self.net(pairs).squeeze(-1)  # [B,N,N]
        return scores


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
        # MINE统计网络: 估计互信息
        self.mine_net = MINEStatisticsNet(
            self.all_feat_dim, hidden_dim=self.hidden_dim)
        # 特征投影(降维到统一空间)
        self.feat_proj = nn.Linear(
            self.all_feat_dim, self.all_feat_dim)
        # BN: 用于时间序列特征
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)
        # MINE EMA基线(用于方差缩减)
        self.register_buffer('ema_baseline',
                             torch.tensor(0.0))
        self.ema_momentum = 0.01

    def _mine_adjacency(self, X1, X2):
        """MINE互信息 → 邻接矩阵
        MI越高 → 连接越强
        """
        # 联合分布得分: T(x_i, y_j) 对 matched pairs
        joint_scores = self.mine_net(X1, X2)  # [B, N, N]

        # 边际乘积分布: shuffle Y → T(x_i, y_π(j))
        B, N, D = X2.shape
        perm = torch.randperm(N, device=X2.device)
        X2_shuffled = X2[:, perm, :]
        marginal_scores = self.mine_net(X1, X2_shuffled)

        # MINE下界: E[T_joint] - log(E[e^{T_marginal}])
        # 但我们要的是每对的"距离"而非全局MI
        # 用单对T(x_i, y_j)作为连接强度的代理
        # 高T值 → 高互信息 → 强连接
        adj = torch.sigmoid(joint_scores)  # [B, N, N]

        # 记录MINE估计值用于诊断
        with torch.no_grad():
            mi_est = (joint_scores.mean()
                      - torch.logsumexp(marginal_scores.reshape(B, -1),
                                        dim=-1).mean()
                      + math.log(N))
            _mine_tracker.record(mi_est.item())

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

        X1 = self.feat_proj(
            torch.cat([dy_feat, T_D, D_W, emb1], dim=-1))
        X2 = self.feat_proj(
            torch.cat([dy_feat, T_D, D_W, emb2], dim=-1))

        dataflow_checkpoint("distance.features", X1)

        adj1 = self._mine_adjacency(X1, X2)
        adj2 = self._mine_adjacency(X2, X1)

        _dbg("distance.mine_adj_density",
             f"{(adj1 > 0.5).float().mean():.3f}", "graph")
        _dbg("distance.adj_symmetry",
             f"{(adj1 - adj2).abs().mean():.4f}", "graph")

        return [adj1, adj2]
