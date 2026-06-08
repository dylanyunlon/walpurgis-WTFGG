import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_reverie import _dbg

_TAG = "distance"


class DistanceFunction(nn.Module):
    """upstream: QK attention dot-product distance → softmax
    改动: Radial Basis Function (RBF) kernel distance
    RBF: exp(-||q-k||^2 / (2*sigma^2)), 可学习bandwidth sigma
    比dot-product更能捕获局部结构: dot-product对magnitude敏感, RBF对距离敏感
    traffic graph中相邻节点的特征应该距离近, 而非投影对齐
    """

    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']

        # Time Series Feature Extraction — 保持upstream结构
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim

        # Time Slot Embedding
        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        # Feature projection
        self.all_feat_dim = (self.ts_feat_dim + self.node_dim +
                            model_args['time_emb_dim'] * 2)
        self.proj = nn.Linear(self.all_feat_dim, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # 改动: 可学习RBF bandwidth, 初始化为sqrt(hidden_dim)
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(math.sqrt(self.hidden_dim))))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        # 改动: RBF kernel distance代替QK attention
        sigma_sq = (2 * self.log_sigma).exp()
        adjacent_list = []
        for feat in [X1, X2]:
            proj_feat = self.proj(feat)  # [B, N, D]
            # pairwise L2 distance: ||q_i - k_j||^2
            dist_sq = torch.cdist(proj_feat, proj_feat, p=2).pow(2)
            # RBF kernel
            W = torch.exp(-dist_sq / (sigma_sq + 1e-6))
            # row normalize to get stochastic matrix
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            adjacent_list.append(W)

        _dbg(f"{_TAG}/rbf_sigma", self.log_sigma.exp(), _TAG)
        _dbg(f"{_TAG}/adj_0_sparsity",
             (adjacent_list[0] < 0.01).float().mean(), _TAG)
        return adjacent_list
