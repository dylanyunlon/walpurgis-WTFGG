"""
Aphelion DistanceFunction — 算法改写 #4:
  upstream: QKT attention score for adjacency
  corona: learned metric embedding + cosine similarity + temperature
  aphelion: SimCLR contrastive learning adjacency — 用SimCLR框架的
            投影头+NT-Xent相似度构建邻接矩阵。通过对比学习使相似节点
            靠近、不同节点远离, 比简单的内积/余弦距离有更好的表示学习能力。
  改动幅度: ~30% (对比学习投影头替代简单distance)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        # Aphelion改写: SimCLR投影头 — 两层MLP投影到对比学习空间
        # upstream/corona用单层投影; SimCLR用两层非线性投影, 更好地学习表示
        proj_dim = self.hidden_dim
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.all_feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),  # SimCLR标准: BN + ReLU
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # 可学习的温度参数 (NT-Xent loss中的tau)
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        adjacent_list = []
        for feat in [X1, X2]:
            # Aphelion: SimCLR投影头 + NT-Xent相似度
            # feat: [B, N, all_feat_dim]
            B, N, D = feat.shape
            feat_flat = feat.reshape(B * N, D)
            # 通过SimCLR投影头
            z = self.simclr_projector(feat_flat)  # [B*N, proj_dim]
            z = z.reshape(B, N, -1)
            # L2归一化 (SimCLR标准)
            z_norm = F.normalize(z, p=2, dim=-1)
            # NT-Xent相似度矩阵: sim(i,j) = z_i · z_j / tau
            tau = torch.clamp(self.temperature, min=0.07)
            sim_matrix = torch.bmm(z_norm, z_norm.transpose(-1, -2)) / tau
            # 转为邻接权重 (softmax归一化)
            W = torch.softmax(sim_matrix, dim=-1)
            adjacent_list.append(W)
        return adjacent_list
