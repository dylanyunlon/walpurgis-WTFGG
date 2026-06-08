"""
Corona DistanceFunction — 算法改写:
  upstream: QKT attention score for adjacency
  corona: learned metric embedding — 先投影到metric space,
          然后用余弦相似度替代内积, 加上可学习的temperature scaling
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
        # Corona改写: metric embedding投影 + 温度缩放余弦距离
        self.metric_proj = nn.Linear(self.all_feat_dim, self.hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 可学习温度
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
            # Corona: metric embedding + 余弦相似度 + 温度缩放
            proj = self.metric_proj(feat)  # [B, N, D]
            proj_norm = F.normalize(proj, p=2, dim=-1)
            cosine_sim = torch.bmm(proj_norm, proj_norm.transpose(-1, -2))
            # 温度缩放 + softmax
            temp = torch.clamp(self.temperature, min=0.01)
            W = torch.softmax(cosine_sim / temp, dim=-1)
            adjacent_list.append(W)
        return adjacent_list
