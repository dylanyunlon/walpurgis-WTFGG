"""
Cathexis DistanceFunction — 算法改写 #4
upstream: inner-product attention → softmax
cathexis: Exponential kernel with learned bandwidth — exp(-||q-k||²/σ²)
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
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim']*2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim*2)
        # Cathexis改写: learned bandwidth for exponential kernel
        self.log_bandwidth = nn.Parameter(torch.zeros(1))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        adjacent_list = []
        for feat in [X1, X2]:
            Q = self.WQ(feat)
            K = self.WK(feat)
            # Cathexis改写: exponential kernel distance
            bandwidth = torch.exp(self.log_bandwidth).clamp(min=0.1, max=10.0)
            diff = Q.unsqueeze(2) - K.unsqueeze(1)
            sq_dist = (diff ** 2).sum(dim=-1)
            W = torch.exp(-sq_dist / (2.0 * bandwidth ** 2))
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            adjacent_list.append(W)
        return adjacent_list
