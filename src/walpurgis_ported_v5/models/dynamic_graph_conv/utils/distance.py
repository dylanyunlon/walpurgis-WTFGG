import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Delta vs upstream:
#   1. Distance score: QK^T → QK^T with learned bias term per node pair
#   2. TS feature extraction: 2-layer FC → FC + GroupNorm + FC

class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len     = model_args['seq_length']

        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        # ── delta 2: GroupNorm replaces BN on the TS branch ──
        self.gn = nn.GroupNorm(4, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim

        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = (self.ts_feat_dim + self.node_dim +
                             model_args['time_emb_dim'] * 2)
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)

        # ── delta 1: learned bias for pairwise affinity ──
        num_nodes = model_args.get('num_nodes', 207)
        self.pair_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, L = X.shape
        X = X.view(B * N, L)

        # ── delta 2: GroupNorm ──
        h = F.relu(self.fc_ts_emb1(X))
        h = self.gn(h)
        dy_feat = self.fc_ts_emb2(self.dropout(h))
        dy_feat = dy_feat.view(B, N, -1)

        emb1 = E_d.unsqueeze(0).expand(B, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(B, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        adjacent_list = []
        for feat in [X1, X2]:
            Q = self.WQ(feat)
            K = self.WK(feat)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            # ── delta 1: add learned bias ──
            QKT = QKT + self.pair_bias.unsqueeze(0)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list
