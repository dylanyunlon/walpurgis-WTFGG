"""
DistanceFunction — walpurgis_ported_v4
Modifications:
  - Attention score: added learnable temperature parameter (initialized to 1.0)
    that scales the softmax logits, allowing the model to learn sharper/softer
    attention distributions
  - forward() prints attention entropy and sparsity stats for debugging
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

_V4_DEBUG = True


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']

        # Time Series Feature Extraction
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim

        # Time Slot Embedding
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        # Distance Score
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # v4: learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, X, E_d, E_u, T_D, D_W):
        # last pooling
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        # distance calculation
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X = [X1, X2]

        adjacent_list = []
        for idx, feat in enumerate(X):
            Q = self.WQ(feat)
            K = self.WK(feat)
            # v4: temperature-scaled attention
            scale = math.sqrt(self.hidden_dim) * self.temperature.clamp(min=0.1)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / scale
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)

            if _V4_DEBUG:
                # compute attention entropy and sparsity
                entropy = -(W * (W + 1e-8).log()).sum(-1).mean().item()
                sparsity = (W < 0.01).float().mean().item()
                print(f"[v4-DBG][DistanceFunction] modality={idx} "
                      f"temp={self.temperature.item():.4f} "
                      f"attn_entropy={entropy:.4f} "
                      f"sparsity(<0.01)={sparsity:.4f} "
                      f"W_shape={tuple(W.shape)}",
                      file=sys.stderr)

        return adjacent_list
