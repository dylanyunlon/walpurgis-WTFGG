import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

_DBG = ("--dbg" in sys.argv)


class DistanceFunction(nn.Module):
    """算法改动:
    1. 时序嵌入: fc1 -> ELU -> fc2 (原版 ReLU+BN, 改为 ELU 无需 BN 的激活)
       加 residual shortcut: 原始序列的线性投影 + MLP 输出
    2. 距离计算: 原版 scaled dot-product attention (QKT/sqrt(d))
       改为 cosine similarity + learnable temperature:
         sim = cos(Q, K) * exp(tau)
       cosine 天然归一化, temperature 控制锐度
    """

    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']

        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # 算法改动: residual shortcut for ts embedding
        self.ts_skip = nn.Linear(self.input_seq_len, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim

        self.time_slot_embedding = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = (self.ts_feat_dim + self.node_dim
                             + model_args['time_emb_dim'] * 2)
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)

        # 算法改动: learnable temperature for cosine similarity
        self.tau = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(self.hidden_dim))))

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X_flat = X.view(batch_size * num_nodes, seq_len)

        # 算法改动: ELU + residual
        dy_feat = self.fc_ts_emb2(
            self.dropout(F.elu(self.fc_ts_emb1(X_flat))))
        skip = self.ts_skip(X_flat)
        dy_feat = dy_feat + skip  # residual
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X_pair = [X1, X2]

        adjacent_list = []
        for feat in X_pair:
            Q = self.WQ(feat)
            K = self.WK(feat)
            # 算法改动: cosine similarity + learnable temperature
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            sim = torch.bmm(Q_norm, K_norm.transpose(-1, -2))
            temperature = torch.exp(self.tau)
            W = torch.softmax(sim * temperature, dim=-1)
            adjacent_list.append(W)

        if _DBG:
            with torch.no_grad():
                print(f"[DBG][DistanceFunction] tau={self.tau.item():.4f}  "
                      f"temp={temperature.item():.4f}  "
                      f"adj[0] sparsity="
                      f"{(adjacent_list[0] < 0.01).float().mean().item():.3f}",
                      flush=True)
        return adjacent_list
