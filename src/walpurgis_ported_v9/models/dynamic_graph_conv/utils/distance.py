"""
distance.py — v9 port
Algo delta:
  1. 单头 QK attention → 双头 (n_heads=2) 分别算 QKT 再 average
  2. BatchNorm1d → LayerNorm (对变 batch_size 更鲁棒)
  3. TS 特征提取加残差 shortcut: dy_feat += X_proj
  4. softmax 后接 tanh 压缩, 限制邻接权重上界
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v9 import _dbg

_TAG = "distance"
_N_DIST_HEADS = 2


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.input_seq_len = model_args['seq_length']

        # TS feature extraction
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # v9: residual projection for TS shortcut
        self.ts_proj = nn.Linear(self.input_seq_len, self.hidden_dim)
        # v9: LayerNorm instead of BN
        self.ln = nn.LayerNorm(self.hidden_dim * 2)

        # time slot
        self.time_slot_emb = nn.Linear(model_args['time_emb_dim'], self.hidden_dim)

        # distance score — v9: multi-head
        all_feat = self.hidden_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.n_heads = _N_DIST_HEADS
        self.head_dim = self.hidden_dim // self.n_heads
        self.WQ = nn.Linear(all_feat, self.hidden_dim, bias=False)
        self.WK = nn.Linear(all_feat, self.hidden_dim, bias=False)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X_raw = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X_raw.shape
        X_flat = X_raw.view(B * N, S)

        # v9: LayerNorm path
        h = F.relu(self.fc_ts1(X_flat))
        h = self.ln(h)
        dy_feat = self.fc_ts2(self.dropout(h))
        # v9: residual shortcut
        dy_feat = dy_feat + self.ts_proj(X_flat)
        dy_feat = dy_feat.view(B, N, -1)

        emb1 = E_d.unsqueeze(0).expand(B, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(B, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        adj_list = []
        for feat in [X1, X2]:
            Q = self.WQ(feat)   # [B, N, hidden]
            K = self.WK(feat)
            # v9: reshape to multi-head
            Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
            K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)
            # v9: average over heads
            W = attn.mean(dim=1)  # [B, N, N]
            # v9: tanh compression
            W = torch.tanh(W)
            adj_list.append(W)

        _dbg(_TAG, f"dist  heads={self.n_heads}  "
                    f"adj0∈[{adj_list[0].min().item():.4f},{adj_list[0].max().item():.4f}]  "
                    f"adj1∈[{adj_list[1].min().item():.4f},{adj_list[1].max().item():.4f}]")
        return adj_list
