"""Distance function — dual-head attention + residual time-series encoding.

Changes
-------
1. Attention distance uses 2 independent heads and averages them.  The
   upstream uses a single Q/K projection; dual-head captures both
   symmetric and asymmetric distance patterns simultaneously.
2. Time-series feature extraction: the two-layer MLP now has a residual
   shortcut (project input to hidden_dim, then add back).  This helps
   preserve raw temporal patterns that the MLP might otherwise wash out.
3. BatchNorm on the TS path is replaced by LayerNorm for batch-size
   invariance (same motivation as GroupNorm in dif_model).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v6 import _dbg


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.input_seq_len = model_args['seq_length']

        # TS feature extraction with residual shortcut
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # residual projection: seq_len → hidden_dim
        self.ts_skip = nn.Linear(self.input_seq_len, self.hidden_dim)
        # LayerNorm instead of BatchNorm
        self.ln = nn.LayerNorm(self.hidden_dim * 2)

        self.time_slot_emb_dim = self.hidden_dim
        self.time_slot_proj = nn.Linear(
            model_args['time_emb_dim'], self.time_slot_emb_dim)

        all_dim = self.hidden_dim + self.node_dim + model_args['time_emb_dim'] * 2
        # 2-head attention
        self.n_heads = 2
        self.WQ = nn.ModuleList([
            nn.Linear(all_dim, self.hidden_dim, bias=False)
            for _ in range(self.n_heads)])
        self.WK = nn.ModuleList([
            nn.Linear(all_dim, self.hidden_dim, bias=False)
            for _ in range(self.n_heads)])

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X_raw = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, L = X_raw.shape
        X_flat = X_raw.view(B * N, L)

        # residual TS encoding
        h = F.relu(self.fc_ts1(X_flat))
        h = self.ln(h)
        h = self.dropout(h)
        dy_feat = self.fc_ts2(h) + self.ts_skip(X_flat)    # ← residual
        dy_feat = dy_feat.view(B, N, -1)

        emb1 = E_d.unsqueeze(0).expand(B, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(B, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        adj_list = []
        for X_in in (X1, X2):
            head_results = []
            for h_idx in range(self.n_heads):
                Q = self.WQ[h_idx](X_in)
                K = self.WK[h_idx](X_in)
                attn = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
                W = torch.softmax(attn, dim=-1)
                head_results.append(W)
            # average over heads
            avg_W = sum(head_results) / self.n_heads
            adj_list.append(avg_W)

        _dbg("Distance", adj_list[0], n_heads=self.n_heads)
        return adj_list
