"""
Distance function for dynamic graph construction via attention.
"""
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_DBG = ("--debug-dist" in sys.argv)


class DistanceFunction(nn.Module):

    def __init__(self, **kw):
        super().__init__()
        d_h   = kw['num_hidden']
        d_n   = kw['node_hidden']
        d_t   = kw['time_emb_dim']
        L_in  = kw['seq_length']

        # time-series feature extractor
        self.drop = nn.Dropout(kw['dropout'])
        self.ts_fc1 = nn.Linear(L_in, d_h * 2)
        self.ts_fc2 = nn.Linear(d_h * 2, d_h)
        self.ts_bn  = nn.BatchNorm1d(d_h * 2)

        # attention projections
        feat_dim = d_h + d_n + d_t * 2
        self.W_Q = nn.Linear(feat_dim, d_h, bias=False)
        self.W_K = nn.Linear(feat_dim, d_h, bias=False)
        self.scale = math.sqrt(d_h)

    def forward(self, X, E_d, E_u, T_D, D_W):
        # last-step pooling for temporal embeddings
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        # dynamic feature from raw signal
        ts = X[:, :, :, 0].transpose(1, 2).contiguous()   # (B, N, L)
        B, N, L = ts.shape
        ts_flat = ts.view(B * N, L)
        dyn = self.ts_fc2(self.drop(self.ts_bn(F.relu(self.ts_fc1(ts_flat)))))
        dyn = dyn.view(B, N, -1)

        # expand node embeddings
        e1 = E_d.unsqueeze(0).expand(B, -1, -1)
        e2 = E_u.unsqueeze(0).expand(B, -1, -1)

        feat_src = torch.cat([dyn, T_D, D_W, e1], dim=-1)
        feat_tgt = torch.cat([dyn, T_D, D_W, e2], dim=-1)

        adj_list = []
        for f in [feat_src, feat_tgt]:
            Q = self.W_Q(f)
            K = self.W_K(f)
            attn = torch.bmm(Q, K.transpose(-1, -2)) / self.scale
            A = torch.softmax(attn, dim=-1)
            adj_list.append(A)

        if _DBG:
            print(f"[DBG:dist] forward  B={B} N={N}  "
                  f"A0_mean={adj_list[0].mean().item():.4f}  "
                  f"A1_mean={adj_list[1].mean().item():.4f}")

        return adj_list
