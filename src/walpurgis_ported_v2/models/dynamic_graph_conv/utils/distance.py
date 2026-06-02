"""
Distance function for dynamic graph construction.
Computes pairwise attention-based adjacency from node embeddings,
time features, and historical traffic signals.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

_DBG_DIST = ("--debug-dist" in sys.argv) or False


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.d_hidden = model_args['num_hidden']
        self.d_node   = model_args['node_hidden']
        self.d_slot   = self.d_hidden
        self.input_len = model_args['seq_length']

        # time-series feature extraction (FC encoder)
        self.dropout = nn.Dropout(model_args['dropout'])
        self.ts_fc1  = nn.Linear(self.input_len, self.d_hidden * 2)
        self.ts_fc2  = nn.Linear(self.d_hidden * 2, self.d_hidden)
        self.ts_bn   = nn.BatchNorm1d(self.d_hidden * 2)

        # time-slot embedding projection
        self.slot_proj = nn.Linear(model_args['time_emb_dim'], self.d_slot)

        # query / key projections for attention distance
        all_dim = self.d_hidden + self.d_node + model_args['time_emb_dim'] * 2
        self.W_Q = nn.Linear(all_dim, self.d_hidden, bias=False)
        self.W_K = nn.Linear(all_dim, self.d_hidden, bias=False)

    def forward(self, X, E_d, E_u, T_D, D_W):
        """
        Parameters
        ----------
        X   : [B, L, N, D]  — raw traffic signals
        E_d : [N, d_node]
        E_u : [N, d_node]
        T_D : [B, L, N, d_time]  — time-of-day embedding
        D_W : [B, L, N, d_time]  — day-of-week embedding

        Returns
        -------
        list[Tensor]  — two soft adjacency matrices, each [B, N, N]
        """
        # use last time step only for temporal context
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        # extract dynamic traffic feature
        ts = X[:, :, :, 0].transpose(1, 2).contiguous()        # [B, N, L]
        B, N, L = ts.shape
        ts_flat = ts.reshape(B * N, L)
        dyn_feat = self.ts_fc2(
            self.dropout(self.ts_bn(F.relu(self.ts_fc1(ts_flat))))
        )
        dyn_feat = dyn_feat.view(B, N, -1)

        # expand node embeddings to batch dim
        e_d = E_d.unsqueeze(0).expand(B, -1, -1)
        e_u = E_u.unsqueeze(0).expand(B, -1, -1)

        # build feature vectors for Q/K attention
        feat_1 = torch.cat([dyn_feat, T_D, D_W, e_d], dim=-1)
        feat_2 = torch.cat([dyn_feat, T_D, D_W, e_u], dim=-1)

        adj_list = []
        for feat in (feat_1, feat_2):
            Q = self.W_Q(feat)
            K = self.W_K(feat)
            attn = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.d_hidden)
            W = torch.softmax(attn, dim=-1)
            adj_list.append(W)

        if _DBG_DIST:
            for idx, adj in enumerate(adj_list):
                print(f"[DBG:dist] adj[{idx}]  shape={tuple(adj.shape)}  "
                      f"sparsity={(adj < 1e-4).float().mean().item():.3f}  "
                      f"max={adj.max().item():.4f}  min={adj.min().item():.6f}")
        return adj_list
