import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceFunction(nn.Module):
    """Compute pairwise node distance (attention) from dynamic features,
    time embeddings, and trainable node embeddings.

    Walpurgis notes:
    - The QK^T / sqrt(d) attention is O(N^2) per batch — this is the
      primary memory bottleneck in graph construction.
    - BN layer on the time-series features can cause issues with very
      small batch sizes; tracked via probe.
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']
        self.time_slot_emb_dim  = self.hidden_dim
        self.input_seq_len      = model_args['seq_length']
        # Time Series Feature Extraction
        self.dropout    = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        # Time Slot Embedding Extraction
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        # Distance Score
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        print(f"[Walpurgis::DistanceFunction] init hidden={self.hidden_dim} "
              f"node_dim={self.node_dim} all_feat_dim={self.all_feat_dim} "
              f"seq_len={self.input_seq_len}")

    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)

    def forward(self, X, E_d, E_u, T_D, D_W):
        DistanceFunction._call_count += 1
        _verbose = (DistanceFunction._call_count <= 3 or
                     DistanceFunction._call_count % 500 == 0)
        t0 = time.perf_counter()

        # last pooling
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        # dynamic information
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)

        # BN + FC feature extraction
        bn_input = F.relu(self.fc_ts_emb1(X))
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(bn_input)))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        # node embedding
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        # distance calculation via scaled dot-product attention
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X  = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"[Walpurgis::DistanceFunction::forward] call#{DistanceFunction._call_count} "
                  f"batch={batch_size} nodes={num_nodes} elapsed={elapsed_ms:.3f}ms")
            for idx, adj in enumerate(adjacent_list):
                print(f"  adj[{idx}] shape={list(adj.shape)} "
                      f"mean={adj.mean().item():.6f} max={adj.max().item():.6f} "
                      f"min={adj.min().item():.6f}")
                _nan = torch.isnan(adj).any().item()
                if _nan:
                    print(f"  ⚠ adj[{idx}] contains NaN!")

        return adjacent_list
