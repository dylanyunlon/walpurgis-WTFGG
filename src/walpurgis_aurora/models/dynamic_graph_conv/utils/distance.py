import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:dist:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DistanceFunction(nn.Module):
    """upstream: scaled-dot attention O(N^2)
    aurora: ELU-based linear attention O(N), GroupNorm替代BN"""
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
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        # upstream: BN; aurora: GroupNorm
        self.gn = nn.GroupNorm(min(4, self.hidden_dim * 2), self.hidden_dim * 2)

    def _elu_feature_map(self, x):
        """aurora: ELU+1 kernel for linear attention"""
        return F.elu(x) + 1

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)
        # upstream: BN(ReLU(fc1))
        # aurora: GroupNorm(GELU(fc1))
        h = self.fc_ts_emb1(X)
        h = h.view(B, N, -1).permute(0, 2, 1).unsqueeze(0)
        h = h.reshape(B, -1, N)
        h = self.gn(h)
        h = F.gelu(h)
        h = h.view(B * N, -1)
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
            # upstream: scaled-dot QK^T O(N^2)
            # aurora: linear attention via ELU kernel O(N)
            Q_prime = self._elu_feature_map(Q)
            K_prime = self._elu_feature_map(K)
            KV = torch.bmm(K_prime.transpose(-1, -2), torch.ones_like(K_prime))
            Z = 1.0 / (torch.bmm(Q_prime, KV.sum(dim=-1, keepdim=True)) + 1e-6)
            W = torch.bmm(Q_prime, K_prime.transpose(-1, -2)) * Z
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            _adbg("attn_score", W)
            adjacent_list.append(W)
        return adjacent_list
