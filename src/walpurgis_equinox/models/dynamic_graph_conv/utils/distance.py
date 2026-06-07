import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nnutils
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:dist:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DistanceFunction(nn.Module):
    """upstream: scaled-dot attention O(N^2)
    equinox: Linformer低秩投影注意力 O(N*k), WeightNorm替代BN"""
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        self.num_nodes = model_args['num_nodes']

        self.dropout = nn.Dropout(model_args['dropout'])
        # equinox: WeightNorm包裹FC层
        self.fc_ts_emb1 = nnutils.weight_norm(nn.Linear(self.input_seq_len, self.hidden_dim * 2))
        self.fc_ts_emb2 = nnutils.weight_norm(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        self.ts_feat_dim = self.hidden_dim
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)

        # equinox: Linformer低秩投影矩阵 — 将N维投影到k维
        self.linformer_k = max(4, self.num_nodes // 4)
        self.E_proj = nn.Parameter(torch.randn(self.num_nodes, self.linformer_k) * 0.02)
        self.F_proj = nn.Parameter(torch.randn(self.num_nodes, self.linformer_k) * 0.02)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)
        # equinox: WeightNorm层 + GELU
        h = self.fc_ts_emb1(X)
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
            # equinox: Linformer — K投影到低秩空间 K'=E^T K, 然后 W=softmax(Q K'^T / sqrt(d)) F^T
            # 这样注意力从 O(N^2) 降到 O(N*k)
            E = self.E_proj[:N, :].to(K.device)  # [N, k]
            F_mat = self.F_proj[:N, :].to(K.device)  # [N, k]
            K_proj = torch.bmm(E.T.unsqueeze(0).expand(B, -1, -1),
                              K)  # [B, k, D]
            # Q @ K_proj^T -> [B, N, k]
            attn_low = torch.bmm(Q, K_proj.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            attn_low = F.softmax(attn_low, dim=-1)  # [B, N, k]
            # 将低秩注意力重建到 [B, N, N] 空间
            W = torch.bmm(attn_low, F_mat.T.unsqueeze(0).expand(B, -1, -1))  # [B, N, N]
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            _edbg("linformer_attn", W)
            adjacent_list.append(W)
        return adjacent_list
