"""Vortex distance: Mahalanobis distance with learnable covariance.
Unlike upstream (scaled dot-product attention) and eclipse (cosine similarity + bias),
Vortex uses a learnable Mahalanobis metric: d(q,k) = (q-k)^T M (q-k) where M = L^T L
is a learnable positive semi-definite matrix. This captures feature correlations
in the distance computation for richer graph construction."""
import math, torch, torch.nn as nn, torch.nn.functional as F, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']; self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim; self.input_seq_len = model_args['seq_length']
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        self.all_feat_dim = self.hidden_dim + self.node_dim + model_args['time_emb_dim'] * 2
        # Project to a common space for Mahalanobis distance
        self.proj = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        # Learnable lower-triangular matrix L for M = L^T L (positive semi-definite)
        self.L_raw = nn.Parameter(torch.eye(self.hidden_dim) * 0.1)
        # LayerNorm for feature stability
        self.ln = nn.LayerNorm(self.hidden_dim * 2)

    def _mahalanobis_adjacency(self, X_proj):
        """Compute Mahalanobis distance-based adjacency.
        M = L^T L ensures positive semi-definite metric.
        sim(i,j) = -||proj_i - proj_j||_M then softmax."""
        B, N, D = X_proj.shape
        # Construct PSD metric M = L^T L via tril
        L = torch.tril(self.L_raw)
        M = torch.matmul(L.T, L)  # [D, D] PSD
        # Compute pairwise Mahalanobis distance
        diff = X_proj.unsqueeze(2) - X_proj.unsqueeze(1)  # [B, N, N, D]
        # d_M(i,j) = diff^T M diff = sum over d1,d2 of diff_d1 * M_d1d2 * diff_d2
        Md = torch.matmul(diff, M)  # [B, N, N, D]
        dist_sq = (Md * diff).sum(dim=-1)  # [B, N, N]
        # Convert distance to similarity via negative exponential
        sim = torch.exp(-dist_sq / max(D, 1))
        W = torch.softmax(sim, dim=-1)
        return W

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]; D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)
        dy_feat = self.fc_ts_emb2(self.dropout(self.ln(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(B, N, -1)
        emb1 = E_d.unsqueeze(0).expand(B, -1, -1); emb2 = E_u.unsqueeze(0).expand(B, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        adj_list = []
        for Xi in [X1, X2]:
            proj = self.proj(Xi)  # [B, N, hidden_dim]
            W = self._mahalanobis_adjacency(proj)
            adj_list.append(W)
        if _VX_DBG:
            L = torch.tril(self.L_raw)
            print(f"[VX:distance@distance] L_norm={L.norm().item():.4f} adj_mean={adj_list[0].mean().item():.6f} "
                  f"adj_sparsity={(adj_list[0]<0.01).float().mean().item():.2%}", file=sys.stderr)
        return adj_list
