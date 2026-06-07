"""Eclipse distance: cosine similarity + learnable bias + LayerNorm."""
import math, torch, torch.nn as nn, torch.nn.functional as F, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

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
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.ln = nn.LayerNorm(self.hidden_dim * 2)  # LayerNorm instead of BN
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # learnable temperature
        self.cos_bias = nn.Parameter(torch.zeros(1))    # learnable bias

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
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        for Xi in [X1, X2]:
            Q = self.WQ(Xi); K = self.WK(Xi)
            # Cosine similarity + learnable bias (vs upstream scaled-dot)
            Q_norm = F.normalize(Q, dim=-1); K_norm = F.normalize(K, dim=-1)
            cos_sim = torch.bmm(Q_norm, K_norm.transpose(-1, -2))
            W = torch.softmax((cos_sim * tau + self.cos_bias), dim=-1)
            adj_list.append(W)
        if _ECL_DBG: print(f"[ECL:distance] tau={tau.item():.4f} bias={self.cos_bias.item():.4f} attn_mean={adj_list[0].mean().item():.6f}", file=sys.stderr)
        return adj_list
