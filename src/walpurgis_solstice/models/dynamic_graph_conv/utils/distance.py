import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:dist:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class ScaleNorm(nn.Module):
    """solstice: ScaleNorm替代GroupNorm — 可学习scale, 无偏置"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * dim ** 0.5)
        self.eps = eps
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm * self.scale

def mish(x):
    """solstice: Mish激活 x·tanh(softplus(x))"""
    return x * torch.tanh(F.softplus(x))

class DistanceFunction(nn.Module):
    """upstream: scaled-dot attention O(N^2)
    aurora: ELU-based linear attention O(N), GroupNorm
    solstice: FAVOR+ Performer随机特征近似 O(N), ScaleNorm"""
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        self.num_random_features = max(self.hidden_dim, 16)

        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        # solstice: ScaleNorm替代GroupNorm
        self.sn = ScaleNorm(self.hidden_dim * 2)

    def _random_feature_map(self, x):
        """solstice: FAVOR+ random feature map for linear attention O(N)
        Uses random orthogonal projections + positive random features"""
        d = x.shape[-1]
        m = self.num_random_features
        if not hasattr(self, '_omega') or self._omega.shape[0] != m or self._omega.shape[1] != d:
            omega = torch.randn(m, d, device=x.device)
            # Orthogonalize via QR for better approximation
            if m <= d:
                Q, _ = torch.linalg.qr(omega.T)
                omega = Q.T[:m]
            self.register_buffer('_omega', omega, persistent=False)
        proj = torch.einsum('...d,md->...m', x, self._omega.to(x.device))
        # FAVOR+ positive random features: exp(proj - ||x||^2/2)
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
        features = torch.exp(proj - norm_sq + math.log(m) / 2.0)
        return features / (features.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)
        # solstice: ScaleNorm(Mish(fc1))
        h = self.fc_ts_emb1(X)
        h = h.view(B, N, -1)
        h = self.sn(h)
        h = mish(h)
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
            # solstice: FAVOR+ random feature attention O(N)
            Q_prime = self._random_feature_map(Q)
            K_prime = self._random_feature_map(K)
            KV = torch.bmm(K_prime.transpose(-1, -2), torch.ones_like(K_prime))
            Z = 1.0 / (torch.bmm(Q_prime, KV.sum(dim=-1, keepdim=True)) + 1e-6)
            W = torch.bmm(Q_prime, K_prime.transpose(-1, -2)) * Z
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            _sdbg("attn_score", W)
            adjacent_list.append(W)
        return adjacent_list
