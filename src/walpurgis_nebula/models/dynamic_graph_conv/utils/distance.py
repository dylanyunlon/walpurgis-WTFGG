"""Nebula distance: hyperbolic distance on Poincaré ball model."""
import math, torch, torch.nn as nn, torch.nn.functional as F, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class PoincareDistance(nn.Module):
    """Compute pairwise distance in the Poincaré ball model of hyperbolic space.
    d(u,v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    Points are projected inside the ball via clipping norm < 1-eps."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def _project(self, x):
        """Project points into the Poincaré ball (norm < 1-eps)."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        max_norm = 1.0 - self.eps
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    def forward(self, u, v):
        """u: [B, N, D], v: [B, N, D] -> dist: [B, N, N]"""
        u = self._project(u)
        v = self._project(v)
        # ||u - v||^2 pairwise
        diff_sq = torch.cdist(u, v, p=2).pow(2)  # [B, N, N]
        u_sq = (u ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        v_sq = (v ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # [B, 1, N]
        denom = (1.0 - u_sq) * (1.0 - v_sq)  # [B, N, N]
        denom = denom.clamp(min=self.eps)
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = arg.clamp(min=1.0 + self.eps)
        dist = torch.acosh(arg)
        return dist


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_length']
        self.dropout = nn.Dropout(model_args['dropout'])
        # Time series feature extraction
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        # Time slot embedding
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)
        # Nebula: project combined features into Poincaré ball
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2
        self.hyp_proj_1 = nn.Linear(self.all_feat_dim, self.hidden_dim)
        self.hyp_proj_2 = nn.Linear(self.all_feat_dim, self.hidden_dim)
        self.poincare = PoincareDistance()
        # Learnable temperature for distance-to-similarity
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        batch_size, num_nodes, seq_len = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        # Nebula: project to Poincaré ball and compute hyperbolic distance
        h1 = torch.tanh(self.hyp_proj_1(X1)) * 0.9  # keep inside ball
        h2 = torch.tanh(self.hyp_proj_2(X2)) * 0.9
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        adjacent_list = []
        for h in [h1, h2]:
            dist = self.poincare(h, h)  # [B, N, N]
            # Convert distance to similarity: softmax(-dist/temperature)
            W = torch.softmax(-dist / temperature, dim=-1)
            adjacent_list.append(W)
        if _NEB_DBG:
            print(f"[NEB:hyp_dist@distance] temp={temperature.item():.4f} "
                  f"dist_mean={dist.mean().item():.4f} W_sparsity={(W < 0.01).float().mean().item():.2%}", file=sys.stderr)
        return adjacent_list
