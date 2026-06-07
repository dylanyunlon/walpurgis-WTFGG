import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:dist:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)

class DistanceFunction(nn.Module):
    """upstream: scaled-dot attention O(N^2)
    solstice: Performer随机正交特征距离 — 用正交随机矩阵投影, 更低方差"""
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
        # upstream: BN; solstice: InstanceNorm1d
        self.in_norm = nn.InstanceNorm1d(self.hidden_dim * 2, affine=True)
        # solstice: 随机特征维度
        self._n_features = max(16, self.hidden_dim)

    def _performer_feature_map(self, x):
        """solstice: 正随机特征映射 — 正交初始化减少方差"""
        d = x.shape[-1]
        m = self._n_features
        device = x.device
        # 正交随机矩阵
        omega = torch.randn(d, m, device=device)
        q, _ = torch.linalg.qr(omega)
        omega = q * math.sqrt(d)
        proj = torch.matmul(x, omega[:, :m])
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(m)
        return F.relu(phi) + 1e-6

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        B, N, S = X.shape
        X = X.view(B * N, S)
        # upstream: BN(ReLU(fc1))
        # solstice: InstanceNorm(Mish(fc1))
        h = self.fc_ts_emb1(X)
        h = h.view(B, N, -1).permute(0, 2, 1)  # [B, C, N]
        h = self.in_norm(h)
        h = F.mish(h)
        h = h.permute(0, 2, 1).reshape(B * N, -1)
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
            # solstice: Performer随机特征距离
            Q_prime = self._performer_feature_map(Q)
            K_prime = self._performer_feature_map(K)
            # FAVOR+ approx: W ≈ Q'K'^T / (Q' * sum(K'^T))
            K_sum = K_prime.sum(dim=1, keepdim=True)  # [B, 1, 2m]
            denom = torch.bmm(Q_prime, K_sum.transpose(-1, -2)) + 1e-6  # [B, N, 1]
            W = torch.bmm(Q_prime, K_prime.transpose(-1, -2)) / denom  # [B, N, N]
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            _adbg("attn_score", W)
            adjacent_list.append(W)
        return adjacent_list
