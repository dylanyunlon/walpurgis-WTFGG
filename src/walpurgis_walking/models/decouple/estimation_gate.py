import torch
import torch.nn as nn
from walpurgis_walking import _dbg

_TAG = "gate"


class EstimationGate(nn.Module):
    """upstream: FC→ReLU→FC→sigmoid
    改动: 双头FC投影→SiLU→GroupNorm→FC→温度缩放sigmoid
    """
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # 改动1: 双头 (upstream 单条)
        self.head_a = nn.Linear(in_dim, hidden_dim)
        self.head_b = nn.Linear(in_dim, hidden_dim)
        # 改动2: SiLU 替代 ReLU
        self.act = nn.SiLU()
        # 改动3: GroupNorm (upstream 无 norm)
        self.gn = nn.GroupNorm(min(4, hidden_dim), hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, 1)
        # 改动4: 可学习温度 τ
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, N, _ = time_in_day_feat.shape
        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        cat_feat = torch.cat([time_in_day_feat, day_in_week_feat, eu, ed], dim=-1)

        _dbg(_TAG, "input", cat_feat=cat_feat, history=history_data)

        ha = self.act(self.head_a(cat_feat))
        hb = self.act(self.head_b(cat_feat))
        h = (ha + hb) * 0.5

        # GroupNorm: reshape (B,L,N,D)→(B*L*N, D, 1)→GN→还原
        osh = h.shape
        h = self.gn(h.reshape(-1, osh[-1]).unsqueeze(-1)).squeeze(-1).reshape(osh)

        tau = torch.exp(self.log_tau).clamp(0.1, 10.0)
        logit = self.out_fc(h) / tau
        gate = torch.sigmoid(logit)[:, -history_data.shape[1]:, :, :]

        _dbg(_TAG, "gate", tau=tau, g_mean=gate.mean(), g_min=gate.min(), g_max=gate.max())

        # 改动5: gate 通过率监控
        ge = gate.detach().norm()
        ie = history_data.detach().norm().clamp(min=1e-8)
        _dbg(_TAG, "passthrough", ratio=ge / ie)

        return history_data * gate
