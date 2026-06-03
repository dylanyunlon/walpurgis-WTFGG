import torch
import torch.nn as nn
from walpurgis_ported_v10 import _dbg

_TAG = "gate"


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2

        # 改动1: 双头投影 — upstream 只有单条 FC→ReLU→FC
        # 这里用两组独立的 W 投影到 hidden, 再平均, 增加表达多样性
        self.head_a_fc1 = nn.Linear(in_dim, hidden_dim)
        self.head_b_fc1 = nn.Linear(in_dim, hidden_dim)

        # 改动2: ReLU → SiLU (Swish), 处处可微且允许负值通过
        self.act = nn.SiLU()

        # 改动3: GroupNorm 替代无归一化 — upstream 中间无任何 norm
        # 4 groups, 对 hidden_dim 做 group norm
        self.gn = nn.GroupNorm(min(4, hidden_dim), hidden_dim)

        self.out_fc = nn.Linear(hidden_dim, 1)

        # 改动4: 可学习温度 τ, 控制 sigmoid 的锐度
        self.log_tau = nn.Parameter(torch.zeros(1))  # init τ=1.0

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, N, _ = time_in_day_feat.shape

        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        cat_feat = torch.cat([time_in_day_feat, day_in_week_feat, eu, ed], dim=-1)

        _dbg(_TAG, "input", cat_feat=cat_feat, history=history_data)

        # 双头各自投影 + 激活
        h_a = self.act(self.head_a_fc1(cat_feat))
        h_b = self.act(self.head_b_fc1(cat_feat))

        # 平均融合两头
        h = (h_a + h_b) * 0.5

        # GroupNorm — reshape 让 GN 在 channel 维工作
        # GN expects (N, C, *), 把 (B,L,N,D) → (B*L*N, D, 1) → GN → 还原
        orig_shape = h.shape
        h_flat = h.reshape(-1, orig_shape[-1]).unsqueeze(-1)
        h_flat = self.gn(h_flat)
        h = h_flat.squeeze(-1).reshape(orig_shape)

        # 输出投影 + 温度缩放 sigmoid
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        gate_logit = self.out_fc(h) / tau
        gate_val = torch.sigmoid(gate_logit)[:, -history_data.shape[1]:, :, :]

        _dbg(_TAG, "gate", tau=tau, gate_mean=gate_val.mean(),
             gate_min=gate_val.min(), gate_max=gate_val.max())

        out = history_data * gate_val

        _dbg(_TAG, "output", gated=out)
        return out
