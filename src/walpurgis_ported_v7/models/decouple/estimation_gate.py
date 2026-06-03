import torch
import torch.nn as nn
import sys

_DBG_GATE = ("--dbg-gate" in sys.argv)


def _dump_gate(tag, t):
    if not _DBG_GATE:
        return
    with torch.no_grad():
        print(f"[DBG-GATE][{tag}] shape={list(t.shape)}  "
              f"min={t.min().item():.5f}  max={t.max().item():.5f}  "
              f"mean={t.mean().item():.5f}  std={t.std().item():.5f}")


class EstimationGate(nn.Module):
    """Estimation gate — 算法改动:
    1. 用 GELU 替换 ReLU (更平滑的梯度流)
    2. 在拼接特征上先做 LayerNorm 再过 FC, 稳定训练初期的梯度幅度
    3. 增加 residual 缩放: gate * data + (1-gate) * data 的加权形式
    """

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        cat_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.layer_norm = nn.LayerNorm(cat_dim)
        self.fc1 = nn.Linear(cat_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape

        eu = node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_length, -1, -1)
        ed = node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_length, -1, -1)

        cat_feat = torch.cat([time_in_day_feat, day_in_week_feat, eu, ed],
                             dim=-1)
        _dump_gate("cat_feat_pre_ln", cat_feat)

        # LayerNorm on concatenated features
        cat_feat = self.layer_norm(cat_feat)
        _dump_gate("cat_feat_post_ln", cat_feat)

        hidden = self.fc1(cat_feat)
        hidden = self.activation(hidden)       # GELU instead of ReLU
        gate_val = torch.sigmoid(self.fc2(hidden))
        gate_val = gate_val[:, -history_data.shape[1]:, :, :]

        _dump_gate("gate_value", gate_val)

        # 算法改动: soft residual gating — 不是纯乘法 gate*x,
        # 而是 gate*x + (1-gate)*x * 0.1 (leak), 保证即使 gate→0
        # 也有微弱梯度流过
        leak = 0.1
        history_data = gate_val * history_data + (1.0 - gate_val) * history_data * leak

        _dump_gate("gated_output", history_data)
        return history_data
