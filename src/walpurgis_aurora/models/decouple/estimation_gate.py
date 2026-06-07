import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:gate:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[AUR:gate:{tag}] {val}", file=sys.stderr)

class EstimationGate(nn.Module):
    """upstream: 2层FC+ReLU门控
    aurora: 3层FC + GELU激活 + Squeeze-Excitation通道注意力 + 可学习温度τ"""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # aurora: 3层FC代替2层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # aurora: GELU代替ReLU
        self.act = nn.GELU()
        # aurora: Squeeze-Excitation通道注意力
        se_mid = max(1, hidden_dim // 4)
        self.se_squeeze = nn.Linear(hidden_dim, se_mid)
        self.se_excite = nn.Linear(se_mid, hidden_dim)
        # aurora: 可学习温度参数控制sigmoid锐度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ], dim=-1)

        h = self.act(self.fc1(feat))
        # aurora: SE通道注意力
        se = h.mean(dim=(0, 1))  # [N, hidden]
        se = torch.sigmoid(self.se_excite(self.act(self.se_squeeze(se))))
        h = h * se.unsqueeze(0).unsqueeze(0)

        h = self.act(self.fc2(h))
        # aurora: 温度缩放sigmoid
        tau = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        _adbg("gate_val", gate)
        _adbg("temperature", tau)
        return history_data * gate
