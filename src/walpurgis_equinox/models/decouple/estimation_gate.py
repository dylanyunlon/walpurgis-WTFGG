import torch
import torch.nn as nn
import torch.nn.utils as nnutils
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:gate:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:gate:{tag}] {val}", file=sys.stderr)

class EstimationGate(nn.Module):
    """upstream: 2层FC+ReLU门控
    equinox: 3层WeightNorm-FC + GELU激活 + 通道缩放 + 可学习温度τ"""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # equinox: WeightNorm包裹的3层FC代替普通FC
        self.fc1 = nnutils.weight_norm(nn.Linear(in_dim, hidden_dim))
        self.fc2 = nnutils.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.Linear(hidden_dim, 1)
        # equinox: GELU代替ReLU
        self.act = nn.GELU()
        # equinox: 可学习通道缩放向量 (替代SE模块, 更轻量)
        self.channel_scale = nn.Parameter(torch.ones(hidden_dim))
        # equinox: 可学习温度参数控制sigmoid锐度
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
        # equinox: 可学习通道缩放
        h = h * self.channel_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        h = self.act(self.fc2(h))
        # equinox: 温度缩放sigmoid
        tau = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        _edbg("gate_val", gate)
        _edbg("temperature", tau)
        return history_data * gate
