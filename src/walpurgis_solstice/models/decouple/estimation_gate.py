import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:gate:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:gate:{tag}] {val}", file=sys.stderr)


class ScaleNorm(nn.Module):
    """solstice: ScaleNorm"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return self.g * x / norm


class EstimationGate(nn.Module):
    """upstream: 2层FC+ReLU门控
    solstice: 3层FC + Mish激活 + ScaleNorm + 可学习温度τ"""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # solstice: ScaleNorm中间层
        self.sn = ScaleNorm(hidden_dim)
        # solstice: 可学习温度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ], dim=-1)

        h = self._mish(self.fc1(feat))
        h = self.sn(h)
        h = self._mish(self.fc2(h))
        # solstice: 温度缩放sigmoid
        tau = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        _sdbg("gate_val", gate)
        _sdbg("temperature", tau)
        return history_data * gate
