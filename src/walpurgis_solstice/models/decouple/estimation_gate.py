import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:gate:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:gate:{tag}] {val}", file=sys.stderr)

class EstimationGate(nn.Module):
    """upstream: 2层FC+ReLU门控
    solstice: 3层FC + Mish激活 + channel-shuffle跨通道混合 + 可学习温度τ"""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        in_dim = 2 * node_emb_dim + time_emb_dim * 2
        # solstice: 3层FC代替2层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # solstice: Mish激活 x*tanh(softplus(x))
        self.act = nn.Mish()
        # solstice: channel-shuffle groups
        self._shuffle_groups = max(1, hidden_dim // 8)
        # solstice: 可学习温度参数控制sigmoid锐度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _channel_shuffle(self, x):
        """solstice: channel shuffle — 分组后交错排列通道, 促进跨通道信息流动"""
        B_dims = x.shape[:-1]
        C = x.shape[-1]
        g = self._shuffle_groups
        if C % g != 0:
            return x
        x = x.view(*B_dims, g, C // g)
        x = x.transpose(-2, -1).contiguous()
        x = x.view(*B_dims, C)
        return x

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        B, L, _, _ = time_in_day_feat.shape
        feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        ], dim=-1)

        h = self.act(self.fc1(feat))
        # solstice: channel shuffle
        h = self._channel_shuffle(h)

        h = self.act(self.fc2(h))
        # solstice: 温度缩放sigmoid
        tau = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate = torch.sigmoid(self.fc3(h) / tau)[:, -history_data.shape[1]:, :, :]
        _adbg("gate_val", gate)
        _adbg("temperature", tau)
        return history_data * gate
