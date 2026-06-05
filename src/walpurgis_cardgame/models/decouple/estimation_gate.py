"""
estimation_gate.py — CardGame EstimationGate
算法改写 (vs upstream):
  - 2层FC → 3层FC + 瓶颈结构 (dim → dim//4 → dim//2 → 1)
  - ReLU → Swish (SiLU) 激活
  - 新增LayerNorm在瓶颈后
  - 新增可学习温度参数 τ 控制sigmoid锐度
"""
import os
import sys
import torch
import torch.nn as nn

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="EstGate"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class EstimationGate(nn.Module):
    """CardGame EstimationGate: 3层瓶颈FC + Swish + LayerNorm + learnable temperature

    改写:
      - 输入拼接(time_in_day, day_in_week, node_emb_u, node_emb_d)
      - 3层FC瓶颈: full_dim → bottleneck_dim → mid_dim → 1
      - Swish激活替代ReLU
      - 瓶颈层后接LayerNorm
      - 可学习温度τ控制sigmoid(x/τ)的锐度
    """

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        full_dim = 2 * node_emb_dim + time_emb_dim * 2
        bottleneck_dim = max(full_dim // 4, 8)
        mid_dim = max(full_dim // 2, 16)

        # 3层FC瓶颈
        self.fc_bottleneck = nn.Linear(full_dim, bottleneck_dim)
        self.ln_bottleneck = nn.LayerNorm(bottleneck_dim)
        self.fc_mid = nn.Linear(bottleneck_dim, mid_dim)
        self.fc_out = nn.Linear(mid_dim, 1)

        # Swish激活 (SiLU)
        self.activation = nn.SiLU()

        # 可学习温度参数 (初始化为1.0, 范围约0.1~10)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        _dbg("input.history", history_data)
        _dbg("input.tid", time_in_day_feat)

        batch_size, seq_length, _, _ = time_in_day_feat.shape

        # 拼接特征
        estimation_gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        _dbg("concat_feat", estimation_gate_feat)

        # 3层瓶颈FC + Swish + LayerNorm
        hidden = self.fc_bottleneck(estimation_gate_feat)
        hidden = self.ln_bottleneck(hidden)
        hidden = self.activation(hidden)
        _dbg("after_bottleneck", hidden)

        hidden = self.fc_mid(hidden)
        hidden = self.activation(hidden)
        _dbg("after_mid", hidden)

        # 可学习温度sigmoid
        temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        gate_logits = self.fc_out(hidden)
        estimation_gate = torch.sigmoid(gate_logits / temperature)
        estimation_gate = estimation_gate[:, -history_data.shape[1]:, :, :]
        _dbg("gate_values", estimation_gate)
        _dbg("temperature", temperature)

        history_data = history_data * estimation_gate
        _dbg("gated_output", history_data)
        return history_data
