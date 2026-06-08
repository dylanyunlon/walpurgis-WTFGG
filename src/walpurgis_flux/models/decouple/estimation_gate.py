"""Flux EstimationGate: 流式感知的门控 + 因果时序掩码.
与upstream(直接全序列门控)和vortex(同upstream门控)不同,
Flux在门控中加入因果时序掩码: 门控权重随时间步指数衰减,
使得更近的时间步获得更大的门控值, 配合流式推理的因果性."""
import torch
import torch.nn as nn
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class EstimationGate(nn.Module):
    """The estimation gate module with causal temporal decay."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(
            2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(
            hidden_dim, 1)
        # Flux: 可学习的因果衰减速率
        self.causal_decay_rate = nn.Parameter(
            torch.tensor(0.1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat,
                history_data):
        """Generate gate value with causal temporal decay.
        近的时间步门控值更大, 远的指数衰减."""
        batch_size, seq_length, _, _ = \
            time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)
        hidden = self.fully_connected_layer_1(
            estimation_gate_feat)
        hidden = self.activation(hidden)
        estimation_gate = torch.sigmoid(
            self.fully_connected_layer_2(hidden))[
            :, -history_data.shape[1]:, :, :]
        # Flux: 因果时序衰减
        L = history_data.shape[1]
        decay_rate = torch.sigmoid(self.causal_decay_rate)
        # 从最后一步(权重1.0)向前指数衰减
        time_positions = torch.arange(
            L, device=history_data.device).float()
        causal_weight = torch.exp(
            -decay_rate * (L - 1 - time_positions))
        causal_weight = causal_weight.view(1, L, 1, 1)
        # 门控值乘以因果衰减
        estimation_gate = estimation_gate * causal_weight
        history_data = history_data * estimation_gate
        if _FX_DBG:
            print(f"[FX:est_gate] gate_range="
                  f"[{estimation_gate.min().item():.4f},"
                  f"{estimation_gate.max().item():.4f}] "
                  f"decay_rate={decay_rate.item():.4f}",
                  file=sys.stderr)
        return history_data
