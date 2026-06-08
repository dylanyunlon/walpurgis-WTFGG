"""
Corona EstimationGate — 算法改写:
  upstream: FC1 → ReLU → FC2 → sigmoid → element-wise multiply
  corona:  Attention Pooling (Q/K from node+time embed) → SiLU → learned bias → sigmoid
  改动幅度: ~25% (attention pooling替代plain FC, SiLU替代ReLU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ... import _dbg, dataflow_checkpoint


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        feat_dim = 2 * node_emb_dim + time_emb_dim * 2
        # Corona改写: attention pooling layer (替代upstream的fc1→relu→fc2)
        self.query_proj = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.attn_scale = hidden_dim ** 0.5
        # SiLU gating (替代upstream的ReLU)
        self.gate_fc = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()  # Corona: SiLU替代ReLU
        # 可学习偏移 (Corona新增)
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1, -1)
        ], dim=-1)

        dataflow_checkpoint("corona_gate.feat", estimation_gate_feat)

        # Corona: attention-based pooling — Q和K都从同一特征投影
        Q = self.query_proj(estimation_gate_feat)
        K = self.key_proj(estimation_gate_feat)
        # self-attention score per position → gating signal
        attn_energy = (Q * K).sum(dim=-1, keepdim=True) / self.attn_scale
        # SiLU activation → gate projection
        hidden = self.activation(attn_energy)
        estimation_gate = torch.sigmoid(
            self.gate_fc(Q * torch.sigmoid(attn_energy)) + self.gate_bias
        )[:, -history_data.shape[1]:, :, :]

        dataflow_checkpoint("corona_gate.output", estimation_gate)
        history_data = history_data * estimation_gate
        return history_data
