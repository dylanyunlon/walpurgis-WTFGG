"""Meridian InhModel — Highway GRU + Relative Position Bias Transformer.
Changes vs upstream:
  - GRU with highway gate: h_new = gate*gru_out + (1-gate)*x (upstream: vanilla GRU)
  - Transformer with additive relative position bias (upstream: no position bias)
  - Debug: prints hidden state evolution and attention entropy
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class HighwayGRULayer(nn.Module):
    """GRU with highway gating — allows input to bypass recurrence."""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.highway_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_len, num_nodes, hidden_dim = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for t in range(X.shape[1]):
            x_t = X[:, t, :]
            gru_out = self.gru_cell(x_t, hx)
            # highway gate
            gate_input = torch.cat([x_t, gru_out], dim=-1)
            gate = torch.sigmoid(self.highway_gate(gate_input))
            hx = gate * gru_out + (1.0 - gate) * x_t
            output.append(hx)
        output = torch.stack(output, dim=0)  # [seq, batch*nodes, hidden]
        output = self.dropout(output)
        if _DBG and output.shape[0] > 0:
            h_last = output[-1].detach()
            print(f"[MER:highway_gru] seq_len={output.shape[0]} "
                  f"h_last_norm={h_last.norm().item():.4f} "
                  f"gate_mean={gate.detach().mean().item():.4f}", file=sys.stderr)
        return output


class RelPosTransformerLayer(nn.Module):
    """Multi-head attention with additive relative position bias."""
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # relative position bias table (max 64 positions)
        self.max_pos = 64
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, self.max_pos * 2 - 1))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self.num_heads = num_heads

    def _get_rel_pos_bias(self, seq_len):
        """Compute relative position bias matrix."""
        positions = torch.arange(seq_len, device=self.rel_pos_bias.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos + self.max_pos - 1
        rel_pos = rel_pos.clamp(0, 2 * self.max_pos - 2)
        bias = self.rel_pos_bias[:, rel_pos]  # [heads, seq, seq]
        return bias

    def forward(self, X, K, V):
        seq_len = X.shape[0]
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
