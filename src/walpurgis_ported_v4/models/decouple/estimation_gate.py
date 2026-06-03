"""
EstimationGate — walpurgis_ported_v4
Modifications:
  - Added information bottleneck: 3-layer FC with LayerNorm between hidden layers
    (original: 2-layer FC). Bottleneck dim = hidden_dim // 2.
  - forward() prints gate statistics (mean, std, min, max) for debugging
"""
import torch
import torch.nn as nn
import sys

_V4_DEBUG = True


class EstimationGate(nn.Module):
    """Gate module that estimates the proportion of diffusion vs. inherent signals.
    v4: 3-layer with LayerNorm bottleneck for more stable gating.
    """

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        input_dim = 2 * node_emb_dim + time_emb_dim * 2
        bottleneck_dim = max(hidden_dim // 2, 4)  # v4: information bottleneck

        self.fully_connected_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)          # v4: added
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)  # v4: added
        self.activation_2 = nn.ReLU()                       # v4: added
        self.fully_connected_layer_2 = nn.Linear(bottleneck_dim, 1)  # v4: from bottleneck

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat,
                day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat,
            day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ], dim=-1)

        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        hidden = self.layer_norm(hidden)                    # v4: stabilize
        hidden = self.bottleneck(hidden)                    # v4: compress
        hidden = self.activation_2(hidden)                  # v4: nonlinear
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))
        estimation_gate = estimation_gate[:, -history_data.shape[1]:, :, :]

        if _V4_DEBUG:
            g = estimation_gate
            print(f"[v4-DBG][EstimationGate] gate stats: "
                  f"mean={g.mean().item():.4f} std={g.std().item():.4f} "
                  f"min={g.min().item():.4f} max={g.max().item():.4f} "
                  f"shape={tuple(g.shape)}",
                  file=sys.stderr)

        history_data = history_data * estimation_gate
        return history_data
