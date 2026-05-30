import time

import torch
import torch.nn as nn


class EstimationGate(nn.Module):
    """Estimation gate — learns a soft gate in (0,1) to control the
    proportion of diffusion vs inherent signals in the decoupled
    representation.

    Walpurgis notes:
    - Gate values near 0 or 1 indicate strong signal separation;
      values near 0.5 suggest the model cannot distinguish the two
      components, which may indicate insufficient training or
      degenerate embeddings.
    - Gate statistics are tracked for diagnostics.
    """

    _call_count = 0

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)
        print(f"[Walpurgis::EstimationGate] init input_dim={2 * node_emb_dim + time_emb_dim * 2} "
              f"hidden_dim={hidden_dim} output_dim=1")

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step
        embeddings to roughly estimate the proportion of the two hidden time series."""
        EstimationGate._call_count += 1
        _verbose = (EstimationGate._call_count <= 5 or EstimationGate._call_count % 300 == 0)

        t0 = time.perf_counter()

        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            gate_mean = estimation_gate.mean().item()
            gate_std  = estimation_gate.std().item()
            gate_min  = estimation_gate.min().item()
            gate_max  = estimation_gate.max().item()
            # Diagnose gate health: values near 0.5 = poor separation
            near_half = ((estimation_gate > 0.4) & (estimation_gate < 0.6)).float().mean().item()
            print(f"[Walpurgis::EstimationGate::forward] call#{EstimationGate._call_count} "
                  f"elapsed={elapsed_ms:.3f}ms "
                  f"gate mean={gate_mean:.4f} std={gate_std:.4f} "
                  f"range=[{gate_min:.4f},{gate_max:.4f}] "
                  f"near_0.5_ratio={near_half:.4f}")
            if near_half > 0.8:
                print(f"  ⚠ EstimationGate: {near_half*100:.1f}% of gate values near 0.5 — "
                      f"signal decoupling may be ineffective")

        return history_data
