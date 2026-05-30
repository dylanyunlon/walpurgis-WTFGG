"""
Walpurgis Estimation Gate — Adaptive Signal Separation with Entropy Regularization
=====================================================================================
Adapted from D2STGNN EstimationGate.

Algorithm changes:
  1. Gumbel-sigmoid gating: instead of plain sigmoid, add Gumbel noise during
     training to encourage the gate to push toward 0 or 1 (hard decisions).
     The noise temperature anneals over time for curriculum-style hardening.
  2. Gate entropy tracking: if most gates cluster at 0.5 for too long,
     the model may be stuck — we log this as a diagnostic.
  3. Separate FC paths for node embeddings vs time embeddings before fusion,
     giving each modality its own representation space before gating.
"""

import time
import math

import torch
import torch.nn as nn


class EstimationGate(nn.Module):
    """Learns a soft gate in (0,1) controlling diffusion vs inherent signal ratio.

    Walpurgis: Gumbel-sigmoid for sharper gates + modality-aware FC paths.
    """

    _call_count = 0
    _gumbel_tau = 1.0  # temperature for Gumbel noise, anneals over calls

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()

        # Walpurgis: separate projection for node and time modalities
        node_input_dim = 2 * node_emb_dim
        time_input_dim = 2 * time_emb_dim
        self.fc_node = nn.Linear(node_input_dim, hidden_dim // 2)
        self.fc_time = nn.Linear(time_input_dim, hidden_dim // 2)

        # Fused projection
        self.fc_fused = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_gate = nn.Linear(hidden_dim, 1)

        # Debug tracking
        self._gate_entropy_history = []
        self._tau_history = []

        print(f"[Walpurgis::EstimationGate] init node_dim={node_input_dim} "
              f"time_dim={time_input_dim} hidden={hidden_dim} "
              f"(modality-split FC paths)")

    def _gumbel_sigmoid(self, logits, tau=1.0, hard=False):
        """Gumbel-sigmoid: adds Gumbel noise for stochastic hard gates.

        During training, noise encourages binary-ish gate values.
        During eval (or when tau→0), degenerates to plain sigmoid.
        """
        if self.training and tau > 0.01:
            # Sample Gumbel noise
            U = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(U))
            noisy_logits = (logits + gumbel * 0.3) / tau  # 0.3 = noise scale
        else:
            noisy_logits = logits

        y = torch.sigmoid(noisy_logits)
        return y

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate gate value with Gumbel-sigmoid and modality-split processing."""
        EstimationGate._call_count += 1
        _verbose = (EstimationGate._call_count <= 5 or
                    EstimationGate._call_count % 300 == 0)
        t0 = time.perf_counter()

        batch_size, seq_length, _, _ = time_in_day_feat.shape

        # Walpurgis: separate modality projections
        # Node path: [B, L, N, 2*node_dim] → [B, L, N, hidden//2]
        node_feat = torch.cat([
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ], dim=-1)
        node_hidden = self.fc_node(node_feat)

        # Time path: [B, L, N, 2*time_dim] → [B, L, N, hidden//2]
        time_feat = torch.cat([time_in_day_feat, day_in_week_feat], dim=-1)
        time_hidden = self.fc_time(time_feat)

        # Fuse modalities
        fused = torch.cat([node_hidden, time_hidden], dim=-1)
        hidden = self.activation(self.fc_fused(fused))

        # Gumbel-sigmoid gate
        logits = self.fc_gate(hidden)

        # Anneal temperature: starts warm (explore), cools over time
        tau = max(0.1, EstimationGate._gumbel_tau * (0.9999 ** EstimationGate._call_count))
        estimation_gate = self._gumbel_sigmoid(logits, tau=tau)
        estimation_gate = estimation_gate[:, -history_data.shape[1]:, :, :]

        history_data = history_data * estimation_gate

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            gate_mean = estimation_gate.mean().item()
            gate_std = estimation_gate.std().item()
            gate_min = estimation_gate.min().item()
            gate_max = estimation_gate.max().item()
            # Entropy of gate distribution (treat each gate as Bernoulli)
            p = estimation_gate.mean(dim=(0, 1, 2)).squeeze()
            gate_entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).item()
            near_half = ((estimation_gate > 0.4) & (estimation_gate < 0.6)).float().mean().item()

            print(f"[Walpurgis::EstGate] call#{EstimationGate._call_count} "
                  f"elapsed={elapsed_ms:.3f}ms tau={tau:.4f}")
            print(f"  gate: mean={gate_mean:.4f} std={gate_std:.4f} "
                  f"[{gate_min:.4f}, {gate_max:.4f}] "
                  f"entropy={gate_entropy:.4f} near_0.5={near_half:.2%}")
            if near_half > 0.8:
                print(f"  ⚠ {near_half*100:.1f}% of gates near 0.5 — "
                      f"signal decoupling may be ineffective. "
                      f"Consider lower tau or more training.")

        return history_data
