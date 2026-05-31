"""
Walpurgis Estimation Gate — Temporal-Spatial Feature Routing
=============================================================
Derived from D2STGNN estimation_gate.py with ~20% restructuring.

Changes:
  1. Gate output uses tanh squashing instead of direct multiplication
     to prevent gate values from blowing up early in training
  2. Separate node/time embedding fusion paths for cleaner gradient flow
  3. Per-call debug probe with gate distribution statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class EstimationGate(nn.Module):
    """Gates history data using learned node + temporal embeddings.
    
    The gate learns which spatial-temporal positions should pass through
    to the diffusion path vs the inherent path. High gate values → more
    signal routed to diffusion; low values → inherent path dominates.
    
    Walpurgis: uses tanh activation on the fused embedding before
    element-wise multiplication. Raw linear output can have arbitrarily
    large magnitude, causing the gated signal to explode. Tanh bounds
    the gate to [-1, 1], then we shift to [0, 1] via (tanh + 1) / 2.
    """
    
    _call_count = 0
    
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim=64):
        super().__init__()
        # Node embedding projection
        self.node_fc = nn.Linear(node_emb_dim * 2, hidden_dim)
        # Temporal embedding projection (separate path for cleaner gradients)
        self.time_fc = nn.Linear(time_emb_dim * 2, hidden_dim)
        # Fusion
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim, 1)
        
        self._debug_on = True
        self._gate_stats = []  # track gate value distributions

    def forward(self, node_emb_u, node_emb_d, time_in_day, day_in_week, history):
        """Compute gate and apply to history data.
        
        Args:
            node_emb_u, node_emb_d: [N, D_node] node embeddings
            time_in_day:  [B, L, N, D_time] time-of-day embedding
            day_in_week:  [B, L, N, D_time] day-of-week embedding
            history:      [B, L, N, D_feat] raw history data
            
        Returns:
            gated_history: [B, L, N, D_feat] — history modulated by gate
        """
        EstimationGate._call_count += 1
        
        # Node path: concatenate up/down embeddings
        node_cat = torch.cat([node_emb_u, node_emb_d], dim=-1)
        node_feat = F.relu(self.node_fc(node_cat))
        # Expand to match batch/seq dims
        node_feat = node_feat.unsqueeze(0).unsqueeze(0)
        node_feat = node_feat.expand(history.shape[0], history.shape[1], -1, -1)
        
        # Time path: concatenate day/week embeddings
        time_cat = torch.cat([time_in_day, day_in_week], dim=-1)
        time_feat = F.relu(self.time_fc(time_cat))
        
        # Fuse node + time paths
        fused = torch.cat([node_feat, time_feat], dim=-1)
        fused = F.relu(self.fusion_fc(fused))
        
        # Gate: use tanh squashing → shift to [0, 1]
        # Upstream uses raw linear output which can be unbounded
        raw_gate = self.gate_fc(fused)  # [B, L, N, 1]
        gate = (torch.tanh(raw_gate) + 1.0) / 2.0  # [0, 1]
        
        gated_history = history * gate
        
        # Debug: track gate distribution
        if self._debug_on and EstimationGate._call_count % 200 == 1:
            with torch.no_grad():
                g_flat = gate.detach()
                print(f"      [EstGate #{EstimationGate._call_count}] "
                      f"gate: μ={g_flat.mean().item():.4f} σ={g_flat.std().item():.4f} "
                      f"∈[{g_flat.min().item():.4f}, {g_flat.max().item():.4f}] "
                      f"near_0(<0.1)={((g_flat < 0.1).float().mean().item()*100):.1f}% "
                      f"near_1(>0.9)={((g_flat > 0.9).float().mean().item()*100):.1f}%")
        
        return gated_history
