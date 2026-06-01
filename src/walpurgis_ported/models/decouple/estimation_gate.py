"""
Walpurgis v2 Estimation Gate — Softplus-Bounded Feature Routing
=================================================================
Delta: tanh→[0,1] mapping → *softplus-based* gate that naturally
saturates at large values but never hard-clips at 1.  This allows
the gate to stay slightly above 1 when both paths should amplify,
which the tanh mapping can't express.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimationGate(nn.Module):
    """Gates history data for spatial-diffusion vs temporal-inherent routing.

    Gate = softplus(linear(fused)) / (1 + softplus(linear(fused)))
    This is bounded in [0, 1) but with a softer saturation than sigmoid.
    """

    _n = 0

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim=64):
        super().__init__()
        self.node_fc = nn.Linear(node_emb_dim * 2, hidden_dim)
        self.time_fc = nn.Linear(time_emb_dim * 2, hidden_dim)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_fc = nn.Linear(hidden_dim, 1)
        self._debug = True

    def forward(self, emb_u, emb_d, tod, dow, history):
        EstimationGate._n += 1

        B, L = history.shape[:2]
        # Node path
        nc = torch.cat([emb_u, emb_d], dim=-1)
        nf = F.relu(self.node_fc(nc)).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        # Time path
        tc = torch.cat([tod, dow], dim=-1)
        tf = F.relu(self.time_fc(tc))
        # Fuse
        fused = F.relu(self.fusion_fc(torch.cat([nf, tf], dim=-1)))
        raw = self.gate_fc(fused)

        # Softplus-bounded gate: sp(x)/(1+sp(x)) ∈ [0, 1)
        sp = F.softplus(raw)
        gate = sp / (1.0 + sp)

        gated = history * gate

        if self._debug and EstimationGate._n % 200 == 1:
            with torch.no_grad():
                gf = gate.detach()
                print(
                    f"      [EstGate #{EstimationGate._n}] "
                    f"gate: μ={gf.mean().item():.4f} σ={gf.std().item():.4f} "
                    f"∈[{gf.min().item():.4f},{gf.max().item():.4f}] "
                    f"<0.1={((gf<0.1).float().mean()*100).item():.1f}% "
                    f">0.9={((gf>0.9).float().mean()*100).item():.1f}%"
                )
        return gated
