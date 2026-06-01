"""
Walpurgis v2 Estimation Gate — GEGLU-Bounded Feature Routing
================================================================
Delta vs prior:
  - softplus/(1+softplus) → *GEGLU-style* gating:
      h_fused = Linear(cat(node, time))
      gate  = σ(h_fused[:, :half])
      value = GELU(h_fused[:, half:])
      out   = gate ⊙ value ⊙ history
    This keeps the output bounded via sigmoid, but the value branch
    adds nonlinear expressivity that the prior pure-sigmoid path lacked.
  - Fusion layer uses a single wider projection (2× hidden) instead of
    separate node_fc + time_fc + fusion_fc, reducing sequential depth.
  - Gate activation statistics tracked in a ring buffer for post-mortem.

Breakpoint helpers:
    self._diag_last       # dict with last forward stats
    self.gate_trend(n)    # print last n gate means
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class EstimationGate(nn.Module):
    """GEGLU-style gating for diffusion vs inherent path routing."""

    _n = 0

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim=64):
        super().__init__()
        in_dim = node_emb_dim * 2 + time_emb_dim * 2
        # Single wide projection: first half → sigmoid gate, second half → GELU value
        self.fused_proj = nn.Linear(in_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self._debug = True
        self._diag_last = {}
        self._gate_means = deque(maxlen=500)

    def gate_trend(self, n=20):
        """Print recent gate activation means — call from pdb."""
        entries = list(self._gate_means)[-n:]
        print(f"  [EstGate] last {len(entries)} gate means:")
        for i, (step, mu, lo, hi) in enumerate(entries):
            bar_pos = int((mu - lo) / max(hi - lo, 1e-8) * 30)
            bar = "░" * bar_pos + "█" + "░" * (30 - bar_pos)
            print(f"    #{step}: μ={mu:.4f} [{lo:.4f}|{bar}|{hi:.4f}]")

    def forward(self, emb_u, emb_d, tod, dow, history):
        EstimationGate._n += 1
        B, L = history.shape[:2]

        # Node features: expand to (B, L, N, 2·node_dim)
        nc = torch.cat([emb_u, emb_d], dim=-1)
        nc = nc.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # Time features: (B, L, N, 2·time_dim)
        tc = torch.cat([tod, dow], dim=-1)

        # Fused input → single wide projection
        combined = torch.cat([nc, tc], dim=-1)
        h_wide = self.fused_proj(combined)  # [B, L, N, 2H]
        H = h_wide.shape[-1] // 2
        gate_branch = torch.sigmoid(h_wide[..., :H])
        value_branch = F.gelu(h_wide[..., H:])
        h_fused = gate_branch * value_branch  # GEGLU

        # Final scalar gate per node
        raw = self.out_proj(h_fused)  # [B, L, N, 1]
        gate = torch.sigmoid(raw)     # bounded [0, 1]

        gated = history * gate

        if self._debug:
            with torch.no_grad():
                gf = gate.detach()
                mu = gf.mean().item()
                lo = gf.min().item()
                hi = gf.max().item()
                sigma = gf.std().item()
                frac_low = ((gf < 0.1).float().mean() * 100).item()
                frac_high = ((gf > 0.9).float().mean() * 100).item()
                self._gate_means.append((EstimationGate._n, mu, lo, hi))
                self._diag_last = {
                    "step": EstimationGate._n,
                    "gate_mean": round(mu, 5),
                    "gate_std": round(sigma, 5),
                    "gate_range": (round(lo, 5), round(hi, 5)),
                    "pct_below_0.1": round(frac_low, 2),
                    "pct_above_0.9": round(frac_high, 2),
                }
            if EstimationGate._n % 200 == 1:
                d = self._diag_last
                print(
                    f"      [EstGate #{EstimationGate._n}] "
                    f"gate: μ={d['gate_mean']:.4f} σ={d['gate_std']:.4f} "
                    f"∈[{d['gate_range'][0]:.4f},{d['gate_range'][1]:.4f}] "
                    f"<0.1={d['pct_below_0.1']:.1f}% "
                    f">0.9={d['pct_above_0.9']:.1f}% "
                    f"| GEGLU active"
                )
        return gated
