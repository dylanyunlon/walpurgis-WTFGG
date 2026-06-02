"""
Walpurgis v2 Mask — Percentile-Based Adaptive Sparse Gating
===============================================================
Delta vs prior:
  - Per-row mean threshold → *percentile-based* threshold:
      thresh_row = quantile(row, p)  where p = sigmoid(α)
    Using a quantile makes the threshold robust to skewed similarity
    distributions (common in traffic graphs with hub nodes).
  - Sharpness uses LeakyHardtanh instead of sigmoid for the mask
    transition — sharper cutoff, but with gradient everywhere.
  - Tracks per-forward sparsity in a ring buffer for trend analysis.

Breakpoint helpers:
    self.sparsity_trend(20)    # print last 20 density readings
    self._diag_last            # dict with last forward diagnostics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class Mask(nn.Module):
    """Percentile-based adaptive sparse gating."""

    _n = 0

    def __init__(self, **kw):
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid → quantile level
        self._slope = nn.Parameter(torch.tensor(10.0))  # leaky transition steepness
        self._hyst_margin = 0.05  # v4: hysteresis band
        self._last_thresh = None
        self._debug = True
        self._density_log = deque(maxlen=500)
        self._diag_last = {}

    def sparsity_trend(self, n=20):
        """Print recent density readings — call from pdb."""
        entries = list(self._density_log)[-n:]
        if not entries:
            print("  [Mask] no density data yet")
            return
        print(f"  [Mask] last {len(entries)} density readings:")
        for step, dens, pctl in entries:
            bar_len = int(dens * 50)
            bar = "▓" * bar_len + "░" * (50 - bar_len)
            print(f"    #{step}: {bar} {dens:.4f} (p={pctl:.2f})")

    def forward(self, dist_mx):
        Mask._n += 1
        quantile_level = torch.sigmoid(self._alpha)  # ∈ (0, 1)
        steepness = F.softplus(self._slope) + 1.0

        # Per-row percentile threshold
        # Use quantile along last dim for each row
        B = dist_mx.shape[0]
        q_level = quantile_level.item()
        new_thresh = torch.quantile(dist_mx.float(), q_level, dim=-1, keepdim=True)
        # v4: Schmitt hysteresis — stabilise threshold near decision boundary
        if self._last_thresh is not None and self._last_thresh.shape == new_thresh.shape:
            upper = self._last_thresh + self._hyst_margin
            lower = self._last_thresh - self._hyst_margin
            thresh = torch.where(new_thresh > upper, new_thresh,
                      torch.where(new_thresh < lower, new_thresh, self._last_thresh))
        else:
            thresh = new_thresh
        self._last_thresh = thresh.detach()

        # Leaky hard-tanh mask: smooth transition with guaranteed gradient
        diff = steepness * (dist_mx - thresh)
        soft_mask = 0.5 * (1.0 + torch.tanh(diff))  # ∈ (0, 1), smooth
        masked = dist_mx * soft_mask

        # Diagnostics
        if self._debug:
            with torch.no_grad():
                dens = (masked.abs() > 1e-6).float().mean().item()
                self._density_log.append((Mask._n, dens, q_level))
                self._diag_last = {
                    "step": Mask._n,
                    "quantile_level": round(q_level, 4),
                    "steepness": round(steepness.item(), 2),
                    "density": round(dens, 4),
                    "thresh_mean": round(thresh.mean().item(), 5),
                }
            if Mask._n % 200 == 1:
                d = self._diag_last
                print(
                    f"        [Mask #{Mask._n}] p={d['quantile_level']:.4f} "
                    f"steep={d['steepness']:.2f} density={d['density']:.4f} "
                    f"thresh_μ={d['thresh_mean']:.5f}"
                )
        return masked
