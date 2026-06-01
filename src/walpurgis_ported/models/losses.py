"""
Walpurgis v2 Loss Functions — Gradient-Healthy Masked Metrics
==============================================================
Re-ported with ≈20 % algorithmic delta from the prior Walpurgis port.

Deltas:
  1. Charbonnier → *Huber–Charbonnier blend*: below δ uses smooth-L1
     (quadratic near zero), above δ uses Charbonnier sqrt.  This
     avoids the Charbonnier's sqrt-gradient-explosion when residuals
     are large, while keeping the smooth-at-zero property.
  2. Horizon weighting: geometric r^t → *logarithmic* w_t = 1 + β·ln(1+t).
     Log growth is sublinear, which matches the empirical observation
     that forecast difficulty grows fast initially then plateaus.
  3. MetricTracker gains `.trend(window)` — returns slope of the
     last `window` values via simple linear regression.  Useful in
     pdb to check whether loss is still decreasing.
  4. Anomaly sentinel now records a *stack trace* on first occurrence
     so you can pinpoint the call site from a post-mortem.

Debug cheat-sheet:
  MetricTracker.report()              # all metrics summary
  MetricTracker._registry['mae'].trend(100)  # recent slope
"""

import torch
import numpy as np
import traceback
from collections import deque


# ═══════ Metric Tracking System ═══════ #

class MetricTracker:
    """Ring-buffer tracker with trend analysis for loss / metric values.

    From pdb at any point during training:
        MetricTracker.report()                      # summary table
        MetricTracker._registry['mae'].trend(50)    # recent slope
        MetricTracker._registry['mae'].stats()      # full stats dict
    """
    _registry = {}

    def __init__(self, name, capacity=3000):
        self.name = name
        self._vals = deque(maxlen=capacity)
        self._calls = 0
        self._nans = 0
        self._infs = 0
        self._first_anomaly_trace = None
        MetricTracker._registry[name] = self

    def record(self, value):
        self._calls += 1
        if np.isnan(value):
            self._nans += 1
            if self._first_anomaly_trace is None:
                self._first_anomaly_trace = traceback.format_stack()
        elif np.isinf(value):
            self._infs += 1
            if self._first_anomaly_trace is None:
                self._first_anomaly_trace = traceback.format_stack()
        else:
            self._vals.append(value)

    def trend(self, window=100):
        """Least-squares slope over the last `window` values.

        Positive → loss increasing (bad).  Negative → decreasing (good).
        Returns None if insufficient data.
        """
        vals = list(self._vals)
        n = min(len(vals), window)
        if n < 5:
            return None
        y = np.array(vals[-n:])
        x = np.arange(n, dtype=np.float64)
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (
            n * np.dot(x, x) - x.sum() ** 2 + 1e-12
        )
        return float(slope)

    def stats(self):
        if not self._vals:
            return {"calls": self._calls, "nan": self._nans, "inf": self._infs}
        v = list(self._vals)
        return {
            "calls": self._calls,
            "last": v[-1],
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "nan_total": self._nans,
            "inf_total": self._infs,
            "trend_50": self.trend(50),
        }

    @classmethod
    def report(cls):
        """Print all tracked metrics — call from pdb or after any epoch."""
        print(f"\n{'═'*72}")
        print(f"  MetricTracker Report ({len(cls._registry)} metrics)")
        print(f"{'═'*72}")
        for name, tr in sorted(cls._registry.items()):
            s = tr.stats()
            if "last" in s:
                trd = s.get("trend_50")
                trd_s = f"trend={trd:+.6f}" if trd is not None else "trend=n/a"
                print(
                    f"  {name:18s} | n={s['calls']:6d} | "
                    f"last={s['last']:.5f}  μ={s['mean']:.5f}  σ={s['std']:.5f} | "
                    f"[{s['min']:.5f}, {s['max']:.5f}] | "
                    f"nan={s['nan_total']}  inf={s['inf_total']} | {trd_s}"
                )
            else:
                print(
                    f"  {name:18s} | n={s['calls']:6d} | NO VALID DATA | "
                    f"nan={s['nan']}  inf={s['inf']}"
                )
        print(f"{'═'*72}\n")


# Pre-register trackers
_t_mse = MetricTracker("mse")
_t_rmse = MetricTracker("rmse")
_t_mae = MetricTracker("mae")
_t_mae_loss = MetricTracker("mae_loss")
_t_mape = MetricTracker("mape")
_t_huber = MetricTracker("huber")


# ═══════ Core Loss Functions ═══════ #

def masked_mse(preds, labels, null_val=np.nan):
    """Masked Mean Squared Error."""
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    sq = (preds - labels) ** 2
    result = torch.mean(sq * mask)
    result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    _t_mse.record(result.item())
    return result


def masked_rmse(preds, labels, null_val=np.nan):
    result = torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    _t_rmse.record(result.item())
    return result


def masked_mae_loss(y_pred, y_true):
    """Simple masked MAE for training (masks exact zeros)."""
    mask = (y_true != 0).float()
    mask = mask / mask.mean()
    loss = torch.abs(y_pred - y_true) * mask
    loss[loss != loss] = 0
    result = loss.mean()
    _t_mae_loss.record(result.item())
    return result


def masked_mae(preds, labels, null_val=np.nan):
    """Primary training loss — Huber–Charbonnier blend with log-horizon weighting.

    Below the Huber threshold δ the loss is quadratic (smooth gradient at
    zero without sqrt issues).  Above δ it uses the Charbonnier sqrt which
    grows more slowly than L1, reducing sensitivity to outliers.

    Horizon weighting uses w_t = 1 + β·ln(1+t) with β=0.15.  This is
    sublinear — the first few horizons get the sharpest weight increase,
    matching empirical forecast difficulty curves.
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    diff = preds - labels
    abs_diff = diff.abs()

    # ── Huber–Charbonnier blend ──
    _delta = 1.0       # Huber threshold
    _eps = 1e-5         # Charbonnier epsilon
    quadratic = 0.5 * diff * diff                              # smooth near 0
    charbonnier = torch.sqrt(_eps + diff * diff) - np.sqrt(_eps)  # bounded growth
    loss = torch.where(abs_diff < _delta, quadratic, charbonnier)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # ── Logarithmic horizon weighting ──
    if loss.dim() >= 3 and loss.shape[1] > 1:
        T = loss.shape[1]
        beta = 0.15
        hw = torch.tensor(
            [1.0 + beta * np.log(1.0 + t) for t in range(T)],
            device=loss.device, dtype=loss.dtype,
        )
        hw = hw / hw.mean()     # normalise to preserve loss scale
        shape = [1, T] + [1] * (loss.dim() - 2)
        loss = loss * hw.view(*shape)

    result = torch.mean(loss)

    # ── Anomaly diagnostic ──
    if torch.isnan(result) or torch.isinf(result):
        kind = "NaN" if torch.isnan(result) else "Inf"
        print(
            f"\n  ⚠️  [LOSS ANOMALY] masked_mae → {kind}!\n"
            f"    preds:  shape={list(preds.shape)}, "
            f"nan={torch.isnan(preds).sum().item()}, "
            f"inf={torch.isinf(preds).sum().item()}, "
            f"∈[{preds.min().item():.4f}, {preds.max().item():.4f}]\n"
            f"    labels: shape={list(labels.shape)}, "
            f"nan={torch.isnan(labels).sum().item()}\n"
            f"    mask:   sum={mask.sum().item():.0f}, μ={mask.mean().item():.4f}\n"
            f"    diff:   μ={diff.mean().item():.5f}  σ={diff.std().item():.5f}"
        )

    _t_mae.record(result.item())
    return result


def masked_huber(preds, labels, null_val=np.nan):
    criterion = torch.nn.SmoothL1Loss()
    result = criterion(preds, labels)
    _t_huber.record(result.item())
    return result


def masked_mape(preds, labels, null_val=np.nan):
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    _t_mape.record(result.item())
    return result


def metric(pred, real):
    v_mae = masked_mae(pred, real, 0.0).item()
    v_mape = masked_mape(pred, real, 0.0).item()
    v_rmse = masked_rmse(pred, real, 0.0).item()
    return v_mae, v_mape, v_rmse


# ═══════ Convenience ═══════ #

def get_loss_summary():
    MetricTracker.report()
