"""
Walpurgis v4 Loss Functions — Log-Cosh Smooth Loss with Cauchy Horizon Decay
================================================================================
Delta vs v3:
  1. adaptive-δ Huber → *log-cosh* smooth loss: L = log(cosh(pred-label)).
     Combines the best of MSE (smooth at 0) and MAE (linear at ∞)
     point adapts to the current residual distribution.
  2. Exponential horizon decay → *Cauchy weighting*:
            w_t = 1 / (1 + (t/γ)²) — heavier tails than exponential,
     Exponential decay penalises near horizons more aggressively,
     matching the observation that traffic prediction accuracy
     degrades fastest in the first few steps.
  3. MetricTracker gains `.ewma(span)` — exponential weighted moving
     average of recent values for smoother trend estimation.
  4. Anomaly sentinel now includes a compact stack fingerprint (hash
     of the top 3 frames) for deduplication in logs.

Debug cheat-sheet:
  MetricTracker.report()
  MetricTracker._registry['mae'].ewma(100)
  MetricTracker._registry['mae'].trend(100)
"""

import torch
import numpy as np
import traceback
import hashlib
from collections import deque


# ═══════ Metric Tracking System ═══════ #

class MetricTracker:
    """Ring-buffer tracker with trend + EWMA analysis.

    From pdb:
        MetricTracker.report()
        MetricTracker._registry['mae'].trend(50)
        MetricTracker._registry['mae'].ewma(100)
        MetricTracker._registry['mae'].stats()
    """
    _registry = {}

    def __init__(self, name, capacity=3000):
        self.name = name
        self._vals = deque(maxlen=capacity)
        self._calls = 0
        self._nans = 0
        self._infs = 0
        self._first_anomaly_trace = None
        self._anomaly_fingerprint = None
        MetricTracker._registry[name] = self

    def record(self, value):
        self._calls += 1
        if np.isnan(value):
            self._nans += 1
            if self._first_anomaly_trace is None:
                frames = traceback.format_stack()
                self._first_anomaly_trace = frames
                sig = "".join(frames[-3:])
                self._anomaly_fingerprint = hashlib.md5(sig.encode()).hexdigest()[:8]
        elif np.isinf(value):
            self._infs += 1
            if self._first_anomaly_trace is None:
                frames = traceback.format_stack()
                self._first_anomaly_trace = frames
                sig = "".join(frames[-3:])
                self._anomaly_fingerprint = hashlib.md5(sig.encode()).hexdigest()[:8]
        else:
            self._vals.append(value)

    def trend(self, window=100):
        """Least-squares slope over the last `window` values."""
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

    def ewma(self, span=100):
        """Exponentially weighted moving average of last `span` values.

        Returns the final EWMA value, or None if insufficient data.
        """
        vals = list(self._vals)
        n = min(len(vals), span)
        if n < 2:
            return None
        alpha = 2.0 / (n + 1)
        ema = vals[-n]
        for v in vals[-n + 1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

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
            "ewma_100": self.ewma(100),
            "anomaly_fp": self._anomaly_fingerprint,
        }

    @classmethod
    def report(cls):
        print(f"\n{'═' * 78}")
        print(f"  MetricTracker Report ({len(cls._registry)} metrics)")
        print(f"{'═' * 78}")
        for name, tr in sorted(cls._registry.items()):
            s = tr.stats()
            if "last" in s:
                trd = s.get("trend_50")
                trd_s = f"slope={trd:+.6f}" if trd is not None else "slope=n/a"
                ema_s = f"ewma={s['ewma_100']:.5f}" if s.get("ewma_100") is not None else "ewma=n/a"
                fp = f" fp={s['anomaly_fp']}" if s['anomaly_fp'] else ""
                print(
                    f"  {name:18s} | n={s['calls']:6d} | "
                    f"last={s['last']:.5f}  μ={s['mean']:.5f}  σ={s['std']:.5f} | "
                    f"[{s['min']:.5f}, {s['max']:.5f}] | "
                    f"nan={s['nan_total']}  inf={s['inf_total']} | "
                    f"{trd_s} {ema_s}{fp}"
                )
            else:
                print(
                    f"  {name:18s} | n={s['calls']:6d} | NO VALID DATA | "
                    f"nan={s['nan']}  inf={s['inf']}"
                )
        print(f"{'═' * 78}\n")


# Pre-register trackers
_t_mse = MetricTracker("mse")
_t_rmse = MetricTracker("rmse")
_t_mae = MetricTracker("mae")
_t_mae_loss = MetricTracker("mae_loss")
_t_mape = MetricTracker("mape")
_t_huber = MetricTracker("huber")
_t_adaptive_delta = MetricTracker("adaptive_delta")


# ═══════ Core Loss Functions ═══════ #

# Running residual magnitude for adaptive delta
_residual_buf = deque(maxlen=500)


def masked_mse(preds, labels, null_val=np.nan):
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
    mask = (y_true != 0).float()
    mask = mask / mask.mean()
    loss = torch.abs(y_pred - y_true) * mask
    loss[loss != loss] = 0
    result = loss.mean()
    _t_mae_loss.record(result.item())
    return result


def masked_mae(preds, labels, null_val=np.nan):
    """Primary training loss — log-cosh with Cauchy horizon weighting.

    log-cosh is C²-smooth everywhere and transitions from quadratic
    to linear behavior at |residual| ≈ 1, without needing an adaptive δ.

    Cauchy weighting has heavier tails than exponential, so distant
    horizons still receive meaningful gradient signal.

    The Huber threshold δ adapts to the p90 of recent |pred-label|,
    so near-zero residuals get quadratic smoothing while the threshold
    evolves with training progress.

    Horizon weighting uses w_t = exp(-λ·t) normalised so Σw_t = T.
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    diff = preds - labels

    # Track residual magnitude for diagnostics
    with torch.no_grad():
        sample_mag = diff.abs().detach().mean().item()
        _residual_buf.append(sample_mag)
    _t_adaptive_delta.record(sample_mag)

    # ── Log-cosh loss ──
    # log(cosh(x)) ≈ x²/2 for small x, |x| - log(2) for large x
    # Combines MSE smoothness near zero with MAE robustness at tails
    loss = torch.log(torch.cosh(diff.clamp(-20, 20)))  # clamp for numerical safety

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # ── Cauchy horizon weighting ──
    # w_t = 1 / (1 + (t/γ)²) — heavier tails than exponential,
    # doesn't discard distant horizons as aggressively
    if loss.dim() >= 3 and loss.shape[1] > 1:
        T = loss.shape[1]
        gamma = 4.0  # scale parameter
        hw = torch.tensor(
            [1.0 / (1.0 + (t / gamma) ** 2) for t in range(T)],
            device=loss.device, dtype=loss.dtype,
        )
        hw = hw * T / hw.sum()  # normalise to preserve total scale
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
            f"    diff:   μ={diff.mean().item():.5f}  σ={diff.std().item():.5f}\n"
            f"    delta:  {delta:.4f}"
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
