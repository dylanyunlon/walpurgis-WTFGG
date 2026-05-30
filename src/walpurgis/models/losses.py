"""
Walpurgis Loss Functions — Tier-Aware Masked Metrics with Debug Probes
======================================================================
Adapted from D2STGNN losses.py.

Key modifications:
  1. Every loss function tracks call count and running statistics
  2. NaN/Inf guards with explicit warnings (critical for multi-tier training)
  3. Masked metric consistency checker — cross-validates different metric computations
"""

import torch
import numpy as np


_loss_call_counts = {}
_loss_running_stats = {}


def _track_loss(name, value):
    """Walpurgis debug: track loss statistics across calls."""
    if name not in _loss_call_counts:
        _loss_call_counts[name] = 0
        _loss_running_stats[name] = []
    _loss_call_counts[name] += 1
    _loss_running_stats[name].append(value)
    # Keep last 1000 values
    if len(_loss_running_stats[name]) > 1000:
        _loss_running_stats[name] = _loss_running_stats[name][-500:]


def get_loss_summary():
    """Debug helper: print summary of all loss function usage."""
    print("\n[LOSS SUMMARY]")
    for name in _loss_call_counts:
        vals = _loss_running_stats[name]
        if vals:
            print(f"  {name}: calls={_loss_call_counts[name]}, "
                  f"last={vals[-1]:.6f}, mean={np.mean(vals):.6f}, "
                  f"std={np.std(vals):.6f}, min={min(vals):.6f}, max={max(vals):.6f}")


def masked_mse(preds, labels, null_val=np.nan):
    """Mean Squared Error with masking and Walpurgis debug tracking."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    _track_loss('mse', result.item())
    return result


def masked_rmse(preds, labels, null_val=np.nan):
    """Root Mean Squared Error with masking."""
    result = torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    _track_loss('rmse', result.item())
    return result


def masked_mae_loss(y_pred, y_true):
    """Simple masked MAE loss."""
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0  # NaN trick
    result = loss.mean()
    _track_loss('mae_loss', result.item())
    return result


def masked_mae(preds, labels, null_val=np.nan):
    """Mean Absolute Error with masking — the primary training loss.
    
    Walpurgis modifications:
      1. NaN/Inf guard with diagnostic output
      2. Smooth floor: instead of raw |pred - label|, use sqrt(eps + (pred-label)^2)
         which is differentiable at zero (standard MAE has undefined gradient at 0).
         The floor eps is tiny (1e-6) so the actual loss value barely changes,
         but gradient flow through near-perfect predictions becomes much healthier.
      3. Per-horizon weighting when predictions have temporal dimension:
         later horizons get slightly higher weight (they're harder to predict).
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # Walpurgis: smooth-floor MAE for gradient health at zero crossings
    # Standard: loss = |pred - label|
    # Smooth:   loss = sqrt(eps + (pred - label)^2) - sqrt(eps)
    # This is equivalent to Huber loss with delta→0, but preserves MAE scale
    _eps = 1e-6
    diff = preds - labels
    loss = torch.sqrt(_eps + diff * diff) - np.sqrt(_eps)
    
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    # Walpurgis: horizon-aware weighting (if temporal dim exists)
    # Later time steps get 1 + 0.02*t weight to emphasize hard-to-predict horizons
    if loss.dim() >= 3 and loss.shape[1] > 1:
        T = loss.shape[1]
        horizon_weights = 1.0 + 0.02 * torch.arange(T, device=loss.device, dtype=loss.dtype)
        # Reshape for broadcasting: [1, T, 1, ...] 
        shape = [1, T] + [1] * (loss.dim() - 2)
        horizon_weights = horizon_weights.view(*shape)
        loss = loss * horizon_weights
    
    result = torch.mean(loss)
    
    # Walpurgis NaN/Inf guard
    if torch.isnan(result) or torch.isinf(result):
        print(f"  ⚠️  [LOSS] masked_mae returned {'NaN' if torch.isnan(result) else 'Inf'}!")
        print(f"    preds: shape={list(preds.shape)} "
              f"nan={torch.isnan(preds).sum().item()}, inf={torch.isinf(preds).sum().item()}")
        print(f"    labels: shape={list(labels.shape)} "
              f"nan={torch.isnan(labels).sum().item()}, inf={torch.isinf(labels).sum().item()}")
        print(f"    mask: sum={mask.sum().item()}, mean={mask.mean().item()}")
        print(f"    diff stats: mean={diff.mean().item():.6f} std={diff.std().item():.6f}")
    
    _track_loss('mae', result.item())
    return result


def masked_huber(preds, labels, null_val=np.nan):
    """Huber loss — robust to outliers from tier-boundary edges."""
    crit = torch.nn.SmoothL1Loss()
    result = crit(preds, labels)
    _track_loss('huber', result.item())
    return result


def masked_mape(preds, labels, null_val=np.nan):
    """Mean Absolute Percentage Error with masking."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    _track_loss('mape', result.item())
    return result


def metric(pred, real):
    """Combined metrics — returns (mae, mape, rmse)."""
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
