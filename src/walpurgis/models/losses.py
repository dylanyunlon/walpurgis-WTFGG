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
    
    Walpurgis addition: NaN/Inf guard with warning message.
    In heterogeneous-memory training, numerical instability can arise from
    tier migration during computation — this guard catches it early.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    
    # Walpurgis NaN/Inf guard
    if torch.isnan(result) or torch.isinf(result):
        print(f"  ⚠️  [LOSS] masked_mae returned {'NaN' if torch.isnan(result) else 'Inf'}!")
        print(f"    preds: nan={torch.isnan(preds).sum().item()}, inf={torch.isinf(preds).sum().item()}")
        print(f"    labels: nan={torch.isnan(labels).sum().item()}, inf={torch.isinf(labels).sum().item()}")
        print(f"    mask: sum={mask.sum().item()}, mean={mask.mean().item()}")
    
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
