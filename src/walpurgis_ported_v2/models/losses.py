"""
Masked loss functions for spatial-temporal prediction.
Handles missing / zero-padded entries gracefully.
"""

import torch
import numpy as np
import sys

_DBG_LOSS = ("--debug-loss" in sys.argv) or False


def _build_mask(labels, null_val):
    """Construct a float mask that is 1 where labels are valid."""
    if np.isnan(null_val):
        valid = ~torch.isnan(labels)
    else:
        valid = (labels != null_val)
    mask = valid.float()
    mask = mask / torch.mean(mask)                       # normalize so mean(mask) = 1
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(predictions, targets, null_val=np.nan):
    """Mean squared error ignoring entries marked by *null_val*."""
    mask = _build_mask(targets, null_val)
    sq_err = (predictions - targets) ** 2
    sq_err = sq_err * mask
    sq_err = torch.where(torch.isnan(sq_err), torch.zeros_like(sq_err), sq_err)
    result = torch.mean(sq_err)
    if _DBG_LOSS:
        print(f"[DBG:loss] masked_mse={result.item():.6f}  mask_ratio={mask.mean().item():.4f}")
    return result


def masked_rmse(predictions, targets, null_val=np.nan):
    """Root mean squared error (masked)."""
    return torch.sqrt(masked_mse(predictions, targets, null_val))


def masked_mae_loss(y_pred, y_true):
    """MAE where true == 0 is treated as missing (legacy interface)."""
    active = (y_true != 0).float()
    active = active / active.mean()
    err = torch.abs(y_pred - y_true) * active
    err[err != err] = 0   # NaN → 0 trick
    return err.mean()


def masked_mae(predictions, targets, null_val=np.nan):
    """Mean absolute error ignoring entries marked by *null_val*."""
    mask = _build_mask(targets, null_val)
    abs_err = torch.abs(predictions - targets) * mask
    abs_err = torch.where(torch.isnan(abs_err), torch.zeros_like(abs_err), abs_err)
    result = torch.mean(abs_err)
    if _DBG_LOSS:
        print(f"[DBG:loss] masked_mae={result.item():.6f}  "
              f"pred_range=[{predictions.min().item():.4g},{predictions.max().item():.4g}]  "
              f"target_range=[{targets.min().item():.4g},{targets.max().item():.4g}]")
    return result


def masked_huber(predictions, targets, null_val=np.nan):
    """Smooth-L1 (Huber) loss (no explicit masking, used as a drop-in)."""
    criterion = torch.nn.SmoothL1Loss()
    return criterion(predictions, targets)


def masked_mape(predictions, targets, null_val=np.nan):
    """Mean absolute percentage error (masked)."""
    mask = _build_mask(targets, null_val)
    pct_err = torch.abs(predictions - targets) / targets
    pct_err = pct_err * mask
    pct_err = torch.where(torch.isnan(pct_err), torch.zeros_like(pct_err), pct_err)
    return torch.mean(pct_err)


def metric(pred, real):
    """Convenience: return (MAE, MAPE, RMSE) in one call."""
    mae_val  = masked_mae(pred, real, 0.0).item()
    mape_val = masked_mape(pred, real, 0.0).item()
    rmse_val = masked_rmse(pred, real, 0.0).item()
    if _DBG_LOSS:
        print(f"[DBG:loss] metric  MAE={mae_val:.4f}  MAPE={mape_val:.4f}  RMSE={rmse_val:.4f}")
    return mae_val, mape_val, rmse_val
