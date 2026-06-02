"""
Masked loss functions (MAE, MSE, RMSE, MAPE, Huber) and composite metric.
Ported with debug probes for loss landscape inspection.
"""
import sys
import torch
import numpy as np

_DBG = ("--debug-loss" in sys.argv)


def _build_mask(labels, null_val):
    """Return a float mask that is 1 where labels are valid."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(pred, target, null_val=np.nan):
    mask = _build_mask(target, null_val)
    sq_err = (pred - target) ** 2
    sq_err = sq_err * mask
    sq_err = torch.where(torch.isnan(sq_err), torch.zeros_like(sq_err), sq_err)
    out = torch.mean(sq_err)
    if _DBG:
        print(f"[DBG:loss] masked_mse={out.item():.6f}  "
              f"mask_frac={mask.mean().item():.4f}")
    return out


def masked_rmse(pred, target, null_val=np.nan):
    return torch.sqrt(masked_mse(pred, target, null_val))


def masked_mae_loss(y_pred, y_true):
    """Variant that treats 0 as the null sentinel (no null_val arg)."""
    mask = (y_true != 0).float()
    mask = mask / mask.mean()
    err = torch.abs(y_pred - y_true) * mask
    err[err != err] = 0         # nan -> 0
    return err.mean()


def masked_mae(pred, target, null_val=np.nan):
    mask = _build_mask(target, null_val)
    abs_err = torch.abs(pred - target) * mask
    abs_err = torch.where(torch.isnan(abs_err), torch.zeros_like(abs_err), abs_err)
    out = torch.mean(abs_err)
    if _DBG:
        print(f"[DBG:loss] masked_mae={out.item():.6f}")
    return out


def masked_huber(pred, target, null_val=np.nan):
    return torch.nn.SmoothL1Loss()(pred, target)


def masked_mape(pred, target, null_val=np.nan):
    mask = _build_mask(target, null_val)
    pct = torch.abs(pred - target) / target
    pct = pct * mask
    pct = torch.where(torch.isnan(pct), torch.zeros_like(pct), pct)
    out = torch.mean(pct)
    if _DBG:
        print(f"[DBG:loss] masked_mape={out.item():.6f}")
    return out


def metric(pred, real):
    """Convenience: return (mae, mape, rmse) as floats."""
    mae  = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    if _DBG:
        print(f"[DBG:loss] metric  MAE={mae:.4f}  MAPE={mape:.4f}  RMSE={rmse:.4f}")
    return mae, mape, rmse
