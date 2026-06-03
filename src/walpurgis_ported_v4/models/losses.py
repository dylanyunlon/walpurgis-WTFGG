"""
Loss functions — walpurgis_ported_v4
Modifications:
  - All masked_* functions: use eps-guarded division (mask / (mean + eps))
    instead of raw mask /= mean, preventing NaN when mask is all-zero
  - metric(): prints per-call breakdown to stderr
  - masked_mae: optional gradient-scale tracking
"""
import torch
import numpy as np
import sys

_V4_DEBUG = True
_V4_EPS = 1e-8  # v4: guard against division by zero in mask normalization


def _safe_mask_norm(mask):
    """v4: eps-guarded mask normalization."""
    mask = mask.float()
    mask_mean = torch.mean(mask)
    mask = mask / (mask_mean + _V4_EPS)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = _safe_mask_norm(mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask = mask / (mask.mean() + _V4_EPS)  # v4: eps guard
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = _safe_mask_norm(mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    # v4: debug print for first few calls (controlled by module-level counter)
    if _V4_DEBUG and not hasattr(masked_mae, '_call_count'):
        masked_mae._call_count = 0
    if _V4_DEBUG:
        masked_mae._call_count += 1
        if masked_mae._call_count <= 5:
            print(f"[v4-DBG][masked_mae] call#{masked_mae._call_count} "
                  f"loss={result.item():.6f} "
                  f"pred_range=[{preds.min().item():.4f},{preds.max().item():.4f}] "
                  f"label_range=[{labels.min().item():.4f},{labels.max().item():.4f}] "
                  f"mask_ratio={mask.mean().item():.4f}",
                  file=sys.stderr)
    return result


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = _safe_mask_norm(mask)
    loss = torch.abs(preds - labels) / (labels + _V4_EPS)  # v4: eps in denominator too
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    if _V4_DEBUG:
        print(f"[v4-DBG][metric] MAE={mae:.4f} MAPE={mape:.4f} RMSE={rmse:.4f}",
              file=sys.stderr)
    return mae, mape, rmse
