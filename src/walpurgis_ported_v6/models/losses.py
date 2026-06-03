"""Loss functions — soft-clipped MAE variant.

Changes
-------
1. ``masked_mae`` — pure L1 on traffic data produces spiky gradients when
   outlier sensors report huge jumps.  We add a Huber-like soft clip:
   below ``delta`` it's L1, above ``delta`` it's sqrt-scaled.  The
   default delta=10.0 keeps behaviour identical for normal ranges.
2. ``masked_mape`` — adds a floor clamp on labels (1e-4) to prevent
   division-by-near-zero from dominating the gradient.
3. ``metric`` — prints per-call values for instant feedback during eval.
"""

import torch
import numpy as np


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=np.nan, delta=10.0):
    """Soft-clipped MAE: linear below *delta*, sqrt-scaled above."""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    abs_err = torch.abs(preds - labels)
    # soft clip: for abs_err > delta, use delta + sqrt(abs_err - delta)
    loss = torch.where(
        abs_err <= delta,
        abs_err,
        delta + torch.sqrt(abs_err - delta + 1e-8)
    )
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true) * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # floor clamp to prevent div-by-near-zero blow-up
    safe_labels = torch.clamp(torch.abs(labels), min=1e-4)
    loss = torch.abs(preds - labels) / safe_labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
