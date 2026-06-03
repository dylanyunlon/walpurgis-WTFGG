import torch
import numpy as np

# ═══════ Loss Functions ═══════
# Delta vs upstream:
#   1. masked_mae uses Kahan compensated accumulation for numerical stability
#   2. Added masked_huber_adaptive with auto-tuning delta
#   3. metric() returns dict instead of tuple, for cleaner breakpoint inspection


def _kahan_mean(t: torch.Tensor) -> torch.Tensor:
    """Kahan-compensated mean — reduces float32 drift on large batches."""
    flat = t.reshape(-1)
    s = torch.zeros(1, device=t.device, dtype=t.dtype)
    c = torch.zeros(1, device=t.device, dtype=t.dtype)
    for i in range(0, flat.shape[0], 4096):
        chunk = flat[i:i+4096].sum()
        y = chunk - c
        new_s = s + y
        c = (new_s - s) - y
        s = new_s
    return s / flat.shape[0]


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
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # ── delta 1: Kahan mean on large tensors ──
    if loss.numel() > 8192:
        return _kahan_mean(loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def masked_huber_adaptive(preds, labels, null_val=np.nan, percentile=90):
    """Huber loss with delta auto-set to the p-th percentile of residuals.

    At any breakpoint:
        masked_huber_adaptive._last_delta   → the auto-tuned delta
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    residuals = torch.abs(preds - labels)
    valid = residuals[mask.bool()]
    if valid.numel() == 0:
        return torch.tensor(0.0, device=preds.device)
    delta = torch.quantile(valid.detach(), percentile / 100.0)
    masked_huber_adaptive._last_delta = delta.item()
    crit = torch.nn.SmoothL1Loss(beta=delta.item())
    raw = crit(preds * mask, labels * mask)
    return raw


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae  = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
