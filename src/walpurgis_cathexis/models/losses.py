"""
Cathexis Losses — 算法改写 #9
upstream: masked_mae (symmetric absolute error)
cathexis: Asymmetric Winsorized loss — clips extreme errors, asymmetric penalty
"""
import torch
import numpy as np

def winsorized_mae(preds, labels, null_val=0.0, lo_pct=0.05, hi_pct=0.95, asymmetry=1.2):
    """Asymmetric Winsorized MAE: clip extreme errors + asymmetric penalty for under/over prediction"""
    if null_val == 0.0:
        mask = (labels != 0).float()
    elif np.isnan(null_val):
        mask = ~torch.isnan(labels)
        mask = mask.float()
    else:
        mask = (labels != null_val).float()
    mask /= mask.mean().clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    error = preds - labels
    abs_error = error.abs()
    # Winsorize: clip extreme errors
    with torch.no_grad():
        flat = abs_error[mask > 0.5].detach()
        if flat.numel() > 10:
            lo_val = torch.quantile(flat, lo_pct)
            hi_val = torch.quantile(flat, hi_pct)
        else:
            lo_val = torch.tensor(0.0, device=preds.device)
            hi_val = torch.tensor(1e6, device=preds.device)
    clipped = abs_error.clamp(min=lo_val, max=hi_val)
    # Asymmetric penalty: penalize under-prediction more
    weight = torch.where(error < 0, torch.tensor(asymmetry, device=error.device),
                                     torch.tensor(1.0, device=error.device))
    loss = clipped * weight * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
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
    return torch.mean(loss)


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
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
