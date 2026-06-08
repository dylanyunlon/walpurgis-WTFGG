"""Meridian losses — annealed focal regression loss.
Changes vs upstream:
  - Focal regression loss: up-weights hard samples by (1-exp(-|err|))^gamma
  - Annealing: gamma decreases over training (hard→easy focus shift)
  - All original metrics preserved for evaluation
"""
import torch
import numpy as np
import sys, os

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


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


def focal_mae(preds, labels, null_val=np.nan, gamma=2.0, anneal_factor=1.0):
    """Focal regression loss: focuses on hard-to-predict samples.
    focal_weight = (1 - exp(-|error|))^(gamma * anneal_factor)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    abs_error = torch.abs(preds - labels)
    focal_weight = (1.0 - torch.exp(-abs_error)).pow(gamma * anneal_factor)
    loss = abs_error * focal_weight * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if _DBG:
        fw_mean = focal_weight.detach().mean().item()
        print(f"[MER:focal_loss] gamma={gamma:.2f} anneal={anneal_factor:.3f} "
              f"fw_mean={fw_mean:.4f} loss={loss.mean().item():.4f}", file=sys.stderr)
    return torch.mean(loss)


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
