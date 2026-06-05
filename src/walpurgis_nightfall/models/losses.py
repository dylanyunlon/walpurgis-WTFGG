"""
losses.py — Nightfall变体
算法改写:
  1. masked_mae → Charbonnier loss sqrt(|pred-label|^2 + eps) (平滑L1近似)
  2. 新增 temporal_consistency_penalty (惩罚pred比real更粗糙的时序跳变)
  3. masked_mse加权: 距离越远的horizon权重越高 (远期预测更重要)
"""
import torch
import numpy as np
from .. import _dbg

_CHARB_EPS = 1e-6  # Charbonnier平滑常数


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
    """Charbonnier-smoothed MAE: 比L1更平滑, 梯度在0处不跳跃"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # Charbonnier: sqrt((pred-label)^2 + eps) ≈ |pred-label| 但0处平滑
    diff_sq = (preds - labels) ** 2
    loss = torch.sqrt(diff_sq + _CHARB_EPS)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _dbg("loss.charb_mae", loss, "loss")
    return torch.mean(loss)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.sqrt((y_pred - y_true) ** 2 + _CHARB_EPS)
    loss = loss * mask
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
    loss = torch.abs(preds - labels) / (labels.abs() + 1e-8)  # eps防止除零
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def temporal_consistency_penalty(preds, labels, alpha=0.05):
    """惩罚预测序列比真实序列更粗糙的时序跳变
    只罚|pred_diff| > |real_diff|的部分 (单侧)"""
    if preds.dim() < 2 or preds.shape[1] < 2:
        return torch.tensor(0.0, device=preds.device)
    pred_diff = torch.diff(preds, dim=1)
    real_diff = torch.diff(labels, dim=1)
    excess = torch.relu(pred_diff.abs() - real_diff.abs())
    penalty = alpha * excess.mean()
    _dbg("loss.temporal_penalty", penalty, "loss")
    return penalty


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
