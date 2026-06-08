"""
Losses — Penumbra变体
算法改动: Log-Cosh损失 替代 MAE
  log(cosh(x)) ≈ |x| - log(2) for large |x|
  log(cosh(x)) ≈ x^2/2 for small |x|
  比MAE更平滑(处处可微), 比MSE对异常值更鲁棒
  可学习的scale参数控制MAE/MSE切换点
"""
import torch
import numpy as np
from .. import _dbg


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(
        masked_mse(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def log_cosh_loss(preds, labels, null_val=np.nan, scale=1.0):
    """Log-Cosh损失: log(cosh(error * scale))
    scale控制MAE↔MSE行为切换"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    diff = (preds - labels) * scale
    # log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    # 数值稳定实现
    loss = diff + torch.nn.functional.softplus(
        -2.0 * diff.abs()) - 0.6931  # log(2)
    loss = loss.abs()  # 确保非负
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
