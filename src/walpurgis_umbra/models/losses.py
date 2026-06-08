"""
Losses — Umbra变体
算法改动: Adaptive Huber Loss 替代 MAE
  原版: MAE (|y - ŷ|)
  Penumbra: Log-Cosh (log(cosh(e)))
  Umbra: Adaptive Huber Loss
    L(e) = 0.5 * e^2          if |e| <= δ
         = δ * (|e| - 0.5δ)   if |e| > δ
    其中 δ 是可学习参数(通过sigmoid映射到正数区间)
    自适应: 训练初期δ大→类似MSE(稳定梯度)
            训练后期δ自动收缩→类似MAE(对异常值鲁棒)

保留所有评估用的standard metrics (MAE/RMSE/MAPE)
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


def adaptive_huber_loss(preds, labels, null_val=np.nan,
                        delta=1.0):
    """Adaptive Huber Loss:
    δ控制MSE↔MAE切换阈值
    |e| <= δ: quadratic (MSE-like)
    |e| > δ:  linear (MAE-like)
    delta可以是标量或tensor(由trainer的可学习参数提供)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)

    error = preds - labels
    abs_error = torch.abs(error)
    # Huber分段
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    loss = torch.where(abs_error <= delta, quadratic, linear)

    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)

    _dbg("huber_loss.delta", f"{delta:.4f}"
         if isinstance(delta, float) else
         f"{delta.item():.4f}", "loss")
    _dbg("huber_loss.frac_quadratic",
         f"{(abs_error <= delta).float().mean().item():.4f}",
         "loss")

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
