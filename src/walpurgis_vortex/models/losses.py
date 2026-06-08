"""
Vortex losses — 算法改写:
  新增 huber_mae_adaptive: Huber和MAE的自适应混合
  delta参数根据预测误差的running统计自动调节
  小误差时近似MSE(更强梯度), 大误差时近似MAE(鲁棒)
"""
import torch
import numpy as np
from .. import _dbg

_running_delta = [1.0]  # mutable container for running EMA


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
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
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


def huber_mae_adaptive(preds, labels, null_val=np.nan,
                       ema_decay=0.95):
    """Vortex特有: Huber-MAE自适应混合损失
    delta根据误差分布的EMA自动调节:
      - 小delta: 小误差附近近似MSE, 梯度更强
      - 大delta: 退化为MAE, 对outlier鲁棒
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = (preds - labels).abs()
    # 自适应delta: 用误差中位数的EMA
    with torch.no_grad():
        current_median = torch.median(
            residual[mask.bool()]).item()
        _running_delta[0] = (
            ema_decay * _running_delta[0] +
            (1 - ema_decay) * current_median)
    delta = max(_running_delta[0], 0.01)
    # Huber loss with adaptive delta
    quadratic = torch.clamp(residual, max=delta)
    linear = residual - quadratic
    loss = 0.5 * quadratic ** 2 / delta + linear
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    _dbg("huber_delta", f"{delta:.4f}", "loss")
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
