"""
Helix losses — 算法改写:
  新增 label_smoothing_mae: Label Smoothing + MAE混合损失
  通过将目标值与全局均值混合, 避免过度拟合极端值
  smoothing_factor控制混合比例, 实现soft target regression
"""
import torch
import numpy as np
from .. import _dbg

_smooth_mean_ema = [0.0]  # running EMA of target mean
_smooth_initialized = [False]


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


def label_smoothing_mae(preds, labels, null_val=np.nan,
                        smoothing=0.1, ema_decay=0.99):
    """Helix特有: Label Smoothing MAE损失
    将hard target与全局均值(EMA)混合, 形成soft target:
      smooth_labels = (1 - smoothing) * labels + smoothing * global_mean
    对regression任务的效果类似分类中的label smoothing:
      - 防止模型过度自信于极端值
      - 提供更平滑的梯度landscape
      - 对噪声标签更鲁棒
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    # 计算当前batch的标签均值并用EMA更新全局均值
    with torch.no_grad():
        valid_labels = labels[mask.bool()]
        if valid_labels.numel() > 0:
            batch_mean = valid_labels.mean().item()
        else:
            batch_mean = 0.0
        if not _smooth_initialized[0]:
            _smooth_mean_ema[0] = batch_mean
            _smooth_initialized[0] = True
        else:
            _smooth_mean_ema[0] = (
                ema_decay * _smooth_mean_ema[0] +
                (1 - ema_decay) * batch_mean)
    # Label smoothing: 混合hard target和全局均值
    global_mean = _smooth_mean_ema[0]
    smooth_labels = ((1 - smoothing) * labels +
                     smoothing * global_mean)
    # MAE on smoothed labels
    loss = torch.abs(preds - smooth_labels)
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    _dbg("label_smooth.global_mean",
         f"{global_mean:.4f}", "loss")
    _dbg("label_smooth.smoothing",
         f"{smoothing:.4f}", "loss")
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
