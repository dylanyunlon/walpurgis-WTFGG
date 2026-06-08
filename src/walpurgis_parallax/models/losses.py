"""
Losses — Parallax变体 (M054)
算法改动: Cauchy Loss 替代 Log-Cosh / MAE
  cauchy_loss(x) = log(1 + (x/scale)^2)
  - 比MSE对异常值鲁棒(重尾分布)
  - 比MAE更平滑(处处可微)
  - 比Log-Cosh对大偏差更宽容(对数增长而非线性)
  - scale参数控制"正常偏差"范围: 小scale=对偏差敏感, 大scale=宽容
  - 当scale→∞时退化为MSE, 当scale→0时退化为0-1损失

  与Penumbra的Log-Cosh对比:
    Log-Cosh: 小误差≈MSE, 大误差≈MAE (线性增长)
    Cauchy: 小误差≈MSE, 大误差≈log增长 (更慢, 更鲁棒)
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


def cauchy_loss(preds, labels, null_val=np.nan, scale=1.0):
    """Cauchy损失: log(1 + ((pred-label)/scale)^2)
    
    重尾鲁棒: 对大偏差的惩罚增长为log而非线性
    scale控制正常范围: 小=敏感, 大=宽容
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    # Cauchy核心: log(1 + (diff/scale)^2)
    diff = (preds - labels) / max(scale, 1e-6)
    loss = torch.log1p(diff ** 2)
    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)

    _dbg("cauchy_loss.diff_abs_mean",
         f"{diff.abs().mean().item():.6f}", "loss")
    _dbg("cauchy_loss.raw_mean",
         f"{loss.mean().item():.6f}", "loss")
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
