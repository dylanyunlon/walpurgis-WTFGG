"""
Losses — Perihelion变体
算法改动: Linex非对称损失 替代 MAE
  Linex(e) = b * (exp(a*e) - a*e - 1)
  a > 0: 过预测(正误差)惩罚更大(指数增长)
  a < 0: 欠预测(负误差)惩罚更大
  b: 整体缩放系数
  当 a→0 时退化为 MSE/2
  相比Log-Cosh: 提供明确的方向性非对称惩罚
  相比MAE: 处处可微, 方向可控
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


def linex_loss(preds, labels, null_val=np.nan,
               a=0.5, b=1.0):
    """Linex非对称损失: b * (exp(a*error) - a*error - 1)
    a > 0: 过预测惩罚更大 (traffic场景常需)
    a < 0: 欠预测惩罚更大
    b: 缩放系数
    数值稳定: clamp a*error 防止exp溢出
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
    ae = a * error
    # 数值稳定: clamp防止exp溢出
    ae_clamped = torch.clamp(ae, min=-10.0, max=10.0)
    loss = b * (torch.exp(ae_clamped) - ae - 1.0)
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
