"""
Corona losses — 算法改写:
  upstream: masked_mae as primary loss
  corona: quantile loss — 多分位数回归损失, 对称地惩罚过低和过高预测,
          比MAE更robust且能输出预测区间
"""
import torch
import numpy as np


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


def quantile_loss(preds, labels, quantiles=None, weights=None, null_val=0):
    """Corona特有: 分位数损失 — 对不同分位数的偏差做非对称加权"""
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    if weights is None:
        weights = [0.15, 0.2, 0.3, 0.2, 0.15]

    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    errors = preds - labels
    total_loss = torch.zeros(1, device=preds.device)
    for q, w in zip(quantiles, weights):
        # 分位数回归: q * max(e, 0) + (1-q) * max(-e, 0)
        q_loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
        q_loss = q_loss * mask
        q_loss = torch.where(torch.isnan(q_loss), torch.zeros_like(q_loss), q_loss)
        total_loss = total_loss + w * torch.mean(q_loss)
    return total_loss.squeeze()


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
