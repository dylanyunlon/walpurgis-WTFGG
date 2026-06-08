"""
Rift losses — 算法改写:
  新增 logcosh_adaptive: Log-cosh和MAE的自适应混合损失
  alpha参数根据训练进度(global step)动态调节
"""
import torch
import numpy as np
from .. import _dbg

_step_counter = [0]

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

def logcosh_adaptive(preds, labels, null_val=np.nan, warmup_steps=100, decay_rate=0.005):
    """Rift特有: Log-cosh自适应混合损失
    log(cosh(x)) 在x小时≈x^2/2 (像MSE), x大时≈|x|-log(2) (像MAE)
    但比Huber loss更平滑, 处处二阶可微
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = preds - labels
    abs_res = residual.abs()
    logcosh = abs_res + torch.log1p(torch.exp(-2.0 * abs_res)) - np.log(2.0)
    _step_counter[0] += 1
    alpha = max(0.2, 1.0 / (1.0 + decay_rate * max(_step_counter[0] - warmup_steps, 0)))
    mae_part = abs_res
    combined = alpha * logcosh + (1 - alpha) * mae_part
    combined = combined * mask
    combined = torch.where(torch.isnan(combined), torch.zeros_like(combined), combined)
    _dbg("logcosh_alpha", f"{alpha:.4f}", "loss")
    return torch.mean(combined)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
