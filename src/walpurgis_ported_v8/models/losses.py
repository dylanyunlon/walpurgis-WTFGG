import torch
import numpy as np
import sys

_DBG = ("--dbg" in sys.argv)


def _dp(tag, msg):
    if _DBG:
        print(f"[DBG][losses][{tag}] {msg}", flush=True)


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
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan, huber_delta=1.0):
    """算法改动: Huber-MAE 混合
    原版: 纯 L1 (|pred - label|)
    改为: 当 |error| < huber_delta 时用 0.5*error^2/delta (二次项, 梯度更平滑),
          当 |error| >= huber_delta 时用 |error| - 0.5*delta (线性项, 和原版一致)
    效果: 小误差区间梯度不会跳变, 大误差区间保持 MAE 的鲁棒性
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    error = preds - labels
    abs_error = torch.abs(error)
    # Huber-MAE: smooth L1 transition
    quadratic = torch.clamp(abs_error, max=huber_delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic.pow(2) / huber_delta + linear

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)

    if _DBG:
        _dp("masked_mae",
            f"mean_abs_err={abs_error.mean().item():.5f}  "
            f"loss={result.item():.5f}  "
            f"pct_quadratic={((abs_error < huber_delta).float().mean().item()*100):.1f}%")
    return result


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
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
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
