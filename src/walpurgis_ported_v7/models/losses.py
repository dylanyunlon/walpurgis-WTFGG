import torch
import numpy as np
import sys

# ---- runtime debug helper ------------------------------------------------- #
_DBG_LOSS = ("--dbg-loss" in sys.argv)

def _loss_snapshot(tag, tensor):
    """Print a compact statistical snapshot of any loss-related tensor."""
    if not _DBG_LOSS:
        return
    with torch.no_grad():
        flat = tensor.detach().float().flatten()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            print(f"[DBG-LOSS][{tag}] ALL non-finite! shape={list(tensor.shape)}")
            return
        print(f"[DBG-LOSS][{tag}] shape={list(tensor.shape)}  "
              f"min={finite.min().item():.6f}  max={finite.max().item():.6f}  "
              f"mean={finite.mean().item():.6f}  std={finite.std().item():.6f}  "
              f"nan%={100.0*(flat.isnan().sum().item()/flat.numel()):.2f}  "
              f"inf%={100.0*(flat.isinf().sum().item()/flat.numel()):.2f}")


# ---- core loss functions -------------------------------------------------- #

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
    _loss_snapshot("mse_raw", loss)
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
    """
    算法改动: 用 smooth-L1 (Huber, delta=1.0) 替代纯 L1,
    保持 mask 逻辑不变。Huber 在残差 < delta 时是 0.5*x^2,
    大于 delta 时是 delta*(|x|-0.5*delta), 对离群值更鲁棒。
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    delta = 1.0
    residual = preds - labels
    abs_res  = torch.abs(residual)
    # Huber 核
    loss = torch.where(abs_res < delta,
                       0.5 * residual ** 2,
                       delta * (abs_res - 0.5 * delta))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _loss_snapshot("huber_mae", loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # 算法改动: 加 clamp 避免除零爆炸
    safe_labels = labels.clone()
    safe_labels[safe_labels.abs() < 1e-7] = 1e-7
    loss = torch.abs(preds - labels) / safe_labels.abs()
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _loss_snapshot("mape", loss)
    return torch.mean(loss)


def metric(pred, real):
    mae  = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
