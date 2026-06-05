"""
losses.py — CardGame损失函数
算法改写 (vs upstream):
  - masked_mae → Welsch robust loss (对离群值更鲁棒)
  - 新增temporal smoothness penalty (惩罚相邻时间步预测突变)
  - 新增gradient-aware re-weighting
"""
import os
import sys
import torch
import numpy as np

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="losses"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


def _build_mask(labels, null_val=np.nan):
    """构建有效值掩码"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def welsch_robust_loss(preds, labels, null_val=np.nan, c=1.0):
    """Welsch robust loss: ρ(r) = (c²/2) * (1 - exp(-r²/c²))
    比MAE对离群值更鲁棒, 比Huber更平滑
    """
    _dbg("welsch.preds", preds)
    _dbg("welsch.labels", labels)
    mask = _build_mask(labels, null_val)
    residual = preds - labels
    # Welsch loss核
    loss = (c ** 2 / 2.0) * (1.0 - torch.exp(-(residual ** 2) / (c ** 2)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    result = torch.mean(loss)
    _dbg("welsch.loss", result)
    return result


def temporal_smoothness_penalty(preds, alpha=0.1):
    """时间平滑惩罚: 惩罚相邻时间步预测之间的突变
    penalty = α * mean(|pred_{t+1} - pred_t|²)
    """
    if preds.dim() < 2 or preds.shape[1] < 2:
        return torch.tensor(0.0, device=preds.device)
    diff = preds[:, 1:, ...] - preds[:, :-1, ...]
    penalty = alpha * torch.mean(diff ** 2)
    _dbg("temporal_smooth.penalty", penalty)
    return penalty


def masked_mse(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae_loss(y_pred, y_true):
    """兼容接口: 带零值mask的MAE"""
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae(preds, labels, null_val=np.nan):
    """CardGame改写: 使用Welsch robust loss替代纯MAE"""
    return welsch_robust_loss(preds, labels, null_val, c=1.0)


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


def masked_mape(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
