"""
losses.py — v9 port
Algo delta:
  1. masked_mae → Huber-like soft-clip: |e| < δ 用 L1, |e| ≥ δ 用 δ·(log(|e|/δ)+1)
     压制极端离群点的梯度贡献 (upstream 是纯 L1)
  2. masked_mape: 分母加 floor clamp max(|y|, 1e-5) 防小值除零
  3. 新增 masked_quantile_loss(q=0.5): 非对称分位数损失,
     q>0.5 偏罚低估, q<0.5 偏罚高估
  4. metric() 返回 (mae, mape, rmse, q50_loss)
"""
import torch
import numpy as np
from walpurgis_ported_v9 import _dbg

_TAG = "losses"
_HUBER_DELTA = 10.0
_MAPE_FLOOR  = 1e-5


def _build_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    denom = torch.mean(mask)
    mask = mask / (denom + 1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    loss = (preds - labels) ** 2 * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val) + 1e-8)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask = mask / (mask.mean() + 1e-8)
    loss = torch.abs(y_pred - y_true) * mask
    loss[loss != loss] = 0
    return loss.mean()


# ── v9: Huber-like soft-clip MAE ──

def masked_mae(preds, labels, null_val=np.nan):
    """
    v9: 当 |error| < δ 时等价于普通 MAE;
    |error| ≥ δ 时用 δ·(log(|error|/δ) + 1) 做对数压缩,
    避免极端离群点主导梯度.
    """
    mask = _build_mask(labels, null_val)
    err = torch.abs(preds - labels)
    # soft-clip: 小误差 → L1, 大误差 → 对数压缩
    within = err * (err < _HUBER_DELTA).float()
    beyond = (_HUBER_DELTA * (torch.log(err / _HUBER_DELTA + 1e-8) + 1.0)) * (err >= _HUBER_DELTA).float()
    loss = (within + beyond) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    out = torch.mean(loss)
    _dbg(_TAG, f"masked_mae(huber-clip δ={_HUBER_DELTA})  loss={out.item():.6g}  "
               f"err_max={err.max().item():.4g}  pct_clipped={(err>=_HUBER_DELTA).float().mean().item()*100:.1f}%")
    return out


# ── v9: floor-clamped MAPE ──

def masked_mape(preds, labels, null_val=np.nan):
    mask = _build_mask(labels, null_val)
    # v9: clamp denominator to avoid near-zero division
    safe_labels = torch.clamp(torch.abs(labels), min=_MAPE_FLOOR)
    loss = torch.abs(preds - labels) / safe_labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    out = torch.mean(loss)
    _dbg(_TAG, f"masked_mape(floor={_MAPE_FLOOR})  mape={out.item():.6g}")
    return out


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    return crit(preds, labels)


# ── v9: 分位数损失 (新增) ──

def masked_quantile_loss(preds, labels, q=0.5, null_val=np.nan):
    """
    Pinball / quantile loss.
    q = 0.5 退化为 0.5 * MAE;  q = 0.9 重罚低估.
    在交通预测中 q > 0.5 常用于保守估计 (宁可高估不要低估).
    """
    mask = _build_mask(labels, null_val)
    err = preds - labels
    loss = torch.where(err >= 0, q * err, (q - 1.0) * err)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    out = torch.mean(loss)
    _dbg(_TAG, f"quantile_loss(q={q})  loss={out.item():.6g}")
    return out


def metric(pred, real):
    mae  = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    q50  = masked_quantile_loss(pred, real, q=0.5, null_val=0.0).item()
    return mae, mape, rmse, q50
