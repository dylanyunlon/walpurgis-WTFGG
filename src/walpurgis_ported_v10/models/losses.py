import torch
import numpy as np
from walpurgis_ported_v10 import _dbg

_TAG = "loss"

# ---- 改动1: smooth Huber + log-cosh 混合 ----
# upstream 用纯 masked_mae; 这里改成 δ=5 的 Huber 和 log-cosh 的加权和
# Huber 在大残差处梯度更稳, log-cosh 在零附近更光滑
_HUBER_DELTA = 5.0
_LOGCOSH_WEIGHT = 0.3   # 30% log-cosh + 70% huber


def _build_mask(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    denom = torch.mean(mask)
    denom = torch.clamp(denom, min=1e-8)  # 改动: clamp防零除, upstream直接除
    mask = mask / denom
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_huber_logcosh(preds, labels, null_val=np.nan):
    """改动: upstream是纯MAE, 这里替换为Huber+log-cosh混合.
    Huber(δ=5): 小误差用L2, 大误差用L1, 拐点在δ
    log-cosh: log(cosh(x)) ≈ x²/2 (小x), |x|-log2 (大x), 处处二阶可微
    """
    mask = _build_mask(preds, labels, null_val)
    diff = preds - labels

    # Huber 部分
    abs_diff = torch.abs(diff)
    huber = torch.where(
        abs_diff <= _HUBER_DELTA,
        0.5 * diff * diff,
        _HUBER_DELTA * (abs_diff - 0.5 * _HUBER_DELTA)
    )

    # log-cosh 部分: log(cosh(x)), 数值稳定写法
    logcosh = diff + torch.nn.functional.softplus(-2.0 * diff) - np.log(2.0)

    combined = (1.0 - _LOGCOSH_WEIGHT) * huber + _LOGCOSH_WEIGHT * logcosh
    combined = combined * mask
    combined = torch.where(torch.isnan(combined), torch.zeros_like(combined), combined)

    result = torch.mean(combined)
    _dbg(_TAG, "huber_logcosh", loss=result, abs_diff_mean=abs_diff.mean())
    return result


def masked_mse(preds, labels, null_val=np.nan):
    mask = _build_mask(preds, labels, null_val)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    out = torch.mean(loss)
    _dbg(_TAG, "mse", loss=out)
    return out


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=np.nan):
    """保留接口兼容, 但内部走 huber+logcosh."""
    return masked_huber_logcosh(preds, labels, null_val)


# ---- 改动2: MAPE floor clamp 5e-6 ----
# upstream 直接 abs(pred-real)/real, real=0时爆炸
# 这里加 floor clamp, 比 upstream 更鲁棒
_MAPE_FLOOR = 5e-6


def masked_mape(preds, labels, null_val=np.nan):
    mask = _build_mask(preds, labels, null_val)
    # 改动: 分母 clamp 到 floor, upstream 无此保护
    safe_labels = torch.clamp(torch.abs(labels), min=_MAPE_FLOOR)
    loss = torch.abs(preds - labels) / safe_labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    out = torch.mean(loss)
    _dbg(_TAG, "mape", loss=out)
    return out


# ---- 改动3: 新增 quantile loss ----
# upstream 完全没有 quantile loss; 这里新增, 可用于不确定性估计
def quantile_loss(preds, labels, tau=0.5, null_val=np.nan):
    """Pinball / quantile loss at quantile τ.
    τ=0.5 退化为 scaled MAE.
    """
    mask = _build_mask(preds, labels, null_val)
    diff = labels - preds
    ql = torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    ql = ql * mask
    ql = torch.where(torch.isnan(ql), torch.zeros_like(ql), ql)
    out = torch.mean(ql)
    _dbg(_TAG, f"quantile(tau={tau})", loss=out)
    return out


# ---- 改动4: metric 返回4元组 ----
def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    huber = masked_huber_logcosh(pred, real, 0.0).item()
    _dbg(_TAG, "metric", mae=torch.tensor(mae), mape=torch.tensor(mape),
         rmse=torch.tensor(rmse), huber=torch.tensor(huber))
    return mae, mape, rmse, huber
