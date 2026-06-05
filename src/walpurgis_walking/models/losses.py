import torch
import numpy as np
from walpurgis_walking import _dbg

_TAG = "loss"

# ---- 改动1: Huber(δ=5) + log-cosh(30%) 混合 ----
# upstream 纯 masked_mae, 梯度在零点不连续
# Huber 在大残差处限制梯度, log-cosh 在零附近二阶光滑
_HUBER_DELTA = 5.0
_LC_W = 0.3


def _mask(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        m = ~torch.isnan(labels)
    else:
        m = (labels != null_val)
    m = m.float()
    d = torch.clamp(m.mean(), min=1e-8)          # 改动: clamp 防零除
    m = m / d
    m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
    return m


def masked_huber_logcosh(preds, labels, null_val=np.nan):
    """Huber + log-cosh 混合损失, upstream 是纯 L1."""
    mk = _mask(preds, labels, null_val)
    diff = preds - labels
    ad = torch.abs(diff)
    # Huber
    hub = torch.where(ad <= _HUBER_DELTA,
                      0.5 * diff * diff,
                      _HUBER_DELTA * (ad - 0.5 * _HUBER_DELTA))
    # log-cosh (数值稳定)
    lc = diff + torch.nn.functional.softplus(-2.0 * diff) - np.log(2.0)
    c = (1.0 - _LC_W) * hub + _LC_W * lc
    c = c * mk
    c = torch.where(torch.isnan(c), torch.zeros_like(c), c)
    out = c.mean()
    _dbg(_TAG, "huber_lc", loss=out, ad_mean=ad.mean())
    return out


def masked_mse(preds, labels, null_val=np.nan):
    mk = _mask(preds, labels, null_val)
    l = (preds - labels) ** 2 * mk
    l = torch.where(torch.isnan(l), torch.zeros_like(l), l)
    return l.mean()


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=np.nan):
    """接口兼容, 内部走 huber+logcosh."""
    return masked_huber_logcosh(preds, labels, null_val)


# ---- 改动2: MAPE floor clamp ----
_MAPE_FLOOR = 5e-6


def masked_mape(preds, labels, null_val=np.nan):
    mk = _mask(preds, labels, null_val)
    safe = torch.clamp(torch.abs(labels), min=_MAPE_FLOOR)
    l = torch.abs(preds - labels) / safe * mk
    l = torch.where(torch.isnan(l), torch.zeros_like(l), l)
    return l.mean()


# ---- 改动3: 新增 quantile loss (upstream 无) ----
def quantile_loss(preds, labels, tau=0.5, null_val=np.nan):
    mk = _mask(preds, labels, null_val)
    d = labels - preds
    ql = torch.where(d >= 0, tau * d, (tau - 1) * d) * mk
    ql = torch.where(torch.isnan(ql), torch.zeros_like(ql), ql)
    return ql.mean()


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    _dbg(_TAG, "metric",
         mae=torch.tensor(mae), mape=torch.tensor(mape), rmse=torch.tensor(rmse))
    return mae, mape, rmse


# ---- 改动4: 时间一致性惩罚 (upstream 无) ----
def temporal_consistency_penalty(preds, labels, alpha=0.1):
    """惩罚预测比真值更抖的情况, 交通流相邻步通常平滑."""
    if preds.dim() < 2 or preds.shape[1] < 2:
        return torch.tensor(0.0, device=preds.device)
    pd = torch.abs(preds[:, 1:] - preds[:, :-1])
    rd = torch.abs(labels[:, 1:] - labels[:, :-1])
    excess = torch.clamp(pd - rd, min=0.0)
    pen = alpha * excess.mean()
    _dbg(_TAG, "temporal_pen", penalty=pen,
         pred_rough=pd.mean(), real_rough=rd.mean())
    return pen
