"""
Aphelion losses — 算法改写 #9:
  upstream: masked_mae as primary loss
  corona: quantile loss (多分位数回归)
  aphelion: Tilted Empirical Risk Minimization (TERM) loss —
            通过可学习的tilt参数t控制对尾部风险的关注程度:
            当t>0时更关注高损失样本(鲁棒性), t<0时关注低损失样本(效率)。
            TERM loss = (1/t) * log(E[exp(t * loss_i)]), 在t→0时退化为ERM。
            比MAE/quantile loss能更灵活地控制风险偏好。
  改动幅度: ~30% (TERM loss替代MAE/quantile)
"""
import torch
import numpy as np
from .. import _dbg, TERMLossTracker

_term_tracker = TERMLossTracker()


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


def tilted_erm_loss(preds, labels, tilt_param, null_val=0):
    """Aphelion特有: Tilted Empirical Risk Minimization (TERM) loss
    TERM loss = (1/t) * log( (1/n) * sum(exp(t * loss_i)) )
    当t>0: 关注高损失样本(worst-case), 提升鲁棒性
    当t<0: 关注低损失样本(best-case), 提升效率
    当t→0: 退化为标准ERM (即普通的均值loss)

    参数:
      tilt_param: 可学习的tilt参数, 控制风险偏好
      null_val: 缺失值标记
    """
    # 基础逐元素损失: 使用Huber-like损失替代纯MAE, 更平滑
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    errors = torch.abs(preds - labels)
    # Huber平滑: 在|e|<δ时用二次, 否则用线性
    delta = 1.0
    huber = torch.where(errors < delta, 0.5 * errors ** 2, delta * (errors - 0.5 * delta))
    per_sample_loss = huber * mask
    per_sample_loss = torch.where(torch.isnan(per_sample_loss),
                                   torch.zeros_like(per_sample_loss), per_sample_loss)

    # 展平为1D的per-sample loss
    flat_loss = per_sample_loss.flatten()

    # TERM公式: (1/t) * log( mean(exp(t * loss_i)) )
    # 为数值稳定性使用log-sum-exp trick
    t = tilt_param  # 已经是scalar tensor
    if t.abs() < 1e-4:
        # t≈0时退化为普通均值
        term_loss = flat_loss.mean()
    else:
        # log-sum-exp稳定化: max trick
        t_loss = t * flat_loss
        max_val = t_loss.max().detach()
        term_loss = (1.0 / t) * (max_val + torch.log(torch.mean(torch.exp(t_loss - max_val))))

    # 计算尾部风险比率 (用于诊断)
    with torch.no_grad():
        threshold = flat_loss.mean() + flat_loss.std()
        tail_ratio = (flat_loss > threshold).float().mean().item()
    _term_tracker.record(tilt_param.detach(), term_loss.detach(), tail_ratio)

    return term_loss


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
