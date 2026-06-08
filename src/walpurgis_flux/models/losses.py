"""
Flux losses — 算法改写:
  新增 focal_mae: Focal Loss变体应用于MAE
  对预测误差大的样本(hard samples)施加更大权重,
  gamma参数控制聚焦强度: gamma=0退化为MAE, gamma越大越聚焦hard sample
  alpha参数做类别平衡(此处用误差分位数区分easy/hard)
"""
import torch
import numpy as np
from .. import _dbg

_focal_stats = {"call_count": 0, "hard_frac_ema": 0.5}


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
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
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


def focal_mae(preds, labels, null_val=np.nan,
              gamma=2.0, alpha=0.75):
    """Flux特有: Focal MAE损失
    将Focal Loss的思想应用于回归任务:
      - 计算每个样本的归一化误差(相对于当前batch最大误差)
      - 误差越大的样本(hard sample)权重越高: w = (err_norm)^gamma
      - alpha控制整体缩放, gamma控制聚焦强度
      - gamma=0退化为标准masked_mae

    与Vortex的Huber-MAE不同: Focal关注样本难度分布,
    Huber关注outlier鲁棒性; Focal让模型更关注难预测的时间步
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)
    # 绝对误差
    residual = torch.abs(preds - labels)
    # 归一化误差到[0,1]用于focal weighting
    with torch.no_grad():
        valid_residual = residual[mask.bool()]
        if valid_residual.numel() > 0:
            r_max = valid_residual.max().clamp(min=1e-6)
            r_median = valid_residual.median()
        else:
            r_max = torch.tensor(1.0)
            r_median = torch.tensor(0.5)
        err_normalized = (residual / r_max).clamp(0, 1)
        # Focal权重: hard sample(大误差)获得更高权重
        focal_weight = alpha * (err_normalized ** gamma)
        # 确保权重均值归一化, 避免loss scale漂移
        focal_weight = focal_weight / (focal_weight.mean() + 1e-8)
        # 统计hard sample比例(误差>中位数的)
        hard_frac = (valid_residual > r_median).float().mean().item()
        _focal_stats["call_count"] += 1
        _focal_stats["hard_frac_ema"] = (
            0.95 * _focal_stats["hard_frac_ema"] +
            0.05 * hard_frac)
    # 加权MAE
    loss = residual * focal_weight * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    _dbg("focal_gamma", f"{gamma}", "loss")
    _dbg("focal_hard_frac",
         f"{_focal_stats['hard_frac_ema']:.4f}", "loss")
    _dbg("focal_weight_range",
         f"[{focal_weight.min().item():.4f},"
         f"{focal_weight.max().item():.4f}]", "loss")
    return torch.mean(loss)


def progressive_refinement_loss(coarse_pred, fine_pred,
                                labels, null_val=np.nan,
                                coarse_weight=0.3):
    """Flux特有: 渐进式解码的多级损失
    对粗预测和细预测分别计算focal_mae, 用权重混合
    coarse_weight控制粗预测损失的贡献比例
    """
    coarse_loss = focal_mae(coarse_pred, labels, null_val)
    fine_loss = focal_mae(fine_pred, labels, null_val)
    combined = coarse_weight * coarse_loss + (1 - coarse_weight) * fine_loss
    _dbg("prog_coarse_loss", f"{coarse_loss.item():.6f}", "loss")
    _dbg("prog_fine_loss", f"{fine_loss.item():.6f}", "loss")
    return combined


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
