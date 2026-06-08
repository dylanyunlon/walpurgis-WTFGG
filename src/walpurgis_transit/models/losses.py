"""
Losses — Transit变体 (M055)
算法改动 #9: Tweedie Loss 替代 Log-Cosh / MAE
  Tweedie分布: 指数色散模型的特例, 方差与均值的幂次关系
  variance = φ · μ^p, p ∈ (1,2) 对应 compound Poisson-Gamma
  特别适合零膨胀(zero-inflated)和右偏(right-skewed)数据
  可学习的power参数p控制分布形状:
    p→1: Poisson-like (纯计数)
    p=1.5: 经典Tweedie (零膨胀连续)
    p→2: Gamma-like (正连续)
  损失 = -log_lik ∝ (y·μ^(1-p))/(1-p) - μ^(2-p)/(2-p)
  数值稳定: clamp μ>ε, p∈[1.01,1.99]
"""
import torch
import numpy as np
from .. import _dbg


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


def tweedie_loss(preds, labels, null_val=np.nan,
                 power=1.5):
    """Tweedie deviance loss: 适合零膨胀/右偏数据
    power ∈ (1,2): compound Poisson-Gamma
    loss = 2 * [ y^(2-p)/((1-p)(2-p)) - y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p) ]
    简化为: -y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p) 的负对数似然部分
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)

    # clamp power到安全范围, 避免除零
    p = torch.clamp(torch.as_tensor(power), 1.01, 1.99)

    # μ必须>0, 用softplus确保正值
    mu = torch.nn.functional.softplus(preds) + 1e-8
    y = torch.clamp(labels, min=0.0)  # Tweedie要求y≥0

    # Tweedie deviance (unit deviance):
    # d(y,μ) = 2·[ y^(2-p)/((1-p)(2-p)) - y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p) ]
    # 但训练只需梯度方向, 用负对数似然的核心项:
    # nll ∝ μ^(2-p)/(2-p) - y·μ^(1-p)/(1-p)
    term1 = torch.pow(mu, 2.0 - p) / (2.0 - p)
    term2 = y * torch.pow(mu, 1.0 - p) / (1.0 - p)
    loss = term1 - term2

    loss = loss * mask
    loss = torch.where(
        torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(
        torch.isinf(loss), torch.zeros_like(loss), loss)

    _dbg("tweedie.mu_range",
         f"[{mu.min().item():.4f},{mu.max().item():.4f}]",
         "loss")
    _dbg("tweedie.power", f"{p.item():.3f}", "loss")
    _dbg("tweedie.raw_loss", f"{loss.mean().item():.6f}",
         "loss")

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


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
