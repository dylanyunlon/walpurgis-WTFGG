import torch
import numpy as np
from walpurgis_reverie import _dbg

_TAG = "losses"


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
    return torch.sqrt(masked_mse(
        preds=preds, labels=labels, null_val=null_val))


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


class LogCoshLoss(torch.nn.Module):
    """upstream: masked_mae (L1 loss)
    改动: Log-Cosh loss with adaptive temperature
    log(cosh(x/T)) ≈ |x|/T - log(2)/T when |x| >> T (like L1)
    log(cosh(x/T)) ≈ x^2/(2*T^2) when |x| << T (like L2)
    自适应T根据误差分布调整: 小误差时像MSE (平滑梯度), 大误差时像MAE (抗异常值)
    """

    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(np.log(init_temperature)))

    def forward(self, preds, labels, null_val=0.0):
        mask = (labels != null_val).float()
        mask = mask / (mask.mean() + 1e-8)
        mask = torch.where(torch.isnan(mask),
                          torch.zeros_like(mask), mask)

        T = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        diff = (preds - labels) / T
        loss = T * torch.log(torch.cosh(diff.clamp(-20, 20)))
        loss = loss * mask
        loss = torch.where(torch.isnan(loss),
                          torch.zeros_like(loss), loss)

        _dbg(f"{_TAG}/logcosh_temperature", T, _TAG)
        return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
