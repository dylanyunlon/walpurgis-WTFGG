"""
Cascade losses — 算法改写:
  新增 cascade_aware_loss: 级联感知损失
  对不同预测时间步(horizon)施加递增权重
  近期预测权重大, 远期预测权重递增(鼓励模型关注难样本)
  同时加入gradient-scaled penalty: 梯度大的样本额外惩罚
"""
import torch
import numpy as np
from .. import _dbg

_horizon_weights_cache = {}


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


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


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


def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()
    # crit = torch.nn.MSELoss()
    return crit(preds, labels)


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


def cascade_aware_loss(preds, labels, null_val=np.nan,
                       base_weight=1.0, horizon_scale=0.12,
                       grad_penalty=0.002):
    """Cascade特有: 级联感知损失

    对不同预测horizon施加指数递增权重:
      w_t = exp(horizon_scale * t)
    远期horizon(更难)获得指数级更高权重, 迫使模型关注长程预测质量

    gradient-scaled penalty: 对残差大的样本施加额外二次惩罚
    penalty系数降低(0.002)以减少早期训练噪声干扰收敛
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(
        torch.isnan(mask), torch.zeros_like(mask), mask)

    residual = torch.abs(preds - labels)

    # 生成horizon权重 [1, T, 1] — 指数递增
    T = preds.shape[1]
    cache_key = (T, preds.device, horizon_scale)
    if cache_key not in _horizon_weights_cache:
        hw = torch.arange(T, dtype=torch.float32,
                          device=preds.device)
        hw = torch.exp(horizon_scale * hw)  # 指数递增
        hw = hw / hw.mean()  # 归一化使均值为1
        _horizon_weights_cache[cache_key] = hw.unsqueeze(0).unsqueeze(-1)
    horizon_w = _horizon_weights_cache[cache_key]

    # weighted MAE
    weighted_residual = residual * horizon_w
    weighted_loss = weighted_residual * mask
    weighted_loss = torch.where(
        torch.isnan(weighted_loss),
        torch.zeros_like(weighted_loss), weighted_loss)

    # gradient-scaled penalty: 大残差额外惩罚 (tighter clamp for stability)
    with torch.no_grad():
        residual_std = residual[mask.bool()].std().clamp(min=0.5)
    penalty = grad_penalty * (residual / residual_std).clamp(max=3.0).pow(2)
    penalty = penalty * mask
    penalty = torch.where(
        torch.isnan(penalty),
        torch.zeros_like(penalty), penalty)

    total = torch.mean(weighted_loss) + torch.mean(penalty)

    _dbg("cascade_loss.weighted_mae",
         torch.mean(weighted_loss), "loss")
    _dbg("cascade_loss.penalty",
         torch.mean(penalty), "loss")
    _dbg("cascade_loss.horizon_range",
         f"[{horizon_w.min().item():.3f}, "
         f"{horizon_w.max().item():.3f}]", "loss")

    return total


class LogCoshHorizonLoss(torch.nn.Module):
    """融合: LogCosh平滑梯度 + cascade的horizon-weighted策略
    LogCosh: 小误差像MSE(平滑), 大误差像MAE(鲁棒)
    Horizon权重: 远期预测权重递增
    自适应温度: 早期高温(平滑)→后期低温(精确)
    """

    def __init__(self, init_temperature=1.0, horizon_scale=0.1):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(np.log(init_temperature)))
        self.horizon_scale = horizon_scale
        self._epoch = 0
        self._temp_schedule_alpha = 0.02  # 温度退火速率

    def set_epoch(self, epoch):
        """由trainer每epoch调用,驱动自适应温度"""
        self._epoch = epoch

    def forward(self, preds, labels, null_val=0.0):
        mask = (labels != null_val).float()
        mask = mask / (mask.mean() + 1e-8)
        mask = torch.where(torch.isnan(mask),
                          torch.zeros_like(mask), mask)

        # 自适应温度: sigmoid退火 — 早期T大(平滑),后期T小(精确)
        epoch_factor = 1.0 / (1.0 + np.exp(self._temp_schedule_alpha * (self._epoch - 50)))
        T_base = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        T = T_base * (0.3 + 0.7 * epoch_factor)  # T范围: [0.3*base, base]

        diff = (preds - labels) / T
        loss = T * torch.log(torch.cosh(diff.clamp(-20, 20)))

        # horizon weighting — exponential (consistent with cascade_aware_loss)
        num_horizons = preds.shape[1]
        hw = torch.arange(num_horizons, dtype=torch.float32,
                          device=preds.device)
        hw = torch.exp(self.horizon_scale * hw)
        hw = hw / hw.mean()
        hw = hw.unsqueeze(0).unsqueeze(-1)
        loss = loss * hw

        loss = loss * mask
        loss = torch.where(torch.isnan(loss),
                          torch.zeros_like(loss), loss)

        _dbg("logcosh_horizon/temperature", T, "loss")
        _dbg("logcosh_horizon/epoch_factor",
             f"epoch={self._epoch} factor={epoch_factor:.4f}", "loss")
        return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
