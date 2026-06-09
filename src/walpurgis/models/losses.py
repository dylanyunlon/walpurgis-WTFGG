"""
Cascade losses — 算法改写:
  新增 cascade_aware_loss: 级联感知损失
  对不同预测时间步(horizon)施加递增权重
  近期预测权重大, 远期预测权重递增(鼓励模型关注难样本)
  同时加入gradient-scaled penalty: 梯度大的样本额外惩罚
"""
import torch
import torch.nn.functional as F
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


def compute_psd_uncertainty(labels, null_val=0.0, quantile=0.7):
    """从TITAN移植: PSD-based不确定性估计
    原理: 对label序列的时间差分做功率谱密度分析
    高PSD变化率 = 高不确定性(不可预测的突变) → 降低该样本权重
    低PSD变化率 = 低不确定性(规律性强的样本) → 保持/增加权重

    Args:
        labels: [B, T, N] — 真实值序列
        null_val: 空值标记
        quantile: 不确定性阈值分位数

    Returns:
        uncertainty: [B, 1, N] — 每个样本每个节点的不确定性权重
    """
    with torch.no_grad():
        vals = labels.clone()
        vals[vals == null_val] = float('nan')
        # 时间差分: 捕获变化率
        diff = vals[:, 1:, :] - vals[:, :-1, :]
        # 沿时间维度计算均值和标准差
        abs_mean = torch.nanmean(torch.abs(diff), dim=1, keepdim=True)
        # nanstd: sqrt(nanmean((x - nanmean(x))^2))
        diff_centered = diff - torch.nanmean(diff, dim=1, keepdim=True)
        diff_var = torch.nanmean(diff_centered ** 2, dim=1, keepdim=True)
        diff_std = torch.sqrt(diff_var + 1e-6)
        # 变异系数: mean(|diff|) / std(diff) — TITAN的核心指标
        cv = abs_mean / (diff_std + 1e-6)
        cv = torch.where(torch.isnan(cv), torch.zeros_like(cv), cv)
        # 低cv = 不确定(噪声大于信号) → uncertainty=1
        # 高cv = 确定(信号大于噪声) → uncertainty=0
        threshold = torch.quantile(cv[cv > 0].flatten(), quantile) if (cv > 0).any() else cv.mean()
        uncertainty = (cv < threshold).float()
        return uncertainty


def cascade_aware_loss(preds, labels, null_val=np.nan,
                       base_weight=1.0, horizon_scale=0.15,
                       grad_penalty=0.002,
                       _learnable_hw=None,
                       _use_uncertainty=True,
                       _current_epoch=0):
    """Cascade特有: 级联感知损失

    对不同预测horizon施加指数递增权重:
      w_t = exp(horizon_scale * t)
    远期horizon(更难)获得指数级更高权重, 迫使模型关注长程预测质量

    gradient-scaled penalty: 对残差大的样本施加额外二次惩罚
    penalty系数降低(0.002)以减少早期训练噪声干扰收敛

    _learnable_hw: optional Parameter[T] for data-driven horizon weighting
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

    # Data-driven horizon adaptation: blend fixed schedule with learnable weights
    if _learnable_hw is not None and _learnable_hw.shape[0] >= T:
        learned_w = F.softplus(_learnable_hw[:T])  # ensure positive
        learned_w = learned_w / (learned_w.mean() + 1e-8)
        learned_w = learned_w.unsqueeze(0).unsqueeze(-1)
        # 70% fixed schedule + 30% learned (stable but adaptive)
        horizon_w = 0.7 * horizon_w + 0.3 * learned_w.to(horizon_w.device)

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

    # ═══ 从TITAN移植: PSD不确定性感知加权 ═══
    # 高不确定性样本(突变/噪声)降权 → 防止模型被不可预测的样本带偏
    # epoch < 20时不启用，让模型先学会基本模式
    if _use_uncertainty and _current_epoch >= 20:
        uncertainty = compute_psd_uncertainty(labels, null_val=0.0)
        # uncertainty=1: 不确定(降权到0.6), uncertainty=0: 确定(权重1.0)
        unc_weight = 1.0 - 0.4 * uncertainty  # [B, 1, N]
        weighted_loss = weighted_loss * unc_weight
        _dbg("cascade_loss.uncertainty_mean",
             uncertainty.mean(), "loss")
        _dbg("cascade_loss.unc_weight_range",
             f"[{unc_weight.min().item():.3f}, {unc_weight.max().item():.3f}]", "loss")

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
