import torch
import numpy as np
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:loss:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:loss:{tag}] {val}", file=sys.stderr)

def logcosh_loss(preds, labels, null_val=np.nan, scale=1.0):
    """upstream: masked_mae (L1)
    equinox: LogCosh鲁棒损失 L(r)=log(cosh(r/scale)), scale=1.0 -> 平滑L1近似"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = (preds - labels) / scale
    # log(cosh(x)) = x + softplus(-2x) - log(2), numerically stable
    loss = residual + torch.nn.functional.softplus(-2.0 * residual) - np.log(2.0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _edbg("logcosh_raw", loss)
    return torch.mean(loss)

def spectral_smoothness_penalty(preds, beta=0.03):
    """equinox新增: 频域平滑惩罚 — 高频分量L2正则"""
    if preds.dim() < 3 or preds.shape[1] < 4:
        return torch.tensor(0.0, device=preds.device)
    # 沿时间轴做实数FFT, 惩罚高频能量
    fft_out = torch.fft.rfft(preds, dim=1)
    n_freq = fft_out.shape[1]
    # 高频: 后半部分
    high_start = n_freq // 2
    high_freq_energy = torch.mean(torch.abs(fft_out[:, high_start:, :]) ** 2)
    pen = beta * high_freq_energy
    _edbg("spectral_pen", pen)
    return pen

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / torch.clamp(torch.abs(labels), min=1e-8)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
