import torch
import numpy as np
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:loss:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:loss:{tag}] {val}", file=sys.stderr)

def huber_loss(preds, labels, null_val=np.nan, delta=1.35):
    """upstream: masked_mae (L1)
    solstice: Huber鲁棒损失 — 小误差用MSE平滑, 大误差用L1截断, delta=1.35 (95%高斯效率)"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    quadratic = torch.clamp(residual, max=delta)
    linear = residual - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _adbg("huber_raw", loss)
    return torch.mean(loss)

def spectral_smoothness_penalty(preds, beta=0.03):
    """solstice新增: 频域平滑惩罚 — 高频分量能量占比过大则惩罚, 抑制预测锯齿"""
    if preds.dim() < 3 or preds.shape[1] < 4:
        return torch.tensor(0.0, device=preds.device)
    fft_out = torch.fft.rfft(preds, dim=1)
    magnitudes = torch.abs(fft_out)
    n_freq = magnitudes.shape[1]
    cutoff = max(1, n_freq // 2)
    high_energy = torch.mean(magnitudes[:, cutoff:, :] ** 2)
    total_energy = torch.mean(magnitudes ** 2) + 1e-8
    ratio = high_energy / total_energy
    pen = beta * ratio
    _adbg("spectral_pen", pen)
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
