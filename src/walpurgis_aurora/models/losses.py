import torch
import numpy as np
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:loss:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[AUR:loss:{tag}] {val}", file=sys.stderr)

def cauchy_loss(preds, labels, null_val=np.nan, gamma=2.385):
    """upstream: masked_mae (L1)
    aurora: Cauchy鲁棒损失 L(r)=log(1+(r/gamma)^2), gamma=2.385 -> 95%效率"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = preds - labels
    loss = torch.log1p((residual / gamma) ** 2)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _adbg("cauchy_raw", loss)
    return torch.mean(loss)

def temporal_coherence_penalty(preds, alpha=0.05):
    """aurora新增: 二阶差分惩罚抑制预测跳变"""
    if preds.dim() < 3 or preds.shape[1] < 3:
        return torch.tensor(0.0, device=preds.device)
    d1 = preds[:, 1:, :] - preds[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    pen = alpha * torch.mean(torch.abs(d2))
    _adbg("temporal_pen", pen)
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
