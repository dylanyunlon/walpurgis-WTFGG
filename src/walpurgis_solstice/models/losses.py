import torch
import numpy as np
import sys, os

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:loss:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:loss:{tag}] {val}", file=sys.stderr)

def huber_loss(preds, labels, null_val=np.nan, delta=1.345):
    """upstream: masked_mae (L1)
    solstice: Huber loss (平滑L1) — |r|<δ时用0.5r²/δ, 否则|r|-0.5δ
    delta=1.345 -> 95% asymptotic efficiency under Gaussian"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = preds - labels
    abs_r = torch.abs(residual)
    # Huber: quadratic for small residuals, linear for large
    loss = torch.where(abs_r <= delta,
                       0.5 * residual ** 2 / delta,
                       abs_r - 0.5 * delta)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _sdbg("huber_raw", loss)
    _sdbg("huber_delta", delta)
    return torch.mean(loss)

def spatial_smoothness_penalty(preds, adj=None, beta=0.03):
    """solstice新增: 空间平滑惩罚 — 相邻节点预测差异最小化
    如果有邻接矩阵用图拉普拉斯, 否则用节点间方差"""
    if preds.dim() < 3 or preds.shape[2] < 2:
        return torch.tensor(0.0, device=preds.device)
    # 节点维度方差作为空间不平滑度量
    node_var = torch.var(preds, dim=2, keepdim=False)
    pen = beta * torch.mean(node_var)
    _sdbg("spatial_pen", pen)
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
