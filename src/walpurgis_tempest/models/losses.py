"""Tempest losses: Focal regression loss + L2 temporal smoothness penalty.
Focal regression: modulates loss by (1-exp(-|r|/beta))^gamma to focus on hard examples.
L2 smoothness: penalizes squared 2nd-order temporal differences for smooth predictions."""
import torch, numpy as np, sys, os
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

def _ldbg(tag, v):
    if not _TEM_DBG: return
    if isinstance(v, torch.Tensor):
        print(f"[TEM:{tag}@losses] shape={list(v.shape)} min={v.min().item():.6f} max={v.max().item():.6f} mean={v.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[TEM:{tag}@losses] value={v}", file=sys.stderr)

def focal_regression_loss(preds, labels, null_val=0.0, gamma=2.0, beta=1.0):
    """Focal regression: weight hard samples more via modulating factor.
    loss_i = (1 - exp(-|r_i|/beta))^gamma * |r_i|
    This focuses optimization on hard-to-predict samples, unlike upstream MAE
    (uniform weighting) or eclipse Tukey (downweight outliers)."""
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    denom = mask.mean().clamp(min=1e-8)
    mask = mask / denom
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = (preds - labels).abs()
    # Focal modulating factor: harder samples (larger |r|) get higher weight
    focal_weight = (1.0 - torch.exp(-residual / beta)).pow(gamma)
    loss = focal_weight * residual * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("focal.preds", preds); _ldbg("focal.weight", focal_weight); _ldbg("focal.loss", loss)
    return loss.mean()

def l2_smoothness_penalty(preds, alpha=0.005):
    """L2 temporal smoothness: squared 2nd-order differences.
    Encourages smooth prediction curves vs eclipse gradient_penalty (L1 on 2nd diff)."""
    if preds.dim() < 2: return torch.tensor(0.0, device=preds.device)
    d1 = preds[:, 1:, ...] - preds[:, :-1, ...]
    if d1.shape[1] < 2: return torch.tensor(0.0, device=preds.device)
    d2 = d1[:, 1:, ...] - d1[:, :-1, ...]
    penalty = alpha * (d2 ** 2).mean()  # L2 squared (vs eclipse L1 abs)
    _ldbg("l2_smooth", penalty)
    return penalty

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= mask.mean().clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2 * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val).clamp(min=1e-10))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= mask.mean().clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("mae.preds", preds); _ldbg("mae.labels", labels); _ldbg("mae.loss", loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= mask.mean().clamp(min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels.abs().clamp(min=1e-8) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
