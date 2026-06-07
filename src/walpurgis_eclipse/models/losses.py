"""Eclipse losses: Tukey biweight + gradient_penalty."""
import torch, numpy as np, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

def _ldbg(tag, v):
    if not _ECL_DBG: return
    if isinstance(v, torch.Tensor):
        print(f"[ECL:{tag}@losses] shape={list(v.shape)} min={v.min().item():.6f} max={v.max().item():.6f} mean={v.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[ECL:{tag}@losses] value={v}", file=sys.stderr)

def tukey_biweight_loss(preds, labels, null_val=0.0, c=4.685):
    """Tukey biweight: robust to outliers. rho(r)=(c^2/6)[1-(1-(r/c)^2)^3] if |r|<=c."""
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    denom = mask.mean().clamp(min=1e-8)
    mask = mask / denom
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = (preds - labels) * mask
    r_abs = residual.abs()
    c2_6 = (c**2) / 6.0
    inlier = r_abs <= c
    ratio_sq = (residual / c) ** 2
    biweight = c2_6 * (1.0 - (1.0 - ratio_sq).pow(3))
    loss = torch.where(inlier, biweight, torch.full_like(biweight, c2_6))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("tukey.preds", preds); _ldbg("tukey.loss", loss)
    return loss.mean()

def gradient_penalty(preds, alpha=0.01):
    """Penalize excessive 2nd-order temporal differences."""
    if preds.dim() < 2: return torch.tensor(0.0, device=preds.device)
    d1 = preds[:, 1:, ...] - preds[:, :-1, ...]
    if d1.shape[1] < 2: return torch.tensor(0.0, device=preds.device)
    d2 = d1[:, 1:, ...] - d1[:, :-1, ...]
    p = alpha * d2.abs().mean()
    _ldbg("grad_penalty", p)
    return p

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2 * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("mae.preds", preds); _ldbg("mae.labels", labels); _ldbg("mae.loss", loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float(); mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
