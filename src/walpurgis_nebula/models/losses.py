"""Nebula losses: Log-Cosh + quantile regression composite loss."""
import torch, numpy as np, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

def _ldbg(tag, v):
    if not _NEB_DBG: return
    if isinstance(v, torch.Tensor):
        print(f"[NEB:{tag}@losses] shape={list(v.shape)} min={v.min().item():.6f} max={v.max().item():.6f} mean={v.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[NEB:{tag}@losses] value={v}", file=sys.stderr)

def log_cosh_loss(preds, labels, null_val=0.0):
    """Log-Cosh loss: smooth approx of L1 for small errors, L2-like for large.
    L(x) = log(cosh(x)) ≈ (x^2)/2 for small x, |x| - log(2) for large x."""
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    denom = mask.mean().clamp(min=1e-8)
    mask = mask / denom
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = (preds - labels) * mask
    # Numerically stable log-cosh: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    abs_r = residual.abs()
    loss = abs_r + torch.log1p(torch.exp(-2.0 * abs_r)) - 0.6931471805599453
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("logcosh.preds", preds); _ldbg("logcosh.loss", loss)
    return loss.mean()

def quantile_loss(preds, labels, tau=0.5, null_val=0.0):
    """Pinball / quantile loss for asymmetric error penalization.
    L_tau(y, f) = tau * max(y-f, 0) + (1-tau) * max(f-y, 0)."""
    if np.isnan(null_val): mask = ~torch.isnan(labels)
    else: mask = (labels != null_val)
    mask = mask.float()
    denom = mask.mean().clamp(min=1e-8)
    mask = mask / denom
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = labels - preds
    loss = torch.where(residual >= 0, tau * residual, (tau - 1.0) * residual)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _ldbg("quantile.tau", tau); _ldbg("quantile.loss", loss)
    return loss.mean()

def nebula_composite_loss(preds, labels, null_val=0.0, alpha=0.7, tau=0.5):
    """Composite: alpha * LogCosh + (1-alpha) * QuantileLoss."""
    lc = log_cosh_loss(preds, labels, null_val)
    ql = quantile_loss(preds, labels, tau, null_val)
    composite = alpha * lc + (1.0 - alpha) * ql
    _ldbg("composite", composite)
    return composite

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
