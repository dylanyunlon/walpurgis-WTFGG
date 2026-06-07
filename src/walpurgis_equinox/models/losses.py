import torch
import numpy as np
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:loss:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:loss:{tag}] {val}", file=sys.stderr)


def logcosh_loss(preds, labels, null_val=np.nan):
    """upstream: masked_mae (L1)
    equinox: LogCosh损失 L(r)=log(cosh(r)), 平滑可微的Huber替代
    在小残差时近似L2, 大残差时近似L1, 兼顾灵敏度和鲁棒性"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask / torch.clamp(torch.mean(mask), min=1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = preds - labels
    # log(cosh(x)) = x + softplus(-2x) - log(2), 数值稳定版本
    loss = residual + torch.nn.functional.softplus(-2.0 * residual) - np.log(2.0)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    _edbg("logcosh_raw", loss)
    return torch.mean(loss)


def cutmix_spatiotemporal(x, y, alpha=1.0):
    """equinox新增: CutMix时空块替换数据增强
    随机选取时间窗口和节点子集, 在batch内交换对应区域
    增强模型对局部缺失/异常的鲁棒性
    x: [B, T, N, D] or [B, T, N]"""
    if alpha <= 0 or x.shape[0] < 2:
        return x, y
    B, T, N = x.shape[0], x.shape[1], x.shape[2]
    lam = np.random.beta(alpha, alpha)
    # 随机选择混合对象
    indices = torch.randperm(B, device=x.device)
    # 时间维度CutMix窗口
    t_len = max(1, int(T * (1 - lam)))
    t_start = np.random.randint(0, max(1, T - t_len))
    # 空间维度CutMix节点子集
    n_cut = max(1, int(N * (1 - lam)))
    n_perm = torch.randperm(N, device=x.device)[:n_cut]
    # 执行替换 — 使用切片避免广播问题
    x_mixed = x.clone()
    y_mixed = y.clone()
    # 逐节点替换, 避免高级索引广播不兼容
    for ni in n_perm:
        x_mixed[:, t_start:t_start+t_len, ni] = x[indices][:, t_start:t_start+t_len, ni]
    if y.dim() >= 3 and y.shape[2] == N:
        for ni in n_perm:
            y_mixed[:, :, ni] = y[indices][:, :, ni]
    _edbg("cutmix", f"lam={lam:.3f} t_window=[{t_start},{t_start+t_len}) n_nodes={n_cut}/{N}")
    return x_mixed, y_mixed


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
