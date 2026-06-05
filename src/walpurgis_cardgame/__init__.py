"""
walpurgis_cardgame — D2STGNN CardGame变体
==================================================
算法改写要点 (vs upstream D2STGNN):
  - Welsch robust loss + temporal smoothness penalty
  - 3层FC瓶颈EstimationGate + LayerNorm + Swish + learnable temperature
  - CELU激活 + learnable residual scaling beta
  - WeightNorm + Mish + gconv残差skip
  - 2层MLP + tanh门控 backcast
  - spectral dropout in diffusion forecast
  - DropNode + learnable receive gain
  - Mahalanobis距离 + learnable协方差
  - sigmoid soft-gating + Gumbel noise mask
  - Laplacian归一化 I - D^{-1/2}AD^{-1/2}
  - GELU + gradient checkpoint in inherent
  - learnable step decay + ALiBi位置偏移
  - tanh gated residual
  - attention加权聚合 + Swish输出头 + spatial dropout
  - RAdam + OneCycleLR + gradient norm tracking + cosine temperature
  - Cauchy kernel邻接 + symmetric closure
  - Winsorize离群值 + cyclic feature encoding
  - EarlyStopping plateau slope detection + deterministic seed derivation
  - JSONL structured log + per-epoch metric CSV dump
  - mixup augmentation + circular padding
  - DataParallel + AMP + activation probe + ensemble test

调试体系 (CARDGAME_DEBUG=1):
  - _dbg(): 打印tensor统计 (shape/dtype/min/max/mean/std/NaN/Inf)
  - snapshot_model(): 每epoch参数快照
  - ActivationTracker: 注册前向hook跟踪各层激活
  - gradient_health_check(): 检查梯度健康状态
  - weight_diff(): 比较两次快照间参数变化
"""
import os
import sys

__version__ = "0.1.0"
__variant__ = "cardgame"

# ─── CARDGAME_DEBUG 全局调试开关 ───
_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _is_debug():
    """检查当前是否开启调试模式"""
    return os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _dbg(tag, tensor, module=""):
    """CardGame调试输出: 打印tensor/value的完整统计信息

    Args:
        tag: 标识标签 (如 'fwd.input', 'loss.welsch')
        tensor: 待检查的tensor或标量
        module: 可选的模块名
    """
    if not _is_debug():
        return
    prefix = f"[CG-DBG:{tag}]"
    if module:
        prefix = f"[CG-DBG:{tag}@{module}]"

    if hasattr(tensor, 'shape'):
        import torch
        stats = (f" shape={list(tensor.shape)} dtype={tensor.dtype}"
                 f" min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
                 f" mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        alerts = ""
        if nan_count > 0:
            alerts += f" *** NaN={nan_count} ***"
        if inf_count > 0:
            alerts += f" *** Inf={inf_count} ***"
        msg = f"{prefix}{stats}{alerts}"
    elif hasattr(tensor, '__len__'):
        msg = f"{prefix} len={len(tensor)} type={type(tensor).__name__}"
    else:
        msg = f"{prefix} value={tensor}"
    print(msg, file=sys.stderr)


def snapshot_model(model, epoch=0, step=0):
    """参数快照: 打印所有参数的统计信息

    Returns:
        dict: {name: (mean, std, norm)} 用于后续weight_diff比较
    """
    if not _is_debug():
        return {}
    import torch
    print(f"\n[CG-SNAPSHOT] epoch={epoch} step={step}", file=sys.stderr)
    snap = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            m = p.data.mean().item()
            s = p.data.std().item()
            n = p.data.norm().item()
            snap[name] = (m, s, n)
            print(f"  {name:50s} shape={list(p.shape)} "
                  f"mean={m:.6f} std={s:.6f} norm={n:.6f}", file=sys.stderr)
    print(f"[CG-SNAPSHOT] total params: {len(snap)}\n", file=sys.stderr)
    return snap


def weight_diff(snap_before, snap_after, threshold=1e-7):
    """比较两次参数快照间的变化量

    Args:
        snap_before: snapshot_model返回的前一次快照
        snap_after: snapshot_model返回的后一次快照
        threshold: 变化阈值, 低于此值视为未更新
    """
    if not _is_debug():
        return
    print("[CG-WDIFF] Parameter changes:", file=sys.stderr)
    frozen_count = 0
    for name in snap_before:
        if name in snap_after:
            delta_norm = abs(snap_after[name][2] - snap_before[name][2])
            flag = " ** FROZEN **" if delta_norm < threshold else ""
            if flag:
                frozen_count += 1
            print(f"  {name:50s} Δnorm={delta_norm:.8f}{flag}", file=sys.stderr)
    if frozen_count > 0:
        print(f"[CG-WDIFF] WARNING: {frozen_count} params appear frozen!", file=sys.stderr)


class ActivationTracker:
    """前向hook跟踪各层激活分布"""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            if hasattr(output, 'shape'):
                self.activations[name] = {
                    'shape': list(output.shape),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'nan': output.isnan().sum().item(),
                    'inf': output.isinf().sum().item(),
                }
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def report(self):
        print("\n[CG-ACT] Activation Report:", file=sys.stderr)
        for name, stats in self.activations.items():
            alerts = ""
            if stats['nan'] > 0:
                alerts += " NaN!"
            if stats['inf'] > 0:
                alerts += " Inf!"
            if stats['std'] < 1e-6:
                alerts += " DEAD?"
            print(f"  {name:50s} shape={stats['shape']} "
                  f"μ={stats['mean']:.4f} σ={stats['std']:.4f} "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]{alerts}",
                  file=sys.stderr)
        print(f"[CG-ACT] Tracked {len(self.activations)} layers\n",
              file=sys.stderr)


def register_activation_hooks(model):
    """为模型所有子模块注册activation tracking hooks"""
    tracker = ActivationTracker()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            h = module.register_forward_hook(tracker._hook_fn(name))
            tracker._hooks.append(h)
    return tracker


def gradient_health_check(model):
    """检查梯度健康: 打印每个参数的梯度统计"""
    if not _is_debug():
        return
    print("\n[CG-GRAD] Gradient Health Check:", file=sys.stderr)
    no_grad_count = 0
    exploding = 0
    vanishing = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad_count += 1
                print(f"  {name:50s} grad=None", file=sys.stderr)
            else:
                g = p.grad
                gn = g.norm().item()
                flag = ""
                if gn > 100:
                    flag = " ** EXPLODING **"
                    exploding += 1
                elif gn < 1e-8:
                    flag = " ** VANISHING **"
                    vanishing += 1
                print(f"  {name:50s} grad_norm={gn:.6f} "
                      f"grad_mean={g.mean().item():.6f}{flag}",
                      file=sys.stderr)
    print(f"[CG-GRAD] no_grad={no_grad_count} exploding={exploding} "
          f"vanishing={vanishing}\n", file=sys.stderr)
