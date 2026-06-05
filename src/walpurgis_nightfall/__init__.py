"""
walpurgis_nightfall — 鲁迅式移植 D2STGNN (Nightfall 变体)
算法改写点: spectral gating + adaptive温度 + 梯度噪声退火
全局调试: NIGHTFALL_DEBUG=1 开启, 逗号分隔指定模块
"""
import os
import sys
import torch
import numpy as np
from collections import defaultdict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("NIGHTFALL_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()

def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES

def _dbg(tag, tensor_or_msg, module=""):
    """通用断点诊断: 打印tensor统计或字符串消息"""
    if not _is_debug(module) and not _DEBUG_ALL:
        return
    prefix = f"[NF-DBG:{tag}]"
    if isinstance(tensor_or_msg, torch.Tensor):
        t = tensor_or_msg
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        sparse_ratio = (t.abs() < 1e-8).float().mean().item() * 100
        alert = ""
        if has_nan:
            alert += " ⚠NaN!"
        if has_inf:
            alert += " ⚠Inf!"
        if sparse_ratio > 95.0:
            alert += f" SPARSE({sparse_ratio:.1f}%)"
        print(f"{prefix} shape={list(t.shape)} dtype={t.dtype} "
              f"min={t.min().item():.6f} max={t.max().item():.6f} "
              f"mean={t.mean().item():.6f} std={t.std().item():.6f}"
              f"{alert}", file=sys.stderr, flush=True)
    elif isinstance(tensor_or_msg, np.ndarray):
        a = tensor_or_msg
        print(f"{prefix} np_shape={a.shape} dtype={a.dtype} "
              f"min={a.min():.6f} max={a.max():.6f} "
              f"mean={a.mean():.6f} nan_count={np.isnan(a).sum()}", file=sys.stderr, flush=True)
    else:
        print(f"{prefix} {tensor_or_msg}", file=sys.stderr, flush=True)


# ─── 模型参数快照 ────────────────────────────────────────
def snapshot_model(model, epoch=0, step=0, top_k=5):
    """打印模型参数统计快照: 各层norm/mean/std + 异常检测"""
    if not _is_debug():
        return
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[SNAPSHOT] epoch={epoch} step={step}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    stats = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            norm_val = p.data.norm().item()
            mean_val = p.data.mean().item()
            std_val = p.data.std().item()
            alert_flags = []
            if std_val < 1e-6:
                alert_flags.append("COLLAPSED_SCALE")
            if abs(mean_val) > 3 * max(std_val, 1e-9):
                alert_flags.append("MEAN_DRIFT")
            if p.grad is not None and p.grad.norm().item() > 50:
                alert_flags.append("GRAD_SPIKE")
            stats.append((norm_val, name, mean_val, std_val, alert_flags))
    stats.sort(key=lambda x: -x[0])
    for norm_val, name, mean_val, std_val, flags in stats[:top_k]:
        flag_str = " ".join(flags) if flags else ""
        print(f"  {name:50s} norm={norm_val:10.4f} μ={mean_val:+.6f} σ={std_val:.6f} {flag_str}",
              file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


# ─── 激活值追踪器 ────────────────────────────────────────
class ActivationTracker:
    """Hook-based activation tracker: 记录每层激活值统计"""
    def __init__(self):
        self.stats = defaultdict(list)
        self._hooks = []

    def _hook_fn(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.stats[name].append({
                    'mean': out.mean().item(),
                    'std': out.std().item(),
                    'abs_max': out.abs().max().item(),
                    'dead_frac': (out.abs() < 1e-7).float().mean().item(),
                })
        return hook

    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.GRUCell,
                                torch.nn.MultiheadAttention, torch.nn.Conv1d)):
                h = mod.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)
        return self

    def report(self, top_k=8):
        print(f"\n[ACT-TRACK] Activation statistics ({len(self.stats)} layers):", file=sys.stderr)
        items = []
        for name, records in self.stats.items():
            if records:
                avg_std = np.mean([r['std'] for r in records])
                avg_dead = np.mean([r['dead_frac'] for r in records])
                items.append((avg_std, name, avg_dead))
        items.sort(key=lambda x: -x[0])
        for avg_std, name, avg_dead in items[:top_k]:
            warn = " ⚠DEAD" if avg_dead > 0.5 else ""
            print(f"  {name:50s} avg_std={avg_std:.6f} dead_frac={avg_dead:.3f}{warn}", file=sys.stderr)

    def check_dead(self, threshold=0.5):
        dead_layers = []
        for name, records in self.stats.items():
            if records:
                avg_dead = np.mean([r['dead_frac'] for r in records])
                if avg_dead > threshold:
                    dead_layers.append((name, avg_dead))
        return dead_layers

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.stats.clear()


def register_activation_hooks(model):
    tracker = ActivationTracker()
    tracker.register(model)
    return tracker


# ─── 梯度健康检查 ────────────────────────────────────────
def gradient_health_check(model, explode_thresh=100.0, vanish_thresh=1e-7):
    """检查全部参数的梯度: 爆炸/消失/NaN三类"""
    issues = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            if torch.isnan(p.grad).any():
                issues.append(f"  ✗ NaN grad: {name}")
            elif g_norm > explode_thresh:
                issues.append(f"  ✗ EXPLODING grad: {name} (norm={g_norm:.2f})")
            elif g_norm < vanish_thresh:
                issues.append(f"  ✗ VANISHING grad: {name} (norm={g_norm:.2e})")
    if issues:
        for iss in issues:
            print(iss, file=sys.stderr)
    return issues


# ─── 权重差异对比 ────────────────────────────────────────
def weight_diff(state_a, state_b, top_k=5):
    """对比两个state_dict间的参数变化量"""
    diffs = []
    for key in state_a:
        if key in state_b:
            delta = (state_a[key].float() - state_b[key].float()).norm().item()
            diffs.append((delta, key))
    diffs.sort(key=lambda x: -x[0])
    print(f"\n[WEIGHT-DIFF] top-{top_k} changed parameters:", file=sys.stderr)
    for delta, key in diffs[:top_k]:
        print(f"  {key:50s} Δ={delta:.6f}", file=sys.stderr)
    frozen = [key for delta, key in diffs if delta < 1e-10]
    if frozen:
        print(f"  ⚠ {len(frozen)} parameters appear frozen (Δ<1e-10)", file=sys.stderr)
