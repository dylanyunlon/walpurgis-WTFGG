"""
walpurgis_reverie — D2STGNN Reverie变体 (幻想/白日梦)
算法改写 (~20% modification vs upstream):
  1. Bilinear + Swish estimation gate (替代FC→ReLU→FC→sigmoid)
  2. Instance-norm residual decomposition with learnable affine (替代LayerNorm)
  3. Chebyshev polynomial graph conv with learnable order weights (替代固定k_s阶MatMul)
  4. Radial basis function distance (替代Q/K dot-product attention)
  5. Straight-through Gumbel top-k mask (替代static adj masking)
  6. Row-stochastic doubly-normalized adjacency (替代简单degree归一化)
  7. Minimal GRU + exponential positional decay (替代标准GRU + sinusoidal PE)
  8. LayerScale + GELU output head (替代ReLU FC→FC)
  9. Log-cosh loss with adaptive temperature (替代masked MAE)
  10. RAdam + CosineAnnealingWarmRestarts (替代Adam + MultiStepLR)

调试基础设施:
  - 每一层forward输出tensor统计 (shape/min/max/mean/std/nan/inf)
  - 结构体全状态dump (所有Parameter的norm/mean/std)
  - 激活值dead neuron检测
  - 梯度直方图 + 梯度爆炸/消失告警
  - 训练阶段耗时拆分
  - 学习率追踪
  - 每epoch数据流断言

全局调试: REVERIE_DEBUG=1 开启
"""
import os
import sys
import time
import json
import math
import torch
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("REVERIE_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()
_DIAG_LOG_PATH = os.environ.get("REVERIE_DIAG_LOG", "")


def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断日志到JSONL文件"""
    if not _DIAG_LOG_PATH:
        return
    try:
        with open(_DIAG_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _dbg(tag, tensor_or_msg, module=""):
    """通用断点诊断: 打印tensor全部统计 + 异常检测, 模拟现实世界print调试"""
    if not _is_debug(module) and not _DEBUG_ALL:
        return
    prefix = f"[RV-DBG:{tag}]"
    ts = time.time()
    if isinstance(tensor_or_msg, torch.Tensor):
        t = tensor_or_msg
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        sparse_frac = (t.abs() < 1e-8).float().mean().item()
        neg_frac = (t < 0).float().mean().item()
        q01 = torch.quantile(t.float().flatten(), 0.01).item() if t.numel() > 10 else t.min().item()
        q99 = torch.quantile(t.float().flatten(), 0.99).item() if t.numel() > 10 else t.max().item()
        alert = ""
        if has_nan:
            alert += " !!NaN"
        if has_inf:
            alert += " !!Inf"
        if sparse_frac > 0.9:
            alert += f" SPARSE({sparse_frac*100:.1f}%)"
        if neg_frac > 0.95:
            alert += " ALL_NEG"
        msg = (f"shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
               f"q01={q01:.6f} q99={q99:.6f}"
               f"{alert}")
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)
        _diag_write({
            "ts": ts, "tag": tag, "type": "tensor",
            "shape": list(t.shape), "min": t.min().item(),
            "max": t.max().item(), "mean": t.mean().item(),
            "std": t.std().item(), "nan": has_nan, "inf": has_inf,
            "sparse_frac": sparse_frac, "neg_frac": neg_frac,
            "q01": q01, "q99": q99,
        })
    elif isinstance(tensor_or_msg, np.ndarray):
        a = tensor_or_msg
        msg = (f"np shape={a.shape} dtype={a.dtype} "
               f"min={a.min():.6f} max={a.max():.6f} "
               f"mean={a.mean():.6f} nan_count={np.isnan(a).sum()}")
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)
        _diag_write({
            "ts": ts, "tag": tag, "type": "ndarray",
            "shape": list(a.shape), "min": float(a.min()),
            "max": float(a.max()), "mean": float(a.mean()),
        })
    else:
        print(f"{prefix} {tensor_or_msg}", file=sys.stderr, flush=True)
        _diag_write({"ts": ts, "tag": tag, "type": "msg", "msg": str(tensor_or_msg)})


# ─── 结构体全状态dump ────────────────────────────────────
def snapshot_model(model, epoch=0, step=0, top_k=10):
    """打印模型全部参数统计快照 — 像gdb的info locals"""
    if not _is_debug():
        return
    total_p = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[RV-SNAPSHOT] epoch={epoch} step={step} "
          f"total={total_p:,} trainable={trainable:,}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    stats = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = p.data
        norm_val = d.norm().item()
        mean_val = d.mean().item()
        std_val = d.std().item()
        alerts = []
        if std_val < 1e-6:
            alerts.append("COLLAPSED")
        if abs(mean_val) > 3 * max(std_val, 1e-9):
            alerts.append("MEAN_DRIFT")
        if torch.isnan(d).any():
            alerts.append("NaN_PARAM")
        g_norm = 0.0
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            if g_norm > 50:
                alerts.append(f"GRAD_SPIKE({g_norm:.1f})")
            elif g_norm < 1e-8:
                alerts.append("GRAD_ZERO")
        stats.append((norm_val, name, mean_val, std_val, g_norm, alerts))
    stats.sort(key=lambda x: -x[0])
    for norm_val, name, mean_val, std_val, g_norm, alerts in stats[:top_k]:
        alert_str = " ".join(alerts) if alerts else ""
        print(f"  {name:55s} |w|={norm_val:10.4f} μ={mean_val:+.6f} "
              f"σ={std_val:.6f} |∇|={g_norm:.4f} {alert_str}",
              file=sys.stderr)
    remaining = len(stats) - top_k
    if remaining > 0:
        print(f"  ... and {remaining} more parameters", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    _diag_write({
        "ts": time.time(), "tag": "snapshot", "epoch": epoch, "step": step,
        "total_params": total_p, "trainable_params": trainable,
        "layers": [{
            "name": n, "norm": nv, "mean": mv, "std": sv, "grad_norm": gn
        } for nv, n, mv, sv, gn, _ in stats[:top_k]]
    })


# ─── 激活值追踪器 ────────────────────────────────────────
class ActivationTracker:
    """Hook-based activation tracker: 类似TensorBoard的实时追踪"""
    def __init__(self):
        self.stats = defaultdict(list)
        self._hooks = []
        self._step_count = 0

    def _hook_fn(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.stats[name].append({
                    'step': self._step_count,
                    'mean': out.mean().item(),
                    'std': out.std().item(),
                    'abs_max': out.abs().max().item(),
                    'dead_frac': (out.abs() < 1e-7).float().mean().item(),
                    'q99': torch.quantile(out.abs().float(), 0.99).item() if out.numel() > 1 else out.abs().item(),
                    'entropy': -(out.softmax(-1) * out.log_softmax(-1)).sum(-1).mean().item() if out.dim() >= 2 else 0.0,
                })
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                self.stats[name].append({
                    'step': self._step_count,
                    'mean': out[0].mean().item(),
                    'std': out[0].std().item(),
                    'abs_max': out[0].abs().max().item(),
                    'dead_frac': (out[0].abs() < 1e-7).float().mean().item(),
                    'q99': torch.quantile(out[0].abs().float(), 0.99).item() if out[0].numel() > 1 else 0.0,
                })
        return hook

    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.GRUCell,
                                torch.nn.MultiheadAttention, torch.nn.Conv1d,
                                torch.nn.LayerNorm, torch.nn.BatchNorm2d,
                                torch.nn.InstanceNorm1d)):
                h = mod.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)
        return self

    def tick(self):
        self._step_count += 1

    def report(self, top_k=12):
        print(f"\n[RV-ACTIVATION] {len(self.stats)} layers tracked:",
              file=sys.stderr)
        items = []
        for name, records in self.stats.items():
            if records:
                avg_std = np.mean([r['std'] for r in records])
                avg_dead = np.mean([r['dead_frac'] for r in records])
                max_q99 = max(r['q99'] for r in records)
                items.append((avg_std, name, avg_dead, max_q99))
        items.sort(key=lambda x: -x[0])
        for avg_std, name, avg_dead, max_q99 in items[:top_k]:
            warn = ""
            if avg_dead > 0.5:
                warn += " !!DEAD"
            if max_q99 > 100:
                warn += f" !!HOT(q99={max_q99:.1f})"
            print(f"  {name:55s} avg_std={avg_std:.6f} "
                  f"dead={avg_dead:.3f} q99_max={max_q99:.4f}{warn}",
                  file=sys.stderr)

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
def gradient_health_check(model, explode_thresh=80.0, vanish_thresh=1e-7):
    """逐参数梯度检查: 爆炸/消失/NaN — 模拟valgrind风格"""
    issues = []
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            grad_norms.append((name, g_norm))
            if torch.isnan(p.grad).any():
                issues.append(f"  ✗ NaN grad: {name}")
            elif g_norm > explode_thresh:
                issues.append(f"  ✗ EXPLODING: {name} (|∇|={g_norm:.2f})")
            elif g_norm < vanish_thresh:
                issues.append(f"  ✗ VANISHING: {name} (|∇|={g_norm:.2e})")
    if issues and _is_debug():
        print("[RV-GRAD] Issues detected:", file=sys.stderr)
        for iss in issues:
            print(iss, file=sys.stderr)
    return issues, grad_norms


# ─── 梯度直方图 ─────────────────────────────────────────
def gradient_histogram(model, n_bins=8):
    """将所有梯度按数量级分桶, 打印直方图 (Reverie特有: 含per-layer分解)"""
    if not _is_debug():
        return
    all_grads = []
    per_layer = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            flat = p.grad.abs().flatten()
            all_grads.append(flat)
            per_layer[name] = flat.mean().item()
    if not all_grads:
        print("[RV-GRAD-HIST] No gradients yet.", file=sys.stderr)
        return
    flat = torch.cat(all_grads)
    log_grads = torch.log10(flat.clamp(min=1e-12))
    lo, hi = log_grads.min().item(), log_grads.max().item()
    edges = np.linspace(lo, hi, n_bins + 1)
    counts = []
    for i in range(n_bins):
        mask = (log_grads >= edges[i]) & (log_grads < edges[i + 1])
        counts.append(mask.sum().item())
    total = sum(counts) or 1
    print(f"[RV-GRAD-HIST] log10 gradient distribution (total={flat.numel():,}):", file=sys.stderr)
    for i in range(n_bins):
        bar_len = int(counts[i] / total * 40)
        bar = "█" * bar_len
        print(f"  [{edges[i]:+6.2f}, {edges[i+1]:+6.2f}): "
              f"{counts[i]:8d} ({counts[i]/total*100:5.1f}%) {bar}",
              file=sys.stderr)
    # top-5 layers by avg gradient magnitude
    sorted_layers = sorted(per_layer.items(), key=lambda x: -x[1])[:5]
    print(f"[RV-GRAD-HIST] Top-5 layers by |∇| mean:", file=sys.stderr)
    for name, val in sorted_layers:
        print(f"  {name:50s} {val:.6f}", file=sys.stderr)


# ─── 权重差异对比 ────────────────────────────────────────
def weight_diff(state_a, state_b, top_k=5):
    """对比两个state_dict间参数变化量 — 像git diff for weights"""
    diffs = []
    for key in state_a:
        if key in state_b:
            delta = (state_a[key].float() - state_b[key].float()).norm().item()
            relative = delta / (state_a[key].float().norm().item() + 1e-12)
            diffs.append((delta, key, relative))
    diffs.sort(key=lambda x: -x[0])
    print(f"\n[RV-WEIGHT-DIFF] top-{top_k} changed parameters:", file=sys.stderr)
    for delta, key, rel in diffs[:top_k]:
        print(f"  {key:50s} Δ={delta:.6f} (rel={rel:.4f})", file=sys.stderr)
    frozen = [key for delta, key, _ in diffs if delta < 1e-10]
    if frozen:
        print(f"  !! {len(frozen)} parameters appear frozen (Δ<1e-10)",
              file=sys.stderr)
    return diffs


# ─── 数据流检查点 ────────────────────────────────────────
def dataflow_checkpoint(name, tensor, expected_shape=None):
    """在关键数据流节点插入断点检查: 形状断言 + 统计输出"""
    if not _is_debug():
        return
    prefix = f"[RV-FLOW:{name}]"
    if expected_shape is not None:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"{prefix} !!SHAPE MISMATCH dim[{i}]: "
                      f"expected {e} got {a}, full shape={actual}",
                      file=sys.stderr)
    _dbg(name, tensor, "flow")


# ─── 学习率追踪 ─────────────────────────────────────────
class LRTracker:
    """记录每个step的学习率, epoch结束时打印趋势"""
    def __init__(self):
        self.history = []

    def log(self, step, lr):
        self.history.append((step, lr))

    def report_epoch(self, epoch):
        if not _is_debug() or not self.history:
            return
        lrs = [lr for _, lr in self.history]
        print(f"[RV-LR] epoch={epoch} "
              f"start={lrs[0]:.6f} end={lrs[-1]:.6f} "
              f"min={min(lrs):.6f} max={max(lrs):.6f} "
              f"steps={len(lrs)}", file=sys.stderr)


# ─── 训练速度计时器 ──────────────────────────────────────
class PerfTimer:
    """细粒度训练阶段计时器 — 类似perf profiling"""
    def __init__(self):
        self._timers = OrderedDict()
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(name)
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)

    def report(self):
        if not _is_debug():
            return
        print(f"\n[RV-PERF] Timing breakdown:", file=sys.stderr)
        total_all = sum(sum(v) for v in self._timers.values())
        for name, times in self._timers.items():
            avg_ms = np.mean(times) * 1000
            total_s = np.sum(times)
            pct = (total_s / total_all * 100) if total_all > 0 else 0
            print(f"  {name:30s} avg={avg_ms:8.2f}ms "
                  f"total={total_s:.2f}s ({len(times)} calls, {pct:.1f}%)",
                  file=sys.stderr)


# ─── Loss追踪器 (Reverie特有) ───────────────────────────
class LossTracker:
    """追踪loss的移动平均和方差, 检测训练不稳定性"""
    def __init__(self, window=50):
        self.window = window
        self.values = []

    def update(self, loss_val):
        self.values.append(loss_val)

    def report(self, epoch):
        if not _is_debug() or len(self.values) < 2:
            return
        recent = self.values[-self.window:]
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        trend = "↓" if len(self.values) > self.window and np.mean(self.values[-self.window:]) < np.mean(self.values[:self.window]) else "→"
        warn = ""
        if std_val > mean_val * 0.5:
            warn = " !!UNSTABLE"
        if len(self.values) > 2 and self.values[-1] > self.values[-2] * 2:
            warn += " !!SPIKE"
        print(f"[RV-LOSS] epoch={epoch} recent_mean={mean_val:.6f} "
              f"std={std_val:.6f} trend={trend}{warn}",
              file=sys.stderr)
