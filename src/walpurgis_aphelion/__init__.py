"""
walpurgis_aphelion — D2STGNN Aphelion(远日点)变体
算法改写 (~20%):
  - Hypernetwork gate — 小网络生成gate权重(替代固定FC)
  - VMD (Variational Mode Decomposition) 风格分解
  - GAT v2 + edge features (替代标准图卷积)
  - SimCLR contrastive learning adjacency (对比学习邻接)
  - Entropy-regularized mask (最大化mask信息熵)
  - Wavelet-based normalizer (小波归一化)
  - Retention network + cross-scale fusion (替代GRU)
  - FPN (Feature Pyramid Network) multi-scale output
  - Tilted Empirical Risk Minimization loss
  - Sophia optimizer + exponential decay scheduler

全局调试: APHELION_DEBUG=1 开启
"""
import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("APHELION_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()

_DIAG_LOG_PATH = os.environ.get("APHELION_DIAG_LOG", "")


def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断日志到JSONL（若配置了APHELION_DIAG_LOG）"""
    if not _DIAG_LOG_PATH:
        return
    try:
        with open(_DIAG_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _dbg(tag, tensor_or_msg, module=""):
    """断点诊断: 打印tensor统计或消息, 写入JSONL"""
    if not _is_debug(module) and not _DEBUG_ALL:
        return
    prefix = f"[APH-DBG:{tag}]"
    ts = time.time()
    if isinstance(tensor_or_msg, torch.Tensor):
        t = tensor_or_msg
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        sparse_frac = (t.abs() < 1e-8).float().mean().item()
        alert = ""
        if has_nan:
            alert += " !!NaN"
        if has_inf:
            alert += " !!Inf"
        if sparse_frac > 0.95:
            alert += f" SPARSE({sparse_frac*100:.1f}%)"
        msg = (f"shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f}"
               f"{alert}")
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)
        _diag_write({
            "ts": ts, "tag": tag, "type": "tensor",
            "shape": list(t.shape), "min": t.min().item(),
            "max": t.max().item(), "mean": t.mean().item(),
            "std": t.std().item(), "nan": has_nan, "inf": has_inf,
            "sparse_frac": sparse_frac
        })
    elif isinstance(tensor_or_msg, np.ndarray):
        a = tensor_or_msg
        msg = (f"np_shape={a.shape} dtype={a.dtype} "
               f"min={a.min():.6f} max={a.max():.6f} "
               f"mean={a.mean():.6f} nan_count={np.isnan(a).sum()}")
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)
        _diag_write({
            "ts": ts, "tag": tag, "type": "ndarray",
            "shape": list(a.shape), "min": float(a.min()),
            "max": float(a.max()), "mean": float(a.mean())
        })
    else:
        print(f"{prefix} {tensor_or_msg}", file=sys.stderr, flush=True)
        _diag_write({"ts": ts, "tag": tag, "type": "msg", "msg": str(tensor_or_msg)})


# ─── 模型参数快照 ────────────────────────────────────────
def snapshot_model(model, epoch=0, step=0, top_k=8):
    """打印模型参数统计快照 + 异常检测"""
    if not _is_debug():
        return
    print(f"\n{'='*65}", file=sys.stderr)
    print(f"[APH-SNAPSHOT] epoch={epoch} step={step} "
          f"total_params={sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)
    print(f"{'='*65}", file=sys.stderr)
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
        g_norm = p.grad.norm().item() if p.grad is not None else 0.0
        if p.grad is not None and g_norm > 50:
            alerts.append(f"GRAD_SPIKE({g_norm:.1f})")
        stats.append((norm_val, name, mean_val, std_val, g_norm, alerts))
    stats.sort(key=lambda x: -x[0])
    for norm_val, name, mean_val, std_val, g_norm, alerts in stats[:top_k]:
        alert_str = " ".join(alerts) if alerts else ""
        print(f"  {name:50s} norm={norm_val:10.4f} μ={mean_val:+.6f} "
              f"σ={std_val:.6f} |∇|={g_norm:.4f} {alert_str}",
              file=sys.stderr)
    print(f"{'='*65}\n", file=sys.stderr)


# ─── 激活值追踪器 ────────────────────────────────────────
class ActivationTracker:
    """Hook-based activation tracker"""
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
                    'q99': torch.quantile(out.abs().float(), 0.99).item(),
                })
        return hook

    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.GRUCell,
                                torch.nn.LSTMCell,
                                torch.nn.MultiheadAttention, torch.nn.Conv1d,
                                torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
                h = mod.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)
        return self

    def tick(self):
        self._step_count += 1

    def report(self, top_k=10):
        print(f"\n[APH-ACTIVATION] {len(self.stats)} layers tracked:",
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
            print(f"  {name:50s} avg_std={avg_std:.6f} "
                  f"dead={avg_dead:.3f} q99_max={max_q99:.4f}{warn}",
                  file=sys.stderr)

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
    """检查全部参数梯度: 爆炸/消失/NaN"""
    issues = []
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            grad_norms.append((name, g_norm))
            if torch.isnan(p.grad).any():
                issues.append(f"  ✗ NaN grad: {name}")
            elif g_norm > explode_thresh:
                issues.append(f"  ✗ EXPLODING: {name} (norm={g_norm:.2f})")
            elif g_norm < vanish_thresh:
                issues.append(f"  ✗ VANISHING: {name} (norm={g_norm:.2e})")
    if issues and _is_debug():
        print("[APH-GRAD] Issues detected:", file=sys.stderr)
        for iss in issues:
            print(iss, file=sys.stderr)
    return issues, grad_norms


# ─── Aphelion特有: VMD分解状态监控 ────────────────────────
class VMDStateMonitor:
    """监控VMD分解模块的模态分离状态, 检测频带收敛/重叠"""
    def __init__(self):
        self._history = defaultdict(list)

    def record(self, layer_name, mode_energies, residual_val):
        self._history[layer_name].append({
            'mode_energies': [e.item() if torch.is_tensor(e) else e for e in mode_energies],
            'res_mean': residual_val.mean().item(),
            'res_std': residual_val.std().item(),
        })

    def report(self):
        if not _is_debug():
            return
        print(f"\n[APH-VMD] VMD decomposition state:", file=sys.stderr)
        for name, records in self._history.items():
            if len(records) < 2:
                continue
            last = records[-1]
            energies = last['mode_energies']
            overlap_flag = ""
            if len(energies) > 1:
                ratios = [max(energies[i], energies[i+1]) / (min(energies[i], energies[i+1]) + 1e-8)
                          for i in range(len(energies)-1)]
                if any(r < 1.5 for r in ratios):
                    overlap_flag = " !!MODE_OVERLAP"
            print(f"  {name:40s} modes={energies} "
                  f"res_std={last['res_std']:.6f}{overlap_flag}", file=sys.stderr)


# ─── Aphelion特有: TERM损失追踪 ──────────────────────────
class TERMLossTracker:
    """追踪Tilted ERM损失的tilt参数和尾部风险"""
    def __init__(self):
        self._records = []

    def record(self, tilt_param, loss_val, tail_ratio):
        self._records.append({
            'tilt': tilt_param.item() if torch.is_tensor(tilt_param) else tilt_param,
            'loss': loss_val.item() if torch.is_tensor(loss_val) else loss_val,
            'tail_ratio': tail_ratio,
        })

    def report(self):
        if not _is_debug() or not self._records:
            return
        print(f"\n[APH-TERM] Tilted ERM loss tracking:", file=sys.stderr)
        last = self._records[-1]
        print(f"  tilt={last['tilt']:.4f} loss={last['loss']:.6f} "
              f"tail_ratio={last['tail_ratio']:.4f}", file=sys.stderr)


# ─── Aphelion特有: Retention状态诊断 ─────────────────────
def retention_state_dump(name, state, decay_factors):
    """打印Retention网络的状态和衰减因子统计"""
    if not _is_debug():
        return
    s_norm = state.norm().item()
    decay_range = (decay_factors.min().item(), decay_factors.max().item())
    alert = ""
    if s_norm < 1e-6:
        alert = " !!STATE_VANISHED"
    if decay_range[0] < 0.01:
        alert += " !!FAST_DECAY"
    print(f"[APH-RET:{name}] state_norm={s_norm:.6f} "
          f"decay_range=[{decay_range[0]:.4f}, {decay_range[1]:.4f}]{alert}",
          file=sys.stderr, flush=True)


# ─── 梯度直方图 ─────────────────────────────────────────
def gradient_histogram(model, n_bins=8):
    """所有梯度按数量级分桶打印"""
    if not _is_debug():
        return
    all_grads = []
    for p in model.parameters():
        if p.grad is not None:
            all_grads.append(p.grad.abs().flatten())
    if not all_grads:
        print("[APH-GRAD-HIST] No gradients yet.", file=sys.stderr)
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
    print(f"[APH-GRAD-HIST] log10 gradient distribution:", file=sys.stderr)
    for i in range(n_bins):
        bar_len = int(counts[i] / total * 40)
        bar = "█" * bar_len
        print(f"  [{edges[i]:+6.2f}, {edges[i+1]:+6.2f}): "
              f"{counts[i]:8d} ({counts[i]/total*100:5.1f}%) {bar}",
              file=sys.stderr)


# ─── 权重差异对比 ────────────────────────────────────────
def weight_diff(state_a, state_b, top_k=5):
    """对比两个state_dict间的参数变化量"""
    diffs = []
    for key in state_a:
        if key in state_b:
            delta = (state_a[key].float() - state_b[key].float()).norm().item()
            diffs.append((delta, key))
    diffs.sort(key=lambda x: -x[0])
    print(f"\n[APH-WEIGHT-DIFF] top-{top_k} changed parameters:", file=sys.stderr)
    for delta, key in diffs[:top_k]:
        print(f"  {key:50s} Δ={delta:.6f}", file=sys.stderr)
    frozen = [key for delta, key in diffs if delta < 1e-10]
    if frozen:
        print(f"  !! {len(frozen)} parameters appear frozen (Δ<1e-10)",
              file=sys.stderr)
    return diffs


# ─── 数据流检查点 ────────────────────────────────────────
def dataflow_checkpoint(name, tensor, expected_shape=None):
    """在关键数据流节点检查shape + 统计"""
    if not _is_debug():
        return
    prefix = f"[APH-FLOW:{name}]"
    if expected_shape is not None:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"{prefix} !!SHAPE MISMATCH dim[{i}]: "
                      f"expected {e} got {a}, full shape={actual}",
                      file=sys.stderr)
    _dbg(name, tensor, "flow")


# ─── 计时器 ──────────────────────────────────────────────
class PerfTimer:
    """细粒度训练阶段计时器"""
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
        print(f"\n[APH-PERF] Timing breakdown:", file=sys.stderr)
        for name, times in self._timers.items():
            avg_ms = np.mean(times) * 1000
            total_s = np.sum(times)
            print(f"  {name:30s} avg={avg_ms:8.2f}ms "
                  f"total={total_s:.2f}s ({len(times)} calls)",
                  file=sys.stderr)


# ─── Aphelion全局结构体状态dump ──────────────────────────
def struct_dump(model, label=""):
    """一次性dump所有关键结构体的形状和数值范围"""
    if not _is_debug():
        return
    print(f"\n[APH-STRUCT-DUMP] {label}", file=sys.stderr)
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            grad_info = f"|∇|={param.grad.norm().item():.4f}" if has_grad else "no_grad"
            print(f"  {name:55s} {str(list(param.shape)):20s} "
                  f"μ={param.data.mean().item():+.6f} "
                  f"σ={param.data.std().item():.6f} {grad_info}",
                  file=sys.stderr)
    print("", file=sys.stderr)
