"""
walpurgis_prism — D2STGNN Prism变体
算法改写 (~20%):
  - Multi-view编码: 空间/时间/频率三视角独立编码后加权融合
  - Contrastive loss: 相邻节点embedding对比正则化
  - Mixup数据增强: 训练时对输入进行线性插值增强
  - 频率域特征: FFT提取频谱特征与时域融合
  - 视角注意力融合: 可学习的视角权重做自适应融合
  - 对比温度退火: contrastive loss的温度参数随训练退火

全局调试: PRISM_DEBUG=1 开启
诊断日志: PRISM_DIAG_LOG=<path> 写入JSONL
"""
import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("PRISM_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()
_DIAG_LOG_PATH = os.environ.get("PRISM_DIAG_LOG", "")


def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断记录到JSONL（若设置了PRISM_DIAG_LOG）"""
    if not _DIAG_LOG_PATH:
        return
    try:
        with open(_DIAG_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _dbg(tag, tensor_or_msg, module=""):
    """通用断点诊断: 打tensor统计或字符串, 同时写JSONL"""
    if not _is_debug(module) and not _DEBUG_ALL:
        return
    prefix = f"[PR-DBG:{tag}]"
    ts = time.time()
    if isinstance(tensor_or_msg, torch.Tensor):
        t = tensor_or_msg
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        dead_frac = (t.abs() < 1e-8).float().mean().item()
        alert = ""
        if has_nan:
            alert += " !!NaN"
        if has_inf:
            alert += " !!Inf"
        if dead_frac > 0.90:
            alert += f" DEAD({dead_frac*100:.1f}%)"
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
            "dead_frac": dead_frac
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
        _diag_write({"ts": ts, "tag": tag, "type": "msg",
                      "msg": str(tensor_or_msg)})


# ─── 模型参数快照 ────────────────────────────────────────
def snapshot_model(model, epoch=0, step=0, top_k=10):
    """打印模型参数统计快照 + 异常检测"""
    if not _is_debug():
        return
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[PR-SNAPSHOT] epoch={epoch} step={step} "
          f"params={sum(p.numel() for p in model.parameters()):,}",
          file=sys.stderr)
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
        if std_val < 1e-7:
            alerts.append("COLLAPSED")
        if abs(mean_val) > 4 * max(std_val, 1e-9):
            alerts.append("MEAN_DRIFT")
        if torch.isnan(d).any():
            alerts.append("NaN_PARAM")
        g_norm = p.grad.norm().item() if p.grad is not None else 0.0
        if g_norm > 60:
            alerts.append(f"GRAD_SPIKE({g_norm:.1f})")
        stats.append((norm_val, name, mean_val, std_val, g_norm, alerts))
    stats.sort(key=lambda x: -x[0])
    for norm_val, name, mean_val, std_val, g_norm, alerts in stats[:top_k]:
        alert_str = " ".join(alerts) if alerts else ""
        print(f"  {name:55s} norm={norm_val:10.4f} mu={mean_val:+.6f} "
              f"sig={std_val:.6f} |g|={g_norm:.4f} {alert_str}",
              file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    _diag_write({
        "ts": time.time(), "tag": "snapshot", "epoch": epoch, "step": step,
        "layers": [{
            "name": n, "norm": nv, "mean": mv, "std": sv, "grad_norm": gn
        } for nv, n, mv, sv, gn, _ in stats[:top_k]]
    })


# ─── 激活值追踪器 ────────────────────────────────────────
class ActivationTracker:
    """Hook-based activation tracker: 记录每层输出统计"""
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
                    'q95': torch.quantile(
                        out.abs().float(), 0.95).item(),
                })
        return hook

    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.GRUCell,
                                torch.nn.MultiheadAttention, torch.nn.Conv1d,
                                torch.nn.LayerNorm, torch.nn.GroupNorm,
                                torch.nn.BatchNorm2d)):
                h = mod.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)
        return self

    def tick(self):
        self._step_count += 1

    def report(self, top_k=12):
        print(f"\n[PR-ACTIVATION] {len(self.stats)} layers tracked:",
              file=sys.stderr)
        items = []
        for name, records in self.stats.items():
            if records:
                avg_std = np.mean([r['std'] for r in records])
                avg_dead = np.mean([r['dead_frac'] for r in records])
                max_q95 = max(r['q95'] for r in records)
                items.append((avg_std, name, avg_dead, max_q95))
        items.sort(key=lambda x: -x[0])
        for avg_std, name, avg_dead, max_q95 in items[:top_k]:
            warn = ""
            if avg_dead > 0.5:
                warn += " !!DEAD"
            if max_q95 > 80:
                warn += f" !!HOT(q95={max_q95:.1f})"
            print(f"  {name:55s} avg_std={avg_std:.6f} "
                  f"dead={avg_dead:.3f} q95_max={max_q95:.4f}{warn}",
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
def gradient_health_check(model, explode_thresh=70.0,
                          vanish_thresh=1e-7):
    """检查全部参数梯度: 爆炸/消失/NaN"""
    issues = []
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item()
            grad_norms.append((name, g_norm))
            if torch.isnan(p.grad).any():
                issues.append(f"  x NaN grad: {name}")
            elif g_norm > explode_thresh:
                issues.append(
                    f"  x EXPLODING: {name} (norm={g_norm:.2f})")
            elif g_norm < vanish_thresh:
                issues.append(
                    f"  x VANISHING: {name} (norm={g_norm:.2e})")
    if issues and _is_debug():
        print("[PR-GRAD] Issues detected:", file=sys.stderr)
        for iss in issues:
            print(iss, file=sys.stderr)
    return issues, grad_norms


# ─── 梯度分桶直方图 ─────────────────────────────────────
def gradient_histogram(model, n_bins=10):
    """将所有梯度按数量级分桶, 打印直方图"""
    if not _is_debug():
        return
    all_grads = []
    for p in model.parameters():
        if p.grad is not None:
            all_grads.append(p.grad.abs().flatten())
    if not all_grads:
        print("[PR-GRAD-HIST] No gradients computed.",
              file=sys.stderr)
        return
    flat = torch.cat(all_grads)
    log_grads = torch.log10(flat.clamp(min=1e-12))
    lo, hi = log_grads.min().item(), log_grads.max().item()
    edges = np.linspace(lo, hi, n_bins + 1)
    counts = []
    for i in range(n_bins):
        mask = ((log_grads >= edges[i]) &
                (log_grads < edges[i + 1]))
        counts.append(mask.sum().item())
    total = sum(counts) or 1
    print(f"[PR-GRAD-HIST] log10 gradient distribution:",
          file=sys.stderr)
    for i in range(n_bins):
        bar_len = int(counts[i] / total * 40)
        bar = "#" * bar_len
        print(f"  [{edges[i]:+6.2f}, {edges[i+1]:+6.2f}): "
              f"{counts[i]:8d} ({counts[i]/total*100:5.1f}%) {bar}",
              file=sys.stderr)


# ─── 权重差异对比 ────────────────────────────────────────
def weight_diff(state_a, state_b, top_k=6):
    """对比两个state_dict间的参数变化量"""
    diffs = []
    for key in state_a:
        if key in state_b:
            delta = (state_a[key].float() -
                     state_b[key].float()).norm().item()
            diffs.append((delta, key))
    diffs.sort(key=lambda x: -x[0])
    print(f"\n[PR-WEIGHT-DIFF] top-{top_k} changed:",
          file=sys.stderr)
    for delta, key in diffs[:top_k]:
        print(f"  {key:55s} delta={delta:.6f}",
              file=sys.stderr)
    frozen = [key for delta, key in diffs if delta < 1e-10]
    if frozen:
        print(f"  !! {len(frozen)} params appear frozen",
              file=sys.stderr)
    return diffs


# ─── 数据流检查点 ────────────────────────────────────────
def dataflow_checkpoint(name, tensor, expected_shape=None):
    """在数据流关键节点插入: shape验证 + 统计"""
    if not _is_debug():
        return
    prefix = f"[PR-FLOW:{name}]"
    if expected_shape is not None:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"{prefix} !!SHAPE MISMATCH dim[{i}]: "
                      f"expected {e} got {a}",
                      file=sys.stderr)
    _dbg(name, tensor, "flow")


# ─── 结构体状态dump（Prism特有）─────────────────────────
def dump_struct_state(label, **kwargs):
    """打印当前所有传入的结构体/tensor状态, 用于断点调试"""
    if not _is_debug():
        return
    print(f"\n[PR-STRUCT:{label}] === State Dump ===",
          file=sys.stderr)
    record = {"ts": time.time(), "tag": f"struct_{label}",
              "fields": {}}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            info = (f"Tensor shape={list(v.shape)} "
                    f"dtype={v.dtype} "
                    f"range=[{v.min().item():.4f},"
                    f"{v.max().item():.4f}] "
                    f"mean={v.mean().item():.4f}")
            print(f"  {k:30s}: {info}", file=sys.stderr)
            record["fields"][k] = {
                "type": "tensor", "shape": list(v.shape),
                "min": v.min().item(), "max": v.max().item()}
        elif isinstance(v, (list, tuple)):
            print(f"  {k:30s}: {type(v).__name__} "
                  f"len={len(v)}", file=sys.stderr)
            record["fields"][k] = {
                "type": type(v).__name__, "len": len(v)}
        elif isinstance(v, (int, float, bool, str)):
            print(f"  {k:30s}: {v}", file=sys.stderr)
            record["fields"][k] = {"type": "scalar",
                                    "value": v}
        else:
            print(f"  {k:30s}: {type(v).__name__}",
                  file=sys.stderr)
    print(f"[PR-STRUCT:{label}] === End Dump ===\n",
          file=sys.stderr)
    _diag_write(record)


# ─── 训练速度计时器 ──────────────────────────────────────
class PerfTimer:
    """细粒度训练阶段计时器"""
    def __init__(self):
        self._timers = OrderedDict()
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self._starts:
            elapsed = (time.perf_counter() -
                       self._starts.pop(name))
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)

    def report(self):
        if not _is_debug():
            return
        print(f"\n[PR-PERF] Timing breakdown:",
              file=sys.stderr)
        for name, times in self._timers.items():
            avg_ms = np.mean(times) * 1000
            total_s = np.sum(times)
            print(f"  {name:35s} avg={avg_ms:8.2f}ms "
                  f"total={total_s:.2f}s ({len(times)} calls)",
                  file=sys.stderr)


# ─── 对比学习温度追踪 (Prism特有) ────────────────────────
class ContrastiveTemperatureTracker:
    """记录对比学习温度退火过程, 用于验证温度schedule"""
    def __init__(self):
        self.history = []

    def record(self, step, temperature, contrastive_loss_val):
        self.history.append((step, temperature, contrastive_loss_val))
        if _is_debug() and len(self.history) % 30 == 0:
            print(f"[PR-CONTRAST] step={step} tau={temperature:.6f} "
                  f"cl_loss={contrastive_loss_val:.6f}",
                  file=sys.stderr)

    def report(self):
        if not self.history:
            return
        temps = [t for _, t, _ in self.history]
        losses = [l for _, _, l in self.history]
        print(f"\n[PR-CONTRAST] Summary: "
              f"tau_init={temps[0]:.4f} tau_final={temps[-1]:.4f} "
              f"cl_loss_init={losses[0]:.4f} cl_loss_final={losses[-1]:.4f} "
              f"steps={len(self.history)}", file=sys.stderr)


# ─── 视角融合权重追踪 (Prism特有) ────────────────────────
class ViewFusionTracker:
    """追踪空间/时间/频率三视角的融合权重变化"""
    def __init__(self):
        self.history = []

    def record(self, step, spatial_w, temporal_w, freq_w):
        self.history.append((step, spatial_w, temporal_w, freq_w))
        if _is_debug() and len(self.history) % 30 == 0:
            print(f"[PR-VIEW] step={step} spatial={spatial_w:.4f} "
                  f"temporal={temporal_w:.4f} freq={freq_w:.4f}",
                  file=sys.stderr)

    def report(self):
        if not self.history:
            return
        last = self.history[-1]
        print(f"\n[PR-VIEW] Final weights: spatial={last[1]:.4f} "
              f"temporal={last[2]:.4f} freq={last[3]:.4f} "
              f"steps={len(self.history)}", file=sys.stderr)
