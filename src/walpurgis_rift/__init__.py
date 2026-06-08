"""
walpurgis_rift — D2STGNN Rift变体
算法改写 (~20%):
  - Split-Recombine注意力: hidden分K组独立处理后重组
  - FFT域特征增强: 频域特征与时域特征concat后投影
  - Polynomial decay learning rate: lr = base_lr * (1 - t/T)^power
  - SiLU激活 + RMSNorm输出头 + 频域残差旁路
  - Split-Recombine decouple层: dif/inh各自分组处理后交叉拼接
  - 频谱正则化 (spectral regularization on forecast)
  - Log-cosh自适应混合损失

全局调试: RIFT_DEBUG=1 开启
诊断日志: RIFT_DIAG_LOG=<path> 写入JSONL
"""
import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("RIFT_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()
_DIAG_LOG_PATH = os.environ.get("RIFT_DIAG_LOG", "")


def _is_debug(module_name=""):
    # 动态检查环境变量, 支持运行时设置RIFT_DEBUG=1
    env = os.environ.get("RIFT_DEBUG", "0")
    if env.strip() == "1":
        return True
    if _DEBUG_ALL:
        return True
    return module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断记录到JSONL（若设置了RIFT_DIAG_LOG）"""
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
    prefix = f"[RF-DBG:{tag}]"
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
    print(f"[RF-SNAPSHOT] epoch={epoch} step={step} "
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


# ─── 分组统计追踪 (Rift特有: 追踪Split-Recombine各组状态) ───
class SplitGroupTracker:
    """追踪Split-Recombine中各组的激活值统计, 检测组间不均衡"""
    def __init__(self, num_groups):
        self.num_groups = num_groups
        self.group_stats = defaultdict(list)

    def record(self, group_idx, tensor):
        self.group_stats[group_idx].append({
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'norm': tensor.norm().item(),
        })

    def report(self):
        if not _is_debug():
            return
        print(f"\n[RF-SPLIT] Group balance report ({self.num_groups} groups):",
              file=sys.stderr)
        means = []
        for g in range(self.num_groups):
            records = self.group_stats.get(g, [])
            if records:
                avg_mean = np.mean([r['mean'] for r in records])
                avg_std = np.mean([r['std'] for r in records])
                avg_norm = np.mean([r['norm'] for r in records])
                means.append(avg_norm)
                print(f"  group[{g}]: avg_mean={avg_mean:.6f} "
                      f"avg_std={avg_std:.6f} avg_norm={avg_norm:.4f}",
                      file=sys.stderr)
        if len(means) >= 2:
            imbalance = max(means) / (min(means) + 1e-8)
            if imbalance > 3.0:
                print(f"  !! GROUP IMBALANCE: ratio={imbalance:.2f}",
                      file=sys.stderr)

    def clear(self):
        self.group_stats.clear()


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
        print(f"\n[RF-ACTIVATION] {len(self.stats)} layers tracked:",
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
        print("[RF-GRAD] Issues detected:", file=sys.stderr)
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
        print("[RF-GRAD-HIST] No gradients computed.",
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
    print(f"[RF-GRAD-HIST] log10 gradient distribution:",
          file=sys.stderr)
    for i in range(n_bins):
        bar_len = int(counts[i] / total * 40)
        bar = "#" * bar_len
        print(f"  [{edges[i]:+6.2f}, {edges[i+1]:+6.2f}): "
              f"{counts[i]:8d} ({counts[i]/total*100:5.1f}%) {bar}",
              file=sys.stderr)


# ─── FFT频谱监控 (Rift特有) ─────────────────────────────
def fft_spectrum_monitor(tensor, tag="", top_k=5):
    """监控tensor的FFT频谱分布, 检测频域特征质量"""
    if not _is_debug():
        return
    if tensor.dim() < 2:
        return
    flat = tensor.detach().float().reshape(-1, tensor.shape[-1])
    spec = torch.fft.rfft(flat, dim=-1).abs()
    energy = spec.mean(dim=0)
    total_energy = energy.sum().item()
    if total_energy < 1e-10:
        print(f"[RF-FFT:{tag}] spectrum DEAD (total_energy={total_energy:.2e})",
              file=sys.stderr)
        return
    normalized = energy / total_energy
    top_freqs = torch.topk(normalized, min(top_k, len(normalized)))
    dc_ratio = energy[0].item() / total_energy
    print(f"[RF-FFT:{tag}] DC_ratio={dc_ratio:.4f} "
          f"top_freq_idx={top_freqs.indices.tolist()} "
          f"top_energy={[f'{v:.4f}' for v in top_freqs.values.tolist()]}",
          file=sys.stderr)
    _diag_write({
        "ts": time.time(), "tag": f"fft_{tag}",
        "dc_ratio": dc_ratio,
        "total_energy": total_energy,
        "top_indices": top_freqs.indices.tolist(),
    })


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
    print(f"\n[RF-WEIGHT-DIFF] top-{top_k} changed:",
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
    prefix = f"[RF-FLOW:{name}]"
    if expected_shape is not None:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"{prefix} !!SHAPE MISMATCH dim[{i}]: "
                      f"expected {e} got {a}",
                      file=sys.stderr)
    _dbg(name, tensor, "flow")


# ─── 结构体状态dump ─────────────────────────────────────
def dump_struct_state(label, **kwargs):
    """打印当前所有传入的结构体/tensor状态, 用于断点调试"""
    if not _is_debug():
        return
    print(f"\n[RF-STRUCT:{label}] === State Dump ===",
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
    print(f"[RF-STRUCT:{label}] === End Dump ===\n",
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
        print(f"\n[RF-PERF] Timing breakdown:",
              file=sys.stderr)
        for name, times in self._timers.items():
            avg_ms = np.mean(times) * 1000
            total_s = np.sum(times)
            print(f"  {name:35s} avg={avg_ms:8.2f}ms "
                  f"total={total_s:.2f}s ({len(times)} calls)",
                  file=sys.stderr)


# ─── Polynomial LR追踪 (Rift特有) ───────────────────────
class PolyLRTracker:
    """记录每个step的polynomial decay学习率, 验证衰减曲线"""
    def __init__(self):
        self.history = []

    def record(self, step, lr):
        self.history.append((step, lr))
        if _is_debug() and len(self.history) % 50 == 0:
            print(f"[RF-LR] step={step} lr={lr:.8f}",
                  file=sys.stderr)

    def report(self):
        if not self.history:
            return
        lrs = [lr for _, lr in self.history]
        print(f"\n[RF-LR] Summary: "
              f"initial={lrs[0]:.6f} final={lrs[-1]:.6f} "
              f"min={min(lrs):.6f} max={max(lrs):.6f} "
              f"steps={len(lrs)}", file=sys.stderr)
        # 检查是否单调递减 (polynomial decay应该是)
        monotone = all(lrs[i] >= lrs[i+1] - 1e-10
                       for i in range(len(lrs)-1))
        print(f"  monotone_decreasing={monotone}",
              file=sys.stderr)
