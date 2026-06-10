"""
walpurgis — D2STGNN 统一变体 (Cascade+Reverie融合)
算法改写 (~20%):
  - 级联残差学习 (Cascade Residual): 每层输出直接跳连到输出聚合
  - 动态深度选择 (Dynamic Depth Gate): sigmoid门控决定实际用几层
  - SE通道注意力 (Squeeze-and-Excitation): 通道维度自适应加权
  - 级联感知损失 (Cascade-Aware Loss): 逐层贡献加权
  - ReduceLROnPlateau调度器 (替代MultiStepLR)
  - 梯度裁剪自适应 (adaptive gradient clipping)

全局调试: WALPURGIS_DEBUG=1 开启
诊断日志: CASCADE_DIAG_LOG=<path> 写入JSONL
"""
import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("WALPURGIS_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()
_DIAG_LOG_PATH = os.environ.get("CASCADE_DIAG_LOG", "")


def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断记录到JSONL（若设置了CASCADE_DIAG_LOG）"""
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
    prefix = f"[CAS-DBG:{tag}]"
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
    print(f"[CAS-SNAPSHOT] epoch={epoch} step={step} "
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
        print(f"\n[CAS-ACTIVATION] {len(self.stats)} layers tracked:",
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
        print("[CAS-GRAD] Issues detected:", file=sys.stderr)
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
        print("[CAS-GRAD-HIST] No gradients computed.",
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
    print(f"[CAS-GRAD-HIST] log10 gradient distribution:",
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
    print(f"\n[CAS-WEIGHT-DIFF] top-{top_k} changed:",
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
    prefix = f"[CAS-FLOW:{name}]"
    if expected_shape is not None:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"{prefix} !!SHAPE MISMATCH dim[{i}]: "
                      f"expected {e} got {a}",
                      file=sys.stderr)
    _dbg(name, tensor, "flow")


# ─── 结构体状态dump ──────────────────────────────────────
def dump_struct_state(label, **kwargs):
    """打印当前所有传入的结构体/tensor状态, 用于断点调试"""
    if not _is_debug():
        return
    print(f"\n[CAS-STRUCT:{label}] === State Dump ===",
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
    print(f"[CAS-STRUCT:{label}] === End Dump ===\n",
          file=sys.stderr)
    _diag_write(record)


# ─── Cascade特有: 深度门控状态追踪 ─────────────────────
class DepthGateTracker:
    """记录Dynamic Depth Gate每层的激活概率, 追踪有效深度"""
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.gate_history = [[] for _ in range(num_layers)]
        self._step = 0

    def record(self, layer_idx, gate_value):
        if isinstance(gate_value, torch.Tensor):
            gate_value = gate_value.mean().item()
        self.gate_history[layer_idx].append(gate_value)

    def tick(self):
        self._step += 1

    def report(self):
        if not _is_debug():
            return
        print(f"\n[CAS-DEPTH] Dynamic Depth Gate Summary "
              f"({self._step} steps):", file=sys.stderr)
        for i in range(self.num_layers):
            vals = self.gate_history[i]
            if vals:
                avg = np.mean(vals)
                std = np.std(vals)
                recent = np.mean(vals[-20:]) if len(vals) >= 20 else avg
                status = "ACTIVE" if avg > 0.5 else "DORMANT"
                print(f"  Layer {i}: avg_gate={avg:.4f} "
                      f"std={std:.4f} recent={recent:.4f} "
                      f"[{status}]", file=sys.stderr)

    def effective_depth(self):
        """返回平均有效深度 (gate > 0.5 的层数)"""
        active = 0
        for i in range(self.num_layers):
            vals = self.gate_history[i]
            if vals and np.mean(vals[-20:] if len(vals) >= 20 else vals) > 0.5:
                active += 1
        return active


# ─── Cascade特有: SE通道注意力追踪 ────────────────────────
class SETracker:
    """追踪SE attention的通道权重分布"""
    def __init__(self):
        self.weight_stats = []

    def record(self, se_weights):
        if isinstance(se_weights, torch.Tensor):
            self.weight_stats.append({
                'mean': se_weights.mean().item(),
                'std': se_weights.std().item(),
                'max': se_weights.max().item(),
                'min': se_weights.min().item(),
                'entropy': -(se_weights * torch.log(
                    se_weights.clamp(min=1e-8))).sum(-1).mean().item()
            })

    def report(self):
        if not _is_debug() or not self.weight_stats:
            return
        print(f"\n[CAS-SE] Squeeze-Excitation Channel Attention "
              f"({len(self.weight_stats)} records):", file=sys.stderr)
        avg_entropy = np.mean([s['entropy'] for s in self.weight_stats])
        avg_std = np.mean([s['std'] for s in self.weight_stats])
        print(f"  avg_entropy={avg_entropy:.4f} "
              f"avg_channel_std={avg_std:.6f}", file=sys.stderr)
        if avg_std < 1e-4:
            print("  !! SE weights nearly uniform — "
                  "attention may be ineffective", file=sys.stderr)


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
        print(f"\n[CAS-PERF] Timing breakdown:",
              file=sys.stderr)
        for name, times in self._timers.items():
            avg_ms = np.mean(times) * 1000
            total_s = np.sum(times)
            print(f"  {name:35s} avg={avg_ms:8.2f}ms "
                  f"total={total_s:.2f}s ({len(times)} calls)",
                  file=sys.stderr)


# ─── Cascade特有: 级联残差贡献追踪 ────────────────────────
class CascadeResidualTracker:
    """追踪每层级联残差对最终输出的贡献比例"""
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.contribution_history = [[] for _ in range(num_layers)]

    def record(self, layer_idx, contribution_norm):
        if isinstance(contribution_norm, torch.Tensor):
            contribution_norm = contribution_norm.item()
        self.contribution_history[layer_idx].append(contribution_norm)

    def report(self):
        if not _is_debug():
            return
        print(f"\n[CAS-RESIDUAL] Cascade Residual Contributions:",
              file=sys.stderr)
        total_norms = []
        for i in range(self.num_layers):
            vals = self.contribution_history[i]
            if vals:
                avg = np.mean(vals)
                total_norms.append(avg)
        total = sum(total_norms) or 1.0
        for i, avg in enumerate(total_norms):
            pct = avg / total * 100
            bar = "#" * int(pct / 2)
            print(f"  Layer {i}: norm={avg:.4f} "
                  f"({pct:.1f}%) {bar}", file=sys.stderr)


# ─── Phase 3: 空间自注意力追踪 ───────────────────────────
class SpatialAttnTracker:
    """追踪SpatialSelfAttention的门控强度与注意力熵轨迹
    断点诊断要点:
      - gate应从~0.12逐渐上升; 若长期<0.05说明模块被骨干抑制(无效)
      - entropy接近log(N)=均匀分布(没学到结构); 接近0=单点坍缩(过拟合某节点)
      - 健康区间: 大约0.3*log(N) ~ 0.85*log(N)
    """
    def __init__(self, num_nodes=None):
        self.records = []
        self.num_nodes = num_nodes

    def record(self, gate, entropy, out_norm=None):
        if isinstance(gate, torch.Tensor):
            gate = gate.item()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.item()
        self.records.append({
            'gate': gate, 'entropy': entropy,
            'out_norm': (out_norm.item()
                         if isinstance(out_norm, torch.Tensor)
                         else out_norm)})

    def report(self):
        if not _is_debug() or not self.records:
            return
        gates = [r['gate'] for r in self.records]
        ents = [r['entropy'] for r in self.records]
        recent_g = np.mean(gates[-20:])
        recent_e = np.mean(ents[-20:])
        print(f"\n[CAS-SPATTN] Spatial Self-Attention "
              f"({len(self.records)} records):", file=sys.stderr)
        print(f"  gate: first={gates[0]:.4f} "
              f"recent={recent_g:.4f} "
              f"trend={'UP' if recent_g > gates[0] else 'DOWN'}",
              file=sys.stderr)
        if self.num_nodes:
            max_ent = float(np.log(self.num_nodes))
            ratio = recent_e / max_ent
            print(f"  entropy: recent={recent_e:.4f} "
                  f"/ log(N)={max_ent:.4f} (ratio={ratio:.2f})",
                  file=sys.stderr)
            if ratio > 0.95:
                print("  !! attention nearly UNIFORM — "
                      "spatial structure not learned yet",
                      file=sys.stderr)
            elif ratio < 0.10:
                print("  !! attention COLLAPSED to few nodes — "
                      "check dropout/lr", file=sys.stderr)
        if recent_g < 0.05:
            print("  !! gate suppressed (<0.05) — module "
                  "effectively disabled by backbone",
                  file=sys.stderr)
        _diag_write({
            "ts": time.time(), "tag": "spatial_attn_summary",
            "gate_first": gates[0], "gate_recent": recent_g,
            "entropy_recent": recent_e})
