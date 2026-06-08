"""
walpurgis_cathexis — D2STGNN Cathexis(精神贯注)变体
算法改写 (~20%):
  1. Bilinear + SiLU estimation gate (替代FC+ReLU+sigmoid)
  2. AdaIN residual decomposition (替代LayerNorm+ReLU)
  3. GATv2-style attention per hop graph conv (替代uniform matmul)
  4. Exponential kernel distance (替代inner-product softmax)
  5. Bernoulli sampling mask (替代element-wise adj mask)
  6. Doubly-stochastic Sinkhorn normalizer (替代row-normalize)
  7. Mamba-SSM + SwishGLU inherent (替代GRU+sinusoidal PE+Transformer)
  8. Exponential recency weighted aggregation (替代简单sum)
  9. Asymmetric Winsorized loss (替代masked_mae)
  10. AdamW+GradCentralization + WarmupCosine (替代Adam+MultiStepLR)
全局调试: CATHEXIS_DEBUG=1
"""
import os, sys, time, json, math, torch, numpy as np
from collections import defaultdict, OrderedDict

_DEBUG_ENV = os.environ.get("CATHEXIS_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = set(_DEBUG_ENV.split(",")) if not _DEBUG_ALL and _DEBUG_ENV != "0" else set()
_DIAG_LOG_PATH = os.environ.get("CATHEXIS_DIAG_LOG", "")

def _is_debug(module_name=""): return _DEBUG_ALL or module_name in _DEBUG_MODULES

def _diag_write(record):
    if not _DIAG_LOG_PATH: return
    try:
        with open(_DIAG_LOG_PATH, "a") as f: f.write(json.dumps(record, default=str) + "\n")
    except: pass

def _dbg(tag, tensor_or_msg, module=""):
    if not _is_debug(module) and not _DEBUG_ALL: return
    prefix = f"[CX-DBG:{tag}]"
    ts = time.time()
    if isinstance(tensor_or_msg, torch.Tensor):
        t = tensor_or_msg.detach().float()
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        sparse_frac = (t.abs() < 1e-8).float().mean().item()
        alert = ""
        if has_nan: alert += " !!NaN"
        if has_inf: alert += " !!Inf"
        if sparse_frac > 0.95: alert += f" SPARSE({sparse_frac*100:.1f}%)"
        msg = (f"shape={list(tensor_or_msg.shape)} dtype={tensor_or_msg.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f}{alert}")
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)
        _diag_write({"ts":ts,"tag":tag,"type":"tensor","shape":list(tensor_or_msg.shape),
                     "min":t.min().item(),"max":t.max().item(),"mean":t.mean().item(),
                     "std":t.std().item(),"nan":has_nan,"inf":has_inf,"sparse":sparse_frac})
    elif isinstance(tensor_or_msg, np.ndarray):
        a = tensor_or_msg
        print(f"{prefix} np shape={a.shape} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f}", file=sys.stderr, flush=True)
    else:
        print(f"{prefix} {tensor_or_msg}", file=sys.stderr, flush=True)
        _diag_write({"ts":ts,"tag":tag,"type":"msg","msg":str(tensor_or_msg)})

def snapshot_model(model, epoch=0, step=0, top_k=8):
    if not _is_debug(): return
    print(f"\n{'='*65}", file=sys.stderr)
    print(f"[CX-SNAPSHOT] epoch={epoch} step={step} params={sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)
    stats = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        d = p.data.float()
        alerts = []
        if d.std().item() < 1e-6: alerts.append("COLLAPSED")
        if torch.isnan(d).any(): alerts.append("NaN")
        g_norm = p.grad.norm().item() if p.grad is not None else 0.0
        stats.append((d.norm().item(), name, d.mean().item(), d.std().item(), g_norm, alerts))
    stats.sort(key=lambda x: -x[0])
    for norm_val, name, mean_val, std_val, g_norm, alerts in stats[:top_k]:
        print(f"  {name:50s} norm={norm_val:10.4f} μ={mean_val:+.6f} σ={std_val:.6f} |∇|={g_norm:.4f} {' '.join(alerts)}", file=sys.stderr)
    print(f"{'='*65}\n", file=sys.stderr)

class ActivationTracker:
    def __init__(self):
        self.stats = defaultdict(list); self._hooks = []; self._step_count = 0
    def _hook_fn(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                o = out.detach().float()
                self.stats[name].append({'step':self._step_count,'mean':o.mean().item(),'std':o.std().item(),
                    'abs_max':o.abs().max().item(),'dead_frac':(o.abs()<1e-7).float().mean().item(),
                    'q99':torch.quantile(o.abs().float(),0.99).item()})
        return hook
    def register(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear,torch.nn.GRUCell,torch.nn.LSTMCell,
                                torch.nn.MultiheadAttention,torch.nn.Conv1d,
                                torch.nn.LayerNorm,torch.nn.BatchNorm2d,torch.nn.BatchNorm1d)):
                self._hooks.append(mod.register_forward_hook(self._hook_fn(name)))
        return self
    def report(self, top_k=10):
        print(f"\n[CX-ACTIVATION] {len(self.stats)} layers tracked:", file=sys.stderr)
        items = []
        for name, records in self.stats.items():
            if records:
                items.append((np.mean([r['std'] for r in records]), name,
                              np.mean([r['dead_frac'] for r in records]),
                              max(r['q99'] for r in records)))
        items.sort(key=lambda x: -x[0])
        for avg_std, name, avg_dead, max_q99 in items[:top_k]:
            warn = ""
            if avg_dead > 0.5: warn += " !!DEAD"
            if max_q99 > 100: warn += f" !!HOT(q99={max_q99:.1f})"
            print(f"  {name:50s} avg_std={avg_std:.6f} dead={avg_dead:.3f} q99_max={max_q99:.4f}{warn}", file=sys.stderr)
    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear(); self.stats.clear()

def register_activation_hooks(model):
    tracker = ActivationTracker(); tracker.register(model); return tracker

def gradient_health_check(model, explode_thresh=80.0, vanish_thresh=1e-7):
    issues = []; grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            g_norm = p.grad.norm().item(); grad_norms.append((name, g_norm))
            if torch.isnan(p.grad).any(): issues.append(f"  ✗ NaN grad: {name}")
            elif g_norm > explode_thresh: issues.append(f"  ✗ EXPLODING: {name} (norm={g_norm:.2f})")
            elif g_norm < vanish_thresh: issues.append(f"  ✗ VANISHING: {name} (norm={g_norm:.2e})")
    if issues and _is_debug():
        print("[CX-GRAD] Issues:", file=sys.stderr)
        for iss in issues: print(iss, file=sys.stderr)
    return issues, grad_norms

def gradient_histogram(model, n_bins=8):
    if not _is_debug(): return
    all_grads = [p.grad.abs().flatten() for p in model.parameters() if p.grad is not None]
    if not all_grads: return
    flat = torch.cat(all_grads)
    log_grads = torch.log10(flat.clamp(min=1e-12))
    lo, hi = log_grads.min().item(), log_grads.max().item()
    edges = np.linspace(lo, hi, n_bins + 1)
    total = flat.numel()
    print(f"[CX-GRAD-HIST] log10 gradient distribution:", file=sys.stderr)
    for i in range(n_bins):
        cnt = ((log_grads >= edges[i]) & (log_grads < edges[i+1])).sum().item()
        bar = "█" * int(cnt / total * 40)
        print(f"  [{edges[i]:+6.2f},{edges[i+1]:+6.2f}): {cnt:8d} ({cnt/total*100:5.1f}%) {bar}", file=sys.stderr)

def weight_diff(state_a, state_b, top_k=5):
    diffs = [(( state_a[k].float() - state_b[k].float()).norm().item(), k) for k in state_a if k in state_b]
    diffs.sort(key=lambda x: -x[0])
    print(f"\n[CX-WEIGHT-DIFF] top-{top_k}:", file=sys.stderr)
    for delta, key in diffs[:top_k]:
        print(f"  {key:50s} Δ={delta:.6f}", file=sys.stderr)
    frozen = sum(1 for d, _ in diffs if d < 1e-10)
    if frozen: print(f"  !! {frozen} frozen params", file=sys.stderr)
    return diffs

def dataflow_checkpoint(name, tensor, expected_shape=None):
    if not _is_debug(): return
    if expected_shape:
        actual = list(tensor.shape)
        for i, (a, e) in enumerate(zip(actual, expected_shape)):
            if e is not None and a != e:
                print(f"[CX-FLOW:{name}] !!SHAPE dim[{i}]: expect {e} got {a}", file=sys.stderr)
    _dbg(name, tensor, "flow")

class PerfTimer:
    def __init__(self): self._timers = OrderedDict(); self._starts = {}
    def start(self, name): self._starts[name] = time.perf_counter()
    def stop(self, name):
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(name)
            self._timers.setdefault(name, []).append(elapsed)
    def report(self):
        if not _is_debug(): return
        print(f"\n[CX-PERF] Timing:", file=sys.stderr)
        for name, times in self._timers.items():
            print(f"  {name:30s} avg={np.mean(times)*1000:8.2f}ms total={np.sum(times):.2f}s ({len(times)} calls)", file=sys.stderr)

def struct_dump(model, label=""):
    if not _is_debug(): return
    print(f"\n[CX-STRUCT-DUMP] {label}", file=sys.stderr)
    for name, p in model.named_parameters():
        if p.requires_grad:
            d = p.data.float()
            grad_info = f"|∇|={p.grad.norm().item():.4f}" if p.grad is not None else "no_grad"
            print(f"  {name:55s} {str(list(p.shape)):20s} μ={d.mean().item():+.6f} σ={d.std().item():.6f} {grad_info}", file=sys.stderr)
