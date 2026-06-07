"""walpurgis_tempest — D2STGNN Tempest variant.
Tempest: Focal regression, Squeeze-Excitation gate, PowerNorm+Swish decomp,
SpectralNorm+Chebyshev conv, cross-attention backcast, Mahalanobis distance,
straight-through Bernoulli mask, Sinkhorn normalizer, MinGRU+relative-bias
Transformer, conditional PE, NAS-style layer agg, Swish+LayerNorm output,
Adan+OneCycleLR optimizer, fBm synth, stratified shuffle."""
import os, sys, torch

_TEMPEST_DEBUG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

def _is_debug():
    return _TEMPEST_DEBUG

def _dbg(tag, tensor_or_val, module="tempest"):
    if not _TEMPEST_DEBUG:
        return
    if isinstance(tensor_or_val, torch.Tensor):
        t = tensor_or_val.detach().float()
        if t.numel() == 0:
            print(f"[TEM:{tag}@{module}] EMPTY shape={list(t.shape)}", file=sys.stderr)
            return
        nan_ct = torch.isnan(t).sum().item()
        inf_ct = torch.isinf(t).sum().item()
        sp = (t == 0).float().mean().item()
        msg = (f"[TEM:{tag}@{module}] shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f}")
        if nan_ct: msg += f" ***NaN={nan_ct}***"
        if inf_ct: msg += f" ***Inf={inf_ct}***"
        if sp > 0.95: msg += f" ***SPARSE({sp:.1%})***"
    else:
        msg = f"[TEM:{tag}@{module}] value={tensor_or_val}"
    print(msg, file=sys.stderr)

def snapshot_model(model, epoch=0, step=0):
    if not _TEMPEST_DEBUG: return
    print(f"\n[TEM] === Model Snapshot (epoch={epoch}, step={step}) ===", file=sys.stderr)
    recs = []
    for name, p in model.named_parameters():
        info = {"name": name, "shape": list(p.shape),
                "mean": p.data.mean().item(), "std": p.data.std().item(),
                "has_nan": bool(torch.isnan(p.data).any().item())}
        if p.data.std().item() < 1e-6: info["diag"] = "COLLAPSED_SCALE"
        if abs(p.data.mean().item()) > 3.0 * max(p.data.std().item(), 1e-8):
            info["diag"] = info.get("diag","") + " MEAN_DRIFT"
        if p.grad is not None:
            gn = p.grad.norm().item()
            info["grad_norm"] = gn
            if gn > 50: info["diag"] = info.get("diag","") + " GRAD_SPIKE"
        recs.append(info)
    recs.sort(key=lambda x: x.get("grad_norm", 0), reverse=True)
    for r in recs[:12]:
        d = f" [{r['diag'].strip()}]" if r.get("diag","").strip() else ""
        g = f"{r['grad_norm']:.4f}" if "grad_norm" in r else "N/A"
        print(f"  {r['name']}: shape={r['shape']} mean={r['mean']:.6f} std={r['std']:.6f} grad={g}{d}", file=sys.stderr)
    print(f"[TEM] === End Snapshot ===\n", file=sys.stderr)

class _ActivationTracker:
    def __init__(self):
        self.records = {}
        self.handles = []
    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor) and out.numel() > 0:
                self.records[name] = {
                    "mean": out.detach().float().mean().item(),
                    "std": out.detach().float().std().item(),
                    "zero_frac": (out.detach() == 0).float().mean().item()}
        return fn
    def report(self):
        print(f"\n[TEM] === Activation Report ({len(self.records)} layers) ===", file=sys.stderr)
        for name, r in self.records.items():
            flag = " ***DEAD***" if r["zero_frac"] > 0.9 else ""
            print(f"  {name}: mean={r['mean']:.6f} std={r['std']:.6f} zero={r['zero_frac']:.2%}{flag}", file=sys.stderr)
        print(f"[TEM] === End Report ===\n", file=sys.stderr)
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def register_activation_hooks(model):
    tracker = _ActivationTracker()
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            h = mod.register_forward_hook(tracker._hook(name))
            tracker.handles.append(h)
    return tracker

def gradient_health_check(model):
    if not _TEMPEST_DEBUG: return
    print(f"\n[TEM] === Gradient Health Check ===", file=sys.stderr)
    issues = 0
    for name, p in model.named_parameters():
        if p.grad is None: continue
        gn = p.grad.norm().item()
        if gn > 100:
            print(f"  EXPLODING: {name} grad_norm={gn:.2f}", file=sys.stderr); issues += 1
        elif gn < 1e-7 and p.requires_grad:
            print(f"  VANISHING: {name} grad_norm={gn:.2e}", file=sys.stderr); issues += 1
        if torch.isnan(p.grad).any():
            print(f"  NaN GRAD: {name}", file=sys.stderr); issues += 1
    if not issues: print(f"  All gradients healthy.", file=sys.stderr)
    print(f"[TEM] === End Check ({issues} issues) ===\n", file=sys.stderr)

def weight_diff(state_a, state_b, top_k=5):
    if not _TEMPEST_DEBUG: return
    diffs = []
    for k in state_a:
        if k in state_b:
            d = (state_a[k].float() - state_b[k].float()).norm().item()
            diffs.append((k, d))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[TEM] === Weight Diff (top {top_k}) ===", file=sys.stderr)
    for name, delta in diffs[:top_k]:
        print(f"  {name}: delta_norm={delta:.6f}", file=sys.stderr)
    frozen = [n for n, d in diffs if d < 1e-10]
    if frozen: print(f"  FROZEN ({len(frozen)}): {frozen[:5]}", file=sys.stderr)
    print(f"[TEM] === End Diff ===\n", file=sys.stderr)
