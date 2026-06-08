"""walpurgis_meridian — D2STGNN Meridian variant.
Meridian: Adaptive kernel temporal conv, Lanczos graph spectral filtering,
dual-pathway residual gating, learnable positional frequency encoding,
annealed focal loss, AdamW+ReduceLROnPlateau, fBm synth data, stratified split.

Algorithm changes vs upstream (~20%):
  - EstimationGate: bilinear interaction + Mish activation (upstream: concat+ReLU+sigmoid)
  - ResidualDecomp: gated residual with learnable interpolation (upstream: subtract+LayerNorm)
  - DifBlock: Lanczos spectral filter replaces k-hop polynomial (upstream: multi-hop matmul)
  - InhBlock: GRU with highway gate + relative position bias (upstream: vanilla GRU)
  - D2STGNN: adaptive kernel sizes per layer + geometric mean aggregation (upstream: fixed k_t + sum)
  - Output: GEGLU projection head (upstream: ReLU+Linear)
  - Loss: annealed focal regression (upstream: masked MAE)
  - Optimizer: AdamW + ReduceLROnPlateau (upstream: Adam + MultiStepLR)
  - EarlyStopping: trend+curvature aware (upstream: simple patience)
"""
import os, sys, json, time, torch

_MER_DEBUG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'

def _is_debug():
    return _MER_DEBUG

def _dbg(tag, tensor_or_val, module="meridian"):
    """Debug print: tensor stats or scalar values."""
    if not _MER_DEBUG:
        return
    if isinstance(tensor_or_val, torch.Tensor):
        t = tensor_or_val.detach().float()
        if t.numel() == 0:
            print(f"[MER:{tag}@{module}] EMPTY shape={list(t.shape)}", file=sys.stderr)
            return
        nan_ct = torch.isnan(t).sum().item()
        inf_ct = torch.isinf(t).sum().item()
        sp = (t == 0).float().mean().item()
        msg = (f"[MER:{tag}@{module}] shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f}")
        if nan_ct: msg += f" ***NaN={nan_ct}***"
        if inf_ct: msg += f" ***Inf={inf_ct}***"
        if sp > 0.95: msg += f" ***SPARSE({sp:.1%})***"
    else:
        msg = f"[MER:{tag}@{module}] value={tensor_or_val}"
    print(msg, file=sys.stderr)


def snapshot_model(model, epoch=0, step=0):
    """Print full model parameter snapshot for debugging."""
    if not _MER_DEBUG:
        return
    print(f"\n[MER] === Model Snapshot (epoch={epoch}, step={step}) ===", file=sys.stderr)
    recs = []
    for name, p in model.named_parameters():
        info = {"name": name, "shape": list(p.shape),
                "requires_grad": p.requires_grad,
                "mean": round(p.data.float().mean().item(), 6),
                "std": round(p.data.float().std().item(), 6),
                "norm": round(p.data.float().norm().item(), 4)}
        if p.grad is not None:
            g = p.grad.detach().float()
            info["grad_norm"] = round(g.norm().item(), 6)
            info["grad_max"] = round(g.abs().max().item(), 6)
        recs.append(info)
    for r in recs:
        print(f"  {r}", file=sys.stderr)
    print(f"[MER] === End Snapshot ({len(recs)} params) ===\n", file=sys.stderr)


class ActivationTracker:
    """Hook-based activation monitor for all layers."""
    def __init__(self):
        self.records = {}
        self.handles = []

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                t = out.detach().float()
                self.records[name] = {
                    "shape": list(t.shape),
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "min": t.min().item(),
                    "max": t.max().item(),
                    "nan": torch.isnan(t).sum().item(),
                    "inf": torch.isinf(t).sum().item(),
                }
        return fn

    def report(self):
        print(f"\n[MER] === Activation Report ({len(self.records)} layers) ===", file=sys.stderr)
        for name, info in sorted(self.records.items()):
            flags = ""
            if info["nan"] > 0: flags += " ***NaN***"
            if info["inf"] > 0: flags += " ***Inf***"
            if info["std"] < 1e-6: flags += " ***DEAD***"
            print(f"  {name}: mean={info['mean']:.4f} std={info['std']:.4f} "
                  f"range=[{info['min']:.4f},{info['max']:.4f}]{flags}", file=sys.stderr)
        print("[MER] === End Activation Report ===\n", file=sys.stderr)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def register_activation_hooks(model):
    """Attach activation tracking hooks to all named modules."""
    tracker = ActivationTracker()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(tracker._hook(name))
            tracker.handles.append(h)
    return tracker


def gradient_health_check(model):
    """Check gradient statistics across all parameters."""
    if not _MER_DEBUG:
        return
    print(f"\n[MER] === Gradient Health Check ===", file=sys.stderr)
    total_params = 0
    zero_grad = 0
    nan_grad = 0
    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_params += 1
            gn = g.norm().item()
            if gn == 0:
                zero_grad += 1
            if torch.isnan(g).any():
                nan_grad += 1
                print(f"  *** NaN grad: {name}", file=sys.stderr)
            max_grad = max(max_grad, g.abs().max().item())
    print(f"  Total: {total_params} | ZeroGrad: {zero_grad} | NaNGrad: {nan_grad} | MaxGrad: {max_grad:.6f}",
          file=sys.stderr)
    print("[MER] === End Gradient Check ===\n", file=sys.stderr)


class LRTracker:
    """Track and print learning rate changes."""
    def __init__(self):
        self.history = []

    def log(self, epoch, lr):
        self.history.append((epoch, lr))
        if _MER_DEBUG:
            print(f"[MER:lr_track] epoch={epoch} lr={lr:.8f}", file=sys.stderr)

    def summary(self):
        if not self.history:
            return
        print(f"\n[MER] LR History ({len(self.history)} entries):", file=sys.stderr)
        for ep, lr in self.history:
            print(f"  epoch {ep}: {lr:.8f}", file=sys.stderr)
