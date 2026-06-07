# walpurgis_solstice — D2STGNN Solstice variant
# 算法方向: Huber loss, FAVOR+ attention, ScaleNorm, RAdam+CosineWarmRestarts, LSTM, attention-weighted pooling, Mixup
import os, sys, torch

_SOLSTICE_DEBUG = os.environ.get('SOLSTICE_DEBUG', '0') == '1'

def _is_debug():
    return _SOLSTICE_DEBUG

def _dbg(tag, tensor_or_val, module="solstice"):
    """统一调试桩: 打印tensor统计或标量值, NaN/Inf自动告警"""
    if not _SOLSTICE_DEBUG:
        return
    if isinstance(tensor_or_val, torch.Tensor):
        t = tensor_or_val
        numel = t.numel()
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        zero_frac = (t == 0).float().mean().item()
        msg = (f"[SOL:{tag}@{module}] shape={list(t.shape)} dtype={t.dtype} "
               f"min={t.min().item():.6f} max={t.max().item():.6f} "
               f"mean={t.mean().item():.6f} std={t.std().item():.6f} "
               f"zero_frac={zero_frac:.2%}")
        if nan_count > 0:
            msg += f" *** ALERT NaN={nan_count}/{numel} ***"
        if inf_count > 0:
            msg += f" *** ALERT Inf={inf_count}/{numel} ***"
        if zero_frac > 0.95:
            msg += f" *** SPARSE({zero_frac:.1%}) ***"
    else:
        msg = f"[SOL:{tag}@{module}] value={tensor_or_val}"
    print(msg, file=sys.stderr)


def snapshot_model(model, epoch=0, step=0):
    """全参数快照: grad_norm降序排列, nan/冻结参数检测"""
    if not _SOLSTICE_DEBUG:
        return
    print(f"\n[SOL] === Model Snapshot (epoch={epoch}, step={step}) ===", file=sys.stderr)
    params = []
    frozen_count = 0
    for name, p in model.named_parameters():
        info = {"name": name, "shape": list(p.shape), "mean": p.data.mean().item(),
                "std": p.data.std().item(), "has_nan": bool(torch.isnan(p.data).any().item()),
                "numel": p.numel()}
        if p.grad is not None:
            info["grad_norm"] = p.grad.norm().item()
            info["grad_mean"] = p.grad.mean().item()
        else:
            frozen_count += 1
        if not p.requires_grad:
            frozen_count += 1
        params.append(info)
    params.sort(key=lambda x: x.get("grad_norm", 0), reverse=True)
    for p in params[:15]:
        nan_flag = " [NaN!]" if p["has_nan"] else ""
        grad_str = f"grad={p['grad_norm']:.6f}" if "grad_norm" in p else "no_grad"
        print(f"  {p['name']}: shape={p['shape']} mean={p['mean']:.6f} std={p['std']:.6f} {grad_str}{nan_flag}", file=sys.stderr)
    total_params = sum(p["numel"] for p in params)
    print(f"[SOL] total_params={total_params:,} frozen={frozen_count}", file=sys.stderr)
    print(f"[SOL] === End Snapshot ===\n", file=sys.stderr)


class _ActivationTracker:
    """Forward hook追踪每层activation统计, 检测死神经元"""
    def __init__(self):
        self.records = {}
        self.handles = []

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor) and out.numel() > 0:
                self.records[name] = {
                    "mean": out.mean().item(), "std": out.std().item(),
                    "zero_frac": (out.abs() < 1e-7).float().mean().item(),
                    "max_abs": out.abs().max().item()
                }
        return fn

    def check_dead(self, threshold=0.90):
        dead = {k: v for k, v in self.records.items() if v["zero_frac"] > threshold}
        return dead

    def report(self):
        print(f"\n[SOL] === Activation Report ({len(self.records)} layers) ===", file=sys.stderr)
        for name, r in sorted(self.records.items()):
            flag = " *** DEAD ***" if r["zero_frac"] > 0.9 else ""
            flag += " *** EXPLODING ***" if r["max_abs"] > 1e4 else ""
            print(f"  {name}: mean={r['mean']:.6f} std={r['std']:.6f} zero={r['zero_frac']:.2%} max_abs={r['max_abs']:.4f}{flag}", file=sys.stderr)
        print(f"[SOL] === End Report ===\n", file=sys.stderr)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def register_activation_hooks(model):
    """注册forward hooks追踪每层activation"""
    tracker = _ActivationTracker()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(tracker._hook(name))
            tracker.handles.append(h)
    return tracker


def gradient_health_check(model):
    """检测梯度健康: 爆炸(>100)/消失(<1e-7)/NaN三类"""
    if not _SOLSTICE_DEBUG:
        return
    issues = {"exploding": [], "vanishing": [], "nan": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.norm().item()
        if gn > 100:
            issues["exploding"].append((name, gn))
        elif gn < 1e-7:
            issues["vanishing"].append((name, gn))
        if torch.isnan(p.grad).any():
            issues["nan"].append(name)
    print(f"\n[SOL] === Gradient Health Check ===", file=sys.stderr)
    for category, items in issues.items():
        if items:
            print(f"  {category.upper()}: {len(items)} params", file=sys.stderr)
            for item in items[:5]:
                if isinstance(item, tuple):
                    print(f"    {item[0]} grad_norm={item[1]:.4f}", file=sys.stderr)
                else:
                    print(f"    {item}", file=sys.stderr)
    if not any(issues.values()):
        print("  All gradients healthy", file=sys.stderr)
    print(f"[SOL] === End Check ===\n", file=sys.stderr)


def weight_diff(state_a, state_b, top_k=10):
    """比较两个state_dict间参数变化量, 检测冻结参数"""
    if not _SOLSTICE_DEBUG:
        return
    diffs = []
    frozen = []
    for key in state_a:
        if key in state_b:
            delta = (state_a[key].float() - state_b[key].float()).norm().item()
            denom = state_a[key].float().norm().item()
            rel = delta / max(denom, 1e-8)
            if delta < 1e-10:
                frozen.append(key)
            diffs.append((key, delta, rel))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[SOL] === Weight Diff (top-{top_k}) ===", file=sys.stderr)
    for k, d, r in diffs[:top_k]:
        print(f"  {k}: delta={d:.6f} rel={r:.4%}", file=sys.stderr)
    if frozen:
        print(f"  FROZEN: {len(frozen)} params unchanged", file=sys.stderr)
    print(f"[SOL] === End Diff ===\n", file=sys.stderr)
