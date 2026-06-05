"""
walpurgis_walking — D2STGNN 移植版 (LLM4Walking)
==========================================================
Upstream: D2STGNN (Shao et al., VLDB 2022)
改动 ≥20% 的要点:
  - 损失: Huber(δ=5) + log-cosh(0.3) 混合, 替代纯MAE
  - 估计门: SiLU激活 + 双头投影 + GroupNorm + 可学习温度
  - 残差分解: Mish激活 + 可学习残差缩放 α(init=0.9)
  - 扩散卷积: InstanceNorm替代BN, gconv内加skip, dropout衰减
  - 扩散块: 3层MLP backcast + GELU + 残差门控
  - 动态图: 3-head QK注意力 + InstanceNorm + cosine辅助
  - 图掩码: softplus阈值 + 温度退火 + 对角清零
  - 归一化: 对称 D^{-1/2}AD^{-1/2} + 高阶指数衰减 λ^k
  - GRU: 步间RMSNorm + gradient checkpoint
  - Transformer: Rotary PE + attention entropy监控
  - 训练器: 自适应p90梯度裁剪 + warmup-cosine LR
  - MAPE: 分母floor clamp 5e-6

调试系统: WALPURGIS_DEBUG=1 开启全量 / =model,trainer 按模块开
"""

import os as _os

_DEBUG_ENV = _os.environ.get("WALPURGIS_DEBUG", "")
_DEBUG_ALL = (_DEBUG_ENV == "1")
_DEBUG_TAGS = set(_DEBUG_ENV.split(",")) if _DEBUG_ENV and not _DEBUG_ALL else set()


def _dbg(tag: str, msg: str, **kw):
    """统一调试入口. NaN/Inf 即使 tag 关闭也会告警."""
    import torch as _th
    if not (_DEBUG_ALL or tag in _DEBUG_TAGS):
        for name, t in kw.items():
            if isinstance(t, _th.Tensor):
                if t.isnan().any() or t.isinf().any():
                    print(f"[ALERT:{tag}] {msg} | {name}: "
                          f"nan={t.isnan().sum().item()} inf={t.isinf().sum().item()} "
                          f"shape={tuple(t.shape)}", flush=True)
        return
    parts = [f"[{tag}] {msg}"]
    for name, t in kw.items():
        if isinstance(t, _th.Tensor):
            ne = t.numel()
            sp = (t.abs() < 1e-8).sum().item() / max(ne, 1)
            s = (f"{name}: shape={tuple(t.shape)} "
                 f"min={t.min().item():.4g} max={t.max().item():.4g} "
                 f"mean={t.float().mean().item():.4g} "
                 f"nan={t.isnan().sum().item()}")
            if sp > 0.95:
                s += f" SPARSE({sp:.1%})"
            parts.append(s)
        elif isinstance(t, (float, int)):
            parts.append(f"{name}={t}")
        else:
            parts.append(f"{name}: {type(t).__name__}")
    print(" | ".join(parts), flush=True)


def snapshot_model(model, epoch=None, step=None, top_k=8):
    """全模型参数+梯度快照，类似 pdb 里 print(locals())."""
    import torch as _th
    hdr = "[snapshot]"
    if epoch is not None:
        hdr += f" ep={epoch}"
    if step is not None:
        hdr += f" step={step}"

    recs = []
    total_p = train_p = has_g = zero_g = 0
    nan_list = []

    for nm, p in model.named_parameters():
        n = p.numel()
        total_p += n
        if p.requires_grad:
            train_p += n
        gn = None
        if p.grad is not None:
            has_g += 1
            gn = p.grad.data.norm().item()
            if p.grad.data.abs().max().item() < 1e-10:
                zero_g += 1
        if p.data.isnan().any():
            nan_list.append(nm)
        recs.append(dict(name=nm, shape=tuple(p.shape),
                         mean=p.data.float().mean().item(),
                         std=p.data.float().std().item(),
                         amax=p.data.float().abs().max().item(),
                         gn=gn))

    recs.sort(key=lambda r: -(r["gn"] or 0))

    # 检测初始化病态
    sick = []
    for r in recs:
        fl = []
        if r["std"] < 1e-6 and r["amax"] > 1e-10:
            fl.append("COLLAPSED")
        if r["std"] > 1e-8 and abs(r["mean"]) > 3 * r["std"]:
            fl.append("MEAN_DRIFT")
        if r["gn"] is not None and r["gn"] > 50:
            fl.append("GRAD_SPIKE")
        if fl:
            sick.append((r["name"], fl))

    print(f"\n{'━'*72}\n{hdr}  params={total_p:,}  trainable={train_p:,}  "
          f"has_grad={has_g}  zero_grad={zero_g}  nan={len(nan_list)}")
    if nan_list:
        print(f"  ⚠ NaN in: {nan_list[:5]}")
    if sick:
        for pn, pf in sick[:5]:
            print(f"  ⚠ {pn}: {', '.join(pf)}")
    print(f"{'─'*72}")
    for r in recs[:top_k]:
        g = f"{r['gn']:.4g}" if r['gn'] is not None else "-"
        print(f"  {r['name']:42s} {str(r['shape']):16s} "
              f"μ={r['mean']:+.4f} σ={r['std']:.4f} |max|={r['amax']:.4f} ∇={g}")
    if len(recs) > top_k:
        rem = [r["gn"] for r in recs[top_k:] if r["gn"] is not None]
        if rem:
            print(f"  ... {len(recs)-top_k} more, grad=[{min(rem):.4g}, {max(rem):.4g}]")
    print(f"{'━'*72}\n")
    return recs


class _ActTracker:
    """Forward hook 集合，持续记录每层激活统计."""
    def __init__(self):
        self._hooks, self._data = [], {}

    def _make_hook(self, name):
        def _h(mod, inp, out):
            import torch
            t = out if isinstance(out, torch.Tensor) else (
                out[0] if isinstance(out, tuple) and isinstance(out[0], torch.Tensor) else None)
            if t is None:
                return
            self._data[name] = dict(
                shape=tuple(t.shape),
                mean=t.float().mean().item(),
                std=t.float().std().item(),
                amax=t.float().abs().max().item(),
                zf=(t.abs() < 1e-7).float().mean().item(),
                nan=t.isnan().any().item())
        return _h

    def report(self):
        print(f"\n{'─'*68}\n[activations] {len(self._data)} layers")
        for nm, s in self._data.items():
            flag = " ⚠NaN" if s["nan"] else (" ⚠DEAD" if s["zf"] > 0.9 else "")
            print(f"  {nm:42s} {str(s['shape']):16s} "
                  f"μ={s['mean']:+.5f} σ={s['std']:.5f} dead={s['zf']:.1%}{flag}")
        print(f"{'─'*68}\n")

    def check_dead(self, th=0.9):
        return {n: s["zf"] for n, s in self._data.items() if s["zf"] > th}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._data.clear()


def register_activation_hooks(model, prefix=""):
    trk = _ActTracker()
    for nm, mod in model.named_modules():
        if nm:
            h = mod.register_forward_hook(trk._make_hook(f"{prefix}{nm}" if prefix else nm))
            trk._hooks.append(h)
    return trk


def gradient_health_check(model, explode=100.0, vanish=1e-7):
    import torch
    issues = []
    for nm, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.data.norm().item()
        if torch.isnan(p.grad).any():
            issues.append(("NaN", nm, gn))
        elif gn > explode:
            issues.append(("EXPLODE", nm, gn))
        elif gn < vanish and p.grad.data.abs().max().item() < vanish:
            issues.append(("VANISH", nm, gn))
    if issues:
        print(f"[grad_health] {len(issues)} issues:")
        for k, nm, v in issues:
            print(f"  {k:8s} {nm:40s} norm={v:.4g}")
    return issues


def weight_diff(model, prev_sd, top_k=10):
    """比较当前权重与之前的 state_dict, 显示变化最大的 top_k 参数."""
    diffs = []
    for nm, p in model.named_parameters():
        if nm in prev_sd:
            delta = (p.data - prev_sd[nm]).float().norm().item()
            scale = prev_sd[nm].float().norm().item()
            rel = delta / max(scale, 1e-10)
            diffs.append((nm, delta, rel))
    diffs.sort(key=lambda x: -x[1])
    print(f"\n[weight_diff] top {top_k}:")
    for nm, d, r in diffs[:top_k]:
        print(f"  {nm:42s} Δ={d:.6f}  rel={r:.4%}")
    frozen = [n for n, d, _ in diffs if d < 1e-10]
    if frozen:
        print(f"  ... {len(frozen)} params unchanged")
    return diffs
