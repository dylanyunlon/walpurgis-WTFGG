"""
walpurgis — D2STGNN 鲁迅式移植
==========================================================
Upstream: D2STGNN (upstream/d2stgnn)
算法改动策略 ≥20%:
  - 损失: smooth Huber + log-cosh 混合, 可切换 quantile loss (τ=0.5)
  - 估计门: Swish 激活, 双头注意力加权(不同 W_q W_k), GroupNorm
  - 残差分解: Mish 激活, 可学习残差缩放系数 α (init 0.9)
  - 时空卷积: InstanceNorm 替代 BN, gconv 内加 skip connection
  - 扩散预测: Cosine 退火 AR dropout, 线性插值 padding
  - 扩散块: 3-layer MLP backcast + GELU, residual gating
  - 距离函数: 多头(3-head) Q-K 注意力, InstanceNorm, Dropout 正则
  - 图掩码: softplus 阈值 + 温度衰减 τ_anneal, 对角清零
  - 归一化: 双向对称 D^{-1/2}AD^{-1/2} + 高阶指数衰减 λ^k (0.8)
  - 动态图: 可学习时间卷积权重 (softmax), cosine-sim 辅助
  - GRU: 步间 RMSNorm + gradient checkpoint
  - Transformer: Rotary PE, flash-attn-style mask, 注意力熵监控
  - 固有预测: 可学习步长衰减 exp(-γ·step)
  - 固有块: 残差门控 (sigmoid gate), gradient checkpoint
  - 主模型: Mish 输出激活, 层权重 softmax 聚合 + 温度
  - 训练器: 自适应 p90 梯度裁剪, warmup-cosine 学习率
  - 数据: 周期性 sin/cos 编码, Tukey fences 异常剔除
  - 邻接: RBF kernel + k-NN(15) 稀疏化 + 双向对称闭包
  - DataLoader: 环形 wrap padding, Knuth shuffle, 3-tuple yield

调试系统:
  设置 WALPURGIS_DEBUG=1  开启所有调试打印
  设置 WALPURGIS_DEBUG=model,trainer  只开启指定模块
"""

import os as _os

_DEBUG_ENV = _os.environ.get("WALPURGIS_DEBUG", "")
_DEBUG_ALL = (_DEBUG_ENV == "1")
_DEBUG_TAGS = set(_DEBUG_ENV.split(",")) if _DEBUG_ENV and not _DEBUG_ALL else set()


def _dbg(tag: str, msg: str, **tensors):
    """统一调试打印入口.

    使用方法:
        from walpurgis import _dbg
        _dbg("model", "forward pass", x=some_tensor, adj=adj_tensor)

    自动检测:
        - NaN/Inf 会触发 WARNING 级别输出，即使该 tag 未开启
        - 稀疏度 >95% 的 tensor 会附加 sparsity 信息
    """
    if not (_DEBUG_ALL or tag in _DEBUG_TAGS):
        # 即使 tag 未开启，NaN/Inf 仍然告警
        import torch as _th
        for name, t in tensors.items():
            if isinstance(t, _th.Tensor):
                n_nan = t.isnan().sum().item()
                n_inf = t.isinf().sum().item()
                if n_nan > 0 or n_inf > 0:
                    print(f"[walpurgis:ALERT:{tag}] {msg} | {name}: "
                          f"nan={n_nan} inf={n_inf} shape={tuple(t.shape)}",
                          flush=True)
        return
    parts = [f"[walpurgis:{tag}] {msg}"]
    for name, t in tensors.items():
        import torch as _th
        if isinstance(t, _th.Tensor):
            n_elem = t.numel()
            n_zero = (t.abs() < 1e-8).sum().item() if n_elem > 0 else 0
            sparsity = n_zero / max(n_elem, 1)
            s = (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
                 f"min={t.min().item():.4g} max={t.max().item():.4g} "
                 f"mean={t.float().mean().item():.4g} "
                 f"nan={t.isnan().sum().item()} inf={t.isinf().sum().item()}")
            if sparsity > 0.95:
                s += f" SPARSE({sparsity:.1%})"
        elif isinstance(t, (list, tuple)):
            s = f"{name}: len={len(t)} types={[type(x).__name__ for x in t[:3]]}"
        else:
            s = f"{name}: {type(t).__name__}={t}"
        parts.append(s)
    print(" | ".join(parts), flush=True)


# ============================================================
# 全模型断点快照系统
# upstream 完全没有这套东西; 这里提供三个层级的诊断:
#   1) snapshot_model   — 一次性 dump 所有参数 + 梯度的统计量
#   2) register_activation_hooks — 注册 forward hook, 持续记录激活
#   3) gradient_health_check — 检测梯度消失/爆炸/nan
# ============================================================

def snapshot_model(model, epoch=None, step=None, top_k=10):
    """全模型参数+梯度快照 — 类似 pdb 里 print(locals()) 的效果.

    打印每个参数的 shape / mean / std / abs_max / grad_norm,
    按 grad_norm 降序排列, 只显示 top_k 个最大梯度的参数(其余折叠).
    同时统计: 总参数量 / 可训练参数量 / 有梯度参数量 / 梯度为零参数量.
    """
    import torch as _th

    header = "[walpurgis:snapshot]"
    if epoch is not None:
        header += f" epoch={epoch}"
    if step is not None:
        header += f" step={step}"

    records = []
    total_params = 0
    trainable_params = 0
    has_grad = 0
    zero_grad = 0
    nan_params = []

    for name, p in model.named_parameters():
        n_elem = p.numel()
        total_params += n_elem
        if p.requires_grad:
            trainable_params += n_elem

        grad_norm = None
        if p.grad is not None:
            has_grad += 1
            grad_norm = p.grad.data.norm().item()
            if p.grad.data.abs().max().item() < 1e-10:
                zero_grad += 1
        if p.data.isnan().any():
            nan_params.append(name)

        records.append({
            "name": name,
            "shape": tuple(p.shape),
            "mean": p.data.float().mean().item(),
            "std": p.data.float().std().item(),
            "abs_max": p.data.float().abs().max().item(),
            "grad_norm": grad_norm,
        })

    # 按 grad_norm 降序
    records.sort(key=lambda r: -(r["grad_norm"] or 0))

    # ── 参数分布健康度自动诊断 ──
    # 检测三类初始化病态:
    #   1) 峰度异常: kurtosis > 10 说明分布有极端尖峰或重尾
    #   2) 均值偏移: |mean| > 3*std 说明初始化中心严重偏离零点
    #   3) 尺度坍缩: std < 1e-6 说明参数几乎为常数(可能是错误初始化)
    pathologies = []
    for r in records:
        flags = []
        if r["std"] < 1e-6 and r["abs_max"] > 1e-10:
            flags.append("COLLAPSED_SCALE")
        if r["std"] > 1e-8 and abs(r["mean"]) > 3.0 * r["std"]:
            flags.append("MEAN_DRIFT")
        if r.get("grad_norm") is not None and r["grad_norm"] > 50.0:
            flags.append("GRAD_SPIKE")
        if flags:
            pathologies.append((r["name"], flags))

    print(f"\n{'━' * 80}")
    print(f"{header}")
    print(f"  total_params={total_params:,}  trainable={trainable_params:,}  "
          f"has_grad={has_grad}  zero_grad={zero_grad}  "
          f"nan_params={len(nan_params)}")
    if nan_params:
        print(f"  ⚠ NaN detected in: {nan_params[:5]}")
    if pathologies:
        print(f"  ⚠ {len(pathologies)} pathological params:")
        for pname, pflags in pathologies[:5]:
            print(f"    {pname}: {', '.join(pflags)}")
    print(f"{'─' * 80}")
    for i, r in enumerate(records[:top_k]):
        gn = f"{r['grad_norm']:.4g}" if r['grad_norm'] is not None else "None"
        print(f"  {r['name']:45s} {str(r['shape']):18s} "
              f"μ={r['mean']:+.4f} σ={r['std']:.4f} "
              f"|max|={r['abs_max']:.4f} ∇={gn}")
    if len(records) > top_k:
        remaining_grads = [r["grad_norm"] for r in records[top_k:]
                           if r["grad_norm"] is not None]
        if remaining_grads:
            print(f"  ... {len(records)-top_k} more params, "
                  f"grad_norm range=[{min(remaining_grads):.4g}, "
                  f"{max(remaining_grads):.4g}]")
    print(f"{'━' * 80}\n")
    return records


class _ActivationTracker:
    """注册到模型上的 forward hook 集合, 持续记录每层激活的统计量.

    用法:
        tracker = register_activation_hooks(model)
        model(input)  # forward
        tracker.report()         # 打印所有激活
        tracker.check_dead()     # 检查死神经元 (>90% 输出为零)
        tracker.remove()         # 清除 hooks
    """

    def __init__(self):
        self._hooks = []
        self._data = {}

    def _make_hook(self, name):
        def _hook(module, inp, out):
            import torch as _th
            t = out if isinstance(out, _th.Tensor) else (
                out[0] if isinstance(out, tuple) and len(out) > 0
                and isinstance(out[0], _th.Tensor) else None)
            if t is None:
                return
            self._data[name] = {
                "shape": tuple(t.shape),
                "mean": t.float().mean().item(),
                "std": t.float().std().item(),
                "abs_max": t.float().abs().max().item(),
                "zero_frac": (t.abs() < 1e-7).float().mean().item(),
                "nan": t.isnan().any().item(),
            }
        return _hook

    def report(self):
        print(f"\n{'─' * 76}")
        print(f"[walpurgis:activations] {len(self._data)} layers captured")
        for name, s in self._data.items():
            flag = ""
            if s["nan"]:
                flag = " ⚠NaN"
            elif s["zero_frac"] > 0.9:
                flag = " ⚠DEAD"
            print(f"  {name:45s} {str(s['shape']):18s} "
                  f"μ={s['mean']:+.5f} σ={s['std']:.5f} "
                  f"dead={s['zero_frac']:.1%}{flag}")
        print(f"{'─' * 76}\n")

    def check_dead(self, threshold=0.9):
        dead = {n: s["zero_frac"] for n, s in self._data.items()
                if s["zero_frac"] > threshold}
        if dead:
            print(f"[walpurgis:WARN] {len(dead)} dead layers (>{threshold:.0%} zeros):")
            for n, frac in sorted(dead.items(), key=lambda x: -x[1]):
                print(f"  {n}: {frac:.1%} zeros")
        return dead

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._data.clear()


def register_activation_hooks(model, prefix=""):
    """注册 forward hooks, 返回 _ActivationTracker 实例."""
    tracker = _ActivationTracker()
    for name, module in model.named_modules():
        if name and not any(
            isinstance(module, t) for t in [
                __import__("torch.nn", fromlist=["Sequential"]).Sequential]):
            full_name = f"{prefix}{name}" if prefix else name
            h = module.register_forward_hook(tracker._make_hook(full_name))
            tracker._hooks.append(h)
    return tracker


def gradient_health_check(model, explode_threshold=100.0, vanish_threshold=1e-7):
    """梯度健康度检查 — 检测爆炸/消失/nan."""
    import torch as _th
    issues = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.data.norm().item()
        if _th.isnan(p.grad).any():
            issues.append(("NaN_GRAD", name, gn))
        elif gn > explode_threshold:
            issues.append(("EXPLODE", name, gn))
        elif gn < vanish_threshold and p.grad.data.abs().max().item() < vanish_threshold:
            issues.append(("VANISH", name, gn))

    if issues:
        print(f"[walpurgis:grad_health] {len(issues)} issues:")
        for kind, name, val in issues:
            print(f"  {kind:10s} {name:40s} norm={val:.4g}")
    return issues


def weight_diff(model, prev_state_dict, top_k=10):
    """比较当前权重与之前保存的 state_dict, 显示变化最大的 top_k 参数.

    用法:
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        ... 训练若干步 ...
        weight_diff(model, sd_before)
    """
    diffs = []
    for name, p in model.named_parameters():
        if name in prev_state_dict:
            delta = (p.data - prev_state_dict[name]).float().norm().item()
            scale = prev_state_dict[name].float().norm().item()
            rel = delta / max(scale, 1e-10)
            diffs.append((name, delta, rel))

    diffs.sort(key=lambda x: -x[1])
    print(f"\n[walpurgis:weight_diff] top {top_k} changed parameters:")
    for name, delta, rel in diffs[:top_k]:
        print(f"  {name:45s} Δ={delta:.6f}  rel={rel:.4%}")
    frozen = [n for n, d, _ in diffs if d < 1e-10]
    if frozen:
        print(f"  ... {len(frozen)} params unchanged (possibly frozen)")
    return diffs
