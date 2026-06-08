"""
walpurgis_umbra — D2STGNN Umbra变体 (M051)
算法改写 (~20%):
  - MoE Gating (2-expert mixture): EstimationGate用双专家路由, 学习路由概率
  - Haar小波分解: 替代LayerNorm, 将信号拆为低频+高频分量再重组
  - Random Walk Diffusion + Restart: 带重启的随机游走, teleport概率可学习
  - Hyperbolic Poincaré距离: 双曲空间距离度量, 适合层次结构数据
  - Differentiable Top-K (Gumbel-Sinkhorn): 可微分top-k选择替代硬掩码
  - PageRank-style归一化: PersonalizedPageRank, 带damping因子
  - Mamba SSM + ALiBi位置偏差: 状态空间模型替代GRU, ALiBi替代正弦PE
  - DenseNet-style Skip Connections: 所有层到所有层的密集连接
  - Adaptive Huber Loss: 自适应delta参数的Huber损失
  - LAMB + Polynomial Decay LR: LAMB优化器配合多项式衰减调度

全局调试: UMBRA_DEBUG=1 开启
诊断日志: UMBRA_DIAG_LOG=<path> 写入JSONL
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, OrderedDict

# ─── 全局调试开关 ─────────────────────────────────────────
_DEBUG_ENV = os.environ.get("UMBRA_DEBUG", "0")
_DEBUG_ALL = _DEBUG_ENV.strip() == "1"
_DEBUG_MODULES = (
    set(_DEBUG_ENV.split(","))
    if not _DEBUG_ALL and _DEBUG_ENV != "0"
    else set()
)
_DIAG_LOG_PATH = os.environ.get("UMBRA_DIAG_LOG", "")
_STEP_COUNTER = {"global": 0, "epoch": 0, "batch": 0}


def _is_debug(module_name=""):
    return _DEBUG_ALL or module_name in _DEBUG_MODULES


def _diag_write(record: dict):
    """写入诊断记录到JSONL"""
    if not _DIAG_LOG_PATH:
        return
    try:
        record["_ts"] = time.time()
        record["_step"] = _STEP_COUNTER.copy()
        with open(_DIAG_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _dbg(tag, tensor_or_msg, module=""):
    """通用断点诊断: 打tensor统计或字符串, 同时写JSONL"""
    if not _is_debug(module) and not _DEBUG_ALL:
        return
    prefix = f"[UMB-DBG:{tag}]"
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
        if dead_frac > 0.5:
            alert += f" !!DEAD({dead_frac:.1%})"
        stats = (
            f"shape={list(t.shape)} "
            f"μ={t.mean().item():.6f} σ={t.std().item():.6f} "
            f"min={t.min().item():.6f} max={t.max().item():.6f} "
            f"dead%={dead_frac:.3f}{alert}"
        )
        print(f"{prefix} {stats}")
        _diag_write({
            "tag": tag, "module": module, "type": "tensor",
            "shape": list(t.shape),
            "mean": t.mean().item(), "std": t.std().item(),
            "min": t.min().item(), "max": t.max().item(),
            "dead_frac": dead_frac,
            "has_nan": has_nan, "has_inf": has_inf,
        })
    else:
        print(f"{prefix} {tensor_or_msg}")
        _diag_write({
            "tag": tag, "module": module, "type": "msg",
            "content": str(tensor_or_msg),
        })


def dataflow_checkpoint(name, tensor):
    """在数据流关键节点插入检查点 — 类似断点调试"""
    if not _is_debug():
        return
    if isinstance(tensor, torch.Tensor):
        _dbg(f"CHECKPOINT:{name}", tensor)
        grad_info = "requires_grad" if tensor.requires_grad else "no_grad"
        _dbg(f"CHECKPOINT:{name}.grad_status", grad_info)
    elif isinstance(tensor, (list, tuple)):
        _dbg(f"CHECKPOINT:{name}", f"container len={len(tensor)}")


def dump_struct_state(label, **kwargs):
    """打印当前所有结构体的完整状态 — 类似GDB的info locals"""
    if not _is_debug():
        return
    print(f"\n{'='*60}")
    print(f"  STRUCT DUMP: {label}")
    print(f"  step={_STEP_COUNTER}")
    print(f"{'='*60}")
    for name, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            t = val
            print(
                f"  {name}: Tensor shape={list(t.shape)} "
                f"dtype={t.dtype} device={t.device}"
            )
            print(
                f"    μ={t.mean().item():.6f} σ={t.std().item():.6f} "
                f"[{t.min().item():.4f}, {t.max().item():.4f}]"
            )
            if t.requires_grad and t.grad is not None:
                g = t.grad
                print(
                    f"    grad: μ={g.mean().item():.6f} "
                    f"norm={g.norm().item():.6f}"
                )
        elif isinstance(val, nn.Module):
            n_params = sum(p.numel() for p in val.parameters())
            print(f"  {name}: Module params={n_params}")
        elif isinstance(val, (list, tuple)):
            print(f"  {name}: {type(val).__name__} len={len(val)}")
            for i, v in enumerate(val[:3]):
                if isinstance(v, torch.Tensor):
                    print(f"    [{i}]: shape={list(v.shape)}")
        elif isinstance(val, dict):
            print(f"  {name}: dict keys={list(val.keys())[:10]}")
        else:
            print(f"  {name}: {type(val).__name__} = {val}")
    print(f"{'='*60}\n")


def register_activation_hooks(model, tag_prefix=""):
    """注册forward hook打印每层激活统计 — 调试激活分布"""
    hooks = []
    activation_stats = {}

    def _make_hook(name):
        def hook(module, inp, output):
            if not _is_debug():
                return
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "max_abs": output.abs().max().item(),
                    "dead_pct": (output.abs() < 1e-8).float().mean().item(),
                }
        return hook

    for name, module in model.named_modules():
        full_name = f"{tag_prefix}.{name}" if tag_prefix else name
        h = module.register_forward_hook(_make_hook(full_name))
        hooks.append(h)

    return hooks, activation_stats


def gradient_health_check(model, step=None):
    """检查梯度健康状态 — 发现vanishing/exploding梯度"""
    if not _is_debug():
        return {}
    report = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            grad_norm = g.norm().item()
            grad_max = g.abs().max().item()
            is_vanishing = grad_norm < 1e-7
            is_exploding = grad_norm > 1e3
            status = "OK"
            if is_vanishing:
                status = "VANISHING"
            elif is_exploding:
                status = "EXPLODING"
            report[name] = {
                "norm": grad_norm, "max": grad_max,
                "status": status,
            }
            if status != "OK":
                print(
                    f"[UMB-GRAD] {status}: {name} "
                    f"norm={grad_norm:.6f} max={grad_max:.6f}"
                )
    if step is not None:
        _diag_write({"tag": "grad_health", "step": step, "report": report})
    return report


def gradient_histogram(model, bins=10):
    """打印梯度直方图 — 可视化梯度分布"""
    if not _is_debug():
        return
    print("\n[UMB-GRAD-HIST] Gradient Distribution:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.flatten()
            hist = torch.histc(g, bins=bins)
            total = g.numel()
            edges = torch.linspace(g.min(), g.max(), bins + 1)
            bars = ""
            for i in range(bins):
                pct = hist[i].item() / total
                bar = "█" * int(pct * 40)
                bars += f"  [{edges[i]:.4f},{edges[i+1]:.4f}] {bar} {pct:.1%}\n"
            print(f"  {name}:\n{bars}")


def weight_diff(model, old_state_dict, top_k=5):
    """比较当前权重与旧snapshot的差异 — 看哪些层学到了最多"""
    if not _is_debug():
        return
    diffs = {}
    for name, param in model.named_parameters():
        if name in old_state_dict:
            diff = (param.data - old_state_dict[name]).norm().item()
            diffs[name] = diff
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[UMB-WDIFF] Top {top_k} weight changes:")
    for name, diff in sorted_diffs[:top_k]:
        print(f"  {name}: Δ={diff:.6f}")


def snapshot_model(model):
    """拍快照 — 供后续weight_diff比较"""
    return {name: param.data.clone() for name, param in model.named_parameters()}


class PerfTimer:
    """性能计时器 — 测量各阶段耗时"""

    def __init__(self, name=""):
        self.name = name
        self.records = {}
        self._start = {}

    def start(self, phase):
        self._start[phase] = time.time()

    def stop(self, phase):
        if phase in self._start:
            elapsed = time.time() - self._start[phase]
            if phase not in self.records:
                self.records[phase] = []
            self.records[phase].append(elapsed)
            if _is_debug():
                print(f"[UMB-TIMER:{self.name}] {phase}: {elapsed:.4f}s")
            return elapsed
        return 0.0

    def summary(self):
        if not _is_debug():
            return
        print(f"\n[UMB-TIMER:{self.name}] Summary:")
        for phase, times in self.records.items():
            avg = np.mean(times)
            total = np.sum(times)
            print(f"  {phase}: avg={avg:.4f}s total={total:.2f}s ({len(times)} calls)")


class MoERouterTracker:
    """追踪MoE路由的负载均衡情况"""

    def __init__(self):
        self.expert_loads = []

    def record(self, route_probs):
        """route_probs: [B,..., num_experts] 路由概率"""
        load = route_probs.detach().mean(dim=tuple(range(route_probs.dim()-1)))
        self.expert_loads.append(load.cpu().numpy())
        if _is_debug():
            _dbg("moe_router.load_balance",
                 f"expert_loads={load.tolist()}")

    def summary(self):
        if not self.expert_loads:
            return
        avg = np.mean(self.expert_loads, axis=0)
        print(f"[UMB-MoE] avg_expert_loads={avg.tolist()}")


class HaarWaveletTracker:
    """追踪Haar小波分解的低频/高频能量比"""

    def __init__(self):
        self.ratios = []

    def record(self, low_energy, high_energy):
        ratio = low_energy / (high_energy + 1e-8)
        self.ratios.append(ratio)
        if _is_debug():
            _dbg("haar.energy_ratio",
                 f"low/high={ratio:.4f} low={low_energy:.4f} high={high_energy:.4f}")

    def summary(self):
        if not self.ratios:
            return
        print(f"[UMB-HAAR] avg_lh_ratio={np.mean(self.ratios):.4f}")


class PageRankTracker:
    """追踪PageRank归一化的收敛情况"""

    def __init__(self):
        self.iterations_per_call = []
        self.residuals = []

    def record(self, n_iters, final_residual):
        self.iterations_per_call.append(n_iters)
        self.residuals.append(final_residual)
        if _is_debug():
            _dbg("pagerank.convergence",
                 f"iters={n_iters} residual={final_residual:.6f}")

    def summary(self):
        if not self.iterations_per_call:
            return
        print(
            f"[UMB-PPR] avg_iters={np.mean(self.iterations_per_call):.1f} "
            f"avg_residual={np.mean(self.residuals):.6f}"
        )


# 全局追踪器
_moe_tracker = MoERouterTracker()
_haar_tracker = HaarWaveletTracker()
_pagerank_tracker = PageRankTracker()
_perf_timer = PerfTimer("global")
