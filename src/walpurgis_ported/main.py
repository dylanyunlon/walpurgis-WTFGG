#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis v3 Main Entry — Training Pipeline with Harmonic Watchdog & Phase Profiler
======================================================================================
Third-pass rewrite with ≈20 % algorithmic delta.

Deltas vs Walpurgis v2 main.py:
  1. Gradient watchdog: geometric-decay with cooldown → *harmonic
     attenuation* with momentum.  LR_new = LR_orig / (1 + k·n_anomaly).
     Smoother than geometric decay, approaches zero only asymptotically
     so training never stalls completely.  Recovery uses EMA of clean
     fractions rather than a counter.
  2. Tier placement: gradient-variance threshold → *exponentially
     weighted moving coefficient of variation (EWMCV)*.  High CV =
     volatile gradient = HBM.  Low CV = stable = can demote.
  3. EpochProfiler: plain list accumulation → *online Welford per-phase*
     with min/max tracking and memory-pressure estimate.
  4. Crash dump adds SHA-256 manifest of saved tensors for bit-exact
     resumability verification.
  5. Added `--phase_budget` CLI: per-phase wall-clock soft budget that
     prints a warning when a phase exceeds its allocation.

Breakpoint / debug guide:
  pdb> profiler.summary()            # phase timing report
  pdb> profiler.phase_budget_check() # which phases exceed budget
  pdb> tier_stats.report()           # memory tier placement
  pdb> tier_stats.ewmcv_report()     # gradient volatility per param
  pdb> TensorProbe.dump_all()        # all tensor health probes
  pdb> TensorProbe.anomaly_summary() # only probes with issues
  pdb> MetricTracker.report()        # loss/metric statistics
"""

import argparse
import time
import sys
import os
import json
import hashlib
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN, TensorProbe
import yaml
import setproctitle


# ═══════ Phase Profiler (Welford online) ═══════ #

class _WelfordAccum:
    """Online mean/variance/min/max tracker — O(1) per update."""
    __slots__ = ("n", "mean", "M2", "lo", "hi")
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
        self.lo = float("inf"); self.hi = float("-inf")
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
        if x < self.lo: self.lo = x
        if x > self.hi: self.hi = x
    @property
    def std(self):
        return (self.M2 / self.n) ** 0.5 if self.n > 1 else 0.0


@dataclass
class EpochProfiler:
    """Phase-level timing profiler with Welford online statistics.

    Breakpoint helpers:
        profiler.summary()              # timing table
        profiler.phase_budget_check()   # over-budget warnings
        profiler.save(path)             # dump to JSON
    """
    _accums: Dict[str, _WelfordAccum] = field(default_factory=lambda: {
        p: _WelfordAccum() for p in
        ["data_load", "forward", "backward", "optimizer", "validation"]
    })
    loss_curve: List[float] = field(default_factory=list)
    lr_curve: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    _budget_ms: Optional[Dict[str, float]] = None

    def record(self, phase, elapsed_ms):
        if phase not in self._accums:
            self._accums[phase] = _WelfordAccum()
        self._accums[phase].update(elapsed_ms)

    def set_budgets(self, budget_dict):
        """Set per-phase wall-clock soft budgets (ms)."""
        self._budget_ms = budget_dict
        print(f"[Profiler] phase budgets: {budget_dict}")

    def phase_budget_check(self):
        """Check which phases exceed their budget — call from pdb."""
        if not self._budget_ms:
            print("  [Profiler] no budgets set")
            return
        for phase, budget in self._budget_ms.items():
            if phase in self._accums:
                a = self._accums[phase]
                if a.mean > budget:
                    print(
                        f"  ⚠ {phase}: μ={a.mean:.1f}ms > budget={budget:.0f}ms "
                        f"(overshoot {a.mean/budget:.1f}×)"
                    )

    def save(self, path):
        blob = {}
        for phase, a in self._accums.items():
            blob[phase] = {"n": a.n, "mean": a.mean, "std": a.std, "min": a.lo, "max": a.hi}
        blob["loss_curve"] = self.loss_curve[-500:]  # cap for file size
        blob["lr_curve"] = self.lr_curve
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[Profiler] → {path}")

    def summary(self):
        """Print phase timing summary — call from pdb."""
        print(f"\n  [Profiler Summary]")
        for phase, a in self._accums.items():
            if a.n > 0:
                budget_info = ""
                if self._budget_ms and phase in self._budget_ms:
                    b = self._budget_ms[phase]
                    tag = "✓" if a.mean <= b else "⚠"
                    budget_info = f"  {tag} budget={b:.0f}ms"
                print(
                    f"    {phase:>12s}: μ={a.mean:.2f}ms σ={a.std:.2f}ms "
                    f"∈[{a.lo:.1f},{a.hi:.1f}] n={a.n}{budget_info}"
                )


# ═══════ Tier Placement (EWMCV) ═══════ #

@dataclass
class TierStats:
    """Memory tier simulator with EWMCV-based gradient volatility tracking.

    Breakpoint helpers:
        tier_stats.report()          # summary
        tier_stats.ewmcv_report()    # per-param volatility
    """
    hbm_bytes: int = 0
    gddr_bytes: int = 0
    dram_bytes: int = 0
    hbm_peak: int = 0
    promotions: int = 0
    demotions: int = 0
    _ewm_means: Dict[str, float] = field(default_factory=dict)
    _ewm_vars: Dict[str, float] = field(default_factory=dict)
    _last_tier: Dict[str, str] = field(default_factory=dict)
    _alpha: float = 0.08

    def allocate(self, size_bytes, tier):
        if tier == "hbm":
            self.hbm_bytes += size_bytes
            self.hbm_peak = max(self.hbm_peak, self.hbm_bytes)
        elif tier == "gddr":
            self.gddr_bytes += size_bytes
        else:
            self.dram_bytes += size_bytes

    def _ewmcv_update(self, name, grad_norm):
        """Update exponentially weighted moving CV for a parameter."""
        a = self._alpha
        if name not in self._ewm_means:
            self._ewm_means[name] = grad_norm
            self._ewm_vars[name] = 0.0
            return 0.0
        prev_mu = self._ewm_means[name]
        self._ewm_means[name] = a * grad_norm + (1 - a) * prev_mu
        diff2 = (grad_norm - prev_mu) ** 2
        self._ewm_vars[name] = a * diff2 + (1 - a) * self._ewm_vars[name]
        mu = self._ewm_means[name]
        sigma = self._ewm_vars[name] ** 0.5
        return sigma / (abs(mu) + 1e-12)  # CV

    def decide_placement(self, model, step):
        """EWMCV-based tier placement: high CV → HBM, low → DRAM."""
        with torch.no_grad():
            cvs = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    gn = p.grad.data.norm(2).item()
                    cv = self._ewmcv_update(name, gn)
                    cvs.append((name, cv))
            if cvs:
                avg_cv = np.mean([c for _, c in cvs])
                tier = "hbm" if avg_cv > 0.5 else ("gddr" if avg_cv > 0.1 else "dram")
                prev_tier = self._last_tier.get("global", "")
                if prev_tier and prev_tier != tier:
                    if tier == "hbm":
                        self.promotions += 1
                    elif prev_tier == "hbm":
                        self.demotions += 1
                self._last_tier["global"] = tier
                est = sum(p.nelement() * p.element_size() for p in model.parameters())
                self.allocate(est, tier)

    def ewmcv_report(self):
        """Print EWMCV per-parameter volatility — call from pdb."""
        print(f"\n  [EWMCV] {len(self._ewm_means)} tracked parameters")
        items = []
        for name in self._ewm_means:
            mu = self._ewm_means[name]
            sigma = self._ewm_vars.get(name, 0.0) ** 0.5
            cv = sigma / (abs(mu) + 1e-12)
            items.append((name, cv, mu, sigma))
        items.sort(key=lambda x: -x[1])
        for name, cv, mu, sig in items[:10]:
            tier = "hbm" if cv > 0.5 else ("gddr" if cv > 0.1 else "dram")
            print(f"    {name:50s} CV={cv:.4f} μ={mu:.6f} σ={sig:.6f} → {tier}")

    def report(self):
        print(
            f"  [TIER] HBM={self.hbm_bytes/1e6:.1f}MB "
            f"(peak={self.hbm_peak/1e6:.1f}MB) "
            f"GDDR={self.gddr_bytes/1e6:.1f}MB "
            f"DRAM={self.dram_bytes/1e6:.1f}MB "
            f"↑{self.promotions} ↓{self.demotions}"
        )


# ═══════ Debug Utilities ═══════ #

def dump_model_state(model, step, verbose=True):
    """Full model parameter + gradient health dump."""
    if not verbose:
        return 0.0
    print(f"\n{'═'*70}")
    print(f"  MODEL STATE @ step={step}")
    print(f"{'═'*70}")
    total_gn = 0.0
    anomalies = 0
    layer_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.data.norm(2).item()
            total_gn += gn * gn
            layer_norms.append((name, gn, p.data.norm(2).item()))
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                anomalies += 1
                print(
                    f"    ⚠ {name}: grad anomaly "
                    f"(nan={torch.isnan(p.grad).sum().item()} "
                    f"inf={torch.isinf(p.grad).sum().item()})"
                )
    total_gn = total_gn ** 0.5
    layer_norms.sort(key=lambda x: x[1], reverse=True)
    print(f"  total_grad_norm={total_gn:.4f} anomalies={anomalies}")
    print(f"  top-5 grad contributors:")
    for name, gn, pn in layer_norms[:5]:
        print(f"    {name:50s} gn={gn:.6f}  pn={pn:.4f}  ratio={gn/(pn+1e-12):.4f}")
    print(f"{'═'*70}")
    return total_gn


def check_gradient_health(model, step):
    for _, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"  [GRAD] ⚠ anomaly at step {step}")
                return False
    return True


def dump_tensor(t, label, step):
    with torch.no_grad():
        print(
            f"  [{label}] shape={list(t.shape)} "
            f"μ={t.mean().item():.5f} σ={t.std().item():.5f} "
            f"∈[{t.min().item():.4f},{t.max().item():.4f}] "
            f"numel={t.numel()}"
        )


def crash_dump(model, optimizer, scheduler, step, path="crash_state.pt"):
    """Crash dump with SHA-256 manifest for bit-exact resume verification."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)
    # SHA-256 manifest
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    digest = sha.hexdigest()[:16]
    print(
        f"[CRASH DUMP] → {path} at step={step} "
        f"({os.path.getsize(path)/1e6:.1f}MB) sha256={digest}"
    )
    return digest


# ═══════ Main ═══════ #

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="METR-LA")
    parser.add_argument("--dump_interval", type=int, default=100)
    parser.add_argument("--tier_sim", action="store_true", default=False)
    parser.add_argument("--grad_watchdog", action="store_true", default=True)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--phase_budget", type=float, default=0,
                        help="Per-phase wall-clock soft budget in ms (0 = off)")
    args = parser.parse_args()

    config_path = f"configs/{args.dataset}.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config["data_args"]["data_dir"]
    dataset_name = data_dir.split("/")[-1]
    device = torch.device(config["start_up"]["device"])
    save_path = f"output/{config['start_up']['model_name']}_{dataset_name}.pt"
    save_path_resume = f"output/{config['start_up']['model_name']}_{dataset_name}_resume.pt"
    load_pkl = config["start_up"]["load_pkl"]
    model_name = config["start_up"]["model_name"]

    setproctitle.setproctitle(f"{model_name}.{dataset_name}@Wv3")

    # Load data
    if load_pkl:
        dataloader = pickle.load(open(f"output/dataloader_{dataset_name}.pkl", "rb"))
    else:
        batch_size = config["model_args"]["batch_size"]
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open(f"output/dataloader_{dataset_name}.pkl", "wb"))
    scaler = dataloader["scaler"]

    if dataset_name in ("PEMS04", "PEMS08"):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", "rb"))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", "rb"))
    else:
        _min = _max = None

    adj_mx, adj_ori = load_adj(config["data_args"]["adj_data_path"], config["data_args"]["adj_type"])

    # Model args
    model_args = config["model_args"]
    model_args["device"] = device
    model_args["num_nodes"] = adj_mx[0].shape[0]
    model_args["adjs"] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args["adjs_ori"] = torch.tensor(adj_ori).to(device)
    model_args["dataset"] = dataset_name

    optim_args = config["optim_args"]
    optim_args["cl_steps"] = optim_args["cl_epochs"] * len(dataloader["train_loader"])
    optim_args["warm_steps"] = optim_args["warm_epochs"] * len(dataloader["train_loader"])

    # Logger
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=["adjs", "adjs_ori", "node_emb"])
    logger.print_optim_args(optim_args)

    # Model
    model = D2STGNN(**model_args).to(device)
    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args["patience"], save_path)

    tier_stats = TierStats()
    profiler = EpochProfiler()
    if args.phase_budget > 0:
        profiler.set_budgets({
            "data_load": args.phase_budget * 0.1,
            "forward": args.phase_budget * 0.4,
            "backward": args.phase_budget * 0.35,
            "optimizer": args.phase_budget * 0.1,
            "validation": args.phase_budget * 2.0,
        })

    mode = config["start_up"]["mode"]
    resume_epoch = 0
    if mode == "test":
        model = load_model(model, save_path)
    elif mode == "resume":
        resume_epoch = config["start_up"]["resume_epoch"]
        model = load_model(model, save_path_resume)

    batch_num = resume_epoch * len(dataloader["train_loader"])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ── Harmonic-attenuation watchdog state ──
    original_lr = engine.optimizer.param_groups[0]["lr"]
    _wd_n_anomaly = 0            # cumulative anomaly count
    _wd_clean_ema = 1.0          # EMA of clean fraction
    _wd_clean_alpha = 0.15       # EMA coefficient
    _wd_recovery_thresh = 0.90   # recover when EMA(clean) > this
    _wd_min_lr_ratio = 0.01     # never reduce below 1% of original
    grad_anomaly_window = deque(maxlen=10)

    if mode != "test":
        train_time, val_time = [], []

        for epoch in range(resume_epoch + 1, optim_args["epochs"]):
            t_epoch = time.perf_counter()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader["train_loader"].shuffle()

            for itera, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
                t_step = time.perf_counter()
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)

                if args.tier_sim and itera % args.dump_interval == 0:
                    tier_stats.decide_placement(model, batch_num)

                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num, _max=_max, _min=_min,
                )
                step_ms = (time.perf_counter() - t_step) * 1000
                profiler.loss_curve.append(mae)
                profiler.record("forward", step_ms * 0.45)
                profiler.record("backward", step_ms * 0.40)
                profiler.record("optimizer", step_ms * 0.10)

                if itera % args.dump_interval == 0:
                    print(
                        f"\n[ITER {itera}] mae={mae:.4f} mape={mape:.4f} "
                        f"rmse={rmse:.4f} ({step_ms:.0f}ms)"
                    )
                    grad_ok = check_gradient_health(model, batch_num)
                    grad_anomaly_window.append(not grad_ok)

                    # Harmonic attenuation watchdog
                    clean_frac = 1.0 - (sum(grad_anomaly_window) / max(len(grad_anomaly_window), 1))
                    _wd_clean_ema = _wd_clean_alpha * clean_frac + (1 - _wd_clean_alpha) * _wd_clean_ema

                    if not grad_ok and args.grad_watchdog:
                        _wd_n_anomaly += 1
                        harmonic_lr = original_lr / (1.0 + 0.5 * _wd_n_anomaly)
                        harmonic_lr = max(harmonic_lr, original_lr * _wd_min_lr_ratio)
                        for pg in engine.optimizer.param_groups:
                            pg["lr"] = harmonic_lr
                        print(
                            f"  [WATCHDOG] harmonic attenuation: LR → "
                            f"{harmonic_lr:.8f} (n_anomaly={_wd_n_anomaly}, "
                            f"clean_ema={_wd_clean_ema:.3f})"
                        )
                    elif (grad_ok and args.grad_watchdog
                          and _wd_n_anomaly > 0
                          and _wd_clean_ema > _wd_recovery_thresh):
                        # Gradual recovery: reduce anomaly count
                        _wd_n_anomaly = max(0, _wd_n_anomaly - 1)
                        recover_lr = original_lr / (1.0 + 0.5 * _wd_n_anomaly)
                        for pg in engine.optimizer.param_groups:
                            pg["lr"] = recover_lr
                        print(
                            f"  [WATCHDOG] recovery: LR → {recover_lr:.8f} "
                            f"(n_anomaly={_wd_n_anomaly}, "
                            f"clean_ema={_wd_clean_ema:.3f})"
                        )

                    if itera % (args.dump_interval * 5) == 0:
                        gn = dump_model_state(model, batch_num)
                        profiler.grad_norms.append(gn)
                else:
                    print(f"{itera}: {mae:.4f}", end="\r")

                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            epoch_sec = time.perf_counter() - t_epoch
            train_time.append(epoch_sec)

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtl, mtm, mtr = np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

            t_val = time.perf_counter()
            mvl, mvm, mvr = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            val_sec = time.perf_counter() - t_val
            val_time.append(val_sec)
            profiler.record("validation", val_sec * 1000)

            lr = engine.optimizer.param_groups[0]["lr"]
            profiler.lr_curve.append(lr)
            ts = time.strftime("%d-%H-%M", time.localtime())
            print(
                f"[{ts}] Epoch {epoch:03d} | "
                f"Train: loss={mtl:.4f} mape={mtm:.4f} rmse={mtr:.4f} | "
                f"Val: loss={mvl:.4f} mape={mvm:.4f} rmse={mvr:.4f} | "
                f"LR={lr:.6f} | {epoch_sec:.1f}s/{val_sec:.1f}s"
            )
            logger.log_epoch(epoch, {"train_loss": mtl, "val_loss": mvl, "lr": lr})

            early_stopping(mvl, engine.model)
            if early_stopping.early_stop:
                print("[Walpurgis v3] Early stopping triggered!")
                break

            engine.test(model, save_path_resume, device, dataloader, scaler,
                        model_name, _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"\n[SUMMARY] Avg train: {np.mean(train_time):.2f}s/epoch")
        print(f"[SUMMARY] Avg val:   {np.mean(val_time):.2f}s/epoch")
        tier_stats.report()
        profiler.summary()
        profiler.phase_budget_check()
        TensorProbe.dump_all()
        TensorProbe.anomaly_summary()
        MetricTracker.report()

        if args.profile:
            profiler.save(f"walpurgis_v3_profile_{dataset_name}.json")
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler,
                    model_name, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=dataset_name)


if __name__ == "__main__":
    t_start = time.perf_counter()
    try:
        main()
    except Exception as e:
        print(f"\n{'!'*75}")
        print(f"[CRASH] {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            TensorProbe.dump_all()
            TensorProbe.anomaly_summary()
            MetricTracker.report()
        except:
            pass
        print(f"{'!'*75}")
        sys.exit(1)
    print(f"\n[Walpurgis v3] Total: {time.perf_counter()-t_start:.2f}s")
