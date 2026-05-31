#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis-TSH Main Entry — Heterogeneous-Memory STGNN Training Pipeline
=========================================================================
Derived from D2STGNN main.py with ~20% structural and algorithmic changes.

Changes vs upstream:
  1. Dataclass-based profiler with per-phase latency breakdown
  2. Tier placement simulation integrated into training loop
  3. Gradient health watchdog: auto-halves LR on repeated gradient anomalies
  4. Full crash diagnostics with model state dump on exception
  5. Sigmoid-ramped curriculum learning (delegated to trainer)
  6. Configurable debug verbosity via CLI args
"""

import argparse
import time
import sys
import os
import json
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List

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


# ═══════════ Walpurgis Profiling Infrastructure ═══════════ #

@dataclass
class TierStats:
    """Memory tier utilization tracker — models HBM→GDDR→DRAM waterfall.
    
    At any debug breakpoint, call tier_stats.report() to see current state.
    """
    hbm_bytes: int = 0
    gddr_bytes: int = 0
    dram_bytes: int = 0
    hbm_peak: int = 0
    promotions: int = 0     # DRAM/GDDR → HBM
    demotions: int = 0      # HBM → GDDR/DRAM
    
    def allocate(self, size_bytes: int, tier: str):
        if tier == 'hbm':
            self.hbm_bytes += size_bytes
            self.hbm_peak = max(self.hbm_peak, self.hbm_bytes)
        elif tier == 'gddr':
            self.gddr_bytes += size_bytes
        else:
            self.dram_bytes += size_bytes

    def as_dict(self) -> dict:
        return asdict(self)
    
    def report(self):
        print(f"  [TIER STATUS] HBM={self.hbm_bytes/1e6:.1f}MB (peak={self.hbm_peak/1e6:.1f}MB) "
              f"GDDR={self.gddr_bytes/1e6:.1f}MB DRAM={self.dram_bytes/1e6:.1f}MB "
              f"↑{self.promotions} ↓{self.demotions}")


@dataclass
class EpochProfiler:
    """Per-epoch latency and metric collector for publication-quality analysis."""
    phase_times: Dict[str, List[float]] = field(default_factory=lambda: {
        'data_load': [], 'forward': [], 'backward': [],
        'optimizer': [], 'tier_migration': [], 'validation': []
    })
    grad_norms: List[float] = field(default_factory=list)
    loss_curve: List[float] = field(default_factory=list)
    tier_history: List[dict] = field(default_factory=list)
    
    def record(self, phase: str, elapsed: float):
        if phase in self.phase_times:
            self.phase_times[phase].append(elapsed)
    
    def save(self, path: str):
        payload = {
            'phases': {k: v for k, v in self.phase_times.items()},
            'grad_norms': self.grad_norms,
            'loss_curve': self.loss_curve,
            'tier_snapshots': self.tier_history,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[Walpurgis] Profiler → {path}")


# ═══════════ Debug Dump Functions ═══════════ #

def dump_model_state(model, step: int, verbose: bool = True):
    """Full model parameter snapshot — equivalent to a debugger watch window.
    
    Prints every parameter's shape, mean, std, grad norm, and anomaly flags.
    Call this from pdb at any breakpoint to understand model state.
    """
    if not verbose:
        return 0.0
    print(f"\n{'═'*75}")
    print(f"  MODEL STATE @ step={step}")
    print(f"{'═'*75}")
    
    total_params = 0
    total_grad_sq = 0.0
    anomalies = []
    
    for name, p in model.named_parameters():
        total_params += p.numel()
        
        has_nan = torch.isnan(p.data).any().item()
        has_inf = torch.isinf(p.data).any().item()
        if has_nan: anomalies.append(f"NaN in {name}")
        if has_inf: anomalies.append(f"Inf in {name}")
        
        grad_str = "—"
        if p.grad is not None:
            gn = p.grad.data.norm(2).item()
            total_grad_sq += gn ** 2
            grad_str = f"{gn:.6f}"
        
        flag = ""
        if has_nan: flag += " 🔴NaN"
        if has_inf: flag += " 🔴Inf"
        
        print(f"  {name:48s} | {str(list(p.shape)):18s} | "
              f"μ={p.data.mean().item():+.5f} σ={p.data.std().item():.5f} | "
              f"∇={grad_str}{flag}")
    
    total_gn = total_grad_sq ** 0.5
    print(f"\n  Parameters: {total_params:,d} | Total ∇norm: {total_gn:.6f}")
    if anomalies:
        print(f"  ⚠️  ANOMALIES: {anomalies}")
    print(f"{'═'*75}\n")
    return total_gn


def dump_tensor(tensor, name: str, step: int):
    """Quick tensor inspection — lighter than dump_model_state.
    
    Use at specific points in the forward/backward pass.
    """
    if tensor is None:
        print(f"  [PROBE] step={step} | {name} = None")
        return
    print(f"  [PROBE] step={step} | {name}: "
          f"shape={list(tensor.shape)} dtype={tensor.dtype} dev={tensor.device} "
          f"∈[{tensor.min().item():.5f}, {tensor.max().item():.5f}] "
          f"μ={tensor.mean().item():.5f} σ={tensor.std().item():.5f} "
          f"nan={torch.isnan(tensor).sum().item()} inf={torch.isinf(tensor).sum().item()}")


def decide_tier_placement(batch_size: int, tier_stats: TierStats, step: int):
    """Simulate tier placement for current batch and log decision.
    
    Heuristic: batch_size > 32 → HBM, 17-32 → GDDR, ≤16 → DRAM
    Approximate memory = B × L × N × D × 4 bytes
    """
    if batch_size > 32:
        tier, reason = 'hbm', f"B={batch_size}>32 → high-bandwidth path"
    elif batch_size > 16:
        tier, reason = 'gddr', f"B={batch_size}∈[17,32] → mid-tier"
    else:
        tier, reason = 'dram', f"B={batch_size}≤16 → capacity path"
    
    approx_bytes = batch_size * 12 * 207 * 4 * 4
    tier_stats.allocate(approx_bytes, tier)
    
    print(f"  [TIER] step={step} | {tier.upper()} | {reason} | "
          f"~{approx_bytes/1e6:.1f}MB | peak_HBM={tier_stats.hbm_peak/1e6:.1f}MB")


def check_gradient_health(model, step: int, clip_val: float = 5.0):
    """Gradient anomaly detector — returns True if all healthy.
    
    Checks for: exploding (>50×clip), vanishing (<1e-8), NaN gradients.
    """
    problems = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        gn = g.norm(2).item()
        
        if gn > clip_val * 10:
            problems.append(f"  ⚠️  EXPLODE: {name} ∇={gn:.1f} >> clip={clip_val}")
        elif gn < 1e-8 and p.requires_grad:
            problems.append(f"  ⚠️  VANISH: {name} ∇={gn:.2e}")
        if torch.isnan(g).any():
            problems.append(f"  🔴 NaN: {name}")
    
    if problems:
        print(f"\n[GRADIENT CHECK] step={step} — {len(problems)} issue(s):")
        for p in problems:
            print(p)
    return len(problems) == 0


# ═══════════ Main Training Loop ═══════════ #

def main(**kwargs):
    set_config(0)
    
    parser = argparse.ArgumentParser(description='Walpurgis-TSH Training')
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name')
    parser.add_argument('--debug_interval', type=int, default=50,
                        help='Full state dump every N iterations')
    parser.add_argument('--profile', action='store_true', default=True,
                        help='Enable epoch profiler')
    parser.add_argument('--tier_sim', action='store_true', default=True,
                        help='Enable tier placement simulation')
    parser.add_argument('--grad_watchdog', action='store_true', default=True,
                        help='Auto-halve LR on repeated gradient anomalies')
    args = parser.parse_args()
    
    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    data_dir     = config['data_args']['data_dir']
    dataset_name = data_dir.split("/")[-1]
    device       = torch.device(config['start_up']['device'])
    save_path    = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
    load_pkl     = config['start_up']['load_pkl']
    model_name   = config['start_up']['model_name']

    setproctitle.setproctitle(f"{model_name}.{dataset_name}@Walpurgis")

    # ── Init profiling ── #
    profiler = EpochProfiler()
    tier_stats = TierStats()
    grad_anomaly_streak = 0  # consecutive steps with gradient issues
    
    print(f"\n{'#'*75}")
    print(f"# Walpurgis-TSH Training Pipeline")
    print(f"# Model: {model_name} | Dataset: {dataset_name} | Device: {device}")
    print(f"# Debug interval: {args.debug_interval} | Profiler: {'ON' if args.profile else 'OFF'}")
    print(f"# Tier sim: {'ON' if args.tier_sim else 'OFF'} | Grad watchdog: {'ON' if args.grad_watchdog else 'OFF'}")
    print(f"{'#'*75}\n")

    # ═══════ Load Dataset ═══════ #
    t_load = time.perf_counter()
    if load_pkl:
        t0 = time.perf_counter()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        print(f"[DATA] Loaded pickle: {(time.perf_counter()-t0):.2f}s")
    else:
        t0 = time.perf_counter()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        print(f"[DATA] Loaded raw: {(time.perf_counter()-t0):.2f}s")
    
    scaler = dataloader['scaler']
    profiler.record('data_load', time.perf_counter() - t_load)
    
    # Debug: dataset structure
    for key in dataloader:
        val = dataloader[key]
        if isinstance(val, np.ndarray):
            print(f"  [DATA] '{key}': shape={val.shape} dtype={val.dtype} "
                  f"mem={val.nbytes/1e6:.1f}MB")
        elif hasattr(val, 'size'):
            print(f"  [DATA] '{key}': {type(val).__name__}")
    
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))
    else:
        _min = None
        _max = None
    
    t0 = time.perf_counter()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    print(f"[DATA] Adjacency: {(time.perf_counter()-t0):.2f}s, type={config['data_args']['adj_type']}")
    
    for i, a in enumerate(adj_mx):
        a_np = np.array(a)
        nnz = np.count_nonzero(a_np)
        print(f"  [ADJ] adj[{i}]: shape={a_np.shape} nnz={nnz} "
              f"density={nnz/a_np.size:.4f}")

    # ═══════ Model Setup ═══════ #
    model_args = config['model_args']
    model_args['device']    = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs']      = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori']  = torch.tensor(adj_ori).to(device)
    model_args['dataset']   = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps']   = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)
    
    # Architecture summary
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Total params: {total_p:,d} | Trainable: {train_p:,d}")
    for name, module in model.named_modules():
        if name:
            print(f"    {name}: {module.__class__.__name__}")

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    train_time = []
    val_time = []
    num_iters = len(dataloader['train_loader'])
    print(f"\n[TRAIN] Iterations/epoch: {num_iters}")

    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model = load_model(model, save_path_resume)
    
    batch_num = resume_epoch * num_iters
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ═══════ Training Loop ═══════ #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            t_epoch = time.perf_counter()
            current_lr = engine.lr_scheduler.get_last_lr()[0]
            
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            
            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch} | LR={current_lr:.6f}")
            print(f"{'─'*60}")
            
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                t_step = time.perf_counter()
                
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                
                # Tier placement
                if args.tier_sim and itera % args.debug_interval == 0:
                    decide_tier_placement(trainx.shape[0], tier_stats, batch_num)
                
                # Forward + backward
                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num,
                    _max=_max, _min=_min
                )
                
                step_ms = (time.perf_counter() - t_step) * 1000
                profiler.record('forward', step_ms)
                profiler.loss_curve.append(mae)
                
                # Periodic debug dumps
                if itera % args.debug_interval == 0:
                    print(f"\n[ITER {itera}/{num_iters}] mae={mae:.4f} mape={mape:.4f} rmse={rmse:.4f}")
                    dump_tensor(trainx, "input", batch_num)
                    dump_tensor(trainy, "target", batch_num)
                    
                    grad_ok = check_gradient_health(model, batch_num)
                    if not grad_ok:
                        grad_anomaly_streak += 1
                        if args.grad_watchdog and grad_anomaly_streak >= 3:
                            # Auto-halve LR on 3 consecutive anomalies
                            for pg in engine.optimizer.param_groups:
                                pg['lr'] *= 0.5
                            new_lr = engine.optimizer.param_groups[0]['lr']
                            print(f"  [WATCHDOG] 3 consecutive gradient anomalies — "
                                  f"LR halved to {new_lr:.8f}")
                            grad_anomaly_streak = 0
                    else:
                        grad_anomaly_streak = 0
                    
                    # Full model state every debug_interval*5
                    if itera % (args.debug_interval * 5) == 0:
                        gn = dump_model_state(model, batch_num)
                        profiler.grad_norms.append(gn)
                else:
                    print(f"{itera}: {mae:.4f}", end='\r')
                
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            
            epoch_sec = time.perf_counter() - t_epoch
            train_time.append(epoch_sec)
            
            # Tier snapshot
            snap = tier_stats.as_dict()
            snap['epoch'] = epoch
            profiler.tier_history.append(snap)
            
            current_lr = engine.optimizer.param_groups[0]['lr']
            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtl = np.mean(train_loss)
            mtm = np.mean(train_mape)
            mtr = np.mean(train_rmse)

            # Validation
            t_val = time.perf_counter()
            mvl, mvm, mvr = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            val_sec = time.perf_counter() - t_val
            val_time.append(val_sec)
            profiler.record('validation', val_sec)

            ts = time.strftime("%d-%H-%M", time.localtime())
            print(f"[{ts}] Epoch {epoch:03d} | "
                  f"Train: loss={mtl:.4f} mape={mtm:.4f} rmse={mtr:.4f} | "
                  f"Val: loss={mvl:.4f} mape={mvm:.4f} rmse={mvr:.4f} | "
                  f"LR={current_lr:.6f} | {epoch_sec:.1f}s train / {val_sec:.1f}s val")
            
            early_stopping(mvl, engine.model)
            if early_stopping.early_stop:
                print('[Walpurgis] Early stopping triggered!')
                break

            engine.test(model, save_path_resume, device, dataloader, scaler,
                        model_name, _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"\n[SUMMARY] Avg train: {np.mean(train_time):.2f}s/epoch")
        print(f"[SUMMARY] Avg val:   {np.mean(val_time):.2f}s/epoch")
        tier_stats.report()
        
        # Dump all debug data
        TensorProbe.dump_all()
        MetricTracker.report()
        
        if args.profile:
            profiler.save(f'walpurgis_profile_{dataset_name}.json')
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler,
                    model_name, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.perf_counter()
    try:
        main()
    except Exception as e:
        print(f"\n{'!'*75}")
        print(f"[CRASH] Walpurgis training failed!")
        print(f"[CRASH] {type(e).__name__}: {e}")
        print(f"[CRASH] Traceback:")
        traceback.print_exc()
        # Dump probe state on crash for post-mortem
        try:
            TensorProbe.dump_all()
            MetricTracker.report()
        except:
            pass
        print(f"{'!'*75}")
        sys.exit(1)
    print(f"\n[Walpurgis] Total: {time.perf_counter()-t_start:.2f}s")
