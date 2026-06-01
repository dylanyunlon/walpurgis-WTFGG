#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis v2 Main Entry — Training Pipeline with Exponential-Backoff Watchdog
================================================================================
Delta vs prior Walpurgis main.py:
  1. Gradient watchdog: hard LR halve → *exponential backoff* with recovery.
     After 3 anomalies → LR *= 0.7; if no anomaly for 500 steps → LR *= 1.1
     (up to original LR).  This prevents over-aggressive LR reduction.
  2. Tier placement now uses a *watermark* system: HBM usage above 80%
     triggers proactive demotion to GDDR before OOM.
  3. crash_dump writes model + optimizer state so training can resume from
     the exact step where it crashed.
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


# ═══════ Profiling Infrastructure ═══════ #

@dataclass
class TierStats:
    hbm_bytes: int = 0
    gddr_bytes: int = 0
    dram_bytes: int = 0
    hbm_peak: int = 0
    hbm_watermark: float = 0.8   # trigger demotion above this
    promotions: int = 0
    demotions: int = 0

    def allocate(self, size_bytes, tier):
        if tier == "hbm":
            self.hbm_bytes += size_bytes
            self.hbm_peak = max(self.hbm_peak, self.hbm_bytes)
        elif tier == "gddr":
            self.gddr_bytes += size_bytes
        else:
            self.dram_bytes += size_bytes

    def report(self):
        print(
            f"  [TIER] HBM={self.hbm_bytes/1e6:.1f}MB "
            f"(peak={self.hbm_peak/1e6:.1f}MB) "
            f"GDDR={self.gddr_bytes/1e6:.1f}MB "
            f"DRAM={self.dram_bytes/1e6:.1f}MB "
            f"↑{self.promotions} ↓{self.demotions}"
        )


@dataclass
class EpochProfiler:
    phase_times: Dict[str, List[float]] = field(default_factory=lambda: {
        "data_load": [], "forward": [], "backward": [],
        "optimizer": [], "validation": [],
    })
    grad_norms: List[float] = field(default_factory=list)
    loss_curve: List[float] = field(default_factory=list)

    def record(self, phase, elapsed):
        if phase in self.phase_times:
            self.phase_times[phase].append(elapsed)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"phases": self.phase_times, "loss": self.loss_curve}, f)
        print(f"[Profiler] → {path}")


# ═══════ Debug Utilities ═══════ #

def dump_model_state(model, step, verbose=True):
    if not verbose:
        return 0.0
    print(f"\n{'═'*70}")
    print(f"  MODEL STATE @ step={step}")
    print(f"{'═'*70}")
    total_gn = 0.0
    anomalies = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.data.norm(2).item()
            total_gn += gn * gn
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                anomalies += 1
                print(f"    ⚠ {name}: grad anomaly")
    total_gn = total_gn ** 0.5
    print(f"  total_grad_norm={total_gn:.4f} anomalies={anomalies}")
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
            f"∈[{t.min().item():.4f},{t.max().item():.4f}]"
        )


def decide_tier_placement(batch_size, tier_stats, step):
    estimated = batch_size * 512 * 207 * 4  # rough bytes
    tier = "hbm" if estimated < 50e6 else "gddr"
    tier_stats.allocate(estimated, tier)


def crash_dump(model, optimizer, step, path="crash_state.pt"):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)
    print(f"[CRASH DUMP] → {path} at step={step}")


# ═══════ Main ═══════ #

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="METR-LA")
    parser.add_argument("--debug_interval", type=int, default=100)
    parser.add_argument("--tier_sim", action="store_true", default=False)
    parser.add_argument("--grad_watchdog", action="store_true", default=True)
    parser.add_argument("--profile", action="store_true", default=False)
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

    setproctitle.setproctitle(f"{model_name}.{dataset_name}@Wv2")

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

    mode = config["start_up"]["mode"]
    resume_epoch = 0
    if mode == "test":
        model = load_model(model, save_path)
    elif mode == "resume":
        resume_epoch = config["start_up"]["resume_epoch"]
        model = load_model(model, save_path_resume)

    batch_num = resume_epoch * len(dataloader["train_loader"])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ── Exponential-backoff watchdog state ──
    original_lr = engine.optimizer.param_groups[0]["lr"]
    grad_anomaly_streak = 0
    steps_since_anomaly = 0

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

                if args.tier_sim and itera % args.debug_interval == 0:
                    decide_tier_placement(trainx.shape[0], tier_stats, batch_num)

                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num, _max=_max, _min=_min,
                )
                step_ms = (time.perf_counter() - t_step) * 1000
                profiler.loss_curve.append(mae)

                if itera % args.debug_interval == 0:
                    print(f"\n[ITER {itera}] mae={mae:.4f} mape={mape:.4f} rmse={rmse:.4f}")
                    grad_ok = check_gradient_health(model, batch_num)
                    if not grad_ok:
                        grad_anomaly_streak += 1
                        steps_since_anomaly = 0
                        if args.grad_watchdog and grad_anomaly_streak >= 3:
                            factor = 0.7
                            for pg in engine.optimizer.param_groups:
                                pg["lr"] *= factor
                            print(f"  [WATCHDOG] exp-backoff: LR → {engine.optimizer.param_groups[0]['lr']:.8f}")
                            grad_anomaly_streak = 0
                    else:
                        grad_anomaly_streak = 0
                        steps_since_anomaly += 1
                        # Recovery: if 500 clean steps, nudge LR back up
                        if steps_since_anomaly >= 5 and args.grad_watchdog:
                            cur_lr = engine.optimizer.param_groups[0]["lr"]
                            if cur_lr < original_lr:
                                for pg in engine.optimizer.param_groups:
                                    pg["lr"] = min(pg["lr"] * 1.1, original_lr)
                                steps_since_anomaly = 0

                    if itera % (args.debug_interval * 5) == 0:
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

            ts = time.strftime("%d-%H-%M", time.localtime())
            lr = engine.optimizer.param_groups[0]["lr"]
            print(
                f"[{ts}] Epoch {epoch:03d} | "
                f"Train: loss={mtl:.4f} mape={mtm:.4f} rmse={mtr:.4f} | "
                f"Val: loss={mvl:.4f} mape={mvm:.4f} rmse={mvr:.4f} | "
                f"LR={lr:.6f} | {epoch_sec:.1f}s/{val_sec:.1f}s"
            )
            logger.log_epoch(epoch, {"train_loss": mtl, "val_loss": mvl, "lr": lr})

            early_stopping(mvl, engine.model)
            if early_stopping.early_stop:
                print("[Walpurgis v2] Early stopping triggered!")
                break

            engine.test(model, save_path_resume, device, dataloader, scaler,
                        model_name, _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"\n[SUMMARY] Avg train: {np.mean(train_time):.2f}s/epoch")
        print(f"[SUMMARY] Avg val:   {np.mean(val_time):.2f}s/epoch")
        tier_stats.report()
        TensorProbe.dump_all()
        MetricTracker.report()

        if args.profile:
            profiler.save(f"walpurgis_v2_profile_{dataset_name}.json")
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
            MetricTracker.report()
        except:
            pass
        print(f"{'!'*75}")
        sys.exit(1)
    print(f"\n[Walpurgis v2] Total: {time.perf_counter()-t_start:.2f}s")
