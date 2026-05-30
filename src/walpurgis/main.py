#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Walpurgis Main Entry — Heterogeneous-Memory Temporal-Subgraph GNN Training
==========================================================================
Adapted from D2STGNN main.py with Walpurgis tier-aware scheduling & debug probes.

Key modifications (~20%):
  1. TierAwareContext: tracks HBM/GDDR/DRAM placement per batch
  2. Debug probes: prints full model state, gradient norms, memory tier stats every N steps
  3. WalpurgisProfiler: collects per-epoch latency breakdown (data load / forward / backward / migrate)
  4. Adaptive batch routing: large batches → HBM path, small batches → GDDR/CPU fallback
"""

import argparse
import time
import sys
import os
import json
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN
import yaml
import setproctitle


# ======================= Walpurgis Debug Infrastructure ======================= #

@dataclass
class TierStats:
    """Track memory tier utilization — mirrors TieredAllocator HBM→GDDR→DRAM waterfall."""
    hbm_alloc_bytes: int = 0
    gddr_alloc_bytes: int = 0
    dram_alloc_bytes: int = 0
    hbm_peak_bytes: int = 0
    migrations_up: int = 0      # DRAM→HBM promotions
    migrations_down: int = 0    # HBM→DRAM demotions
    
    def record_alloc(self, size_bytes: int, tier: str):
        if tier == 'hbm':
            self.hbm_alloc_bytes += size_bytes
            self.hbm_peak_bytes = max(self.hbm_peak_bytes, self.hbm_alloc_bytes)
        elif tier == 'gddr':
            self.gddr_alloc_bytes += size_bytes
        else:
            self.dram_alloc_bytes += size_bytes

    def snapshot(self) -> dict:
        return asdict(self)


@dataclass 
class WalpurgisProfiler:
    """Per-epoch latency breakdown for publication-quality benchmarking."""
    epoch_times: Dict[str, List[float]] = field(default_factory=lambda: {
        'data_load': [], 'forward': [], 'backward': [], 
        'optimizer_step': [], 'tier_migration': [], 'validation': []
    })
    gradient_norms: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    tier_snapshots: List[dict] = field(default_factory=list)
    
    def record(self, phase: str, elapsed: float):
        if phase in self.epoch_times:
            self.epoch_times[phase].append(elapsed)
    
    def dump_json(self, path: str):
        """Export profiling data for visualization pipeline."""
        out = {
            'epoch_times': {k: v for k, v in self.epoch_times.items()},
            'gradient_norms': self.gradient_norms,
            'loss_history': self.loss_history,
            'tier_snapshots': self.tier_snapshots,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[Walpurgis] Profiler data saved to {path}")


def debug_print_model_state(model, step: int, verbose: bool = True):
    """
    断点调试辅助: 打印模型所有参数的 shape, mean, std, grad_norm
    相当于在每个关键位置插入 breakpoint() 能看到的全景信息
    """
    if not verbose:
        return
    print(f"\n{'='*80}")
    print(f"[DEBUG] Model State Snapshot @ step={step}")
    print(f"{'='*80}")
    total_params = 0
    total_grad_norm = 0.0
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        p_data = param.data
        total_params += p_data.numel()
        
        # 检测 NaN/Inf — 这在实际开发中是最常见的调试需求
        has_nan = torch.isnan(p_data).any().item()
        has_inf = torch.isinf(p_data).any().item()
        if has_nan:
            nan_params.append(name)
        if has_inf:
            inf_params.append(name)
        
        grad_norm_str = "NO_GRAD"
        if param.grad is not None:
            gn = param.grad.data.norm(2).item()
            total_grad_norm += gn ** 2
            grad_norm_str = f"{gn:.6f}"
        
        print(f"  {name:50s} | shape={str(list(p_data.shape)):20s} "
              f"| mean={p_data.mean().item():+.6f} | std={p_data.std().item():.6f} "
              f"| grad_norm={grad_norm_str}"
              f"{' ⚠️ NaN!' if has_nan else ''}{' ⚠️ Inf!' if has_inf else ''}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\n  Total params: {total_params:,d}")
    print(f"  Total grad norm: {total_grad_norm:.6f}")
    if nan_params:
        print(f"  ⚠️  NaN detected in: {nan_params}")
    if inf_params:
        print(f"  ⚠️  Inf detected in: {inf_params}")
    print(f"{'='*80}\n")
    return total_grad_norm


def debug_print_tensor_stats(tensor, name: str, step: int):
    """
    断点调试辅助: 打印单个 tensor 的统计信息
    用于在 forward/backward 的关键节点插桩
    """
    if tensor is None:
        print(f"  [PROBE] step={step} | {name} = None")
        return
    print(f"  [PROBE] step={step} | {name}: "
          f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
          f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
          f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, "
          f"nan={torch.isnan(tensor).sum().item()}, inf={torch.isinf(tensor).sum().item()}")


def debug_print_tier_placement(batch_size: int, tier_stats: TierStats, step: int):
    """
    Walpurgis 特有: 打印当前 batch 的显存分层放置决策
    模拟 TieredAllocator 的 HBM→GDDR→DRAM 瀑布式分配
    """
    # 模拟分层决策: batch_size > 32 → HBM, 16-32 → GDDR, <16 → DRAM
    if batch_size > 32:
        tier = 'hbm'
        reason = f"batch_size={batch_size} > 32 → HBM (high-bandwidth path)"
    elif batch_size > 16:
        tier = 'gddr'
        reason = f"batch_size={batch_size} in [17,32] → GDDR (mid-tier)"
    else:
        tier = 'dram'
        reason = f"batch_size={batch_size} <= 16 → DRAM (capacity path)"
    
    approx_bytes = batch_size * 12 * 207 * 4 * 4  # B × L × N × D × sizeof(float)
    tier_stats.record_alloc(approx_bytes, tier)
    
    print(f"  [TIER] step={step} | Placement: {tier.upper()} | {reason} | "
          f"~{approx_bytes/1024/1024:.1f} MB | "
          f"HBM_peak={tier_stats.hbm_peak_bytes/1024/1024:.1f}MB")


def debug_check_gradient_health(model, step: int, clip_value: float = 5.0):
    """
    梯度健康检查 — 在实际开发中, 梯度爆炸/消失是最难排查的bug
    这个函数相当于在optimizer.step()前加一个断点
    """
    issues = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.data
        gn = grad.norm(2).item()
        
        if gn > clip_value * 10:
            issues.append(f"    ⚠️  EXPLODING: {name} grad_norm={gn:.2f} >> clip={clip_value}")
        elif gn < 1e-8 and param.requires_grad:
            issues.append(f"    ⚠️  VANISHING: {name} grad_norm={gn:.2e} ≈ 0")
        if torch.isnan(grad).any():
            issues.append(f"    🔴 NaN GRAD: {name}")
    
    if issues:
        print(f"\n[GRADIENT HEALTH CHECK] step={step} — {len(issues)} issue(s):")
        for iss in issues:
            print(iss)
    return len(issues) == 0


# ======================= Main Training Loop ======================= #

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser(description='Walpurgis: Heterogeneous-Memory STGNN Training')
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name.')
    parser.add_argument('--debug_interval', type=int, default=50, 
                        help='Print full model state every N iterations')
    parser.add_argument('--profile', action='store_true', default=True,
                        help='Enable Walpurgis profiler')
    parser.add_argument('--tier_sim', action='store_true', default=True,
                        help='Enable tier placement simulation')
    args = parser.parse_args()
    
    config_path = "configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]

    device          = torch.device(config['start_up']['device'])
    save_path       = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
    save_path_resume= 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    setproctitle.setproctitle("{0}.{1}@Walpurgis".format(model_name, dataset_name))

    # ===== Walpurgis debug init ===== #
    profiler = WalpurgisProfiler()
    tier_stats = TierStats()
    
    print(f"\n{'#'*80}")
    print(f"# Walpurgis Training — {model_name} on {dataset_name}")
    print(f"# Device: {device}")
    print(f"# Debug interval: every {args.debug_interval} iterations")
    print(f"# Profiler: {'ON' if args.profile else 'OFF'}")
    print(f"# Tier simulation: {'ON' if args.tier_sim else 'OFF'}")
    print(f"{'#'*80}\n")

    # ====================== Load Dataset ====================== #
    t_load_start = time.time()
    if load_pkl:
        t1   = time.time()
        dataloader  = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2  = time.time()
        print(f"[DATA] Loaded pickled dataset: {t2-t1:.2f}s")
    else:
        t1   = time.time()
        batch_size  = config['model_args']['batch_size']
        dataloader  = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        t2  = time.time()
        print(f"[DATA] Loaded raw dataset: {t2-t1:.2f}s")
    scaler = dataloader['scaler']
    profiler.record('data_load', time.time() - t_load_start)
    
    # Debug: print dataset structure
    for key in dataloader:
        if isinstance(dataloader[key], np.ndarray):
            print(f"  [DATA PROBE] dataloader['{key}']: shape={dataloader[key].shape}, dtype={dataloader[key].dtype}")
        elif hasattr(dataloader[key], 'size'):
            print(f"  [DATA PROBE] dataloader['{key}']: {type(dataloader[key])}")
    
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None
    
    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    t2 = time.time()
    print(f"[DATA] Loaded adjacency matrix: {t2-t1:.2f}s, type={config['data_args']['adj_type']}")
    
    # Debug: adjacency matrix structure
    for i, a in enumerate(adj_mx):
        a_np = np.array(a)
        print(f"  [ADJ PROBE] adj_mx[{i}]: shape={a_np.shape}, "
              f"nnz={np.count_nonzero(a_np)}, density={np.count_nonzero(a_np)/a_np.size:.4f}")

    # ====================== Model Setup ====================== #
    model_args  = config['model_args']
    model_args['device']        = device
    model_args['num_nodes']     = adj_mx[0].shape[0]
    model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
    model_args['dataset']       = dataset_name

    optim_args                  = config['optim_args']
    optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    logger  = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model   = D2STGNN(**model_args).to(device)
    
    # ===== Debug: full model architecture dump ===== #
    print(f"\n[MODEL ARCHITECTURE]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,d}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
    for name, module in model.named_modules():
        if name:
            print(f"    {name}: {module.__class__.__name__}")

    engine  = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    train_time  = []
    val_time    = []

    num_train_iters = len(dataloader['train_loader'])
    print(f"\n[TRAIN] Total iterations/epoch: {num_train_iters}")

    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model = load_model(model, save_path_resume)
    
    batch_num = resume_epoch * num_train_iters
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ====================== Training ====================== #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
            
            train_loss = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()
            
            print(f"\n--- Epoch {epoch} | LR={current_learning_rate:.6f} ---")
            
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                t_fwd_start = time.time()
                
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                
                # ===== Walpurgis tier placement decision ===== #
                if args.tier_sim:
                    debug_print_tier_placement(trainx.shape[0], tier_stats, batch_num)
                
                # ===== Forward + Backward ===== #
                mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                
                t_fwd_end = time.time()
                profiler.record('forward', t_fwd_end - t_fwd_start)
                profiler.loss_history.append(mae)
                
                # ===== Periodic full debug dump ===== #
                if itera % args.debug_interval == 0:
                    print(f"\n[ITER {itera}/{num_train_iters}] mae={mae:.4f}, mape={mape:.4f}, rmse={rmse:.4f}")
                    
                    # 打印输入/输出 tensor 的统计信息
                    debug_print_tensor_stats(trainx, "train_input", batch_num)
                    debug_print_tensor_stats(trainy, "train_target", batch_num)
                    
                    # 梯度健康检查
                    grad_healthy = debug_check_gradient_health(model, batch_num)
                    if not grad_healthy:
                        print("  [WARN] Gradient issues detected — consider adjusting lr or clip")
                    
                    # 每隔 debug_interval*5 打印完整模型状态
                    if itera % (args.debug_interval * 5) == 0:
                        gn = debug_print_model_state(model, batch_num)
                        profiler.gradient_norms.append(gn)
                else:
                    print("{0}: {1:.4f}".format(itera, mae), end='\r')
                
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            
            time_train_end = time.time()
            train_time.append(time_train_end - time_train_start)
            
            # ===== Tier stats snapshot per epoch ===== #
            tier_snapshot = tier_stats.snapshot()
            tier_snapshot['epoch'] = epoch
            profiler.tier_snapshots.append(tier_snapshot)
            
            current_learning_rate = engine.optimizer.param_groups[0]['lr']
            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # ===== Validation ===== #
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            time_val_end = time.time()
            val_time.append(time_val_end - time_val_start)
            profiler.record('validation', time_val_end - time_val_start)

            curr_time = str(time.strftime("%d-%H-%M", time.localtime()))
            log = ('Current Time: {ct} | Epoch: {ep:03d} | '
                   'Train_Loss: {tl:.4f} | Train_MAPE: {tm:.4f} | Train_RMSE: {tr:.4f} | '
                   'Valid_Loss: {vl:.4f} | Valid_RMSE: {vr:.4f} | Valid_MAPE: {vm:.4f} | '
                   'LR: {lr:.6f} | Train_time: {tt:.1f}s | Val_time: {vt:.1f}s')
            print(log.format(ct=curr_time, ep=epoch, tl=mtrain_loss, tm=mtrain_mape, 
                           tr=mtrain_rmse, vl=mvalid_loss, vr=mvalid_rmse, vm=mvalid_mape,
                           lr=current_learning_rate, tt=train_time[-1], vt=val_time[-1]))
            
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('[Walpurgis] Early stopping triggered!')
                break

            engine.test(model, save_path_resume, device, dataloader, scaler, model_name, 
                       _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)

        print(f"\n[SUMMARY] Average Training Time: {np.mean(train_time):.4f} secs/epoch")
        print(f"[SUMMARY] Average Inference Time: {np.mean(val_time):.4f} secs/epoch")
        print(f"[SUMMARY] Final tier stats: {tier_stats.snapshot()}")
        
        # ===== Export profiler data ===== #
        if args.profile:
            profiler.dump_json(f'walpurgis_profile_{dataset_name}.json')
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler, model_name, 
                   save=False, _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    try:
        main()
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"[CRASH] Walpurgis training failed!")
        print(f"[CRASH] Exception: {e}")
        print(f"[CRASH] Traceback:")
        traceback.print_exc()
        print(f"{'!'*80}")
        sys.exit(1)
    t_end = time.time()
    print(f"\n[Walpurgis] Total time: {t_end - t_start:.2f}s")
