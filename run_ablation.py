"""
run_ablation.py — Phase 2 消融实验 (M376-M400)
在SYNTH上跑3种配置各3 epoch，记录MAE差异。
配置:
  1. baseline     — 完整模型
  2. no_adp_emb   — adp_gate=0 (关闭自适应时空嵌入)
  3. no_temp_attn — bypass output_temporal_attn
"""
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from walpurgis.utils.train import (
    set_config, EarlyStopping, data_reshaper)
from walpurgis.utils.load_data import (
    load_dataset, load_adj)
from walpurgis.utils.log import TrainLogger
from walpurgis.models.losses import masked_mae
from walpurgis.models import trainer
from walpurgis.models.model import D2STGNN


def _resolve_path(rel_path, base=None):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(base or _REPO_ROOT, rel_path)


def build_model_and_data(config_path, seed=42):
    """加载数据和构建基础model_args，返回供反复复用的对象。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = _resolve_path(config['data_args']['data_dir'])
    adj_path = _resolve_path(config['data_args']['adj_data_path'])
    if not os.path.exists(data_dir):
        cas_dir = os.path.join(_REPO_ROOT, 'src', 'walpurgis')
        data_dir = _resolve_path(config['data_args']['data_dir'], cas_dir)
        adj_path = _resolve_path(config['data_args']['adj_data_path'], cas_dir)

    device = torch.device('cpu')
    model_args = config['model_args'].copy()
    optim_args = config['optim_args'].copy()

    dataset_name = os.path.basename(data_dir)
    dataloader = load_dataset(
        data_dir, model_args['batch_size'],
        model_args['batch_size'], model_args['batch_size'],
        dataset_name)

    scaler = dataloader['scaler']
    _max = _min = None

    adj_mx, adj_ori = load_adj(adj_path, config['data_args']['adj_type'])
    model_args.update({
        'adjs': [torch.tensor(a).to(device) for a in adj_mx],
        'adjs_ori': torch.tensor(adj_ori).to(device),
        'num_nodes': adj_mx[0].shape[0],
        'dataset': dataset_name,
        'device': device,
    })
    optim_args['_steps_per_epoch'] = len(dataloader['train_loader'])
    optim_args['cl_steps'] = (
        optim_args['cl_epochs'] * len(dataloader['train_loader']))
    optim_args['warm_steps'] = (
        optim_args['warm_epochs'] * len(dataloader['train_loader']))
    if 'training' in config:
        optim_args['cl_ramp_mode'] = config['training'].get('cl_ramp_mode', 'sigmoid')

    return config, model_args, optim_args, dataloader, scaler, _max, _min, device


def run_config(name, config, model_args, optim_args,
               dataloader, scaler, _max, _min, device,
               seed=42, patch_fn=None):
    """训练一个配置并返回best_val_mae和各epoch MAE列表。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    set_config(0)
    model = D2STGNN(**model_args).to(device)

    # 应用消融patch（如有）
    if patch_fn is not None:
        patch_fn(model)

    total_epochs = optim_args['epochs']
    save_path = os.path.join(_REPO_ROOT, 'output', f'ablation_{name}.pt')
    os.makedirs(os.path.join(_REPO_ROOT, 'output'), exist_ok=True)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    val_maes = []

    for epoch in range(1, total_epochs + 1):
        train_loss = []
        dataloader['train_loader'].shuffle()
        batch_num = (epoch - 1) * len(dataloader['train_loader'])

        for x, y in dataloader['train_loader'].get_iterator():
            trainx = data_reshaper(x, device)
            trainy = data_reshaper(y, device)
            mae, mape, rmse = engine.train(
                trainx, trainy,
                batch_num=batch_num,
                _max=_max, _min=_min)
            train_loss.append(mae)
            batch_num += 1

        engine.step_lr_scheduler(epoch)

        mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
            device, dataloader, name,
            _max=_max, _min=_min)
        val_maes.append(float(mvalid_loss))

        print(f"  [{name}] Epoch {epoch:02d} | "
              f"Train MAE={np.mean(train_loss):.4f} | "
              f"Val MAE={mvalid_loss:.4f}")

        early_stopping(mvalid_loss, engine.model)
        if early_stopping.early_stop:
            print(f"  [{name}] Early stop at epoch {epoch}")
            break

    best_val = early_stopping.val_loss_min
    print(f"  [{name}] Best Val MAE = {best_val:.4f}\n")
    return best_val, val_maes


# ── 消融patch函数 ──────────────────────────────────────────────

def patch_no_adp_emb(model):
    """关闭自适应时空嵌入: 将adp_gate固定为极大负值使sigmoid→0，
    并冻结该参数，使gate_val≈0从而不注入adaptive_embedding。"""
    with torch.no_grad():
        # sigmoid(-20) ≈ 2e-9，等效于gate=0
        model.adp_gate.data.fill_(-20.0)
    model.adp_gate.requires_grad_(False)


def patch_no_temporal_attn(model):
    """bypass output_temporal_attn: 将forward替换为恒等函数。"""
    model.output_temporal_attn.forward = lambda x: x


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    config_path = os.path.join(
        _REPO_ROOT, 'src', 'walpurgis', 'configs', 'SYNTH.yaml')

    print("=" * 60)
    print("  Walpurgis 消融实验 (Phase 2 · M376-M400)")
    print("  Dataset: SYNTH | 每配置3 epoch | seed=42")
    print("=" * 60)

    (config, model_args, optim_args,
     dataloader, scaler, _max, _min, device) = \
        build_model_and_data(config_path, seed=42)

    configs_to_run = [
        ("baseline",      None),
        ("no_adp_emb",    patch_no_adp_emb),
        ("no_temp_attn",  patch_no_temporal_attn),
    ]

    results = {}
    t0 = time.time()

    for cfg_name, patch_fn in configs_to_run:
        print(f"\n─── 配置: {cfg_name} ───")
        best_val, per_epoch = run_config(
            cfg_name, config, model_args, optim_args,
            dataloader, scaler, _max, _min, device,
            seed=42, patch_fn=patch_fn)
        results[cfg_name] = {
            "best_val_mae": round(best_val, 4),
            "per_epoch_val_mae": [round(v, 4) for v in per_epoch],
        }

    # 计算相对基线的MAE差异
    baseline_mae = results["baseline"]["best_val_mae"]
    for name in ("no_adp_emb", "no_temp_attn"):
        delta = round(results[name]["best_val_mae"] - baseline_mae, 4)
        results[name]["delta_vs_baseline"] = delta
        results[name]["pct_change"] = round(
            delta / baseline_mae * 100, 2)

    ablation_doc = {
        "experiment": "ablation",
        "dataset": "SYNTH",
        "epochs_per_config": 3,
        "seed": 42,
        "device": "cpu",
        "baseline_mae": baseline_mae,
        "configs": results,
        "summary": {
            "adaptive_emb_contribution": {
                "delta_mae": results["no_adp_emb"]["delta_vs_baseline"],
                "pct_change": results["no_adp_emb"]["pct_change"],
                "interpretation": (
                    "positive = adaptive_emb helps reduce MAE"
                    if results["no_adp_emb"]["delta_vs_baseline"] > 0
                    else "negative = removing adaptive_emb improves MAE (unexpected)"
                ),
            },
            "temporal_attn_contribution": {
                "delta_mae": results["no_temp_attn"]["delta_vs_baseline"],
                "pct_change": results["no_temp_attn"]["pct_change"],
                "interpretation": (
                    "positive = temporal_attn helps reduce MAE"
                    if results["no_temp_attn"]["delta_vs_baseline"] > 0
                    else "negative = removing temporal_attn improves MAE (unexpected)"
                ),
            },
        },
        "total_time_s": round(time.time() - t0, 1),
        "conducted_by": "Claude-4 M376-M400 (Phase 2 ablation)",
    }

    out_path = os.path.join(
        _REPO_ROOT, 'experiments', 'results', 'ablation.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_doc, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  消融实验完成! 结果写入: {out_path}")
    print(f"  Baseline MAE        : {baseline_mae:.4f}")
    print(f"  w/o adaptive_emb    : "
          f"{results['no_adp_emb']['best_val_mae']:.4f} "
          f"(Δ{results['no_adp_emb']['delta_vs_baseline']:+.4f}, "
          f"{results['no_adp_emb']['pct_change']:+.1f}%)")
    print(f"  w/o temporal_attn   : "
          f"{results['no_temp_attn']['best_val_mae']:.4f} "
          f"(Δ{results['no_temp_attn']['delta_vs_baseline']:+.4f}, "
          f"{results['no_temp_attn']['pct_change']:+.1f}%)")
    print("=" * 60)

    return ablation_doc


if __name__ == '__main__':
    main()
