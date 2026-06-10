"""
run_multi_seed.py — Phase 2 多种子评估 (M401-M425)
在SYNTH上跑 SEED=42, 123, 456 各3 epoch，写入 experiments/results/multi_seed.json。
直接复用 run_ablation.py 的 trainer engine 模式。
"""
import os
import sys
import time
import json
import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from walpurgis.utils.train import (
    set_config, EarlyStopping, data_reshaper)
from walpurgis.utils.load_data import (
    load_dataset, load_adj)
from walpurgis.models import trainer
from walpurgis.models.model import D2STGNN


def _resolve_path(rel_path, base=None):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(base or _REPO_ROOT, rel_path)


def build_model_and_data(config_path, seed):
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

    return (config, model_args, optim_args,
            dataloader, scaler, _max, _min, device)


def run_one_seed(seed, config, model_args, optim_args,
                 dataloader, scaler, _max, _min, device,
                 num_epochs=3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = D2STGNN(**model_args).to(device)

    save_path = os.path.join(_REPO_ROOT, 'output', f'multi_seed_{seed}.pt')
    os.makedirs(os.path.join(_REPO_ROOT, 'output'), exist_ok=True)

    engine = trainer(dataloader['scaler'], model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    val_maes = []

    for epoch in range(1, num_epochs + 1):
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
            device, dataloader, f'seed_{seed}',
            _max=_max, _min=_min)
        val_maes.append(round(float(mvalid_loss), 4))

        print(f"  [seed={seed}] Epoch {epoch:02d} | "
              f"Train MAE={np.mean(train_loss):.4f} | "
              f"Val MAE={mvalid_loss:.4f}")

        early_stopping(mvalid_loss, engine.model)
        if early_stopping.early_stop:
            print(f"  [seed={seed}] Early stop at epoch {epoch}")
            break

    best_val = early_stopping.val_loss_min
    print(f"  [seed={seed}] Best Val MAE = {best_val:.4f}\n")
    return best_val, val_maes


def main():
    config_path = os.path.join(
        _REPO_ROOT, 'src', 'walpurgis', 'configs', 'SYNTH.yaml')

    seeds = [42, 123, 456]
    num_epochs = 3

    print("=" * 60)
    print("  Walpurgis 多种子评估 (Phase 2 · M401-M425)")
    print(f"  Dataset: SYNTH | {num_epochs} epoch/seed | seeds={seeds}")
    print("=" * 60)

    results = {}
    t0 = time.time()

    for seed in seeds:
        print(f"\n─── Seed = {seed} ───")
        (config, model_args, optim_args,
         dataloader, scaler, _max, _min, device) = \
            build_model_and_data(config_path, seed=seed)

        best_val, per_epoch = run_one_seed(
            seed, config, model_args, optim_args,
            dataloader, scaler, _max, _min, device,
            num_epochs=num_epochs)

        results[f"seed_{seed}"] = {
            "seed": seed,
            "best_val_mae": round(best_val, 4),
            "per_epoch_val_mae": per_epoch,
        }

    # 以seed=42为基准，计算相对变化
    baseline_mae = results["seed_42"]["best_val_mae"]
    for key, v in results.items():
        if v["seed"] == 42:
            v["delta_vs_seed42"] = 0.0
            v["pct_change_vs_seed42"] = 0.0
        else:
            delta = round(v["best_val_mae"] - baseline_mae, 4)
            v["delta_vs_seed42"] = delta
            v["pct_change_vs_seed42"] = round(delta / baseline_mae * 100, 2)

    all_best = [results[f"seed_{s}"]["best_val_mae"] for s in seeds]
    multi_seed_doc = {
        "experiment": "multi_seed",
        "dataset": "SYNTH",
        "epochs_per_seed": num_epochs,
        "seeds_run": seeds,
        "device": "cpu",
        "results": results,
        "summary": {
            "all_seeds_best_mae": {
                f"seed_{s}": results[f"seed_{s}"]["best_val_mae"]
                for s in seeds
            },
            "mean_best_mae": round(float(np.mean(all_best)), 4),
            "std_best_mae": round(float(np.std(all_best)), 4),
            "min_best_mae": round(float(np.min(all_best)), 4),
            "max_best_mae": round(float(np.max(all_best)), 4),
            "interpretation": (
                "Multi-seed SYNTH validation (3 epoch CPU). "
                "METR-LA full 200-epoch runs require server (ags1 GPU). "
                "See experiments/run_server_experiment.sh for server instructions."
            ),
        },
        "metrla_note": {
            "status": "requires_server",
            "reason": "METR-LA has 207 nodes x 23974 samples; "
                      "full training needs GPU (ags1 A6000/H100).",
            "server_cmd": "for SEED in 42 123 456; do SEED=$SEED GPU=2 EPOCHS=200 "
                          "bash experiments/run_server_experiment.sh; done",
            "seed42_verified_mae": 2.93,
            "seed42_source": "walpurgis_metrla_verified.json (14 epochs on ags1)"
        },
        "total_time_s": round(time.time() - t0, 1),
        "conducted_by": "Claude-5 M401-M425 (Phase 2 multi-seed)",
    }

    out_path = os.path.join(
        _REPO_ROOT, 'experiments', 'results', 'multi_seed.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(multi_seed_doc, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  多种子实验完成! 结果写入: {out_path}")
    for s in seeds:
        r = results[f"seed_{s}"]
        delta_str = f"(Δ{r['delta_vs_seed42']:+.4f})" if r['seed'] != 42 else "(baseline)"
        print(f"  seed={s:<4}  best_val_MAE={r['best_val_mae']:.4f}  {delta_str}")
    print(f"  mean±std : {multi_seed_doc['summary']['mean_best_mae']:.4f} "
          f"± {multi_seed_doc['summary']['std_best_mae']:.4f}")
    print(f"  [METR-LA] seed=42 verified MAE=2.93 (server run). "
          f"Full multi-seed needs ags1 GPU.")
    print("=" * 60)

    return multi_seed_doc


if __name__ == '__main__':
    main()
