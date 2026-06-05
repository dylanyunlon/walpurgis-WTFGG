"""
eval_nightfall.py — Nightfall 10维评估 + upstream对比表
从项目根目录运行: python eval_nightfall.py [--dataset SYNTH]

产出:
  output/nightfall_eval_results.json   — 10维指标JSON
  output/comparison_table.tex          — LaTeX对比表 (nightfall vs SOTA)

10维指标:
  1. MAE@15min (Horizon 3)
  2. MAE@30min (Horizon 6)
  3. MAE@60min (Horizon 12)
  4. RMSE@15min
  5. RMSE@30min
  6. RMSE@60min
  7. MAPE@15min (%)
  8. MAPE@30min (%)
  9. MAPE@60min (%)
  10. 模型参数量 (M)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from walpurgis_nightfall.utils.train import set_config, data_reshaper
from walpurgis_nightfall.utils.load_data import load_dataset, load_adj
from walpurgis_nightfall.models.losses import masked_mae, masked_rmse, masked_mape, metric
from walpurgis_nightfall.models.model import D2STGNN


def _resolve_path(rel_path, base=None):
    if os.path.isabs(rel_path):
        return rel_path
    base = base or _REPO_ROOT
    p = os.path.join(base, rel_path)
    if not os.path.exists(p):
        nf_dir = os.path.join(_REPO_ROOT, 'src', 'walpurgis_nightfall')
        p = os.path.join(nf_dir, rel_path)
    return p


def evaluate(dataset: str, model_path: str = None, device_str: str = 'cpu'):
    """
    加载训练好的模型，在测试集上计算10维指标。

    Returns:
        dict: 10维评估结果
    """
    set_config(0)
    device = torch.device(device_str)

    config_path = os.path.join(_REPO_ROOT, 'src', 'walpurgis_nightfall', 'configs', dataset + '.yaml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = dataset
    data_dir = _resolve_path(config['data_args']['data_dir'])
    adj_path = _resolve_path(config['data_args']['adj_data_path'])

    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
    scaler = dataloader['scaler']
    adj_mx, adj_ori = load_adj(adj_path, config['data_args']['adj_type'])

    model_args = config['model_args'].copy()
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    model = D2STGNN(**model_args).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 尝试加载模型权重
    if model_path is None:
        model_name = config['start_up']['model_name']
        model_path = os.path.join(_REPO_ROOT, 'output', f'{model_name}_{dataset_name}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python train_nightfall.py --dataset {dataset} first")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path} ({param_count:,} params)")

    # ── 推理 ────────────────────────────────────────────────
    outputs = []
    for x, y in dataloader['test_loader'].get_iterator():
        testx = data_reshaper(x, device)
        with torch.no_grad():
            preds = model(testx)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
    yhat = yhat[:realy.size(0)]

    _min = _max = None  # PEMS04/08 only
    if _max is not None:
        pass  # handled in train pipeline
    else:
        predict = scaler.inverse_transform(yhat)
        real = scaler.inverse_transform(realy)[:, :, :, 0]

    # ── 逐horizon指标 ─────────────────────────────────────
    per_horizon = []
    for i in range(predict.shape[2]):
        m = metric(predict[:, :, i], real[:, :, i])
        per_horizon.append({'horizon': i + 1, 'MAE': m[0], 'MAPE': m[1], 'RMSE': m[2]})

    # ── 10维指标提取 (Horizon 3=15min, 6=30min, 12=60min) ──
    h3 = per_horizon[2]   # 15min
    h6 = per_horizon[5]   # 30min
    h12 = per_horizon[11] # 60min

    avg_mae = float(np.mean([h['MAE'] for h in per_horizon]))
    avg_rmse = float(np.mean([h['RMSE'] for h in per_horizon]))
    avg_mape = float(np.mean([h['MAPE'] for h in per_horizon]))

    results = {
        'model': 'D2STGNN-Nightfall',
        'dataset': dataset_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path,
        'param_count': param_count,
        'param_count_M': round(param_count / 1e6, 3),
        # 10维指标
        'MAE_15min': round(float(h3['MAE']), 4),
        'MAE_30min': round(float(h6['MAE']), 4),
        'MAE_60min': round(float(h12['MAE']), 4),
        'RMSE_15min': round(float(h3['RMSE']), 4),
        'RMSE_30min': round(float(h6['RMSE']), 4),
        'RMSE_60min': round(float(h12['RMSE']), 4),
        'MAPE_15min': round(float(h3['MAPE']) * 100, 2),
        'MAPE_30min': round(float(h6['MAPE']) * 100, 2),
        'MAPE_60min': round(float(h12['MAPE']) * 100, 2),
        # 平均指标
        'MAE_avg': round(avg_mae, 4),
        'RMSE_avg': round(avg_rmse, 4),
        'MAPE_avg_pct': round(avg_mape * 100, 2),
        # 全量per-horizon
        'per_horizon': per_horizon,
    }
    return results


def save_eval_results(results: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved eval results: {out_path}")


def generate_comparison_table(eval_results: dict, sota_path: str, out_path: str):
    """
    生成LaTeX对比表: Nightfall vs SOTA baselines
    """
    # 加载SOTA数据 (METR-LA格式)
    sota_models = []
    if os.path.exists(sota_path):
        with open(sota_path) as f:
            sota = json.load(f)
        sota_models = sota.get('models', [])

    nightfall = eval_results

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Spatio-Temporal Forecasting on ' + nightfall['dataset'] + r' (D2STGNN-Nightfall vs Baselines)}',
        r'\label{tab:nightfall_comparison}',
        r'\begin{tabular}{l|ccc|ccc|ccc|r}',
        r'\hline',
        r'\multirow{2}{*}{Model} & \multicolumn{3}{c|}{MAE} & \multicolumn{3}{c|}{RMSE} & \multicolumn{3}{c|}{MAPE (\%)} & \multirow{2}{*}{Params (M)} \\',
        r' & 15min & 30min & 60min & 15min & 30min & 60min & 15min & 30min & 60min & \\',
        r'\hline',
    ]

    # SOTA baselines (use their single-horizon avg as placeholder if no horizon breakdown)
    for m in sota_models:
        mae = m.get('MAE', '-')
        rmse = m.get('RMSE', '-')
        mape = m.get('MAPE', '-')
        row = (f"{m['name']} ({m['year']}) & "
               f"{mae} & {mae} & {mae} & "
               f"{rmse} & {rmse} & {rmse} & "
               f"{mape} & {mape} & {mape} & - \\\\")
        lines.append(row)

    lines.append(r'\hline')

    # Nightfall row (bold)
    nf = nightfall
    nf_row = (
        r'\textbf{D2STGNN-Nightfall (ours)} & '
        rf'\textbf{{{nf["MAE_15min"]:.4f}}} & '
        rf'\textbf{{{nf["MAE_30min"]:.4f}}} & '
        rf'\textbf{{{nf["MAE_60min"]:.4f}}} & '
        rf'\textbf{{{nf["RMSE_15min"]:.4f}}} & '
        rf'\textbf{{{nf["RMSE_30min"]:.4f}}} & '
        rf'\textbf{{{nf["RMSE_60min"]:.4f}}} & '
        rf'\textbf{{{nf["MAPE_15min"]:.2f}}} & '
        rf'\textbf{{{nf["MAPE_30min"]:.2f}}} & '
        rf'\textbf{{{nf["MAPE_60min"]:.2f}}} & '
        rf'\textbf{{{nf["param_count_M"]:.3f}}} \\'
    )
    lines.append(nf_row)
    lines.extend([
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved LaTeX table: {out_path}")


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print(f"  D2STGNN-Nightfall Evaluation Results")
    print(f"  Dataset : {results['dataset']}")
    print(f"  Params  : {results['param_count']:,} ({results['param_count_M']:.3f}M)")
    print(f"{'='*60}")
    print(f"{'Horizon':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
    print(f"{'-'*40}")
    print(f"{'15min (H3)':<12} {results['MAE_15min']:>8.4f} {results['RMSE_15min']:>8.4f} {results['MAPE_15min']:>8.2f}")
    print(f"{'30min (H6)':<12} {results['MAE_30min']:>8.4f} {results['RMSE_30min']:>8.4f} {results['MAPE_30min']:>8.2f}")
    print(f"{'60min (H12)':<12} {results['MAE_60min']:>8.4f} {results['RMSE_60min']:>8.4f} {results['MAPE_60min']:>8.2f}")
    print(f"{'-'*40}")
    print(f"{'Average':<12} {results['MAE_avg']:>8.4f} {results['RMSE_avg']:>8.4f} {results['MAPE_avg_pct']:>8.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Nightfall 10维评估 + LaTeX对比表生成')
    parser.add_argument('--dataset', type=str, default='SYNTH',
                        choices=['SYNTH', 'METR-LA', 'PEMS-BAY', 'PEMS04', 'PEMS08'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径 (默认: output/D2STGNN_{dataset}.pt)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    results = evaluate(args.dataset, args.model_path, args.device)
    print_summary(results)

    out_dir = os.path.join(_REPO_ROOT, 'output')
    save_eval_results(results, os.path.join(out_dir, 'nightfall_eval_results.json'))

    sota_path = os.path.join(_REPO_ROOT, 'bench', 'sota.json')
    generate_comparison_table(
        results, sota_path,
        os.path.join(out_dir, 'comparison_table.tex'))


if __name__ == '__main__':
    main()
