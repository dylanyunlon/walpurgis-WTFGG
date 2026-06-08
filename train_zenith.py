"""
train_zenith.py — Zenith训练入口
从项目根目录运行: python train_zenith.py --dataset SYNTH --debug

算法特性 (vs upstream):
  - 频域衰减门控 (spectral decay gating)
  - layer-wise attention聚合
  - GELU输出头 + 残差shortcut
  - OneCycleLR (cosine + built-in warmup)
  - temporal_smoothness正则
  - 可学习残差缩放 per layer
  - 完整梯度直方图 + activation probe
"""
import argparse
import os
import sys
import time
import pickle

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from walpurgis_zenith.utils.train import (
    set_config, EarlyStopping, data_reshaper, load_model)
from walpurgis_zenith.utils.load_data import (
    load_dataset, load_adj)
from walpurgis_zenith.utils.log import TrainLogger
from walpurgis_zenith.models.losses import (
    masked_mae, masked_rmse, masked_mape, metric)
from walpurgis_zenith.models import trainer
from walpurgis_zenith.models.model import D2STGNN
from walpurgis_zenith import (
    _dbg, _is_debug, snapshot_model,
    register_activation_hooks,
    gradient_health_check, gradient_histogram,
    weight_diff, PerfTimer)


def _resolve_path(rel_path, base=None):
    if os.path.isabs(rel_path):
        return rel_path
    base = base or _REPO_ROOT
    return os.path.join(base, rel_path)


def run(dataset, device_str='cpu', epochs_override=None,
        debug=False):
    if debug:
        os.environ['ZENITH_DEBUG'] = '1'

    set_config(0)
    config_dir = os.path.join(
        _REPO_ROOT, 'src', 'walpurgis_zenith', 'configs')
    config_path = os.path.join(config_dir, dataset + '.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = _resolve_path(config['data_args']['data_dir'])
    adj_path = _resolve_path(
        config['data_args']['adj_data_path'])

    if not os.path.exists(data_dir):
        nf_dir = os.path.join(
            _REPO_ROOT, 'src', 'walpurgis_zenith')
        data_dir = _resolve_path(
            config['data_args']['data_dir'], nf_dir)
        adj_path = _resolve_path(
            config['data_args']['adj_data_path'], nf_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}")

    device = torch.device(device_str)
    dataset_name = os.path.basename(data_dir)
    model_name = config['start_up']['model_name']
    mode = config['start_up']['mode']

    os.makedirs(os.path.join(_REPO_ROOT, 'output'),
                exist_ok=True)
    save_path = os.path.join(
        _REPO_ROOT, 'output',
        f'{model_name}_{dataset_name}.pt')
    save_path_resume = os.path.join(
        _REPO_ROOT, 'output',
        f'{model_name}_{dataset_name}_resume.pt')

    print(f"\n{'='*60}")
    print(f"  Zenith Training Pipeline")
    print(f"  Dataset : {dataset_name}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {mode}")
    print(f"  Debug   : {_is_debug()}")
    print(f"{'='*60}\n")

    t0 = time.time()
    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(
        data_dir, batch_size, batch_size,
        batch_size, dataset_name)
    print(f"Loaded dataset: {time.time()-t0:.2f}s")

    scaler = dataloader['scaler']
    _min = _max = None
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(
            open(os.path.join(data_dir, 'min.pkl'), 'rb'))
        _max = pickle.load(
            open(os.path.join(data_dir, 'max.pkl'), 'rb'))

    t0 = time.time()
    adj_mx, adj_ori = load_adj(
        adj_path, config['data_args']['adj_type'])
    print(f"Loaded adj: {time.time()-t0:.2f}s")

    model_args = config['model_args'].copy()
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [
        torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(
        adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args'].copy()
    if epochs_override is not None:
        optim_args['epochs'] = epochs_override
    optim_args['cl_steps'] = (
        optim_args['cl_epochs'] *
        len(dataloader['train_loader']))
    optim_args['warm_steps'] = (
        optim_args['warm_epochs'] *
        len(dataloader['train_loader']))
    optim_args['_steps_per_epoch'] = len(
        dataloader['train_loader'])

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(
        model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)
    param_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    print(f"Model params: {param_count:,}")

    initial_state = {
        k: v.clone() for k, v in model.state_dict().items()}

    if _is_debug():
        snapshot_model(model, epoch=0, step=0)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(
        optim_args['patience'], save_path)

    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        if os.path.exists(save_path_resume):
            model = load_model(model, save_path_resume)
        batch_num_init = (
            resume_epoch * len(dataloader['train_loader']))
        engine.set_resume_lr_and_cl(
            resume_epoch, batch_num_init)

    # 训练前activation probe
    if _is_debug() and mode != 'test':
        print("\n[ZN] === Activation probe ===")
        tracker = register_activation_hooks(model)
        model.train()
        for x, y in dataloader[
                'train_loader'].get_iterator():
            probe_x = data_reshaper(x, device)
            with torch.no_grad():
                _ = model(probe_x)
            break
        tracker.report()
        tracker.remove()
        print("[ZN] === Probe complete ===\n")

    total_epochs = optim_args['epochs']
    train_time, val_time = [], []

    if mode != 'test':
        for epoch in range(
                resume_epoch + 1, total_epochs + 1):
            t_train = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            batch_num = (
                (epoch - 1) *
                len(dataloader['train_loader']))

            for x, y in dataloader[
                    'train_loader'].get_iterator():
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy,
                    batch_num=batch_num,
                    _max=_max, _min=_min)
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - t_train)

            t_val = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = \
                engine.eval(
                    device, dataloader, model_name,
                    _max=_max, _min=_min)
            val_time.append(time.time() - t_val)

            current_lr = engine.optimizer.param_groups[0]['lr']
            import time as _time
            curr_time = _time.strftime(
                "%d-%H-%M", _time.localtime())
            log = (f'{curr_time} | Epoch {epoch:03d} | '
                   f'Train MAE={np.mean(train_loss):.4f} '
                   f'MAPE={np.mean(train_mape):.4f} '
                   f'RMSE={np.mean(train_rmse):.4f} | '
                   f'Val MAE={mvalid_loss:.4f} '
                   f'RMSE={mvalid_rmse:.4f} '
                   f'MAPE={mvalid_mape:.4f} | '
                   f'LR={current_lr:.6f}')
            print(log)

            logger.log_epoch(epoch, {
                'train_mae': np.mean(train_loss),
                'train_mape': np.mean(train_mape),
                'train_rmse': np.mean(train_rmse),
                'val_mae': mvalid_loss,
                'val_mape': mvalid_mape,
                'val_rmse': mvalid_rmse,
                'lr': current_lr,
            })

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            if _is_debug() and epoch % 2 == 0:
                gradient_health_check(model)
                weight_diff(initial_state,
                            model.state_dict())

            engine.test(
                model, save_path_resume, device,
                dataloader, scaler, model_name,
                _max=_max, _min=_min,
                loss=engine.loss,
                dataset_name=dataset_name)

        print(f"\nAvg Train: "
              f"{np.mean(train_time):.4f}s/epoch")
        print(f"Avg Val  : "
              f"{np.mean(val_time):.4f}s/epoch")
        print(f"Best Val : "
              f"{early_stopping.val_loss_min:.4f}")

        if _is_debug():
            engine.perf.report()

    else:
        engine.test(
            model, save_path_resume, device,
            dataloader, scaler, model_name,
            save=False, _max=_max, _min=_min,
            loss=engine.loss,
            dataset_name=dataset_name)

    print(f"\n[ZN] Pipeline complete. Saved: {save_path}")
    return early_stopping.val_loss_min


def main():
    parser = argparse.ArgumentParser(
        description='Zenith (D2STGNN) Training')
    parser.add_argument('--dataset', type=str,
                        default='SYNTH',
                        choices=['SYNTH', 'METR-LA',
                                 'PEMS-BAY', 'PEMS04',
                                 'PEMS08'])
    parser.add_argument('--device', type=str,
                        default='cpu')
    parser.add_argument('--epochs', type=int,
                        default=None)
    parser.add_argument('--debug',
                        action='store_true')
    args = parser.parse_args()
    t_start = time.time()
    run(args.dataset, args.device,
        args.epochs, args.debug)
    print(f"Total time: {time.time()-t_start:.2f}s")


if __name__ == '__main__':
    main()
