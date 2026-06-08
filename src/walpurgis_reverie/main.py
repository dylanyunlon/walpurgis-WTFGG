"""
main.py — Reverie variant entry point
Run from repo root: python -m walpurgis_reverie.main --dataset SYNTH
"""
import argparse
import time
import torch
import pickle
import os
import sys
import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, '..'))

from walpurgis_reverie.utils.train import set_config, EarlyStopping, data_reshaper, load_model
from walpurgis_reverie.utils.load_data import load_dataset, load_adj
from walpurgis_reverie.utils.log import TrainLogger
from walpurgis_reverie.models.losses import masked_mae, masked_rmse, masked_mape, metric
from walpurgis_reverie.models import trainer
from walpurgis_reverie.models.model import D2STGNN
from walpurgis_reverie import _dbg, _is_debug, snapshot_model, register_activation_hooks, gradient_health_check, gradient_histogram, weight_diff, PerfTimer


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SYNTH')
    args = parser.parse_args()

    config_path = os.path.join(_REPO_ROOT, "configs", args.dataset + ".yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_REPO_ROOT, data_dir)
    adj_path = config['data_args']['adj_data_path']
    if not os.path.isabs(adj_path):
        adj_path = os.path.join(_REPO_ROOT, adj_path)

    device = torch.device(config['start_up']['device'])
    dataset_name = os.path.basename(data_dir)
    model_name = config['start_up']['model_name']

    os.makedirs(os.path.join(_REPO_ROOT, '..', '..', 'output'), exist_ok=True)
    save_path = os.path.join(_REPO_ROOT, '..', '..', 'output',
                             f'{model_name}_REVERIE_{dataset_name}.pt')
    save_path_resume = save_path.replace('.pt', '_resume.pt')

    print(f"\n{'='*60}")
    print(f"  Reverie Training Pipeline")
    print(f"  Dataset : {dataset_name}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    t0 = time.time()
    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
    print(f"Loaded dataset: {time.time()-t0:.2f}s")

    scaler = dataloader['scaler']
    _min = _max = None

    t0 = time.time()
    adj_mx, adj_ori = load_adj(adj_path, config['data_args']['adj_type'])
    print(f"Loaded adj: {time.time()-t0:.2f}s")

    model_args = config['model_args'].copy()
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args'].copy()
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count:,}")

    if _is_debug():
        snapshot_model(model, epoch=0, step=0)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    mode = config['start_up']['mode']
    if mode == 'test':
        model = load_model(model, save_path)
    else:
        total_epochs = optim_args['epochs']
        train_time, val_time = [], []
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        for epoch in range(1, total_epochs + 1):
            t_train = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            batch_num = (epoch - 1) * len(dataloader['train_loader'])

            for x, y in dataloader['train_loader'].get_iterator():
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - t_train)

            t_val = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - t_val)

            if engine.lr_scheduler:
                engine.lr_scheduler.step()

            current_lr = engine.optimizer.param_groups[0]['lr']
            curr_time = time.strftime("%d-%H-%M", time.localtime())

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
                'val_mae': mvalid_loss,
                'lr': current_lr,
            })

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            if _is_debug() and epoch % 2 == 0:
                gradient_health_check(model)
                gradient_histogram(model)
                weight_diff(initial_state, model.state_dict())

            engine.test(model, save_path_resume, device, dataloader,
                       scaler, model_name, _max=_max, _min=_min,
                       loss=engine.loss, dataset_name=dataset_name)

        print(f"\nAvg Train: {np.mean(train_time):.4f}s/epoch")
        print(f"Best Val : {early_stopping.val_loss_min:.4f}")
        if _is_debug():
            engine.perf.report()

    print(f"\n[RV] Pipeline complete. Saved: {save_path}")


if __name__ == '__main__':
    t_start = time.time()
    main()
    print(f"Total time: {time.time()-t_start:.2f}s")
