"""
train_parallax.py — Parallax训练入口 (M054)
从项目根目录运行: python train_parallax.py --dataset SYNTH --debug

算法特性 (vs upstream D2STGNN):
  - Bayesian MC-Dropout EstimationGate: Monte Carlo不确定性估计门控
  - STL/LOESS ResidualDecomp: 可学习趋势+FFT季节+残差分解
  - MixHop多分辨率GCN: 混合不同跳数邻居信息, 可学习hop权重
  - KDE核密度估计距离: RBF+Laplacian双核, 可学习带宽
  - REINFORCE重要性掩码: 策略梯度+Bernoulli采样+熵正则
  - Spectral Clustering归一化: 拉普拉斯特征分解→簇感知缩放
  - xLSTM: 指数门控+归一化器+记忆混合(扩展LSTM)
  - Positional Interpolation: 比例缩放位置编码, 支持序列外推
  - Mixture Output Router: 路由网络动态分配各层权重
  - Cauchy Loss: log(1+(x/s)^2)重尾鲁棒损失
  - Prodigy优化器: dual averaging自适应学习率
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

from walpurgis_parallax.utils.train import (
    set_config, EarlyStopping, data_reshaper, load_model)
from walpurgis_parallax.utils.load_data import (
    load_dataset, load_adj)
from walpurgis_parallax.utils.log import TrainLogger
from walpurgis_parallax.models.losses import (
    masked_mae, masked_rmse, masked_mape, metric)
from walpurgis_parallax.models.trainer import trainer
from walpurgis_parallax.models.model import D2STGNN
from walpurgis_parallax import (
    _dbg, _is_debug, dump_struct_state,
    gradient_health_check, PerfTimer)


def _resolve_path(rel_path, base=None):
    if os.path.isabs(rel_path):
        return rel_path
    base = base or _REPO_ROOT
    return os.path.join(base, rel_path)


def run(dataset, device_str='cpu', epochs_override=None,
        debug=False):
    if debug:
        os.environ['PARALLAX_DEBUG'] = '1'

    set_config(0)
    config_dir = os.path.join(
        _REPO_ROOT, 'src', 'walpurgis_parallax', 'configs')
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
        par_dir = os.path.join(
            _REPO_ROOT, 'src', 'walpurgis_parallax')
        data_dir = _resolve_path(
            config['data_args']['data_dir'], par_dir)
        adj_path = _resolve_path(
            config['data_args']['adj_data_path'], par_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}")

    device = torch.device(device_str)
    dataset_name = os.path.basename(data_dir)
    model_name = config['start_up']['model_name']
    mode = config['start_up']['mode']

    os.makedirs(
        os.path.join(_REPO_ROOT, 'output'), exist_ok=True)
    save_path = os.path.join(
        _REPO_ROOT, 'output',
        f'{model_name}_{dataset_name}.pt')
    save_path_resume = os.path.join(
        _REPO_ROOT, 'output',
        f'{model_name}_{dataset_name}_resume.pt')

    print(f"\n{'='*60}")
    print(f"  Parallax (D2STGNN) Training Pipeline — M054")
    print(f"  Dataset : {dataset_name}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {mode}")
    print(f"  Debug   : {_is_debug()}")
    print(f"  Algos   : MC-Dropout Gate, STL/LOESS, MixHop,")
    print(f"            KDE Distance, REINFORCE Mask,")
    print(f"            Spectral Normalizer, xLSTM+PosInterp,")
    print(f"            MixtureRouter, Cauchy Loss, Prodigy")
    print(f"{'='*60}\n")

    timer = PerfTimer("parallax_pipeline")
    timer.start("data_load")
    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(
        data_dir, batch_size, batch_size,
        batch_size, dataset_name)
    timer.stop("data_load")
    print(f"Loaded dataset: {timer.summary()}")

    scaler = dataloader['scaler']
    _min = _max = None

    timer.start("adj_load")
    adj_mx, adj_ori = load_adj(
        adj_path, config['data_args']['adj_type'])
    timer.stop("adj_load")
    print(f"Loaded adj: {timer.summary()}")

    model_args = config['model_args'].copy()
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [
        torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args'].copy()
    if epochs_override is not None:
        optim_args['epochs'] = epochs_override
    optim_args['cl_steps'] = (
        optim_args['cl_epochs']
        * len(dataloader['train_loader']))
    optim_args['warm_steps'] = (
        optim_args['warm_epochs']
        * len(dataloader['train_loader']))
    optim_args['total_batches'] = len(
        dataloader['train_loader'])
    # Parallax特有参数
    optim_args['cauchy_scale'] = config['model_args'].get(
        'cauchy_scale', 1.0)
    optim_args['prodigy_beta3'] = config['model_args'].get(
        'prodigy_beta3', 0.0)

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(
        model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)
    param_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    print(f"Model params: {param_count:,}")

    dump_struct_state("model_init",
        param_count=param_count, num_layers=5,
        hidden_dim=model_args['num_hidden'],
        forecast_dim=256, device=str(device),
        variant="parallax_m054")

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

    total_epochs = optim_args['epochs']
    train_time, val_time = [], []

    if mode != 'test':
        for epoch in range(resume_epoch + 1,
                           total_epochs + 1):
            timer.start(f"epoch_{epoch}")
            t_train = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            batch_num = ((epoch - 1)
                         * len(dataloader['train_loader']))

            for x, y in \
                    dataloader['train_loader'].get_iterator():
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num,
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

            current_lr = \
                engine.optimizer.param_groups[0]['lr']
            curr_time = time.strftime(
                "%d-%H-%M", time.localtime())
            # Prodigy诊断
            prodigy_d = engine.optimizer.effective_d
            log = (f'{curr_time} | Epoch {epoch:03d} | '
                   f'Train MAE={np.mean(train_loss):.4f} '
                   f'MAPE={np.mean(train_mape):.4f} '
                   f'RMSE={np.mean(train_rmse):.4f} | '
                   f'Val MAE={mvalid_loss:.4f} '
                   f'RMSE={mvalid_rmse:.4f} '
                   f'MAPE={mvalid_mape:.4f} | '
                   f'LR={current_lr:.6f} '
                   f'd={prodigy_d:.6f}')
            print(log)
            timer.stop(f"epoch_{epoch}")

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            if _is_debug() and epoch % 2 == 0:
                gradient_health_check(model, step=batch_num)

            engine.test(
                model, save_path_resume, device,
                dataloader, scaler, model_name,
                _max=_max, _min=_min,
                dataset_name=dataset_name)

        print(f"\nAvg Train: {np.mean(train_time):.4f}s/epoch")
        print(f"Avg Val  : {np.mean(val_time):.4f}s/epoch")
        print(f"Best Val : {early_stopping.val_loss_min:.4f}")
    else:
        engine.test(
            model, save_path_resume, device,
            dataloader, scaler, model_name,
            save=False, _max=_max, _min=_min,
            dataset_name=dataset_name)

    print(timer.summary())
    print(f"\n[PAR] Pipeline complete. Saved: {save_path}")
    return early_stopping.val_loss_min


def main():
    parser = argparse.ArgumentParser(
        description='Parallax (D2STGNN) Training — M054')
    parser.add_argument(
        '--dataset', type=str, default='SYNTH',
        choices=['SYNTH', 'METR-LA', 'PEMS-BAY',
                 'PEMS04', 'PEMS08'])
    parser.add_argument(
        '--device', type=str, default='cpu')
    parser.add_argument(
        '--epochs', type=int, default=None)
    parser.add_argument(
        '--debug', action='store_true')
    args = parser.parse_args()
    t_start = time.time()
    run(args.dataset, args.device, args.epochs, args.debug)
    print(f"Total time: {time.time()-t_start:.2f}s")


if __name__ == '__main__':
    main()
