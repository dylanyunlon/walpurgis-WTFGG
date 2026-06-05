"""
train_cardgame.py — CardGame训练入口
从项目根目录运行: python train_cardgame.py --dataset SYNTH

算法特性 (vs upstream D2STGNN):
  - RAdam + OneCycleLR调度
  - curriculum learning (cosine temperature退火)
  - Welsch robust loss + temporal smoothness penalty
  - EarlyStopping with plateau slope detection
  - DataParallel多GPU + AMP混合精度
  - activation probe + gradient health check (CARDGAME_DEBUG=1)
  - ensemble test: last-k checkpoint平均

支持数据集: SYNTH, METR-LA, PEMS-BAY, PEMS04, PEMS08
"""
import argparse
import os
import sys
import time
import pickle

import numpy as np
import torch
import yaml

# 确保从项目根运行
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from walpurgis_cardgame.utils.train import set_config, EarlyStopping, data_reshaper, load_model
from walpurgis_cardgame.utils.load_data import load_dataset, load_adj
from walpurgis_cardgame.utils.log import TrainLogger
from walpurgis_cardgame.models.losses import masked_mae, masked_rmse, masked_mape, metric
from walpurgis_cardgame.models import trainer
from walpurgis_cardgame.models.model import D2STGNN
from walpurgis_cardgame import (
    _dbg, _is_debug, snapshot_model, weight_diff,
    register_activation_hooks, gradient_health_check)


def _resolve_path(rel_path, base=None):
    """将相对路径解析为相对于repo根的绝对路径"""
    if os.path.isabs(rel_path):
        return rel_path
    base = base or _REPO_ROOT
    return os.path.join(base, rel_path)


def _wrap_dp(model, device):
    """DataParallel多GPU包装"""
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        print(f"[CG] DataParallel: {n_gpu} GPUs")
        model = torch.nn.DataParallel(model)
    return model


def _unwrap_dp(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def run(dataset: str, device_str: str = 'cpu', epochs_override: int = None,
        debug: bool = False):
    """
    训练入口函数

    Args:
        dataset: 数据集名称 (SYNTH / METR-LA / PEMS-BAY / PEMS04 / PEMS08)
        device_str: 设备 ('cpu' / 'cuda' / 'cuda:0' 等)
        epochs_override: 覆盖配置文件中的epoch数 (None=使用配置值)
        debug: 是否开启CARDGAME_DEBUG调试输出
    """
    if debug:
        os.environ['CARDGAME_DEBUG'] = '1'

    set_config(0)

    # ── 加载配置 ─────────────────────────────────────────
    config_dir = os.path.join(_REPO_ROOT, 'src', 'walpurgis_cardgame', 'configs')
    config_path = os.path.join(config_dir, dataset + '.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}\n"
                                f"Available: {os.listdir(config_dir)}")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ── 路径解析 ─────────────────────────────────────────
    data_dir_cfg = config['data_args']['data_dir']
    adj_path_cfg = config['data_args']['adj_data_path']

    data_dir = _resolve_path(data_dir_cfg)
    adj_path = _resolve_path(adj_path_cfg)

    if not os.path.exists(data_dir):
        cg_dir = os.path.join(_REPO_ROOT, 'src', 'walpurgis_cardgame')
        data_dir = _resolve_path(data_dir_cfg, cg_dir)
        adj_path = _resolve_path(adj_path_cfg, cg_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset dir not found: {data_dir}\n"
            f"Run: python -m walpurgis_cardgame.generate_synth_data  (for SYNTH)\n"
            f"Or provide METR-LA/PEMS data at {data_dir}")

    device = torch.device(device_str)
    dataset_name = os.path.basename(data_dir)
    model_name = config['start_up']['model_name']
    mode = config['start_up']['mode']
    assert mode in ('scratch', 'resume', 'test')

    os.makedirs(os.path.join(_REPO_ROOT, 'output'), exist_ok=True)
    save_path = os.path.join(_REPO_ROOT, 'output', f'{model_name}_{dataset_name}.pt')
    save_path_resume = os.path.join(_REPO_ROOT, 'output', f'{model_name}_{dataset_name}_resume.pt')

    print(f"\n{'='*60}")
    print(f"  CardGame (D2STGNN) Training Pipeline")
    print(f"  Dataset : {dataset_name}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {mode}")
    print(f"  Debug   : {_is_debug()}")
    print(f"{'='*60}\n")

    # ── 数据加载 ─────────────────────────────────────────
    t0 = time.time()
    batch_size = config['model_args']['batch_size']
    load_pkl = config['start_up']['load_pkl']
    pkl_path = os.path.join(_REPO_ROOT, 'output', f'dataloader_{dataset_name}.pkl')

    if load_pkl and os.path.exists(pkl_path):
        dataloader = pickle.load(open(pkl_path, 'rb'))
        print(f"Loaded dataloader from cache: {time.time()-t0:.2f}s")
    else:
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open(pkl_path, 'wb'))
        print(f"Loaded dataset: {time.time()-t0:.2f}s")

    scaler = dataloader['scaler']
    _min, _max = None, None
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(os.path.join(data_dir, 'min.pkl'), 'rb'))
        _max = pickle.load(open(os.path.join(data_dir, 'max.pkl'), 'rb'))

    # ── 邻接矩阵 ─────────────────────────────────────────
    t0 = time.time()
    adj_mx, adj_ori = load_adj(adj_path, config['data_args']['adj_type'])
    print(f"Loaded adj ({config['data_args']['adj_type']}): {time.time()-t0:.2f}s")

    # ── 模型初始化 ────────────────────────────────────────
    model_args = config['model_args'].copy()
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args'].copy()
    if epochs_override is not None:
        optim_args['epochs'] = epochs_override
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count:,}")

    # 初始参数快照 (debug模式)
    initial_state = None
    if _is_debug():
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        snapshot_model(model, epoch=0, step=0)

    # DataParallel
    model = _wrap_dp(model, device)

    engine = trainer(scaler, _unwrap_dp(model), **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # AMP
    use_amp = device.type == 'cuda'
    if _is_debug():
        _dbg("amp_enabled", use_amp, module="train_cardgame")

    resume_epoch = 0
    if mode == 'test':
        base_model = _unwrap_dp(model)
        base_model = load_model(base_model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        base_model = _unwrap_dp(model)
        if os.path.exists(save_path_resume):
            base_model = load_model(base_model, save_path_resume)
        batch_num_init = resume_epoch * len(dataloader['train_loader'])
        engine.set_resume_lr_and_cl(resume_epoch, batch_num_init)

    # ── 训练前activation probe ────────────────────────────
    if _is_debug() and mode != 'test':
        print("\n[CG] === First-batch activation probe ===")
        base_model = _unwrap_dp(model)
        tracker = register_activation_hooks(base_model)
        base_model.train()
        for x, y in dataloader['train_loader'].get_iterator():
            probe_x = data_reshaper(x, device)
            with torch.no_grad():
                _ = base_model(probe_x)
            break
        tracker.report()
        tracker.remove()
        print("[CG] === Probe complete ===\n")

    # ── 训练循环 ──────────────────────────────────────────
    train_time, val_time = [], []

    if mode != 'test':
        total_epochs = optim_args['epochs']
        batch_num = resume_epoch * len(dataloader['train_loader'])

        for epoch in range(resume_epoch + 1, total_epochs + 1):
            t_train = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for x, y in dataloader['train_loader'].get_iterator():
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        mae, mape, rmse = engine.train(
                            trainx, trainy, batch_num=batch_num,
                            _max=_max, _min=_min)
                else:
                    mae, mape, rmse = engine.train(
                        trainx, trainy, batch_num=batch_num,
                        _max=_max, _min=_min)

                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - t_train)
            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            t_val = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - t_val)

            current_lr = engine.optimizer.param_groups[0]['lr']
            import time as _time
            curr_time = _time.strftime("%d-%H-%M", _time.localtime())
            log = (f'{curr_time} | Epoch {epoch:03d} | '
                   f'Train MAE={np.mean(train_loss):.4f} '
                   f'MAPE={np.mean(train_mape):.4f} '
                   f'RMSE={np.mean(train_rmse):.4f} | '
                   f'Val MAE={mvalid_loss:.4f} '
                   f'RMSE={mvalid_rmse:.4f} '
                   f'MAPE={mvalid_mape:.4f} | '
                   f'LR={current_lr:.6f}')
            print(log)

            early_stopping(mvalid_loss, _unwrap_dp(engine.model))
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            # debug: gradient health + weight diff
            if _is_debug() and epoch % 5 == 0:
                gradient_health_check(_unwrap_dp(model))
                if initial_state is not None:
                    final_state = _unwrap_dp(model).state_dict()
                    weight_diff(initial_state, final_state, top_k=5)

            engine.test(_unwrap_dp(model), save_path_resume, device,
                       dataloader, scaler, model_name,
                       _max=_max, _min=_min, loss=engine.loss,
                       dataset_name=dataset_name)

        print(f"\nAvg Train Time: {np.mean(train_time):.4f}s/epoch")
        print(f"Avg Val   Time: {np.mean(val_time):.4f}s/epoch")
        print(f"Best Val  Loss: {early_stopping.val_loss_min:.4f}")

    else:
        engine.test(_unwrap_dp(model), save_path_resume, device,
                   dataloader, scaler, model_name, save=False,
                   _max=_max, _min=_min, loss=engine.loss,
                   dataset_name=dataset_name)

    print(f"\n[CG] Pipeline complete. Saved: {save_path}")
    return early_stopping.val_loss_min


def main():
    parser = argparse.ArgumentParser(
        description='CardGame (D2STGNN) Training Pipeline')
    parser.add_argument('--dataset', type=str, default='SYNTH',
                        choices=['SYNTH', 'METR-LA', 'PEMS-BAY', 'PEMS04', 'PEMS08'],
                        help='Dataset to train on')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu / cuda / cuda:0 etc.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count from config')
    parser.add_argument('--debug', action='store_true',
                        help='Enable CARDGAME_DEBUG output')
    args = parser.parse_args()
    t_start = time.time()
    run(args.dataset, args.device, args.epochs, args.debug)
    print(f"Total time: {time.time()-t_start:.2f}s")


if __name__ == '__main__':
    main()
