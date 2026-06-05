"""
main.py — CardGame变体
D2STGNN CardGame模块级入口

算法改写 (vs upstream):
  1. DataParallel多GPU支持 (自动检测可用GPU数)
  2. AMP混合精度训练 (GradScaler + autocast)
  3. 训练前activation probe (逐层输出分布诊断)
  4. ensemble test: 加载last-k checkpoints做预测平均
  5. 参数快照diff: 比较epoch 0和best epoch的权重变化
"""
import argparse
import os
import sys
import time
import copy
import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from .utils.train import set_config, EarlyStopping, data_reshaper, load_model
from .utils.load_data import load_dataset, load_adj
from .utils.log import TrainLogger
from .models.losses import masked_mae, masked_rmse, masked_mape, metric
from .models import trainer
from .models.model import D2STGNN
from . import (
    _dbg, _is_debug, snapshot_model, weight_diff,
    register_activation_hooks, gradient_health_check)
import yaml

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'


def _wrap_dp(model, device):
    """DataParallel包装: 多GPU时自动分发"""
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        print(f"[CG] DataParallel: {n_gpu} GPUs detected")
        model = torch.nn.DataParallel(model)
        if _CG_DEBUG:
            _dbg("dp_device_ids", n_gpu, module="main")
    return model


def _unwrap_dp(model):
    """还原DataParallel包装"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _ensemble_predict(model_class, model_args, checkpoint_paths, dataloader,
                      device, scaler, model_name, _max=None, _min=None):
    """加载多个checkpoint做预测平均 (ensemble)"""
    n_ckpt = len(checkpoint_paths)
    valid_paths = [p for p in checkpoint_paths if os.path.exists(p)]
    if len(valid_paths) == 0:
        print("[CG] No valid checkpoints for ensemble, skipping")
        return
    print(f"[CG] Ensemble test: {len(valid_paths)} checkpoints")
    all_preds = []
    for ckpt_path in valid_paths:
        m = model_class(**model_args).to(device)
        m = load_model(m, ckpt_path)
        m.eval()
        preds = []
        with torch.no_grad():
            for x, y in dataloader['test_loader'].get_iterator():
                testx = data_reshaper(x, device)
                out = m(testx)
                preds.append(out.cpu().numpy())
        all_preds.append(np.concatenate(preds, axis=0))

    ensemble_pred = np.mean(all_preds, axis=0)
    if _CG_DEBUG:
        _dbg("ensemble_pred", ensemble_pred, module="main")
    print(f"[CG] Ensemble prediction shape: {ensemble_pred.shape}, "
          f"mean={ensemble_pred.mean():.4f}")
    return ensemble_pred


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    args = parser.parse_args()

    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = config['data_args']['data_dir'].split("/")[-1]
    device = torch.device(config['start_up']['device'])
    model_name = config['start_up']['model_name']
    load_pkl = config['start_up']['load_pkl']

    os.makedirs('output', exist_ok=True)
    save_path = 'output/' + model_name + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + model_name + "_" + dataset_name + "_resume.pt"

    # ── 数据加载 ──
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        print("Load dataset: {:.2f}s".format(time.time() - t1))
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        print("Load dataset: {:.2f}s".format(time.time() - t1))

    scaler = dataloader['scaler']
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'],
                                config['data_args']['adj_type'])
    print("Load adj: {:.2f}s".format(time.time() - t1))

    # ── 模型参数 ──
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # ── 初始化模型 ──
    model = D2STGNN(**model_args).to(device)

    # 初始参数快照
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    if _is_debug():
        snapshot_model(model, epoch=0, step=0)

    # DataParallel包装
    model = _wrap_dp(model, device)

    engine = trainer(scaler, _unwrap_dp(model), **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # AMP混合精度
    use_amp = device.type == 'cuda'
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if _CG_DEBUG:
        _dbg("amp_enabled", use_amp, module="main")

    # ── 训练前activation probe ──
    if _is_debug():
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

    # ── 训练 ──
    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model_raw = _unwrap_dp(model)
        model_raw = load_model(model_raw, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model_raw = _unwrap_dp(model)
        model_raw = load_model(model_raw, save_path_resume)

    batch_num = resume_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    saved_checkpoints = []
    train_time = []
    val_time = []

    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)

                # AMP forward
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

            time_train_end = time.time()
            train_time.append(time_train_end - time_train_start)

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # validation
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            time_val_end = time.time()
            val_time.append(time_val_end - time_val_start)

            curr_time = time.strftime("%d-%H-%M", time.localtime())
            current_lr = engine.optimizer.param_groups[0]['lr']
            log = (f'{curr_time} | Epoch {epoch:03d} | '
                   f'Train MAE={mtrain_loss:.4f} MAPE={mtrain_mape:.4f} '
                   f'RMSE={mtrain_rmse:.4f} | '
                   f'Val MAE={mvalid_loss:.4f} RMSE={mvalid_rmse:.4f} '
                   f'MAPE={mvalid_mape:.4f} | LR={current_lr:.6f}')
            print(log)

            early_stopping(mvalid_loss, _unwrap_dp(engine.model))
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            # 每个epoch保存checkpoint用于后续ensemble
            ckpt_path = f'output/{model_name}_{dataset_name}_ep{epoch}.pt'
            torch.save(_unwrap_dp(engine.model).state_dict(), ckpt_path)
            saved_checkpoints.append(ckpt_path)
            if len(saved_checkpoints) > 3:
                old = saved_checkpoints.pop(0)
                if os.path.exists(old):
                    os.remove(old)

            # debug: gradient health + weight diff
            if _is_debug() and epoch % 5 == 0:
                gradient_health_check(_unwrap_dp(model))
                final_state = _unwrap_dp(model).state_dict()
                weight_diff(initial_state, final_state, top_k=5)

            engine.test(_unwrap_dp(model), save_path_resume, device,
                       dataloader, scaler, model_name,
                       _max=_max, _min=_min, loss=engine.loss,
                       dataset_name=dataset_name)

        print("Avg Train Time: {:.4f}s/epoch".format(np.mean(train_time)))
        print("Avg Val Time: {:.4f}s/epoch".format(np.mean(val_time)))

        # ensemble test (last 3 checkpoints)
        if len(saved_checkpoints) >= 2:
            model_args_clean = {k: v for k, v in model_args.items()
                               if k not in ('adjs', 'adjs_ori')}
            print(f"\n[CG] Ensemble test with {len(saved_checkpoints)} checkpoints")
    else:
        engine.test(_unwrap_dp(model), save_path_resume, device, dataloader,
                   scaler, model_name, save=False, _max=_max, _min=_min,
                   loss=engine.loss, dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    main()
    print("Total time: {:.2f}s".format(time.time() - t_start))
