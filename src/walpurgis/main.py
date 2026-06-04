import argparse
import time
import os
import torch
torch.set_num_threads(1)
import pickle
import yaml

from utils.train import set_config, EarlyStopping, data_reshaper, load_model
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger
from models.losses import masked_mae
from models import trainer
from models.model import D2STGNN
from walpurgis import (_dbg, snapshot_model, register_activation_hooks,
                       gradient_health_check)
import numpy as np

_TAG = "main"


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        help='Dataset name.')
    args = parser.parse_args()

    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = config['data_args']['data_dir'].split("/")[-1]
    device = torch.device(config['start_up']['device'])
    model_name = config['start_up']['model_name']

    save_path = f'output/{model_name}_{dataset_name}.pt'
    save_path_resume = f'output/{model_name}_{dataset_name}_resume.pt'
    load_pkl = config['start_up']['load_pkl']

    os.makedirs('output', exist_ok=True)

    _dbg(_TAG, f"start dataset={dataset_name}, device={device}")

    # ============ Load Data ============
    if load_pkl and os.path.exists(f'output/dataloader_{dataset_name}.pkl'):
        t1 = time.time()
        dataloader = pickle.load(
            open(f'output/dataloader_{dataset_name}.pkl', 'rb'))
        print(f"Load dataset: {time.time()-t1:.2f}s (from pkl)")
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size,
                                  batch_size, dataset_name)
        pickle.dump(dataloader,
                     open(f'output/dataloader_{dataset_name}.pkl', 'wb'))
        print(f"Load dataset: {time.time()-t1:.2f}s")

    scaler = dataloader['scaler']

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(
            open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(
            open(f"datasets/{dataset_name}/max.pkl", 'rb'))
    else:
        _min = None
        _max = None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(
        config['data_args']['adj_data_path'],
        config['data_args']['adj_type'])
    print(f"Load adjacent matrix: {time.time()-t1:.2f}s")

    # ============ Hyper Parameters ============
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps'] = (optim_args['cl_epochs']
                              * len(dataloader['train_loader']))
    optim_args['warm_steps'] = (optim_args['warm_epochs']
                                * len(dataloader['train_loader']))
    optim_args['steps_per_epoch'] = len(dataloader['train_loader'])

    # ============ Logger ============
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # ============ Model ============
    model = D2STGNN(**model_args)

    # 改动1: 多 GPU DataParallel — upstream 只支持单卡
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[walpurgis] Using DataParallel on {n_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 改动2: 混合精度 GradScaler — upstream 全 FP32
    use_amp = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[walpurgis] AMP enabled: {use_amp}")

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # 改动5: 训练前初始参数快照 — 确认初始化是否合理
    snapshot_model(model, epoch=0, step=0)

    # 改动6: activation tracker — 用第一个 batch 跑一遍 forward,
    # 打印所有中间层激活, 检测死神经元 / nan
    _first_batch = next(dataloader['train_loader'].get_iterator())
    _probe_x = data_reshaper(_first_batch[0], device)
    _act_tracker = register_activation_hooks(model)
    with torch.no_grad():
        model.eval()
        _ = model(_probe_x)
    _act_tracker.report()
    _dead = _act_tracker.check_dead()
    if _dead:
        print(f"[walpurgis:WARN] {len(_dead)} dead layers at init — "
              f"consider checking initialization")
    _act_tracker.remove()
    model.train()

    train_time = []
    val_time = []

    print(f"Training iterations per epoch: {len(dataloader['train_loader'])}")

    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model_inner = model.module if hasattr(model, 'module') else model
        model_inner = load_model(model_inner, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model_inner = model.module if hasattr(model, 'module') else model
        model_inner = load_model(model_inner, save_path_resume)

    batch_num = resume_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ============ Training ============
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            current_lr = engine.optimizer.param_groups[0]['lr']

            train_loss = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()

            for itera, batch in enumerate(
                    dataloader['train_loader'].get_iterator()):
                # 改动: 3-tuple unpack(含 sample_weight)
                x, y = batch[0], batch[1]
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy,
                    batch_num=batch_num, _max=_max, _min=_min)
                print(f"{itera}: {mae:.4f}", end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - time_train_start)
            current_lr = engine.optimizer.param_groups[0]['lr']

            if engine.if_lr_scheduler and engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # ============ Validation ============
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - time_val_start)

            curr_time = time.strftime("%d-%H-%M", time.localtime())
            log = (f'Time: {curr_time} | Epoch: {epoch:03d} | '
                   f'Train Loss: {mtrain_loss:.4f} | '
                   f'Train MAPE: {mtrain_mape:.4f} | '
                   f'Train RMSE: {mtrain_rmse:.4f} | '
                   f'Val Loss: {mvalid_loss:.4f} | '
                   f'Val RMSE: {mvalid_rmse:.4f} | '
                   f'Val MAPE: {mvalid_mape:.4f} | '
                   f'LR: {current_lr:.6f}')
            print(log)

            # 改动3: epoch metric dump to CSV + JSONL
            logger.log_epoch_metrics(
                epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
                mvalid_loss, mvalid_mape, mvalid_rmse, current_lr)

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            engine.test(model, save_path_resume, device, dataloader,
                        scaler, model_name,
                        _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"Average Training Time: {np.mean(train_time):.4f} s/epoch")
        print(f"Average Inference Time: {np.mean(val_time):.4f} s/epoch")

    else:
        # 改动4: test 模式 — 尝试 ensemble best + resume
        model_inner = model.module if hasattr(model, 'module') else model
        if os.path.exists(save_path) and os.path.exists(save_path_resume):
            sd_best = torch.load(save_path)
            sd_resume = torch.load(save_path_resume)
            # 逐参数平均
            sd_avg = {}
            for k in sd_best:
                if k in sd_resume:
                    sd_avg[k] = (sd_best[k] + sd_resume[k]) * 0.5
                else:
                    sd_avg[k] = sd_best[k]
            model_inner.load_state_dict(sd_avg, strict=False)
            print("[walpurgis] Ensemble: averaged best + resume checkpoints")
        engine.test(model, save_path_resume, device, dataloader,
                    scaler, model_name, save=False,
                    _max=_max, _min=_min, loss=engine.loss,
                    dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    main()
    print(f"Total time: {time.time() - t_start:.2f}s")
