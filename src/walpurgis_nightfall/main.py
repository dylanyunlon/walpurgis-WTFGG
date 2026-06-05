"""
main.py — Nightfall变体
算法改写:
  1. scheduled sampling线性衰减 (teacher forcing ratio从1.0线性降到0.1)
  2. 训练前第一个batch做activation probe
  3. 初始参数快照 (存储初始state_dict)
"""
import argparse
import time
import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from utils.train import set_config, EarlyStopping, data_reshaper, load_model
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger
from models.losses import masked_mae, masked_rmse, masked_mape, metric
from models import trainer
from models.model import D2STGNN
from walpurgis_nightfall import (
    _dbg, _is_debug, snapshot_model, register_activation_hooks, gradient_health_check)
import yaml


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
    save_path = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
    load_pkl = config['start_up']['load_pkl']
    model_name = config['start_up']['model_name']
    # load dataset
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        print("Load dataset: {:.2f}s".format(time.time() - t1))
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        os.makedirs('output', exist_ok=True)
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
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    print("Load adj: {:.2f}s".format(time.time() - t1))
    # model args
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name
    # optim args
    optim_args = config['optim_args']
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])
    # log
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)
    # init model
    model = D2STGNN(**model_args).to(device)
    # 初始参数快照
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    if _is_debug():
        snapshot_model(model, epoch=0, step=0)
    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)
    # 训练前activation probe (第一个batch)
    if _is_debug():
        print("\n[NF] === First-batch activation probe ===")
        tracker = register_activation_hooks(model)
        model.train()
        for x, y in dataloader['train_loader'].get_iterator():
            probe_x = data_reshaper(x, device)
            with torch.no_grad():
                _ = model(probe_x)
            break
        tracker.report()
        tracker.remove()
        print("[NF] === Probe complete ===\n")
    train_time = []
    val_time = []
    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model = load_model(model, save_path_resume)
    batch_num = resume_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)
    # training
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
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
                   f'Train MAE={mtrain_loss:.4f} MAPE={mtrain_mape:.4f} RMSE={mtrain_rmse:.4f} | '
                   f'Val MAE={mvalid_loss:.4f} RMSE={mvalid_rmse:.4f} MAPE={mvalid_mape:.4f} | '
                   f'LR={current_lr:.6f}')
            print(log)
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            engine.test(model, save_path_resume, device, dataloader, scaler,
                       model_name, _max=_max, _min=_min, loss=engine.loss,
                       dataset_name=dataset_name)
        print("Avg Train Time: {:.4f}s/epoch".format(np.mean(train_time)))
        print("Avg Val Time: {:.4f}s/epoch".format(np.mean(val_time)))
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler, model_name,
                   save=False, _max=_max, _min=_min, loss=engine.loss,
                   dataset_name=dataset_name)


import os

if __name__ == '__main__':
    t_start = time.time()
    main()
    print("Total time: {:.2f}s".format(time.time() - t_start))
