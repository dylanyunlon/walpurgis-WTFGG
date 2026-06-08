"""
main.py — Aurora变体模块入口
"""
import argparse
import time
import os
import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from .utils.train import (set_config, EarlyStopping,
                           data_reshaper, load_model)
from .utils.load_data import load_dataset, load_adj
from .utils.log import TrainLogger
from .models.losses import (masked_mae, masked_rmse,
                             masked_mape, metric)
from .models import trainer
from .models.model import D2STGNN
from . import (_dbg, _is_debug, snapshot_model,
               register_activation_hooks,
               gradient_health_check)
import yaml


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='METR-LA')
    args = parser.parse_args()
    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_dir = config['data_args']['data_dir']
    dataset_name = data_dir.split("/")[-1]
    device = torch.device(config['start_up']['device'])
    model_name = config['start_up']['model_name']
    save_path = (f'output/{model_name}_{dataset_name}.pt')
    save_path_resume = (
        f'output/{model_name}_{dataset_name}_resume.pt')
    batch_size = config['model_args']['batch_size']
    dataloader = load_dataset(
        data_dir, batch_size, batch_size,
        batch_size, dataset_name)
    scaler = dataloader['scaler']
    _min = _max = None
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(
            open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(
            open(f"datasets/{dataset_name}/max.pkl", 'rb'))
    adj_mx, adj_ori = load_adj(
        config['data_args']['adj_data_path'],
        config['data_args']['adj_type'])
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [
        torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name
    optim_args = config['optim_args']
    optim_args['cl_steps'] = (
        optim_args['cl_epochs'] *
        len(dataloader['train_loader']))
    optim_args['warm_steps'] = (
        optim_args['warm_epochs'] *
        len(dataloader['train_loader']))
    model = D2STGNN(**model_args).to(device)
    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(
        optim_args['patience'], save_path)
    mode = config['start_up']['mode']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model = load_model(model, save_path_resume)
    batch_num = (
        resume_epoch * len(dataloader['train_loader']))
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)
    if mode != 'test':
        for epoch in range(
                resume_epoch + 1, optim_args['epochs']):
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
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
            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            mvalid_loss, mvalid_mape, mvalid_rmse = \
                engine.eval(
                    device, dataloader, model_name,
                    _max=_max, _min=_min)
            print(f"Epoch {epoch}: "
                  f"train_mae={np.mean(train_loss):.4f} "
                  f"val_mae={mvalid_loss:.4f}")
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break


if __name__ == '__main__':
    t_start = time.time()
    main()
    print(f"Total: {time.time()-t_start:.2f}s")
