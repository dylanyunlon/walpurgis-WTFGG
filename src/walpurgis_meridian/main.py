"""Meridian main.py — D2STGNN Meridian variant entry point."""
import argparse
import time
import torch
torch.set_num_threads(1)
import pickle
import numpy as np
import yaml
import os, sys

# ensure parent is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walpurgis_meridian.utils.train import set_config, EarlyStopping, data_reshaper, load_model, save_model
from walpurgis_meridian.utils.load_data import load_dataset, load_adj
from walpurgis_meridian.utils.log import TrainLogger
from walpurgis_meridian.models.losses import masked_mae, metric
from walpurgis_meridian.models.trainer import trainer
from walpurgis_meridian.models.model import D2STGNN
from walpurgis_meridian import _dbg, snapshot_model, _is_debug


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SYNTH', help='Dataset name.')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__),
                               "configs", args.dataset + ".yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = data_dir.split("/")[-1]
    device = torch.device(config['start_up']['device'])
    os.makedirs('output', exist_ok=True)
    save_path = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
    load_pkl = config['start_up']['load_pkl']
    model_name = config['start_up']['model_name']

    # load data
    if load_pkl and os.path.exists('output/dataloader_' + dataset_name + '.pkl'):
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        print(f"Load dataset: {time.time()-t1:.2f}s...")
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        os.makedirs('output', exist_ok=True)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        print(f"Load dataset: {time.time()-t1:.2f}s...")
    scaler = dataloader['scaler']

    _min, _max = None, None
    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))

    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    print(f"Load adj: {time.time()-t1:.2f}s...")

    # model args
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    # init model
    model = D2STGNN(**model_args).to(device)
    if _is_debug():
        snapshot_model(model, epoch=0, step=0)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    train_time = []
    val_time = []

    print(f"Total training iterations: {len(dataloader['train_loader'])}")

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

    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num,
                    _max=_max, _min=_min, epoch=epoch)
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            train_time.append(time.time() - time_train_start)

            current_lr = engine.optimizer.param_groups[0]['lr']

            # ReduceLROnPlateau needs val loss
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - time_val_start)

            if engine.if_lr_scheduler and engine.lr_scheduler is not None:
                engine.lr_scheduler.step(mvalid_loss)

            mtrain_loss = np.mean(train_loss)
            curr_time = time.strftime("%d-%H-%M", time.localtime())
            log = (f'{curr_time} | Epoch {epoch:03d} | '
                   f'Train MAE: {mtrain_loss:.4f} | Val MAE: {mvalid_loss:.4f} | '
                   f'Val RMSE: {mvalid_rmse:.4f} | LR: {current_lr:.6f}')
            print(log)
            _dbg("epoch_summary", f"epoch={epoch} train={mtrain_loss:.4f} val={mvalid_loss:.4f}")

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            engine.test(model, save_path_resume, device, dataloader, scaler,
                        model_name, _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"Avg Train Time: {np.mean(train_time):.4f} s/epoch")
        print(f"Avg Val Time: {np.mean(val_time):.4f} s/epoch")
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler,
                    model_name, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    main()
    print(f"Total time: {time.time() - t_start:.2f}s")
