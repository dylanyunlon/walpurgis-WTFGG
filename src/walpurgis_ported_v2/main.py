#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walpurgis-WTFGG main entry point.
Ported from D2STGNN with instrumented debug output.

Debug flags (append to command line):
  --debug-model   --debug-trainer  --debug-data
  --debug-adj     --debug-train    --debug-loss
  --debug-gate    --debug-stconv   --debug-inhblk
  ... (each module has its own flag)
"""

import argparse
import time
import sys
import torch
torch.set_num_threads(1)
import pickle

from utils.train import set_config, EarlyStopping, data_reshaper, load_model
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger
from models.losses import masked_mae, masked_rmse, masked_mape
from models import trainer
from models.model import D2STGNN
import yaml
import numpy as np

try:
    import setproctitle
except ImportError:
    setproctitle = None

_DBG_MAIN = ("--debug-main" in sys.argv) or False


def _timestamp():
    return time.strftime("%d-%H:%M:%S", time.localtime())


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser(description="Walpurgis-WTFGG training / evaluation")
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        help='Dataset: METR-LA | PEMS-BAY | PEMS04 | PEMS08')
    args, _ = parser.parse_known_args()

    cfg_path = f"configs/{args.dataset}.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)

    data_dir     = cfg['data_args']['data_dir']
    ds_name      = data_dir.split("/")[-1]
    device       = torch.device(cfg['start_up']['device'])
    model_tag    = cfg['start_up']['model_name']
    save_best    = f'output/{model_tag}_{ds_name}.pt'
    save_resume  = f'output/{model_tag}_{ds_name}_resume.pt'
    load_pkl     = cfg['start_up']['load_pkl']

    if setproctitle:
        setproctitle.setproctitle(f"{model_tag}.{ds_name}@walpurgis")

    if _DBG_MAIN:
        print(f"[DBG:main] dataset={ds_name}  device={device}  "
              f"model={model_tag}  load_pkl={load_pkl}")

    # ═══════════ Load data ═══════════

    t0 = time.time()
    if load_pkl:
        dataloader = pickle.load(open(f'output/dataloader_{ds_name}.pkl', 'rb'))
    else:
        bs = cfg['model_args']['batch_size']
        dataloader = load_dataset(data_dir, bs, bs, bs, ds_name)
        pickle.dump(dataloader, open(f'output/dataloader_{ds_name}.pkl', 'wb'))
    print(f"[TIMING] Dataset loaded in {time.time()-t0:.2f}s")

    scaler = dataloader['scaler']

    # flow-dataset min/max
    _min, _max = None, None
    if ds_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{ds_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{ds_name}/max.pkl", 'rb'))

    # adjacent matrix
    t0 = time.time()
    adj_processed, adj_raw = load_adj(
        cfg['data_args']['adj_data_path'], cfg['data_args']['adj_type']
    )
    print(f"[TIMING] Adjacency loaded in {time.time()-t0:.2f}s")

    # ═══════════ Hyperparameters ═══════════

    m_args = cfg['model_args']
    m_args['device']    = device
    m_args['num_nodes'] = adj_processed[0].shape[0]
    m_args['adjs']      = [torch.tensor(a).to(device) for a in adj_processed]
    m_args['adjs_ori']  = torch.tensor(adj_raw).to(device)
    m_args['dataset']   = ds_name

    o_args = cfg['optim_args']
    o_args['cl_steps']   = o_args['cl_epochs'] * len(dataloader['train_loader'])
    o_args['warm_steps'] = o_args['warm_epochs'] * len(dataloader['train_loader'])

    if _DBG_MAIN:
        print(f"[DBG:main] num_nodes={m_args['num_nodes']}  "
              f"train_batches={len(dataloader['train_loader'])}  "
              f"cl_steps={o_args['cl_steps']}  warm_steps={o_args['warm_steps']}")

    # ═══════════ Model + Trainer ═══════════

    logger = TrainLogger(model_tag, ds_name)
    logger.print_model_args(m_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(o_args)

    model = D2STGNN(**m_args).to(device)
    engine = trainer(scaler, model, **o_args)
    stopper = EarlyStopping(o_args['patience'], save_best)

    # ═══════════ Resume / mode ═══════════

    mode = cfg['start_up']['mode']
    assert mode in ('test', 'resume', 'scratch')
    start_epoch = 0
    if mode == 'test':
        model = load_model(model, save_best)
    elif mode == 'resume':
        start_epoch = cfg['start_up']['resume_epoch']
        model = load_model(model, save_resume)

    batch_counter = start_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(start_epoch, batch_counter)

    train_times = []
    val_times   = []
    n_train_batches = len(dataloader['train_loader'])
    print(f"Training iterations per epoch: {n_train_batches}")

    # ═══════════ Training loop ═══════════

    if mode != 'test':
        for epoch in range(start_epoch + 1, o_args['epochs']):
            # ──── train one epoch ────
            t_train = time.time()
            lr_now = engine.optimizer.param_groups[0]['lr']

            epoch_mae, epoch_mape, epoch_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for it, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                tx = data_reshaper(x, device)
                ty = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    tx, ty, batch_num=batch_counter, _max=_max, _min=_min
                )
                print(f"  batch {it}: mae={mae:.4f}", end='\r')
                epoch_mae.append(mae)
                epoch_mape.append(mape)
                epoch_rmse.append(rmse)
                batch_counter += 1

            dt_train = time.time() - t_train
            train_times.append(dt_train)

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            lr_now = engine.optimizer.param_groups[0]['lr']

            # ──── validate ────
            t_val = time.time()
            v_loss, v_mape, v_rmse = engine.eval(
                device, dataloader, model_tag, _max=_max, _min=_min
            )
            dt_val = time.time() - t_val
            val_times.append(dt_val)

            log_line = (
                f"[{_timestamp()}] Epoch {epoch:03d} | "
                f"Train MAE={np.mean(epoch_mae):.4f} MAPE={np.mean(epoch_mape):.4f} "
                f"RMSE={np.mean(epoch_rmse):.4f} | "
                f"Val MAE={v_loss:.4f} RMSE={v_rmse:.4f} MAPE={v_mape:.4f} | "
                f"LR={lr_now:.6f} | t_train={dt_train:.1f}s t_val={dt_val:.1f}s"
            )
            print(log_line)

            stopper(v_loss, engine.model)
            if stopper.early_stop:
                print('Early stopping triggered.')
                break

            # ──── test at end of each epoch ────
            engine.test(
                model, save_resume, device, dataloader, scaler,
                model_tag, _max=_max, _min=_min,
                loss=engine.loss, dataset_name=ds_name,
            )

        print(f"\nAvg train time: {np.mean(train_times):.4f}s/epoch")
        print(f"Avg val time:   {np.mean(val_times):.4f}s/epoch")

    else:
        engine.test(
            model, save_resume, device, dataloader, scaler,
            model_tag, save=False, _max=_max, _min=_min,
            loss=engine.loss, dataset_name=ds_name,
        )


if __name__ == '__main__':
    t_global = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - t_global:.2f}s")
