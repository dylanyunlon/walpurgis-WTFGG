#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walpurgis-WTFGG v3 main entry — ported from D2STGNN with full debug flags.

Usage:
  python main.py --dataset METR-LA
  python main.py --dataset PEMS04 --debug-model --debug-trainer --debug-loss

Available debug flags (combine freely):
  --debug-main     --debug-model    --debug-trainer  --debug-data
  --debug-adj      --debug-train    --debug-loss     --debug-gate
  --debug-stconv   --debug-difblk   --debug-diffc    --debug-inhblk
  --debug-inhmod   --debug-inhfc    --debug-dygraph  --debug-dist
  --debug-mask     --debug-norm     --debug-loader   --debug-log
  --debug-resdecomp
"""

import argparse
import time
import sys
import os
import torch
torch.set_num_threads(1)
import pickle
import numpy as np

from utils.train import set_config, EarlyStopping, data_reshaper, load_model
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger
from models.losses import masked_mae, masked_rmse, masked_mape
from models import trainer
from models.model import D2STGNN
import yaml

try:
    import setproctitle
except ImportError:
    setproctitle = None

_DBG = ("--debug-main" in sys.argv)


def _ts():
    return time.strftime("%d-%H:%M:%S", time.localtime())


def main():
    set_config(0)

    parser = argparse.ArgumentParser(
        description="Walpurgis-WTFGG v3 training / evaluation")
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        help='METR-LA | PEMS-BAY | PEMS04 | PEMS08')
    args, _ = parser.parse_known_args()

    cfg_path = f"configs/{args.dataset}.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)

    data_dir  = cfg['data_args']['data_dir']
    ds_name   = data_dir.split("/")[-1]
    device    = torch.device(cfg['start_up']['device'])
    tag       = cfg['start_up']['model_name']
    ckpt_best = f'output/{tag}_{ds_name}.pt'
    ckpt_last = f'output/{tag}_{ds_name}_resume.pt'
    use_pkl   = cfg['start_up']['load_pkl']

    if setproctitle:
        setproctitle.setproctitle(f"{tag}.{ds_name}@walpurgis_v3")

    os.makedirs('output', exist_ok=True)

    if _DBG:
        print(f"[DBG:main] ds={ds_name}  device={device}  "
              f"tag={tag}  pkl={use_pkl}")

    # ════════ data ════════
    t0 = time.time()
    if use_pkl:
        dataloader = pickle.load(open(f'output/dataloader_{ds_name}.pkl', 'rb'))
    else:
        bs = cfg['model_args']['batch_size']
        dataloader = load_dataset(data_dir, bs, bs, bs, ds_name)
        pickle.dump(dataloader,
                     open(f'output/dataloader_{ds_name}.pkl', 'wb'))
    print(f"[TIMING] data loaded in {time.time()-t0:.2f}s")

    scaler = dataloader['scaler']

    _min, _max = None, None
    if ds_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{ds_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{ds_name}/max.pkl", 'rb'))

    t0 = time.time()
    adj_proc, adj_raw = load_adj(
        cfg['data_args']['adj_data_path'], cfg['data_args']['adj_type'])
    print(f"[TIMING] adj loaded in {time.time()-t0:.2f}s")

    # ════════ hyper-params ════════
    m = cfg['model_args']
    m['device']    = device
    m['num_nodes'] = adj_proc[0].shape[0]
    m['adjs']      = [torch.tensor(a).to(device) for a in adj_proc]
    m['adjs_ori']  = torch.tensor(adj_raw).to(device)
    m['dataset']   = ds_name

    o = cfg['optim_args']
    o['cl_steps']   = o['cl_epochs'] * len(dataloader['train_loader'])
    o['warm_steps'] = o['warm_epochs'] * len(dataloader['train_loader'])

    if _DBG:
        print(f"[DBG:main] nodes={m['num_nodes']}  "
              f"train_iters={len(dataloader['train_loader'])}  "
              f"cl_steps={o['cl_steps']}  warm={o['warm_steps']}")

    # ════════ model ════════
    logger = TrainLogger(tag, ds_name)
    logger.print_model_args(m, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(o)

    model  = D2STGNN(**m).to(device)
    engine = trainer(scaler, model, **o)
    stopper = EarlyStopping(o['patience'], ckpt_best)

    mode = cfg['start_up']['mode']
    assert mode in ('test', 'resume', 'scratch')
    start_ep = 0
    if mode == 'test':
        model = load_model(model, ckpt_best)
    elif mode == 'resume':
        start_ep = cfg['start_up']['resume_epoch']
        model = load_model(model, ckpt_last)

    batch_ctr = start_ep * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(start_ep, batch_ctr)

    t_train_hist, t_val_hist = [], []
    n_batches = len(dataloader['train_loader'])
    print(f"Iterations per epoch: {n_batches}")

    # ════════ training loop ════════
    if mode != 'test':
        for epoch in range(start_ep + 1, o['epochs']):
            tt = time.time()
            ep_mae, ep_mape, ep_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for it, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                tx = data_reshaper(x, device)
                ty = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    tx, ty, batch_num=batch_ctr, _max=_max, _min=_min)
                print(f"  batch {it}: mae={mae:.4f}", end='\r')
                ep_mae.append(mae)
                ep_mape.append(mape)
                ep_rmse.append(rmse)
                batch_ctr += 1

            dt_tr = time.time() - tt
            t_train_hist.append(dt_tr)

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            lr_now = engine.optimizer.param_groups[0]['lr']

            tv = time.time()
            vl, vm, vr = engine.eval(
                device, dataloader, tag, _max=_max, _min=_min)
            dt_val = time.time() - tv
            t_val_hist.append(dt_val)

            print(f"[{_ts()}] Ep {epoch:03d} | "
                  f"Train MAE={np.mean(ep_mae):.4f} "
                  f"MAPE={np.mean(ep_mape):.4f} "
                  f"RMSE={np.mean(ep_rmse):.4f} | "
                  f"Val MAE={vl:.4f} RMSE={vr:.4f} MAPE={vm:.4f} | "
                  f"LR={lr_now:.6f} | "
                  f"t_tr={dt_tr:.1f}s t_val={dt_val:.1f}s")

            stopper(vl, engine.model)
            if stopper.early_stop:
                print("Early stopping triggered.")
                break

            engine.test(model, ckpt_last, device, dataloader, scaler,
                        tag, _max=_max, _min=_min,
                        loss=engine.loss, dataset_name=ds_name)

        print(f"\nAvg train: {np.mean(t_train_hist):.4f}s/ep  "
              f"Avg val: {np.mean(t_val_hist):.4f}s/ep")
    else:
        engine.test(model, ckpt_last, device, dataloader, scaler,
                    tag, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=ds_name)


if __name__ == '__main__':
    t_global = time.time()
    main()
    print(f"\nTotal wall-clock: {time.time() - t_global:.2f}s")
