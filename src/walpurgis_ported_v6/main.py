"""Entry point — with configurable seed, gradient diagnostics, and
output distribution health check.

Algorithm changes vs upstream
------------------------------
1. **Configurable seed** — ``set_config`` takes seed from config file
   instead of hardcoding 0.  Prints a hash of the initial model state
   so you can verify reproducibility across machines.

2. **3-tuple dataloader** — unpacks ``(x, y, meta)`` everywhere.

3. **Per-epoch gradient diagnostics** — hooks from ``utils.log`` are
   registered at startup; ``print_grad_summary`` is called once per
   epoch after the last backward pass.

4. **Output distribution health check** — after each validation epoch,
   runs a quick statistical check on predictions: if >5% of outputs are
   near-zero or if variance collapses below a threshold, prints a
   warning.  This catches mode collapse early.
"""

import argparse
import time
import hashlib
import torch
import pickle
import numpy as np

torch.set_num_threads(1)

from utils.train import set_config, EarlyStopping, load_model, data_reshaper
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger, register_grad_hooks, get_grad_norms, print_grad_summary
from models.losses import masked_mae
from models import trainer
from models.model import D2STGNN
import yaml


def _model_hash(model):
    """MD5 of the initial parameter blob — reproducibility fingerprint."""
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def _output_health_check(model, device, dataloader, tag="val"):
    """Gradient-free sanity check on prediction distribution."""
    model.eval()
    preds = []
    loader = dataloader[f'{tag}_loader']
    n_check = min(5, len(loader))
    for i, batch in enumerate(loader.get_iterator()):
        if i >= n_check:
            break
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
        with torch.no_grad():
            out = model(data_reshaper(x, device))
        preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    p_std = preds.std()
    p_near_zero = (np.abs(preds) < 1e-3).mean()
    if p_std < 0.01:
        print(f"  ⚠ OUTPUT HEALTH: variance collapse (std={p_std:.6f})")
    if p_near_zero > 0.05:
        print(f"  ⚠ OUTPUT HEALTH: {p_near_zero*100:.1f}% near-zero outputs")


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    args = parser.parse_args()

    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = data_dir.split("/")[-1]

    device = torch.device(config['start_up']['device'])
    model_name = config['start_up']['model_name']
    save_path = f'output/{model_name}_{dataset_name}.pt'
    save_path_resume = f'output/{model_name}_{dataset_name}_resume.pt'
    load_pkl = config['start_up']['load_pkl']

    # configurable seed from yaml (default 0)
    seed = config['start_up'].get('seed', 0)
    set_config(seed)

    # ── load data ──
    t1 = time.time()
    if load_pkl:
        dataloader = pickle.load(
            open(f'output/dataloader_{dataset_name}.pkl', 'rb'))
    else:
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(
            data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(
            dataloader,
            open(f'output/dataloader_{dataset_name}.pkl', 'wb'))
    print(f"Load dataset: {time.time()-t1:.2f}s")
    scaler = dataloader['scaler']

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))
    else:
        _min, _max = None, None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(
        config['data_args']['adj_data_path'],
        config['data_args']['adj_type'])
    print(f"Load adj: {time.time()-t1:.2f}s")

    # ── model args ──
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

    # ── logger ──
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(
        model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # ── init model ──
    model = D2STGNN(**model_args).to(device)
    print(f"[init] model hash: {_model_hash(model)}")

    # register gradient diagnostic hooks
    register_grad_hooks(model)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    print(f"Training iterations per epoch: {len(dataloader['train_loader'])}")

    mode = config['start_up']['mode']
    assert mode in ('test', 'resume', 'scratch')
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    elif mode == 'resume':
        resume_epoch = config['start_up']['resume_epoch']
        model = load_model(model, save_path_resume)

    batch_num = resume_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

    # ── training loop ──
    if mode != 'test':
        train_time, val_time = [], []

        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            t_start = time.time()
            current_lr = engine.optimizer.param_groups[0]['lr']

            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for itera, batch in enumerate(
                    dataloader['train_loader'].get_iterator()):
                if len(batch) == 3:
                    x, y, meta = batch
                else:
                    x, y = batch
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(
                    trainx, trainy,
                    batch_num=batch_num, _max=_max, _min=_min)
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - t_start)

            # per-epoch gradient summary
            gnorms = get_grad_norms()
            print_grad_summary(gnorms, top_k=5)

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # ── validation ──
            t_val = time.time()
            mval_loss, mval_mape, mval_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - t_val)

            # output distribution health check
            _output_health_check(model, device, dataloader, tag="val")

            log = ('Ep {:03d} | trn {:.4f} | val {:.4f} | '
                   'val_rmse {:.4f} | val_mape {:.4f} | lr {:.6f}')
            print(log.format(epoch, mtrain_loss, mval_loss,
                             mval_rmse, mval_mape, current_lr))

            logger.log_json({
                "event": "epoch",
                "epoch": epoch,
                "train_loss": mtrain_loss,
                "val_loss": mval_loss,
                "lr": current_lr,
            })

            early_stopping(mval_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            engine.test(model, save_path_resume, device, dataloader,
                        scaler, model_name,
                        _max=_max, _min=_min,
                        loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"Avg train time: {np.mean(train_time):.4f}s/epoch")
        print(f"Avg val time:   {np.mean(val_time):.4f}s/epoch")
    else:
        engine.test(model, save_path_resume, device, dataloader,
                    scaler, model_name, save=False,
                    _max=_max, _min=_min,
                    loss=engine.loss,
                    dataset_name=dataset_name)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Total time: {time.time()-t0:.1f}s")
