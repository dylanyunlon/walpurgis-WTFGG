#!/usr/bin/env python3
"""Tempest training entry point — mirrors train_eclipse.py structure."""
import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from walpurgis_tempest.utils.load_data import load_dataset, load_adj
from walpurgis_tempest.utils.train import set_config, EarlyStopping, data_reshaper
from walpurgis_tempest.utils.log import TrainLogger
from walpurgis_tempest.models.model import D2STGNN
from walpurgis_tempest.models import trainer
from walpurgis_tempest import _dbg, _is_debug, snapshot_model, register_activation_hooks, gradient_health_check


def main():
    parser = argparse.ArgumentParser(description='D2STGNN Tempest Variant')
    parser.add_argument('--dataset', type=str, default='SYNTH')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        os.environ['TEMPEST_DEBUG'] = '1'

    # Load config
    cfg_path = os.path.join('src', 'walpurgis_tempest', 'configs',
                            f'{args.dataset}.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override epochs if specified
    if args.epochs is not None:
        cfg['optim_args']['epochs'] = args.epochs
    epochs_env = os.environ.get('EPOCHS')
    if epochs_env is not None:
        cfg['optim_args']['epochs'] = int(epochs_env)

    device = torch.device(args.device)
    model_name = cfg['start_up']['model_name']
    dataset_name = args.dataset

    print(f"\n[TEM] D2STGNN Tempest Variant")
    print(f"[TEM] Dataset: {dataset_name}")
    print(f"[TEM] Device: {device}")
    print(f"[TEM] Epochs: {cfg['optim_args']['epochs']}")

    # Set seed (SplitMix64)
    set_config(42)

    # Generate synth data if needed (fBm + Gabriel graph)
    if dataset_name == 'SYNTH':
        synth_dir = os.path.join('datasets', 'SYNTH')
        if not os.path.exists(os.path.join(synth_dir, 'train.npz')):
            print("[TEM] Generating synthetic data (fBm + Gabriel graph)...")
            from walpurgis_tempest.generate_synth_data import generate_synth_traffic
            generate_synth_traffic(output_dir=synth_dir)

    # Load adjacency
    adj_path = cfg['data_args']['adj_data_path']
    adj_type = cfg['data_args']['adj_type']
    adjs, adj_mx = load_adj(adj_path, adj_type)
    adjs = [torch.tensor(a).float().to(device) for a in adjs]

    # Load data
    data_dir = cfg['data_args']['data_dir']
    batch_size = cfg['model_args']['batch_size']
    dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size,
                              dataset_name)

    # Model args
    num_nodes = dataloader['x_train'].shape[2]
    model_args = cfg['model_args'].copy()
    model_args['num_nodes'] = num_nodes
    model_args['adjs'] = adjs
    model_args['device'] = device

    # Build model
    model = D2STGNN(**model_args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[TEM] Model params: {num_params:,}")

    # Logger
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(
        {k: str(v) for k, v in model_args.items() if k != 'adjs'})

    # Scaler
    scaler = dataloader['scaler']

    # Optim args
    optim_args = cfg['optim_args'].copy()
    optim_args['cl_steps'] = optim_args.get('cl_epochs', 3) * \
        len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args.get('warm_epochs', 0) * \
        len(dataloader['train_loader'])
    optim_args['steps_per_epoch'] = len(dataloader['train_loader'])

    # Output paths
    os.makedirs('output', exist_ok=True)
    save_path = os.path.join('output', f'{model_name}_{dataset_name}.pt')
    save_path_resume = os.path.join(
        'output', f'{model_name}_{dataset_name}_resume.pt')

    # Trainer (Adan + OneCycleLR)
    engine = trainer(scaler, model, **optim_args)

    # Extra kwargs for PEMS datasets
    extra = {}
    if dataset_name in ('PEMS04', 'PEMS08'):
        import pickle
        _max = pickle.load(
            open(os.path.join("datasets", dataset_name, "max.pkl"), 'rb'))
        _min = pickle.load(
            open(os.path.join("datasets", dataset_name, "min.pkl"), 'rb'))
        extra['_max'] = torch.Tensor(_max).to(device)
        extra['_min'] = torch.Tensor(_min).to(device)
    extra['dataset_name'] = dataset_name

    # Early stopping (trend + curvature)
    patience = optim_args.get('patience', 20)
    early_stopping = EarlyStopping(patience, save_path, verbose=True)

    # Activation tracker
    tracker = None
    if _is_debug():
        tracker = register_activation_hooks(model)

    # Training loop
    total_epochs = optim_args['epochs']
    print(f"\n[TEM] Starting training: {total_epochs} epochs")
    t_start = time.time()

    for epoch in range(1, total_epochs + 1):
        t_epoch = time.time()
        train_loss, train_mape, train_rmse = [], [], []
        dataloader['train_loader'].shuffle()

        for batch_idx, (x, y) in enumerate(
                dataloader['train_loader'].get_iterator()):
            trainx = data_reshaper(x, device)
            trainy = data_reshaper(y, device)
            batch_num = (epoch - 1) * len(dataloader['train_loader']) + \
                batch_idx

            loss, mape, rmse = engine.train(
                trainx, trainy, batch_num=batch_num, **extra)
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)

        # Validation
        val_loss, val_mape, val_rmse = engine.eval(
            device, dataloader, model_name, **extra)

        # OneCycleLR steps per batch (already done in train), no extra step here

        train_time = time.time() - t_epoch
        lr = engine.optimizer.param_groups[0]['lr']

        log = (f"Epoch {epoch:3d}/{total_epochs} | "
               f"Train Loss: {np.mean(train_loss):.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"Val MAPE: {val_mape:.4f} | "
               f"Val RMSE: {val_rmse:.4f} | "
               f"LR: {lr:.6f} | "
               f"Time: {train_time:.1f}s")
        print(log)

        # Log metrics
        logger.log_metrics(
            epoch,
            train_loss=float(np.mean(train_loss)),
            val_loss=float(val_loss),
            val_mape=float(val_mape),
            val_rmse=float(val_rmse),
            lr=lr,
            train_time=train_time)

        # Debug: gradient health + activation report every 5 epochs
        if _is_debug() and epoch % 5 == 0:
            gradient_health_check(model)
            if tracker:
                tracker.report()

        # Debug: model snapshot every 10 epochs
        if _is_debug() and epoch % 10 == 0:
            snapshot_model(model, epoch=epoch)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start

    # Test
    print(f"\n[TEM] Testing best model...")
    best_model = D2STGNN(**model_args).to(device)
    best_model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True))

    trainer.test(best_model, save_path_resume, device, dataloader,
                 scaler, model_name, **extra)

    print(f"\nAvg Train Time: {total_time/epoch:.4f}s/epoch")
    print(f"Best Val  Loss: {early_stopping.val_loss_min:.4f}")
    print(f"\n[TEM] Pipeline complete. Saved: {save_path_resume}")
    print(f"Total time: {total_time:.2f}s")

    # Clean up tracker
    if tracker:
        tracker.remove()


if __name__ == '__main__':
    main()
