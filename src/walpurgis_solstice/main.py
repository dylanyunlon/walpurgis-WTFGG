import os
import sys
import argparse
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from walpurgis_solstice.utils.train import set_config, data_reshaper, save_model, EarlyStopping, get_num_params
from walpurgis_solstice.utils.load_data import load_dataset, load_adj
from walpurgis_solstice.utils.log import TrainLogger
from walpurgis_solstice.models.model import D2STGNN
from walpurgis_solstice.models.trainer import trainer
from walpurgis_solstice import snapshot_model, register_activation_hooks, gradient_health_check, _is_debug

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    print(f"[SOL:main:{tag}] {val}", file=sys.stderr)


def main(config_path, device_str='cpu', epochs_override=None):
    cfg = set_config(config_path)
    device = torch.device(device_str)

    data_cfg = cfg['DATA']
    model_cfg = cfg['MODEL']
    train_cfg = cfg['TRAIN']

    dataset_name = data_cfg['DATASET_NAME']
    num_nodes = data_cfg['NUM_NODES']

    if epochs_override is not None:
        train_cfg['EPOCHS'] = epochs_override

    # Load data
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'datasets', 'raw_data', dataset_name)
    adj_dir = os.path.join(base_dir, 'datasets', 'sensor_graph')

    dataloader = load_dataset(
        data_dir, batch_size=train_cfg['BATCH_SIZE'],
        dataset_name=dataset_name,
        seq_length_x=data_cfg.get('SEQ_LENGTH_X', 12),
        seq_length_y=data_cfg.get('SEQ_LENGTH_Y', 12),
        normalizer=data_cfg.get('NORMALIZER', 'std'))

    if dataloader is None:
        print(f"ERROR: Could not load data from {data_dir}")
        sys.exit(1)

    # Load adjacency
    adj_file = os.path.join(adj_dir, f'adj_{dataset_name.lower()}.npy')
    if os.path.exists(adj_file):
        adjs = load_adj(adj_file, num_nodes=num_nodes)
    else:
        _adbg("adj", f"No adj file, using identity {num_nodes}x{num_nodes}")
        adjs = [torch.eye(num_nodes)]

    adjs = [a.to(device) for a in adjs]

    # Model args
    model_args = {
        'num_feat': model_cfg['NUM_FEAT'],
        'num_hidden': model_cfg['NUM_HIDDEN'],
        'node_hidden': model_cfg['NODE_HIDDEN'],
        'time_emb_dim': model_cfg['TIME_EMB_DIM'],
        'k_s': model_cfg['K_S'],
        'k_t': model_cfg['K_T'],
        'gap': model_cfg['GAP'],
        'seq_length': model_cfg['SEQ_LENGTH'],
        'num_nodes': num_nodes,
        'dropout': model_cfg['DROPOUT'],
        'adjs': adjs
    }

    model = D2STGNN(**model_args).to(device)
    total_params = get_num_params(model)
    print(f"Solstice D2STGNN: {total_params:,} trainable parameters")
    _adbg("model_params", total_params)

    # Trainer
    optim_args = {
        'output_seq_len': model_cfg['SEQ_LENGTH'],
        'print_model': train_cfg.get('PRINT_MODEL', False),
        'lrate': train_cfg['LRATE'],
        'wdecay': train_cfg['WDECAY'],
        'eps': train_cfg['EPS'],
        'lr_schedule': train_cfg.get('LR_SCHEDULE', False),
        'lr_sche_steps': train_cfg.get('LR_SCHE_STEPS', [50, 80]),
        'lr_decay_ratio': train_cfg.get('LR_DECAY_RATIO', 0.5),
        'if_cl': train_cfg.get('IF_CL', True),
        'cl_steps': train_cfg.get('CL_STEPS', 3),
        'warm_steps': train_cfg.get('WARM_STEPS', 30)
    }

    scaler = dataloader['scaler']
    if hasattr(scaler, 'inverse_transform'):
        engine = trainer(scaler, model, **optim_args)
    else:
        engine = trainer(scaler, model, **optim_args)

    logger = TrainLogger('solstice', dataset_name)
    early_stop = EarlyStopping(patience=train_cfg.get('PATIENCE', 15))

    save_path = train_cfg.get('SAVE_PATH', f'checkpoints/solstice_{dataset_name.lower()}')

    # Debug hooks
    tracker = None
    if _is_debug():
        tracker = register_activation_hooks(model)
        snapshot_model(model, epoch=0, step=0)

    # Train
    epochs = train_cfg['EPOCHS']
    print(f"\nStarting training: {epochs} epochs, batch_size={train_cfg['BATCH_SIZE']}")
    print(f"Dataset: {dataset_name}, Nodes: {num_nodes}")
    print(f"Debug: {_is_debug()}\n")

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss_list = []
        dataloader['train_loader'].shuffle()
        for batch_num, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = data_reshaper(x, device)
            trainy = data_reshaper(y, device)
            kwargs = {'batch_num': (epoch - 1) * dataloader['train_loader'].num_batch + batch_num}
            if hasattr(scaler, '_min'):
                kwargs['_max'] = dataloader['_max'].to(device)
                kwargs['_min'] = dataloader['_min'].to(device)
            loss, mape, rmse = engine.train(trainx, trainy, **kwargs)
            train_loss_list.append(loss)

        train_avg = np.mean(train_loss_list)

        # Validation
        kwargs_eval = {}
        if hasattr(scaler, '_min'):
            kwargs_eval['_max'] = dataloader['_max'].to(device)
            kwargs_eval['_min'] = dataloader['_min'].to(device)
        val_loss, val_mape, val_rmse = engine.eval(device, dataloader, 'solstice', **kwargs_eval)

        lr = engine.optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_avg, val_loss, val_mape, val_rmse, lr)

        if engine.lr_scheduler is not None:
            engine.lr_scheduler.step(epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_model(model, save_path + '_best.pt')

        early_stop(val_loss)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        # Debug snapshots
        if _is_debug() and epoch % max(1, epochs // 3) == 0:
            snapshot_model(model, epoch=epoch)
            gradient_health_check(model)
            if tracker:
                tracker.report()

    # Test
    print("\n--- Testing ---")
    best_path = save_path + '_best.pt'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    kwargs_test = {}
    if hasattr(scaler, '_min'):
        kwargs_test['_max'] = dataloader['_max'].to(device)
        kwargs_test['_min'] = dataloader['_min'].to(device)
    trainer.test(model, save_path + '_final.pt', device, dataloader,
                 scaler if hasattr(scaler, 'inverse_transform') else scaler,
                 'solstice', **kwargs_test)
    logger.summary()
    logger.save_log()

    if tracker:
        tracker.remove()

    print("\n[Solstice] Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/SYNTH.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    main(config_path, args.device, args.epochs)
