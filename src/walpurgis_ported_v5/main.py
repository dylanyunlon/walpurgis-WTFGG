#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import torch
torch.set_num_threads(1)
import pickle
import traceback
import sys

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN, TensorProbe
import yaml
import setproctitle

# Delta vs upstream:
#   1. Loads debug_args from config and wires probes/profiler/watchdog
#   2. Crash handler: dumps TensorProbe state + grad watchdog on exception
#   3. Periodic snapshot every N epochs (configurable)
#   4. Phase profiler wraps train/val/test loops
#   5. Budget check at end of each epoch


def _crash_dump(model, engine, epoch, batch_num):
    """Emergency state dump on unhandled exception."""
    print("\n" + "=" * 60)
    print(f"CRASH DUMP  epoch={epoch}  batch={batch_num}")
    print("=" * 60)
    try:
        print("\n── TensorProbe registry ──")
        TensorProbe.dump_all()
        print("\n── Gradient Watchdog ──")
        engine.grad_watchdog.report()
        print("\n── Phase Profiler ──")
        engine.profiler.report()
        print("\n── Model Snapshot ──")
        print(model.snapshot())
    except Exception as e2:
        print(f"(crash dump itself failed: {e2})")
    print("=" * 60 + "\n")


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name.')
    args = parser.parse_args()

    config_path = "configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]

    device          = torch.device(config['start_up']['device'])
    save_path       = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    setproctitle.setproctitle("{0}.{1}@S22".format(model_name, dataset_name))

    # ── delta 1: load debug args ──
    debug_args = config.get('debug_args', {})
    probe_active    = debug_args.get('probe_depth', 0) > 0
    dump_on_nan     = debug_args.get('dump_on_nan', True)
    budget_sec      = debug_args.get('phase_budget_sec', 0)
    snapshot_every  = debug_args.get('snapshot_every_n', 0)

# ========================== load dataset ====================== #
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    scaler = dataloader['scaler']

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    t2 = time.time()
    print("Load adjacent matrix: {:.2f}s...".format(t2 - t1))

# ================================ Hyper Parameters ================================= #
    model_args  = config['model_args']
    model_args['device']        = device
    model_args['num_nodes']     = adj_mx[0].shape[0]
    model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
    model_args['dataset']       = dataset_name

    optim_args                  = config['optim_args']
    optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])

# ============================= Model and Trainer ============================= #
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model  = D2STGNN(**model_args).to(device)

    # ── delta 1: activate probes if configured ──
    if probe_active:
        for name, probe in TensorProbe._registry.items():
            probe._active = True
        print(f"[DEBUG] {len(TensorProbe._registry)} probes activated")

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    train_time = []
    val_time   = []

    print("Whole training iteration is " + str(len(dataloader['train_loader'])))

    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = load_model(model, save_path)
    else:
        if mode == 'resume':
            resume_epoch = config['start_up']['resume_epoch']
            model = load_model(model, save_path_resume)
        else:
            resume_epoch = 0

    batch_num = resume_epoch * len(dataloader['train_loader'])
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)

# =============================================================== Training ================================================================= #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            try:
                # ── delta 4: phase profiler ──
                engine.profiler.begin("train")
                time_train_start = time.time()

                current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
                train_loss = []
                train_mape = []
                train_rmse = []
                dataloader['train_loader'].shuffle()
                for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                    trainx = data_reshaper(x, device)
                    trainy = data_reshaper(y, device)

                    # ── delta 2: NaN guard ──
                    if dump_on_nan and (torch.isnan(trainx).any() or torch.isnan(trainy).any()):
                        print(f"\033[91m[NaN in input] epoch={epoch} iter={itera}\033[0m")

                    mae, mape, rmse = engine.train(
                        trainx, trainy, batch_num=batch_num,
                        _max=_max, _min=_min)
                    print("{0}: {1}".format(itera, mae), end='\r')
                    train_loss.append(mae)
                    train_mape.append(mape)
                    train_rmse.append(rmse)
                    batch_num += 1

                time_train_end = time.time()
                train_time.append(time_train_end - time_train_start)
                engine.profiler.end()

                current_learning_rate = engine.optimizer.param_groups[0]['lr']

                if engine.if_lr_scheduler:
                    engine.lr_scheduler.step()

                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

# =============================================================== Validation ================================================================= #
                time_val_start = time.time()
                mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                    device, dataloader, model_name, _max=_max, _min=_min)
                time_val_end = time.time()
                val_time.append(time_val_end - time_val_start)

                curr_time = str(time.strftime("%d-%H-%M", time.localtime()))
                log = ('Current Time: ' + curr_time +
                       ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} '
                       '| Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} '
                       '| Valid_MAPE: {:.4f} | LR: {:.6f}')
                print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
                                 mvalid_loss, mvalid_rmse, mvalid_mape,
                                 current_learning_rate))

                # ── delta 5: budget check ──
                if budget_sec > 0:
                    engine.profiler.budget_check(budget_sec)

                # ── delta 3: periodic snapshot ──
                if snapshot_every > 0 and epoch % snapshot_every == 0:
                    print(f"[SNAPSHOT epoch={epoch}] {model.snapshot()}")
                    TensorProbe.anomaly_summary()

                early_stopping(mvalid_loss, engine.model)
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break

# =============================================================== Test ================================================================= #
                engine.profiler.begin("test")
                engine.test(model, save_path_resume, device, dataloader,
                            scaler, model_name,
                            _max=_max, _min=_min, loss=engine.loss,
                            dataset_name=dataset_name)
                engine.profiler.end()

            except Exception as exc:
                # ── delta 2: crash dump ──
                _crash_dump(model, engine, epoch, batch_num)
                raise

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))

        # final profiler report
        engine.profiler.report()
    else:
        engine.test(model, save_path_resume, device, dataloader,
                    scaler, model_name, save=False,
                    _max=_max, _min=_min, loss=engine.loss,
                    dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Total time spent: {0}".format(t_end - t_start))
