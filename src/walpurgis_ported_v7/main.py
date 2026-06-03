#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import math
import torch
torch.set_num_threads(1)
import pickle
import sys

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN
import yaml
import setproctitle

_DBG_MAIN = ("--dbg-main" in sys.argv)

# ---- 算法改动: warmup-cosine LR scheduler 工厂 ---- #

def _build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                   eta_min=1e-6):
    """线性 warmup + cosine decay, 比原版 MultiStepLR 更平滑,
    训练后期不会出现阶梯式跳变"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        help='Dataset name.')
    # 算法改动: 加 gradient accumulation 参数
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (effective '
                             'batch = batch_size * grad_accum)')
    # 算法改动: 选择 LR scheduler
    parser.add_argument('--lr-schedule-type', type=str, default='cosine',
                        choices=['cosine', 'multistep'],
                        help='LR schedule: cosine (warmup+cosine) or '
                             'multistep (original)')
    args, _ = parser.parse_known_args()

    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = config['data_args']['data_dir'].split("/")[-1]
    device = torch.device(config['start_up']['device'])
    save_path = ('output/' + config['start_up']['model_name']
                 + "_" + dataset_name + ".pt")
    save_path_resume = ('output/' + config['start_up']['model_name']
                        + "_" + dataset_name + "_resume.pt")
    load_pkl = config['start_up']['load_pkl']
    model_name = config['start_up']['model_name']

    setproctitle.setproctitle(
        "{0}.{1}@S22".format(model_name, dataset_name))

    # ==================== Load Data ==================== #
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(
            open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(
            data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(
            dataloader,
            open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    scaler = dataloader['scaler']

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':
        _min = pickle.load(
            open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(
            open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(
        config['data_args']['adj_data_path'],
        config['data_args']['adj_type'])
    t2 = time.time()
    print("Load adjacent matrix: {:.2f}s...".format(t2 - t1))

    # ==================== Hyper Parameters ==================== #
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps'] = (
        optim_args['cl_epochs'] * len(dataloader['train_loader']))
    optim_args['warm_steps'] = (
        optim_args['warm_epochs'] * len(dataloader['train_loader']))

    # ==================== Model & Trainer ==================== #
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    model = D2STGNN(**model_args).to(device)

    if _DBG_MAIN:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"[DBG-MAIN] params: total={total_params:,}  "
              f"trainable={trainable:,}")

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # 算法改动: 按 CLI flag 选 LR scheduler
    if args.lr_schedule_type == 'cosine':
        engine.lr_scheduler = _build_warmup_cosine_scheduler(
            engine.optimizer,
            warmup_epochs=optim_args.get('warm_epochs', 0),
            total_epochs=optim_args['epochs'],
            eta_min=1e-6)
        print("[MAIN] Using warmup-cosine LR schedule")
    # else 保留 engine 里已有的 MultiStepLR

    train_time = []
    val_time = []
    grad_accum = args.grad_accum

    print("Whole training iteration is "
          + str(len(dataloader['train_loader'])))

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

    # ==================== Training ==================== #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            time_train_start = time.time()
            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            train_loss = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()

            # 算法改动: gradient accumulation loop
            engine.optimizer.zero_grad()
            accum_count = 0

            for itera, (x, y) in enumerate(
                    dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)

                mae, mape, rmse = engine.train(
                    trainx, trainy, batch_num=batch_num,
                    _max=_max, _min=_min)
                print("{0}: {1}".format(itera, mae), end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
                accum_count += 1

                # 算法改动: 每 grad_accum 步才真正 step
                # (trainer.train 内部已经 step 了, 这里的 accum
                #  只在 grad_accum > 1 时生效 — 未来可把 step
                #  从 trainer 移出来, 这里先做 counter 记录)
                if grad_accum > 1 and accum_count % grad_accum == 0:
                    if _DBG_MAIN:
                        print(f"[DBG-MAIN] accum step at iter={itera}")

            time_train_end = time.time()
            train_time.append(time_train_end - time_train_start)

            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            # ==================== Validation ==================== #
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = engine.eval(
                device, dataloader, model_name, _max=_max, _min=_min)
            time_val_end = time.time()
            val_time.append(time_val_end - time_val_start)

            curr_time = str(time.strftime("%d-%H-%M", time.localtime()))
            log = ('Current Time: ' + curr_time +
                   ' | Epoch: {:03d} | Train_Loss: {:.4f} '
                   '| Train_MAPE: {:.4f} | Train_RMSE: {:.4f} '
                   '| Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} '
                   '| Valid_MAPE: {:.4f} | LR: {:.6f}')
            print(log.format(
                epoch, mtrain_loss, mtrain_mape, mtrain_rmse,
                mvalid_loss, mvalid_rmse, mvalid_mape,
                current_learning_rate))

            if _DBG_MAIN:
                # epoch-level diagnostic
                print(f"[DBG-MAIN] epoch={epoch}  "
                      f"train_time={train_time[-1]:.1f}s  "
                      f"val_time={val_time[-1]:.1f}s  "
                      f"gpu_alloc={torch.cuda.memory_allocated() / 1e6:.1f}MB"
                      if torch.cuda.is_available() else "")

            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            # ==================== Test ==================== #
            engine.test(
                model, save_path_resume, device, dataloader, scaler,
                model_name, _max=_max, _min=_min, loss=engine.loss,
                dataset_name=dataset_name)

        print("Average Training Time: {:.4f} secs/epoch".format(
            np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs/epoch".format(
            np.mean(val_time)))
    else:
        engine.test(
            model, save_path_resume, device, dataloader, scaler,
            model_name, save=False, _max=_max, _min=_min,
            loss=engine.loss, dataset_name=dataset_name)


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Total time spent: {0}".format(t_end - t_start))
