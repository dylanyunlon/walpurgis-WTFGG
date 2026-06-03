"""
main.py — v9 port
Algo delta:
  1. 可配置随机种子 (--seed CLI), 控制 torch/numpy/cuda 三路
  2. 训练前对模型参数做 SHA-256 hash → 可复现性校验
  3. 训练结束后对最终输出做 health check:
     检测 NaN / Inf / 极端值 (>10σ) 并警告
  4. 集成 WALPURGIS_V9_DEBUG=1 全局 debug 标志
"""
import argparse, time, os, hashlib
import torch
import pickle
import numpy as np
import yaml
import setproctitle

torch.set_num_threads(1)

from utils.train import set_config, data_reshaper, EarlyStopping, save_model, load_model
from utils.load_data import load_dataset, load_adj
from utils.log import TrainLogger, register_gradient_hooks
from models.losses import masked_mae, masked_rmse, masked_mape, metric
from models.trainer import trainer
from models.model import D2STGNN
from walpurgis_ported_v9 import _dbg

_TAG = "main"


def _seed_everything(seed):
    """v9: 三路种子同步."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _dbg(_TAG, f"seed={seed}  cuda_deterministic=True")


def _param_hash(model):
    """v9: SHA-256 over all parameters → 用于检测权重是否一致."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    digest = h.hexdigest()[:16]
    _dbg(_TAG, f"param_hash={digest}  n_params={sum(p.numel() for p in model.parameters())}")
    return digest


def _output_health_check(tensor, label="output"):
    """v9: NaN / Inf / extreme value 检测."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    mu = tensor.mean().item()
    sigma = tensor.std().item()
    extreme = ((tensor - mu).abs() > 10 * sigma).float().mean().item() * 100

    status = "OK"
    if has_nan:
        status = "NaN DETECTED"
    elif has_inf:
        status = "Inf DETECTED"
    elif extreme > 5.0:
        status = f"EXTREME ({extreme:.1f}% > 10σ)"

    _dbg(_TAG, f"health[{label}]  {status}  mean={mu:.4g}  std={sigma:.4g}  "
               f"range=[{tensor.min().item():.4g},{tensor.max().item():.4g}]")
    if has_nan or has_inf:
        print(f"⚠️  {label} health check FAILED: {status}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    parser.add_argument('--seed', type=int, default=42)        # v9: configurable seed
    args = parser.parse_args()

    # v9: seed
    _seed_everything(args.seed)

    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir     = config['data_args']['data_dir']
    dataset_name = data_dir.split("/")[-1]
    device       = torch.device(config['start_up']['device'])
    model_name   = config['start_up']['model_name']
    save_path    = 'output/' + model_name + "_" + dataset_name + ".pt"
    save_path_resume = 'output/' + model_name + "_" + dataset_name + "_resume.pt"
    load_pkl     = config['start_up']['load_pkl']

    setproctitle.setproctitle(f"{model_name}.{dataset_name}@v9")

    # ── load data ──
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        print(f"Load dataset: {time.time()-t1:.2f}s...")
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        print(f"Load dataset: {time.time()-t1:.2f}s...")
    scaler = dataloader['scaler']

    if dataset_name in ('PEMS04', 'PEMS08'):
        _min = pickle.load(open(f"datasets/{dataset_name}/min.pkl", 'rb'))
        _max = pickle.load(open(f"datasets/{dataset_name}/max.pkl", 'rb'))
    else:
        _min, _max = None, None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    print(f"Load adj: {time.time()-t1:.2f}s...")

    # ── model args ──
    model_args = config['model_args']
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    optim_args = config['optim_args']
    optim_args['cl_steps']   = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])

    # ── logger ──
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # ── model ──
    model = D2STGNN(**model_args).to(device)
    _param_hash(model)

    # v9: register gradient hooks when debug is on
    if os.environ.get('WALPURGIS_V9_DEBUG', '0') == '1':
        register_gradient_hooks(model, logger)

    engine = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

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

    # ── training ──
    if mode != 'test':
        train_time, val_time = [], []
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            t_start = time.time()
            train_loss, train_mape, train_rmse = [], [], []
            dataloader['train_loader'].shuffle()

            for itera, (x, y, _meta) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(trainx, trainy,
                                               batch_num=batch_num, _max=_max, _min=_min)
                print(f"{itera}: {mae:.4f}", end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1

            train_time.append(time.time() - t_start)
            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()

            t_val = time.time()
            ml, mm, mr = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            val_time.append(time.time() - t_val)

            curr = time.strftime("%d-%H-%M", time.localtime())
            print(f"[{curr}] Ep {epoch:03d} | "
                  f"TrLoss {np.mean(train_loss):.4f} MAPE {np.mean(train_mape):.4f} RMSE {np.mean(train_rmse):.4f} | "
                  f"Val {ml:.4f} MAPE {mm:.4f} RMSE {mr:.4f} | "
                  f"LR {engine.optimizer.param_groups[0]['lr']:.6f}")

            early_stopping(ml, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            engine.test(model, save_path_resume, device, dataloader, scaler,
                        model_name, _max=_max, _min=_min, loss=engine.loss,
                        dataset_name=dataset_name)

        print(f"Avg train time: {np.mean(train_time):.4f}s  val time: {np.mean(val_time):.4f}s")
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler,
                    model_name, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=dataset_name)

    # v9: final output health check
    model.eval()
    with torch.no_grad():
        sample_x, sample_y, _ = next(dataloader['test_loader'].get_iterator())
        sample_x = data_reshaper(sample_x, device)
        sample_out = model(sample_x)
        _output_health_check(sample_out, "final_test_sample")


if __name__ == '__main__':
    t_start = time.time()
    main()
    print(f"Total time: {time.time()-t_start:.1f}s")
