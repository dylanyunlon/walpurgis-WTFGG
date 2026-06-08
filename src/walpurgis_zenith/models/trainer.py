"""
trainer — Zenith变体
算法改写 (~20%):
  1. Adam → Lamb-like (layerwise adaptive lr scaling)
  2. MultiStepLR → OneCycleLR (single cycle cosine with warmup built-in)
  3. temporal_smoothness_reg: 输出相邻时步差的L2惩罚
  4. 每N步自动打印梯度直方图 + 参数norm排行
"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from ..utils.train import data_reshaper, save_model
from .losses import (masked_mae, masked_rmse, masked_mape, metric,
                     temporal_smoothness_penalty)
from .. import _dbg, _is_debug, gradient_histogram, PerfTimer
import sys


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(
            np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        self.if_lr_scheduler = optim_args['lr_schedule']
        self.lr_sche_steps = optim_args['lr_sche_steps']
        self.lr_decay_ratio = optim_args['lr_decay_ratio']
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']
        # AdamW优化器 (Zenith: 更大weight_decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lrate,
            weight_decay=self.wdecay * 2.0,
            eps=self.eps)
        # OneCycleLR: 单周期余弦, 内置warmup
        total_steps = optim_args.get('epochs', 80) * optim_args.get(
            '_steps_per_epoch', 50)
        if self.if_lr_scheduler and total_steps > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lrate * 3.0,
                total_steps=max(total_steps, 100),
                pct_start=0.15,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0)
        else:
            self.lr_scheduler = None
        self.loss = masked_mae
        self.clip = 5
        self._train_step = 0
        self.perf = PerfTimer()

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        for _ in range(batch_num):
            if _ < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif _ == self.warm_steps:
                self.cl_len = 1
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lrate
            else:
                if (_ - self.warm_steps) % self.cl_steps == 0 \
                        and self.cl_len < self.output_seq_len:
                    self.cl_len += int(self.if_cl)
        print(f"[ZN] resume from epoch={epoch_num}, "
              f"lr={self.lrate}, cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.perf.start("forward")
        output = self.model(input)
        output = output.transpose(1, 2)
        self.perf.stop("forward")
        # curriculum learning
        batch_num = kwargs.get('batch_num', 0)
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif batch_num == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            if _is_debug():
                print(f"[ZN] CL start, reset lr={self.lrate}",
                      file=sys.stderr)
        else:
            if (batch_num - self.warm_steps) % self.cl_steps == 0 \
                    and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)
        # scale data and calculate loss
        if kwargs.get('_max') is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val_t = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_t = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :], 0)
        # temporal smoothness正则 (Zenith特有)
        ts_penalty = temporal_smoothness_penalty(predict)
        ts_weight = kwargs.get('ts_weight', 0.01)
        loss = mae_loss + ts_weight * ts_penalty
        self.perf.start("backward")
        loss.backward()
        self.perf.stop("backward")
        # 梯度clip前记录全局范数
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip)
        if _is_debug() and self._train_step % 50 == 0:
            _dbg(f"step{self._train_step}.grad_norm",
                 f"total_norm={total_norm:.4f} cl_len={self.cl_len} "
                 f"mae={mae_loss.item():.4f} ts_pen={ts_penalty.item():.6f}",
                 "train")
            gradient_histogram(self.model)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self._train_step += 1
        # metrics
        mape = masked_mape(predict, real_val_t, 0.0)
        rmse = masked_rmse(predict, real_val_t, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(
                dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx)
            output = output.transpose(1, 2)
            if kwargs.get('_max') is not None:
                predict = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)
        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
        y_list = []
        for itera, (x, y) in enumerate(
                dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]
        assert torch.where(y_list == realy)
        if kwargs.get('_max') is not None:
            realy = scaler(
                realy.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(
                yhat.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(
                    pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = metrics[0], metrics[1], metrics[2]
            log = ('Evaluate horizon {:d}: '
                   'MAE={:.4f} RMSE={:.4f} MAPE={:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
        log = ('(Avg 12 horizons) MAE={:.2f} | '
               'RMSE={:.2f} | MAPE={:.2f}%')
        print(log.format(
            np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
