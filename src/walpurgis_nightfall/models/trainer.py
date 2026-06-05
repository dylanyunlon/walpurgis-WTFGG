"""
trainer — Nightfall变体
算法改写:
  1. Adam → AdamW (decoupled weight decay)
  2. MultiStepLR → CosineAnnealingWarmRestarts (周期性重启)
  3. 梯度clip前记录范数 (追踪梯度健康)
  4. 训练step中打印全量状态: loss, grad_norm, lr, cl_len
"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from ..utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, temporal_consistency_penalty
from .. import _dbg, _is_debug
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
        # AdamW替代Adam
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)
        # CosineAnnealingWarmRestarts替代MultiStepLR
        if self.if_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=self.lrate * 0.01)
        else:
            self.lr_scheduler = None
        self.loss = masked_mae
        self.clip = 5
        # 梯度范数历史
        self._grad_norm_history = []

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
                if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                    self.cl_len += int(self.if_cl)
        _dbg("trainer.resume", f"epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}", "trainer")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            _dbg("trainer.params", f"Trainable parameters: {param_count:,}", "trainer")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)
        output = self.model(input)
        output = output.transpose(1, 2)
        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            _dbg("trainer.cl_start", f"CL started, lr reset to {self.lrate}", "trainer")
        else:
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 \
                    and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)
        # scale data and calculate loss
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real_val_scaled = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_scaled[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)
        # temporal consistency penalty
        tc_penalty = temporal_consistency_penalty(
            predict[:, :self.cl_len, :],
            real_val_inv[:, :self.cl_len, :] if kwargs['_max'] is None else real_val_scaled[:, :self.cl_len, :])
        loss = mae_loss + tc_penalty
        loss.backward()
        # 梯度范数追踪 (clip前)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float('inf'))
        self._grad_norm_history.append(total_grad_norm.item())
        if _is_debug("trainer") and kwargs['batch_num'] % 50 == 0:
            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"[NF-TRAIN] step={kwargs['batch_num']} loss={loss.item():.4f} "
                  f"mae={mae_loss.item():.4f} tc={tc_penalty.item():.4f} "
                  f"grad_norm={total_grad_norm.item():.2f} lr={lr_now:.6f} "
                  f"cl_len={self.cl_len}", file=sys.stderr, flush=True)
        # gradient clip
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # metrics
        if kwargs['_max'] is not None:
            mape = masked_mape(predict, real_val_scaled, 0.0)
            rmse = masked_rmse(predict, real_val_scaled, 0.0)
        else:
            mape = masked_mape(predict, real_val_inv, 0.0)
            rmse = masked_rmse(predict, real_val_inv, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx)
            output = output.transpose(1, 2)
            if kwargs['_max'] is not None:
                predict = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
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
    def test(model, save_path_resume, device, dataloader, scaler, model_name,
             save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
        y_list = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]
        assert torch.where(y_list == realy)
        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1),
                          kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1),
                         kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = metrics[0], metrics[1], metrics[2]
            log = 'Horizon {:d} | MAE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f}'
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
        log = '(Avg 12h) MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
