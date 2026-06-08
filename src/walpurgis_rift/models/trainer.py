"""
Rift trainer — 算法改写:
  1. Polynomial decay LR替代MultiStepLR: lr = base * (1 - t/T)^power
  2. 频谱正则化 (spectral regularization): 惩罚forecast的高频分量能量
  3. 每N步自动dump训练状态
  4. 集成PerfTimer用于训练阶段计时
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis_rift.utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, logcosh_adaptive
from walpurgis_rift import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    PolyLRTracker, fft_spectrum_monitor)


class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial decay: lr = base_lr * (1 - step/total_steps)^power (Rift特有)"""
    def __init__(self, optimizer, total_steps, power=2.0, min_lr=1e-7, last_epoch=-1):
        self.total_steps = max(total_steps, 1)
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch, self.total_steps)
        factor = max(0.0, (1 - step / self.total_steps)) ** self.power
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrate,
                                    weight_decay=self.wdecay, eps=self.eps)
        total_steps = optim_args.get('_steps_per_epoch', 50) * optim_args.get('epochs', 3)
        self.lr_scheduler = PolynomialDecayLR(
            self.optimizer, total_steps=total_steps, power=2.0, min_lr=self.lrate * 0.01)
        self.loss = masked_mae
        self.logcosh_loss = logcosh_adaptive
        self._use_logcosh = True
        self.clip = 5
        self._spectral_reg_weight = 0.001
        self._spectral_reg_warmup = 50
        self.perf = PerfTimer()
        self.lr_tracker = PolyLRTracker()
        self._global_step = 0

    def _spectral_regularization(self, forecast_output):
        """频谱正则化: 惩罚预测中高频分量的能量 (Rift特有)"""
        if self._global_step < self._spectral_reg_warmup:
            return torch.tensor(0.0, device=forecast_output.device)
        spec = torch.fft.rfft(forecast_output, dim=1)
        amplitude = spec.abs()
        n_freqs = amplitude.shape[1]
        high_freq_start = max(n_freqs // 2, 1)
        high_freq_energy = amplitude[:, high_freq_start:, :].pow(2).mean()
        total_energy = amplitude.pow(2).mean() + 1e-8
        reg = high_freq_energy / total_energy
        _dbg("spectral_reg", f"hf_ratio={reg.item():.6f}", "loss")
        return reg * self._spectral_reg_weight

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
                if ((_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len):
                    self.cl_len += int(self.if_cl)

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.perf.start("forward")
        output = self.model(input)
        output = output.transpose(1, 2)
        self.perf.stop("forward")
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning, "
                  f"lr reset to {self.lrate} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)
        self.perf.start("loss")
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(real_val[:, :, :, 0])
            if self._use_logcosh:
                mae_loss = self.logcosh_loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :], 0)
            else:
                mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :], 0)
        spec_reg = self._spectral_regularization(predict[:, :self.cl_len, :])
        loss = mae_loss + spec_reg
        self.perf.stop("loss")
        self.perf.start("backward")
        loss.backward()
        self.perf.stop("backward")
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_tracker.record(self._global_step, current_lr)
        if _is_debug() and self._global_step % 20 == 0:
            fft_spectrum_monitor(predict.detach(), tag=f"step_{self._global_step}")
        self._global_step += 1
        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
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
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
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
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            metrics = metric(pred, real)
            mae, mape, rmse = metrics[0], metrics[1], metrics[2]
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
        log = '(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
