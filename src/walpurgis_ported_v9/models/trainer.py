"""
trainer.py — v9 port
Algo delta:
  1. 梯度裁剪: 固定 clip=5 → 自适应: 维护最近 100 步梯度范数,
     取 p95 作为 clip 阈值. 训练初期自动松, 收敛期自动紧.
  2. CL warmup: 阶梯增长 → cosine ramp-up:
     cl_len = round(L * 0.5*(1 - cos(π * progress)))
     更平滑, 减少 CL 阶跃带来的 loss 震荡.
  3. 每 50 步打印完整的 per-layer 梯度范数 snapshot
  4. eval 打印 per-horizon MAE (debug 时可看哪个 horizon 最差)
"""
import collections
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric
from walpurgis_ported_v9 import _dbg

_TAG = "trainer"

_GRAD_WINDOW = 100
_GRAD_PRINT_EVERY = 50


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer:
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

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio)
            if self.if_lr_scheduler else None)

        self.loss = masked_mae
        # v9: adaptive gradient clip (p95 rolling)
        self._grad_history = collections.deque(maxlen=_GRAD_WINDOW)
        self._step_counter = 0

    def _adaptive_clip(self):
        """Return p95 of recent gradient norms as clip threshold."""
        if len(self._grad_history) < 10:
            return 5.0        # fallback at start
        arr = np.array(self._grad_history)
        threshold = float(np.percentile(arr, 95))
        return max(threshold, 0.1)

    def _cosine_cl(self, batch_num):
        """v9: cosine ramp-up for curriculum length."""
        if batch_num < self.warm_steps:
            return self.output_seq_len
        progress = min((batch_num - self.warm_steps) / max(self.cl_steps * self.output_seq_len, 1), 1.0)
        frac = 0.5 * (1.0 - math.cos(math.pi * progress))
        cl = max(1, round(self.output_seq_len * frac))
        return min(cl, self.output_seq_len)

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        # just restore CL length via cosine formula
        self.cl_len = self._cosine_cl(batch_num)
        _dbg(_TAG, f"resume  epoch={epoch_num}  cl_len={self.cl_len}  lr={self.lrate}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input).transpose(1, 2)

        # v9: cosine CL
        batch_num = kwargs['batch_num']
        if self.if_cl:
            self.cl_len = self._cosine_cl(batch_num)

        if kwargs['_max'] is not None:
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0],
                                  kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0],
                                  kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :],
                                real_val[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(predict[:, :self.cl_len, :],
                                real_val[:, :self.cl_len, :], 0)
        loss = mae_loss
        loss.backward()

        # v9: record grad norm then adaptive clip
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e9).item()
        self._grad_history.append(total_norm)
        clip_val = self._adaptive_clip()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        self.optimizer.step()

        self._step_counter += 1
        # v9: periodic per-layer grad snapshot
        if self._step_counter % _GRAD_PRINT_EVERY == 0:
            _dbg(_TAG, f"step={self._step_counter}  clip={clip_val:.3f}  "
                        f"grad_norm={total_norm:.4f}  cl_len={self.cl_len}")
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    gn = p.grad.norm(2).item()
                    if gn > clip_val * 0.5:
                        _dbg(_TAG, f"  ∇ {name}  norm={gn:.4f}")

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)

            if kwargs['_max'] is not None:
                predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                      kwargs["_max"][0, 0, 0, 0],
                                      kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1),
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

        ml, mm, mr = np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)
        _dbg(_TAG, f"eval  loss={ml:.4f}  mape={mm:.4f}  rmse={mr:.4f}")
        return ml, mm, mr

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        model.eval()
        outputs, y_list = [], []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)

        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)

        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]

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
            if kwargs['dataset_name'] in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                mae, mape, rmse, _ = metric(pred, real)
            log = 'Horizon {:2d} | MAE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f}'
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        print('Avg 12h | MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%'.format(
            np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
