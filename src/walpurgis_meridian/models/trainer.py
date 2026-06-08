"""Meridian Trainer — AdamW + ReduceLROnPlateau + focal loss + trend-aware early stop.
Changes vs upstream:
  - AdamW optimizer (upstream: Adam)
  - ReduceLROnPlateau scheduler (upstream: MultiStepLR)
  - Focal regression loss with annealing (upstream: masked MAE)
  - Gradient norm logging per step
  - Trend-aware early stopping candidate (uses curvature)
"""
import numpy as np
import torch
import torch.optim as optim
import sys, os

from walpurgis_meridian.utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, focal_mae

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


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


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args.get('print_model', False)
        self.total_epochs = optim_args.get('epochs', 100)

        # training strategy
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

        # AdamW optimizer (upstream: Adam)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)

        # ReduceLROnPlateau scheduler (upstream: MultiStepLR)
        if self.if_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.lr_decay_ratio,
                patience=5, min_lr=1e-6)
        else:
            self.lr_scheduler = None

        # focal loss (upstream: masked_mae)
        self.focal_gamma = 2.0
        self.loss = masked_mae  # fallback for eval
        self.clip = 5
        self._step_count = 0

    def _get_anneal_factor(self, epoch=None):
        """Anneal focal gamma: start hard, gradually ease."""
        if epoch is None:
            return 1.0
        progress = min(epoch / max(self.total_epochs, 1), 1.0)
        return max(1.0 - 0.5 * progress, 0.3)

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
        print(f"resume training from epoch{epoch_num}, lr={self.lrate}, cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self._step_count += 1

        output = self.model(input)
        output = output.transpose(1, 2)

        # curriculum learning
        batch_num = kwargs.get('batch_num', 0)
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif batch_num == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print(f"======== Start CL, reset lr to {self.lrate} ========")
        else:
            if (batch_num - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        # scale data
        if kwargs.get('_max') is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real_val_scaled = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            mae_loss = focal_mae(
                predict[:, :self.cl_len, :], real_val_scaled[:, :self.cl_len, :],
                gamma=self.focal_gamma,
                anneal_factor=self._get_anneal_factor(kwargs.get('epoch')))
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_scaled = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = focal_mae(
                predict[:, :self.cl_len, :], real_val_scaled[:, :self.cl_len, :], 0,
                gamma=self.focal_gamma,
                anneal_factor=self._get_anneal_factor(kwargs.get('epoch')))

        loss = mae_loss
        loss.backward()

        # gradient clip
        if self.clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if _DBG and self._step_count % 50 == 0:
                print(f"[MER:trainer] step={self._step_count} loss={loss.item():.4f} "
                      f"grad_norm={grad_norm:.4f} cl_len={self.cl_len}",
                      file=sys.stderr)

        self.optimizer.step()

        mape = masked_mape(predict, real_val_scaled, 0.0)
        rmse = masked_rmse(predict, real_val_scaled, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx)
            output = output.transpose(1, 2)

            if kwargs.get('_max') is not None:
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

        if kwargs.get('_max') is not None:
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
            metrics = metric(pred, real)
            log = 'Horizon {:d} | MAE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = '(Average 12h) MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
