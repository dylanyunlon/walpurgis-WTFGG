"""
D2STGNN CardGame variant — trainer.py
Algorithm changes vs upstream:
  1. Adam → RAdam optimizer (rectified Adam with variance rectification)
  2. MultiStepLR → OneCycleLR scheduler (super-convergence)
  3. Gradient norm tracking (logged per train step in debug mode)
  4. Cosine temperature annealing for curriculum learning pace
"""

import os
import sys
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from walpurgis_cardgame.utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


def _compute_grad_norm(model):
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


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
        self.model  = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']

        # training strategy parameters
        self.lrate  = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        ## learning rate scheduler
        self.if_lr_scheduler    = optim_args['lr_schedule']
        self.lr_sche_steps      = optim_args['lr_sche_steps']
        self.lr_decay_ratio     = optim_args['lr_decay_ratio']
        ## curriculum learning
        self.if_cl          = optim_args['if_cl']
        self.cl_steps       = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        ## warmup
        self.warm_steps     = optim_args['warm_steps']

        # --- CARDGAME: RAdam optimizer (replaces Adam) ---
        self.optimizer = optim.RAdam(
            self.model.parameters(),
            lr=self.lrate,
            weight_decay=self.wdecay,
            eps=self.eps)

        # --- CARDGAME: OneCycleLR scheduler (replaces MultiStepLR) ---
        total_epochs = optim_args.get('epochs', 50)
        steps_per_epoch = max(optim_args.get('cl_steps', 100) // max(optim_args.get('cl_epochs', 1), 1), 1)
        if self.if_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lrate * 5,
                total_steps=total_epochs * steps_per_epoch + 1,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0)
        else:
            self.lr_scheduler = None

        # loss
        self.loss   = masked_mae
        self.clip   = 5

        # --- CARDGAME: gradient norm tracking ---
        self._grad_norms = []

        # --- CARDGAME: cosine temperature for curriculum learning ---
        self._total_epochs = total_epochs

    def _cosine_cl_temperature(self, batch_num):
        """Cosine annealing temperature: smoothly transitions curriculum length."""
        if not self.if_cl:
            return self.output_seq_len
        if batch_num < self.warm_steps:
            return self.output_seq_len
        progress = min((batch_num - self.warm_steps) / max(self.cl_steps * self.output_seq_len, 1), 1.0)
        # cosine annealing from 1 to output_seq_len
        cosine_val = 0.5 * (1 - math.cos(math.pi * progress))
        cl_len = max(1, int(1 + cosine_val * (self.output_seq_len - 1)))
        return min(cl_len, self.output_seq_len)

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
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
            print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                tmp = 1
                for _ in param.shape:
                    tmp = tmp * _
                parameter_num += tmp
            print("Parameter size: {0}".format(parameter_num))

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # --- CARDGAME: cosine curriculum learning temperature ---
        self.cl_len = self._cosine_cl_temperature(kwargs['batch_num'])
        _dbg("trainer.cl_len", self.cl_len, "trainer")

        # scale data and calculate loss
        if kwargs['_max'] is not None:  # traffic flow
            predict  = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1,2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            predict  = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:,:,:,0])
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)
        loss = mae_loss
        _dbg("trainer.loss", loss, "trainer")

        loss.backward()

        # --- CARDGAME: gradient norm tracking ---
        grad_norm = _compute_grad_norm(self.model)
        self._grad_norms.append(grad_norm)
        if _CG_DEBUG:
            print(f"[CG-DBG:trainer.grad_norm] value={grad_norm:.6f}", file=sys.stderr)

        # gradient clip and optimization
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # metrics
        if kwargs['_max'] is not None:
            mape = masked_mape(predict, real_val, 0.0)
            rmse = masked_rmse(predict, real_val, 0.0)
        else:
            mape = masked_mape(predict, real_val_inv, 0.0)
            rmse = masked_rmse(predict, real_val_inv, 0.0)
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

            if kwargs['_max'] is not None:
                predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:,:,:,0])

            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)
        realy   = realy.transpose(1, 2)
        y_list  = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)
        yhat   = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]

        assert torch.where(y_list == realy)

        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat  = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat  = scaler.inverse_transform(yhat)

        amae  = []
        amape = []
        armse = []

        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):
                mae  = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
                log  = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i+1, mae, rmse, mape))
                amae.append(mae)
                amape.append(mape)
                armse.append(rmse)
            else:
                metrics = metric(pred, real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i+1, metrics[0], metrics[2], metrics[1]))
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

        log = '(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
