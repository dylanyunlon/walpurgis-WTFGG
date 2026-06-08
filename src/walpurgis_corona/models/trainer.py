"""
Corona trainer — 算法改写:
  upstream: Adam + MultiStepLR
  corona: RAdam + CosineAnnealingWarmRestarts
  upstream loss: masked_mae
  corona loss: quantile_loss (分位数回归)
"""
import numpy as np
import torch
import torch.optim as optim
from ..utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, quantile_loss
from .. import _dbg, PerfTimer


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']
        self.perf = PerfTimer()

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

        # Corona改写: RAdam替代Adam
        self.optimizer = optim.RAdam(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)

        # Corona改写: CosineAnnealingWarmRestarts替代MultiStepLR
        if self.if_lr_scheduler:
            t0 = optim_args.get('cosine_t0', 10)
            t_mult = optim_args.get('cosine_t_mult', 2)
            eta_min = optim_args.get('cosine_eta_min', 1e-5)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min)
        else:
            self.lr_scheduler = None

        # Corona改写: quantile loss作为训练损失
        self.quantiles = optim_args.get('quantiles', [0.1, 0.25, 0.5, 0.75, 0.9])
        self.quantile_weights = optim_args.get('quantile_weights', [0.15, 0.2, 0.3, 0.2, 0.15])

        self.loss = masked_mae  # 用于eval
        self.clip = 5

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
        else:
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        if kwargs['_max'] is not None:
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            loss = quantile_loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :],
                                self.quantiles, self.quantile_weights)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(real_val[:, :, :, 0])
            loss = quantile_loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :],
                                self.quantiles, self.quantile_weights, null_val=0)

        self.perf.start("backward")
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.perf.stop("backward")

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)
        return loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)
            if kwargs['_max'] is not None:
                predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
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
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(min(12, yhat.shape[2])):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            metrics_val = metric(pred, real)
            log = 'Horizon {:d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'
            print(log.format(i + 1, metrics_val[0], metrics_val[2], metrics_val[1]))
            amae.append(metrics_val[0])
            amape.append(metrics_val[1])
            armse.append(metrics_val[2])
        print('Average | MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%'.format(
            np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
