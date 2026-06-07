import numpy as np
import torch
import torch.optim as optim
import sys, os
from sklearn.metrics import mean_absolute_error

from walpurgis_aurora.utils.train import data_reshaper, save_model
from .losses import cauchy_loss, temporal_coherence_penalty, masked_mae, masked_rmse, masked_mape, metric

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[AUR:trainer:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[AUR:trainer:{tag}] {val}", file=sys.stderr)


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
        # aurora: 对数CL增长代替线性
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']

        # upstream: Adam + MultiStepLR
        # aurora: AdamW + ReduceLROnPlateau (基于验证loss自动调参)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lrate,
                                     weight_decay=self.wdecay, eps=self.eps)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5,
            min_lr=1e-6) if self.if_lr_scheduler else None

        # upstream: masked_mae
        # aurora: cauchy_loss + temporal_coherence_penalty
        self.loss = cauchy_loss
        # upstream: 固定clip=5
        # aurora: 自适应裁剪 (初始5, 按p90梯度动态调)
        self.clip = 5.0
        self._grad_history = []

    def _adaptive_clip(self):
        """aurora: 根据最近梯度范数的p90分位数动态调整clip阈值"""
        if len(self._grad_history) < 10:
            return self.clip
        recent = self._grad_history[-50:]
        p90 = np.percentile(recent, 90)
        self.clip = max(1.0, min(p90 * 1.5, 20.0))
        return self.clip

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        import math
        for _ in range(batch_num):
            if _ < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif _ == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lrate
            else:
                # aurora: 对数增长 cl_len = 1 + int(log2(1 + steps/cl_steps) * out_len)
                elapsed = _ - self.warm_steps
                if self.cl_steps > 0:
                    self.cl_len = 1 + int(math.log2(1 + elapsed / self.cl_steps) * self.output_seq_len)
                    self.cl_len = min(self.cl_len, self.output_seq_len)
        _adbg("resume", f"epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs.get('batch_num', 1)) == 0:
            total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            _adbg("param_count", total)

    def train(self, input, real_val, **kwargs):
        import math
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # aurora: 对数CL增长
        bn = kwargs.get('batch_num', 0)
        if bn < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            _adbg("cl_start", f"reset lr={self.lrate}")
        else:
            elapsed = bn - self.warm_steps
            if self.cl_steps > 0 and self.if_cl:
                self.cl_len = 1 + int(math.log2(1 + elapsed / self.cl_steps) * self.output_seq_len)
                self.cl_len = min(self.cl_len, self.output_seq_len)

        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            main_loss = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:, :, :, 0])
            main_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)

        # aurora: temporal coherence penalty
        tp = temporal_coherence_penalty(predict)
        loss = main_loss + tp
        loss.backward()

        # aurora: 自适应梯度裁剪
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e9).item()
        self._grad_history.append(total_norm)
        clip_val = self._adaptive_clip()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

        self.optimizer.step()

        _adbg("step", f"loss={loss.item():.6f} cauchy={main_loss.item():.6f} tp={tp.item():.6f} "
               f"grad={total_norm:.4f} clip={clip_val:.2f} cl_len={self.cl_len} "
               f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

        if kwargs.get('_max') is not None:
            mape = masked_mape(predict, real_val, 0.0)
            rmse = masked_rmse(predict, real_val, 0.0)
        else:
            mape = masked_mape(predict, real_val_inv, 0.0)
            rmse = masked_rmse(predict, real_val_inv, 0.0)
        return main_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)
            if kwargs.get('_max') is not None:
                predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                      kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1),
                                       kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            valid_loss.append(loss); valid_mape.append(mape); valid_rmse.append(rmse)
        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs, y_list = [], []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds); y_list.append(testy)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]
        if kwargs.get('_max') is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(min(12, yhat.shape[-1])):
            pred = yhat[:, :, i]; real = realy[:, :, i]
            metrics = metric(pred, real)
            log = 'Horizon {:d}: MAE={:.4f} RMSE={:.4f} MAPE={:.4f}'
            print(log.format(i+1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0]); amape.append(metrics[1]); armse.append(metrics[2])
        print('Average: MAE={:.2f} RMSE={:.2f} MAPE={:.2f}%'.format(
            np.mean(amae), np.mean(armse), np.mean(amape)*100))
        if save:
            save_model(model, save_path_resume)
