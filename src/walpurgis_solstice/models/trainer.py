import numpy as np
import torch
import torch.optim as optim
import sys, os
from sklearn.metrics import mean_absolute_error

from walpurgis_solstice.utils.train import data_reshaper, save_model
from .losses import huber_loss, spectral_smoothness_penalty, masked_mae, masked_rmse, masked_mape, metric

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:trainer:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:trainer:{tag}] {val}", file=sys.stderr)


def _mixup_batch(x, y, alpha=0.2):
    """solstice新增: Mixup数据增强 — 线性插值两个样本, alpha控制Beta分布形状"""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # 保证主样本权重 >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


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
        # solstice: cosine CL增长
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']

        # upstream: Adam + MultiStepLR
        # solstice: RAdam + CosineAnnealingWarmRestarts (周期性重启学习率)
        self.optimizer = optim.RAdam(self.model.parameters(), lr=self.lrate,
                                     weight_decay=self.wdecay, eps=self.eps)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2,
            eta_min=1e-6) if self.if_lr_scheduler else None

        # upstream: masked_mae
        # solstice: huber_loss + spectral_smoothness_penalty
        self.loss = huber_loss
        self.clip = 5.0
        # solstice: mixup alpha参数
        self.mixup_alpha = 0.2

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
                # solstice: cosine CL增长 cl_len = 1 + int((1 - cos(pi * steps / total)) / 2 * out_len)
                elapsed = _ - self.warm_steps
                if self.cl_steps > 0:
                    progress = min(elapsed / (self.cl_steps * self.output_seq_len), 1.0)
                    self.cl_len = 1 + int((1 - math.cos(math.pi * progress)) / 2 * (self.output_seq_len - 1))
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

        # solstice: Mixup数据增强 (在训练阶段对输入和目标做线性插值)
        input_mixed, real_val_mixed = _mixup_batch(input, real_val, self.mixup_alpha)

        output = self.model(input_mixed)
        output = output.transpose(1, 2)

        # solstice: cosine CL增长
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
                progress = min(elapsed / (self.cl_steps * self.output_seq_len), 1.0)
                self.cl_len = 1 + int((1 - math.cos(math.pi * progress)) / 2 * (self.output_seq_len - 1))
                self.cl_len = min(self.cl_len, self.output_seq_len)

        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val_inv = self.scaler(real_val_mixed.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            main_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val_mixed[:, :, :, 0])
            main_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)

        # solstice: spectral smoothness penalty (频域平滑惩罚)
        sp = spectral_smoothness_penalty(predict)
        loss = main_loss + sp
        loss.backward()

        # solstice: 固定梯度裁剪 (不用aurora的自适应)
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip).item()

        self.optimizer.step()

        _adbg("step", f"loss={loss.item():.6f} huber={main_loss.item():.6f} sp={sp.item():.6f} "
               f"grad={total_norm:.4f} clip={self.clip:.2f} cl_len={self.cl_len} "
               f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

        if kwargs.get('_max') is not None:
            mape = masked_mape(predict, real_val_inv, 0.0)
            rmse = masked_rmse(predict, real_val_inv, 0.0)
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
