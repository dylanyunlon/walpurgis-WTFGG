import numpy as np
import torch
import torch.optim as optim
import sys, os
from collections import OrderedDict

from walpurgis_equinox.utils.train import data_reshaper, save_model
from .losses import logcosh_loss, cutmix_spatiotemporal, masked_mae, masked_rmse, masked_mape, metric

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:trainer:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:trainer:{tag}] {val}", file=sys.stderr)


class Lookahead(optim.Optimizer):
    """equinox: Lookahead包装器 — 慢权重平滑加速收敛
    k步快权重更新后, 慢权重向快权重方向插值一步
    减少训练震荡, 提升泛化能力"""
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        # 缓存慢权重
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            slow = []
            for p in group['params']:
                slow.append(p.data.clone())
            self.slow_params.append(slow)
        # 需要暴露param_groups以兼容lr_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for g_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    slow = self.slow_params[g_idx][p_idx]
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)
            _edbg("lookahead_sync", f"step={self._step_count} k={self.k} alpha={self.alpha}")
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']

        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        self.if_lr_scheduler = optim_args.get('lr_schedule', True)
        self.if_cl = optim_args.get('if_cl', True)
        self.cl_steps = optim_args.get('cl_steps', 3)
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args.get('warm_steps', 30)

        # upstream: Adam + MultiStepLR
        # equinox: Lookahead(Adam, k, alpha) + OneCycleLR
        lookahead_k = optim_args.get('lookahead_k', 5)
        lookahead_alpha = optim_args.get('lookahead_alpha', 0.5)
        base_adam = optim.Adam(self.model.parameters(), lr=self.lrate,
                               weight_decay=self.wdecay, eps=self.eps)
        self.optimizer = Lookahead(base_adam, k=lookahead_k, alpha=lookahead_alpha)

        # equinox: OneCycleLR — 余弦退火
        total_steps = optim_args.get('total_steps', 500)
        max_lr = optim_args.get('max_lr', self.lrate * 2)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer.base_optimizer,
            max_lr=max_lr,
            total_steps=max(total_steps, 10),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        ) if self.if_lr_scheduler else None

        # upstream: masked_mae
        # equinox: LogCosh loss
        self.loss = logcosh_loss
        self.clip = 5.0
        # equinox: CutMix增强开关
        self.cutmix_alpha = optim_args.get('cutmix_alpha', 0.5)

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
                elapsed = _ - self.warm_steps
                if self.cl_steps > 0:
                    self.cl_len = min(int(elapsed / self.cl_steps) + 1, self.output_seq_len)
        _edbg("resume", f"epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs.get('batch_num', 1)) == 0:
            total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            _edbg("param_count", total)

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        # equinox: CutMix时空数据增强
        if self.training_cutmix_enabled:
            input_aug, _ = cutmix_spatiotemporal(
                input[:, :, :, 0], real_val[:, :, :, 0] if real_val.dim() == 4 else real_val,
                alpha=self.cutmix_alpha)
            input = input.clone()
            input[:, :, :, 0] = input_aug

        output = self.model(input)
        output = output.transpose(1, 2)

        # CL调度
        bn = kwargs.get('batch_num', 0)
        if bn < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            _edbg("cl_start", f"reset lr={self.lrate}")
        else:
            elapsed = bn - self.warm_steps
            if self.cl_steps > 0 and self.if_cl:
                self.cl_len = min(int(elapsed / self.cl_steps) + 1, self.output_seq_len)

        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1, 2).unsqueeze(-1),
                                  kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            loss = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:, :, :, 0])
            loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # equinox: OneCycleLR每步更新
        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step()
            except ValueError:
                pass  # OneCycleLR超出total_steps时忽略

        _edbg("step", f"loss={loss.item():.6f} cl_len={self.cl_len} "
               f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

        if kwargs.get('_max') is not None:
            mape = masked_mape(predict, real_val, 0.0)
            rmse = masked_rmse(predict, real_val, 0.0)
        else:
            mape = masked_mape(predict, real_val_inv, 0.0)
            rmse = masked_rmse(predict, real_val_inv, 0.0)
        return loss.item(), mape.item(), rmse.item()

    @property
    def training_cutmix_enabled(self):
        return self.cutmix_alpha > 0 and self.model.training

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
