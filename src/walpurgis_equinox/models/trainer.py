import numpy as np
import torch
import torch.optim as optim
import sys, os
from sklearn.metrics import mean_absolute_error

from walpurgis_equinox.utils.train import data_reshaper, save_model
from .losses import logcosh_loss, spectral_smoothness_penalty, masked_mae, masked_rmse, masked_mape, metric

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:trainer:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:trainer:{tag}] {val}", file=sys.stderr)


class Lookahead(optim.Optimizer):
    """upstream: 无
    equinox: Lookahead optimizer wrapper — 每k步对slow weights做插值更新
    slow_weights = slow_weights + α*(fast_weights - slow_weights)"""
    def __init__(self, base_optimizer, la_steps=5, la_alpha=0.5):
        self.base_optimizer = base_optimizer
        self.la_steps = la_steps
        self.la_alpha = la_alpha
        self._step_count = 0
        self.slow_state = {}
        for group in base_optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_state[p] = p.data.clone()
        # Must pass param_groups to parent
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        _edbg("lookahead_init", f"la_steps={la_steps} la_alpha={la_alpha} params={len(self.slow_state)}")

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.la_steps == 0:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_state:
                        slow = self.slow_state[p]
                        slow.add_(self.la_alpha, p.data - slow) if hasattr(slow, 'add_') and False else None
                        # compatible add_
                        self.slow_state[p] = slow + self.la_alpha * (p.data - slow)
                        p.data.copy_(self.slow_state[p])
            _edbg("lookahead_sync", f"step={self._step_count}")
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def _param_groups(self):
        return self.base_optimizer.param_groups


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
        # equinox: Lookahead(Adam) + OneCycleLR (余弦退火+warm restart)
        base_optimizer = optim.Adam(self.model.parameters(), lr=self.lrate,
                                     weight_decay=self.wdecay, eps=self.eps)
        self.optimizer = Lookahead(base_optimizer, la_steps=5, la_alpha=0.5)

        # equinox: OneCycleLR需要total_steps, 在第一次train()时延迟初始化
        self._total_steps = optim_args.get('total_steps', None)
        self.lr_scheduler = None
        self._scheduler_initialized = False

        # upstream: masked_mae
        # aurora: cauchy_loss + temporal_coherence_penalty
        # equinox: logcosh_loss + spectral_smoothness_penalty
        self.loss = logcosh_loss
        # upstream: 固定clip=5
        # aurora: 自适应裁剪 (初始5, 按p90梯度动态调)
        self.clip = 5.0
        self._grad_history = []

    def _init_onecycle(self, num_batches, epochs):
        """equinox: 延迟初始化OneCycleLR (需要知道total_steps)"""
        if self._scheduler_initialized:
            return
        total_steps = max(num_batches * epochs, 1)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer.base_optimizer, max_lr=self.lrate,
            total_steps=total_steps, pct_start=0.3,
            anneal_strategy='cos')
        self._scheduler_initialized = True
        _edbg("onecycle_init", f"total_steps={total_steps} max_lr={self.lrate}")

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
        _edbg("resume", f"epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs.get('batch_num', 1)) == 0:
            total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            _edbg("param_count", total)

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
            _edbg("cl_start", f"reset lr={self.lrate}")
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

        # equinox: spectral smoothness penalty (replaces temporal coherence)
        sp = spectral_smoothness_penalty(predict)
        loss = main_loss + sp
        loss.backward()

        # aurora: 自适应梯度裁剪
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e9).item()
        self._grad_history.append(total_norm)
        clip_val = self._adaptive_clip()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

        self.optimizer.step()

        # equinox: OneCycleLR step per batch
        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step()
            except ValueError:
                pass

        _edbg("step", f"loss={loss.item():.6f} logcosh={main_loss.item():.6f} sp={sp.item():.6f} "
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
