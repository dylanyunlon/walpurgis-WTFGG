"""
Trainer — Perihelion变体
算法改动:
  - Linex非对称损失 替代 MAE
  - AdaFactor替代AdamW: 分解二阶矩(row/col因子), 显存更省
    自带自适应学习率, 不需要外部存储动量
  - Inverse Square Root Schedule: lr = lr_init * sqrt(warm) / sqrt(step)
    warm-up阶段线性增长, 之后按1/sqrt(t)衰减
    灵感来自Transformer原始论文的调度策略
  - 梯度裁剪自适应: 根据梯度历史动态调整clip值
  - 完整诊断: 训练/验证的逐batch统计
"""
import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from ..utils.train import data_reshaper, save_model
from .losses import (masked_mae, masked_rmse, masked_mape,
                     metric, linex_loss)
from .. import (_dbg, _is_debug, gradient_health_check,
                _STEP_COUNTER)


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(
            np.subtract(y_pred, y_true).astype('float32'),
            y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class AdaFactorOptimizer(optim.Optimizer):
    """AdaFactor: 分解二阶矩估计的自适应优化器
    核心思想: 对于2D参数(weight矩阵), 不存完整二阶矩
    而是分解为行因子R和列因子C, 显存从O(mn)降到O(m+n)
    对于1D参数(bias), 退化为标准Adam-like
    """

    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3),
                 clip_threshold=1.0, decay_rate=-0.8,
                 beta1=None, weight_decay=0.0,
                 relative_step=False):
        defaults = dict(
            lr=lr, eps=eps, clip_threshold=clip_threshold,
            decay_rate=decay_rate, beta1=beta1,
            weight_decay=weight_decay,
            relative_step=relative_step)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(
            dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _get_rho(self, group, step):
        """二阶矩的衰减率: rho = min(rho_max, 1 - step^decay_rate)"""
        decay = group['decay_rate']
        rho = min(1.0, 1.0 - math.pow(step, decay))
        return max(rho, 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaFactor不支持稀疏梯度")

                state = self.state[p]
                grad_shape = grad.shape
                factored = len(grad_shape) >= 2

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    if factored:
                        state['exp_avg_sq_row'] = \
                            torch.zeros(
                                grad_shape[:-1],
                                device=grad.device)
                        state['exp_avg_sq_col'] = \
                            torch.zeros(
                                grad_shape[:-2] + grad_shape[-1:],
                                device=grad.device)
                    else:
                        state['exp_avg_sq'] = \
                            torch.zeros_like(grad)
                    if group['beta1'] is not None:
                        state['exp_avg'] = \
                            torch.zeros_like(grad)

                state['step'] += 1
                rho = self._get_rho(group, state['step'])

                # 更新二阶矩
                if factored:
                    exp_r = state['exp_avg_sq_row']
                    exp_c = state['exp_avg_sq_col']
                    exp_r.mul_(rho).add_(
                        grad.pow(2).mean(dim=-1),
                        alpha=1.0 - rho)
                    exp_c.mul_(rho).add_(
                        grad.pow(2).mean(dim=-2),
                        alpha=1.0 - rho)
                    update = self._approx_sq_grad(
                        exp_r, exp_c)
                    update.mul_(grad)
                else:
                    exp_sq = state['exp_avg_sq']
                    exp_sq.mul_(rho).add_(
                        grad.pow(2), alpha=1.0 - rho)
                    update = exp_sq.rsqrt().mul_(grad)

                # RMS裁剪
                update_rms = self._rms(update)
                threshold = group['clip_threshold']
                if update_rms > threshold:
                    update.mul_(
                        threshold / max(update_rms, 1e-10))

                # 动量(可选)
                if group['beta1'] is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(
                        update, alpha=1 - group['beta1'])
                    update = exp_avg

                # 权重衰减
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay']
                           * group['lr'])

                p.add_(update, alpha=-group['lr'])

        return loss


class InverseSquareRootSchedule:
    """Inverse Square Root Schedule:
    warmup阶段: lr线性从0增到lr_init
    之后: lr = lr_init * sqrt(warmup_steps) / sqrt(step)
    """

    def __init__(self, optimizer, warmup_steps=500,
                 init_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.init_lr = init_lr
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            # 线性warmup
            lr = self.init_lr * self._step / self.warmup_steps
        else:
            # inverse sqrt衰减
            lr = self.init_lr * math.sqrt(
                self.warmup_steps) / math.sqrt(self._step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


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
        self.cl_len = (0 if self.if_cl
                       else self.output_seq_len)
        self.warm_steps = optim_args['warm_steps']

        # AdaFactor: 分解二阶矩, 显存友好
        self.optimizer = AdaFactorOptimizer(
            self.model.parameters(),
            lr=self.lrate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=0.9,
            weight_decay=self.wdecay)

        # Inverse Square Root Schedule
        warmup = optim_args.get('warm_epochs', 1) * max(
            optim_args.get('total_batches', 25), 1)
        self.lr_scheduler = InverseSquareRootSchedule(
            self.optimizer,
            warmup_steps=max(warmup, 10),
            init_lr=self.lrate
        ) if self.if_lr_scheduler else None

        # Linex非对称损失
        self.linex_a = 0.5  # a>0: 过预测惩罚更大
        self.linex_b = 1.0
        self.loss = linex_loss
        self.loss_mae = masked_mae  # 用于评估
        # 自适应梯度裁剪
        self.clip = 5
        self._grad_history = []

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
                if ((_ - self.warm_steps) % self.cl_steps == 0
                        and self.cl_len < self.output_seq_len):
                    self.cl_len += int(self.if_cl)
        print(f"resume from epoch{epoch_num}, "
              f"lr={self.lrate}, cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if (self.print_model_structure
                and int(kwargs['batch_num']) == 0):
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    tmp = 1
                    for s in param.shape:
                        tmp *= s
                    parameter_num += tmp
            print(f"Parameter size: {parameter_num}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)
        _STEP_COUNTER['batch'] = kwargs.get('batch_num', 0)

        output = self.model(input)
        output = output.transpose(1, 2)

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print(f"======== Start curriculum learning... "
                  f"lr={self.lrate} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps)
                    % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)

        # scale + loss
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_s[:, :self.cl_len, :],
                a=self.linex_a, b=self.linex_b)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_t = self.scaler.inverse_transform(
                real_val[:, :, :, 0])
            loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :],
                0, a=self.linex_a, b=self.linex_b)

        loss.backward()

        # 自适应梯度裁剪
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip)
        self._grad_history.append(total_norm.item())
        if len(self._grad_history) > 100:
            self._grad_history = self._grad_history[-50:]
            adaptive_clip = np.percentile(
                self._grad_history, 90)
            self.clip = max(adaptive_clip, 1.0)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 诊断
        if _is_debug():
            gradient_health_check(
                self.model,
                step=kwargs.get('batch_num', 0))
            current_lr = self.optimizer.param_groups[0]['lr']
            _dbg("train.loss", f"{loss.item():.6f}")
            _dbg("train.grad_norm",
                 f"{total_norm.item():.4f}")
            _dbg("train.adaptive_clip",
                 f"{self.clip:.2f}")
            _dbg("train.lr",
                 f"{current_lr:.8f}")

        # 用MAE做metrics以便和其他变体对比
        if kwargs['_max'] is not None:
            mape = masked_mape(predict, real_val_s, 0.0)
            rmse = masked_rmse(predict, real_val_s, 0.0)
            mae_val = self.loss_mae(predict, real_val_s, 0.0)
        else:
            mape = masked_mape(predict, real_val_t, 0.0)
            rmse = masked_rmse(predict, real_val_t, 0.0)
            mae_val = self.loss_mae(predict, real_val_t, 0.0)

        return mae_val.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(
                dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx)
            output = output.transpose(1, 2)

            if kwargs['_max'] is not None:
                predict = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(
                    testy[:, :, :, 0])

            loss = masked_mae(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        return (np.mean(valid_loss), np.mean(valid_mape),
                np.mean(valid_rmse))

    @staticmethod
    def test(model, save_path_resume, device, dataloader,
             scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
        y_list = []
        for itera, (x, y) in enumerate(
                dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)

        yhat = torch.cat(outputs, dim=0)[
            :realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[
            :realy.size(0), ...]
        assert torch.where(y_list == realy)

        if kwargs['_max'] is not None:
            realy = scaler(
                realy.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(
                yhat.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(
                realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if (kwargs.get('dataset_name') == 'PEMS04'
                    or kwargs.get('dataset_name') == 'PEMS08'):
                mae = mean_absolute_error(
                    pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae = metrics[0]
                mape = metrics[1]
                rmse = metrics[2]
            log = ('Horizon {:d}: MAE={:.4f} '
                   'RMSE={:.4f} MAPE={:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        log = ('(Avg 12h) MAE={:.2f} | '
               'RMSE={:.2f} | MAPE={:.2f}%')
        print(log.format(
            np.mean(amae), np.mean(armse),
            np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
