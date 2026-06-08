"""
Trainer — Umbra变体
算法改动:
  - Adaptive Huber Loss: 可学习delta参数, 训练中自适应调节
  - LAMB优化器: Layer-wise Adaptive Moments, 大batch友好
    核心: 对每层独立计算trust ratio = ||w|| / ||Adam_update||
    更新: w -= lr * trust_ratio * Adam_update
  - Polynomial Decay LR: lr = base_lr * (1 - t/T)^power
    配合warmup: 先线性升温, 再多项式衰减
  - 梯度裁剪自适应: 根据梯度历史动态调整clip值
  - 完整诊断: 训练/验证的逐batch统计

"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from ..utils.train import data_reshaper, save_model
from .losses import (masked_mae, masked_rmse, masked_mape,
                     metric, adaptive_huber_loss)
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


class LAMB(optim.Optimizer):
    """LAMB (Layer-wise Adaptive Moments) 优化器
    参考: You et al., "Large Batch Optimization for
    Deep Learning: Training BERT in 76 minutes"

    核心算法:
      1. Adam moment更新: m_t, v_t
      2. 计算Adam方向: r_t = m_t / (sqrt(v_t) + eps) + wd * w
      3. Trust ratio: φ = ||w|| / ||r_t||
      4. 更新: w -= lr * φ * r_t
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'LAMB does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data)

                exp_avg, exp_avg_sq = (state['exp_avg'],
                                       state['exp_avg_sq'])
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Adam moment更新
                exp_avg.mul_(beta1).add_(
                    grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Adam方向 + weight decay
                adam_step = (m_hat
                             / (v_hat.sqrt()
                                + group['eps']))
                if group['weight_decay'] > 0:
                    adam_step.add_(
                        p.data, alpha=group['weight_decay'])

                # Trust ratio: ||w|| / ||adam_step||
                w_norm = p.data.norm(2).clamp(min=1e-8)
                adam_norm = adam_step.norm(2).clamp(min=1e-8)
                trust_ratio = w_norm / adam_norm

                # 对1D参数(bias/norm)跳过trust ratio
                if p.data.ndim <= 1:
                    trust_ratio = 1.0

                p.data.add_(
                    adam_step,
                    alpha=-group['lr'] * trust_ratio)

        return loss


class PolynomialDecayLR:
    """多项式衰减学习率调度
    lr = base_lr * (1 - t/T)^power
    前warmup_steps用线性升温
    """

    def __init__(self, optimizer, total_steps,
                 warmup_steps=0, power=2.0,
                 min_lr=1e-7):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.power = power
        self.min_lr = min_lr
        self.base_lrs = [g['lr']
                         for g in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        self._step_count += 1
        # 动态同步base_lrs以适配后续add_param_group
        while len(self.base_lrs) < len(
                self.optimizer.param_groups):
            self.base_lrs.append(
                self.optimizer.param_groups[
                    len(self.base_lrs)]['lr'])
        for i, group in enumerate(
                self.optimizer.param_groups):
            if self._step_count <= self.warmup_steps:
                # 线性升温
                lr = (self.base_lrs[i]
                      * self._step_count
                      / max(self.warmup_steps, 1))
            else:
                # 多项式衰减
                decay_steps = (self.total_steps
                               - self.warmup_steps)
                current = (self._step_count
                           - self.warmup_steps)
                progress = min(
                    float(current) / max(decay_steps, 1),
                    1.0)
                lr = (self.base_lrs[i]
                      * (1.0 - progress) ** self.power)
                lr = max(lr, self.min_lr)
            group['lr'] = lr

    def get_last_lr(self):
        return [g['lr']
                for g in self.optimizer.param_groups]


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

        # LAMB优化器: 替代AdamW
        self.optimizer = LAMB(
            self.model.parameters(),
            lr=self.lrate,
            weight_decay=self.wdecay,
            eps=self.eps,
            betas=(0.9, 0.999))

        # Polynomial Decay LR: 替代OneCycleLR
        total_steps = optim_args.get('epochs', 3) * max(
            optim_args.get('total_batches', 25), 1)
        warmup_frac = 0.1  # 10%步数用于warmup
        self.lr_scheduler = PolynomialDecayLR(
            self.optimizer,
            total_steps=max(total_steps, 10),
            warmup_steps=int(total_steps * warmup_frac),
            power=2.0,
            min_lr=1e-7
        ) if self.if_lr_scheduler else None

        # Adaptive Huber Loss: 可学习delta
        # delta通过sigmoid映射: δ = δ_min + (δ_max - δ_min) * σ(raw)
        self._huber_delta_raw = torch.nn.Parameter(
            torch.tensor(1.5))  # 初始delta≈1.3
        # 把huber delta也加到优化器
        self.optimizer.add_param_group({
            'params': [self._huber_delta_raw],
            'lr': self.lrate * 0.1,  # delta学得慢一点
            'weight_decay': 0.0
        })
        self._delta_min = 0.1
        self._delta_max = 5.0

        self.loss = adaptive_huber_loss
        self.loss_mae = masked_mae  # 用于评估
        # 自适应梯度裁剪
        self.clip = 5
        self._grad_history = []

    @property
    def huber_delta(self):
        """可学习的Huber delta: sigmoid映射到[δ_min, δ_max]"""
        return (self._delta_min
                + (self._delta_max - self._delta_min)
                * torch.sigmoid(self._huber_delta_raw))

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

        # 获取当前的自适应delta
        delta = self.huber_delta

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
                delta=delta)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_t = self.scaler.inverse_transform(
                real_val[:, :, :, 0])
            loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :],
                0, delta=delta)

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
            _dbg("train.loss", f"{loss.item():.6f}")
            _dbg("train.grad_norm",
                 f"{total_norm.item():.4f}")
            _dbg("train.adaptive_clip",
                 f"{self.clip:.2f}")
            _dbg("train.huber_delta",
                 f"{delta.item():.4f}")
            _dbg("train.lamb_trust_ratio",
                 "LAMB active", "optimizer")

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
