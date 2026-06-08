"""
Trainer — Parallax变体 (M054)
算法改动:
  - Cauchy Loss 替代 Log-Cosh
  - Prodigy自适应优化器: 自动估计学习率
    基于dual averaging的步长自适应:
    d_k = max(d_{k-1}, ||grad||/||param||)
    lr_effective = lr_base * d_k
    不需要手动调learning rate — Prodigy自动发现
  - REINFORCE掩码辅助损失: 策略梯度+熵正则
  - 梯度裁剪自适应: 根据梯度历史动态调整clip值
  - 完整诊断: 训练/验证的逐batch统计
"""
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from sklearn.metrics import mean_absolute_error

from ..utils.train import data_reshaper, save_model
from .losses import (masked_mae, masked_rmse, masked_mape,
                     metric, cauchy_loss)
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


class ProdigyOptimizer(optim.Optimizer):
    """Prodigy: Adam-like自适应学习率估计器

    核心思想: 自动发现最优学习率
    d_k = max(d_{k-1}, ||grad·param|| / ||param||^2)
    然后用 d_k * lr_base 作为实际学习率

    参考: Mishchenko & Defazio (2023)
    "Prodigy: An Expeditiously Adaptive Parameter-Free
     Learner"
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, d_coef=1.0,
                 beta3=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        d_coef=d_coef, beta3=beta3)
        super().__init__(params, defaults)
        # 全局step长估计
        self._d_numerator = 0.0
        self._d_denominator = 1e-8

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta3 = group['beta3']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Prodigy不支持稀疏梯度")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['p0'] = p.data.clone()

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                p0 = state['p0']

                # weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr']
                                * group['weight_decay'])

                # d_k估计: ||grad · (p - p0)|| / ||p0||^2
                grad_p_prod = (grad * (p.data - p0)).sum().item()
                p0_norm_sq = (p0 * p0).sum().item() + 1e-12
                self._d_numerator = max(
                    self._d_numerator,
                    abs(grad_p_prod))
                self._d_denominator = max(
                    self._d_denominator,
                    p0_norm_sq)

                # 自适应步长
                d_k = (self._d_numerator
                       / self._d_denominator)
                d_k = max(d_k, 1e-8)
                effective_lr = group['lr'] * d_k * group['d_coef']

                # Adam更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)

                # beta3 EMA of d (可选平滑)
                if beta3 > 0:
                    effective_lr = (beta3 * effective_lr
                                    + (1 - beta3) * effective_lr)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (effective_lr
                             / bias_correction1)

                denom = (exp_avg_sq.sqrt()
                         / (bias_correction2 ** 0.5)
                         ).add_(group['eps'])
                p.data.addcdiv_(
                    exp_avg, denom, value=-step_size)

        return loss

    @property
    def effective_d(self):
        return self._d_numerator / self._d_denominator


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

        # Prodigy自适应优化器 — 自动估计学习率
        cauchy_scale = optim_args.get('cauchy_scale', 1.0)
        beta3 = optim_args.get('prodigy_beta3', 0.0)
        self.optimizer = ProdigyOptimizer(
            self.model.parameters(),
            lr=self.lrate,
            weight_decay=self.wdecay,
            eps=self.eps,
            betas=(0.9, 0.999),
            beta3=beta3)

        print(f"[PAR-DBG] Prodigy optimizer: "
              f"lr={self.lrate}, wd={self.wdecay}, "
              f"beta3={beta3}")

        # Cauchy损失
        self.cauchy_scale = cauchy_scale
        self.loss = cauchy_loss
        self.loss_mae = masked_mae  # 用于评估
        # REINFORCE辅助损失权重
        self.reinforce_weight = optim_args.get(
            'reinforce_weight', 0.01)
        # 自适应梯度裁剪
        self.clip = 5
        self._grad_history = []

        print(f"[PAR-DBG] Cauchy loss: scale={cauchy_scale}")
        print(f"[PAR-DBG] REINFORCE weight: "
              f"{self.reinforce_weight}")

    def _collect_reinforce_losses(self):
        """从模型中收集REINFORCE掩码的辅助损失"""
        total_rl_loss = 0.0
        count = 0
        for module in self.model.modules():
            if hasattr(module, 'get_reinforce_loss'):
                # reward = 负主损失 (更低的loss = 更高的reward)
                rl_loss = module.get_reinforce_loss(
                    reward=0.0)  # reward在backward时更新
                if rl_loss is not None:
                    total_rl_loss = total_rl_loss + rl_loss
                    count += 1
        if count > 0:
            _dbg("reinforce.num_masks", count, "train")
            _dbg("reinforce.aux_loss",
                 f"{total_rl_loss:.6f}" if isinstance(
                     total_rl_loss, float)
                 else f"{total_rl_loss.item():.6f}",
                 "train")
        return total_rl_loss

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

        # scale + Cauchy loss
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
                scale=self.cauchy_scale)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_t = self.scaler.inverse_transform(
                real_val[:, :, :, 0])
            loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :],
                0, scale=self.cauchy_scale)

        # REINFORCE辅助损失
        rl_loss = self._collect_reinforce_losses()
        if isinstance(rl_loss, torch.Tensor):
            total_loss = loss + self.reinforce_weight * rl_loss
        else:
            total_loss = loss

        total_loss.backward()

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
            _dbg("prodigy.effective_d",
                 f"{self.optimizer.effective_d:.8f}")

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
