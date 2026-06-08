"""
Trainer — Transit变体 (M055)
算法改动 #10:
  - Lion优化器: EvoLved Sign Momentum, 只用sign(momentum)更新
    内存比Adam少1个状态, 更适合大batch+权重衰减
    update = sign(β1·m + (1-β1)·grad), m = β2·m + (1-β2)·grad
  - Warmup-Stable-Decay (WSD) Schedule:
    Phase 1 (warmup): 线性从0升到lr_max (占total的10%)
    Phase 2 (stable): 保持lr_max (占total的70%)
    Phase 3 (decay): 余弦衰减到lr_min (占total的20%)
    比OneCycleLR更稳定, 适合Lion的sign-based更新
  - Tweedie Loss替代Log-Cosh
  - 梯度裁剪: Lion用更小的clip值(因为sign更新幅度固定)
"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from ..utils.train import data_reshaper, save_model
from .losses import (masked_mae, masked_rmse, masked_mape,
                     metric, tweedie_loss)
from .. import (_dbg, _is_debug, gradient_health_check,
                _STEP_COUNTER)
import math


# ─── Lion Optimizer ──────────────────────────────────────
class Lion(optim.Optimizer):
    """Lion优化器 (EvoLved Sign Momentum)
    Chen et al. 2023: "Symbolic Discovery of Optimization Algorithms"
    核心: update = sign(interp(grad, momentum))
    比Adam内存少33%(只需1个状态vs 2个), 更新方向用sign

    Args:
        lr: 学习率 (Lion通常用Adam的1/3~1/10)
        betas: (β1, β2), β1=interp系数, β2=momentum系数
        weight_decay: 权重衰减(解耦, 在更新前应用)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

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
                        "Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # 解耦权重衰减: 先衰减再更新
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr']
                           * group['weight_decay'])

                # 更新方向: sign(β1·m + (1-β1)·grad)
                update = exp_avg.mul(beta1).add(
                    grad, alpha=1 - beta1)
                p.add_(torch.sign(update),
                       alpha=-group['lr'])

                # 更新momentum: m = β2·m + (1-β2)·grad
                exp_avg.mul_(beta2).add_(
                    grad, alpha=1 - beta2)

        return loss


# ─── WSD Schedule ────────────────────────────────────────
class WarmupStableDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup-Stable-Decay学习率调度
    Phase 1 (warmup_frac): 线性 0 → base_lr
    Phase 2 (stable_frac): 常数 base_lr
    Phase 3 (decay_frac):  余弦 base_lr → min_lr
    """
    def __init__(self, optimizer, total_steps,
                 warmup_frac=0.1, stable_frac=0.7,
                 min_lr_ratio=0.01, last_epoch=-1):
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = int(
            self.total_steps * warmup_frac)
        self.stable_steps = int(
            self.total_steps * stable_frac)
        self.decay_steps = (self.total_steps
                            - self.warmup_steps
                            - self.stable_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Phase 1: linear warmup
            frac = step / max(self.warmup_steps, 1)
            return [base_lr * frac
                    for base_lr in self.base_lrs]
        elif step < self.warmup_steps + self.stable_steps:
            # Phase 2: stable
            return list(self.base_lrs)
        else:
            # Phase 3: cosine decay
            decay_step = (step - self.warmup_steps
                          - self.stable_steps)
            frac = decay_step / max(self.decay_steps, 1)
            frac = min(frac, 1.0)
            cos_factor = (1 + math.cos(math.pi * frac)) / 2
            return [
                base_lr * (self.min_lr_ratio
                           + (1 - self.min_lr_ratio)
                           * cos_factor)
                for base_lr in self.base_lrs]


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

        # Lion优化器: lr通常用Adam的1/10 ~ 1/3
        # β1=0.9, β2=0.99 (Lion原论文推荐)
        lion_lr = self.lrate * 0.3  # Lion建议更小lr
        self.optimizer = Lion(
            self.model.parameters(),
            lr=lion_lr,
            betas=(0.9, 0.99),
            weight_decay=self.wdecay)
        _dbg("lion.lr", f"{lion_lr:.6f}", "trainer")
        _dbg("lion.betas", "(0.9, 0.99)", "trainer")

        # WSD Schedule: warmup 10%, stable 70%, decay 20%
        total_steps = optim_args.get('epochs', 3) * max(
            optim_args.get('total_batches', 25), 1)
        self.lr_scheduler = WarmupStableDecayLR(
            self.optimizer,
            total_steps=max(total_steps, 10),
            warmup_frac=0.1,
            stable_frac=0.7,
            min_lr_ratio=0.01
        ) if self.if_lr_scheduler else None

        # Tweedie Loss: 可学习power参数
        self._tweedie_power = 1.5
        self.loss = tweedie_loss
        self.loss_mae = masked_mae  # 用于评估
        # Lion用更小的clip (sign更新幅度固定)
        self.clip = 1.0
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
                    param_group["lr"] = self.lrate * 0.3
            else:
                if ((_ - self.warm_steps) % self.cl_steps == 0
                        and self.cl_len < self.output_seq_len):
                    self.cl_len += int(self.if_cl)
        print(f"resume from epoch{epoch_num}, "
              f"lr={self.lrate*0.3}, cl_len={self.cl_len}")

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
                param_group["lr"] = self.lrate * 0.3
            print(f"======== Start curriculum learning... "
                  f"lr={self.lrate*0.3} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps)
                    % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)

        # scale + Tweedie loss
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
                power=self._tweedie_power)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_t = self.scaler.inverse_transform(
                real_val[:, :, :, 0])
            loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_t[:, :self.cl_len, :],
                0, power=self._tweedie_power)

        loss.backward()

        # 梯度裁剪 (Lion用更小值因为sign更新)
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip)
        self._grad_history.append(total_norm.item())
        if len(self._grad_history) > 100:
            self._grad_history = self._grad_history[-50:]
            adaptive_clip = np.percentile(
                self._grad_history, 90)
            self.clip = max(min(adaptive_clip, 5.0), 0.5)

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
            current_lr = self.optimizer.param_groups[0]['lr']
            _dbg("train.lr", f"{current_lr:.6f}")
            # WSD phase诊断
            if self.lr_scheduler is not None:
                sched = self.lr_scheduler
                step = sched.last_epoch
                if step < sched.warmup_steps:
                    phase = "warmup"
                elif step < (sched.warmup_steps
                             + sched.stable_steps):
                    phase = "stable"
                else:
                    phase = "decay"
                _dbg("train.wsd_phase", phase)

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
