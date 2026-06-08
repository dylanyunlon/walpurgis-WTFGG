"""
Aphelion trainer — 算法改写 #10:
  upstream: Adam + MultiStepLR
  corona: RAdam + CosineAnnealingWarmRestarts
  aphelion: Sophia-style optimizer + ExponentialLR scheduler —
            Sophia用Hessian对角近似做二阶自适应学习率,
            相比Adam的一阶momentum+RMSprop, Sophia对损失曲面曲率敏感,
            能在平坦方向用更大步长、陡峭方向用更小步长。
            这里实现简化版Sophia: 用Gauss-Newton-Bartlett估计Hessian对角,
            配合exponential decay scheduler实现平滑的学习率衰减。
  upstream loss: masked_mae
  aphelion loss: tilted_erm_loss (TERM)
"""
import numpy as np
import torch
import torch.optim as optim
from ..utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, tilted_erm_loss
from .. import _dbg, PerfTimer


class SophiaG(optim.Optimizer):
    """Aphelion特有: 简化版Sophia-G优化器
    用Gauss-Newton-Bartlett (GNB) 估计器近似Hessian对角元素,
    然后用 param -= lr * grad / max(hessian_diag, rho) 更新参数。
    rho是裁剪阈值, 防止Hessian估计过小导致步长爆炸。

    与标准Adam的区别:
    - Adam用一阶动量(grad EMA)和二阶动量(grad^2 EMA)
    - Sophia用一阶动量和Hessian对角估计(每隔k步更新一次)
    - Sophia的步长上界被rho显式控制, 比Adam的epsilon更精确
    """
    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99),
                 rho=0.04, weight_decay=1e-4, eps=1e-15):
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, hessian_estimates=None):
        """
        hessian_estimates: 可选的dict, key是param id, value是Hessian对角估计
        如果不提供, 退化为带momentum的SGD (仍然比Adam在某些场景下更稳定)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            rho = group['rho']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SophiaG不支持稀疏梯度")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)      # 一阶动量
                    state['hessian_diag'] = torch.zeros_like(p)  # Hessian对角EMA

                exp_avg = state['exp_avg']
                hess = state['hessian_diag']
                state['step'] += 1

                # 权重衰减 (decoupled, 与AdamW相同)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # 更新一阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新Hessian对角估计 (如果提供了外部估计)
                pid = id(p)
                if hessian_estimates is not None and pid in hessian_estimates:
                    h_est = hessian_estimates[pid]
                    hess.mul_(beta2).add_(h_est, alpha=1 - beta2)

                # Sophia更新: param -= lr * m / max(h, rho)
                # 裁剪: 分母至少为rho, 防止步长过大
                denom = torch.clamp(hess, min=rho)
                p.addcdiv_(exp_avg, denom, value=-lr)

        return loss

    def estimate_hessian_gnb(self, model, loss_fn, data, target):
        """Gauss-Newton-Bartlett Hessian对角估计
        用mini-batch的梯度外积近似Fisher信息矩阵 ≈ Hessian
        这是每隔k步调用一次的昂贵操作
        """
        model.zero_grad()
        output = model(data)
        # 对每个sample采样一个loss (GNB需要per-sample gradient)
        loss = loss_fn(output, target)
        loss.backward()
        hessian_estimates = {}
        for p in model.parameters():
            if p.grad is not None:
                # GNB估计: h_i ≈ g_i^2 (梯度平方作为Fisher对角的近似)
                hessian_estimates[id(p)] = p.grad.detach() ** 2
        return hessian_estimates


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

        # Aphelion改写: Sophia优化器替代Adam/RAdam
        sophia_rho = optim_args.get('sophia_rho', 0.04)
        sophia_betas = tuple(optim_args.get('sophia_betas', [0.965, 0.99]))
        self.optimizer = SophiaG(
            self.model.parameters(), lr=self.lrate,
            betas=sophia_betas, rho=sophia_rho,
            weight_decay=self.wdecay, eps=self.eps)

        # Aphelion改写: ExponentialLR替代MultiStepLR/CosineAnnealing
        if self.if_lr_scheduler:
            exp_gamma = optim_args.get('exp_decay_gamma', 0.97)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=exp_gamma)
        else:
            self.lr_scheduler = None

        # Aphelion改写: TERM loss的可学习tilt参数
        self.tilt_param = torch.nn.Parameter(
            torch.tensor(optim_args.get('tilt_init', 1.0)))
        # tilt参数也需要优化, 加入一个独立的小学习率优化器
        self.tilt_optimizer = optim.Adam([self.tilt_param], lr=1e-3)

        # Hessian估计间隔 (每hessian_interval步估计一次, 减少计算开销)
        self.hessian_interval = optim_args.get('hessian_interval', 10)
        self._hessian_estimates = None
        self._global_step = 0

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
        self.tilt_optimizer.zero_grad()
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
            # Aphelion: TERM loss
            loss = tilted_erm_loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :],
                                   self.tilt_param)
        else:
            predict = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(real_val[:, :, :, 0])
            # Aphelion: TERM loss
            loss = tilted_erm_loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :],
                                   self.tilt_param, null_val=0)

        self.perf.start("backward")
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Sophia: 周期性估计Hessian对角
        self._global_step += 1
        if self._global_step % self.hessian_interval == 0:
            self.optimizer.step(hessian_estimates=self._hessian_estimates)
            # 下次step前估计新的Hessian (延迟到下一个interval)
            # 这里用当前梯度的平方作为简单估计
            self._hessian_estimates = {}
            for p in self.model.parameters():
                if p.grad is not None:
                    self._hessian_estimates[id(p)] = p.grad.detach() ** 2
        else:
            self.optimizer.step(hessian_estimates=self._hessian_estimates)

        # 更新tilt参数
        self.tilt_optimizer.step()

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
