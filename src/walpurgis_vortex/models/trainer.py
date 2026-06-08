"""
Vortex trainer — 算法改写:
  1. CosineAnnealingWarmRestarts替代MultiStepLR
  2. 梯度噪声注入 (gradient noise injection) — 训练后期注入衰减噪声
  3. 每N步自动dump训练状态(loss/lr/grad_norm)
  4. 集成PerfTimer用于训练阶段计时
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis_vortex.utils.train import (
    data_reshaper, save_model)
from .losses import (
    masked_mae, masked_rmse, masked_mape, metric,
    huber_mae_adaptive)
from walpurgis_vortex import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    LRTracker)


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
        # adam
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        # curriculum learning
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = (0 if self.if_cl
                       else self.output_seq_len)
        # warmup
        self.warm_steps = optim_args['warm_steps']
        # Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)
        # CosineAnnealingWarmRestarts (Vortex特有)
        # T_0=重启周期(epochs), T_mult=周期倍增
        total_steps = optim_args.get(
            '_steps_per_epoch', 50) * optim_args.get(
            'epochs', 3)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(total_steps // 3, 1),
                T_mult=2, eta_min=self.lrate * 0.01))
        # loss: Huber-MAE自适应混合 (Vortex特有)
        self.loss = masked_mae
        self.huber_loss = huber_mae_adaptive
        self._use_huber = True
        self.clip = 5
        # 梯度噪声注入参数
        self._noise_eta = 0.01
        self._noise_gamma = 0.55
        # 诊断工具
        self.perf = PerfTimer()
        self.lr_tracker = LRTracker()
        self._global_step = 0

    def _inject_gradient_noise(self, step):
        """梯度噪声注入: 标准差随步数衰减"""
        std = (self._noise_eta /
               (1 + step) ** self._noise_gamma)
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * std
                p.grad.add_(noise)

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
        print(f"resume from epoch {epoch_num}, "
              f"lr={self.lrate}, cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.perf.start("forward")
        output = self.model(input)
        output = output.transpose(1, 2)
        self.perf.stop("forward")
        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning, "
                  f"lr reset to {self.lrate} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps)
                    % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)
        # scale data and compute loss
        self.perf.start("loss")
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
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_s[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(
                real_val[:, :, :, 0])
            # Huber-MAE自适应混合 (Vortex特有)
            if self._use_huber:
                mae_loss = self.huber_loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
            else:
                mae_loss = self.loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
        loss = mae_loss
        self.perf.stop("loss")
        self.perf.start("backward")
        loss.backward()
        # 梯度噪声注入 (Vortex特有)
        self._inject_gradient_noise(self._global_step)
        self.perf.stop("backward")
        # gradient clip
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip)
        self.optimizer.step()
        # CosineAnnealing step (per batch)
        self.lr_scheduler.step()
        # 诊断: 每N步dump
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_tracker.record(self._global_step, current_lr)
        if _is_debug() and self._global_step % 20 == 0:
            dump_struct_state(
                f"train_step_{self._global_step}",
                loss=loss.item(),
                lr=current_lr,
                cl_len=self.cl_len,
                predict_range=predict,
                real_val_range=real_val_s)
        self._global_step += 1
        # metrics
        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name,
             **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
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
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader,
             scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(
            dataloader['y_test']).to(device)
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
            if kwargs.get('dataset_name') in (
                    'PEMS04', 'PEMS08'):
                from sklearn.metrics import (
                    mean_absolute_error)
                mae = mean_absolute_error(
                    pred.cpu().numpy(),
                    real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = (metrics[0],
                                    metrics[1],
                                    metrics[2])
            log = ('Evaluate best model on test data '
                   'for horizon {:d}, Test MAE: {:.4f}, '
                   'Test RMSE: {:.4f}, Test MAPE: {:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
        log = ('(On average over 12 horizons) '
               'Test MAE: {:.2f} | Test RMSE: {:.2f} '
               '| Test MAPE: {:.2f}% |')
        print(log.format(
            np.mean(amae), np.mean(armse),
            np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
