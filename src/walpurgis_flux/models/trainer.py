"""
Flux trainer — 算法改写:
  1. ExponentialLR + 线性warmup ramp替代MultiStepLR
  2. Focal MAE损失(聚焦hard samples)
  3. 渐进式解码多级损失(粗细两路加权)
  4. 流式推理统计追踪
  5. 每N步自动dump训练状态
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis_flux.utils.train import (
    data_reshaper, save_model)
from .losses import (
    masked_mae, masked_rmse, masked_mape, metric,
    focal_mae, progressive_refinement_loss)
from walpurgis_flux import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    StreamWindowTracker, ProgressiveDecodeTracker)


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
        # Flux: ExponentialLR + 线性warmup ramp
        # ExponentialLR: 每epoch乘以gamma, 平滑衰减
        self.lr_scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95))
        self._warmup_steps = max(
            optim_args.get('_steps_per_epoch', 50), 20)
        self._warmup_done = False
        # Flux: Focal MAE作为主损失
        self.loss = masked_mae
        self.focal_loss = focal_mae
        self._use_focal = True
        # Focal超参
        self._focal_gamma = 2.0
        self._focal_alpha = 0.75
        # progressive decode loss权重
        self._coarse_loss_weight = 0.3
        self.clip = 5
        # 诊断工具
        self.perf = PerfTimer()
        self.stream_tracker = StreamWindowTracker()
        self.progressive_tracker = ProgressiveDecodeTracker()
        self._global_step = 0

    def _warmup_lr(self, step):
        """Flux: 线性warmup ramp — 前N步线性增长lr"""
        if step < self._warmup_steps:
            warmup_factor = (step + 1) / self._warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lrate * warmup_factor
            _dbg("warmup_lr",
                 f"step={step} factor={warmup_factor:.4f}",
                 "trainer")
        elif not self._warmup_done:
            self._warmup_done = True
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lrate
            _dbg("warmup_complete",
                 f"lr={self.lrate}", "trainer")

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
        # Flux: warmup ramp
        self._warmup_lr(self._global_step)
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
            # Flux: Focal MAE (聚焦hard samples)
            if self._use_focal:
                mae_loss = self.focal_loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0,
                    gamma=self._focal_gamma,
                    alpha=self._focal_alpha)
            else:
                mae_loss = self.loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
        loss = mae_loss
        # Flux: progressive decode额外损失
        if hasattr(self.model, '_last_coarse') and \
                self.model._last_coarse is not None:
            coarse = self.model._last_coarse
            coarse_flat = coarse.transpose(
                1, 2).contiguous().view(
                coarse.shape[0], coarse.shape[2], -1)
            # 对齐coarse和real_val维度
            min_len = min(
                coarse_flat.shape[-1],
                real_val_s.shape[-1])
            if min_len > 0:
                coarse_aligned = coarse_flat[
                    :, :, :min_len]
                real_aligned = real_val_s[:, :, :min_len]
                # 只用前cl_len步
                coarse_loss = self.loss(
                    coarse_aligned[:, :self.cl_len, :],
                    real_aligned[:, :self.cl_len, :], 0)
                loss = (loss + self._coarse_loss_weight *
                        coarse_loss)
                self.progressive_tracker.record(
                    "coarse", coarse_loss.item())
            self.progressive_tracker.record(
                "fine", mae_loss.item())
        self.perf.stop("loss")
        self.perf.start("backward")
        loss.backward()
        self.perf.stop("backward")
        # gradient clip
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip)
        self.optimizer.step()
        # 诊断: 每N步dump
        current_lr = self.optimizer.param_groups[0]['lr']
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
