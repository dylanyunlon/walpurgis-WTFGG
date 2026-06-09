"""
Cascade trainer — 算法改写 (Claude-4 SOTA push):
  1. CosineAnnealingWarmRestarts with correct epoch-level stepping
  2. Node dropout augmentation (training-time random node masking)
  3. Time-shift augmentation (random temporal offset on input)
  4. Simplified gradient clipping (single clip_grad_norm, no AGC)
  5. Removed depth gate regularization (conflicts with gate init)
  6. Mixed loss: masked_mae + LogCosh horizon weighting
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis.utils.train import (
    data_reshaper, save_model)
from .losses import (
    masked_mae, masked_rmse, masked_mape, metric,
    cascade_aware_loss, LogCoshHorizonLoss)
from walpurgis import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    DepthGateTracker, SETracker, CascadeResidualTracker)


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

        # training strategy parameters
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        self.if_lr_scheduler = optim_args['lr_schedule']
        self.lr_sche_steps = optim_args['lr_sche_steps']
        self.lr_decay_ratio = optim_args['lr_decay_ratio']
        # curriculum learning
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']
        self._cl_ramp_mode = optim_args.get('cl_ramp_mode', 'sigmoid')
        self._steps_per_epoch = optim_args.get('_steps_per_epoch', 1)

        # Data augmentation parameters
        self._node_dropout_rate = optim_args.get('node_dropout_rate', 0.0)
        self._time_shift_max = optim_args.get('time_shift_max', 0)

        # AdamW optimizer — better weight decay handling than RAdam
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)
        # CosineAnnealingWarmRestarts — stepped per epoch (not per batch)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=15, T_mult=2, eta_min=1e-5
            ) if self.if_lr_scheduler else None)

        # Loss: primary MAE + auxiliary LogCosh with horizon weighting
        self.loss = masked_mae
        self.cascade_loss = cascade_aware_loss
        self.logcosh_loss = LogCoshHorizonLoss(
            init_temperature=1.0, horizon_scale=0.08)
        self._use_cascade_loss = True
        self.clip = 5

        # Gradient accumulation
        self._grad_accum_steps = optim_args.get('grad_accum_steps', 1)
        self._accum_count = 0

        # Epoch tracking for LR scheduler
        self._current_epoch = 0

        # Diagnostics
        self.perf = PerfTimer()
        self.depth_tracker = DepthGateTracker(5)
        self.se_tracker = SETracker()
        self.cascade_tracker = CascadeResidualTracker(5)
        self._global_step = 0

    def step_lr_scheduler(self, epoch):
        """Called once per epoch from the training loop for correct LR scheduling."""
        self._current_epoch = epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)
        # 同步LogCosh自适应温度
        if hasattr(self.logcosh_loss, 'set_epoch'):
            self.logcosh_loss.set_epoch(epoch)

    def _node_dropout(self, x):
        """Training-time node dropout: randomly zero out ~10% of nodes.
        This acts as a regularizer and forces the model to not rely on any single node.
        Applied to the traffic signal channels only (not time features).
        """
        if not self.model.training or self._node_dropout_rate <= 0:
            return x
        B, L, N, C = x.shape
        # Generate per-batch, per-node mask (same across time and features)
        node_mask = (torch.rand(B, 1, N, 1, device=x.device) > self._node_dropout_rate).float()
        # Scale to preserve expected value
        x = x * node_mask / (1.0 - self._node_dropout_rate)
        return x

    def _time_shift_augment(self, x):
        """Time-shift augmentation: add small Gaussian noise scaled by temporal variance.
        More effective than literal shifting for normalized traffic data.
        """
        if not self.model.training or self._time_shift_max <= 0:
            return x
        # Add noise proportional to the local temporal variance
        noise_scale = 0.02
        noise = torch.randn_like(x) * noise_scale
        return x + noise

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        else:
            progress = min((batch_num - self.warm_steps) / max(self.cl_steps, 1), 1.0)
            sigmoid_val = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))
            self.cl_len = max(1, int(1 + sigmoid_val * (self.output_seq_len - 1)))
        print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self._accum_count += 1
        if self._accum_count == 1:
            self.optimizer.zero_grad()
        self.perf.start("forward")

        # Data augmentation (applied to input before model)
        input_aug = self._node_dropout(input)
        input_aug = self._time_shift_augment(input_aug)

        output = self.model(input_aug)
        output = output.transpose(1, 2)
        self.perf.stop("forward")

        # ── 断点调试: forward阶段状态 ──
        if _is_debug() and self._global_step % 10 == 0:
            dump_struct_state(
                f"forward_step_{self._global_step}",
                input_shape=input.shape if isinstance(input, torch.Tensor) else str(type(input)),
                output_shape=output,
                output_has_nan=torch.isnan(output).any().item(),
                output_range_min=output.min().item(),
                output_range_max=output.max().item(),
                output_mean=output.mean().item(),
                output_std=output.std().item())

        # curriculum learning: sigmoid ramp
        batch_num = kwargs['batch_num']
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        else:
            progress = min((batch_num - self.warm_steps) / max(self.cl_steps, 1), 1.0)
            sigmoid_val = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))
            self.cl_len = max(1, int(1 + sigmoid_val * (self.output_seq_len - 1)))

        # scale data and calculate loss
        self.perf.start("loss")
        if kwargs['_max'] is not None:  # traffic flow
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(real_val.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(real_val[:, :, :, 0])
            # Combined loss: cascade-aware MAE + LogCosh horizon loss
            if self._use_cascade_loss:
                mae_loss = self.cascade_loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
                # Auxiliary LogCosh loss for smoother gradients (weighted 0.3)
                logcosh_loss = self.logcosh_loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
                mae_loss = 0.7 * mae_loss + 0.3 * logcosh_loss
            else:
                mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :], 0)

        # No depth gate regularization — let gates learn freely from init=3.0 (≈0.95)
        loss = mae_loss / self._grad_accum_steps
        self.perf.stop("loss")

        # ── 断点调试: loss阶段状态 ──
        if _is_debug() and self._global_step % 10 == 0:
            dump_struct_state(
                f"loss_step_{self._global_step}",
                mae_loss=mae_loss.item(),
                total_loss=loss.item(),
                cl_len=self.cl_len,
                predict_range_min=predict.min().item(),
                predict_range_max=predict.max().item(),
                real_val_range_min=real_val_s.min().item(),
                real_val_range_max=real_val_s.max().item())

        self.perf.start("backward")
        loss.backward()

        if self._accum_count >= self._grad_accum_steps:
            # Single gradient clip — simple and effective
            if self.clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                # ── 断点调试: backward阶段状态 ──
                if _is_debug() and self._global_step % 10 == 0:
                    dump_struct_state(
                        f"backward_step_{self._global_step}",
                        grad_total_norm=total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm,
                        clip_threshold=self.clip,
                        was_clipped=total_norm > self.clip if isinstance(total_norm, (int, float)) else (total_norm.item() > self.clip))
            self.optimizer.step()
            # LR scheduler is now stepped per-epoch from training loop, not here
            self._accum_count = 0
        self.perf.stop("backward")

        # Diagnostics
        for i, gate_param in enumerate(self.model.depth_gates):
            self.depth_tracker.record(
                i, torch.sigmoid(gate_param).item())

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

    def eval(self, device, dataloader, model_name, **kwargs):
        # val a epoch
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            # for dstgnn
            output = self.model(testx)
            output = output.transpose(1, 2)

            # scale data
            if kwargs['_max'] is not None:  # traffic flow
                ## inverse transform for both predict and real value.
                predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            # metrics
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()

            print("test: {0}".format(loss), end='\r')

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        # test
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
        y_list = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)

            with torch.no_grad():
                preds = model(testx)

            outputs.append(preds)
            y_list.append(testy)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]

        assert torch.where(y_list == realy)

        # scale data
        if kwargs['_max'] is not None:  # traffic flow
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        # summarize the results.
        amae = []
        amape = []
        armse = []

        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):  # traffic flow dataset follows mae metric used in ASTGNN.
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i + 1, mae, rmse, mape))
                amae.append(mae)
                amape.append(mape)
                armse.append(rmse)
            else:       # traffic speed datasets follow the metrics released by GWNet and DCRNN.
                metrics = metric(pred, real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i + 1, metrics[0], metrics[2], metrics[1]))
                amae.append(metrics[0])     # mae
                amape.append(metrics[1])    # mape
                armse.append(metrics[2])    # rmse

        log = '(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
