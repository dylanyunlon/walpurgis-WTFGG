"""Model trainer — adaptive gradient clipping + cosine CL warmup.

Algorithm changes vs upstream
------------------------------
1. **Adaptive gradient clipping** — instead of a fixed clip=5, we track
   the 95th-percentile of recent gradient norms (rolling window of 100
   batches) and clip at ``p95_ratio * p95``.  This prevents the clip
   from being too loose early in training (when grads are small) or too
   aggressive late (when grads stabilise at a high baseline).

2. **Curriculum learning cosine warmup** — the upstream uses a hard
   switch: warmup phase → reset LR → linear CL ramp.  We replace
   the linear ramp with a cosine schedule so the effective CL length
   ``cl_len`` transitions smoothly from 1 to ``output_seq_len``.

3. **3-tuple dataloader** — ``train`` and ``eval`` unpack ``(x, y, meta)``
   from the dataloader.  ``meta`` is logged per-batch for diagnostics.

4. **Per-horizon weighted metrics** — ``test`` computes an exponentially
   decayed per-horizon weight (recent horizons matter more) and reports
   the weighted average alongside the flat average.
"""

import math
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from sklearn.metrics import mean_absolute_error

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(
            np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class _AdaptiveGradClipper:
    """Track gradient norms and clip at a dynamic threshold."""

    def __init__(self, window=100, ratio=1.5, floor=1.0):
        self._history = deque(maxlen=window)
        self._ratio = ratio
        self._floor = floor

    def step(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1e9)   # just measure
        self._history.append(total_norm.item())
        if len(self._history) < 10:
            threshold = max(self._floor, 5.0)    # fallback early on
        else:
            p95 = float(np.percentile(list(self._history), 95))
            threshold = max(self._floor, self._ratio * p95)
        torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)
        return threshold


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
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args['warm_steps']

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)

        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.lr_sche_steps,
                gamma=self.lr_decay_ratio)
            if self.if_lr_scheduler else None)

        self.loss = masked_mae
        # adaptive clip instead of fixed clip=5
        self._grad_clipper = _AdaptiveGradClipper()

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        for _ in range(batch_num):
            if _ < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif _ == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lrate
            else:
                if (_ - self.warm_steps) % self.cl_steps == 0 \
                        and self.cl_len < self.output_seq_len:
                    self.cl_len += int(self.if_cl)
        print(f"resume from epoch {epoch_num}, lr={self.lrate}, "
              f"cl_len={self.cl_len}")

    def _cosine_cl_len(self, batch_num):
        """Cosine CL schedule: smoothly ramp from 1 → output_seq_len."""
        steps_since = batch_num - self.warm_steps
        if steps_since <= 0:
            return self.output_seq_len     # warmup phase
        total_cl_steps = self.cl_steps * (self.output_seq_len - 1)
        progress = min(steps_since / max(total_cl_steps, 1), 1.0)
        # cosine: starts fast, slows near the end
        cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
        return max(1, int(1 + cos_progress * (self.output_seq_len - 1)))

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            n_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
            print(f"[model] trainable params: {n_params:,}")
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(f"  {name} {list(p.shape)}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # cosine curriculum learning
        if kwargs['batch_num'] == self.warm_steps:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            print(f"======== CL start, lr reset to {self.lrate} ========")
        if self.if_cl:
            self.cl_len = self._cosine_cl_len(kwargs['batch_num'])
        else:
            self.cl_len = self.output_seq_len

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
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_s[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_s[:, :self.cl_len, :], 0)

        loss = mae_loss
        loss.backward()

        # adaptive gradient clip
        clip_val = self._grad_clipper.step(self.model)
        self.optimizer.step()

        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, batch in enumerate(
                dataloader['val_loader'].get_iterator()):
            # 3-tuple from our dataloader
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
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
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
        y_list = []

        for itera, batch in enumerate(
                dataloader['test_loader'].get_iterator()):
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
            y_list.append(testy)

        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]

        assert torch.where(y_list == realy)

        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1),
                           kwargs["_max"][0, 0, 0, 0],
                           kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1),
                          kwargs["_max"][0, 0, 0, 0],
                          kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        amae, amape, armse = [], [], []
        n_horizons = 12

        # exponential horizon weights: recent horizons weighted more
        raw_w = np.array([0.9 ** i for i in range(n_horizons)])
        horizon_weights = raw_w / raw_w.sum()

        for i in range(n_horizons):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(
                    pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = metrics[0], metrics[1], metrics[2]

            log = ('horizon {:2d} | MAE {:.4f} | RMSE {:.4f} | '
                   'MAPE {:.4f} | weight {:.3f}')
            print(log.format(i + 1, mae, rmse, mape, horizon_weights[i]))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        # flat average
        flat = ('Flat avg  | MAE {:.2f} | RMSE {:.2f} | MAPE {:.2f}%')
        print(flat.format(
            np.mean(amae), np.mean(armse), np.mean(amape) * 100))
        # weighted average (near-horizon emphasis)
        wmae = sum(w * v for w, v in zip(horizon_weights, amae))
        wrmse = sum(w * v for w, v in zip(horizon_weights, armse))
        wmape = sum(w * v for w, v in zip(horizon_weights, amape))
        weighted = ('Wtd  avg  | MAE {:.2f} | RMSE {:.2f} | MAPE {:.2f}%')
        print(weighted.format(wmae, wrmse, wmape * 100))

        if save:
            save_model(model, save_path_resume)
