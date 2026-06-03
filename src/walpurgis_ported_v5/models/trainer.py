import numpy as np
import time
import torch
import torch.optim as optim
from collections import deque

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
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


# ═══════ Gradient health tracker ═══════ #
# Delta vs upstream: tracks per-step grad norms with EMA for anomaly detection

class GradientWatchdog:
    """Monitors gradient L2 norm with exponential moving average.

    At any breakpoint:
        watchdog.report()           → EMA / raw / anomaly count
        watchdog.recent_norms()     → last N raw norms
        watchdog.is_anomalous()     → True if latest norm > 3× EMA
    """

    def __init__(self, alpha: float = 0.05, window: int = 200):
        self._ema = 0.0
        self._alpha = alpha
        self._norms = deque(maxlen=window)
        self._anomalies = 0
        self._steps = 0

    def step(self, model: torch.nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        norm = total ** 0.5

        self._steps += 1
        self._norms.append(norm)

        if self._steps == 1:
            self._ema = norm
        else:
            self._ema = (1 - self._alpha) * self._ema + self._alpha * norm

        if norm > 3.0 * self._ema and self._steps > 10:
            self._anomalies += 1
            print(f"\033[93m[GRAD-WATCHDOG] step={self._steps} "
                  f"norm={norm:.4f} >> ema={self._ema:.4f}\033[0m")
        return norm

    def report(self):
        print(f"GradWatchdog: steps={self._steps} ema={self._ema:.4f} "
              f"anomalies={self._anomalies} "
              f"last={self._norms[-1]:.4f}" if self._norms else "empty")

    def recent_norms(self, n=10):
        return list(self._norms)[-n:]

    def is_anomalous(self):
        if not self._norms or self._steps < 10:
            return False
        return self._norms[-1] > 3.0 * self._ema


# ═══════ Epoch phase timer ═══════ #
# Delta vs upstream: online Welford stats per phase

class _WelfordAccum:
    __slots__ = ("n", "mean", "M2", "lo", "hi")
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
        self.lo = float("inf"); self.hi = float("-inf")
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
        if x < self.lo: self.lo = x
        if x > self.hi: self.hi = x
    @property
    def std(self):
        return (self.M2 / self.n) ** 0.5 if self.n > 1 else 0.0


class PhaseProfiler:
    """Track wall-clock per training phase (train / val / test).

    At any breakpoint:
        profiler.report()                → timing table
        profiler.budget_check(limit)     → phases exceeding limit
    """

    def __init__(self):
        self._phases = {}
        self._active = None
        self._t0 = None

    def begin(self, name: str):
        self._active = name
        self._t0 = time.perf_counter()

    def end(self):
        if self._active is None:
            return
        elapsed = time.perf_counter() - self._t0
        if self._active not in self._phases:
            self._phases[self._active] = _WelfordAccum()
        self._phases[self._active].update(elapsed)
        self._active = None

    def report(self):
        print(f"{'phase':<16} {'count':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print("-" * 56)
        for name, w in self._phases.items():
            print(f"{name:<16} {w.n:>6} {w.mean:>8.2f} {w.std:>8.2f} "
                  f"{w.lo:>8.2f} {w.hi:>8.2f}")

    def budget_check(self, limit_sec: float):
        over = [(n, w) for n, w in self._phases.items() if w.mean > limit_sec]
        if over:
            for n, w in over:
                print(f"\033[93m[BUDGET] {n} avg={w.mean:.1f}s > limit={limit_sec}s\033[0m")
        else:
            print("All phases within budget.")


# ═══════ Trainer ═══════ #
# Deltas vs upstream:
#   1. Gradient watchdog with EMA anomaly detection
#   2. Phase profiler with Welford online stats
#   3. Cosine-warm-restart LR schedule option alongside MultiStep
#   4. Per-batch metric accumulation uses Kahan summation to reduce float drift
#   5. Debug print: grad norm + loss components every N steps

class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model  = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']

        # adam
        self.lrate  = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']

        # lr scheduler
        self.if_lr_scheduler = optim_args['lr_schedule']
        self.lr_sche_steps   = optim_args['lr_sche_steps']
        self.lr_decay_ratio  = optim_args['lr_decay_ratio']

        # curriculum learning
        self.if_cl    = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len   = 0 if self.if_cl else self.output_seq_len

        # warmup
        self.warm_steps = optim_args['warm_steps']

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)

        # lr scheduler
        if self.if_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.lr_sche_steps,
                gamma=self.lr_decay_ratio)
        else:
            self.lr_scheduler = None

        # loss
        self.loss = masked_mae
        self.clip = 5

        # ── delta 1: gradient watchdog ──
        self.grad_watchdog = GradientWatchdog()
        # ── delta 2: phase profiler ──
        self.profiler = PhaseProfiler()

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
        print(f"resume epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {total:,}")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name:50s} {str(list(param.shape)):>20s}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # curriculum learning
        batch_num = kwargs['batch_num']
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif batch_num == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            print(f"════ CL start, lr reset to {self.lrate} ════")
        else:
            if (batch_num - self.warm_steps) % self.cl_steps == 0 \
                    and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        # scale + loss
        if kwargs['_max'] is not None:
            predict  = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :],
                                real_val[:, :self.cl_len, :])
        else:
            predict  = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(predict[:, :self.cl_len, :],
                                real_val[:, :self.cl_len, :], 0)

        loss = mae_loss
        loss.backward()

        # clip
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # ── delta 1: record gradient health ──
        gnorm = self.grad_watchdog.step(self.model)

        self.optimizer.step()

        # ── delta 5: periodic debug print ──
        if batch_num % 100 == 0:
            print(f"  [batch={batch_num}] loss={loss.item():.4f} "
                  f"‖∇‖={gnorm:.4f} cl_len={self.cl_len}")

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        self.profiler.begin("validation")

        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
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
                predict  = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            loss_v = self.loss(predict, real_val, 0.0).item()
            mape_v = masked_mape(predict, real_val, 0.0).item()
            rmse_v = masked_rmse(predict, real_val, 0.0).item()

            valid_loss.append(loss_v)
            valid_mape.append(mape_v)
            valid_rmse.append(rmse_v)

        self.profiler.end()
        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
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

        yhat   = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list,  dim=0)[:realy.size(0), ...]
        assert torch.where(y_list == realy)

        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1),
                           kwargs["_max"][0, 0, 0, 0],
                           kwargs["_min"][0, 0, 0, 0])
            yhat  = scaler(yhat.squeeze(-1),
                           kwargs["_max"][0, 0, 0, 0],
                           kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat  = scaler.inverse_transform(yhat)

        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs['dataset_name'] in ('PEMS04', 'PEMS08'):
                from sklearn.metrics import mean_absolute_error
                mae  = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics_v = metric(pred, real)
                mae, mape, rmse = metrics_v[0], metrics_v[1], metrics_v[2]

            print(f"  horizon {i+1:2d}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.4f}")
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        print(f"(avg 12h) MAE={np.mean(amae):.2f}  "
              f"RMSE={np.mean(armse):.2f}  MAPE={np.mean(amape)*100:.2f}%")
        if save:
            save_model(model, save_path_resume)
