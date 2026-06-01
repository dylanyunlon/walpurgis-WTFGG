"""
Walpurgis v2 Training Engine — Instrumented Trainer
=====================================================
Re-ported with ≈20 % algorithmic delta.

Deltas vs prior Walpurgis trainer:
  1. Adaptive clip: prior used max(5, 2×median); now uses
     *percentile-based* clip = p75 + 1.5·IQR of recent grad norms
     (box-plot fence).  This is more robust to skewed distributions.
  2. CL ramp: prior sigmoid → now *cosine annealing* ramp.
     cl_len = ceil(output_len × (1 - cos(π·progress)) / 2).
  3. Gradient forensics: per-layer *relative gradient norm* (ratio to
     parameter norm) is tracked — detects vanishing better than
     absolute norm.
  4. Per-step JSON log line for external analysis (grep-friendly).
"""

import numpy as np
import time
import math
import json
import torch
import torch.optim as optim
from collections import deque

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, MetricTracker


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = ~np.isnan(y_true) if np.isnan(null_val) else np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask = mask / np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer:
    """Training engine with percentile-based gradient clipping and cosine CL ramp.

    Debug helpers — call at any breakpoint:
        engine._gradient_forensics()      # per-param gradient health
        engine._timing_summary()          # cumulative timing breakdown
        engine._cl_schedule_preview(200)  # preview CL schedule for next 200 steps
        MetricTracker.report()            # all loss/metric statistics
    """

    def __init__(self, scaler, model, **oa):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = oa["output_seq_len"]
        self.print_model_structure = oa["print_model"]

        self.lrate = oa["lrate"]
        self.wdecay = oa["wdecay"]
        self.eps = oa["eps"]

        self.if_lr_scheduler = oa["lr_schedule"]
        self.lr_sche_steps = oa["lr_sche_steps"]
        self.lr_decay_ratio = oa["lr_decay_ratio"]

        self.if_cl = oa["if_cl"]
        self.cl_steps = oa["cl_steps"]
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = oa["warm_steps"]

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps,
        )
        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio,
            )
            if self.if_lr_scheduler
            else None
        )

        self.loss = masked_mae

        # ── Percentile-based adaptive clip ──
        self._base_clip = 5.0
        self._recent_gn = deque(maxlen=300)

        # ── Timing accumulators ──
        self._step = 0
        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._cum_opt = 0.0
        self._cum_loss = 0.0

    # ─── Percentile clip ───
    def _percentile_clip(self):
        """IQR fence: p75 + 1.5·(p75 − p25).  Falls back to base_clip if < 20 samples."""
        if len(self._recent_gn) < 20:
            return self._base_clip
        arr = np.array(self._recent_gn)
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q75 - q25
        fence = q75 + 1.5 * iqr
        return max(self._base_clip, float(fence))

    # ─── Cosine CL ramp ───
    def _cosine_cl(self, step):
        """Curriculum length via cosine annealing: smooth 1→output_seq_len.

        progress = clamp((step - warm_steps) / total_cl_batches, 0, 1)
        cl_len   = ceil(output_seq_len × (1 − cos(π·progress)) / 2)
        """
        total_cl = self.cl_steps * self.output_seq_len
        raw = (step - self.warm_steps) / max(total_cl, 1)
        progress = max(0.0, min(raw, 1.0))
        frac = (1.0 - math.cos(math.pi * progress)) / 2.0
        return max(1, math.ceil(frac * self.output_seq_len))

    def _cl_schedule_preview(self, n_steps=200):
        """Print upcoming CL schedule — call from pdb."""
        print(f"  [CL preview] next {n_steps} steps from current step={self._step}:")
        prev = -1
        for s in range(self._step, self._step + n_steps):
            cl = self._cosine_cl(s) if s >= self.warm_steps else self.output_seq_len
            if cl != prev:
                print(f"    step={s}: cl_len={cl}")
                prev = cl

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            print(f"  [CL] Fresh start — cl_len=0, lr={self.lrate}")
            return
        print(f"  [CL] Resuming from epoch={epoch_num}, batch={batch_num}")
        transitions = []
        for b in range(batch_num):
            if b < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif b == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lrate
                transitions.append(f"    batch={b}: warmup→CL, cl=1")
            else:
                new = self._cosine_cl(b)
                if new != self.cl_len:
                    self.cl_len = new
                    transitions.append(f"    batch={b}: cl→{self.cl_len}")
        print(f"  [CL] Final: lr={self.lrate}, cl_len={self.cl_len}")
        show = transitions[:3] + (["    ..."] if len(transitions) > 6 else []) + transitions[-3:]
        for t in show:
            print(t)

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs["batch_num"]) == 0:
            tp = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  [MODEL] Trainable params: {tp:,d}")

    # ─── Gradient forensics ───
    def _gradient_forensics(self):
        """Per-parameter gradient health with *relative* norm (grad/param)."""
        print(f"\n{'━'*66}")
        print(f"  Gradient Forensics @ step {self._step}")
        print(f"{'━'*66}")
        clip_v = self._percentile_clip()
        issues = 0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data
            gn = g.norm(2).item()
            pn = p.data.norm(2).item()
            rel = gn / (pn + 1e-12)
            zero_frac = (g == 0).float().mean().item()
            flags = []
            if torch.isnan(g).any():
                flags.append("\033[91mNaN\033[0m")
            if torch.isinf(g).any():
                flags.append("\033[91mInf\033[0m")
            if gn > clip_v * 10:
                flags.append(f"\033[91mexploding({gn:.1f})\033[0m")
            if gn < 1e-8:
                flags.append("\033[93mvanishing\033[0m")
            if zero_frac > 0.5:
                flags.append(f"\033[93mdead({zero_frac*100:.0f}%)\033[0m")
            if rel > 100:
                flags.append(f"\033[93mrel_high({rel:.1f})\033[0m")
            if flags:
                issues += 1
                fl = " ".join(flags)
                print(f"    {name:50s} | gn={gn:.6f}  pn={pn:.4f}  rel={rel:.4f} | {fl}")
        if issues == 0:
            print(f"    ✓ all {sum(1 for _,p in self.model.named_parameters() if p.grad is not None)} grads healthy (clip={clip_v:.2f})")
        print(f"{'━'*66}")
        return issues

    def _timing_summary(self):
        if self._step == 0:
            print("  [TIMING] no steps yet")
            return
        print(f"\n  [TIMING] step={self._step}")
        print(f"    fwd:  {self._cum_fwd/self._step:.1f}ms avg ({self._cum_fwd:.0f}ms total)")
        print(f"    loss: {self._cum_loss/self._step:.1f}ms avg")
        print(f"    bwd:  {self._cum_bwd/self._step:.1f}ms avg")
        print(f"    opt:  {self._cum_opt/self._step:.1f}ms avg")

    def train(self, input, real_val, **kwargs):
        self._step += 1
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        # Forward
        t0 = time.perf_counter()
        output = self.model(input).transpose(1, 2)
        fwd_ms = (time.perf_counter() - t0) * 1000
        self._cum_fwd += fwd_ms

        # Cosine CL ramp
        bn = kwargs["batch_num"]
        cl_event = None
        if bn < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            cl_event = f"CL START: cl=1, lr={self.lrate}"
        else:
            new = self._cosine_cl(bn)
            if new != self.cl_len:
                old = self.cl_len
                self.cl_len = new
                cl_event = f"CL RAMP: {old}→{self.cl_len}"
        if cl_event:
            print(f"  [CL] batch={bn}: {cl_event}")

        # Loss
        t0 = time.perf_counter()
        if kwargs["_max"] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0],
            ).transpose(1, 2).squeeze(-1)
            rvs = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0],
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len], rvs[:, :self.cl_len])
        else:
            predict = self.scaler.inverse_transform(output)
            rvi = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(predict[:, :self.cl_len], rvi[:, :self.cl_len], 0)
        self._cum_loss += (time.perf_counter() - t0) * 1000

        # Backward
        t0 = time.perf_counter()
        mae_loss.backward()
        self._cum_bwd += (time.perf_counter() - t0) * 1000

        # Percentile clip + optimize
        t0 = time.perf_counter()
        clip_v = self._percentile_clip()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_v).item()
        self._recent_gn.append(gn)
        self.optimizer.step()
        self._cum_opt += (time.perf_counter() - t0) * 1000

        # Periodic output
        if self._step % 50 == 0:
            print(
                f"  [PERF] step={self._step}: fwd={fwd_ms:.1f}ms  "
                f"grad_norm={gn:.4f} (clip={clip_v:.2f})  cl={self.cl_len}"
            )
        if self._step % 250 == 0:
            self._gradient_forensics()

        # JSON log line (grep-friendly)
        if self._step % 100 == 0:
            print(
                f"  [JSON] {json.dumps({'step': self._step, 'mae': mae_loss.item(), 'gn': round(gn, 5), 'clip': round(clip_v, 3), 'cl': self.cl_len, 'lr': self.optimizer.param_groups[0]['lr']})}"
            )

        rv = rvi if kwargs["_max"] is None else rvs
        mape = masked_mape(predict, rv, 0.0)
        rmse = masked_rmse(predict, rv, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        t_start = time.perf_counter()

        for itera, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)

            if kwargs["_max"] is not None:
                predict = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0],
                )
                real_val = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0],
                )
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            valid_loss.append(self.loss(predict, real_val, 0.0).item())
            valid_mape.append(masked_mape(predict, real_val, 0.0).item())
            valid_rmse.append(masked_rmse(predict, real_val, 0.0).item())

        elapsed = (time.perf_counter() - t_start) * 1000
        ml = np.mean(valid_loss)
        print(
            f"  [EVAL] {len(valid_loss)} batches in {elapsed:.0f}ms, "
            f"loss={ml:.4f}±{np.std(valid_loss):.4f}"
        )
        return ml, np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name,
             save=True, **kwargs):
        model.eval()
        outputs, y_list = [], []
        realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 2)

        print(f"\n  [TEST] Starting evaluation...")
        t0 = time.perf_counter()

        for itera, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                outputs.append(model(testx))
            y_list.append(testy)

        yhat = torch.cat(outputs, dim=0)[: realy.size(0)]
        y_list = torch.cat(y_list, dim=0)[: realy.size(0)]
        assert torch.where(y_list == realy)

        if kwargs["_max"] is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        amae, amape, armse = [], [], []
        print(f"  {'Horizon':>8s} │ {'MAE':>8s} │ {'RMSE':>8s} │ {'MAPE':>8s}")
        print(f"  {'─'*42}")

        for h in range(12):
            ph, rh = yhat[:, :, h], realy[:, :, h]
            if kwargs["dataset_name"] in ("PEMS04", "PEMS08"):
                from sklearn.metrics import mean_absolute_error
                mae_h = mean_absolute_error(ph.cpu().numpy(), rh.cpu().numpy())
                rmse_h = masked_rmse(ph, rh, 0.0).item()
                mape_h = masked_mape(ph, rh, 0.0).item()
            else:
                mae_h, mape_h, rmse_h = metric(ph, rh)
            print(f"  {h+1:>8d} │ {mae_h:>8.4f} │ {rmse_h:>8.4f} │ {mape_h:>8.4f}")
            amae.append(mae_h)
            amape.append(mape_h)
            armse.append(rmse_h)

        print(f"  {'─'*42}")
        print(
            f"  {'Average':>8s} │ {np.mean(amae):>8.2f} │ {np.mean(armse):>8.2f} │ "
            f"{np.mean(amape)*100:>7.2f}%"
        )
        print(f"  [TEST] Done in {(time.perf_counter()-t0)*1000:.0f}ms")

        if save:
            save_model(model, save_path_resume)
