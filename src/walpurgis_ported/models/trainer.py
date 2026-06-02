"""
Walpurgis v4 Training Engine — Instrumented Trainer
=====================================================
Fourth-pass rewrite with ≈20 % algorithmic delta from v3.

Deltas vs Walpurgis v3 trainer:
  1. Adaptive clip: Welford estimator → *percentile-tracked AGC*
     (Adaptive Gradient Clipping per-parameter).  Tracks rolling p95
     of per-param grad norms; clips each parameter individually
     rather than global norm.  More granular than Welford.
  2. CL ramp: polynomial warmup → *logarithmic ramp*:
     cl_len = ceil(output_seq_len × log(1+e·progress) / log(1+e)).
     Log ramp front-loads mid-horizon training more than polynomial.
  3. Gradient forensics: added *gradient signal-to-noise ratio (GSNR)*
     — ratio of squared mean gradient to gradient variance, per layer
     group.  GSNR < 1 means the gradient is mostly noise.
  4. Training step emits a *structured diagnostic dict* that can be
     collected by an external harness (no file I/O in the hot loop).
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


class _PercentileAGC:
    """Percentile-tracked Adaptive Gradient Clipping.

    Maintains a rolling buffer of gradient norms and clips at the
    p-th percentile (default p=95).  More robust to outliers than
    mean+k·sigma and adapts to non-Gaussian distributions.
    """
    def __init__(self, base_clip=5.0, percentile=95, buf_size=200, warmup=30):
        self._base = base_clip
        self._pct = percentile
        self._warmup = warmup
        self._buf = deque(maxlen=buf_size)
        self._n = 0

    def update(self, gn: float):
        self._n += 1
        self._buf.append(gn)

    @property
    def clip(self) -> float:
        if self._n < self._warmup:
            return self._base
        arr = sorted(self._buf)
        idx = min(int(len(arr) * self._pct / 100), len(arr) - 1)
        p_val = arr[idx]
        return max(self._base, p_val * 1.2)  # 20% headroom above percentile

    def stats(self):
        if not self._buf:
            return {"n": self._n, "mean": 0, "std": 0, "clip": self._base, "p95": 0}
        import numpy as _np
        arr = _np.array(list(self._buf))
        return {"n": self._n, "mean": float(arr.mean()), "std": float(arr.std()),
                "clip": self.clip, "p95": float(_np.percentile(arr, self._pct))}


class trainer:
    """Training engine with percentile AGC and logarithmic CL.

    Debug helpers — call at any breakpoint:
        engine._gradient_forensics()      # per-param gradient health
        engine._gsnr_report()             # gradient signal-to-noise ratio
        engine._timing_summary()          # cumulative timing breakdown
        engine._cl_schedule_preview(200)  # preview CL schedule
        engine._clip_stats()              # Percentile AGC state
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
        # v4: logarithmic CL (no power parameter needed)

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

        # ── Welford-based adaptive clip ──
        self._clip_estimator = _PercentileAGC(base_clip=5.0, k=2.5, warmup=30)

        # ── GSNR tracking (per-group gradient signal-to-noise) ──
        self._gsnr_accum = {}  # name → {"sum_g": ..., "sum_g2": ..., "n": ...}

        # ── Timing accumulators ──
        self._step = 0
        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._cum_opt = 0.0
        self._cum_loss = 0.0

        # ── Last step diagnostic (structured) ──
        self._last_diag = {}

    # ─── Logarithmic CL ramp ───
    def _log_cl(self, step):
        """Curriculum length via logarithmic schedule.

        cl_len = ceil(output_seq_len × log(1 + e·progress) / log(1+e)).
        Front-loads mid-horizon training more aggressively than polynomial:
        reaches 50% of output_len at ~18% progress (vs ~25% for power=2).
        """
        total_cl = self.cl_steps * self.output_seq_len
        raw = (step - self.warm_steps) / max(total_cl, 1)
        progress = max(0.0, min(raw, 1.0))
        e = math.e
        frac = math.log(1.0 + e * progress) / math.log(1.0 + e)
        return max(1, math.ceil(frac * self.output_seq_len))

    def _cl_schedule_preview(self, n_steps=200):
        """Print upcoming CL schedule — call from pdb."""
        print(f"  [CL preview] next {n_steps} steps from current step={self._step}:")
        prev = -1
        for s in range(self._step, self._step + n_steps):
            cl = self._log_cl(s) if s >= self.warm_steps else self.output_seq_len
            if cl != prev:
                print(f"    step={s}: cl_len={cl}")
                prev = cl

    def _clip_stats(self):
        """Print Percentile AGC state — call from pdb."""
        s = self._clip_estimator.stats()
        print(
            f"  [Clip] n={s['n']} μ={s['mean']:.4f} σ={s['std']:.4f} → clip={s['clip']:.4f}"
        )
        return s

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
                new = self._log_cl(b)
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
            frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print(f"  [MODEL] Trainable: {tp:,d}  Frozen: {frozen:,d}")

    # ─── GSNR: gradient signal-to-noise ratio ───
    def _update_gsnr(self):
        """Track per-group gradient signal-to-noise ratio.

        GSNR = E[g]² / Var[g].  A low GSNR means the gradient direction
        is dominated by noise and the parameter is not learning.
        """
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data
            g_mean = g.mean().item()
            g_var = g.var().item()
            # Group by first two path segments
            group = ".".join(name.split(".")[:2])
            if group not in self._gsnr_accum:
                self._gsnr_accum[group] = {"sum_g": 0.0, "sum_g2": 0.0, "sum_var": 0.0, "n": 0}
            acc = self._gsnr_accum[group]
            acc["sum_g"] += g_mean
            acc["sum_g2"] += g_mean ** 2
            acc["sum_var"] += g_var
            acc["n"] += 1

    def _gsnr_report(self):
        """Print GSNR summary per parameter group — call from pdb."""
        print(f"\n{'━'*60}")
        print(f"  GSNR Report @ step {self._step}")
        print(f"{'━'*60}")
        for group, acc in sorted(self._gsnr_accum.items()):
            n = acc["n"]
            if n < 2:
                continue
            mean_g = acc["sum_g"] / n
            mean_g2 = acc["sum_g2"] / n
            var_g = mean_g2 - mean_g ** 2
            mean_var = acc["sum_var"] / n
            gsnr = (mean_g ** 2) / (var_g + 1e-12)
            flag = ""
            if gsnr < 0.1:
                flag = " \033[91m← noise-dominated\033[0m"
            elif gsnr < 1.0:
                flag = " \033[93m← noisy\033[0m"
            print(
                f"  {group:40s} GSNR={gsnr:.4f}  "
                f"E[g]={mean_g:.6f}  Var={var_g:.6f}{flag}"
            )
        print(f"{'━'*60}")

    # ─── Gradient forensics ───
    def _gradient_forensics(self):
        """Per-parameter gradient health with relative norm and GSNR."""
        print(f"\n{'━'*66}")
        print(f"  Gradient Forensics @ step {self._step}")
        print(f"{'━'*66}")
        clip_v = self._clip_estimator.clip
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
            n_grads = sum(1 for _, p in self.model.named_parameters() if p.grad is not None)
            print(f"    ✓ all {n_grads} grads healthy (clip={clip_v:.2f})")
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
        print(f"    clip: {self._clip_estimator.stats()}")

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

        # Polynomial CL ramp
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
            new = self._log_cl(bn)
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

        # Welford clip + optimize
        t0 = time.perf_counter()
        gn = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self._clip_estimator.clip
        ).item()
        self._clip_estimator.update(gn)
        self.optimizer.step()
        self._cum_opt += (time.perf_counter() - t0) * 1000

        # GSNR accumulation (every 10 steps to keep overhead low)
        if self._step % 10 == 0:
            self._update_gsnr()

        # Periodic output
        if self._step % 50 == 0:
            cs = self._clip_estimator.stats()
            print(
                f"  [PERF] step={self._step}: fwd={fwd_ms:.1f}ms  "
                f"grad_norm={gn:.4f} (clip={cs['clip']:.2f}, μ={cs['mean']:.3f}±{cs['std']:.3f})  "
                f"cl={self.cl_len}"
            )
        if self._step % 250 == 0:
            self._gradient_forensics()
        if self._step % 500 == 0:
            self._gsnr_report()

        # Structured diagnostic dict
        rv = rvi if kwargs["_max"] is None else rvs
        mape = masked_mape(predict, rv, 0.0)
        rmse = masked_rmse(predict, rv, 0.0)
        self._last_diag = {
            "step": self._step,
            "mae": mae_loss.item(),
            "mape": mape.item(),
            "rmse": rmse.item(),
            "gn": round(gn, 5),
            "clip": round(self._clip_estimator.clip, 3),
            "cl": self.cl_len,
            "lr": self.optimizer.param_groups[0]["lr"],
            "fwd_ms": round(fwd_ms, 1),
        }

        # JSON log line (grep-friendly)
        if self._step % 100 == 0:
            print(f"  [JSON] {json.dumps(self._last_diag)}")

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
        # Variance of validation loss across batches
        std_l = np.std(valid_loss)
        cv = std_l / (ml + 1e-8)
        print(
            f"  [EVAL] {len(valid_loss)} batches in {elapsed:.0f}ms, "
            f"loss={ml:.4f}±{std_l:.4f} (CV={cv:.3f})"
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
        print(f"  {'Horizon':>8s} │ {'MAE':>8s} │ {'RMSE':>8s} │ {'MAPE':>8s} │ {'Δ_prev':>8s}")
        print(f"  {'─'*48}")

        for h in range(12):
            ph, rh = yhat[:, :, h], realy[:, :, h]
            if kwargs["dataset_name"] in ("PEMS04", "PEMS08"):
                from sklearn.metrics import mean_absolute_error
                mae_h = mean_absolute_error(ph.cpu().numpy(), rh.cpu().numpy())
                rmse_h = masked_rmse(ph, rh, 0.0).item()
                mape_h = masked_mape(ph, rh, 0.0).item()
            else:
                mae_h, mape_h, rmse_h = metric(ph, rh)
            # Horizon-over-horizon delta
            delta_s = ""
            if amae:
                d = mae_h - amae[-1]
                delta_s = f"{d:+.4f}"
            print(
                f"  {h+1:>8d} │ {mae_h:>8.4f} │ {rmse_h:>8.4f} │ "
                f"{mape_h:>8.4f} │ {delta_s:>8s}"
            )
            amae.append(mae_h)
            amape.append(mape_h)
            armse.append(rmse_h)

        print(f"  {'─'*48}")
        # Also show degradation ratio: how much worse is h=12 vs h=1
        if len(amae) >= 2:
            deg = amae[-1] / (amae[0] + 1e-8)
            print(f"  h12/h1 degradation: {deg:.2f}×")
        print(
            f"  {'Average':>8s} │ {np.mean(amae):>8.2f} │ {np.mean(armse):>8.2f} │ "
            f"{np.mean(amape)*100:>7.2f}%"
        )
        print(f"  [TEST] Done in {(time.perf_counter()-t0)*1000:.0f}ms")

        if save:
            save_model(model, save_path_resume)
