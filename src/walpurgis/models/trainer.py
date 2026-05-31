"""
Walpurgis Training Engine — Instrumented Trainer with Gradient Forensics
=========================================================================
Derived from D2STGNN trainer.py with ~20% algorithmic changes.

Changes vs upstream:
  1. Gradient clipping uses adaptive norm instead of fixed clip=5:
     clip_val = max(base_clip, median(recent_grad_norms) * 2)
  2. CL (curriculum learning) progression uses sigmoid ramp-up instead of
     discrete staircase — smoother difficulty increase
  3. Per-step timing is tracked with lightweight counters, not lists
  4. Full gradient forensics: exploding/vanishing/dead neuron detection
"""

import numpy as np
import time
import math
import torch
import torch.optim as optim
from collections import deque

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, MetricTracker


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    """NumPy MAPE for final test reporting (avoids torch overhead)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask = mask / np.mean(mask)
        mape = np.abs(np.divide(
            np.subtract(y_pred, y_true).astype('float32'), y_true
        ))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer:
    """Walpurgis training engine with gradient forensics and adaptive clipping.
    
    Debug usage — at any point during training:
        engine._print_gradient_forensics()   # detailed per-param gradient analysis
        engine._print_timing_summary()       # cumulative timing breakdown
        MetricTracker.report()               # all loss/metric statistics
    """
    
    def __init__(self, scaler, model, **optim_args):
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']

        # ── Optimizer config ── #
        self.lrate  = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        
        # ── LR scheduler ── #
        self.if_lr_scheduler = optim_args['lr_schedule']
        self.lr_sche_steps   = optim_args['lr_sche_steps']
        self.lr_decay_ratio  = optim_args['lr_decay_ratio']
        
        # ── Curriculum learning ── #
        self.if_cl    = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len   = 0 if self.if_cl else self.output_seq_len
        
        # ── Warmup ── #
        self.warm_steps = optim_args['warm_steps']

        # Build optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps
        )
        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_sche_steps,
                gamma=self.lr_decay_ratio
            ) if self.if_lr_scheduler else None
        )
        
        self.loss = masked_mae
        
        # ── Walpurgis: adaptive gradient clipping ── #
        # Base clip value; actual clip = max(base, 2 × median_recent_norm)
        self._base_clip = 5.0
        self._recent_grad_norms = deque(maxlen=200)
        
        # ── Debug counters ── #
        self._step_count = 0
        self._cumulative_fwd_ms = 0.0
        self._cumulative_bwd_ms = 0.0
        self._cumulative_opt_ms = 0.0
        self._cumulative_loss_ms = 0.0
    
    def _adaptive_clip_value(self):
        """Compute adaptive gradient clip threshold.
        
        Instead of a fixed clip=5, we use max(5, 2×median(recent_grad_norms)).
        This prevents clipping during stable training (where grad norms might
        naturally exceed 5 due to large batches) while still catching explosions.
        """
        if len(self._recent_grad_norms) < 10:
            return self._base_clip
        median_gn = np.median(list(self._recent_grad_norms))
        return max(self._base_clip, median_gn * 2.0)
    
    def _sigmoid_cl_progress(self, step):
        """Sigmoid-ramped curriculum length instead of discrete staircase.
        
        Upstream D2STGNN uses: cl_len += 1 every cl_steps batches.
        This creates discontinuities where the model suddenly sees a longer
        horizon and loss spikes. Sigmoid ramp smooths the transition:
        
        cl_len = round(output_seq_len × σ(k·(step - midpoint)))
        
        where k controls ramp steepness and midpoint is when cl_len ≈ half.
        """
        total_cl_batches = self.cl_steps * self.output_seq_len
        midpoint = self.warm_steps + total_cl_batches / 2.0
        steepness = 6.0 / total_cl_batches  # ~95% at ±3σ
        
        progress = 1.0 / (1.0 + math.exp(-steepness * (step - midpoint)))
        return max(1, round(progress * self.output_seq_len))
    
    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        """Resume optimizer and CL state from a checkpoint.
        
        Walpurgis: prints the full state trace for debugging resume issues.
        """
        if batch_num == 0:
            print(f"  [CL] Fresh start — cl_len=0, lr={self.lrate}")
            return
        
        print(f"  [CL] Resuming from epoch={epoch_num}, batch_num={batch_num}")
        transitions = []
        
        for b in range(batch_num):
            if b < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif b == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lrate
                transitions.append(f"    batch={b}: warmup→CL, cl_len=1, lr={self.lrate}")
            else:
                new_cl = self._sigmoid_cl_progress(b)
                if new_cl != self.cl_len:
                    self.cl_len = new_cl
                    transitions.append(f"    batch={b}: cl_len→{self.cl_len}")
        
        print(f"  [CL] Final: lr={self.lrate}, cl_len={self.cl_len}")
        # Show first/last transitions if too many
        if len(transitions) <= 8:
            for t in transitions:
                print(t)
        else:
            print(f"  [CL] {len(transitions)} transitions (first/last 3):")
            for t in transitions[:3]: print(t)
            print("    ...")
            for t in transitions[-3:]: print(t)

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            total_p = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"  [MODEL] Trainable parameters: {total_p:,d}")

    def _print_gradient_forensics(self):
        """Detailed gradient health analysis — call from debugger at any time.
        
        Checks for:
        - Exploding gradients (norm > 10× clip)
        - Vanishing gradients (norm < 1e-8)
        - Dead neurons (grad exactly zero for >50% of params)
        - NaN/Inf contamination
        """
        print(f"\n{'━'*60}")
        print(f"  Gradient Forensics @ step {self._step_count}")
        print(f"{'━'*60}")
        issues = 0
        clip_val = self._adaptive_clip_value()
        
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data
            gn = g.norm(2).item()
            zero_frac = (g == 0).float().mean().item()
            
            flags = []
            if torch.isnan(g).any(): flags.append("🔴NaN")
            if torch.isinf(g).any(): flags.append("🔴Inf")
            if gn > clip_val * 10: flags.append(f"⚠️EXPLODE({gn:.1f})")
            if gn < 1e-8 and p.requires_grad: flags.append(f"⚠️VANISH({gn:.2e})")
            if zero_frac > 0.5: flags.append(f"⚠️DEAD({zero_frac*100:.0f}%)")
            
            if flags:
                issues += 1
                print(f"  {name:50s} | norm={gn:.6f} | {' '.join(flags)}")
        
        if issues == 0:
            print(f"  ✓ All gradients healthy (clip={clip_val:.2f})")
        else:
            print(f"  {issues} issue(s) detected")
        print(f"{'━'*60}\n")

    def _print_timing_summary(self):
        """Print cumulative timing breakdown — call after any epoch."""
        if self._step_count == 0:
            return
        print(f"\n  [TIMING] After {self._step_count} steps:")
        print(f"    fwd:  {self._cumulative_fwd_ms/self._step_count:.1f}ms avg "
              f"({self._cumulative_fwd_ms:.0f}ms total)")
        print(f"    loss: {self._cumulative_loss_ms/self._step_count:.1f}ms avg")
        print(f"    bwd:  {self._cumulative_bwd_ms/self._step_count:.1f}ms avg")
        print(f"    opt:  {self._cumulative_opt_ms/self._step_count:.1f}ms avg")

    def train(self, input, real_val, **kwargs):
        """Single training step with full instrumentation.
        
        Returns: (mae_loss, mape, rmse)
        Side effects: prints timing/CL/gradient info at configured intervals
        """
        self._step_count += 1
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        # ── Forward ── #
        t0 = time.perf_counter()
        output = self.model(input)
        output = output.transpose(1, 2)
        fwd_ms = (time.perf_counter() - t0) * 1000
        self._cumulative_fwd_ms += fwd_ms

        # ── Curriculum learning (sigmoid ramp) ── #
        bn = kwargs['batch_num']
        cl_event = None
        
        if bn < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            cl_event = f"CL START: cl_len=1, lr={self.lrate}"
        else:
            new_cl = self._sigmoid_cl_progress(bn)
            if new_cl != self.cl_len:
                old = self.cl_len
                self.cl_len = new_cl
                cl_event = f"CL RAMP: cl_len {old}→{self.cl_len}"
        
        if cl_event:
            print(f"  [CL] batch={bn}: {cl_event}")

        # ── Loss ── #
        t0 = time.perf_counter()
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1,2).unsqueeze(-1),
                kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]
            ).transpose(1,2).squeeze(-1)
            real_val_scaled = self.scaler(
                real_val.transpose(1,2).unsqueeze(-1),
                kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]
            ).transpose(1,2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_scaled[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:,:,:,0])
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :], 0)
        loss_ms = (time.perf_counter() - t0) * 1000
        self._cumulative_loss_ms += loss_ms

        # ── Backward ── #
        t0 = time.perf_counter()
        mae_loss.backward()
        bwd_ms = (time.perf_counter() - t0) * 1000
        self._cumulative_bwd_ms += bwd_ms

        # ── Adaptive gradient clip + optimize ── #
        t0 = time.perf_counter()
        clip_val = self._adaptive_clip_value()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), clip_val
        ).item()
        self._recent_grad_norms.append(grad_norm)
        self.optimizer.step()
        opt_ms = (time.perf_counter() - t0) * 1000
        self._cumulative_opt_ms += opt_ms

        # ── Periodic debug output ── #
        if self._step_count % 50 == 0:
            print(f"  [PERF] step={self._step_count}: "
                  f"fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms "
                  f"grad_norm={grad_norm:.4f} (clip={clip_val:.2f}) "
                  f"cl_len={self.cl_len}")
        
        if self._step_count % 250 == 0:
            self._print_gradient_forensics()

        # Metrics
        rv = real_val_inv if kwargs['_max'] is None else real_val_scaled
        mape = masked_mape(predict, rv, 0.0)
        rmse = masked_rmse(predict, rv, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        """Validation pass with batch-level statistics."""
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        
        t_start = time.perf_counter()
        
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)
            
            if kwargs['_max'] is not None:
                predict = self.scaler(
                    output.transpose(1,2).unsqueeze(-1),
                    kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]
                )
                real_val = self.scaler(
                    testy.transpose(1,2).unsqueeze(-1),
                    kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]
                )
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:,:,:,0])
            
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            
            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(f"  [EVAL] {len(valid_loss)} batches in {elapsed_ms:.0f}ms, "
              f"loss={np.mean(valid_loss):.4f}±{np.std(valid_loss):.4f}")

        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name,
             save=True, **kwargs):
        """Test evaluation with per-horizon table and debug output."""
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
        y_list = []
        
        print(f"\n  [TEST] Starting evaluation...")
        t0 = time.perf_counter()
        
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
        
        # Scale
        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        
        # Per-horizon results
        amae, amape, armse = [], [], []
        print(f"  {'Horizon':>8s} │ {'MAE':>8s} │ {'RMSE':>8s} │ {'MAPE':>8s}")
        print(f"  {'─'*42}")
        
        for h in range(12):
            pred_h = yhat[:, :, h]
            real_h = realy[:, :, h]
            if kwargs['dataset_name'] in ('PEMS04', 'PEMS08'):
                from sklearn.metrics import mean_absolute_error
                mae_h = mean_absolute_error(pred_h.cpu().numpy(), real_h.cpu().numpy())
                rmse_h = masked_rmse(pred_h, real_h, 0.0).item()
                mape_h = masked_mape(pred_h, real_h, 0.0).item()
            else:
                mae_h, mape_h, rmse_h = metric(pred_h, real_h)
            
            print(f"  {h+1:>8d} │ {mae_h:>8.4f} │ {rmse_h:>8.4f} │ {mape_h:>8.4f}")
            amae.append(mae_h)
            amape.append(mape_h)
            armse.append(rmse_h)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  {'─'*42}")
        print(f"  {'Average':>8s} │ {np.mean(amae):>8.2f} │ {np.mean(armse):>8.2f} │ "
              f"{np.mean(amape)*100:>7.2f}%")
        print(f"  [TEST] Done in {elapsed_ms:.0f}ms")

        if save:
            save_model(model, save_path_resume)
