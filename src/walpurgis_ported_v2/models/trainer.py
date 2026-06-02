"""
Training engine: handles forward/backward, curriculum learning,
learning rate scheduling, evaluation, and test-time metrics.
"""

import numpy as np
import torch
import torch.optim as optim
import sys

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

try:
    from torchinfo.torchinfo import summary as model_summary
except ImportError:
    model_summary = None

try:
    from sklearn.metrics import mean_absolute_error as sklearn_mae
except ImportError:
    sklearn_mae = None

_DBG_TRAINER = ("--debug-trainer" in sys.argv) or False


def _mape_np(y_true, y_pred, null_val=np.nan):
    """Numpy-level MAPE for reporting."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        err = np.abs((y_pred - y_true).astype('float32') / y_true)
        err = np.nan_to_num(mask * err)
        return np.mean(err) * 100


class trainer:
    """
    Orchestrates one experiment: optimizer, LR schedule, curriculum
    learning warm-up, gradient clipping, and metric computation.
    """

    def __init__(self, scaler, model, **optim_args):
        self.model  = model
        self.scaler = scaler
        self.out_len = optim_args['output_seq_len']
        self._print_structure = optim_args['print_model']

        # ── optimizer config ──
        self.base_lr    = optim_args['lrate']
        self.wd         = optim_args['wdecay']
        self.eps        = optim_args['eps']

        # ── LR scheduler ──
        self.use_lr_sched   = optim_args['lr_schedule']
        self._lr_milestones = optim_args['lr_sche_steps']
        self._lr_gamma      = optim_args['lr_decay_ratio']

        # ── curriculum learning ──
        self.use_cl     = optim_args['if_cl']
        self.cl_steps   = optim_args['cl_steps']
        self.cl_len     = 0 if self.use_cl else self.out_len

        # ── warm-up ──
        self.warm_steps = optim_args['warm_steps']

        # ── build optimizer & scheduler ──
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.base_lr,
            weight_decay=self.wd, eps=self.eps,
        )
        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self._lr_milestones, gamma=self._lr_gamma
            ) if self.use_lr_sched else None
        )

        # ── loss & gradient clip ──
        self.loss = masked_mae
        self.grad_clip = 5

        if _DBG_TRAINER:
            print(f"[DBG:trainer] init  base_lr={self.base_lr}  wd={self.wd}  "
                  f"cl={self.use_cl}  warm={self.warm_steps}  out_len={self.out_len}")

    # ───── resume helpers ─────

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        """Fast-forward CL state and LR when resuming from a checkpoint."""
        if batch_num == 0:
            return
        for step in range(batch_num):
            if step < self.warm_steps:
                self.cl_len = self.out_len
            elif step == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr
            else:
                if (step - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.out_len:
                    self.cl_len += int(self.use_cl)
        print(f"Resumed from epoch {epoch_num}: lr={self.base_lr}, cl_len={self.cl_len}")

    if_lr_scheduler = property(lambda self: self.use_lr_sched)

    # ───── model structure dump ─────

    def _maybe_print_model(self, batch_num):
        if self._print_structure and batch_num == 0 and model_summary is not None:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[INFO] Trainable parameters: {n_params:,}")
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(f"  {name}  {tuple(p.shape)}")

    # ───── single training step ─────

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        batch_num = kwargs['batch_num']
        self._maybe_print_model(batch_num)

        output = self.model(input).transpose(1, 2)

        # ── curriculum learning state machine ──
        if batch_num < self.warm_steps:
            self.cl_len = self.out_len
        elif batch_num == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr
            print(f"[CL] Warm-up done at batch {batch_num}. "
                  f"Reset LR to {self.base_lr}, cl_len=1")
        else:
            if (batch_num - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.out_len:
                self.cl_len += int(self.use_cl)

        # ── inverse-scale and compute loss ──
        if kwargs.get('_max') is not None:  # flow dataset
            pred = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(pred[:, :self.cl_len, :], real[:, :self.cl_len, :])
        else:                               # speed dataset
            pred = self.scaler.inverse_transform(output)
            real = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(pred[:, :self.cl_len, :], real[:, :self.cl_len, :], 0)

        mae_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        mape_val = masked_mape(pred, real, 0.0)
        rmse_val = masked_rmse(pred, real, 0.0)

        if _DBG_TRAINER and batch_num % 50 == 0:
            grad_norm = sum(
                p.grad.norm().item() for p in self.model.parameters()
                if p.grad is not None
            )
            print(f"[DBG:trainer] step {batch_num}  mae={mae_loss.item():.4f}  "
                  f"cl_len={self.cl_len}  grad_norm={grad_norm:.4f}  "
                  f"pred_range=[{pred.min().item():.3g},{pred.max().item():.3g}]")

        return mae_loss.item(), mape_val.item(), rmse_val.item()

    # ───── validation ─────

    def eval(self, device, dataloader, model_name, **kwargs):
        losses, mapes, rmses = [], [], []
        self.model.eval()

        for idx, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)

            if kwargs.get('_max') is not None:
                pred = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0]
                )
                real = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0]
                )
            else:
                pred = self.scaler.inverse_transform(output)
                real = self.scaler.inverse_transform(testy[:, :, :, 0])

            l = self.loss(pred, real, 0.0).item()
            m = masked_mape(pred, real, 0.0).item()
            r = masked_rmse(pred, real, 0.0).item()
            losses.append(l); mapes.append(m); rmses.append(r)

        avg_loss = np.mean(losses)
        avg_mape = np.mean(mapes)
        avg_rmse = np.mean(rmses)

        if _DBG_TRAINER:
            print(f"[DBG:trainer] eval  batches={len(losses)}  "
                  f"loss={avg_loss:.4f}  mape={avg_mape:.4f}  rmse={avg_rmse:.4f}")
        return avg_loss, avg_mape, avg_rmse

    # ───── test ─────

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        model.eval()
        all_preds = []
        all_y = []
        ground_truth = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)

        for x, y in dataloader['test_loader'].get_iterator():
            tx = data_reshaper(x, device)
            ty = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                p = model(tx)
            all_preds.append(p)
            all_y.append(ty)

        yhat = torch.cat(all_preds, dim=0)[:ground_truth.size(0), ...]
        y_cat = torch.cat(all_y, dim=0)[:ground_truth.size(0), ...]
        assert torch.where(y_cat == ground_truth)

        # inverse scale
        if kwargs.get('_max') is not None:
            ground_truth = scaler(ground_truth.squeeze(-1),
                                  kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1),
                          kwargs['_max'][0, 0, 0, 0], kwargs['_min'][0, 0, 0, 0])
        else:
            ground_truth = scaler.inverse_transform(ground_truth)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        # per-horizon metrics
        all_mae, all_mape, all_rmse = [], [], []
        ds = kwargs.get('dataset_name', '')
        is_flow = ds in ('PEMS04', 'PEMS08')

        for h in range(12):
            p_h = yhat[:, :, h]
            r_h = ground_truth[:, :, h]

            if is_flow and sklearn_mae is not None:
                mae_h  = sklearn_mae(p_h.cpu().numpy(), r_h.cpu().numpy())
                rmse_h = masked_rmse(p_h, r_h, 0.0).item()
                mape_h = masked_mape(p_h, r_h, 0.0).item()
            else:
                mae_h, mape_h, rmse_h = metric(p_h, r_h)

            print(f'Horizon {h+1:2d} | MAE: {mae_h:.4f} | RMSE: {rmse_h:.4f} | MAPE: {mape_h:.4f}')
            all_mae.append(mae_h)
            all_mape.append(mape_h)
            all_rmse.append(rmse_h)

        print(f'\n(Average 12h)  MAE: {np.mean(all_mae):.2f}  '
              f'RMSE: {np.mean(all_rmse):.2f}  '
              f'MAPE: {np.mean(all_mape)*100:.2f}%')

        if _DBG_TRAINER:
            print(f"[DBG:trainer] test done  yhat={tuple(yhat.shape)}  "
                  f"gt={tuple(ground_truth.shape)}")

        if save:
            save_model(model, save_path_resume)
