"""
Trainer: optimizer, curriculum learning, train/eval/test loops.
Ported with per-batch debug probes for gradient & loss inspection.
"""
import sys
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

_DBG = ("--debug-trainer" in sys.argv)


def _masked_mape_np(y_true, y_pred, null_val=np.nan):
    """Numpy MAPE (percentage) for quick reference prints."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        pct = np.abs((y_pred - y_true).astype('float32') / y_true)
        pct = np.nan_to_num(mask * pct)
        return np.mean(pct) * 100


class trainer:

    def __init__(self, scaler, model, **o):
        self.model  = model
        self.scaler = scaler
        self.out_len = o['output_seq_len']

        # ── optimizer ──
        self.lrate  = o['lrate']
        self.wdecay = o['wdecay']
        self.eps    = o['eps']
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)

        # ── lr scheduler ──
        self.if_lr_scheduler = o['lr_schedule']
        if self.if_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=o['lr_sche_steps'],
                gamma=o['lr_decay_ratio'])
        else:
            self.lr_scheduler = None

        # ── curriculum learning ──
        self.if_cl    = o['if_cl']
        self.cl_steps = o['cl_steps']
        self.cl_len   = 0 if self.if_cl else self.out_len

        # ── warmup ──
        self.warm_steps = o['warm_steps']

        # ── loss & clip ──
        self.loss = masked_mae
        self.clip = 5

        self.print_model_structure = o['print_model']

        if _DBG:
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[DBG:trainer] init  lr={self.lrate}  "
                  f"cl_steps={self.cl_steps}  warm={self.warm_steps}  "
                  f"trainable_params={n_p}")

    # ── resume helper ──

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        for b in range(batch_num):
            if b < self.warm_steps:
                self.cl_len = self.out_len
            elif b == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.lrate
            else:
                if (b - self.warm_steps) % self.cl_steps == 0 \
                        and self.cl_len < self.out_len:
                    self.cl_len += int(self.if_cl)
        print(f"Resumed from epoch {epoch_num}: "
              f"lr={self.lrate}  cl_len={self.cl_len}")

    # ── single train step ──

    def train(self, x, y_real, **kw):
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(x).transpose(1, 2)
        bn = kw['batch_num']

        # curriculum learning state machine
        if bn < self.warm_steps:
            self.cl_len = self.out_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.lrate
            print(f"═══ Curriculum learning started, lr reset to {self.lrate} ═══")
        else:
            if (bn - self.warm_steps) % self.cl_steps == 0 \
                    and self.cl_len <= self.out_len:
                self.cl_len += int(self.if_cl)

        # inverse-transform & loss
        if kw.get('_max') is not None:
            hi = kw['_max'][0, 0, 0, 0]
            lo = kw['_min'][0, 0, 0, 0]
            pred_inv = self.scaler(
                pred.transpose(1, 2).unsqueeze(-1), hi, lo
            ).transpose(1, 2).squeeze(-1)
            real_inv = self.scaler(
                y_real.transpose(1, 2).unsqueeze(-1), hi, lo
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(pred_inv[:, :self.cl_len, :],
                                 real_inv[:, :self.cl_len, :])
        else:
            pred_inv = self.scaler.inverse_transform(pred)
            real_inv = self.scaler.inverse_transform(y_real[:, :, :, 0])
            mae_loss = self.loss(pred_inv[:, :self.cl_len, :],
                                 real_inv[:, :self.cl_len, :], 0)

        mae_loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape_v = masked_mape(pred_inv, real_inv, 0.0)
        rmse_v = masked_rmse(pred_inv, real_inv, 0.0)

        if _DBG and bn % 50 == 0:
            grad_norm = sum(p.grad.norm().item()
                           for p in self.model.parameters()
                           if p.grad is not None)
            print(f"[DBG:trainer] batch={bn}  mae={mae_loss.item():.4f}  "
                  f"cl_len={self.cl_len}  grad_norm={grad_norm:.2f}  "
                  f"pred_range=[{pred_inv.min().item():.2f},"
                  f"{pred_inv.max().item():.2f}]")

        return mae_loss.item(), mape_v.item(), rmse_v.item()

    # ── validation ──

    def eval(self, device, dataloader, model_name, **kw):
        losses, mapes, rmses = [], [], []
        self.model.eval()

        for it, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            tx = data_reshaper(x, device)
            ty = data_reshaper(y, device)
            out = self.model(tx).transpose(1, 2)

            if kw.get('_max') is not None:
                hi = kw['_max'][0, 0, 0, 0]
                lo = kw['_min'][0, 0, 0, 0]
                p = self.scaler(out.transpose(1, 2).unsqueeze(-1), hi, lo)
                r = self.scaler(ty.transpose(1, 2).unsqueeze(-1), hi, lo)
            else:
                p = self.scaler.inverse_transform(out)
                r = self.scaler.inverse_transform(ty[:, :, :, 0])

            losses.append(self.loss(p, r, 0.0).item())
            mapes.append(masked_mape(p, r, 0.0).item())
            rmses.append(masked_rmse(p, r, 0.0).item())

        ml, mm, mr = np.mean(losses), np.mean(mapes), np.mean(rmses)
        if _DBG:
            print(f"[DBG:trainer] eval  val_mae={ml:.4f}  "
                  f"val_mape={mm:.4f}  val_rmse={mr:.4f}")
        return ml, mm, mr

    # ── test ──

    @staticmethod
    def test(model, ckpt_path, device, dataloader, scaler,
             model_name, save=True, **kw):
        model.eval()
        all_preds = []
        all_y = []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)

        for it, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            tx = data_reshaper(x, device)
            ty = data_reshaper(y, device).transpose(1, 2)
            with torch.no_grad():
                p = model(tx)
            all_preds.append(p)
            all_y.append(ty)

        yhat  = torch.cat(all_preds, dim=0)[:realy.size(0), ...]
        y_cat = torch.cat(all_y,     dim=0)[:realy.size(0), ...]
        assert torch.where(y_cat == realy)

        # inverse transform
        if kw.get('_max') is not None:
            hi = kw['_max'][0, 0, 0, 0]
            lo = kw['_min'][0, 0, 0, 0]
            realy = scaler(realy.squeeze(-1), hi, lo)
            yhat  = scaler(yhat.squeeze(-1),  hi, lo)
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat  = scaler.inverse_transform(yhat)

        # per-horizon metrics
        acc_mae, acc_mape, acc_rmse = [], [], []
        ds = kw.get('dataset_name', '')

        for h in range(12):
            p_h = yhat[:, :, h]
            r_h = realy[:, :, h]

            if ds in ('PEMS04', 'PEMS08'):
                mae_h  = mean_absolute_error(p_h.cpu().numpy(), r_h.cpu().numpy())
                rmse_h = masked_rmse(p_h, r_h, 0.0).item()
                mape_h = masked_mape(p_h, r_h, 0.0).item()
            else:
                mae_h, mape_h, rmse_h = metric(p_h, r_h)

            print(f"  horizon {h+1:2d}  MAE={mae_h:.4f}  "
                  f"RMSE={rmse_h:.4f}  MAPE={mape_h:.4f}")
            acc_mae.append(mae_h)
            acc_mape.append(mape_h)
            acc_rmse.append(rmse_h)

        print(f"  ── avg 12h  MAE={np.mean(acc_mae):.2f}  "
              f"RMSE={np.mean(acc_rmse):.2f}  "
              f"MAPE={np.mean(acc_mape)*100:.2f}%")

        if _DBG:
            print(f"[DBG:trainer] test complete  "
                  f"yhat={tuple(yhat.shape)}  realy={tuple(realy.shape)}")

        if save:
            save_model(model, ckpt_path)
