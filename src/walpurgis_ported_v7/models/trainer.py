import numpy as np
import torch
import torch.optim as optim
import sys

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

_DBG_TRAIN = ("--dbg-train" in sys.argv)

try:
    from torchinfo.torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(
            np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


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

        # 算法改动: 用 AdamW 替代 Adam — 解耦 weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)

        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_sche_steps,
                gamma=self.lr_decay_ratio)
            if self.if_lr_scheduler else None)

        self.loss = masked_mae

        # 算法改动: adaptive gradient clipping —
        # 初始 clip=5, 之后根据 EMA of grad norm 自适应调整
        self.clip_base = 5.0
        self._grad_norm_ema = 0.0
        self._grad_norm_alpha = 0.99

    def _adaptive_clip(self):
        """返回当前应使用的梯度裁剪阈值。"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1e9)  # 不实际裁剪, 只测量
        total_norm = total_norm.item()

        if self._grad_norm_ema == 0:
            self._grad_norm_ema = total_norm
        else:
            self._grad_norm_ema = (self._grad_norm_alpha * self._grad_norm_ema
                                   + (1 - self._grad_norm_alpha) * total_norm)

        # clip 上界 = max(base, 2 * EMA)
        clip_val = max(self.clip_base, 2.0 * self._grad_norm_ema)
        # 实际裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=clip_val)

        return total_norm, clip_val

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        for i in range(batch_num):
            if i < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif i == self.warm_steps:
                self.cl_len = 1
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lrate
            else:
                if ((i - self.warm_steps) % self.cl_steps == 0
                        and self.cl_len < self.output_seq_len):
                    self.cl_len += int(self.if_cl)
        print(f"resume from epoch{epoch_num}, lr={self.lrate}, "
              f"cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            if torchinfo_summary is not None:
                torchinfo_summary(self.model)
            param_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[DBG-TRAIN] Total trainable params: {param_count:,}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            print(f"======== CL start, lr reset to {self.lrate} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)

        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val = self.scaler.inverse_transform(real_val[:, :, :, 0])
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val[:, :self.cl_len, :], 0)

        loss = mae_loss
        loss.backward()

        # 算法改动: adaptive clip
        grad_norm, clip_used = self._adaptive_clip()

        if _DBG_TRAIN and kwargs['batch_num'] % 50 == 0:
            print(f"[DBG-TRAIN] batch={kwargs['batch_num']}  "
                  f"loss={loss.item():.5f}  cl_len={self.cl_len}  "
                  f"grad_norm={grad_norm:.4f}  clip={clip_used:.4f}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

        self.optimizer.step()

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
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
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()

            if _DBG_TRAIN and itera == 0:
                print(f"[DBG-TRAIN][VAL] first_batch  loss={loss:.5f}  "
                      f"pred_range=[{predict.min().item():.3f}, "
                      f"{predict.max().item():.3f}]")

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        from sklearn.metrics import mean_absolute_error

        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
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

        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[:realy.size(0), ...]

        assert torch.where(y_list == realy)

        if kwargs['_max'] is not None:
            realy = scaler(
                realy.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(
                yhat.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        amae, amape, armse = [], [], []

        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]

            if _DBG_TRAIN:
                with torch.no_grad():
                    print(f"[DBG-TRAIN][TEST] horizon={i+1}  "
                          f"pred_mean={pred.mean().item():.3f}  "
                          f"real_mean={real.mean().item():.3f}  "
                          f"pred_std={pred.std().item():.3f}")

            if (kwargs['dataset_name'] == 'PEMS04'
                    or kwargs['dataset_name'] == 'PEMS08'):
                mae = mean_absolute_error(
                    pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = metrics[0], metrics[1], metrics[2]

            log = ('Evaluate best model on test data for horizon {:d}, '
                   'Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        log = ('(On average over 12 horizons) '
               'MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}% |')
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
