import math
import numpy as np
import torch
import torch.optim as optim
import sys

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

_DBG = ("--dbg" in sys.argv)

try:
    from torchinfo.torchinfo import summary as model_summary
except ImportError:
    model_summary = None

try:
    from sklearn.metrics import mean_absolute_error
except ImportError:
    mean_absolute_error = None


def _dp(tag, msg):
    if _DBG:
        print(f"[DBG][trainer][{tag}] {msg}", flush=True)


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

        # training strategy parameters
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

        # 算法改动: AdamW 替代 Adam — 解耦 weight decay
        # Adam 的 weight decay 实际上是 L2 reg 耦合在梯度里,
        # AdamW 把 weight decay 独立施加在参数更新后, 效果更好
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)

        self.lr_scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.lr_sche_steps,
                gamma=self.lr_decay_ratio)
            if self.if_lr_scheduler else None)

        self.loss = masked_mae

        # 算法改动: adaptive gradient clipping
        # 原版: 固定 clip_grad_norm_(params, 5)
        # 改为: 对每个参数按 fan-in 缩放 clip threshold
        #   clip_value = base_clip * max(1, sqrt(fan_in) / 10)
        # 大参数矩阵 (如 embedding) 允许更大梯度范数, 小参数更紧
        self.base_clip = 5

        _dp("init",
            f"optimizer=AdamW lr={self.lrate} wd={self.wdecay}  "
            f"cl={self.if_cl} warm_steps={self.warm_steps}")

    def _adaptive_clip(self):
        """算法改动: per-param adaptive gradient clipping"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                fan_in = p.shape[0] if p.dim() >= 1 else 1
                scale = max(1.0, math.sqrt(fan_in) / 10.0)
                param_clip = self.base_clip * scale
                pnorm = p.grad.data.norm(2).item()
                if pnorm > param_clip:
                    p.grad.data.mul_(param_clip / (pnorm + 1e-6))
                total_norm += pnorm ** 2
        total_norm = math.sqrt(total_norm)
        _dp("adaptive_clip", f"total_grad_norm={total_norm:.4f}")
        return total_norm

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
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
            print(f"resume training from epoch{epoch_num}, "
                  f"learn_rate={self.lrate}, cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            if model_summary is not None:
                try:
                    model_summary(self.model, input_data=None)
                except Exception:
                    pass
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if _DBG:
                        print(f"  {name}  {list(param.shape)}")
                    tmp = 1
                    for s in param.shape:
                        tmp *= s
                    parameter_num += tmp
            print(f"Parameter size: {parameter_num}")

    def _cl_schedule(self, batch_num):
        """算法改动: cosine warmup curriculum learning
        原版: 线性递增 cl_len (每 cl_steps 批加 1)
        改为: 用 cosine 曲线从 1 平滑增长到 output_seq_len
              progress = (batch_num - warm_steps) / total_cl_span
              cl_len = 1 + (output_seq_len - 1) * (1 - cos(pi * progress)) / 2
        前期增长慢 (简单样本训充分), 中期加速, 后期趋于饱和
        """
        if batch_num < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif batch_num == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print(f"======== Start curriculum learning... "
                  f"reset lr to {self.lrate} ========")
        else:
            if self.if_cl:
                elapsed = batch_num - self.warm_steps
                total_span = self.cl_steps * self.output_seq_len
                progress = min(elapsed / max(total_span, 1), 1.0)
                cos_val = (1.0 - math.cos(math.pi * progress)) / 2.0
                self.cl_len = int(
                    1 + (self.output_seq_len - 1) * cos_val)
                self.cl_len = min(self.cl_len, self.output_seq_len)
        _dp("cl_schedule",
            f"batch={batch_num} cl_len={self.cl_len}/{self.output_seq_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        self.print_model(**kwargs)

        output = self.model(input)
        output = output.transpose(1, 2)

        # curriculum learning
        self._cl_schedule(kwargs['batch_num'])

        # scale data and calculate loss
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

        # 算法改动: adaptive gradient clipping
        self._adaptive_clip()
        self.optimizer.step()

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        _dp("train_step",
            f"loss={mae_loss.item():.5f} mape={mape.item():.5f} "
            f"rmse={rmse.item():.5f} cl_len={self.cl_len}")
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

            print("test: {0}".format(loss), end='\r')

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        _dp("eval",
            f"val_loss={mvalid_loss:.5f} val_mape={mvalid_mape:.5f} "
            f"val_rmse={mvalid_rmse:.5f}")
        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader,
             scaler, model_name, save=True, **kwargs):
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
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(
                yhat.squeeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        amae = []
        amape = []
        armse = []

        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if (kwargs['dataset_name'] == 'PEMS04'
                    or kwargs['dataset_name'] == 'PEMS08'):
                if mean_absolute_error is not None:
                    mae = mean_absolute_error(
                        pred.cpu().numpy(), real.cpu().numpy())
                else:
                    mae = masked_mae(pred, real, 0.0).item()
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
                log = ('Evaluate best model on test data for '
                       'horizon {:d}, MAE: {:.4f}, RMSE: {:.4f}, '
                       'MAPE: {:.4f}')
                print(log.format(i + 1, mae, rmse, mape))
                amae.append(mae)
                amape.append(mape)
                armse.append(rmse)
            else:
                metrics = metric(pred, real)
                log = ('Evaluate best model on test data for '
                       'horizon {:d}, MAE: {:.4f}, RMSE: {:.4f}, '
                       'MAPE: {:.4f}')
                print(log.format(i + 1, metrics[0], metrics[2],
                                 metrics[1]))
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

        log = ('(On average over 12 horizons) '
               'MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}% |')
        print(log.format(
            np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
