import math
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from walpurgis import _dbg

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric

_TAG = "trainer"


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        denom = np.mean(mask)
        if denom == 0:
            denom = 1.0
        mask /= denom
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

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)

        # 改动2: warmup-cosine 调度 — upstream 用 MultiStepLR
        # warmup 阶段线性从 0 升到 lrate, 之后 cosine 退火
        total_steps = optim_args.get('epochs', 80) * optim_args.get('steps_per_epoch', 1000)

        def _lr_lambda(step):
            warmup = self.warm_steps
            if step < warmup:
                return max(step / max(warmup, 1), 1e-6)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _lr_lambda) if self.if_lr_scheduler else None

        self.loss = masked_mae

        # 改动1: 自适应 p90 梯度裁剪 — upstream 固定 clip=5
        self._grad_history = deque(maxlen=200)
        self._adaptive_clip = 5.0
        self._clip_update_freq = 100
        self._clip_percentile = 90

        # 改动4: 梯度 snapshot 存储
        self._grad_snapshots = []

    def _update_adaptive_clip(self):
        """改动1: 每 _clip_update_freq 步, 用历史梯度的 p90 更新 clip 阈值."""
        if len(self._grad_history) >= self._clip_update_freq:
            vals = sorted(self._grad_history)
            idx = int(len(vals) * self._clip_percentile / 100)
            self._adaptive_clip = max(vals[min(idx, len(vals)-1)], 0.5)
            _dbg(_TAG, "clip_update",
                 new_clip=torch.tensor(self._adaptive_clip),
                 n_samples=torch.tensor(float(len(vals))))

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        for step in range(batch_num):
            if step < self.warm_steps:
                self.cl_len = self.output_seq_len
            elif step == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lrate
            else:
                if (step - self.warm_steps) % self.cl_steps == 0 \
                        and self.cl_len < self.output_seq_len:
                    self.cl_len += int(self.if_cl)
        print(f"resume epoch={epoch_num}, lr={self.lrate}, cl_len={self.cl_len}")

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num']) == 0:
            total_p = 0
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    n = 1
                    for s in p.shape:
                        n *= s
                    total_p += n
            print(f"Parameter size: {total_p}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.print_model(**kwargs)

        output = self.model(input).transpose(1, 2)

        # 改动3: CL ramp 用 sigmoid 曲线 — upstream 用线性阶梯
        bn = kwargs['batch_num']
        if bn < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif bn == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lrate
            print(f"======== CL start, lr reset to {self.lrate} ========")
        else:
            if self.if_cl:
                # 改动3: sigmoid CL — 非线性渐进
                progress = (bn - self.warm_steps) / max(self.cl_steps * self.output_seq_len, 1)
                sigmoid_val = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
                self.cl_len = max(1, int(sigmoid_val * self.output_seq_len))

        # scale + loss
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(
                real_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
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

        # 改动1: 自适应梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self._adaptive_clip)
        self._grad_history.append(grad_norm.item())
        if bn % self._clip_update_freq == 0 and bn > 0:
            self._update_adaptive_clip()

        self.optimizer.step()

        # 改动4: 梯度 snapshot
        _dbg(_TAG, "train_step",
             loss=loss, grad_norm=grad_norm,
             clip=torch.tensor(self._adaptive_clip),
             cl_len=torch.tensor(float(self.cl_len)),
             lr=torch.tensor(self.optimizer.param_groups[0]['lr']))

        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)

            if kwargs['_max'] is not None:
                predict = self.scaler(
                    output.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
                real_v = self.scaler(
                    testy.transpose(1, 2).unsqueeze(-1),
                    kwargs["_max"][0, 0, 0, 0],
                    kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_v = self.scaler.inverse_transform(testy[:, :, :, 0])

            l = self.loss(predict, real_v, 0.0).item()
            mp = masked_mape(predict, real_v, 0.0).item()
            rm = masked_rmse(predict, real_v, 0.0).item()

            _dbg(_TAG, f"val_batch_{itera}", loss=torch.tensor(l))

            valid_loss.append(l)
            valid_mape.append(mp)
            valid_rmse.append(rm)

        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler,
             model_name, save=True, **kwargs):
        from sklearn.metrics import mean_absolute_error
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
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
        for i in range(12):
            pred_i = yhat[:, :, i]
            real_i = realy[:, :, i]
            ds = kwargs.get('dataset_name', '')
            if ds in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(
                    pred_i.cpu().numpy(), real_i.cpu().numpy())
                rmse = masked_rmse(pred_i, real_i, 0.0).item()
                mape = masked_mape(pred_i, real_i, 0.0).item()
            else:
                m = metric(pred_i, real_i)
                mae, mape, rmse = m[0], m[1], m[2]

            log = ('Horizon {:d} | MAE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)

        log = ('Avg 12h | MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%')
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        _dbg(_TAG, "test_done",
             avg_mae=torch.tensor(np.mean(amae)),
             avg_rmse=torch.tensor(np.mean(armse)))

        if save:
            save_model(model, save_path_resume)
