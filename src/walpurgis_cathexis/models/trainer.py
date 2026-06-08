"""
Cathexis Trainer — 算法改写 #10
upstream: Adam + MultiStepLR
cathexis: AdamW + gradient centralization + WarmupCosineAnnealing
"""
import numpy as np
import torch
import torch.optim as optim
from ..utils.train import data_reshaper, save_model
from .losses import winsorized_mae, masked_mae, masked_rmse, masked_mape, metric

class GradCentralizedAdamW(optim.AdamW):
    """AdamW with gradient centralization: center gradients before update"""
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.dim() > 1:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
        return super().step(closure)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup + cosine decay"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            ratio = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * ratio for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


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

        # Cathexis改写 #10: GradCentralized AdamW
        self.optimizer = GradCentralizedAdamW(self.model.parameters(),
                                              lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        # Cathexis改写: WarmupCosine scheduler
        total_steps = optim_args.get('epochs', 80)
        warmup_epochs = max(3, total_steps // 10)
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps=warmup_epochs,
            total_steps=total_steps, min_lr=1e-6) if self.if_lr_scheduler else None

        # Cathexis改写 #9: Winsorized loss
        self.loss = winsorized_mae
        self.loss_eval = masked_mae
        self.clip = 5
        self.perf = None
        try:
            from .. import PerfTimer
            self.perf = PerfTimer()
        except: pass

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0: return
        for _ in range(batch_num):
            if _ < self.warm_steps: self.cl_len = self.output_seq_len
            elif _ == self.warm_steps:
                self.cl_len = 1
                for pg in self.optimizer.param_groups: pg["lr"] = self.lrate
            else:
                if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                    self.cl_len += int(self.if_cl)

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        if self.perf: self.perf.start("forward")
        output = self.model(input)
        if self.perf: self.perf.stop("forward")
        output = output.transpose(1, 2)

        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for pg in self.optimizer.param_groups: pg["lr"] = self.lrate
        else:
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        if kwargs['_max'] is not None:
            predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            real_val = self.scaler(real_val.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_inv = self.scaler.inverse_transform(real_val[:,:,:,0])
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_inv[:, :self.cl_len, :])

        if self.perf: self.perf.start("backward")
        mae_loss.backward()
        if self.perf: self.perf.stop("backward")

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = masked_mape(predict, real_val_inv if kwargs['_max'] is None else real_val, 0.0)
        rmse = masked_rmse(predict, real_val_inv if kwargs['_max'] is None else real_val, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
        self.model.eval()
        for x, y in dataloader['val_loader'].get_iterator():
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            output = self.model(testx).transpose(1, 2)
            if kwargs['_max'] is not None:
                predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
                real_val = self.scaler(testy.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:,:,:,0])
            loss = self.loss_eval(predict, real_val, 0.0).item()
            valid_loss.append(loss)
            valid_mape.append(masked_mape(predict, real_val, 0.0).item())
            valid_rmse.append(masked_rmse(predict, real_val, 0.0).item())
        return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2)
        for x, y in dataloader['test_loader'].get_iterator():
            testx = data_reshaper(x, device)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0),...]
        if kwargs['_max'] is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
        else:
            realy = scaler.inverse_transform(realy)[:,:,:,0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred, real = yhat[:,:,i], realy[:,:,i]
            metrics = metric(pred, real)
            log = 'Horizon {:d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'
            print(log.format(i+1, metrics[0], metrics[2], metrics[1]))
            amae.append(metrics[0]); amape.append(metrics[1]); armse.append(metrics[2])
        print('Avg MAE: {:.2f} | RMSE: {:.2f} | MAPE: {:.2f}%'.format(np.mean(amae), np.mean(armse), np.mean(amape)*100))
        if save: save_model(model, save_path_resume)
