"""Nebula trainer: LAMB optimizer + polynomial decay scheduler, log-cosh+quantile loss."""
import math, numpy as np, torch, torch.optim as optim, sys, os
from collections import deque
from ..utils.train import data_reshaper, save_model
from .losses import nebula_composite_loss, masked_mae, masked_rmse, masked_mape, metric
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class LAMB(optim.Optimizer):
    """Layer-wise Adaptive Moments (LAMB) optimizer.
    Combines Adam-style moment estimation with layer-wise trust ratio scaling.
    Better than Adam for large batch training; here used for stable convergence."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                # Moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                exp_avg_corrected = exp_avg / bc1
                exp_avg_sq_corrected = exp_avg_sq / bc2
                # Adam update direction
                adam_update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                # Weight decay
                if group['weight_decay'] != 0:
                    adam_update = adam_update + group['weight_decay'] * p
                # LAMB trust ratio: ||w|| / ||update||
                w_norm = p.norm(2).clamp(min=1e-6)
                u_norm = adam_update.norm(2).clamp(min=1e-6)
                trust_ratio = w_norm / u_norm
                # Clamp trust ratio for stability
                trust_ratio = trust_ratio.clamp(max=10.0)
                p.add_(adam_update, alpha=-group['lr'] * trust_ratio)
        return loss


class PolynomialDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial decay: lr = base_lr * (1 - step/total_steps)^power.
    Smoother decay than step-based scheduling."""
    def __init__(self, optimizer, total_steps, power=2.0, end_lr=1e-7, last_epoch=-1):
        self.total_steps = max(total_steps, 1)
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch, self.total_steps)
        decay = (1 - step / self.total_steps) ** self.power
        return [max(base_lr * decay, self.end_lr) for base_lr in self.base_lrs]


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val): mask = ~np.isnan(y_true)
        else: mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32'); mask /= max(np.mean(mask), 1e-8)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        return np.mean(np.nan_to_num(mask * mape)) * 100


class trainer:
    def __init__(self, scaler, model, **optim_args):
        self.model = model; self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args.get('print_model', False)
        self.lrate = optim_args['lrate']; self.wdecay = optim_args['wdecay']; self.eps = optim_args['eps']
        # Nebula: LAMB optimizer
        self.optimizer = LAMB(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        self.if_lr_scheduler = optim_args.get('lr_schedule', True)
        # Nebula: polynomial decay scheduler
        total_steps = optim_args.get('epochs', 100)
        self.lr_scheduler = PolynomialDecayScheduler(
            self.optimizer, total_steps=total_steps, power=2.0, end_lr=1e-7
        ) if self.if_lr_scheduler else None
        self.if_cl = optim_args.get('if_cl', True)
        self.cl_steps = optim_args.get('cl_steps', 1)
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args.get('warm_steps', 0)
        # Nebula: composite log-cosh + quantile loss
        self.loss = nebula_composite_loss
        self._grad_norms = deque(maxlen=100)
        self._base_clip = 5.0

    def _adaptive_clip(self):
        if len(self._grad_norms) < 10: return self._base_clip
        s = sorted(self._grad_norms)
        return max(s[int(len(s) * 0.95)], 1.0)

    def _sqrt_cl(self, batch_num):
        """Square-root curriculum: cl_len = min(ceil(sqrt(eff_steps)), output_len)."""
        if not self.if_cl or self.cl_steps < 1: return self.output_seq_len
        eff = max(0, batch_num - self.warm_steps)
        return min(max(1, int(math.ceil(math.sqrt(max(eff / max(self.cl_steps, 1), 0))))), self.output_seq_len)

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num > 0:
            self.cl_len = self._sqrt_cl(batch_num)
            print(f"[NEB] Resume: epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train(); self.optimizer.zero_grad()
        output = self.model(input).transpose(1, 2)
        bn = kwargs.get('batch_num', 0)
        self.cl_len = self.output_seq_len if bn < self.warm_steps else self._sqrt_cl(bn)
        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            rv = self.scaler(real_val.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:])
        else:
            predict = self.scaler.inverse_transform(output)
            rv = self.scaler.inverse_transform(real_val[:,:,:,0])
            loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:], 0)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        self._grad_norms.append(total_norm)
        clip_val = self._adaptive_clip()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        self.optimizer.step()
        if _NEB_DBG:
            lr = self.optimizer.param_groups[0]['lr']
            print(f"[NEB:train] loss={loss.item():.6f} grad={total_norm:.4f} clip={clip_val:.2f} lr={lr:.6f} cl={self.cl_len}", file=sys.stderr)
        with torch.no_grad():
            mape = masked_mape(predict, rv, 0.0); rmse = masked_rmse(predict, rv, 0.0)
        return loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        vl, vm, vr = [], [], []
        self.model.eval()
        for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            tx = data_reshaper(x, device); ty = data_reshaper(y, device)
            with torch.no_grad(): output = self.model(tx).transpose(1,2)
            if kwargs.get('_max') is not None:
                p = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
                r = self.scaler(ty.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            else:
                p = self.scaler.inverse_transform(output); r = self.scaler.inverse_transform(ty[:,:,:,0])
            vl.append(masked_mae(p,r,0.0).item()); vm.append(masked_mape(p,r,0.0).item()); vr.append(masked_rmse(p,r,0.0).item())
        return np.mean(vl), np.mean(vm), np.mean(vr)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        from sklearn.metrics import mean_absolute_error
        model.eval(); outputs = []; realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1,2); yl = []
        for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            tx = data_reshaper(x, device); ty = data_reshaper(y, device).transpose(1,2)
            with torch.no_grad(): preds = model(tx)
            outputs.append(preds); yl.append(ty)
        yhat = torch.cat(outputs, dim=0)[:realy.size(0),...]; yl = torch.cat(yl, dim=0)[:realy.size(0),...]
        if kwargs.get('_max') is not None:
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
        else:
            realy = scaler.inverse_transform(realy)[:,:,:,0]; yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:,:,i]; real = realy[:,:,i]
            dn = kwargs.get('dataset_name','')
            if dn in ('PEMS04','PEMS08'):
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy()); rmse = masked_rmse(pred,real,0.0).item(); mape = masked_mape(pred,real,0.0).item()
            else: mae, mape, rmse = metric(pred, real)
            print(f'Evaluate best model on test data for horizon {i+1:d}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}')
            amae.append(mae); amape.append(mape); armse.append(rmse)
        print(f'(On average over 12 horizons) Test MAE: {np.mean(amae):.2f} | Test RMSE: {np.mean(armse):.2f} | Test MAPE: {np.mean(amape)*100:.2f}% |')
        if save: save_model(model, save_path_resume)
