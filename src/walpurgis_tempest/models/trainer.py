"""Tempest trainer: Adan optimizer + OneCycleLR scheduler, sqrt curriculum learning.
Unlike upstream (Adam + MultiStepLR) and eclipse (AdamW + ReduceLROnPlateau + log CL),
Tempest uses Adan (Adaptive Nesterov Momentum, ICLR 2023) for faster convergence,
OneCycleLR for aggressive cosine warmup+decay, and sqrt-based curriculum learning
for smoother difficulty ramp-up."""
import math, numpy as np, torch, torch.optim as optim, sys, os
from collections import deque
from ..utils.train import data_reshaper, save_model
from .losses import focal_regression_loss, l2_smoothness_penalty, masked_mae, masked_rmse, masked_mape, metric
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val): mask = ~np.isnan(y_true)
        else: mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32'); mask /= max(np.mean(mask), 1e-8)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        return np.mean(np.nan_to_num(mask * mape)) * 100

class Adan(optim.Optimizer):
    """Adan optimizer: Adaptive Nesterov Momentum.
    From 'Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models'
    (Xie et al., ICLR 2023). Uses 1st, 2nd, and 3rd order gradient moments.
    Distinct from Adam (upstream) and AdamW (eclipse)."""
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.02):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']; b1, b2, b3 = group['betas']; eps = group['eps']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError('Adan does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)      # 1st moment
                    state['v'] = torch.zeros_like(p)      # gradient difference
                    state['n'] = torch.zeros_like(p)      # 2nd moment of (grad + b2*diff)
                    state['prev_grad'] = torch.zeros_like(p)
                m, v, n = state['m'], state['v'], state['n']
                prev_grad = state['prev_grad']
                state['step'] += 1
                # Gradient difference
                diff = grad - prev_grad
                # Update moments
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).add_(diff, alpha=1 - b2)
                n.mul_(b3).addcmul_(grad + b2 * diff, grad + b2 * diff, value=1 - b3)
                # Weight decay (decoupled)
                p.mul_(1 - lr * wd)
                # Update parameters
                denom = n.sqrt().add_(eps)
                p.addcdiv_(m + b2 * v, denom, value=-lr)
                state['prev_grad'] = grad.clone()
        return loss

class trainer:
    def __init__(self, scaler, model, **optim_args):
        self.model = model; self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args.get('print_model', False)
        self.lrate = optim_args['lrate']; self.wdecay = optim_args['wdecay']; self.eps = optim_args['eps']
        # Adan optimizer (vs upstream Adam, eclipse AdamW)
        self.optimizer = Adan(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        # OneCycleLR scheduler (vs upstream MultiStepLR, eclipse ReduceLROnPlateau)
        total_steps = optim_args.get('epochs', 3) * optim_args.get('steps_per_epoch', 100)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lrate * 3, total_steps=max(total_steps, 10),
            pct_start=0.3, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
        self.if_cl = optim_args.get('if_cl', True)
        self.cl_steps = optim_args.get('cl_steps', 1)
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args.get('warm_steps', 0)
        self.loss = focal_regression_loss
        self.smooth_alpha = 0.005
        self._grad_norms = deque(maxlen=100)
        self._clip = 5.0

    def _sqrt_cl(self, batch_num):
        """Sqrt curriculum learning: cl_len = min(1 + floor(sqrt(eff_steps)), max_len).
        Smoother than log (eclipse) and linear (upstream)."""
        if not self.if_cl or self.cl_steps < 1: return self.output_seq_len
        eff = max(0, batch_num - self.warm_steps)
        return min(1 + int(math.sqrt(eff / max(self.cl_steps, 1)) * 2), self.output_seq_len)

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num > 0:
            self.cl_len = self._sqrt_cl(batch_num)
            print(f"[TEM] Resume: epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train(); self.optimizer.zero_grad()
        output = self.model(input).transpose(1, 2)
        bn = kwargs.get('batch_num', 0)
        self.cl_len = self.output_seq_len if bn < self.warm_steps else self._sqrt_cl(bn)
        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            rv = self.scaler(real_val.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            base_loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:])
        else:
            predict = self.scaler.inverse_transform(output)
            rv = self.scaler.inverse_transform(real_val[:,:,:,0])
            base_loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:], 0)
        smooth = l2_smoothness_penalty(predict, alpha=self.smooth_alpha)
        loss = base_loss + smooth; loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        self._grad_norms.append(total_norm)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip)
        self.optimizer.step()
        # Step OneCycleLR per batch
        try:
            self.lr_scheduler.step()
        except Exception:
            pass  # scheduler may run out of steps
        if _TEM_DBG:
            lr = self.optimizer.param_groups[0]['lr']
            print(f"[TEM:train@trainer] loss={loss.item():.6f} focal={base_loss.item():.6f} smooth={smooth.item():.6f} "
                  f"grad={total_norm:.4f} lr={lr:.6f} cl={self.cl_len}", file=sys.stderr)
        with torch.no_grad():
            mape = masked_mape(predict, rv, 0.0); rmse = masked_rmse(predict, rv, 0.0)
        return base_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        vl, vm, vr = [], [], []
        self.model.eval()
        for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            tx = data_reshaper(x, device); ty = data_reshaper(y, device)
            with torch.no_grad(): output = self.model(tx).transpose(1, 2)
            if kwargs.get('_max') is not None:
                p = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
                r = self.scaler(ty.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0])
            else:
                p = self.scaler.inverse_transform(output); r = self.scaler.inverse_transform(ty[:,:,:,0])
            vl.append(masked_mae(p, r, 0.0).item()); vm.append(masked_mape(p, r, 0.0).item()); vr.append(masked_rmse(p, r, 0.0).item())
        return np.mean(vl), np.mean(vm), np.mean(vr)

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        from sklearn.metrics import mean_absolute_error
        model.eval(); outputs = []; realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 2); yl = []
        for _, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            tx = data_reshaper(x, device); ty = data_reshaper(y, device).transpose(1, 2)
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
            dn = kwargs.get('dataset_name', '')
            if dn in ('PEMS04', 'PEMS08'):
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item(); mape = masked_mape(pred, real, 0.0).item()
            else:
                mae, mape, rmse = metric(pred, real)
            print(f'Evaluate best model on test data for horizon {i+1:d}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}')
            amae.append(mae); amape.append(mape); armse.append(rmse)
        print(f'(On average over 12 horizons) Test MAE: {np.mean(amae):.2f} | Test RMSE: {np.mean(armse):.2f} | Test MAPE: {np.mean(amape)*100:.2f}% |')
        if save: save_model(model, save_path_resume)
