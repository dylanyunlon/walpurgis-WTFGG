"""Eclipse trainer: AdamW + ReduceLROnPlateau, adaptive p95 clip, log CL."""
import math, numpy as np, torch, torch.optim as optim, sys, os
from collections import deque
from ..utils.train import data_reshaper, save_model
from .losses import tukey_biweight_loss, gradient_penalty, masked_mae, masked_rmse, masked_mape, metric
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        self.if_lr_scheduler = optim_args.get('lr_schedule', True)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6) if self.if_lr_scheduler else None
        self.if_cl = optim_args.get('if_cl', True)
        self.cl_steps = optim_args.get('cl_steps', 1)
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps = optim_args.get('warm_steps', 0)
        self.loss = tukey_biweight_loss
        self.grad_penalty_alpha = 0.01
        self._grad_norms = deque(maxlen=100)
        self._base_clip = 5.0

    def _adaptive_clip(self):
        if len(self._grad_norms) < 10: return self._base_clip
        s = sorted(self._grad_norms)
        return max(s[int(len(s)*0.95)], 1.0)

    def _log_cl(self, batch_num):
        if not self.if_cl or self.cl_steps < 1: return self.output_seq_len
        eff = max(0, batch_num - self.warm_steps)
        return min(1 + int(math.log2(1.0 + eff / max(self.cl_steps, 1)) * self.output_seq_len), self.output_seq_len)

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num > 0:
            self.cl_len = self._log_cl(batch_num)
            print(f"[ECL] Resume: epoch={epoch_num} lr={self.lrate} cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train(); self.optimizer.zero_grad()
        output = self.model(input).transpose(1, 2)
        bn = kwargs.get('batch_num', 0)
        self.cl_len = self.output_seq_len if bn < self.warm_steps else self._log_cl(bn)
        if kwargs.get('_max') is not None:
            predict = self.scaler(output.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            rv = self.scaler(real_val.transpose(1,2).unsqueeze(-1), kwargs["_max"][0,0,0,0], kwargs["_min"][0,0,0,0]).transpose(1,2).squeeze(-1)
            base_loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:])
        else:
            predict = self.scaler.inverse_transform(output)
            rv = self.scaler.inverse_transform(real_val[:,:,:,0])
            base_loss = self.loss(predict[:,:self.cl_len,:], rv[:,:self.cl_len,:], 0)
        gp = gradient_penalty(predict, alpha=self.grad_penalty_alpha)
        loss = base_loss + gp; loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        self._grad_norms.append(total_norm)
        clip_val = self._adaptive_clip()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        self.optimizer.step()
        if _ECL_DBG:
            lr = self.optimizer.param_groups[0]['lr']
            print(f"[ECL:train] loss={loss.item():.6f} base={base_loss.item():.6f} gp={gp.item():.6f} grad={total_norm:.4f} clip={clip_val:.2f} lr={lr:.6f} cl={self.cl_len}", file=sys.stderr)
        with torch.no_grad():
            mape = masked_mape(predict, rv, 0.0); rmse = masked_rmse(predict, rv, 0.0)
        return base_loss.item(), mape.item(), rmse.item()

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
