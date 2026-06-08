"""
Cascade trainer — 算法改写:
  1. ReduceLROnPlateau替代MultiStepLR (基于验证loss自适应降LR)
  2. 自适应梯度裁剪 (adaptive gradient clipping, AGC)
     clip阈值根据参数范数自适应调整
  3. 集成PerfTimer用于训练阶段计时
  4. 逐层深度门控loss正则化 (鼓励使用更少层)
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis_cascade.utils.train import (
    data_reshaper, save_model)
from .losses import (
    masked_mae, masked_rmse, masked_mape, metric,
    cascade_aware_loss)
from walpurgis_cascade import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    DepthGateTracker, SETracker, CascadeResidualTracker)


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(
            np.subtract(y_pred, y_true).astype('float32'),
            y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model = model         # init model
        self.scaler = scaler        # data scaler
        self.output_seq_len = optim_args['output_seq_len']  # output sequence length
        self.print_model_structure = optim_args['print_model']

        # training strategy parametes
        ## adam optimizer
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        ## learning rate scheduler
        self.if_lr_scheduler = optim_args['lr_schedule']
        self.lr_sche_steps = optim_args['lr_sche_steps']
        self.lr_decay_ratio = optim_args['lr_decay_ratio']
        ## curriculum learning
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        ## warmup
        self.warm_steps = optim_args['warm_steps']

        # Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)
        # Cascade特有: ReduceLROnPlateau (基于验证loss自适应)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_decay_ratio,
                patience=max(optim_args.get('patience', 10) // 3, 2),
                min_lr=self.lrate * 0.01
            ) if self.if_lr_scheduler else None)

        # loss: cascade-aware loss (Cascade特有)
        self.loss = masked_mae
        self.cascade_loss = cascade_aware_loss
        self._use_cascade_loss = True
        self.clip = 5             # gradient clip

        # Cascade特有: 自适应梯度裁剪参数
        self._agc_clip_factor = 0.01
        self._agc_eps = 1e-3

        # 诊断工具
        self.perf = PerfTimer()
        self.depth_tracker = DepthGateTracker(5)
        self.se_tracker = SETracker()
        self.cascade_tracker = CascadeResidualTracker(5)
        self._global_step = 0

    def _adaptive_gradient_clip(self):
        """Cascade特有: 自适应梯度裁剪 (AGC)
        clip阈值 = max(param_norm, eps) * clip_factor
        相比固定clip, AGC对不同量级的参数更公平
        """
        for p in self.model.parameters():
            if p.grad is None:
                continue
            p_norm = p.data.norm()
            g_norm = p.grad.data.norm()
            max_norm = max(p_norm.item(), self._agc_eps) * self._agc_clip_factor
            if g_norm.item() > max_norm:
                p.grad.data.mul_(max_norm / (g_norm.item() + 1e-8))

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
            for _ in range(batch_num):
                # curriculum learning
                if _ < self.warm_steps:   # warmupping
                    self.cl_len = self.output_seq_len
                elif _ == self.warm_steps:
                    # init curriculum learning
                    self.cl_len = 1
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lrate
                else:
                    # begin curriculum learning
                    if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                        self.cl_len += int(self.if_cl)
            print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.perf.start("forward")

        output = self.model(input)
        output = output.transpose(1, 2)
        self.perf.stop("forward")

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:   # warmupping
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            # init curriculum learning
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.lrate))
        else:
            # begin curriculum learning
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        # scale data and calculate loss
        self.perf.start("loss")
        if kwargs['_max'] is not None:  # traffic flow
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(real_val.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0]).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :])
        else:
            ## inverse transform for both predict and real value.
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(real_val[:, :, :, 0])
            ## Cascade特有: cascade-aware loss
            if self._use_cascade_loss:
                mae_loss = self.cascade_loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
            else:
                mae_loss = self.loss(predict[:, :self.cl_len, :], real_val_s[:, :self.cl_len, :], 0)

        # Cascade特有: 深度门控正则化 — 鼓励稀疏使用层
        depth_reg = torch.tensor(0.0, device=mae_loss.device)
        for gate_param in self.model.depth_gates:
            depth_reg = depth_reg + torch.sigmoid(gate_param)
        depth_reg = depth_reg * 0.001  # 正则化系数

        loss = mae_loss + depth_reg
        self.perf.stop("loss")
        self.perf.start("backward")
        loss.backward()

        # Cascade特有: 自适应梯度裁剪
        self._adaptive_gradient_clip()
        self.perf.stop("backward")

        # gradient clip (传统clip也保留作为安全网)
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # 诊断: 追踪深度门控
        for i, gate_param in enumerate(self.model.depth_gates):
            self.depth_tracker.record(
                i, torch.sigmoid(gate_param).item())

        # 诊断: 每N步dump
        current_lr = self.optimizer.param_groups[0]['lr']
        if _is_debug() and self._global_step % 20 == 0:
            dump_struct_state(
                f"train_step_{self._global_step}",
                loss=loss.item(),
                lr=current_lr,
                cl_len=self.cl_len,
                depth_reg=depth_reg.item(),
                predict_range=predict,
                real_val_range=real_val_s)
        self._global_step += 1

        # metrics
        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        # val a epoch
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = data_reshaper(x, device)
            testy = data_reshaper(y, device)
            # for dstgnn
            output = self.model(testx)
            output = output.transpose(1, 2)

            # scale data
            if kwargs['_max'] is not None:  # traffic flow
                ## inverse transform for both predict and real value.
                predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
                real_val = self.scaler(testy.transpose(1, 2).unsqueeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            else:
                predict = self.scaler.inverse_transform(output)
                real_val = self.scaler.inverse_transform(testy[:, :, :, 0])

            # metrics
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

        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        # test
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)
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

        assert torch.where(y_list == realy)

        # scale data
        if kwargs['_max'] is not None:  # traffic flow
            realy = scaler(realy.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
            yhat = scaler(yhat.squeeze(-1), kwargs["_max"][0, 0, 0, 0], kwargs["_min"][0, 0, 0, 0])
        else:
            realy = scaler.inverse_transform(realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)

        # summarize the results.
        amae = []
        amape = []
        armse = []

        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in ('PEMS04', 'PEMS08'):  # traffic flow dataset follows mae metric used in ASTGNN.
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i + 1, mae, rmse, mape))
                amae.append(mae)
                amape.append(mape)
                armse.append(rmse)
            else:       # traffic speed datasets follow the metrics released by GWNet and DCRNN.
                metrics = metric(pred, real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                print(log.format(i + 1, metrics[0], metrics[2], metrics[1]))
                amae.append(metrics[0])     # mae
                amape.append(metrics[1])    # mape
                armse.append(metrics[2])    # rmse

        log = '(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
