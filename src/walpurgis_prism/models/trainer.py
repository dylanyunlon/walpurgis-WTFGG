"""
Prism trainer — 算法改写:
  1. Mixup数据增强: 训练时对输入batch做线性插值增强
  2. Contrastive loss: 相邻节点embedding对比正则加入总loss
  3. AdamW优化器替代Adam (解耦权重衰减)
  4. OneCycleLR调度器: 单周期余弦退火
  5. 每N步打印对比学习温度和视角融合权重
"""
import numpy as np
import torch
import torch.optim as optim

from walpurgis_prism.utils.train import (
    data_reshaper, save_model)
from .losses import (
    masked_mae, masked_rmse, masked_mape, metric,
    contrastive_node_loss, mixup_masked_mae)
from walpurgis_prism import (
    _dbg, _is_debug, dump_struct_state, PerfTimer,
    ContrastiveTemperatureTracker, ViewFusionTracker)


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
        self.model = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']
        # optimizer params
        self.lrate = optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps = optim_args['eps']
        # curriculum learning
        self.if_cl = optim_args['if_cl']
        self.cl_steps = optim_args['cl_steps']
        self.cl_len = (0 if self.if_cl
                       else self.output_seq_len)
        # warmup
        self.warm_steps = optim_args['warm_steps']
        # Prism特有: AdamW优化器 (解耦权重衰减)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lrate,
            weight_decay=self.wdecay, eps=self.eps)
        # Prism特有: OneCycleLR调度器
        total_steps = optim_args.get(
            '_steps_per_epoch', 50) * optim_args.get(
            'epochs', 3)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lrate * 3,
                total_steps=max(total_steps, 1),
                pct_start=0.3,
                anneal_strategy='cos',
                final_div_factor=100))
        # loss
        self.loss = masked_mae
        self.clip = 5
        # Prism特有: Mixup参数
        self._mixup_alpha = 0.2
        self._use_mixup = True
        # Prism特有: 对比loss权重
        self._contrastive_weight = 0.01
        # 邻接矩阵 (从model_args获取)
        self._adj_ori = optim_args.get('adj_ori', None)
        # 诊断工具
        self.perf = PerfTimer()
        self.contrast_tracker = ContrastiveTemperatureTracker()
        self.view_tracker = ViewFusionTracker()
        self._global_step = 0

    def _apply_mixup(self, input_data, target_data):
        """Prism特有: Mixup数据增强
        对batch内样本做随机线性插值:
        x_mixed = lam * x_i + (1 - lam) * x_j
        y_mixed = lam * y_i + (1 - lam) * y_j
        """
        if not self._use_mixup or not self.model.training:
            return input_data, target_data, 1.0
        lam = np.random.beta(self._mixup_alpha,
                             self._mixup_alpha)
        lam = max(lam, 1.0 - lam)  # 确保lam >= 0.5
        batch_size = input_data.shape[0]
        index = torch.randperm(batch_size,
                               device=input_data.device)
        mixed_input = (lam * input_data +
                       (1 - lam) * input_data[index])
        mixed_target = (lam * target_data +
                        (1 - lam) * target_data[index])
        _dbg("mixup_lam", f"{lam:.4f}", "trainer")
        return mixed_input, mixed_target, lam

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
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
        print(f"resume from epoch {epoch_num}, "
              f"lr={self.lrate}, cl_len={self.cl_len}")

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        self.perf.start("forward")
        # Prism特有: Mixup增强
        mixed_input, mixed_val, lam = self._apply_mixup(
            input, real_val)
        output = self.model(mixed_input)
        output = output.transpose(1, 2)
        self.perf.stop("forward")
        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning, "
                  f"lr reset to {self.lrate} ========")
        else:
            if ((kwargs['batch_num'] - self.warm_steps)
                    % self.cl_steps == 0
                    and self.cl_len <= self.output_seq_len):
                self.cl_len += int(self.if_cl)
        # scale data and compute loss
        self.perf.start("loss")
        if kwargs['_max'] is not None:
            predict = self.scaler(
                output.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            real_val_s = self.scaler(
                mixed_val.transpose(1, 2).unsqueeze(-1),
                kwargs["_max"][0, 0, 0, 0],
                kwargs["_min"][0, 0, 0, 0]
            ).transpose(1, 2).squeeze(-1)
            mae_loss = self.loss(
                predict[:, :self.cl_len, :],
                real_val_s[:, :self.cl_len, :])
        else:
            predict = self.scaler.inverse_transform(output)
            real_val_s = self.scaler.inverse_transform(
                mixed_val[:, :, :, 0])
            # Prism特有: Mixup-aware MAE
            if lam < 1.0:
                mae_loss = mixup_masked_mae(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :],
                    lam, 0)
            else:
                mae_loss = self.loss(
                    predict[:, :self.cl_len, :],
                    real_val_s[:, :self.cl_len, :], 0)
        # Prism特有: 对比loss
        contrast_loss = torch.tensor(
            0.0, device=mae_loss.device)
        if self._contrastive_weight > 0:
            node_emb = self.model.compute_contrastive_embeddings()
            if self._adj_ori is not None:
                adj_np = self._adj_ori
                if isinstance(adj_np, torch.Tensor):
                    adj_np = adj_np.cpu().numpy()
                contrast_loss = contrastive_node_loss(
                    node_emb, adj_np)
        loss = mae_loss + self._contrastive_weight * contrast_loss
        self.perf.stop("loss")
        self.perf.start("backward")
        loss.backward()
        self.perf.stop("backward")
        # gradient clip
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip)
        self.optimizer.step()
        # OneCycleLR step (per batch)
        self.lr_scheduler.step()
        # 诊断: 追踪对比学习和视角融合
        current_lr = self.optimizer.param_groups[0]['lr']
        self.contrast_tracker.record(
            self._global_step,
            0.5 * np.exp(-0.001 * self._global_step),
            contrast_loss.item())
        # 追踪视角融合权重
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            first_layer = self.model.layers[0]
            if hasattr(first_layer, 'multi_view_fusion'):
                vw = torch.softmax(
                    first_layer.multi_view_fusion.view_logits,
                    dim=0).detach()
                self.view_tracker.record(
                    self._global_step,
                    vw[0].item(), vw[1].item(), vw[2].item())
        if _is_debug() and self._global_step % 20 == 0:
            dump_struct_state(
                f"train_step_{self._global_step}",
                loss=loss.item(),
                mae_loss=mae_loss.item(),
                contrast_loss=contrast_loss.item(),
                lr=current_lr,
                cl_len=self.cl_len,
                mixup_lam=lam,
                predict_range=predict,
                real_val_range=real_val_s)
        self._global_step += 1
        # metrics (用原始数据算, 不用mixup版本)
        mape = masked_mape(predict, real_val_s, 0.0)
        rmse = masked_rmse(predict, real_val_s, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name,
             **kwargs):
        valid_loss, valid_mape, valid_rmse = [], [], []
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
                real_val = self.scaler.inverse_transform(
                    testy[:, :, :, 0])
            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict, real_val, 0.0).item()
            rmse = masked_rmse(predict, real_val, 0.0).item()
            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        return mvalid_loss, mvalid_mape, mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader,
             scaler, model_name, save=True, **kwargs):
        model.eval()
        outputs = []
        realy = torch.Tensor(
            dataloader['y_test']).to(device)
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
        yhat = torch.cat(outputs, dim=0)[
            :realy.size(0), ...]
        y_list = torch.cat(y_list, dim=0)[
            :realy.size(0), ...]
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
            realy = scaler.inverse_transform(
                realy)[:, :, :, 0]
            yhat = scaler.inverse_transform(yhat)
        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat[:, :, i]
            real = realy[:, :, i]
            if kwargs.get('dataset_name') in (
                    'PEMS04', 'PEMS08'):
                from sklearn.metrics import (
                    mean_absolute_error)
                mae = mean_absolute_error(
                    pred.cpu().numpy(),
                    real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
            else:
                metrics = metric(pred, real)
                mae, mape, rmse = (metrics[0],
                                    metrics[1],
                                    metrics[2])
            log = ('Evaluate best model on test data '
                   'for horizon {:d}, Test MAE: {:.4f}, '
                   'Test RMSE: {:.4f}, Test MAPE: {:.4f}')
            print(log.format(i + 1, mae, rmse, mape))
            amae.append(mae)
            amape.append(mape)
            armse.append(rmse)
        log = ('(On average over 12 horizons) '
               'Test MAE: {:.2f} | Test RMSE: {:.2f} '
               '| Test MAPE: {:.2f}% |')
        print(log.format(
            np.mean(amae), np.mean(armse),
            np.mean(amape) * 100))
        if save:
            save_model(model, save_path_resume)
