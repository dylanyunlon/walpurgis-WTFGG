"""
Forecast (inherent branch) — Umbra变体
适配Mamba SSM + ALiBi接口
  - Mamba SSM不需要位置编码参数
  - forecast步进使用Mamba的W_z/W_h辅助参数
  - ALiBi注意力的位置偏差自动处理
"""
import torch
import torch.nn as nn
from ... import _dbg


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer,
                rnn_layer):
        """
        Mamba SSM版forecast — 无PE参数
        Args:
            X: 原始输入 [B, T, N, D]
            RNN_H: Mamba SSM输出 (seq形式) [L, B*N, D]
            Z: ALiBi注意力输出 [L, B*N, D]
            transformer_layer: ALiBiAttentionLayer
            rnn_layer: MambaSSMLayer (含W_z, W_h)
        """
        batch_size, _, num_nodes, num_feat = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]

        _dbg("inh_forecast.start",
             f"steps={int(self.output_seq_len / self.model_args['gap']) - 1} "
             f"Z_last_norm={Z[-1].norm().item():.4f}",
             "inherent")

        for step_i in range(
                int(self.output_seq_len
                    / self.model_args['gap']) - 1):
            # Mamba SSM步进: 使用辅助线性层W_z, W_h
            # 模拟SSM的单步更新(简化版)
            prev = predict[-1][0]  # [B*N, D]
            z = torch.sigmoid(
                rnn_layer.W_z(
                    torch.cat([RNN_H[-1], prev], dim=-1)))
            h_tilde = torch.tanh(rnn_layer.W_h(prev))
            _ssm_step = ((1 - z) * RNN_H[-1]
                         + z * h_tilde).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _ssm_step], dim=0)
            # 注意: 不使用PE — ALiBi自动处理位置
            # ALiBi注意力一步
            _Z = transformer_layer(
                _ssm_step, K=RNN_H, V=RNN_H)
            predict.append(_Z)

            if step_i == 0:
                _dbg("inh_forecast.first_step_z_gate",
                     f"mean={z.mean().item():.4f} "
                     f"std={z.std().item():.4f}",
                     "inherent")

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(
            -1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        _dbg("inh_forecast.output",
             predict, "inherent")
        _dbg("inh_forecast.output_range",
             f"[{predict.min().item():.4f}, "
             f"{predict.max().item():.4f}]",
             "inherent")
        return predict
