import torch
import torch.nn as nn
import sys

_DBG_INH_FK = ("--dbg-inh-fk" in sys.argv)


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

        # 算法改动: 指数衰减基数 — 用于自回归步进的逐步衰减
        self.ar_gamma = nn.Parameter(torch.tensor(0.98))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, num_nodes, num_feat] = X.shape
        predict = [Z[-1, :, :].unsqueeze(0)]

        num_ar = int(self.output_seq_len / self.model_args['gap']) - 1

        for step_i in range(num_ar):
            _gru = rnn_layer.gru_cell(
                predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            # 算法改动: 只在 RNN_H 长度不太长时才做 PE
            # 当 RNN_H 已经超出 PE 的 max_len 预设范围时, skip PE
            # 这避免了原版中可能的越界或无效 PE
            if pe is not None and RNN_H.shape[0] <= 5000:
                RNN_H = pe(RNN_H)

            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)

            # 算法改动: 指数衰减
            decay = torch.clamp(self.ar_gamma, 0.8, 1.0) ** (step_i + 1)
            _Z = _Z * decay

            if _DBG_INH_FK:
                with torch.no_grad():
                    print(f"[DBG-INH-FK] step {step_i}/{num_ar}  "
                          f"decay={decay.item():.5f}  "
                          f"Z_norm={_Z.norm().item():.4f}  "
                          f"RNN_H_len={RNN_H.shape[0]}")

            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        if _DBG_INH_FK:
            with torch.no_grad():
                print(f"[DBG-INH-FK] final shape={list(predict.shape)}  "
                      f"range=[{predict.min().item():.4f}, {predict.max().item():.4f}]")
        return predict
