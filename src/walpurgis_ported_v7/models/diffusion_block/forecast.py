import torch
import torch.nn as nn
import sys

_DBG_DIF_FK = ("--dbg-dif-fk" in sys.argv)


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args

        # 算法改动: 给 AR 步进的每一步加一个可学习的 step-decay 因子
        # 越远的预测步信号越弱, 防止长步累积误差
        ar_steps = max(int(self.output_seq_len / self.model_args['gap']) - 1, 1)
        self.step_decay = nn.Parameter(torch.ones(ar_steps) * 0.95)

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        num_ar = int(self.output_seq_len / self.model_args['gap']) - 1

        for step_i in range(num_ar):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)

            new_pred = localized_st_conv(_1, dynamic_graph, static_graph)

            # 算法改动: step-wise decay
            decay = torch.sigmoid(self.step_decay[step_i])
            new_pred = new_pred * decay

            if _DBG_DIF_FK:
                with torch.no_grad():
                    print(f"[DBG-DIF-FK] AR step {step_i}/{num_ar}  "
                          f"decay={decay.item():.4f}  "
                          f"pred_norm={new_pred.norm().item():.4f}")

            predict.append(new_pred)

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)

        if _DBG_DIF_FK:
            with torch.no_grad():
                print(f"[DBG-DIF-FK] final_predict shape={list(predict.shape)}  "
                      f"range=[{predict.min().item():.4f}, {predict.max().item():.4f}]")

        return predict
