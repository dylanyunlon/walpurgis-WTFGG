import torch
import torch.nn as nn
import sys

_DBG = ("--dbg" in sys.argv)


class Forecast(nn.Module):
    """算法改动: AR 预测时引入 history mixing
    原版: 自回归时纯用前一步的预测结果
    改为: 每步把 history 的最后几帧和预测帧做加权混合,
          混合权重随步数线性衰减 (越远离已知数据, history 贡献越小)
    防止自回归误差过快积累
    """

    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args
        self.total_ar_steps = int(
            self.output_seq_len / self.model_args['gap']) - 1
        # 算法改动: 可学习的 history mixing decay
        self.mix_decay = nn.Parameter(torch.tensor(0.9))

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        for step in range(self.total_ar_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)

            new_pred = localized_st_conv(_1, dynamic_graph, static_graph)

            # 算法改动: history mixing — 越往后越少用 history
            if step < self.total_ar_steps:
                decay = self.mix_decay ** (step + 1)
                history_tail = history[:, -1:, :, :]
                new_pred = (1.0 - decay) * new_pred + decay * history_tail

            predict.append(new_pred)

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)

        if _DBG:
            with torch.no_grad():
                print(f"[DBG][DifForecast] predict shape={list(predict.shape)}  "
                      f"mix_decay={self.mix_decay.item():.4f}  "
                      f"ar_steps={self.total_ar_steps}", flush=True)
        return predict
