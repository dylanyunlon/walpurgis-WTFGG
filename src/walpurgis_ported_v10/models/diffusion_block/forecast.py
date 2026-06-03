import torch
import torch.nn as nn
import torch.nn.functional as F
from walpurgis_ported_v10 import _dbg

_TAG = "diffc"


class Forecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.gap = model_args['gap']

        # 改动3: FC 前加 LayerNorm — upstream 直接 Linear
        self.pre_ln = nn.LayerNorm(hidden_dim)
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)

        # 改动1: cosine 退火 dropout 参数
        self._drop_max = 0.15
        self._drop_min = 0.02

    def _cosine_drop_rate(self, step, total_steps):
        """从 drop_max 余弦退火到 drop_min."""
        if total_steps <= 1:
            return self._drop_min
        ratio = step / (total_steps - 1)
        return self._drop_min + 0.5 * (self._drop_max - self._drop_min) * (1 + torch.cos(torch.tensor(ratio * 3.14159)))

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        n_ar_steps = int(self.output_seq_len / self.gap) - 1

        _dbg(_TAG, "ar_start", n_steps=n_ar_steps, kt=self.k_t,
             history_len=history.shape[1])

        for step_i in range(n_ar_steps):
            recent = predict[-self.k_t:]
            if len(recent) < self.k_t:
                deficit = self.k_t - len(recent)
                # 改动2: 线性插值 padding — upstream 直接拼 history 尾部
                # 这里对最早的预测帧做 lerp 与 history 尾部混合
                tail = history[:, -deficit:, :, :]
                first_pred = recent[0]
                interp_parts = []
                for j in range(deficit):
                    w = (j + 1) / (deficit + 1)  # 0→1 线性权重
                    blended = torch.lerp(tail[:, j:j+1, :, :], first_pred, w)
                    interp_parts.append(blended)
                inp = torch.cat(interp_parts + recent, dim=1)
            else:
                inp = torch.cat(recent, dim=1)

            # 改动1: cosine 退火 dropout
            p_drop = self._cosine_drop_rate(step_i, n_ar_steps)
            if self.training and p_drop > 0:
                inp = F.dropout(inp, p=float(p_drop), training=True)

            next_h = localized_st_conv(inp, dynamic_graph, static_graph)
            predict.append(next_h)

            _dbg(_TAG, f"ar_step_{step_i}", drop_rate=p_drop, inp=inp, out=next_h)

        predict = torch.cat(predict, dim=1)

        # 改动3: LayerNorm before FC
        predict = self.pre_ln(predict)
        predict = self.forecast_fc(predict)

        _dbg(_TAG, "ar_done", predict=predict)
        return predict
