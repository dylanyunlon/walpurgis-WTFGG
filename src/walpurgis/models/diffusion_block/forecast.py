import time

import torch
import torch.nn as nn


class Forecast(nn.Module):
    """Diffusion-block forecast branch — autoregressive prediction via
    localized ST convolution on dynamic + static graphs.

    Walpurgis notes:
    - Each AR step invokes localized_st_conv which performs graph
      convolution — this is the heaviest per-step operation.
    - The history buffer concatenation in the warm-up phase (when
      predict length < k_t) causes variable-size allocations.
    - Tier recommendation: HBM (graph conv is compute-bound).
    """

    _call_count = 0

    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc    = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args     = model_args
        self._ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        print(f"[Walpurgis::DifForecast] init hidden_dim={hidden_dim} "
              f"forecast_hidden_dim={forecast_hidden_dim} k_t={self.k_t} "
              f"ar_steps={self._ar_steps}")

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        Forecast._call_count += 1
        _verbose = (Forecast._call_count <= 3 or Forecast._call_count % 300 == 0)

        t0 = time.perf_counter()
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for step_i in range(self._ar_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2  = history[:, -sub:, :, :]
                _1  = torch.cat([_2] + _1, dim=1)
            else:
                _1  = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        ar_ms = (time.perf_counter() - t0) * 1000

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)

        if _verbose:
            print(f"[Walpurgis::DifForecast::forward] call#{Forecast._call_count} "
                  f"ar_steps={self._ar_steps} ar_loop={ar_ms:.3f}ms "
                  f"({ar_ms / max(self._ar_steps, 1):.3f}ms/step) "
                  f"output shape={list(predict.shape)} "
                  f"mean={predict.mean().item():.6f} std={predict.std().item():.6f}")

        return predict
