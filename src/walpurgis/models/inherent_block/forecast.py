import time

import torch
import torch.nn as nn


class Forecast(nn.Module):
    """Inherent-block forecast branch — autoregressive prediction via
    RNN + Transformer unrolling.

    Walpurgis notes:
    - The AR loop length = output_seq_len / gap - 1.  Each iteration
      invokes both the GRU cell and a Transformer forward, so this is
      the single most latency-sensitive component in InhBlock.
    - Tier recommendation: always HBM during training (hot loop).
    """

    _call_count = 0

    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args     = model_args

        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self._ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        print(f"[Walpurgis::InhForecast] init hidden_dim={hidden_dim} fk_dim={fk_dim} "
              f"output_seq_len={self.output_seq_len} gap={model_args['gap']} "
              f"ar_steps={self._ar_steps}")

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        Forecast._call_count += 1
        _verbose = (Forecast._call_count <= 3 or Forecast._call_count % 300 == 0)

        [batch_size, _, num_nodes, num_feat] = X.shape

        t0 = time.perf_counter()
        predict = [Z[-1, :, :].unsqueeze(0)]
        for step_i in range(self._ar_steps):
            # RNN
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)
            # Transformer
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
        ar_ms = (time.perf_counter() - t0) * 1000

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        if _verbose:
            print(f"[Walpurgis::InhForecast::forward] call#{Forecast._call_count} "
                  f"ar_steps={self._ar_steps} ar_loop={ar_ms:.3f}ms "
                  f"({ar_ms / max(self._ar_steps, 1):.3f}ms/step) "
                  f"output shape={list(predict.shape)} "
                  f"mean={predict.mean().item():.6f} std={predict.std().item():.6f}")

        return predict
