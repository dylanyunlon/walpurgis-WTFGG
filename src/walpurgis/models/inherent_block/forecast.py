"""
Walpurgis Inherent Forecast — AR Temporal Prediction with Teacher Forcing Decay
================================================================================
Adapted from D2STGNN inherent Forecast.

Algorithm changes:
  1. Teacher forcing decay: in early AR steps, mix in ground-truth RNN hidden
     state (from the encoder) to stabilize prediction. The mixing ratio
     decays over steps so later predictions are fully auto-regressive.
     This prevents error accumulation in long AR chains.
  2. RNN hidden state norm monitoring: if hidden state norm grows
     beyond a threshold, apply gradient-friendly re-scaling.
  3. Per-step debug output for AR chain diagnosis.
"""

import time
import torch
import torch.nn as nn


class Forecast(nn.Module):
    """Inherent forecast branch: AR prediction via GRU + Transformer.

    Walpurgis: teacher forcing decay for long-horizon stability.
    """

    _call_count = 0

    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.hidden_dim = hidden_dim

        print(f"[Walpurgis::InhForecast] init output_seq_len={self.output_seq_len} "
              f"gap={model_args.get('gap', 1)} "
              f"hidden→fk: {hidden_dim}→{fk_dim}")

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        Forecast._call_count += 1
        _verbose = (Forecast._call_count <= 3 or Forecast._call_count % 300 == 0)
        t0 = time.perf_counter()

        [batch_size, _, num_nodes, num_feat] = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        ar_norms = [predict[0].norm().item()]

        # Walpurgis: save initial RNN state for teacher forcing reference
        rnn_h_anchor = RNN_H[-1].detach()
        initial_rnn_norm = RNN_H.norm().item()

        num_ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1

        for step in range(num_ar_steps):
            # GRU step
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)

            # Walpurgis: teacher forcing decay
            # Mix in anchor state with decaying weight: w = 0.2 * exp(-0.5 * step)
            # Early steps get ~20% anchor (stability), later steps get ~0% (full AR)
            tf_weight = 0.2 * torch.exp(torch.tensor(-0.5 * step,
                                        device=_gru.device, dtype=_gru.dtype))
            _gru = (1.0 - tf_weight) * _gru + tf_weight * rnn_h_anchor.unsqueeze(0)

            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            # Walpurgis: RNN hidden state norm guard
            rnn_norm = RNN_H.norm().item()
            if rnn_norm > initial_rnn_norm * 50 and initial_rnn_norm > 0:
                # Rescale the entire RNN hidden state sequence
                scale = initial_rnn_norm * 10 / (rnn_norm + 1e-8)
                RNN_H = RNN_H * scale
                if _verbose:
                    print(f"    [InhForecast] ⚠ step {step}: RNN norm "
                          f"{rnn_norm:.1f} >> init {initial_rnn_norm:.1f}, rescaled")

            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)

            # Transformer step
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
            ar_norms.append(_Z.norm().item())

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"    [InhForecast] call#{Forecast._call_count}: "
                  f"{num_ar_steps} AR steps, elapsed={elapsed_ms:.2f}ms "
                  f"output={list(predict.shape)}")
            if len(ar_norms) > 1:
                growth = ar_norms[-1] / (ar_norms[0] + 1e-8)
                print(f"    [InhForecast] AR norms: first={ar_norms[0]:.2f} "
                      f"last={ar_norms[-1]:.2f} growth={growth:.2f}x")

        return predict
