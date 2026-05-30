"""
Walpurgis Diffusion Forecast — Auto-Regressive Prediction with Stability Guard
=================================================================================
Adapted from D2STGNN Forecast.

Algorithm changes:
  1. AR stability guard: if any AR step produces output with norm > 100x the
     initial state, we detach and re-scale to prevent gradient explosion
     through the AR chain. This is critical for long output sequences (gap>1).
  2. Residual connection from last history state to each AR step, weighted
     by 1/(step+1) — gives later AR steps a decaying "anchor" to ground truth.
  3. Full AR chain diagnostics: per-step norm tracking and early divergence detection.
"""

import time
import torch
import torch.nn as nn


class Forecast(nn.Module):
    """Diffusion forecast branch with AR stability guards.

    Generates future predictions by auto-regressively applying the
    localized ST convolution. Each step feeds the previous output
    back as input.
    """

    _call_count = 0

    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args
        self.hidden_dim = hidden_dim

        print(f"[Walpurgis::Forecast] init k_t={self.k_t} "
              f"output_seq_len={self.output_seq_len} "
              f"gap={model_args.get('gap', 1)} "
              f"hidden→forecast: {hidden_dim}→{forecast_hidden_dim}")

    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        Forecast._call_count += 1
        _verbose = (Forecast._call_count <= 3 or Forecast._call_count % 300 == 0)
        t0 = time.perf_counter()

        predict = []
        history = gated_history_data
        ar_norms = []

        # Initial prediction from last diffusion state
        init_state = hidden_states_dif[:, -1, :, :].unsqueeze(1)
        init_norm = init_state.norm().item()
        predict.append(init_state)
        ar_norms.append(init_norm)

        # Walpurgis: anchor state for residual AR stabilization
        anchor = init_state.detach()

        num_ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1
        diverged = False

        for step in range(num_ar_steps):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)

            ar_output = localized_st_conv(_1, dynamic_graph, static_graph)
            step_norm = ar_output.norm().item()

            # Walpurgis: AR stability guard
            # If output norm explodes (>100x initial), detach and rescale
            if step_norm > init_norm * 100 and init_norm > 0:
                scale_factor = init_norm * 10 / (step_norm + 1e-8)
                ar_output = ar_output.detach() * scale_factor
                diverged = True
                if _verbose:
                    print(f"    [Forecast] ⚠ AR step {step}: norm={step_norm:.1f} "
                          f">> init={init_norm:.1f}, rescaled by {scale_factor:.4f}")

            # Walpurgis: decaying residual anchor
            # Gives later AR steps a weakening reference to the initial state
            anchor_weight = 0.1 / (step + 1)
            ar_output = ar_output + anchor_weight * anchor

            predict.append(ar_output)
            ar_norms.append(ar_output.norm().item())

        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"    [Forecast] call#{Forecast._call_count}: "
                  f"{num_ar_steps} AR steps, elapsed={elapsed_ms:.2f}ms "
                  f"output={list(predict.shape)}")
            if ar_norms:
                print(f"    [Forecast] AR norms: init={ar_norms[0]:.2f} "
                      f"final={ar_norms[-1]:.2f} "
                      f"max={max(ar_norms):.2f} diverged={diverged}")

        return predict
