"""
Auto-regressive forecast branch for the Inherent Block.
Generates future hidden states by iteratively running GRU + Transformer.
"""

import torch
import torch.nn as nn
import sys

_DBG_IHFCAST = ("--debug-ihfcast" in sys.argv) or False


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.horizon = model_args['seq_length']
        self.gap = model_args['gap']
        self.proj = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, rnn_states, attn_states,
                transformer_layer, rnn_layer, pos_encoder):
        """
        Parameters
        ----------
        X             : [B, L, N, D]     — raw inherent signal
        rnn_states    : [L, B*N, D]      — GRU hidden trajectory
        attn_states   : [L, B*N, D]      — Transformer output trajectory
        transformer_layer, rnn_layer, pos_encoder : reusable modules

        Returns
        -------
        [B, horizon//gap, N, fk_dim]
        """
        B, _, N, D = X.shape
        n_ar_steps = int(self.horizon / self.gap) - 1

        ar_preds = [attn_states[-1, :, :].unsqueeze(0)]    # seed with last Z

        for step in range(n_ar_steps):
            # one GRU step
            gru_h = rnn_layer.gru_cell(ar_preds[-1][0], rnn_states[-1])
            gru_h = gru_h.unsqueeze(0)                     # [1, B*N, D]
            rnn_states = torch.cat([rnn_states, gru_h], dim=0)

            # positional encoding on full RNN trajectory
            if pos_encoder is not None:
                rnn_states = pos_encoder(rnn_states)

            # single-step transformer attention
            z_new = transformer_layer(gru_h, K=rnn_states, V=rnn_states)
            ar_preds.append(z_new)

            if _DBG_IHFCAST:
                print(f"[DBG:ihfcast] AR step {step+1}/{n_ar_steps}  "
                      f"gru_h_norm={gru_h.norm().item():.4f}  "
                      f"z_norm={z_new.norm().item():.4f}")

        # stack and reshape: [steps, B*N, D] → [B, steps, N, D]
        stacked = torch.cat(ar_preds, dim=0)                # [steps, B*N, D]
        stacked = stacked.reshape(-1, B, N, D).transpose(0, 1)
        forecast = self.proj(stacked)

        if _DBG_IHFCAST:
            print(f"[DBG:ihfcast] final  shape={tuple(forecast.shape)}")
        return forecast
