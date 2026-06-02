"""
Auto-regressive forecast head for the inherent (temporal) branch.
Uses GRU + Transformer to predict future hidden states step by step.
"""
import sys
import torch
import torch.nn as nn

_DBG = ("--debug-inhfc" in sys.argv)


class Forecast(nn.Module):

    def __init__(self, hidden_dim, fk_dim, **kw):
        super().__init__()
        self.horizon = kw['seq_length']
        self.gap     = kw['gap']
        self.proj    = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, rnn_H, Z, transformer, rnn_layer, pos_enc):
        B, _, N, D = X.shape
        n_ar_steps = int(self.horizon / self.gap) - 1

        preds = [Z[-1, :, :].unsqueeze(0)]

        for si in range(n_ar_steps):
            # one GRU step
            gru_out = rnn_layer.gru_cell(preds[-1][0], rnn_H[-1]).unsqueeze(0)
            rnn_H = torch.cat([rnn_H, gru_out], dim=0)
            # optional positional encoding
            if pos_enc is not None:
                rnn_H = pos_enc(rnn_H)
            # transformer attention
            z_new = transformer(gru_out, K=rnn_H, V=rnn_H)
            preds.append(z_new)

            if _DBG and si < 2:
                print(f"[DBG:inhfc] ar_step={si}  "
                      f"gru_norm={gru_out.norm().item():.4f}  "
                      f"z_new_mean={z_new.mean().item():.4f}  "
                      f"rnn_H_len={rnn_H.shape[0]}")

        stacked = torch.cat(preds, dim=0)                    # (T', B*N, D)
        stacked = stacked.reshape(-1, B, N, D).transpose(0, 1)  # (B, T', N, D)
        out = self.proj(stacked)
        return out
