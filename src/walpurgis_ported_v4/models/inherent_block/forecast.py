import torch
import torch.nn as nn
import sys

_V4_DEBUG = True
_dbg_call_count = 0


def _dbg(tag, **kw):
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][InhForecast][{tag}]"]
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}|norm={v.detach().norm().item():.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" ".join(parts), file=sys.stderr)


class Forecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_length']
        self.model_args = model_args

        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

        # v4: dropout on AR loop predictions to regularize sequential generation
        self.ar_dropout = nn.Dropout(p=model_args.get('dropout', 0.1))

        # v4: scheduled sampling probability — decay from 1.0 toward 0 during training
        # controls teacher-forcing vs free-running ratio in AR loop
        self._ss_prob = 1.0  # initially all teacher-forcing

    def set_scheduled_sampling(self, prob):
        """Set scheduled sampling probability.
        prob=1.0 -> always teacher-force (early training)
        prob=0.0 -> always free-running  (late training / inference)
        """
        self._ss_prob = max(0.0, min(1.0, prob))

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        global _dbg_call_count
        _dbg_call_count += 1

        [batch_size, _, num_nodes, num_feat] = X.shape
        num_ar_steps = int(self.output_seq_len / self.model_args['gap']) - 1

        predict = [Z[-1, :, :].unsqueeze(0)]

        for step_i in range(num_ar_steps):
            # v4: scheduled sampling — stochastically decide whether to
            # feed ground truth (Z) or own prediction back into the loop
            if self.training and step_i < Z.shape[0] - 1 and torch.rand(1).item() < self._ss_prob:
                # teacher-forcing: use actual hidden state from encoder
                ar_input = Z[-(num_ar_steps - step_i), :, :].unsqueeze(0)
            else:
                # free-running: use own last prediction
                ar_input = predict[-1]

            # v4: apply dropout to AR input for regularization
            ar_input_reg = self.ar_dropout(ar_input)

            # RNN step
            _gru = rnn_layer.gru_cell(ar_input_reg[0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            # Positional Encoding
            if pe is not None:
                RNN_H = pe(RNN_H)

            # Transformer
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

            if _V4_DEBUG and _dbg_call_count <= 3 and step_i < 3:
                mode = "teacher" if (self.training and step_i < Z.shape[0] - 1 and self._ss_prob > 0.5) else "free"
                _dbg(f"AR-step-{step_i}", mode=mode,
                     pred_norm=_Z.detach().norm().item(),
                     rnn_h_len=RNN_H.shape[0])

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, num_nodes, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)

        if _V4_DEBUG and _dbg_call_count <= 5:
            _dbg("output", predict=predict, ss_prob=f"{self._ss_prob:.3f}")

        return predict
