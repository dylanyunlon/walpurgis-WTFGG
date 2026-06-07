"""Eclipse InhBlock: sigmoid gated residual backcast, learnable PE phase."""
import math, torch, torch.nn as nn, sys, os
from ..decouple.residual_decomp import ResidualDecomp
from .inh_model import RNNLayer, TransformerLayer
from .forecast import Forecast
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.phase_offset = nn.Parameter(torch.zeros(d_model))  # learnable phase

    def forward(self, X):
        pe_shifted = self.pe[:X.size(0)] + self.phase_offset.unsqueeze(0).unsqueeze(0)
        X = X + pe_shifted
        return self.dropout(X)

class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim; self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # Sigmoid gated residual backcast (vs upstream single FC)
        self.backcast_val = nn.Linear(hidden_dim, hidden_dim)
        self.backcast_gate = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        B, S, N, F = hidden_inherent_signal.shape
        hs_rnn = self.rnn_layer(hidden_inherent_signal)
        hs_rnn = self.pos_encoder(hs_rnn)
        hs_inh = self.transformer_layer(hs_rnn, hs_rnn, hs_rnn)
        fk = self.forecast_block(hidden_inherent_signal, hs_rnn, hs_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hs_inh = hs_inh.reshape(S, B, N, F).transpose(0, 1)
        gate = torch.sigmoid(self.backcast_gate(hs_inh))
        bc = gate * self.backcast_val(hs_inh) + (1 - gate) * hs_inh  # gated residual
        bc_res = self.residual_decompose(hidden_inherent_signal, bc)
        if _ECL_DBG: print(f"[ECL:inhblk] gate_mean={gate.mean().item():.4f} fk_shape={list(fk.shape)}", file=sys.stderr)
        return bc_res, fk
