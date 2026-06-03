"""
inh_block.py — v9 port
Algo delta:
  1. PositionalEncoding: 加性正弦PE → RoPE (旋转位置编码)
     对相对位置更敏感, 外推能力更好
  2. InhBlock backcast 加 learnable residual gate:
     gate = σ(FC(backcast)), output = gate * backcast + (1-gate) * input
  3. forward 中用 torch.utils.checkpoint 包裹 forecast,
     在长序列时节省显存
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast
from walpurgis_ported_v9 import _dbg

_TAG = "inh_blk"


# ── v9: Rotary Position Encoding (RoPE) ──

class RotaryPE(nn.Module):
    """
    RoPE: 对 query/key 的偶数/奇数维度对做旋转.
    这里实现为加在 RNN hidden states 上的变体:
    对 [S, B*N, D] 的序列, 将 D 维分成 D/2 对,
    每对乘以 (cos(θ), sin(θ)), θ = pos / 10000^(2i/D).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        half_d = d_model // 2
        freq = 1.0 / (10000.0 ** (torch.arange(0, half_d).float() / half_d))
        pos = torch.arange(max_len).float().unsqueeze(1)
        angles = pos * freq.unsqueeze(0)
        self.register_buffer('cos_cached', torch.cos(angles))  # [max_len, half_d]
        self.register_buffer('sin_cached', torch.sin(angles))

    def forward(self, X):
        # X: [S, B*N, D]
        S = X.shape[0]
        half = X.shape[-1] // 2
        x1, x2 = X[..., :half], X[..., half:]
        cos = self.cos_cached[:S, :half].unsqueeze(1)
        sin = self.sin_cached[:S, :half].unsqueeze(1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True,
                 forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        # v9: RoPE instead of sinusoidal PE
        self.pos_encoder = RotaryPE(hidden_dim)
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads,
                                                  model_args['dropout'], bias)

        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        # v9: learnable residual gate
        self.res_gate_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        B, S, N, D = hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        # v9: RoPE
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        hidden_states_inh = self.transformer_layer(
            hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # v9: gradient checkpoint for forecast to save memory
        forecast_hidden = grad_checkpoint(
            self.forecast_block,
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder,
            use_reentrant=False)

        hidden_states_inh = hidden_states_inh.reshape(S, B, N, D).transpose(0, 1)
        bc = self.backcast_fc(hidden_states_inh)

        # v9: learnable residual gate
        gate = torch.sigmoid(self.res_gate_fc(bc))
        bc_gated = gate * bc + (1.0 - gate) * hidden_inherent_signal

        _dbg(_TAG, f"res_gate∈[{gate.min().item():.3f},{gate.max().item():.3f}]  "
                    f"bc_norm={bc.norm(2).item():.4g}")

        backcast_res = self.residual_decompose(hidden_inherent_signal, bc_gated)
        return backcast_res, forecast_hidden
