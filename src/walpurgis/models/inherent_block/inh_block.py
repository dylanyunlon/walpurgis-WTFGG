import math
import time

import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

# ── Walpurgis tier-aware latency thresholds (ms) ──
_TIER_HBM_THRESH  = 3.0
_TIER_GDDR_THRESH = 1.0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal inherent signals.

    Walpurgis note: PE tensors are registered as buffers and tracked for
    tier placement — they are read-only after init so DRAM-resident is
    acceptable for inference, but training benefits from HBM co-location
    with the hidden states they will be added to.
    """

    _call_count = 0

    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        print(f"[Walpurgis::PositionalEncoding] init d_model={d_model} max_len={max_len} "
              f"pe_buffer shape={list(pe.shape)} dtype={pe.dtype}")

    def forward(self, X):
        PositionalEncoding._call_count += 1
        t0 = time.perf_counter()
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if PositionalEncoding._call_count <= 3 or PositionalEncoding._call_count % 500 == 0:
            print(f"[Walpurgis::PE::forward] call#{PositionalEncoding._call_count} "
                  f"input_len={X.size(0)} elapsed={elapsed_ms:.3f}ms "
                  f"output mean={X.mean().item():.6f} std={X.std().item():.6f}")
        return X


class InhBlock(nn.Module):
    """Inherent block — captures the inherent (non-diffusion) temporal patterns.

    Walpurgis adaptation: each sub-component (RNN, PE, Transformer, Forecast,
    Backcast) is individually timed per forward pass.  The block reports a
    tier-placement suggestion based on its aggregate latency:
        >=3 ms  → HBM   (hot path, high reuse)
        >=1 ms  → GDDR  (warm path)
        < 1 ms  → DRAM  (cold / infrequent)
    """

    _call_count = 0

    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        """Inherent block

        Args:
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of MSA. Defaults to 4.
            bias (bool, optional): if use bias. Defaults to True.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
        """
        super().__init__()
        self.num_feat   = hidden_dim
        self.hidden_dim = hidden_dim

        # inherent model
        self.pos_encoder        = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer          = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer  = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)

        # forecast branch
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # backcast branch
        self.backcast_fc    = nn.Linear(hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

        print(f"[Walpurgis::InhBlock] init hidden_dim={hidden_dim} num_heads={num_heads} "
              f"forecast_hidden_dim={forecast_hidden_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Walpurgis::InhBlock] total params={total_params:,} "
              f"({total_params * 4 / 1024:.1f} KB @ fp32)")

    def forward(self, hidden_inherent_signal):
        """Inherent block forward with per-stage Walpurgis profiling.

        Args:
            hidden_inherent_signal (torch.Tensor): [batch_size, seq_len, num_nodes, num_feat]

        Returns:
            backcast_seq_res: [batch_size, seq_len, num_nodes, hidden_dim]
            forecast_hidden:  [batch_size, seq_len'', num_nodes, forecast_hidden_dim]
        """
        InhBlock._call_count += 1
        _verbose = (InhBlock._call_count <= 3 or InhBlock._call_count % 300 == 0)
        timings = {}

        [batch_size, seq_len, num_nodes, num_feat] = hidden_inherent_signal.shape

        if _verbose:
            print(f"[Walpurgis::InhBlock::forward] call#{InhBlock._call_count} "
                  f"input shape=[{batch_size},{seq_len},{num_nodes},{num_feat}]")
            # NaN/Inf guard on input
            _nan = torch.isnan(hidden_inherent_signal).any().item()
            _inf = torch.isinf(hidden_inherent_signal).any().item()
            if _nan or _inf:
                print(f"  ⚠ INPUT HEALTH: nan={_nan} inf={_inf}")

        # ── RNN stage ──
        t0 = time.perf_counter()
        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        timings['rnn'] = (time.perf_counter() - t0) * 1000

        # Walpurgis: layer-norm RNN output before PE/MSA to stabilize deep stacks.
        # D2STGNN feeds raw RNN output directly, which can cause attention score
        # explosion when RNN output variance grows across decouple layers.
        rnn_std = hidden_states_rnn.std().item()
        if rnn_std > 5.0:  # empirical threshold for instability
            hidden_states_rnn = hidden_states_rnn / (rnn_std + 1e-8) * 2.0
            if _verbose:
                print(f"  [InhBlock] RNN output rescaled: std was {rnn_std:.4f}, "
                      f"normalized to ~2.0")

        # ── Positional Encoding ──
        t0 = time.perf_counter()
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)
        timings['pe'] = (time.perf_counter() - t0) * 1000

        # ── Multi-head Self-Attention with adaptive temperature ──
        # Walpurgis: scale Q/K before attention when feature variance is high.
        # This acts as an implicit temperature: prevents softmax saturation
        # in early training when embeddings haven't converged yet.
        t0 = time.perf_counter()
        attn_input = hidden_states_rnn
        input_var = attn_input.var().item()
        if input_var > 3.0:  # attention heads saturating
            temp_scale = (2.0 / (input_var + 1e-8)) ** 0.5
            attn_input = attn_input * temp_scale
            if _verbose:
                print(f"  [InhBlock] Attention temp scaling: var={input_var:.4f} "
                      f"→ scale={temp_scale:.4f}")
        hidden_states_inh = self.transformer_layer(attn_input, attn_input, attn_input)
        timings['msa'] = (time.perf_counter() - t0) * 1000

        # ── Forecast branch ──
        t0 = time.perf_counter()
        forecast_hidden = self.forecast_block(
            hidden_inherent_signal, hidden_states_rnn, hidden_states_inh,
            self.transformer_layer, self.rnn_layer, self.pos_encoder
        )
        timings['forecast'] = (time.perf_counter() - t0) * 1000

        # ── Backcast branch ──
        t0 = time.perf_counter()
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq      = self.backcast_fc(hidden_states_inh)
        backcast_seq_res  = self.residual_decompose(hidden_inherent_signal, backcast_seq)
        timings['backcast'] = (time.perf_counter() - t0) * 1000

        total_ms = sum(timings.values())

        if _verbose:
            # ── Tier placement heuristic ──
            tier = "HBM" if total_ms >= _TIER_HBM_THRESH else ("GDDR" if total_ms >= _TIER_GDDR_THRESH else "DRAM")
            print(f"  [InhBlock timing] "
                  f"rnn={timings['rnn']:.3f}ms  pe={timings['pe']:.3f}ms  "
                  f"msa={timings['msa']:.3f}ms  forecast={timings['forecast']:.3f}ms  "
                  f"backcast={timings['backcast']:.3f}ms  TOTAL={total_ms:.3f}ms  → tier={tier}")
            print(f"  [InhBlock output] backcast shape={list(backcast_seq_res.shape)} "
                  f"forecast shape={list(forecast_hidden.shape)}")
            # Output health
            for name, t in [("backcast", backcast_seq_res), ("forecast", forecast_hidden)]:
                _n = torch.isnan(t).any().item()
                _i = torch.isinf(t).any().item()
                if _n or _i:
                    print(f"  ⚠ OUTPUT {name}: nan={_n} inf={_i}")

        return backcast_seq_res, forecast_hidden
