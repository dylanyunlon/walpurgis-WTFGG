"""
Walpurgis Inherent Block — Temporal Pathway with Positional Encoding
=====================================================================
Derived from D2STGNN inh_block.py.

Change: uses learnable positional encoding instead of fixed sinusoidal.
Small sequences (L≤12) don't benefit much from the sinusoidal inductive
bias, and learnable PE lets the model adapt to dataset-specific patterns.
"""

import math
import time

import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast

_TIER_HBM_MS = 3.0
_TIER_GDDR_MS = 1.0


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding — replaces fixed sinusoidal.
    
    For short sequences (L≤12 typical in traffic forecasting), learnable
    PE can capture dataset-specific temporal patterns that fixed sinusoidal
    encoding misses (e.g., periodic rush-hour structure).
    """
    
    _call_count = 0
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable instead of fixed sinusoidal
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        print(f"[Walpurgis::LearnablePE] d_model={d_model} max_len={max_len}")

    def forward(self, X):
        """X: [L, B*N, D] → [L, B*N, D]"""
        LearnablePositionalEncoding._call_count += 1
        seq_len = X.shape[0]
        X = X + self.pe[:seq_len]
        return self.dropout(X)


class InhBlock(nn.Module):
    """Inherent block: temporal modeling via RNN + Transformer + PE.
    
    Pipeline: input → PE → RNN → Transformer → residual decomp + forecast
    """
    
    _call_count = 0
    
    def __init__(self, hidden_dim, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.pe = LearnablePositionalEncoding(hidden_dim, dropout=model_args.get('dropout', 0.1))
        self.rnn = RNNLayer(hidden_dim)
        self.transformer = TransformerLayer(hidden_dim, n_heads=4)
        self.residual_decomp = ResidualDecomp(hidden_dim)
        self.forecast_branch = Forecast(
            hidden_dim, forecast_hidden_dim,
            output_seq_len=model_args['seq_length'],
            gap=model_args['gap']
        )
        self._debug_on = True

    def forward(self, dif_backcast_seq):
        """
        Args:
            dif_backcast_seq: [B, L, N, D] — residual from diffusion block
        Returns:
            inh_residual: [B, L, N, D] — residual for next layer
            inh_forecast: [B, N, forecast_dim] — temporal forecast embedding
        """
        InhBlock._call_count += 1
        t0 = time.perf_counter()
        verbose = self._debug_on and InhBlock._call_count % 500 == 1
        
        B, L, N, D = dif_backcast_seq.shape
        
        # Reshape for sequence models: [B, L, N, D] → [L, B*N, D]
        x = dif_backcast_seq.permute(1, 0, 2, 3).reshape(L, B * N, D)
        
        # Positional encoding
        x = self.pe(x)
        
        # Reshape for batch-first RNN: [B*N, L, D]
        x = x.permute(1, 0, 2)
        
        # RNN + Transformer
        x = self.rnn(x)
        x = self.transformer(x)
        
        # Reshape back: [B*N, L, D] → [B, L, N, D]
        x = x.reshape(B, N, L, D).permute(0, 2, 1, 3)
        
        # Forecast
        forecast_hidden = self.forecast_branch(x)
        
        # Residual decomposition
        residual = self.residual_decomp(dif_backcast_seq, x)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            tier = "HBM" if elapsed_ms > _TIER_HBM_MS else ("GDDR" if elapsed_ms > _TIER_GDDR_MS else "DRAM")
            print(f"      [InhBlock #{InhBlock._call_count}] {elapsed_ms:.2f}ms → {tier} | "
                  f"residual_norm={residual.norm().item():.4f}")
        
        return residual, forecast_hidden
