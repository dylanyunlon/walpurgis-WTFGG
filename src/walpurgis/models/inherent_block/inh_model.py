"""
Walpurgis Temporal Sequence Models — RNN and Transformer Layers
================================================================
Derived from D2STGNN inh_model.py.

Change: RNN uses GRU instead of LSTM (fewer parameters, comparable
performance on short sequences). Transformer uses pre-norm (LayerNorm
before attention) instead of post-norm for more stable gradients.
"""

import torch
import torch.nn as nn
import time


class RNNLayer(nn.Module):
    """GRU-based temporal encoder.
    
    Upstream D2STGNN uses LSTM. GRU has 2 gates instead of 3,
    so ~25% fewer parameters with comparable sequence modeling.
    """
    
    _call_count = 0
    
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            batch_first=True, dropout=dropout
        )
        self._debug_on = True
    
    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        RNNLayer._call_count += 1
        out, _ = self.rnn(x)
        
        if self._debug_on and RNNLayer._call_count % 500 == 1:
            print(f"        [GRU #{RNNLayer._call_count}] "
                  f"in={list(x.shape)} out_norm={out.norm().item():.4f}")
        return out


class TransformerLayer(nn.Module):
    """Pre-norm Transformer encoder layer.
    
    Upstream D2STGNN uses post-norm (norm after residual). Pre-norm
    (norm before attention) provides more stable gradients, especially
    important when stacked inside decouple layers.
    """
    
    _call_count = 0
    
    def __init__(self, hidden_dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),  # GELU instead of ReLU for smoother gradients
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self._debug_on = True
    
    def forward(self, x):
        """x: [B*N, L, D] → [B*N, L, D]"""
        TransformerLayer._call_count += 1
        
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        
        if self._debug_on and TransformerLayer._call_count % 500 == 1:
            print(f"        [Transformer #{TransformerLayer._call_count}] "
                  f"out_norm={x.norm().item():.4f}")
        return x
