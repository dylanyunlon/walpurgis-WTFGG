import time

import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention


class RNNLayer(nn.Module):
    """GRU-based recurrent layer for inherent temporal pattern extraction.

    Walpurgis notes:
    - The GRU cell is unrolled over seq_len steps; each step's hidden state
      is a candidate for tier migration if the node count is large.
    - Per-step latency is tracked to identify whether the unrolled loop
      dominates InhBlock cost (if so, HBM placement is critical).
    """

    _call_count = 0

    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        print(f"[Walpurgis::RNNLayer] init hidden_dim={hidden_dim} "
              f"gru_cell params={sum(p.numel() for p in self.gru_cell.parameters()):,}")

    def forward(self, X):
        RNNLayer._call_count += 1
        _verbose = (RNNLayer._call_count <= 3 or RNNLayer._call_count % 500 == 0)

        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X[:, 0, :])
        output = []

        t0 = time.perf_counter()
        for step_i in range(X.shape[1]):
            hx = self.gru_cell(X[:, step_i, :], hx)
            output.append(hx)
        loop_ms = (time.perf_counter() - t0) * 1000

        output = th.stack(output, dim=0)
        output = self.dropout(output)

        if _verbose:
            print(f"[Walpurgis::RNNLayer::forward] call#{RNNLayer._call_count} "
                  f"batch*nodes={batch_size * num_nodes} seq_len={seq_len} "
                  f"loop={loop_ms:.3f}ms ({loop_ms / max(seq_len, 1):.3f}ms/step) "
                  f"output mean={output.mean().item():.6f} std={output.std().item():.6f}")
            # Check for dead neurons (all-zero hidden states)
            dead_ratio = (output.abs().sum(dim=-1) == 0).float().mean().item()
            if dead_ratio > 0.01:
                print(f"  ⚠ RNNLayer dead neuron ratio: {dead_ratio:.4f}")

        return output


class TransformerLayer(nn.Module):
    """Multi-head self-attention layer for capturing long-range temporal dependencies.

    Walpurgis notes:
    - Attention weight matrices are O(seq_len^2) — for long sequences,
      these dominate memory and should be HBM-pinned during training.
    - Dropout on attention weights is tracked for effective sparsity.
    """

    _call_count = 0

    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        print(f"[Walpurgis::TransformerLayer] init hidden_dim={hidden_dim} "
              f"num_heads={num_heads} head_dim={hidden_dim // num_heads} bias={bias}")

    def forward(self, X, K, V):
        TransformerLayer._call_count += 1
        _verbose = (TransformerLayer._call_count <= 3 or TransformerLayer._call_count % 500 == 0)

        t0 = time.perf_counter()
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"[Walpurgis::TransformerLayer::forward] call#{TransformerLayer._call_count} "
                  f"Q shape={list(X.shape)} elapsed={elapsed_ms:.3f}ms "
                  f"output mean={hidden_states_MSA.mean().item():.6f} "
                  f"std={hidden_states_MSA.std().item():.6f}")
            _nan = th.isnan(hidden_states_MSA).any().item()
            _inf = th.isinf(hidden_states_MSA).any().item()
            if _nan or _inf:
                print(f"  ⚠ TransformerLayer output: nan={_nan} inf={_inf}")

        return hidden_states_MSA
