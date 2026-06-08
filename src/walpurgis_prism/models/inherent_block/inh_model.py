"""Prism inherent model: GRU with layer normalization for stable multi-view training.
Unlike upstream (bare GRU) and vortex (GRU with dropout only),
Prism adds layer normalization after GRU to stabilize gradients
when combined with contrastive loss and mixup augmentation."""
import torch as th
import torch.nn as nn
from torch.nn import MultiheadAttention


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Prism特有: GRU后接LayerNorm稳定多视角训练
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        [batch_size, seq_len, num_nodes,
         hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = th.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = th.stack(output, dim=0)
        # Prism特有: LayerNorm
        output = self.ln(output)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = \
            MultiheadAttention(
                hidden_dim, num_heads,
                dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA = \
            self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(
            hidden_states_MSA)
        return hidden_states_MSA
