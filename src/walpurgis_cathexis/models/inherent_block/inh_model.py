"""
Cathexis InhModel — 算法改写 #7
upstream: GRUCell sequential → sinusoidal PE → MultiheadAttention
cathexis: Selective-scan SSM (Mamba-style S4) + SwishGLU gating
"""
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class SelectiveScanLayer(nn.Module):
    """Mamba-style selective state space model (simplified S4 for temporal)"""
    def __init__(self, hidden_dim, state_dim=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        # Input-dependent discretization
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        # SSM parameters A, B, C
        self.A_log = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.B_proj = nn.Linear(hidden_dim, state_dim)
        self.C_proj = nn.Linear(hidden_dim, state_dim)
        # SwishGLU gate
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """X: [batch_size*num_nodes, seq_len, hidden_dim]"""
        BN, L, D = X.shape
        # compute dt (discretization step)
        dt = torch.sigmoid(self.dt_proj(X))  # [BN, L, D]
        # A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # [D, state_dim]
        B = self.B_proj(X)  # [BN, L, state_dim]
        C = self.C_proj(X)  # [BN, L, state_dim]
        # Selective scan
        h = torch.zeros(BN, D, self.state_dim, device=X.device)
        outputs = []
        for t in range(L):
            # Discretize: dA = exp(A * dt), dB = dt * B
            dA = torch.exp(A.unsqueeze(0) * dt[:, t, :].unsqueeze(-1))  # [BN, D, state_dim]
            dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)   # [BN, D, state_dim]
            h = h * dA + dB * X[:, t, :].unsqueeze(-1)
            y = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # [BN, D]
            outputs.append(y)
        scan_out = torch.stack(outputs, dim=1)  # [BN, L, D]
        # SwishGLU gating
        gate = self.gate_proj(scan_out) * torch.sigmoid(self.gate_proj(scan_out))
        up = self.up_proj(X)
        out = self.out_proj(gate * up)
        out = self.dropout(out)
        return out

class RNNLayer(nn.Module):
    """Cathexis: SSM-based recurrence instead of GRU"""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Keep GRU for forecast autoregression compatibility
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.ssm = SelectiveScanLayer(hidden_dim, state_dim=16, dropout=dropout or 0.1)
        self.dropout = nn.Dropout(dropout or 0.1)

    def forward(self, X):
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        # Cathexis改写: SSM instead of step-by-step GRU
        output = self.ssm(X)
        output = output.transpose(0, 1)  # [L, BN, D] for transformer compatibility
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
