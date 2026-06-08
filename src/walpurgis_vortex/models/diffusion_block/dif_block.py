"""Vortex DifBlock: cross-attention backcast mechanism.
Unlike upstream (single FC backcast) and eclipse (2-layer MLP + sigmoid gate),
Vortex uses cross-attention where query=backcast, key/value=forecast hidden states.
This lets the backcast branch attend to what the forecast already captured,
improving the separation of diffusion and inherent signals."""
import torch, torch.nn as nn, sys, os
from .forecast import Forecast; from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class CrossAttentionBackcast(nn.Module):
    """Cross-attention: query from backcast, key/value from forecast.
    Allows backcast to know what forecast has captured and subtract accordingly."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value):
        """query: [B, S1, N, D], key_value: [B, S2, N, D]"""
        B, S1, N, D = query.shape
        S2 = key_value.shape[1]
        # Reshape to [B*N, S, D] for per-node attention
        q = query.permute(0, 2, 1, 3).reshape(B * N, S1, D)
        kv = key_value.permute(0, 2, 1, 3).reshape(B * N, S2, D)
        Q = self.wq(q).view(B * N, S1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(kv).view(B * N, S2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(kv).view(B * N, S2, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B * N, S1, D)
        out = self.out_proj(out)
        out = out.reshape(B, N, S1, D).permute(0, 2, 1, 3)
        return out

class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph,
                                                  use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # Project forecast features down to hidden_dim for cross-attention
        self.fk_proj = nn.Linear(forecast_hidden_dim, hidden_dim)
        # Cross-attention backcast: query from hidden_states, key/value from projected forecast
        self.cross_attn_backcast = CrossAttentionBackcast(hidden_dim, num_heads=4, dropout=model_args.get('dropout', 0.1))
        self.backcast_proj = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hsd = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        fk = self.forecast_branch(gated_history_data, hsd, self.localized_st_conv, dynamic_graph, static_graph)
        # Project forecast to hidden_dim, then cross-attention
        fk_hidden = self.fk_proj(fk)
        bc = self.cross_attn_backcast(hsd, fk_hidden)
        bc = self.backcast_proj(bc)
        hd = history_data[:, -bc.shape[1]:, :, :]
        res = self.residual_decompose(hd, bc)
        if _VX_DBG:
            print(f"[VX:difblk@dif_block] bc_e={bc.norm().item():.4f} fk_e={fk.norm().item():.4f} cross_attn_shape={list(bc.shape)}", file=sys.stderr)
        return res, fk
