"""
D2STGNN CardGame variant — model.py
Algorithm changes vs upstream:
  1. sum aggregation → attention-weighted aggregation over layer forecasts
  2. ReLU output head → Swish (SiLU) activation + spatial dropout after embedding
  3. Spatial dropout on embedding output for regularization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


class SpatialDropout(nn.Module):
    """Drop entire node feature channels (spatial dropout)."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (B, L, N, D)
        if not self.training or self.p == 0:
            return x
        # mask shape (B, 1, 1, D) — same mask across L and N dims
        mask = torch.ones(x.shape[0], 1, 1, x.shape[3], device=x.device, dtype=x.dtype)
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask


class AttentionAggregator(nn.Module):
    """Learnable attention-weighted aggregation over layer forecasts,
    replacing the simple sum used in upstream D2STGNN."""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # project each layer's forecast to a scalar attention logit
        self.attn_proj = nn.Linear(hidden_dim, 1, bias=False)
        # layer-specific bias for diversity
        self.layer_bias = nn.Parameter(torch.zeros(num_layers))
        self._num_layers = num_layers

    def forward(self, forecast_list):
        """
        Args:
            forecast_list: list of tensors, each (B, T, N, D) or (B, N, D)
        Returns:
            aggregated: same shape as each element, with layer dim aggregated
        """
        stacked = torch.stack(forecast_list, dim=1)  # (B, L, T, N, D) or (B, L, N, D)
        _dbg("attn_agg.stacked", stacked, "AttentionAggregator")

        n_layers = stacked.shape[1]
        # average-pool each layer's forecast to get per-layer score
        # flatten everything except batch and layer dims for the projection
        orig_shape = stacked.shape
        flat = stacked.mean(dim=list(range(2, stacked.ndim - 1)))  # (B, L, D)
        logits = self.attn_proj(flat).squeeze(-1)  # (B, L)
        logits = logits + self.layer_bias[:n_layers]
        weights = F.softmax(logits, dim=1)  # (B, L)
        _dbg("attn_agg.weights", weights, "AttentionAggregator")

        # reshape weights for broadcast: (B, L, 1, ..., 1)
        extra_dims = stacked.ndim - 2  # number of dims after layer dim
        w_shape = list(weights.shape) + [1] * extra_dims
        weights_broad = weights.view(*w_shape)

        aggregated = (stacked * weights_broad).sum(dim=1)
        _dbg("attn_agg.output", aggregated, "AttentionAggregator")
        return aggregated


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (list of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un decoupled signal in this layer, i.e., the X^{l+1}
            torch.Tensor: the output of the forecast branch of Diffusion Block
            torch.Tensor: the output of the forecast branch of Inherent Block
        """
        _dbg("decouple.input", history_data, "DecoupleLayer")
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)
        _dbg("decouple.gated", gated_history_data, "DecoupleLayer")

        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data,
            gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph,
            static_graph=static_graph)
        _dbg("decouple.dif_forecast", dif_forecast_hidden, "DecoupleLayer")

        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        _dbg("decouple.inh_forecast", inh_forecast_hidden, "DecoupleLayer")
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._forecast_dim  = 256
        self._output_hidden = 512
        self._output_dim    = model_args['seq_length']

        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 5

        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True

        self._model_args    = model_args

        # start embedding layer
        self.embedding      = nn.Linear(self._in_feat, self._hidden_dim)

        # --- CARDGAME: spatial dropout after embedding ---
        self.spatial_dropout = SpatialDropout(p=model_args.get('dropout', 0.1))

        # time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        # dynamic and static hidden graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # --- CARDGAME: attention-weighted aggregation ---
        self.dif_attn_agg = AttentionAggregator(self._forecast_dim, self._num_layers)
        self.inh_attn_agg = AttentionAggregator(self._forecast_dim, self._num_layers)

        # --- CARDGAME: Swish (SiLU) output head replacing ReLU ---
        self.out_fc_1   = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2   = nn.Linear(self._output_hidden, model_args['gap'])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat    = self._model_args['num_feat']
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        """Feed forward of D2STGNN (CardGame variant).

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]
        """

        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)
        _dbg("model.prepared_input", history_data, "D2STGNN")

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # Start embedding layer + CARDGAME: spatial dropout
        history_data = self.embedding(history_data)
        history_data = self.spatial_dropout(history_data)
        _dbg("model.after_emb_dropout", history_data, "D2STGNN")

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for layer_idx, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)
            _dbg(f"model.layer_{layer_idx}_backcast", inh_backcast_seq_res, "D2STGNN")

        # --- CARDGAME: attention-weighted aggregation ---
        dif_forecast_hidden = self.dif_attn_agg(dif_forecast_hidden_list)
        inh_forecast_hidden = self.inh_attn_agg(inh_forecast_hidden_list)
        forecast_hidden     = dif_forecast_hidden + inh_forecast_hidden
        _dbg("model.forecast_hidden_agg", forecast_hidden, "D2STGNN")

        # --- CARDGAME: Swish (SiLU) output head ---
        forecast = self.out_fc_2(F.silu(self.out_fc_1(F.silu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _dbg("model.output", forecast, "D2STGNN")

        return forecast
