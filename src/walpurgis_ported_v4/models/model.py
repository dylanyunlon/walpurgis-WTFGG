import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_V4_DEBUG = True
_dbg_call_count = 0


def _dbg(tag, **kw):
    if not _V4_DEBUG:
        return
    parts = [f"[v4-DBG][D2STGNN][{tag}]"]
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}|norm={v.detach().norm().item():.4f}")
        else:
            parts.append(f"{k}={v}")
    print(" ".join(parts), file=sys.stderr)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64
        )
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

        # v4: layer-wise learnable blending weight between dif and inh branches
        # instead of passing dif_backcast directly to inh, blend with original
        self.branch_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):

        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data
        )

        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data,
            gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph,
            static_graph=static_graph
        )

        # v4: blend dif output with original input before feeding to inh
        # alpha ∈ (0,1) via sigmoid — controls how much diffusion residual
        # vs raw signal the inherent block sees
        alpha = torch.sigmoid(self.branch_alpha)
        inh_input = alpha * dif_backcast_seq_res + (1 - alpha) * history_data

        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(inh_input)

        if _V4_DEBUG:
            _dbg("DecoupleLayer",
                 alpha=f"{alpha.item():.4f}",
                 dif_backcast=dif_backcast_seq_res,
                 inh_input=inh_input)

        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat = model_args['num_feat']
        self._hidden_dim = model_args['num_hidden']
        self._node_dim = model_args['node_hidden']
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = model_args['seq_length']

        self._num_nodes = model_args['num_nodes']
        self._k_s = model_args['k_s']
        self._k_t = model_args['k_t']
        self._num_layers = 5

        model_args['use_pre'] = False
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True

        self._model_args = model_args

        # start embedding layer
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

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

        # v4: output layer uses SiLU (Swish) instead of ReLU
        # SiLU provides smoother gradients around zero, better for regression
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # v4: learnable per-layer forecast weighting instead of uniform sum
        self.layer_weights = nn.Parameter(torch.ones(self._num_layers))

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
        num_feat = self._model_args['num_feat']
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        global _dbg_call_count
        _dbg_call_count += 1

        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat
        )

        # Start embedding layer
        history_data = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat
            )
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # v4: softmax-weighted sum across layers instead of uniform sum
        # lets the model learn which layers' forecasts matter most
        w = F.softmax(self.layer_weights, dim=0)
        dif_forecast_hidden = sum(w[i] * dif_forecast_hidden_list[i] for i in range(len(dif_forecast_hidden_list)))
        inh_forecast_hidden = sum(w[i] * inh_forecast_hidden_list[i] for i in range(len(inh_forecast_hidden_list)))
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        # v4: SiLU (Swish) replaces ReLU in regression head
        forecast = self.out_fc_2(F.silu(self.out_fc_1(F.silu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)

        if _V4_DEBUG and _dbg_call_count <= 3:
            _dbg("forward",
                 forecast=forecast,
                 layer_weights=f"[{', '.join(f'{x:.3f}' for x in w.detach().tolist())}]",
                 num_layers=self._num_layers)

        return forecast
