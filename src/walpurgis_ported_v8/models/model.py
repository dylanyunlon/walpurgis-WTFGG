import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_DBG = ("--dbg" in sys.argv)


def _dp(tag, t):
    if not _DBG:
        return
    with torch.no_grad():
        print(f"[DBG][Model][{tag}] shape={list(t.shape)}  "
              f"mean={t.mean().item():.5f}  std={t.std().item():.5f}  "
              f"absmax={t.abs().max().item():.5f}", flush=True)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            history_data=history_data,
            gated_history_data=gated_history_data,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(
            dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
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

        # 算法改动: embedding 用 1D causal conv 替代 Linear
        # 原版: Linear(in_feat, hidden)
        # 改为: Conv1d(in_feat, hidden, kernel=3, padding=2) 截断 causal
        # 这样 embedding 本身就带了局部时序感受野
        self.embedding_conv = nn.Conv1d(
            self._in_feat, self._hidden_dim, kernel_size=3, padding=2)
        self.embedding_ln = nn.LayerNorm(self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # decouple layers
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim,
                           fk_dim=self._forecast_dim, **model_args)
             for _ in range(self._num_layers)])

        # dynamic graph
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # 算法改动: output layer 加 LayerNorm + GELU
        # 原版: Linear -> ReLU -> Linear -> ReLU
        # 改为: Linear -> GELU -> LayerNorm -> Linear
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # 算法改动: static graph bilateral sharpening
        # 用可学习 temperature 控制 softmax 锐度
        self.static_graph_temp = nn.Parameter(torch.tensor(1.0))

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
            # 算法改动: temperature-scaled softmax
            raw = torch.mm(E_d, E_u.T)
            temp = torch.clamp(self.static_graph_temp, min=0.1)
            static_graph = [F.softmax(F.relu(raw) / temp, dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        time_in_day_feat = self.T_i_D_emb[
            (history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]
        day_in_week_feat = self.D_i_W_emb[
            (history_data[:, :, :, num_feat + 1]).type(torch.LongTensor)]
        history_data = history_data[:, :, :, :num_feat]
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        _dp("input", history_data)

        (history_data, node_embedding_u, node_embedding_d,
         time_in_day_feat, day_in_week_feat) = \
            self._prepare_inputs(history_data)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # 算法改动: 1D causal conv embedding
        # history_data: [B, L, N, C] -> conv over L dim per node
        B, L, N, C = history_data.shape
        h = history_data.permute(0, 2, 3, 1)  # [B, N, C, L]
        h = h.reshape(B * N, C, L)
        h = self.embedding_conv(h)[:, :, :L]  # causal: truncate future
        h = h.reshape(B, N, self._hidden_dim, L)
        h = h.permute(0, 3, 1, 2)  # [B, L, N, hidden]
        history_data = self.embedding_ln(h)
        _dp("after_embedding", history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for layer_idx, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fh, inh_fh = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fh)
            inh_forecast_hidden_list.append(inh_fh)
            _dp(f"layer_{layer_idx}_backcast", inh_backcast_seq_res)

        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden
        _dp("forecast_hidden_sum", forecast_hidden)

        # 算法改动: GELU + LayerNorm output
        forecast = self.out_fc_1(forecast_hidden)
        forecast = F.gelu(forecast)
        forecast = self.out_ln(forecast)
        forecast = self.out_fc_2(forecast)
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _dp("final_output", forecast)

        return forecast
