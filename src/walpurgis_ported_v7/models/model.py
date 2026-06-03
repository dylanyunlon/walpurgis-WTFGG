import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

_DBG_MODEL = ("--dbg-model" in sys.argv)


def _mp(tag, t):
    """Model-level debug print."""
    if not _DBG_MODEL:
        return
    with torch.no_grad():
        print(f"[DBG-MODEL][{tag}] shape={list(t.shape)}  "
              f"mean={t.mean().item():.5f}  std={t.std().item():.5f}  "
              f"absmax={t.abs().max().item():.5f}")


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
            history_data=history_data, gated_history_data=gated_history_data,
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

        # 算法改动: 2-layer MLP embedding + skip connection
        # 原版: 单层 Linear(in_feat, hidden)
        # 改为: Linear(in, hidden) -> GELU -> Linear(hidden, hidden) + skip
        self.embedding_fc1 = nn.Linear(self._in_feat, self._hidden_dim)
        self.embedding_fc2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.embedding_act = nn.GELU()

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layers
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                           **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                              **model_args))

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)

        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # output layer
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        # 算法改动: output residual shortcut
        # 让 forecast_hidden 直接有一条通路到输出, 绕过 2-layer MLP
        self.out_shortcut = nn.Linear(self._forecast_dim, model_args['gap'])

        # 算法改动: static graph 温度参数
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
            # 算法改动: 加可学习 temperature 控制 static graph 的锐度
            temp = torch.clamp(self.static_graph_temp, min=0.1)
            raw = torch.mm(E_d, E_u.T) / temp
            static_graph = [F.softmax(F.relu(raw), dim=1)]

            _mp("static_graph", static_graph[0])
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
        # ==================== Prepare ==================== #
        (history_data, node_embedding_u, node_embedding_d,
         time_in_day_feat, day_in_week_feat) = self._prepare_inputs(
            history_data)

        _mp("raw_input", history_data)

        # ==================== Graphs ==================== #
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # 算法改动: 2-layer MLP embedding with skip
        h = self.embedding_fc1(history_data)
        h_skip = h
        h = self.embedding_act(h)
        h = self.embedding_fc2(h)
        history_data = h + h_skip   # residual

        _mp("post_embedding", history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for layer_idx, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fk, inh_fk = layer(
                inh_backcast_seq_res, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fk)
            inh_forecast_hidden_list.append(inh_fk)

            if _DBG_MODEL:
                with torch.no_grad():
                    print(f"[DBG-MODEL] layer-{layer_idx}  "
                          f"backcast_norm={inh_backcast_seq_res.norm().item():.4f}  "
                          f"dif_fk_norm={dif_fk.norm().item():.4f}  "
                          f"inh_fk_norm={inh_fk.norm().item():.4f}")

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        _mp("forecast_hidden", forecast_hidden)

        # 算法改动: main path + shortcut
        main_out = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        shortcut_out = self.out_shortcut(forecast_hidden)
        forecast = main_out + shortcut_out * 0.1

        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        _mp("final_output", forecast)
        return forecast
