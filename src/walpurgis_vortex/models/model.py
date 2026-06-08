"""
D2STGNN — Vortex变体
算法改写 (~20%):
  1. DecoupleLayer: EMA动量融合门控 — dif/inh分支通过可学习momentum
     做指数移动平均混合, 而非直接pass-through
  2. D2STGNN: 随机深度(stochastic depth) — 训练时以线性增长概率跳过层
  3. D2STGNN: 温度缩放聚合 — 各层forecast用可学习温度做softmax加权
  4. 输出头: Mish激活 + GroupNorm + 双路输出(主路+辅助路gradient-detach)
  5. embedding后接可学习的通道缩放 (channel-wise scaling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


def _mish(x):
    """Mish激活: x * tanh(softplus(x))"""
    return x * torch.tanh(F.softplus(x))


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256,
                 layer_idx=0, drop_rate=0.0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.drop_rate = drop_rate
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64)
        self.dif_layer = DifBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, forecast_hidden_dim=fk_dim,
            **model_args)
        # EMA动量融合: 可学习的momentum参数控制dif/inh混合
        self.ema_momentum = nn.Parameter(torch.tensor(0.9))
        self._ema_state = None
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        dataflow_checkpoint(
            f"decouple_L{self.layer_idx}.input",
            history_data)
        # Stochastic depth: 训练时按概率跳过该层
        if self.training and self.drop_rate > 0:
            if torch.rand(1).item() < self.drop_rate:
                _dbg(f"decouple_L{self.layer_idx}.DROPPED",
                     f"drop_rate={self.drop_rate:.3f}", "model")
                # 返回identity + 零forecast
                B, L, N, D = history_data.shape
                fk_len = self._seq_length // self._gap
                zero_fk = torch.zeros(
                    B, fk_len, N, 256,
                    device=history_data.device)
                return history_data, zero_fk, zero_fk
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat,
            history_data)
        dif_backcast_seq_res, dif_forecast_hidden = \
            self.dif_layer(
                history_data=history_data,
                gated_history_data=gated_history_data,
                dynamic_graph=dynamic_graph,
                static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = \
            self.inh_layer(dif_backcast_seq_res)
        # EMA动量融合: 用momentum混合当前和历史状态
        momentum = torch.sigmoid(self.ema_momentum)
        if self._ema_state is not None and \
                self._ema_state.shape == inh_backcast_seq_res.shape:
            inh_backcast_seq_res = (
                momentum * inh_backcast_seq_res +
                (1 - momentum) * self._ema_state.detach())
        if self.training:
            self._ema_state = inh_backcast_seq_res.detach()
        _dbg(f"decouple_L{self.layer_idx}.ema_m",
             momentum, "model")
        _dbg(f"decouple_L{self.layer_idx}.dif_energy",
             dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy",
             inh_forecast_hidden.detach().norm(), "model")
        return (inh_backcast_seq_res,
                dif_forecast_hidden,
                inh_forecast_hidden)


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
        # embedding + 可学习通道缩放 (Vortex特有)
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        self.channel_scale = nn.Parameter(
            torch.ones(self._hidden_dim))
        self.channel_bias = nn.Parameter(
            torch.zeros(self._hidden_dim))
        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))
        # decouple layers: 随机深度概率线性增长
        drop_rates = [
            0.05 * i / max(self._num_layers - 1, 1)
            for i in range(self._num_layers)]
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, drop_rate=drop_rates[i],
                **model_args)
            for i in range(self._num_layers)
        ])
        # 温度缩放聚合 (Vortex特有)
        self.agg_temperature = nn.Parameter(torch.tensor(1.0))
        self.layer_importance = nn.Parameter(
            torch.zeros(self._num_layers))
        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)
        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        # 输出头: Mish + GroupNorm + 双路
        n_groups = min(4, self._output_hidden)
        while self._output_hidden % n_groups != 0:
            n_groups -= 1
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_gn = nn.GroupNorm(
            n_groups, self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # 辅助路: gradient-detach轻量分支
        self.aux_fc = nn.Linear(
            self._forecast_dim, model_args['gap'])
        self.aux_gate = nn.Parameter(torch.tensor(0.05))
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
            static_graph = [F.softmax(
                F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = \
                self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        t_idx = (history_data[:, :, :, num_feat] * 288
                 ).type(torch.LongTensor).clamp(0, 287)
        d_idx = (history_data[:, :, :, num_feat + 1]
                 ).type(torch.LongTensor).clamp(0, 6)
        time_in_day_feat = self.T_i_D_emb[t_idx]
        day_in_week_feat = self.D_i_W_emb[d_idx]
        history_data = history_data[:, :, :, :num_feat]
        return (history_data, node_emb_u, node_emb_d,
                time_in_day_feat, day_in_week_feat)

    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, \
            time_in_day_feat, day_in_week_feat = \
            self._prepare_inputs(history_data)
        dataflow_checkpoint("model.raw_input", history_data)
        dump_struct_state(
            "pre_graph",
            history_data=history_data,
            node_emb_u=node_embedding_u,
            node_emb_d=node_embedding_d,
            time_feat_shape=time_in_day_feat)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)
        # embedding + 通道缩放
        history_data = self.embedding(history_data)
        history_data = (history_data * self.channel_scale +
                        self.channel_bias)
        dataflow_checkpoint("model.post_embed", history_data)
        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []
        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_fh, inh_fh = layer(
                inh_backcast_seq_res, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat)
            dif_forecast_hidden_list.append(dif_fh)
            inh_forecast_hidden_list.append(inh_fh)
        # 温度缩放聚合
        temp = torch.clamp(self.agg_temperature, min=0.1)
        weights = F.softmax(
            self.layer_importance / temp, dim=0)
        _dbg("agg_temp", temp, "model")
        _dbg("agg_weights", weights, "model")
        dif_forecast_hidden = sum(
            weights[i] * dif_forecast_hidden_list[i]
            for i in range(self._num_layers))
        inh_forecast_hidden = sum(
            weights[i] * inh_forecast_hidden_list[i]
            for i in range(self._num_layers))
        forecast_hidden = (dif_forecast_hidden +
                           inh_forecast_hidden)
        # 主路: Mish + GroupNorm
        B, L, N, D = forecast_hidden.shape
        h = _mish(self.out_fc_1(forecast_hidden))
        h = h.permute(0, 3, 1, 2)  # [B, C, L, N]
        h = self.out_gn(h)
        h = h.permute(0, 2, 3, 1)  # [B, L, N, C]
        main_out = self.out_fc_2(_mish(h))
        # 辅助路: detach梯度, 轻量投影
        aux_input = forecast_hidden.detach()
        aux_out = self.aux_fc(aux_input)
        gate = torch.sigmoid(self.aux_gate)
        forecast = main_out + gate * aux_out
        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.main_range",
             f"[{main_out.min().item():.4f},"
             f"{main_out.max().item():.4f}]", "model")
        _dbg("output.aux_gate", gate, "model")
        return forecast
