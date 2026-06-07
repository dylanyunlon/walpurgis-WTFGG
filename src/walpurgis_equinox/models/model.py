import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[EQX:model:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[EQX:model:{tag}] {val}", file=sys.stderr)


class WeightNormLinear(nn.Module):
    """equinox: WeightNorm替代BatchNorm/GroupNorm"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.utils.parametrizations.weight_norm(
            nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x):
        return self.linear(x)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        # equinox: 门控能量监控 — 可学习分支平衡系数
        self._branch_balance = nn.Parameter(torch.tensor(0.5))

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(node_embedding_u, node_embedding_d,
                                     time_in_day_feat, day_in_week_feat, history_data)
        _edbg("gate_ratio", torch.norm(gated) / (torch.norm(history_data) + 1e-8))
        dif_backcast, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast, inh_fk = self.inh_layer(dif_backcast)
        bal = torch.sigmoid(self._branch_balance)
        _edbg("dif_energy", torch.norm(dif_fk))
        _edbg("inh_energy", torch.norm(inh_fk))
        _edbg("balance_coeff", bal)
        return inh_backcast, dif_fk, inh_fk


class D2STGNN(nn.Module):
    """equinox D2STGNN:
    - WeightNorm替代GroupNorm/BatchNorm (embedding层)
    - DenseNet式layer aggregation: 收集所有层的forecast, concat后gated projection
    - Mish激活替代Swish/ReLU
    """
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

        # equinox: WeightNorm embedding + Mish激活 (替代GroupNorm+Swish)
        self.embedding = WeightNormLinear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)
            for _ in range(self._num_layers)
        ])

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # equinox: DenseNet式layer aggregation
        # 收集所有层的dif_fk和inh_fk, concat后gated projection
        # 每层产出forecast_dim, 共num_layers层, dif+inh各一组
        self._dense_total_dim = self._forecast_dim * self._num_layers
        self.dense_gate_dif = nn.Sequential(
            WeightNormLinear(self._dense_total_dim, self._forecast_dim),
            nn.Sigmoid()
        )
        self.dense_proj_dif = WeightNormLinear(self._dense_total_dim, self._forecast_dim)
        self.dense_gate_inh = nn.Sequential(
            WeightNormLinear(self._dense_total_dim, self._forecast_dim),
            nn.Sigmoid()
        )
        self.dense_proj_inh = WeightNormLinear(self._dense_total_dim, self._forecast_dim)

        # equinox: output head with WeightNorm + Mish
        self.out_fc_1 = WeightNormLinear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = WeightNormLinear(self._output_hidden, model_args['gap'])

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
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        t_idx = (history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)
        t_idx = torch.clamp(t_idx, 0, 287)
        time_in_day_feat = self.T_i_D_emb[t_idx]
        d_idx = history_data[:, :, :, num_feat + 1].type(torch.LongTensor)
        d_idx = torch.clamp(d_idx, 0, 6)
        day_in_week_feat = self.D_i_W_emb[d_idx]
        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        _edbg("input", history_data)
        history_data, node_eu, node_ed, t_feat, d_feat = self._prepare_inputs(history_data)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_eu, node_embedding_d=node_ed,
            history_data=history_data, time_in_day_feat=t_feat, day_in_week_feat=d_feat)

        # equinox: WeightNorm embedding + Mish激活
        history_data = self.embedding(history_data)
        history_data = F.mish(history_data)
        _edbg("post_embed", history_data)

        dif_fk_list = []
        inh_fk_list = []
        inh_backcast = history_data
        for idx, layer in enumerate(self.layers):
            inh_backcast, dif_fk, inh_fk = layer(
                inh_backcast, dynamic_graph, static_graph,
                node_eu, node_ed, t_feat, d_feat)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)
            _edbg(f"layer_{idx}_out", inh_backcast)

        # equinox: DenseNet式聚合 — concat所有层 + gated projection
        # upstream: sum()聚合
        # aurora:   MoE softmax-gate加权聚合
        # equinox:  concat所有层forecast → gated linear projection
        dif_concat = torch.cat(dif_fk_list, dim=-1)   # [B, L, N, forecast_dim*num_layers]
        inh_concat = torch.cat(inh_fk_list, dim=-1)
        _edbg("dense_dif_concat", dif_concat)

        dif_gate = self.dense_gate_dif(dif_concat)
        dif_fk_agg = dif_gate * self.dense_proj_dif(dif_concat)
        inh_gate = self.dense_gate_inh(inh_concat)
        inh_fk_agg = inh_gate * self.dense_proj_inh(inh_concat)
        _edbg("dense_dif_agg", dif_fk_agg)
        _edbg("dense_inh_agg", inh_fk_agg)

        forecast_hidden = dif_fk_agg + inh_fk_agg

        # equinox: Mish(WeightNorm_FC) → FC
        h = self.out_fc_1(forecast_hidden)
        h = F.mish(h)
        forecast = self.out_fc_2(h)
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _edbg("output", forecast)
        return forecast
