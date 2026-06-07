import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

def _sdbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:model:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:model:{tag}] {val}", file=sys.stderr)


class ScaleNorm(nn.Module):
    """solstice: ScaleNorm替代LayerNorm — 仅可学习scale参数, 无偏置
    normalize到单位范数后乘可学习标量g"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return self.g * x / norm


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(node_embedding_u, node_embedding_d,
                                     time_in_day_feat, day_in_week_feat, history_data)
        _sdbg("gate_ratio", torch.norm(gated) / (torch.norm(history_data) + 1e-8))
        dif_backcast, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast, inh_fk = self.inh_layer(dif_backcast)
        _sdbg("dif_energy", torch.norm(dif_fk))
        _sdbg("inh_energy", torch.norm(inh_fk))
        return inh_backcast, dif_fk, inh_fk


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

        # upstream: nn.Linear embedding
        # solstice: Linear + ScaleNorm + Mish激活
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.embed_sn = ScaleNorm(self._hidden_dim)

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

        # upstream: sum()聚合 + ReLU双层FC
        # solstice: attention-weighted pooling聚合 + Mish输出头
        # Attention pooling: 每层产出过一个共享query, 计算attention weight后加权聚合
        self.pool_query = nn.Parameter(torch.randn(1, 1, 1, self._forecast_dim))
        self.pool_key_proj = nn.Linear(self._forecast_dim, self._forecast_dim)

        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
        # solstice: ScaleNorm + Mish替代ReLU
        self.out_sn = ScaleNorm(self._output_hidden)

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

    def _mish(self, x):
        """Mish激活: x * tanh(softplus(x))"""
        return x * torch.tanh(F.softplus(x))

    def forward(self, history_data):
        _sdbg("input", history_data)
        history_data, node_eu, node_ed, t_feat, d_feat = self._prepare_inputs(history_data)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_eu, node_embedding_d=node_ed,
            history_data=history_data, time_in_day_feat=t_feat, day_in_week_feat=d_feat)

        # solstice: embedding后ScaleNorm + Mish
        history_data = self.embedding(history_data)
        history_data = self.embed_sn(history_data)
        history_data = self._mish(history_data)
        _sdbg("post_embed", history_data)

        dif_fk_list = []
        inh_fk_list = []
        inh_backcast = history_data
        for idx, layer in enumerate(self.layers):
            inh_backcast, dif_fk, inh_fk = layer(
                inh_backcast, dynamic_graph, static_graph,
                node_eu, node_ed, t_feat, d_feat)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)
            _sdbg(f"layer_{idx}_out", inh_backcast)

        # upstream: sum()聚合
        # solstice: attention-weighted pooling聚合
        # Stack layers: [num_layers, B, L, N, D]
        dif_stack = torch.stack(dif_fk_list, dim=0)
        inh_stack = torch.stack(inh_fk_list, dim=0)
        combined_stack = dif_stack + inh_stack  # [num_layers, B, L, N, D]

        # Attention scores: query dot keys for each layer
        keys = self.pool_key_proj(combined_stack)  # [num_layers, B, L, N, D]
        # query: [1, 1, 1, D] broadcast → scores [num_layers, B, L, N]
        attn_scores = (keys * self.pool_query).sum(dim=-1) / (self._forecast_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=0)  # softmax over layers
        _sdbg("pool_attn_weights", attn_weights.mean(dim=(1,2,3)))

        # Weighted aggregation
        forecast_hidden = (attn_weights.unsqueeze(-1) * combined_stack).sum(dim=0)

        # upstream: ReLU(FC(ReLU(FC(x))))
        # solstice: Mish(ScaleNorm(FC(x))) -> FC
        h = self.out_fc_1(forecast_hidden)
        h = self.out_sn(h)
        h = self._mish(h)
        forecast = self.out_fc_2(h)
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _sdbg("output", forecast)
        return forecast
