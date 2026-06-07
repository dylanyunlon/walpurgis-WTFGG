import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:model:{tag}] shape={list(val.shape)} mean={val.mean().item():.6f} std={val.std().item():.6f}", file=sys.stderr)
    else:
        print(f"[SOL:model:{tag}] {val}", file=sys.stderr)


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        # solstice: 门控能量监控 — 可学习分支平衡系数
        self._branch_balance = nn.Parameter(torch.tensor(0.5))

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(node_embedding_u, node_embedding_d,
                                     time_in_day_feat, day_in_week_feat, history_data)
        _adbg("gate_ratio", torch.norm(gated) / (torch.norm(history_data) + 1e-8))
        dif_backcast, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        inh_backcast, inh_fk = self.inh_layer(dif_backcast)
        # solstice: 打印dif/inh分支能量比
        bal = torch.sigmoid(self._branch_balance)
        _adbg("dif_energy", torch.norm(dif_fk))
        _adbg("inh_energy", torch.norm(inh_fk))
        _adbg("balance_coeff", bal)
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

        # upstream: nn.Linear embedding直接进layer
        # solstice: Linear + ChannelShuffle + SiLU激活
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self._embed_groups = max(1, self._hidden_dim // 8)

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
        # solstice: Attention-Pooling聚合 — 可学习query向量对各层做注意力加权
        self.attn_pool_query = nn.Parameter(torch.randn(1, self._forecast_dim) * 0.02)
        self.attn_pool_key = nn.Linear(self._forecast_dim, self._forecast_dim)
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
        # solstice: SiLU替代ReLU
        self.out_ln = nn.LayerNorm(self._output_hidden)

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

    def _channel_shuffle(self, x):
        """solstice: embedding后的channel shuffle"""
        B_dims = x.shape[:-1]
        C = x.shape[-1]
        g = self._embed_groups
        if C % g != 0:
            return x
        x = x.view(*B_dims, g, C // g)
        x = x.transpose(-2, -1).contiguous()
        x = x.view(*B_dims, C)
        return x

    def forward(self, history_data):
        _adbg("input", history_data)
        history_data, node_eu, node_ed, t_feat, d_feat = self._prepare_inputs(history_data)
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_eu, node_embedding_d=node_ed,
            history_data=history_data, time_in_day_feat=t_feat, day_in_week_feat=d_feat)

        # solstice: embedding后加ChannelShuffle + SiLU
        history_data = self.embedding(history_data)
        history_data = self._channel_shuffle(history_data)
        history_data = F.silu(history_data)
        _adbg("post_embed", history_data)

        dif_fk_list = []
        inh_fk_list = []
        inh_backcast = history_data
        for idx, layer in enumerate(self.layers):
            inh_backcast, dif_fk, inh_fk = layer(
                inh_backcast, dynamic_graph, static_graph,
                node_eu, node_ed, t_feat, d_feat)
            dif_fk_list.append(dif_fk)
            inh_fk_list.append(inh_fk)
            _adbg(f"layer_{idx}_out", inh_backcast)

        # upstream: sum()聚合
        # solstice: attention-pooling聚合 — query向量对每层做scaled dot-product attention
        n_layers = len(dif_fk_list)
        # 对每层forecast做attention-pooling
        dif_stack = torch.stack(dif_fk_list, dim=0)  # [L, B, T, N, D]
        inh_stack = torch.stack(inh_fk_list, dim=0)  # [L, B, T, N, D]
        # 计算每层的key: 平均池化后投影
        dif_keys = self.attn_pool_key(dif_stack.mean(dim=(2, 3)))  # [L, B, D]
        query = self.attn_pool_query.unsqueeze(0).expand(n_layers, -1, -1)  # [L, 1, D] -> broadcast
        attn_scores = (query * dif_keys).sum(dim=-1) / (self._forecast_dim ** 0.5)  # [L, B]
        attn_weights = F.softmax(attn_scores, dim=0)  # [L, B]
        _adbg("attn_pool_weights", attn_weights)
        aw = attn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [L, B, 1, 1, 1]
        dif_fk_agg = (aw * dif_stack).sum(dim=0)  # [B, T, N, D]
        inh_fk_agg = (aw * inh_stack).sum(dim=0)  # [B, T, N, D]
        forecast_hidden = dif_fk_agg + inh_fk_agg

        # upstream: ReLU(FC(ReLU(FC(x))))
        # solstice: SiLU(LayerNorm(FC(x))) -> FC
        h = self.out_fc_1(forecast_hidden)
        h = self.out_ln(h)
        h = F.silu(h)
        forecast = self.out_fc_2(h)
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        _adbg("output", forecast)
        return forecast
