"""
D2STGNN — Transit变体 (M055)
算法改写 (~20%):
  1. Attention Weighting聚合: 可学习query token + multi-head attention
     对所有层的forecast hidden做注意力加权, 替代EMA固定衰减
     query=可学习token, key/value=各层输出, 自适应学习层间重要性
  2. 嵌入后接GELU + RMSNorm预处理
  3. 输出头: GLU门控 + 残差shortcut
  4. 集成所有子模块的算法改动 (Capsule/EMD/APPNP/Wasserstein/S4等)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import (_dbg, dataflow_checkpoint, dump_struct_state)


class RMSNorm(nn.Module):
    """RMSNorm: 比LayerNorm更轻量, 不需要均值中心化"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True)
            + self.eps)
        return x / rms * self.scale


class LayerAttentionAggregator(nn.Module):
    """注意力加权聚合器: 可学习query token做cross-attention
    query = 可学习的全局token [1, D]
    keys/values = 各层输出 [num_layers, B, N, D]
    输出 = softmax(Q·K^T/√d)·V 的加权和

    相比EMA衰减:
      - 自适应: 不同输入样本可以有不同的层权重
      - 多头: 不同head关注不同层的不同特征子空间
      - 可解释: attention weights直接反映层重要性
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim={hidden_dim} not divisible by heads={num_heads}"

        # 可学习query token: [1, hidden_dim]
        self.query_token = nn.Parameter(
            torch.randn(1, hidden_dim) * 0.02)

        # 投影: Q/K/V各一个线性层
        self.W_q = nn.Linear(hidden_dim, hidden_dim,
                             bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim,
                             bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim,
                             bias=False)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # temperature: 可学习的缩放因子
        self.temperature = nn.Parameter(
            torch.tensor(self.head_dim ** -0.5))

    def forward(self, layer_outputs):
        """
        layer_outputs: list of tensors, len=num_layers
          each tensor can be [B, T, N, D] (4D) or [B, N, D] (3D)
        returns: aggregated tensor (same shape as input minus layer dim)
                 + attention weights [num_layers]
        """
        num_layers = len(layer_outputs)
        orig_shape = layer_outputs[0].shape

        # 统一为3D: 将前面的维度合并为batch
        if len(orig_shape) == 4:
            B, T, N, D = orig_shape
            # reshape each to [B*T, N, D]
            reshaped = [x.reshape(B * T, N, D)
                        for x in layer_outputs]
        elif len(orig_shape) == 3:
            reshaped = layer_outputs
            B_flat, N, D = orig_shape
        else:
            raise ValueError(
                f"Expected 3D or 4D, got {len(orig_shape)}D")

        B_eff = reshaped[0].shape[0]

        # stack → [B_eff, N, num_layers, D]
        stacked = torch.stack(reshaped, dim=2)

        # query: broadcast到 [B_eff, N, 1, D]
        q = self.query_token.unsqueeze(0).unsqueeze(0)
        q = q.expand(B_eff, N, 1, D)
        q = self.W_q(q)  # [B_eff, N, 1, D]

        # key, value: [B_eff, N, num_layers, D]
        k = self.W_k(stacked)
        v = self.W_v(stacked)

        # reshape为multi-head: [B_eff, N, heads, seq, head_dim]
        q = q.view(B_eff, N, 1, self.num_heads,
                    self.head_dim)
        q = q.permute(0, 1, 3, 2, 4)  # [B,N,H,1,d]
        k = k.view(B_eff, N, num_layers, self.num_heads,
                    self.head_dim)
        k = k.permute(0, 1, 3, 2, 4)  # [B,N,H,L,d]
        v = v.view(B_eff, N, num_layers, self.num_heads,
                    self.head_dim)
        v = v.permute(0, 1, 3, 2, 4)  # [B,N,H,L,d]

        # attention: [B,N,H,1,L]
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.temperature
        attn_weights = F.softmax(attn, dim=-1)

        # 加权值: [B,N,H,1,d] → squeeze → [B,N,H,d]
        out = torch.matmul(attn_weights, v).squeeze(-2)
        # merge heads: [B_eff,N,D]
        out = out.reshape(B_eff, N, D)
        out = self.out_proj(out)

        # 恢复原始shape
        if len(orig_shape) == 4:
            out = out.reshape(B, T, N, D)

        # 诊断: 记录各层的平均attention权重
        mean_attn = attn_weights.mean(dim=(0, 1, 2))
        mean_attn = mean_attn.squeeze(0)  # [num_layers]
        _dbg("attn_agg.layer_weights",
             [f"{w:.3f}" for w in mean_attn.tolist()],
             "model")
        _dbg("attn_agg.temperature",
             f"{self.temperature.item():.4f}", "model")

        return out, mean_attn


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
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

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
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

        # 嵌入层 + GELU + RMSNorm预处理
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        self.embed_norm = RMSNorm(self._hidden_dim)

        # 时间嵌入
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decouple层
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim,
                          fk_dim=self._forecast_dim,
                          **model_args)
        ])
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(self._hidden_dim,
                              fk_dim=self._forecast_dim,
                              **model_args))

        # Attention Weighting聚合: 替代EMA衰减
        # 确保forecast_dim能被num_heads整除
        _attn_heads = 4
        if self._forecast_dim % _attn_heads != 0:
            _attn_heads = 1
        self.dif_aggregator = LayerAttentionAggregator(
            self._forecast_dim, num_heads=_attn_heads)
        self.inh_aggregator = LayerAttentionAggregator(
            self._forecast_dim, num_heads=_attn_heads)

        # 动态图构造器
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)

        # 节点嵌入
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # 输出头: GLU门控 + 残差shortcut
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_gate = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # 残差shortcut: 直接从forecast_dim投影到gap
        self.out_shortcut = nn.Linear(
            self._forecast_dim, model_args['gap'])

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
            time_feat=time_in_day_feat)

        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_embedding_u,
            node_embedding_d=node_embedding_d,
            history_data=history_data,
            time_in_day_feat=time_in_day_feat,
            day_in_week_feat=day_in_week_feat)

        # 嵌入 + GELU + RMSNorm
        history_data = self.embedding(history_data)
        history_data = F.gelu(history_data)
        history_data = self.embed_norm(history_data)
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

        # Attention Weighting聚合: 自适应注意力加权
        dif_forecast_hidden, dif_attn = \
            self.dif_aggregator(dif_forecast_hidden_list)
        inh_forecast_hidden, inh_attn = \
            self.inh_aggregator(inh_forecast_hidden_list)

        _dbg("dif_attn_weights",
             [f"{w:.3f}" for w in dif_attn.tolist()],
             "model")
        _dbg("inh_attn_weights",
             [f"{w:.3f}" for w in inh_attn.tolist()],
             "model")

        forecast_hidden = (dif_forecast_hidden
                           + inh_forecast_hidden)

        # 输出: GLU门控 + 残差shortcut
        # GLU: σ(gate) ⊙ relu(fc)
        glu_gate = torch.sigmoid(
            self.out_gate(forecast_hidden))
        main_path = self.out_fc_2(
            glu_gate * F.relu(
                self.out_fc_1(forecast_hidden)))
        shortcut = self.out_shortcut(forecast_hidden)
        forecast = main_path + 0.1 * shortcut

        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)

        dataflow_checkpoint("model.output", forecast)
        _dbg("output.range",
             f"[{forecast.min().item():.4f},"
             f"{forecast.max().item():.4f}]", "model")
        return forecast
