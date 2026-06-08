"""
D2STGNN — Prism变体
算法改写 (~20%):
  1. DecoupleLayer: 多视角融合门控 — dif/inh分支输出经空间/时间/频率
     三视角编码后，用可学习注意力权重做自适应融合
  2. D2STGNN: 频率域特征注入 — embedding后用FFT提取频谱特征concat
  3. D2STGNN: 视角注意力聚合 — 各层forecast用三视角注意力加权
  4. 输出头: GELU激活 + LayerNorm + 对比正则化辅助路
  5. Mixup-ready前向: 返回中间embedding供contrastive loss使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


class SpatialViewEncoder(nn.Module):
    """空间视角编码器: 通过节点间注意力池化提取空间结构特征"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, L, N, D]
        h = torch.tanh(self.spatial_proj(x))
        # 节点维度注意力池化后广播回去
        attn = torch.softmax(self.spatial_attn(h), dim=2)  # [B, L, N, 1]
        context = (x * attn).sum(dim=2, keepdim=True)  # [B, L, 1, D]
        # 用全局空间context增强每个节点
        return x + context.expand_as(x)


class TemporalViewEncoder(nn.Module):
    """时间视角编码器: 沿时间维度做因果卷积提取局部时序模式"""
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=max(1, hidden_dim // 4))
        self.gate_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=max(1, hidden_dim // 4))

    def forward(self, x):
        # x: [B, L, N, D]
        B, L, N, D = x.shape
        # 对每个节点独立做时间卷积
        x_r = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        h = self.temporal_conv(x_r)[..., :L]
        g = torch.sigmoid(self.gate_conv(x_r)[..., :L])
        out = h * g  # 门控时间特征
        out = out.reshape(B, N, D, L).permute(0, 3, 1, 2)
        return x + out


class FrequencyViewEncoder(nn.Module):
    """频率视角编码器: FFT提取频谱特征后投影回时域"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.freq_real_proj = nn.Linear(hidden_dim, hidden_dim)
        self.freq_imag_proj = nn.Linear(hidden_dim, hidden_dim)
        self.freq_gate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        # x: [B, L, N, D]
        B, L, N, D = x.shape
        # 沿时间维度做FFT
        x_freq = torch.fft.rfft(x, dim=1)
        # 分别处理实部和虚部
        real_feat = self.freq_real_proj(x_freq.real)
        imag_feat = self.freq_imag_proj(x_freq.imag)
        # 重构回时域
        combined = torch.complex(real_feat, imag_feat)
        freq_out = torch.fft.irfft(combined, n=L, dim=1)
        # 门控混合
        gate = torch.sigmoid(self.freq_gate)
        return x + gate * freq_out


class MultiViewFusion(nn.Module):
    """多视角自适应融合: 空间/时间/频率三路编码的注意力加权"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_enc = SpatialViewEncoder(hidden_dim)
        self.temporal_enc = TemporalViewEncoder(hidden_dim)
        self.frequency_enc = FrequencyViewEncoder(hidden_dim)
        # 可学习的视角重要性
        self.view_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # 三视角独立编码
        s_view = self.spatial_enc(x)
        t_view = self.temporal_enc(x)
        f_view = self.frequency_enc(x)
        # 自适应融合
        weights = F.softmax(self.view_logits, dim=0)
        _dbg("view_weights",
             f"spatial={weights[0].item():.4f} "
             f"temporal={weights[1].item():.4f} "
             f"freq={weights[2].item():.4f}", "model")
        fused = (weights[0] * s_view +
                 weights[1] * t_view +
                 weights[2] * f_view)
        return fused, weights


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256,
                 layer_idx=0, **model_args):
        super().__init__()
        self.layer_idx = layer_idx
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
        # Prism特有: 多视角融合取代直接pass-through
        self.multi_view_fusion = MultiViewFusion(hidden_dim)
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        dataflow_checkpoint(
            f"decouple_L{self.layer_idx}.input",
            history_data)
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
        # Prism特有: 多视角融合处理inh输出
        inh_backcast_seq_res, view_weights = \
            self.multi_view_fusion(inh_backcast_seq_res)
        _dbg(f"decouple_L{self.layer_idx}.dif_energy",
             dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy",
             inh_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.fused_range",
             f"[{inh_backcast_seq_res.min().item():.4f},"
             f"{inh_backcast_seq_res.max().item():.4f}]",
             "model")
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
        # embedding层
        self.embedding = nn.Linear(
            self._in_feat, self._hidden_dim)
        # Prism特有: 频率域特征注入层
        self.freq_inject_real = nn.Linear(
            self._hidden_dim, self._hidden_dim)
        self.freq_inject_gate = nn.Parameter(
            torch.tensor(0.2))
        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))
        # decouple layers
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i, **model_args)
            for i in range(self._num_layers)
        ])
        # Prism特有: 层级视角注意力聚合
        self.layer_view_attn = nn.Parameter(
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
        # 输出头: GELU + LayerNorm (Prism特有)
        self.out_fc_1 = nn.Linear(
            self._forecast_dim, self._output_hidden)
        self.out_ln = nn.LayerNorm(self._output_hidden)
        self.out_fc_2 = nn.Linear(
            self._output_hidden, model_args['gap'])
        # Prism特有: 对比正则化辅助投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(self._node_dim, self._node_dim * 2),
            nn.ReLU(),
            nn.Linear(self._node_dim * 2, self._node_dim))
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

    def _frequency_injection(self, x):
        """Prism特有: embedding后注入频率域特征"""
        B, L, N, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        # 只处理实部的低频成分
        freq_feat = self.freq_inject_real(x_freq.real)
        freq_reconstructed = torch.fft.irfft(
            torch.complex(freq_feat, x_freq.imag),
            n=L, dim=1)
        gate = torch.sigmoid(self.freq_inject_gate)
        _dbg("freq_inject_gate", gate, "model")
        return x + gate * freq_reconstructed

    def compute_contrastive_embeddings(self):
        """Prism特有: 计算用于对比学习的节点embedding投影"""
        proj_u = self.contrast_proj(self.node_emb_u)
        proj_u = F.normalize(proj_u, dim=-1)
        return proj_u

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
        # embedding + 频率域特征注入 (Prism特有)
        history_data = self.embedding(history_data)
        history_data = self._frequency_injection(history_data)
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
        # Prism特有: 视角注意力聚合
        weights = F.softmax(self.layer_view_attn, dim=0)
        _dbg("layer_agg_weights", weights, "model")
        dif_forecast_hidden = sum(
            weights[i] * dif_forecast_hidden_list[i]
            for i in range(self._num_layers))
        inh_forecast_hidden = sum(
            weights[i] * inh_forecast_hidden_list[i]
            for i in range(self._num_layers))
        forecast_hidden = (dif_forecast_hidden +
                           inh_forecast_hidden)
        # 输出头: GELU + LayerNorm (Prism特有)
        h = F.gelu(self.out_fc_1(forecast_hidden))
        h = self.out_ln(h)
        forecast = self.out_fc_2(F.gelu(h))
        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.range",
             f"[{forecast.min().item():.4f},"
             f"{forecast.max().item():.4f}]", "model")
        return forecast
