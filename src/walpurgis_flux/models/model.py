"""
D2STGNN — Flux变体
算法改写 (~20%):
  1. DecoupleLayer: 滑动窗口流式处理 — 因果窗口截取,
     无需全序列, 训练时模拟流式推理行为
  2. D2STGNN: 渐进式解码(progressive decode) — forecast分两阶段:
     先粗预测(隔步采样)再细化(插值+修正), 粗到细多步
  3. D2STGNN: 因果卷积嵌入 — 替代线性嵌入, 用因果Conv1d
     捕获局部时序模式(只看过去,不看未来)
  4. 输出头: SiLU激活 + LayerNorm + 粗细双路输出
  5. 流式特征缓存: 缓存已计算的中间特征避免重复计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate
from .. import _dbg, dataflow_checkpoint, dump_struct_state


class CausalConvEmbedding(nn.Module):
    """Flux特有: 因果卷积嵌入层
    用因果Conv1d替代线性映射, 捕获局部时序依赖
    padding=(kernel_size-1)确保因果性(只看过去)
    比Linear embedding多了时序上下文感知能力
    """
    def __init__(self, in_feat, hidden_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_feat, hidden_dim, kernel_size,
            padding=0, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, L, N, C] -> 对每个node做因果卷积
        B, L, N, C = x.shape
        # reshape: [B*N, C, L] for Conv1d
        x_conv = x.permute(0, 2, 3, 1).reshape(B * N, C, L)
        # 因果padding: 左侧填充
        x_padded = F.pad(x_conv, (self.causal_pad, 0))
        out = self.conv(x_padded)  # [B*N, hidden, L]
        out = out.reshape(B, N, -1, L).permute(0, 3, 1, 2)
        # LayerNorm
        out = self.norm(out)
        _dbg("causal_conv_embed", out, "model")
        return out


class StreamingWindowBuffer:
    """Flux特有: 流式特征缓存
    维护一个滑动窗口缓冲区, 缓存已计算的层间特征
    当新时间步到达时, 只需计算增量部分
    """
    def __init__(self, window_size, num_layers):
        self.window_size = window_size
        self.num_layers = num_layers
        self._cache = {}
        self._step = 0

    def get_cached(self, layer_idx):
        key = f"layer_{layer_idx}"
        return self._cache.get(key, None)

    def update_cache(self, layer_idx, features):
        key = f"layer_{layer_idx}"
        if key in self._cache:
            # 滑动窗口: 拼接新特征, 截取最近window_size步
            cached = self._cache[key]
            combined = torch.cat(
                [cached, features], dim=1)
            self._cache[key] = combined[
                :, -self.window_size:, :, :]
        else:
            self._cache[key] = features[
                :, -self.window_size:, :, :]
        self._step += 1

    def clear(self):
        self._cache.clear()
        self._step = 0


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256,
                 layer_idx=0, stream_window=8,
                 **model_args):
        super().__init__()
        self.layer_idx = layer_idx
        self.stream_window = stream_window
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
        # Flux: 流式窗口注意力权重 — 对窗口内位置做可学习衰减
        self.window_decay = nn.Parameter(
            torch.linspace(0.5, 1.0, stream_window))
        self._gap = model_args.get('gap', 3)
        self._seq_length = model_args.get('seq_length', 12)

    def forward(self, history_data, dynamic_graph,
                static_graph, node_embedding_u,
                node_embedding_d, time_in_day_feat,
                day_in_week_feat):
        dataflow_checkpoint(
            f"decouple_L{self.layer_idx}.input",
            history_data)
        # Flux: 滑动窗口截取 — 模拟流式推理
        # 只取最近stream_window步, 而非使用全序列
        L = history_data.shape[1]
        if L > self.stream_window:
            # 窗口内位置加权衰减: 越近的时间步权重越高
            window_data = history_data[
                :, -self.stream_window:, :, :]
            decay = torch.sigmoid(self.window_decay)
            decay = decay.view(1, -1, 1, 1).to(
                window_data.device)
            # 对窗口截取数据做衰减加权
            effective_len = min(
                self.stream_window, window_data.shape[1])
            decay_slice = decay[:, -effective_len:, :, :]
            window_data = window_data * decay_slice
            _dbg(f"decouple_L{self.layer_idx}.stream_window",
                 f"truncated {L}->{effective_len}", "model")
        else:
            window_data = history_data
        gated_history_data = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat[:, -window_data.shape[1]:, :, :],
            day_in_week_feat[:, -window_data.shape[1]:, :, :],
            window_data)
        dif_backcast_seq_res, dif_forecast_hidden = \
            self.dif_layer(
                history_data=window_data,
                gated_history_data=gated_history_data,
                dynamic_graph=dynamic_graph,
                static_graph=static_graph)
        inh_backcast_seq_res, inh_forecast_hidden = \
            self.inh_layer(dif_backcast_seq_res)
        _dbg(f"decouple_L{self.layer_idx}.dif_energy",
             dif_forecast_hidden.detach().norm(), "model")
        _dbg(f"decouple_L{self.layer_idx}.inh_energy",
             inh_forecast_hidden.detach().norm(), "model")
        return (inh_backcast_seq_res,
                dif_forecast_hidden,
                inh_forecast_hidden)


class ProgressiveDecoder(nn.Module):
    """Flux特有: 渐进式解码头
    两阶段预测:
      阶段1 (粗): 用forecast_hidden做低分辨率预测(隔步)
      阶段2 (细): 在粗预测基础上插值并修正, 得到全分辨率
    比单步预测更稳定, 减少误差累积
    """
    def __init__(self, forecast_dim, output_hidden, gap,
                 coarse_factor=2):
        super().__init__()
        self.gap = gap
        self.coarse_factor = coarse_factor
        # 粗预测分支
        self.coarse_fc = nn.Linear(
            forecast_dim, output_hidden)
        self.coarse_out = nn.Linear(
            output_hidden,
            max(gap // coarse_factor, 1))
        # 细化修正分支
        self.refine_fc = nn.Linear(
            forecast_dim + max(gap // coarse_factor, 1),
            output_hidden)
        self.refine_out = nn.Linear(output_hidden, gap)
        self.refine_norm = nn.LayerNorm(output_hidden)

    def forward(self, forecast_hidden):
        B, L, N, D = forecast_hidden.shape
        # 阶段1: 粗预测
        coarse_h = F.silu(self.coarse_fc(forecast_hidden))
        coarse_pred = self.coarse_out(coarse_h)
        # 阶段2: 用粗预测作为条件, 生成细预测
        refine_input = torch.cat(
            [forecast_hidden, coarse_pred], dim=-1)
        refine_h = self.refine_fc(refine_input)
        refine_h = self.refine_norm(refine_h)
        fine_pred = self.refine_out(F.silu(refine_h))
        _dbg("progressive.coarse_range",
             f"[{coarse_pred.min().item():.4f},"
             f"{coarse_pred.max().item():.4f}]", "model")
        _dbg("progressive.fine_range",
             f"[{fine_pred.min().item():.4f},"
             f"{fine_pred.max().item():.4f}]", "model")
        return coarse_pred, fine_pred


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
        # Flux: 因果卷积嵌入替代线性映射
        self.embedding = CausalConvEmbedding(
            self._in_feat, self._hidden_dim, kernel_size=3)
        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))
        # Flux: 流式推理窗口大小 (每层可以不同)
        stream_windows = [
            max(6, 12 - i) for i in range(self._num_layers)]
        self.layers = nn.ModuleList([
            DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim,
                layer_idx=i,
                stream_window=stream_windows[i],
                **model_args)
            for i in range(self._num_layers)
        ])
        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = \
                DynamicGraphConstructor(**model_args)
        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        # Flux: 渐进式解码头替代简单FC输出
        self.progressive_decoder = ProgressiveDecoder(
            self._forecast_dim, self._output_hidden,
            model_args['gap'], coarse_factor=2)
        # 层聚合: 指数衰减权重(越深层越重要)
        self.layer_weight_logits = nn.Parameter(
            torch.linspace(-0.5, 0.5, self._num_layers))
        # 流式特征缓存
        self._stream_buffer = StreamingWindowBuffer(
            window_size=12, num_layers=self._num_layers)
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
        # Flux: 因果卷积嵌入
        history_data = self.embedding(history_data)
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
        # Flux: 指数衰减层聚合
        weights = F.softmax(self.layer_weight_logits, dim=0)
        _dbg("layer_agg_weights", weights, "model")
        dif_forecast_hidden = sum(
            weights[i] * dif_forecast_hidden_list[i]
            for i in range(self._num_layers))
        inh_forecast_hidden = sum(
            weights[i] * inh_forecast_hidden_list[i]
            for i in range(self._num_layers))
        forecast_hidden = (dif_forecast_hidden +
                           inh_forecast_hidden)
        # Flux: 渐进式解码 (粗→细)
        coarse_pred, fine_pred = \
            self.progressive_decoder(forecast_hidden)
        # 使用细预测作为最终输出
        forecast = fine_pred
        forecast = forecast.transpose(1, 2).contiguous()
        forecast = forecast.view(
            forecast.shape[0], forecast.shape[1], -1)
        dataflow_checkpoint("model.output", forecast)
        _dbg("output.fine_range",
             f"[{fine_pred.min().item():.4f},"
             f"{fine_pred.max().item():.4f}]", "model")
        # 保存coarse_pred用于progressive_refinement_loss
        self._last_coarse = coarse_pred
        return forecast
