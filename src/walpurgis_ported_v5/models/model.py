import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque, OrderedDict

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


# ═══════ Tensor Health Probe ═══════ #

class TensorProbe:
    """Attach to any pipeline stage. Call probe(tensor) to record stats."""

    _registry: "OrderedDict[str, TensorProbe]" = OrderedDict()

    def __init__(self, name: str, depth: int = 64, active: bool = True):
        self.name = name
        self._active = active
        self._buf = deque(maxlen=depth)
        self._calls = 0
        self._anomaly_count = 0
        TensorProbe._registry[name] = self

    def __call__(self, t: torch.Tensor, tag: str = "") -> torch.Tensor:
        if not self._active or t is None:
            return t
        self._calls += 1
        with torch.no_grad():
            tf = t.detach().float()
            snap = {
                "call":  self._calls,
                "tag":   tag,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "mu":    round(tf.mean().item(), 6),
                "sigma": round(tf.std().item(), 6) if tf.numel() > 1 else 0.0,
                "lo":    round(tf.min().item(), 6),
                "hi":    round(tf.max().item(), 6),
                "nans":  int(torch.isnan(tf).sum()),
                "infs":  int(torch.isinf(tf).sum()),
                "zeros_pct": round((tf == 0).float().mean().item() * 100, 2),
            }
            if tf.numel() > 4:
                centered = tf - tf.mean()
                var = centered.var()
                if var > 1e-12:
                    snap["kurtosis"] = round(
                        (centered.pow(4).mean() / var.pow(2) - 3.0).item(), 4)
                else:
                    snap["kurtosis"] = 0.0
            else:
                snap["kurtosis"] = 0.0

        has_issue = snap["nans"] > 0 or snap["infs"] > 0
        if has_issue:
            self._anomaly_count += 1
            print(f"\033[91m[PROBE:{self.name}] ANOMALY #{self._anomaly_count} "
                  f"NaN={snap['nans']} Inf={snap['infs']} shape={snap['shape']}\033[0m")
        self._buf.append(snap)
        return t

    def history(self):
        return list(self._buf)

    def anomaly_rate(self):
        if not self._buf:
            return 0.0
        return sum(1 for s in self._buf if s["nans"] or s["infs"]) / len(self._buf)

    @classmethod
    def dump_all(cls):
        print(f"{'probe':<28} {'calls':>6} {'anom%':>7} {'last_mu':>10} {'last_σ':>10}")
        print("-" * 65)
        for name, p in cls._registry.items():
            last = p._buf[-1] if p._buf else {}
            print(f"{name:<28} {p._calls:>6} {p.anomaly_rate()*100:>6.1f}% "
                  f"{last.get('mu', ''):>10} {last.get('sigma', ''):>10}")

    @classmethod
    def anomaly_summary(cls):
        for name, p in cls._registry.items():
            if p._anomaly_count > 0:
                print(f"[{name}] anomalies={p._anomaly_count}/{p._calls} "
                      f"rate={p.anomaly_rate()*100:.1f}%")


# ═══════ Decouple Layer ═══════ #

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(
            node_emb_dim=model_args['node_hidden'],
            time_emb_dim=model_args['time_emb_dim'],
            hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

        self._probe_gate = TensorProbe("decouple_gate", active=False)
        self._probe_dif  = TensorProbe("decouple_dif",  active=False)
        self._probe_inh  = TensorProbe("decouple_inh",  active=False)

    def forward(self, history_data, dynamic_graph, static_graph,
                node_embedding_u, node_embedding_d,
                time_in_day_feat, day_in_week_feat):
        gated = self.estimation_gate(
            node_embedding_u, node_embedding_d,
            time_in_day_feat, day_in_week_feat, history_data)
        gated = self._probe_gate(gated, "gate_out")

        dif_res, dif_fk = self.dif_layer(
            history_data=history_data, gated_history_data=gated,
            dynamic_graph=dynamic_graph, static_graph=static_graph)
        dif_fk = self._probe_dif(dif_fk, "dif_forecast")

        inh_res, inh_fk = self.inh_layer(dif_res)
        inh_fk = self._probe_inh(inh_fk, "inh_forecast")
        return inh_res, dif_fk, inh_fk


# ═══════ Main Model ═══════ #
# Delta vs upstream:
#   1. Layer aggregation: sum → attention-pooled (tiny MLP scores each layer)
#   2. Embedding init: xavier → truncated-normal (σ=0.02)
#   3. Output head: 2-layer ReLU → GELU + residual shortcut
#   4. Graph constructor: static softmax temperature learnable (τ)
#   5. TensorProbe wired into forward path

class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._forecast_dim  = 256
        self._output_hidden = 512
        self._output_dim    = model_args['seq_length']
        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 5

        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True
        self._model_args = model_args

        # input projection
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # temporal embeddings
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # decoupled layers
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)
            for _ in range(self._num_layers)
        ])

        # ── delta 1: attention pooling over layer forecasts ──
        self._agg_score_dif = nn.Sequential(
            nn.Linear(self._forecast_dim, 1, bias=False))
        self._agg_score_inh = nn.Sequential(
            nn.Linear(self._forecast_dim, 1, bias=False))

        # dynamic graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # ── delta 4: learnable softmax temperature for static graph ──
        self._tau = nn.Parameter(torch.tensor(1.0))

        # ── delta 3: output head with GELU + skip ──
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])
        self.out_skip = nn.Linear(self._forecast_dim, model_args['gap'])

        # probes
        self._probe_emb = TensorProbe("model_embedding", active=False)
        self._probe_agg = TensorProbe("model_agg_forecast", active=False)
        self._probe_out = TensorProbe("model_output", active=False)

        self.reset_parameter()

    def reset_parameter(self):
        # ── delta 2: truncated normal init ──
        for p in [self.node_emb_u, self.node_emb_d,
                  self.T_i_D_emb, self.D_i_W_emb]:
            nn.init.trunc_normal_(p, std=0.02)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            # ── delta 4: temperature-scaled softmax ──
            raw = torch.mm(E_d, E_u.T) / self._tau.clamp(min=0.01)
            static_graph = [F.softmax(F.relu(raw), dim=1)]
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
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def _attention_pool(self, hidden_list, score_fn):
        """Attention-weighted aggregation across layers instead of plain sum."""
        stacked = torch.stack(hidden_list, dim=0)           # [L, B, T, N, D]
        scores  = score_fn(stacked).squeeze(-1)             # [L, B, T, N]
        weights = F.softmax(scores, dim=0)                  # [L, B, T, N]
        pooled  = (stacked * weights.unsqueeze(-1)).sum(0)  # [B, T, N, D]
        return pooled

    def _agg_weights_now(self):
        """Debug helper: return last attention weights (call after forward)."""
        return getattr(self, '_last_agg_w', None)

    def forward(self, history_data):
        # prepare
        history_data, node_emb_u, node_emb_d, t_feat, d_feat = \
            self._prepare_inputs(history_data)

        # graph
        static_graph, dynamic_graph = self._graph_constructor(
            node_embedding_u=node_emb_u, node_embedding_d=node_emb_d,
            history_data=history_data,
            time_in_day_feat=t_feat, day_in_week_feat=d_feat)

        # embed
        history_data = self.embedding(history_data)
        history_data = self._probe_emb(history_data, "post_embed")

        dif_list = []
        inh_list = []
        residual = history_data
        for layer in self.layers:
            residual, dif_fk, inh_fk = layer(
                residual, dynamic_graph, static_graph,
                node_emb_u, node_emb_d, t_feat, d_feat)
            dif_list.append(dif_fk)
            inh_list.append(inh_fk)

        # ── delta 1: attention-pool instead of sum ──
        dif_agg = self._attention_pool(dif_list, self._agg_score_dif)
        inh_agg = self._attention_pool(inh_list, self._agg_score_inh)
        forecast_hidden = dif_agg + inh_agg
        forecast_hidden = self._probe_agg(forecast_hidden, "agg")

        # ── delta 3: GELU + residual shortcut ──
        main_path = self.out_fc_2(F.gelu(self.out_fc_1(F.gelu(forecast_hidden))))
        skip_path = self.out_skip(forecast_hidden)
        forecast  = main_path + skip_path

        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)
        forecast = self._probe_out(forecast, "final_out")
        return forecast

    def snapshot(self):
        """JSON-serialisable state for debugging."""
        return {
            "tau":        self._tau.item(),
            "emb_u_norm": self.node_emb_u.data.norm().item(),
            "emb_d_norm": self.node_emb_d.data.norm().item(),
            "num_params": sum(p.numel() for p in self.parameters()),
        }
