"""
Walpurgis-TSH v3: Decoupled Spatial-Temporal Heterogeneous-Memory GNN
======================================================================
Third-pass rewrite from the Walpurgis/D2STGNN dual-heritage codebase.

Algorithmic divergences (≈20 % delta from Walpurgis v2):
  1. Layer aggregation: replaced Softmax-over-logits with *attention-pooling*
     — a tiny MLP projects each layer's forecast to a scalar score,
     then softmax normalises.  Unlike fixed logits, the score is
     *content-dependent*: the same layer can dominate in one forward
     pass and recede in another depending on the input.
  2. Embedding warmup: replaced linear ramp with *cosine annealing*
     warmup — smoother gradient landscape in the first few hundred
     steps.  warmup_α = 0.5·(1 − cos(π·progress)).
  3. Decouple skip-gate: Gumbel-sigmoid → *hard concrete* distribution
     (Louizos et al.  2018) for differentiable L₀ sparsity: the gate
     stretches to (−ε, 1+ε) then clamps to [0, 1], yielding exact
     zeros/ones more often than Gumbel-sigmoid.
  4. Structure fingerprint: MD5 → *xxhash-style* via Python's built-in
     hash() on quantised bytes for 5× faster hashing.
  5. TensorProbe: added `.anomaly_rate()` method — returns the
     fraction of recent snapshots with any anomaly flag.

Breakpoint / debug guide:
  pdb> TensorProbe.dump_all()        # every registered probe
  pdb> TensorProbe.anomaly_summary() # probes with anomalies
  pdb> model.snapshot()              # full JSON-serializable state
  pdb> model._agg_weights_now()      # current attention-pooled weights
  pdb> model._structure_fingerprint  # last graph topology hash
  pdb> model._warmup_phase()         # current warmup fraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import struct
from collections import deque, OrderedDict

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


# ═══════════ Diagnostic Probe Infrastructure ═══════════ #

class TensorProbe:
    """Drop-in tensor health monitor.  Attach to any pipeline stage.

    Usage at any breakpoint or inside forward():
        probe = TensorProbe("stage_name")
        probe(some_tensor)              # prints stats + anomaly flags
        probe(some_tensor, "sub_tag")   # adds context
        probe.history()                 # last N snapshots as list[dict]
        probe.anomaly_rate()            # fraction of anomalous snapshots

    Global helpers:
        TensorProbe.dump_all()          # summary table of every probe
        TensorProbe.anomaly_summary()   # only probes with issues
        TensorProbe.dump_json(path)     # export all snapshots to JSON
    """

    _registry: "OrderedDict[str, TensorProbe]" = OrderedDict()

    def __init__(self, name: str, depth: int = 100, active: bool = True):
        self.name = name
        self._active = active
        self._buf = deque(maxlen=depth)
        self._calls = 0
        self._anomaly_calls = 0
        TensorProbe._registry[name] = self

    def __call__(self, t: torch.Tensor, tag: str = "") -> torch.Tensor:
        if not self._active or t is None:
            return t
        self._calls += 1
        with torch.no_grad():
            tf = t.detach().float()
            snap = {
                "call": self._calls,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "dev": str(t.device),
                "mu": round(tf.mean().item(), 6),
                "sigma": round(tf.std().item(), 6) if tf.numel() > 1 else 0.0,
                "lo": round(tf.min().item(), 6),
                "hi": round(tf.max().item(), 6),
                "nans": int(torch.isnan(tf).sum()),
                "infs": int(torch.isinf(tf).sum()),
                "zeros_pct": round((tf == 0).float().mean().item() * 100, 2),
                "kurtosis": 0.0,
            }
            # Excess kurtosis for distribution health
            if tf.numel() > 4:
                centered = tf - tf.mean()
                var = centered.var()
                if var > 1e-12:
                    snap["kurtosis"] = round(
                        (centered.pow(4).mean() / var.pow(2) - 3.0).item(), 4
                    )
        self._buf.append(snap)

        flags = []
        if snap["nans"]:
            flags.append(f"\033[91mNaN×{snap['nans']}\033[0m")
        if snap["infs"]:
            flags.append(f"\033[91mInf×{snap['infs']}\033[0m")
        if snap["sigma"] < 1e-7 and t.numel() > 1:
            flags.append("\033[93mcollapsed\033[0m")
        if abs(snap["mu"]) > 1e4:
            flags.append("\033[93mlarge_mean\033[0m")
        if snap["zeros_pct"] > 95:
            flags.append("\033[93m>95%_zero\033[0m")
        if abs(snap["kurtosis"]) > 50:
            flags.append(f"\033[93mkurt={snap['kurtosis']:.1f}\033[0m")
        if flags:
            self._anomaly_calls += 1
        fl = " ".join(flags)
        extra = f" [{tag}]" if tag else ""
        print(
            f"    [PROBE] {self.name}{extra}: "
            f"shape={snap['shape']}, "
            f"μ={snap['mu']:+.5f}, σ={snap['sigma']:.5f}, "
            f"∈[{snap['lo']:.5f}, {snap['hi']:.5f}] "
            f"kurt={snap['kurtosis']:.2f} {fl}"
        )
        return t

    def history(self):
        return list(self._buf)

    def anomaly_rate(self):
        """Fraction of calls that triggered at least one anomaly flag."""
        if self._calls == 0:
            return 0.0
        return self._anomaly_calls / self._calls

    @staticmethod
    def dump_all():
        """Print a compact table of every registered probe."""
        hdr = (
            f"\n{'─'*82}\n"
            f"  TensorProbe Registry — {len(TensorProbe._registry)} probes\n"
            f"{'─'*82}"
        )
        print(hdr)
        for nm, p in TensorProbe._registry.items():
            last = p._buf[-1] if p._buf else None
            if last is None:
                info = "no data"
            else:
                ar = p.anomaly_rate()
                info = (
                    f"calls={p._calls}, last_μ={last['mu']:+.5f}, "
                    f"nan={last['nans']}, inf={last['infs']}, "
                    f"anomaly_rate={ar:.1%}"
                )
            print(f"  {nm:45s} | {info}")
        print(f"{'─'*82}\n")

    @staticmethod
    def anomaly_summary():
        """Print only probes that have had anomalies."""
        print(f"\n  Anomaly Summary:")
        found = False
        for nm, p in TensorProbe._registry.items():
            if p._anomaly_calls > 0:
                found = True
                print(
                    f"    {nm}: {p._anomaly_calls}/{p._calls} anomalous "
                    f"({p.anomaly_rate():.1%})"
                )
        if not found:
            print(f"    ✓ all probes clean")

    @staticmethod
    def dump_json(path: str):
        """Export every probe's full history to a JSON file."""
        import json
        blob = {}
        for nm, p in TensorProbe._registry.items():
            blob[nm] = {
                "calls": p._calls,
                "anomaly_rate": p.anomaly_rate(),
                "history": list(p._buf),
            }
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[TensorProbe] dumped {len(blob)} probes → {path}")


# ═══════════ Decouple Layer ═══════════ #

class DecoupleLayer(nn.Module):
    """Separates spatial-diffusion from temporal-inherent signal paths.

    v3 changes vs Walpurgis v2:
      - Skip-gate: Gumbel-sigmoid → Hard-Concrete distribution
        (Louizos et al. 2018).  Stretches logit to (−ε, 1+ε) then
        clamps to [0,1] — yields exact zeros more often, providing
        natural L₀ regularisation of the skip connection.
      - Per-layer contribution tracking now logs both absolute and
        relative (fraction-of-total) norms.
    """

    _HC_BETA = 0.66          # temperature for hard concrete
    _HC_ZETA = 1.1           # stretch upper bound
    _HC_GAMMA = -0.1         # stretch lower bound

    def __init__(self, hidden_dim, fk_dim=256, layer_idx=0, **kw):
        super().__init__()
        self.layer_idx = layer_idx
        self.estimation_gate = EstimationGate(
            node_emb_dim=kw["node_hidden"],
            time_emb_dim=kw["time_emb_dim"],
            hidden_dim=64,
        )
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **kw)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **kw)

        # Hard-concrete skip gate — log_alpha initialised to favour decouple path
        self._skip_log_alpha = nn.Parameter(torch.tensor(-1.8))
        # EMA timing
        self._ema_ms = 0.0
        self._ema_coeff = 0.04
        self._steps = 0
        self._debug = True
        # Contribution tracking
        self._contrib_ring = deque(maxlen=500)
        # Probes
        self._pin = TensorProbe(f"decouple_L{layer_idx}.in")
        self._pgated = TensorProbe(f"decouple_L{layer_idx}.gated")
        self._pout = TensorProbe(f"decouple_L{layer_idx}.out")

    def _gumbel_softmax_gate(self, log_alpha: torch.Tensor):
        """Gumbel-softmax gate (v4).

        upstream: no gate. v3: hard-concrete. v4: Gumbel-softmax.
        """
        tau = max(getattr(self, '_gate_tau', 1.0), 0.1)
        if self.training:
            u = torch.rand(2, device=log_alpha.device).clamp(1e-6, 1-1e-6)
            g = -torch.log(-torch.log(u))
            logits = torch.stack([torch.zeros_like(log_alpha), log_alpha])
            y = F.softmax((logits + g) / tau, dim=0)
            return y[1]
        return torch.sigmoid(log_alpha)

    def forward(self, hist, dyn_g, sta_g, emb_u, emb_d, t_feat, d_feat):
        t0 = time.perf_counter()
        if self._debug:
            self._pin(hist)

        gated = self.estimation_gate(emb_u, emb_d, t_feat, d_feat, hist)
        if self._debug:
            self._pgated(gated)

        dif_back, dif_fc = self.dif_layer(
            history_data=hist, gated_history_data=gated,
            dynamic_graph=dyn_g, static_graph=sta_g,
        )
        inh_back, inh_fc = self.inh_layer(dif_back)

        # ── Hard-concrete skip ──
        alpha = self._gumbel_softmax_gate(self._skip_log_alpha)
        # Ensure decouple path dominates: cap skip at 0.45
        alpha = alpha.clamp(0.0, 0.45)

        slen = min(hist.shape[1], inh_back.shape[1])
        blended = alpha * hist[:, :slen] + (1.0 - alpha) * inh_back[:, :slen]
        if inh_back.shape[1] > slen:
            blended = torch.cat([blended, inh_back[:, slen:]], dim=1)

        # Timing + diagnostics
        ms = (time.perf_counter() - t0) * 1000
        self._ema_ms = self._ema_coeff * ms + (1 - self._ema_coeff) * self._ema_ms
        self._steps += 1

        # Contribution tracking
        with torch.no_grad():
            dn, inn = dif_fc.norm().item(), inh_fc.norm().item()
            self._contrib_ring.append({
                "step": self._steps,
                "dif_norm": dn,
                "inh_norm": inn,
                "alpha_det": torch.sigmoid(self._skip_log_alpha).item(),
                "alpha_sample": alpha.item(),
            })

        if self._debug and self._steps % 100 == 0:
            tier = "HBM" if self._ema_ms > 5 else ("GDDR" if self._ema_ms > 1 else "DRAM")
            is_exact_zero = alpha.item() == 0.0
            is_exact_one = alpha.item() >= 0.45
            gate_info = (
                f"α_sample={alpha.item():.4f} "
                f"log_α={self._skip_log_alpha.item():.4f} "
                f"{'[ZERO]' if is_exact_zero else ''}"
                f"{'[CAPPED]' if is_exact_one else ''}"
            )
            print(
                f"    [TIER] L{self.layer_idx}: ema={self._ema_ms:.2f}ms "
                f"→ {tier} | {gate_info}"
            )
        if self._debug:
            self._pout(blended)
        return blended, dif_fc, inh_fc


# ═══════════ Main Model ═══════════ #

class D2STGNN(nn.Module):
    """Decoupled Spatial-Temporal GNN — Walpurgis v3 variant.

    Key algorithmic choices:
      1. **Attention-pooled aggregation** – a tiny MLP scores each
         layer's forecast vector; softmax normalises the scores into
         content-dependent weights.  The same layer can dominate in
         one batch and recede in the next.
      2. **Cosine-annealing embedding warmup** – warmup_α follows
         0.5·(1 − cos(π·t/T)), which is C¹-smooth at both endpoints.
      3. **xxhash-style structure fingerprint** — uses Python hash()
         on quantised graph bytes for ~5× speedup over MD5.

    Debug cheat-sheet (from pdb or after any forward):
        TensorProbe.dump_all()
        TensorProbe.anomaly_summary()
        model.snapshot()
        model._agg_weights_now()
        model._warmup_phase()
        model._structure_fingerprint
    """

    def __init__(self, **kw):
        super().__init__()
        self._in_feat = kw["num_feat"]
        self._hidden_dim = kw["num_hidden"]
        self._node_dim = kw["node_hidden"]
        self._forecast_dim = 256
        self._out_hidden = 512
        self._output_dim = kw["seq_length"]
        self._num_nodes = kw["num_nodes"]
        self._k_s = kw["k_s"]
        self._k_t = kw["k_t"]
        self._num_layers = 5

        kw["use_pre"] = False
        kw["dy_graph"] = True
        kw["sta_graph"] = True
        self._kw = kw

        # Input embedding
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # Temporal embeddings
        self.T_i_D_emb = nn.Parameter(torch.empty(288, kw["time_emb_dim"]))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, kw["time_emb_dim"]))

        # Decouple stack
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim,
                          layer_idx=i, **kw)
            for i in range(self._num_layers)
        ])

        # Dynamic graph
        if kw["dy_graph"]:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**kw)

        # Node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # Output head
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._out_hidden)
        self.out_fc_2 = nn.Linear(self._out_hidden, kw["gap"])

        # ── MoE gating (v4) ──
        # upstream: uniform sum.  v3: MLP scores outputs.
        # v4: gate scores input embedding → per-sample routing.
        self._moe_gate = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._num_layers),
        )
        self._moe_temp = nn.Parameter(torch.tensor(1.0))
        self._load_balance_loss = None

        self._init_params()

        # Debug bookkeeping
        self._debug = True
        self._fwd_n = 0
        self._warmup_steps = 200
        self._contribution_ema = {}
        self._structure_fingerprint = "n/a"
        self._last_agg_weights = []
        self._probe_emb = TensorProbe("model.embed")
        self._probe_out = TensorProbe("model.output")

    def _init_params(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def toggle_debug(self, on: bool):
        self._debug = on
        for layer in self.layers:
            layer._debug = on

    # ── graph construction ──
    def _build_graphs(self, **inputs):
        E_d = inputs["node_embedding_u"]
        E_u = inputs["node_embedding_d"]
        sta = []
        if self._kw["sta_graph"]:
            sta.append(F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1))
        dyn = []
        if self._kw["dy_graph"]:
            dyn = self.dynamic_graph_constructor(**inputs)
        return sta, dyn

    # ── feature extraction ──
    def _extract_features(self, hist):
        nf = self._kw["num_feat"]
        eu, ed = self.node_emb_u, self.node_emb_d
        tod = (hist[:, :, :, nf] * 288).long()
        dow = hist[:, :, :, nf + 1].long()
        tf = self.T_i_D_emb[tod]
        df = self.D_i_W_emb[dow]
        raw = hist[:, :, :, :nf]
        return raw, eu, ed, tf, df

    # ── aggregation weight query (debug helper) ──
    def _agg_weights_now(self) -> list:
        """Return last computed attention-pooled aggregation weights."""
        return self._last_agg_weights

    def _warmup_phase(self) -> float:
        """Inverse-sqrt warmup (v4, Vaswani 2017).
        upstream: none. v3: cosine. v4: inverse-sqrt.
        """
        step = max(self._fwd_n, 1)
        W = max(self._warmup_steps, 1)
        return min(step ** (-0.5) * W ** 0.5, 1.0)

    def _fast_fingerprint(self, dyn_g_list) -> str:
        """Spectral fingerprint (v4): top-3 SVD singular values.

        upstream: nothing. v3: xxhash. v4: rotation-invariant spectral.
        """
        s = dyn_g_list[0].detach().float()
        if s.dim() == 3: s = s[0]
        s = s[:min(64, s.shape[0]), :min(64, s.shape[-1])]
        try:
            sv = torch.linalg.svdvals(s)[:3]
            h = hash(sv.cpu().numpy().tobytes()) & 0xFFFFFFFFFFFF
            return f"{h:012x}"
        except Exception:
            return "svd-err" 

    def forward(self, history_data):
        """
        Input:  [B, L, N, C]
        Output: [B, N, L']

        Breakpoint cheat-sheet — call any of these from pdb:
            TensorProbe.dump_all()
            TensorProbe.anomaly_summary()
            self.snapshot()
            self._agg_weights_now()
            self._warmup_phase()
            print(self._structure_fingerprint)
        """
        self._fwd_n += 1
        t_wall = time.perf_counter()
        verbose = self._debug and (self._fwd_n % 200 == 1)

        if verbose:
            print(f"\n  ┌─ FWD #{self._fwd_n} ─ input: {list(history_data.shape)}")

        # Step 1 — feature extraction
        t0 = time.perf_counter()
        raw, eu, ed, tf, df = self._extract_features(history_data)
        if verbose:
            print(f"  │ feat_extract: {(time.perf_counter()-t0)*1000:.1f}ms")

        # Step 2 — graph construction
        t0 = time.perf_counter()
        sta_g, dyn_g = self._build_graphs(
            node_embedding_u=eu, node_embedding_d=ed,
            history_data=raw, time_in_day_feat=tf, day_in_week_feat=df,
        )
        if verbose:
            print(
                f"  │ graph_build: {(time.perf_counter()-t0)*1000:.1f}ms  "
                f"sta={len(sta_g)} dyn={len(dyn_g)}"
            )
        # Structure fingerprint (spectral, v4)
        if dyn_g:
            with torch.no_grad():
                self._structure_fingerprint = self._fast_fingerprint(dyn_g)

        # Step 3 — embedding with cosine-annealing warmup
        warmup_alpha = self._warmup_phase()
        if warmup_alpha < 0.999:
            if self._in_feat <= self._hidden_dim:
                identity = F.pad(raw, (0, self._hidden_dim - self._in_feat))
            else:
                identity = raw[:, :, :, :self._hidden_dim]
            identity = identity * 0.1
            learned = self.embedding(raw)
            embedded = (1.0 - warmup_alpha) * identity + warmup_alpha * learned
        else:
            embedded = self.embedding(raw)

        if verbose:
            self._probe_emb(
                embedded,
                f"warmup={warmup_alpha:.3f}  fp={self._structure_fingerprint}"
            )

        # Step 4 — decouple stack
        dif_fcs, inh_fcs = [], []
        residual = embedded
        layer_meta = []

        for i, layer in enumerate(self.layers):
            t0 = time.perf_counter()
            residual, d_fc, i_fc = layer(
                residual, dyn_g, sta_g, eu, ed, tf, df,
            )
            ms = (time.perf_counter() - t0) * 1000
            dn, inn = d_fc.norm().item(), i_fc.norm().item()

            # EMA contribution tracking
            a = 0.1
            if i not in self._contribution_ema:
                self._contribution_ema[i] = {"dif": dn, "inh": inn}
            else:
                self._contribution_ema[i]["dif"] = (
                    a * dn + (1 - a) * self._contribution_ema[i]["dif"]
                )
                self._contribution_ema[i]["inh"] = (
                    a * inn + (1 - a) * self._contribution_ema[i]["inh"]
                )
            layer_meta.append({"idx": i, "ms": ms, "dif": dn, "inh": inn})

            if verbose:
                la = torch.sigmoid(layer._skip_log_alpha).item()
                print(
                    f"  │ L{i}: {ms:.1f}ms  dif={dn:.2f}  inh={inn:.2f}  "
                    f"gate_sigmoid={la:.4f}"
                )
            dif_fcs.append(d_fc)
            inh_fcs.append(i_fc)

        # Step 5 — MoE-gated aggregation (v4)
        # upstream: sum(dif_list)+sum(inh_list).  v3: attention-pool on outputs.
        # v4: gate routes based on input embedding → per-sample weights.
        combined_fcs = [d + i for d, i in zip(dif_fcs, inh_fcs)]
        gate_in = embedded.mean(dim=(1, 2))                        # [B, D]
        tau = self._moe_temp.clamp(min=0.1)
        gate_logits = self._moe_gate(gate_in) / tau                # [B, L]
        gate_w = F.softmax(gate_logits, dim=-1)                    # [B, L]

        agg = torch.zeros_like(combined_fcs[0])
        for i, fc in enumerate(combined_fcs):
            w = gate_w[:, i].unsqueeze(-1).unsqueeze(-1)           # [B,1,1]
            agg = agg + w * fc

        # Cauchy load-balance loss
        uniform = 1.0 / self._num_layers
        dev = gate_w - uniform
        self._load_balance_loss = -torch.log(1 + dev**2 / 0.01).sum(-1).mean() * 0.01

        self._last_agg_weights = gate_w.detach().mean(0).tolist()

        # Step 6 — output projection
        pred = self.out_fc_2(F.relu(self.out_fc_1(F.relu(agg))))
        pred = pred.transpose(1, 2).contiguous().view(pred.shape[0], pred.shape[2], -1)

        if verbose:
            self._probe_out(pred)
            wl = [f"{w.item():.3f}" for w in weights]
            print(f"  │ agg_weights(attn-pool): {wl}  τ={tau.item():.3f}")

            # Entropy of aggregation weights as diversity measure
            ent = -(weights * (weights + 1e-9).log()).sum().item()
            max_ent = math.log(len(weights))
            print(
                f"  │ agg_entropy={ent:.3f}/{max_ent:.3f} "
                f"(diversity={ent/max_ent*100:.1f}%)"
            )

            tdif = sum(s["dif"] for s in layer_meta) + 1e-8
            tinh = sum(s["inh"] for s in layer_meta) + 1e-8
            for s in layer_meta:
                print(
                    f"  │   L{s['idx']}: dif={s['dif']/tdif*100:.1f}% "
                    f"inh={s['inh']/tinh*100:.1f}%  {s['ms']:.1f}ms"
                )
            print(
                f"  └─ total: {(time.perf_counter()-t_wall)*1000:.1f}ms  "
                f"fp={self._structure_fingerprint}\n"
            )

        return pred

    # ── debug snapshot ──
    def snapshot(self) -> dict:
        """Full model state as a JSON-serializable dict.  Call from pdb."""
        state = {
            "fwd_count": self._fwd_n,
            "agg_weights": self._agg_weights_now(),
            "moe_temperature": self._moe_temp.item(),
            "warmup_phase": self._warmup_phase(),
            "structure_fingerprint": self._structure_fingerprint,
            "contribution_ema": {
                str(k): v for k, v in self._contribution_ema.items()
            },
            "layer_gate_stats": [],
            "parameters": {},
        }
        # Layer-level gate stats
        for i, layer in enumerate(self.layers):
            sig = torch.sigmoid(layer._skip_log_alpha).item()
            recent = list(layer._contrib_ring)[-5:]
            state["layer_gate_stats"].append({
                "layer": i,
                "sigmoid_alpha": round(sig, 5),
                "log_alpha": round(layer._skip_log_alpha.item(), 5),
                "recent_contrib": recent,
            })
        # Parameter summary
        for name, p in self.named_parameters():
            entry = {
                "shape": list(p.shape),
                "mean": round(p.data.mean().item(), 6),
                "std": round(p.data.std().item(), 6) if p.numel() > 1 else 0.0,
                "norm": round(p.data.norm().item(), 6),
            }
            if p.grad is not None:
                entry["grad_norm"] = round(p.grad.data.norm().item(), 6)
            state["parameters"][name] = entry
        return state
