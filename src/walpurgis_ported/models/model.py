"""
Walpurgis-TSH v2: Decoupled Spatial-Temporal Heterogeneous-Memory GNN
======================================================================
Re-ported from the Walpurgis/D2STGNN dual-heritage codebase.

Algorithmic divergences (≈20 % delta from the prior Walpurgis port):
  1. Layer aggregation: replaced exponential-decay with *Softmax-over-
     learnable-logits* — each layer owns a raw logit that participates
     in a global softmax, yielding a proper probability distribution
     over contributions.  The prior exp(-λ·i) assumed monotone decay;
     softmax lets any layer dominate if it earns it.
  2. Embedding warmup: replaced hard step-function bypass with a
     *linear interpolation ramp* — warmup_α goes from 0→1 over the
     warmup window so the learned embedding fades in smoothly.
  3. Decouple skip-gate: prior version used clamp(sigmoid, 0, 0.5);
     now uses a *Gumbel-sigmoid* during training for stochastic hard
     gating, with deterministic sigmoid at eval.
  4. Added per-forward *structure fingerprint* — a cheap hash of the
     dynamic graph topology that lets you detect silent graph collapse
     from a debugger.

Breakpoint / debug guide:
  pdb> TensorProbe.dump_all()        # every registered probe
  pdb> model.snapshot()               # full JSON-serializable state
  pdb> model._agg_weights_now()       # current softmax aggregation weights
  pdb> model._structure_fingerprint   # last graph topology hash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import hashlib
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

    Global helpers:
        TensorProbe.dump_all()          # summary table of every probe
        TensorProbe.dump_json(path)     # export all snapshots to JSON
    """

    _registry: "OrderedDict[str, TensorProbe]" = OrderedDict()

    def __init__(self, name: str, depth: int = 80, active: bool = True):
        self.name = name
        self._active = active
        self._buf = deque(maxlen=depth)
        self._calls = 0
        TensorProbe._registry[name] = self

    def __call__(self, t: torch.Tensor, tag: str = "") -> torch.Tensor:
        if not self._active or t is None:
            return t
        self._calls += 1
        with torch.no_grad():
            snap = {
                "call": self._calls,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "dev": str(t.device),
                "mu": round(t.mean().item(), 6),
                "sigma": round(t.std().item(), 6) if t.numel() > 1 else 0.0,
                "lo": round(t.min().item(), 6),
                "hi": round(t.max().item(), 6),
                "nans": int(torch.isnan(t).sum()),
                "infs": int(torch.isinf(t).sum()),
                "zeros_pct": round((t == 0).float().mean().item() * 100, 2),
            }
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
        fl = " ".join(flags)
        extra = f" [{tag}]" if tag else ""
        print(
            f"    [PROBE] {self.name}{extra}: "
            f"shape={snap['shape']}, "
            f"μ={snap['mu']:+.5f}, σ={snap['sigma']:.5f}, "
            f"∈[{snap['lo']:.5f}, {snap['hi']:.5f}] {fl}"
        )
        return t

    def history(self):
        return list(self._buf)

    @staticmethod
    def dump_all():
        """Print a compact table of every registered probe."""
        hdr = f"\n{'─'*78}\n  TensorProbe Registry — {len(TensorProbe._registry)} probes\n{'─'*78}"
        print(hdr)
        for nm, p in TensorProbe._registry.items():
            last = p._buf[-1] if p._buf else None
            if last is None:
                info = "no data"
            else:
                info = (
                    f"calls={p._calls}, last_μ={last['mu']:+.5f}, "
                    f"nan={last['nans']}, inf={last['infs']}"
                )
            print(f"  {nm:45s} | {info}")
        print(f"{'─'*78}\n")

    @staticmethod
    def dump_json(path: str):
        """Export every probe's full history to a JSON file."""
        import json
        blob = {}
        for nm, p in TensorProbe._registry.items():
            blob[nm] = list(p._buf)
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[TensorProbe] dumped {len(blob)} probes → {path}")


# ═══════════ Decouple Layer ═══════════ #

class DecoupleLayer(nn.Module):
    """Separates spatial-diffusion from temporal-inherent signal paths.

    v2 changes vs prior Walpurgis:
      - Skip-gate uses Gumbel-sigmoid in training for stochastic hard
        routing; deterministic sigmoid in eval.
      - Per-layer contribution tracking records *relative* share, not
        raw norm (denominator = total contribution of all layers).
    """

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

        # Gumbel-sigmoid skip gate — initialised to bias toward decouple path
        self._skip_logit = nn.Parameter(torch.tensor(-1.5))
        # EMA timing
        self._ema_ms = 0.0
        self._ema_coeff = 0.04
        self._steps = 0
        self._debug = True
        # Probes
        self._pin = TensorProbe(f"decouple_L{layer_idx}.in")
        self._pgated = TensorProbe(f"decouple_L{layer_idx}.gated")
        self._pout = TensorProbe(f"decouple_L{layer_idx}.out")

    def _gumbel_sigmoid(self, logit: torch.Tensor, tau: float = 0.5):
        """Stochastic hard gate during training; deterministic at eval."""
        if self.training:
            u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            return torch.sigmoid((logit + gumbel) / tau)
        return torch.sigmoid(logit)

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

        # ── Gumbel-sigmoid skip ──
        alpha = self._gumbel_sigmoid(self._skip_logit)
        alpha = alpha.clamp(0.0, 0.45)   # decouple always dominates

        slen = min(hist.shape[1], inh_back.shape[1])
        blended = alpha * hist[:, :slen] + (1.0 - alpha) * inh_back[:, :slen]
        if inh_back.shape[1] > slen:
            blended = torch.cat([blended, inh_back[:, slen:]], dim=1)

        # Timing
        ms = (time.perf_counter() - t0) * 1000
        self._ema_ms = self._ema_coeff * ms + (1 - self._ema_coeff) * self._ema_ms
        self._steps += 1
        if self._debug and self._steps % 100 == 0:
            tier = "HBM" if self._ema_ms > 5 else ("GDDR" if self._ema_ms > 1 else "DRAM")
            print(
                f"    [TIER] L{self.layer_idx}: ema={self._ema_ms:.2f}ms "
                f"→ {tier} | gate_logit={self._skip_logit.item():.4f} "
                f"α_det={torch.sigmoid(self._skip_logit).item():.4f}"
            )
        if self._debug:
            self._pout(blended)
        return blended, dif_fc, inh_fc


# ═══════════ Main Model ═══════════ #

class D2STGNN(nn.Module):
    """Decoupled Spatial-Temporal GNN — Walpurgis v2 variant.

    Key algorithmic choices:
      1. **Softmax logit aggregation** – each of the 5 decouple layers
         owns a raw logit; a temperature-scaled softmax converts them
         into normalised weights.  This is a strict superset of the
         prior exp-decay approach: the network *can* learn monotone
         decay, but isn't forced to.
      2. **Linear-ramp embedding warmup** – the embedding output is
         interpolated as  (1-warmup_α)·pad(x) + warmup_α·embed(x),
         where warmup_α ramps from 0→1 over the warmup window.
      3. **Structure fingerprint** – a cheap MD5 of the dynamic graph
         tensor that lets you detect silent graph collapse or frozen
         topology from a debugger.

    Debug cheat-sheet (from pdb or after any forward):
        TensorProbe.dump_all()
        model.snapshot()
        model._agg_weights_now()
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

        # ── Softmax-logit aggregation weights ──
        self._agg_logits = nn.Parameter(torch.zeros(self._num_layers))
        self._agg_temp = nn.Parameter(torch.tensor(1.0))   # temperature

        self._init_params()

        # Debug bookkeeping
        self._debug = True
        self._fwd_n = 0
        self._warmup_steps = 200
        self._contribution_ema = {}
        self._structure_fingerprint = "n/a"
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
        """Return current softmax aggregation weights as a plain list."""
        tau = self._agg_temp.clamp(min=0.1)
        w = F.softmax(self._agg_logits / tau, dim=0)
        return [round(x.item(), 5) for x in w]

    def forward(self, history_data):
        """
        Input:  [B, L, N, C]
        Output: [B, N, L']

        Breakpoint cheat-sheet — call any of these from pdb:
            TensorProbe.dump_all()
            self.snapshot()
            self._agg_weights_now()
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
        # Structure fingerprint (cheap)
        if dyn_g:
            with torch.no_grad():
                sample = dyn_g[0].detach().cpu().float().numpy().tobytes()[:512]
                self._structure_fingerprint = hashlib.md5(sample).hexdigest()[:12]

        # Step 3 — embedding with linear-ramp warmup
        warmup_frac = min(self._fwd_n / max(self._warmup_steps, 1), 1.0)
        if warmup_frac < 1.0:
            if self._in_feat <= self._hidden_dim:
                identity = F.pad(raw, (0, self._hidden_dim - self._in_feat))
            else:
                identity = raw[:, :, :, :self._hidden_dim]
            identity = identity * 0.1
            learned = self.embedding(raw)
            embedded = (1.0 - warmup_frac) * identity + warmup_frac * learned
        else:
            embedded = self.embedding(raw)

        if verbose:
            self._probe_emb(
                embedded,
                f"warmup={warmup_frac:.2f}  fingerprint={self._structure_fingerprint}"
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
                gl = layer._skip_logit.item()
                print(
                    f"  │ L{i}: {ms:.1f}ms  dif={dn:.2f}  inh={inn:.2f}  "
                    f"gate_logit={gl:.4f}"
                )
            dif_fcs.append(d_fc)
            inh_fcs.append(i_fc)

        # Step 5 — softmax-logit aggregation
        tau = self._agg_temp.clamp(min=0.1)
        weights = F.softmax(self._agg_logits / tau, dim=0)  # [num_layers]

        agg_dif = sum(w * h for w, h in zip(weights, dif_fcs))
        agg_inh = sum(w * h for w, h in zip(weights, inh_fcs))
        combined = agg_dif + agg_inh

        # Step 6 — output projection
        pred = self.out_fc_2(F.relu(self.out_fc_1(F.relu(combined))))
        pred = pred.transpose(1, 2).contiguous().view(pred.shape[0], pred.shape[2], -1)

        if verbose:
            self._probe_out(pred)
            wl = [f"{w.item():.3f}" for w in weights]
            print(f"  │ agg_weights(softmax): {wl}  τ={tau.item():.3f}")
            tdif = sum(s["dif"] for s in layer_meta) + 1e-8
            tinh = sum(s["inh"] for s in layer_meta) + 1e-8
            for s in layer_meta:
                print(
                    f"  │   L{s['idx']}: dif={s['dif']/tdif*100:.1f}% "
                    f"inh={s['inh']/tinh*100:.1f}%  {s['ms']:.1f}ms"
                )
            print(f"  └─ total: {(time.perf_counter()-t_wall)*1000:.1f}ms  "
                  f"fp={self._structure_fingerprint}\n")

        return pred

    # ── debug snapshot ──
    def snapshot(self) -> dict:
        """Full model state as a JSON-serializable dict.  Call from pdb."""
        state = {
            "fwd_count": self._fwd_n,
            "agg_weights": self._agg_weights_now(),
            "agg_temperature": self._agg_temp.item(),
            "warmup_frac": min(self._fwd_n / max(self._warmup_steps, 1), 1.0),
            "structure_fingerprint": self._structure_fingerprint,
            "contribution_ema": dict(self._contribution_ema),
            "parameters": {},
        }
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
