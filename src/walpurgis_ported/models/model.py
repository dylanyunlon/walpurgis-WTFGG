"""
Walpurgis-TSH v4: Decoupled Spatial-Temporal Heterogeneous-Memory GNN
======================================================================
Fourth-pass rewrite from the Walpurgis/D2STGNN dual-heritage codebase.

Algorithmic divergences (≈20 % delta from Walpurgis v3):
  1. Layer aggregation: attention-pooling → *Mixture-of-Experts (MoE)
     gating* with auxiliary load-balancing loss.  A gating network
     maps the combined forecast to per-expert probabilities; a Cauchy
     entropy term penalises load imbalance across layers, preventing
     collapse to a single dominant expert.
  2. Embedding warmup: cosine-annealing → *inverse-sqrt schedule*
     (Vaswani et al.): warmup_α = min(step^{-0.5}, step · W^{-1.5}).
     Provides faster initial ramp then slower decay — well-suited to
     transformer-style gradient landscapes.
  3. Decouple skip-gate: hard-concrete → *straight-through Gumbel-
     softmax* (Jang et al. 2017).  Produces differentiable discrete
     samples via the reparametrisation trick; temperature τ anneals
     from 1.0 → 0.1 over training, approaching true hard gating.
  4. Structure fingerprint: xxhash → *spectral hash* — hash of the
     top-3 singular values of a graph sample.  Captures topology
     changes that elementwise hashes miss (e.g. rotational symmetry).
  5. TensorProbe: added *gradient flow tracer* — hooks onto backward
     pass to track gradient magnitude at each probe point.  Also adds
     `.auto_bisect()` to locate the first pipeline stage where
     gradients vanish or explode.

Breakpoint / debug guide:
  pdb> TensorProbe.dump_all()          # every registered probe
  pdb> TensorProbe.anomaly_summary()   # probes with anomalies
  pdb> TensorProbe.grad_flow_report()  # gradient magnitude per probe
  pdb> TensorProbe.auto_bisect()       # locate gradient pathology
  pdb> model.snapshot()                # full JSON-serializable state
  pdb> model._moe_weights_now()        # current MoE gating weights
  pdb> model._structure_fingerprint    # last spectral graph hash
  pdb> model._warmup_alpha()           # current inverse-sqrt warmup
  pdb> model._load_balance_loss        # last auxiliary MoE loss
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
    """Drop-in tensor health monitor with gradient flow tracing.

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
        TensorProbe.grad_flow_report()  # gradient magnitudes per probe
        TensorProbe.auto_bisect()       # locate gradient pathology
    """

    _registry: "OrderedDict[str, TensorProbe]" = OrderedDict()

    def __init__(self, name: str, depth: int = 100, active: bool = True):
        self.name = name
        self._active = active
        self._buf = deque(maxlen=depth)
        self._calls = 0
        self._anomaly_calls = 0
        # v4: gradient flow tracing
        self._grad_norms = deque(maxlen=depth)
        self._last_grad_norm = None
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
                "skewness": 0.0,
            }
            # Excess kurtosis + skewness for distribution health
            if tf.numel() > 4:
                centered = tf - tf.mean()
                var = centered.var()
                if var > 1e-12:
                    std = var.sqrt()
                    snap["kurtosis"] = round(
                        (centered.pow(4).mean() / var.pow(2) - 3.0).item(), 4
                    )
                    snap["skewness"] = round(
                        (centered.pow(3).mean() / std.pow(3)).item(), 4
                    )
        self._buf.append(snap)

        # v4: register backward hook for gradient flow tracing
        if t.requires_grad and self._active:
            _probe_ref = self
            def _grad_hook(grad, ref=_probe_ref):
                with torch.no_grad():
                    gn = grad.float().norm().item()
                    ref._grad_norms.append(gn)
                    ref._last_grad_norm = gn
            try:
                t.register_hook(_grad_hook)
            except RuntimeError:
                pass  # leaf tensors or no grad

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
        if abs(snap["skewness"]) > 5:
            flags.append(f"\033[93mskew={snap['skewness']:.2f}\033[0m")
        if flags:
            self._anomaly_calls += 1
        fl = " ".join(flags)
        extra = f" [{tag}]" if tag else ""
        grad_info = ""
        if self._last_grad_norm is not None:
            grad_info = f" ∇={self._last_grad_norm:.5f}"
        print(
            f"    [PROBE] {self.name}{extra}: "
            f"shape={snap['shape']}, "
            f"μ={snap['mu']:+.5f}, σ={snap['sigma']:.5f}, "
            f"∈[{snap['lo']:.5f}, {snap['hi']:.5f}] "
            f"kurt={snap['kurtosis']:.2f} skew={snap['skewness']:.2f}"
            f"{grad_info} {fl}"
        )
        return t

    def history(self):
        return list(self._buf)

    def anomaly_rate(self):
        """Fraction of calls that triggered at least one anomaly flag."""
        if self._calls == 0:
            return 0.0
        return self._anomaly_calls / self._calls

    def grad_history(self):
        """Return gradient norm history for this probe."""
        return list(self._grad_norms)

    @staticmethod
    def dump_all():
        """Print a compact table of every registered probe."""
        hdr = (
            f"\n{'─'*90}\n"
            f"  TensorProbe Registry — {len(TensorProbe._registry)} probes\n"
            f"{'─'*90}"
        )
        print(hdr)
        for nm, p in TensorProbe._registry.items():
            last = p._buf[-1] if p._buf else None
            if last is None:
                info = "no data"
            else:
                ar = p.anomaly_rate()
                gn = f" ∇={p._last_grad_norm:.5f}" if p._last_grad_norm is not None else ""
                info = (
                    f"calls={p._calls}, last_μ={last['mu']:+.5f}, "
                    f"nan={last['nans']}, inf={last['infs']}, "
                    f"anomaly_rate={ar:.1%}{gn}"
                )
            print(f"  {nm:45s} | {info}")
        print(f"{'─'*90}\n")

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
    def grad_flow_report():
        """Print gradient magnitude at each probe point — detects vanishing/exploding grads."""
        print(f"\n  Gradient Flow Report:")
        probes_with_grads = [
            (nm, p) for nm, p in TensorProbe._registry.items()
            if p._last_grad_norm is not None
        ]
        if not probes_with_grads:
            print("    no gradient data collected yet")
            return
        for nm, p in probes_with_grads:
            gn = p._last_grad_norm
            bar_len = min(int(math.log10(gn + 1e-12) + 10), 30)
            bar_len = max(bar_len, 0)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            flag = ""
            if gn < 1e-7:
                flag = " \033[91m← VANISHING\033[0m"
            elif gn > 1e3:
                flag = " \033[91m← EXPLODING\033[0m"
            print(f"    {nm:40s} ∇={gn:.6f} [{bar}]{flag}")

    @staticmethod
    def auto_bisect():
        """Locate the first probe in the pipeline where gradients become pathological."""
        print(f"\n  Auto-Bisect Gradient Pathology:")
        probes = [(nm, p) for nm, p in TensorProbe._registry.items()
                  if p._last_grad_norm is not None]
        if len(probes) < 2:
            print("    need ≥2 probes with gradient data")
            return
        prev_gn = probes[0][1]._last_grad_norm
        for i in range(1, len(probes)):
            nm, p = probes[i]
            gn = p._last_grad_norm
            ratio = gn / (prev_gn + 1e-12)
            if ratio > 100:
                print(f"    ⚠ EXPLOSION between {probes[i-1][0]} and {nm}: "
                      f"ratio={ratio:.1f}×")
            elif ratio < 0.001:
                print(f"    ⚠ VANISHING between {probes[i-1][0]} and {nm}: "
                      f"ratio={ratio:.6f}×")
            prev_gn = gn
        print("    bisect complete")

    @staticmethod
    def dump_json(path: str):
        """Export every probe's full history to a JSON file."""
        import json
        blob = {}
        for nm, p in TensorProbe._registry.items():
            blob[nm] = {
                "calls": p._calls,
                "anomaly_rate": p.anomaly_rate(),
                "grad_norms": list(p._grad_norms),
                "history": list(p._buf),
            }
        with open(path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[TensorProbe] dumped {len(blob)} probes → {path}")


# ═══════════ Decouple Layer ═══════════ #

class DecoupleLayer(nn.Module):
    """Separates spatial-diffusion from temporal-inherent signal paths.

    v4 changes vs Walpurgis v3:
      - Skip-gate: hard-concrete → Straight-Through Gumbel-Softmax
        (Jang et al. 2017).  During training, samples from the
        Gumbel-softmax distribution with annealing temperature;
        during eval, uses argmax.  Provides gradient through discrete
        choices without the stretched-interval hack.
      - Contribution tracking now logs gradient norms alongside
        activation norms for full signal-flow analysis.
      - Gate temperature anneals from 1.0 → 0.1 over training.
    """

    _TAU_INIT = 1.0
    _TAU_MIN = 0.1
    _TAU_ANNEAL_STEPS = 5000

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

        # Straight-through Gumbel-softmax gate — 2 logits: [skip, decouple]
        self._gate_logits = nn.Parameter(torch.tensor([-1.5, 1.5]))
        # EMA timing
        self._ema_ms = 0.0
        self._ema_coeff = 0.04
        self._steps = 0
        self._debug = True
        # Contribution tracking (with gradient norms)
        self._contrib_ring = deque(maxlen=500)
        # Probes
        self._pin = TensorProbe(f"decouple_L{layer_idx}.in")
        self._pgated = TensorProbe(f"decouple_L{layer_idx}.gated")
        self._pout = TensorProbe(f"decouple_L{layer_idx}.out")

    def _current_tau(self):
        """Annealing temperature for Gumbel-softmax."""
        progress = min(self._steps / max(self._TAU_ANNEAL_STEPS, 1), 1.0)
        return self._TAU_INIT - (self._TAU_INIT - self._TAU_MIN) * progress

    def _gumbel_softmax_gate(self):
        """Straight-through Gumbel-softmax for differentiable discrete gating.

        Training: sample from Gumbel-softmax, apply ST estimator.
        Eval: deterministic softmax.
        Returns: scalar skip weight in [0, 1].
        """
        tau = self._current_tau()
        if self.training:
            # Gumbel noise
            u = torch.rand_like(self._gate_logits).clamp(1e-6, 1 - 1e-6)
            g = -torch.log(-torch.log(u))
            y_soft = F.softmax((self._gate_logits + g) / tau, dim=0)
            # Straight-through: hard in forward, soft in backward
            idx = y_soft.argmax()
            y_hard = torch.zeros_like(y_soft)
            y_hard[idx] = 1.0
            y_out = (y_hard - y_soft).detach() + y_soft
            alpha = y_out[0]  # skip weight
        else:
            probs = F.softmax(self._gate_logits / tau, dim=0)
            alpha = probs[0]
        # Cap skip weight to ensure decouple path dominates
        return alpha.clamp(0.0, 0.45)

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

        # ── Straight-through Gumbel-softmax skip ──
        alpha = self._gumbel_softmax_gate()

        slen = min(hist.shape[1], inh_back.shape[1])
        blended = alpha * hist[:, :slen] + (1.0 - alpha) * inh_back[:, :slen]
        if inh_back.shape[1] > slen:
            blended = torch.cat([blended, inh_back[:, slen:]], dim=1)

        # Timing + diagnostics
        ms = (time.perf_counter() - t0) * 1000
        self._ema_ms = self._ema_coeff * ms + (1 - self._ema_coeff) * self._ema_ms
        self._steps += 1

        # Contribution tracking with gradient norms
        with torch.no_grad():
            dn, inn = dif_fc.norm().item(), inh_fc.norm().item()
            probs = F.softmax(self._gate_logits, dim=0)
            self._contrib_ring.append({
                "step": self._steps,
                "dif_norm": dn,
                "inh_norm": inn,
                "alpha_skip": alpha.item(),
                "tau": self._current_tau(),
                "gate_probs": [round(p.item(), 4) for p in probs],
            })

        if self._debug and self._steps % 100 == 0:
            tier = "HBM" if self._ema_ms > 5 else ("GDDR" if self._ema_ms > 1 else "DRAM")
            tau = self._current_tau()
            is_hard_skip = alpha.item() == 0.0
            is_capped = alpha.item() >= 0.45
            gate_info = (
                f"α_skip={alpha.item():.4f} "
                f"τ={tau:.4f} "
                f"logits=[{self._gate_logits[0].item():.3f},{self._gate_logits[1].item():.3f}] "
                f"{'[ZERO]' if is_hard_skip else ''}"
                f"{'[CAPPED]' if is_capped else ''}"
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
    """Decoupled Spatial-Temporal GNN — Walpurgis v4 variant.

    Key algorithmic choices:
      1. **MoE gating aggregation** – a gating network maps each
         layer's combined forecast to a probability distribution.
         An auxiliary load-balancing loss (Cauchy entropy) prevents
         collapse to a single expert/layer.
      2. **Inverse-sqrt embedding warmup** – warmup_α =
         min(step^{-0.5}, step · W^{-1.5}).  Fast initial ramp,
         then graceful decay.
      3. **Spectral fingerprint** — top-3 singular values of a
         graph sample, hashed for topology change detection.

    Debug cheat-sheet (from pdb or after any forward):
        TensorProbe.dump_all()
        TensorProbe.anomaly_summary()
        TensorProbe.grad_flow_report()
        TensorProbe.auto_bisect()
        model.snapshot()
        model._moe_weights_now()
        model._warmup_alpha()
        model._structure_fingerprint
        model._load_balance_loss
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

        # ── MoE gating aggregation ──
        # Gating network: maps forecast dim → per-expert (per-layer) probability
        self._moe_gate = nn.Sequential(
            nn.Linear(self._forecast_dim, self._forecast_dim // 2),
            nn.GELU(),
            nn.Linear(self._forecast_dim // 2, self._num_layers),
        )
        self._moe_balance_coeff = 0.01  # aux loss weight
        self._load_balance_loss = 0.0   # stored for external access

        self._init_params()

        # Debug bookkeeping
        self._debug = True
        self._fwd_n = 0
        self._warmup_steps = 200
        self._contribution_ema = {}
        self._structure_fingerprint = "n/a"
        self._last_moe_weights = []
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

    # ── MoE weight query (debug helper) ──
    def _moe_weights_now(self) -> list:
        """Return last computed MoE gating weights."""
        return self._last_moe_weights

    def _warmup_alpha(self) -> float:
        """Inverse-sqrt warmup schedule (Vaswani et al.)."""
        step = max(self._fwd_n, 1)
        W = max(self._warmup_steps, 1)
        return min(step ** (-0.5), step * (W ** (-1.5)))

    def _spectral_fingerprint(self, tensor_sample: torch.Tensor) -> str:
        """Spectral hash: top-3 SVD singular values → hash string.

        Captures topology invariants that elementwise hashes miss.
        """
        try:
            with torch.no_grad():
                # Flatten to 2D for SVD
                t2d = tensor_sample.reshape(-1, tensor_sample.shape[-1]).float()
                # Fast truncated SVD via power iteration
                k = min(3, *t2d.shape)
                U, S, V = torch.svd_lowrank(t2d, q=k)
                # Quantise singular values and hash
                s_bytes = S.cpu().to(torch.float16).numpy().tobytes()
                h = hash(s_bytes) & 0xFFFFFFFFFFFF
                return f"svd_{h:012x}"
        except Exception:
            return "svd_fallback"

    def forward(self, history_data):
        """
        Input:  [B, L, N, C]
        Output: [B, N, L']

        Breakpoint cheat-sheet — call any of these from pdb:
            TensorProbe.dump_all()
            TensorProbe.anomaly_summary()
            TensorProbe.grad_flow_report()
            TensorProbe.auto_bisect()
            self.snapshot()
            self._moe_weights_now()
            self._warmup_alpha()
            print(self._structure_fingerprint)
            print(self._load_balance_loss)
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
        # Spectral fingerprint
        if dyn_g:
            with torch.no_grad():
                sample = dyn_g[0].detach()
                self._structure_fingerprint = self._spectral_fingerprint(sample)

        # Step 3 — embedding with inverse-sqrt warmup
        warmup_alpha = self._warmup_alpha()
        warmup_alpha_clamped = min(warmup_alpha * self._warmup_steps, 1.0)
        if warmup_alpha_clamped < 0.999:
            if self._in_feat <= self._hidden_dim:
                identity = F.pad(raw, (0, self._hidden_dim - self._in_feat))
            else:
                identity = raw[:, :, :, :self._hidden_dim]
            identity = identity * 0.1
            learned = self.embedding(raw)
            embedded = (1.0 - warmup_alpha_clamped) * identity + warmup_alpha_clamped * learned
        else:
            embedded = self.embedding(raw)

        if verbose:
            self._probe_emb(
                embedded,
                f"warmup={warmup_alpha_clamped:.3f}  fp={self._structure_fingerprint}"
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
                tau = layer._current_tau()
                probs = F.softmax(layer._gate_logits, dim=0)
                print(
                    f"  │ L{i}: {ms:.1f}ms  dif={dn:.2f}  inh={inn:.2f}  "
                    f"gate_probs=[{probs[0].item():.4f},{probs[1].item():.4f}] "
                    f"τ={tau:.4f}"
                )
            dif_fcs.append(d_fc)
            inh_fcs.append(i_fc)

        # Step 5 — MoE gating aggregation
        # Combine each layer's dif+inh forecast
        combined_fcs = [d + i for d, i in zip(dif_fcs, inh_fcs)]  # [L × [B,N,F]]

        # Global representation for gating: average across all layers
        stacked = torch.stack(combined_fcs, dim=0)  # [L, B, N, F]
        L, B, N, Fd = stacked.shape
        # Per-sample gating: average over N nodes → [B, F]
        gate_input = stacked.mean(dim=(0, 2))  # [B, F]
        gate_logits = self._moe_gate(gate_input)  # [B, L]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, L]

        self._last_moe_weights = [round(w, 5) for w in gate_weights.mean(dim=0).tolist()]

        # Weighted sum: [B, N, F]
        # Reshape weights to broadcast: [B, 1, 1, L] × [L, B, N, F]
        agg = torch.zeros(B, N, Fd, device=stacked.device, dtype=stacked.dtype)
        for li in range(L):
            w = gate_weights[:, li].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            agg = agg + w * combined_fcs[li]

        # Auxiliary load-balancing loss (Cauchy entropy)
        # Encourage uniform distribution across experts
        avg_weights = gate_weights.mean(dim=0)  # [L]
        # Cauchy entropy: -Σ log(1 + (w - 1/L)² / γ²)
        target = 1.0 / L
        gamma = 0.1
        cauchy_terms = torch.log(1.0 + ((avg_weights - target) / gamma).pow(2))
        self._load_balance_loss = self._moe_balance_coeff * cauchy_terms.sum()

        # Step 6 — output projection
        pred = self.out_fc_2(F.relu(self.out_fc_1(F.relu(agg))))
        pred = pred.transpose(1, 2).contiguous().view(pred.shape[0], pred.shape[2], -1)

        if verbose:
            self._probe_out(pred)
            wl = [f"{w:.3f}" for w in self._last_moe_weights]
            print(f"  │ moe_weights: {wl}  balance_loss={self._load_balance_loss:.5f}")

            # Gini coefficient of MoE weights as diversity measure
            ws = sorted(self._last_moe_weights)
            n = len(ws)
            gini = sum((2 * (i + 1) - n - 1) * ws[i] for i in range(n)) / (n * sum(ws) + 1e-9)
            print(
                f"  │ gini={gini:.3f} "
                f"(0=uniform, 1=monopoly)"
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
            "moe_weights": self._moe_weights_now(),
            "load_balance_loss": float(self._load_balance_loss),
            "warmup_alpha": self._warmup_alpha(),
            "structure_fingerprint": self._structure_fingerprint,
            "contribution_ema": {
                str(k): v for k, v in self._contribution_ema.items()
            },
            "layer_gate_stats": [],
            "parameters": {},
        }
        # Layer-level gate stats
        for i, layer in enumerate(self.layers):
            probs = F.softmax(layer._gate_logits, dim=0)
            recent = list(layer._contrib_ring)[-5:]
            state["layer_gate_stats"].append({
                "layer": i,
                "gate_probs": [round(p.item(), 5) for p in probs],
                "tau": round(layer._current_tau(), 4),
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
