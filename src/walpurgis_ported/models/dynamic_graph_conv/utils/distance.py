"""
Walpurgis v4 Distance Function — Asymmetric Kernel Similarity with Gated Fusion
==========================================================================
Delta vs v3:
  - Bilinear similarity → *asymmetric kernel* similarity with per-modality
    learnable weight matrix W:  sim(a,b) = a^T W b / τ.
    Bilinear allows asymmetric similarity (node A similar to B doesn't
    imply B similar to A), which better models directed traffic flow.
  - Modality fusion: softmax logits → *sigmoid gate* per modality,
    allowing partial suppression rather than forced competition.
  - Temperature uses a per-modality learnable τ instead of shared.

Breakpoint helpers:
    self._diag_last                    # last forward diagnostics
    self.modality_balance_report()     # print fusion weight history
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque


class DistanceFunction(nn.Module):
    """Bilinear pairwise similarity with gated modality fusion."""

    _n = 0

    def __init__(self, **kw):
        super().__init__()
        H = kw["num_hidden"]
        ND = kw["node_hidden"]
        TD = kw["time_emb_dim"]
        self.k_t = kw["k_t"]

        self.node_proj_l = nn.Linear(ND, H)
        self.node_proj_r = nn.Linear(ND, H)  # separate for asymmetry
        self.time_proj = nn.Linear(TD * 2, H)
        self.data_proj = nn.Linear(H, H)

        # Per-modality sigmoid gates
        self._mod_gates = nn.Parameter(torch.zeros(3))
        # Per-modality temperatures
        self._taus = nn.Parameter(torch.full((3,), math.sqrt(float(H))))

        self._debug = True
        self._diag_last = {}
        self._balance_log = deque(maxlen=300)

    def modality_balance_report(self, n=15):
        """Print recent modality gate activations — call from pdb."""
        entries = list(self._balance_log)[-n:]
        if not entries:
            print("  [Distance] no balance data yet")
            return
        print(f"  [Distance] modality gates (last {len(entries)}):")
        for step, gn, gt, gd in entries:
            bars = [int(g * 20) for g in [gn, gt, gd]]
            print(
                f"    #{step}: node={'█'*bars[0]+'░'*(20-bars[0])} {gn:.3f} | "
                f"time={'█'*bars[1]+'░'*(20-bars[1])} {gt:.3f} | "
                f"data={'█'*bars[2]+'░'*(20-bars[2])} {gd:.3f}"
            )

    def forward(self, X, E_d, E_u, T_D, D_W):
        DistanceFunction._n += 1
        t0 = time.perf_counter()
        B, L, N, _ = X.shape
        taus = self._taus.clamp(min=1.0)

        # Node modality — bilinear (asymmetric)
        nl = self.node_proj_l(E_d)  # [N, H]
        nr = self.node_proj_r(E_d)  # [N, H]
        node_sim = torch.mm(nl, nr.T) / taus[0]
        node_sim = torch.softmax(node_sim, dim=-1)
        node_sim = node_sim.unsqueeze(0).expand(B, -1, -1)

        # Time modality
        tc = torch.cat([T_D, D_W], dim=-1)
        tf = self.time_proj(tc).mean(dim=1)
        time_sim = torch.bmm(tf, tf.transpose(1, 2)) / taus[1]
        time_sim = torch.softmax(time_sim, dim=-1)

        # Data modality
        df = self.data_proj(X[:, -self.k_t:].mean(dim=1))
        data_sim = torch.bmm(df, df.transpose(1, 2)) / taus[2]
        data_sim = torch.softmax(data_sim, dim=-1)

        # Sigmoid gated fusion (not softmax competition)
        gates = torch.sigmoid(self._mod_gates)  # [3], each ∈ (0, 1)
        dist = gates[0] * node_sim + gates[1] * time_sim + gates[2] * data_sim
        # Renormalize to keep distribution sum ≈ 1 per row
        dist = dist / (gates.sum() + 1e-8)

        if self._debug:
            with torch.no_grad():
                gvals = [gates[i].item() for i in range(3)]
                self._balance_log.append((DistanceFunction._n, *gvals))
                self._diag_last = {
                    "step": DistanceFunction._n,
                    "gates": [round(g, 4) for g in gvals],
                    "taus": [round(taus[i].item(), 2) for i in range(3)],
                    "node_sim_mean": round(node_sim.mean().item(), 5),
                    "time_sim_mean": round(time_sim.mean().item(), 5),
                    "data_sim_mean": round(data_sim.mean().item(), 5),
                }
            if DistanceFunction._n % 200 == 1:
                ms = (time.perf_counter() - t0) * 1000
                d = self._diag_last
                print(
                    f"        [BilinDist #{DistanceFunction._n}] {ms:.2f}ms | "
                    f"gates={d['gates']} τ={d['taus']} | "
                    f"node_μ={d['node_sim_mean']:.4f} "
                    f"time_μ={d['time_sim_mean']:.4f} "
                    f"data_μ={d['data_sim_mean']:.4f}"
                )
        return dist
