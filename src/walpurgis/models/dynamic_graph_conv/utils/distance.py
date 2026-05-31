"""
Walpurgis v2 Distance Function — Multi-Modal Scaled-Dot-Product Similarity
============================================================================
Delta: cosine → *temperature-scaled dot product* with learnable per-modality
fusion weights instead of uniform average.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class DistanceFunction(nn.Module):
    """Compute pairwise node similarity via temperature-scaled dot product.

    Three modalities (node embedding, temporal, historical data) are
    each projected and compared.  A learnable 3-element weight vector
    controls their relative contribution via softmax fusion.
    """

    _n = 0

    def __init__(self, **kw):
        super().__init__()
        H = kw["num_hidden"]
        ND = kw["node_hidden"]
        TD = kw["time_emb_dim"]
        self.k_t = kw["k_t"]

        self.node_proj = nn.Linear(ND, H)
        self.time_proj = nn.Linear(TD * 2, H)
        self.data_proj = nn.Linear(H, H)

        # Learnable modality fusion weights
        self._mod_logits = nn.Parameter(torch.zeros(3))
        # Temperature for dot-product scaling
        self._tau = nn.Parameter(torch.tensor(math.sqrt(float(H))))

        self._debug = True

    def forward(self, X, E_d, E_u, T_D, D_W):
        DistanceFunction._n += 1
        t0 = time.perf_counter()
        B, L, N, _ = X.shape
        tau = self._tau.clamp(min=1.0)

        # Node modality — scaled dot product
        nf = self.node_proj(E_d)  # [N, H]
        node_sim = torch.mm(nf, nf.T) / tau  # [N, N]
        node_sim = torch.softmax(node_sim, dim=-1)
        node_sim = node_sim.unsqueeze(0).expand(B, -1, -1)

        # Time modality
        tc = torch.cat([T_D, D_W], dim=-1)
        tf = self.time_proj(tc).mean(dim=1)  # [B, N, H]
        time_sim = torch.bmm(tf, tf.transpose(1, 2)) / tau
        time_sim = torch.softmax(time_sim, dim=-1)

        # Data modality
        df = self.data_proj(X[:, -self.k_t:].mean(dim=1))
        data_sim = torch.bmm(df, df.transpose(1, 2)) / tau
        data_sim = torch.softmax(data_sim, dim=-1)

        # Learnable fusion
        w = F.softmax(self._mod_logits, dim=0)
        dist = w[0] * node_sim + w[1] * time_sim + w[2] * data_sim

        if self._debug and DistanceFunction._n % 200 == 1:
            ms = (time.perf_counter() - t0) * 1000
            wl = [f"{x.item():.3f}" for x in w]
            print(
                f"        [Distance #{DistanceFunction._n}] {ms:.2f}ms | "
                f"weights={wl} τ={tau.item():.2f} | "
                f"node_μ={node_sim.mean().item():.4f} "
                f"time_μ={time_sim.mean().item():.4f} "
                f"data_μ={data_sim.mean().item():.4f}"
            )

        return dist
