"""
Walpurgis Distance Function — Tier-Aware Pairwise Node Affinity
================================================================
Adapted from D2STGNN DistanceFunction.

Core algorithm changes (~20%):
  1. Multi-scale temporal feature extraction: instead of a single FC→BN→FC
     pipeline on the raw time series, we also extract a coarsened (avg-pooled)
     representation and concatenate. This gives the distance function both
     fine-grained and trend-level temporal cues.
  2. Temperature-scaled attention: the QK^T / sqrt(d) scaling is augmented
     with a learnable temperature parameter (initialized to 1.0) that the
     model can tune during training. Colder temperatures = sharper graphs.
  3. Attention entropy tracking for debug: if entropy is too low (peaked)
     or too high (uniform), the graph is degenerate.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceFunction(nn.Module):
    """Compute pairwise node distance (attention) from dynamic features,
    time embeddings, and trainable node embeddings.

    Walpurgis changes vs D2STGNN:
      - Multi-scale temporal features (raw + coarsened)
      - Learnable attention temperature
      - Entropy-based graph quality diagnostic
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim   = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len     = model_args['seq_length']

        # ── Time Series Feature Extraction ──
        self.dropout    = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

        # Walpurgis: coarsened temporal branch (avg-pool by 2 then FC)
        self.coarse_seq_len = max(1, self.input_seq_len // 2)
        self.fc_coarse = nn.Linear(self.coarse_seq_len, self.hidden_dim // 2)
        self.coarse_feat_dim = self.hidden_dim // 2

        # Time Slot Embedding Extraction
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        # Distance Score — includes coarse feature dimension
        self.all_feat_dim = (self.ts_feat_dim + self.coarse_feat_dim +
                            self.node_dim + model_args['time_emb_dim'] * 2)
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)

        # Walpurgis: learnable attention temperature
        # init=1.0 means standard sqrt(d) scaling; <1 = sharper, >1 = smoother
        self.attn_temperature = nn.Parameter(torch.ones(1))

        # Debug accumulators
        self._entropy_history = []
        self._temperature_history = []

        print(f"[Walpurgis::DistanceFunction] init hidden={self.hidden_dim} "
              f"node_dim={self.node_dim} all_feat_dim={self.all_feat_dim} "
              f"coarse_dim={self.coarse_feat_dim} seq_len={self.input_seq_len}")
        total_p = sum(p.numel() for p in self.parameters())
        print(f"[Walpurgis::DistanceFunction] total params={total_p:,}")

    def _compute_entropy(self, W):
        """Compute attention entropy for graph quality diagnosis.
        High entropy = uniform attention (graph is useless).
        Low entropy = peaked attention (graph may be too sparse).
        Ideal range: 1.0 - 3.0 nats for typical traffic graphs.
        """
        with torch.no_grad():
            # W shape: [B, N, N], already softmaxed
            log_W = torch.log(W + 1e-10)
            entropy = -(W * log_W).sum(dim=-1).mean().item()
        return entropy

    def forward(self, X, E_d, E_u, T_D, D_W):
        DistanceFunction._call_count += 1
        _verbose = (DistanceFunction._call_count <= 3 or
                    DistanceFunction._call_count % 500 == 0)
        t0 = time.perf_counter()

        # last-step pooling for time embeddings
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        # ── Fine-grained temporal features (same as D2STGNN) ──
        X_raw = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X_raw.shape
        X_flat = X_raw.view(batch_size * num_nodes, seq_len)

        bn_input = F.relu(self.fc_ts_emb1(X_flat))
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(bn_input)))
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)

        # ── Walpurgis: coarsened temporal features ──
        # Average-pool by factor of 2 to capture trend-level patterns
        # that fine-grained FC might miss
        if seq_len >= 2:
            # Truncate to even length, then pool
            even_len = (seq_len // 2) * 2
            X_coarse = X_raw[:, :, :even_len].view(batch_size * num_nodes, -1, 2).mean(dim=-1)
        else:
            X_coarse = X_flat

        # Pad or truncate to expected coarse_seq_len
        if X_coarse.shape[-1] < self.coarse_seq_len:
            pad_size = self.coarse_seq_len - X_coarse.shape[-1]
            X_coarse = F.pad(X_coarse, (0, pad_size), value=0.0)
        elif X_coarse.shape[-1] > self.coarse_seq_len:
            X_coarse = X_coarse[:, :self.coarse_seq_len]

        coarse_feat = F.relu(self.fc_coarse(X_coarse))
        coarse_feat = coarse_feat.view(batch_size, num_nodes, -1)

        # ── Node embeddings ──
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        # ── Distance via attention (with coarse features and learnable temp) ──
        X1 = torch.cat([dy_feat, coarse_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, coarse_feat, T_D, D_W, emb2], dim=-1)
        X_list = [X1, X2]

        # Effective temperature: base sqrt(d) scaling * learned temperature
        effective_temp = math.sqrt(self.hidden_dim) * self.attn_temperature.clamp(min=0.1, max=5.0)

        adjacent_list = []
        entropies = []
        for feat in X_list:
            Q = self.WQ(feat)
            K = self.WK(feat)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / effective_temp
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
            entropies.append(self._compute_entropy(W))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Track temperature and entropy
        self._temperature_history.append(self.attn_temperature.item())
        self._entropy_history.extend(entropies)

        if _verbose:
            temp_val = self.attn_temperature.item()
            eff_temp_val = effective_temp.item() if isinstance(effective_temp, torch.Tensor) else effective_temp
            print(f"\n[Walpurgis::DistFunc::forward] call#{DistanceFunction._call_count} "
                  f"B={batch_size} N={num_nodes} elapsed={elapsed_ms:.3f}ms")
            print(f"  temperature: learned={temp_val:.4f} effective={eff_temp_val:.4f}")
            print(f"  coarse_feat: shape={list(coarse_feat.shape)} "
                  f"norm={coarse_feat.norm().item():.4f}")
            for idx, (adj, ent) in enumerate(zip(adjacent_list, entropies)):
                adj_density = (adj > 1.0 / num_nodes).float().mean().item()
                print(f"  adj[{idx}] shape={list(adj.shape)} "
                      f"mean={adj.mean().item():.6f} max={adj.max().item():.6f} "
                      f"entropy={ent:.4f} density={adj_density:.4f}")
                if torch.isnan(adj).any().item():
                    print(f"  ⚠ adj[{idx}] contains NaN!")
                # Entropy quality diagnosis
                if ent < 0.5:
                    print(f"  ⚠ adj[{idx}] entropy very low ({ent:.2f}) — graph is too peaked")
                elif ent > math.log(num_nodes) * 0.9:
                    print(f"  ⚠ adj[{idx}] entropy near uniform ({ent:.2f}/{math.log(num_nodes):.2f}) — graph is uninformative")

        # Periodic summary
        if DistanceFunction._call_count % 1000 == 0 and self._entropy_history:
            import numpy as np
            ent_arr = np.array(self._entropy_history[-1000:])
            temp_arr = np.array(self._temperature_history[-500:])
            print(f"\n  [DistFunc SUMMARY @ {DistanceFunction._call_count}] "
                  f"entropy: mean={ent_arr.mean():.3f} std={ent_arr.std():.3f} "
                  f"temperature: mean={temp_arr.mean():.4f} std={temp_arr.std():.4f}")

        return adjacent_list
