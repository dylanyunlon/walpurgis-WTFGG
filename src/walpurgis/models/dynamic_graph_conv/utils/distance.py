"""
Walpurgis Distance Function — Pairwise Node Similarity
========================================================
Derived from D2STGNN distance.py.

Change: uses cosine similarity instead of L2 distance for the embedding
pair comparison. Cosine similarity is scale-invariant, so it handles
the case where node embeddings have different magnitudes better than
raw dot product or L2 distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DistanceFunction(nn.Module):
    """Compute pairwise node distance/similarity from multi-modal features.
    
    Combines:
    - Node structural embeddings (E_u, E_d)
    - Time-of-day features (T_D)
    - Day-of-week features (D_W)
    - Historical data patterns (X)
    
    Walpurgis: uses cosine similarity between projected feature pairs
    instead of raw dot product. This makes the distance function invariant
    to embedding scale, which improves stability across training.
    """
    
    _call_count = 0
    
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_dim = model_args['time_emb_dim']
        self.k_t = model_args['k_t']
        
        # Projection layers for each modality
        self.node_proj = nn.Linear(self.node_dim, self.hidden_dim)
        self.time_proj = nn.Linear(self.time_dim * 2, self.hidden_dim)  # day + week
        self.data_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Final distance projection
        self.dist_fc = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        
        self._debug_on = True
    
    def forward(self, X, E_d, E_u, T_D, D_W):
        """
        Args:
            X:   [B, L, N, D] history data
            E_d: [N, D_node]  node embedding (downstream)
            E_u: [N, D_node]  node embedding (upstream)
            T_D: [B, L, N, D_time] time-of-day embedding
            D_W: [B, L, N, D_time] day-of-week embedding
        Returns:
            dist_mx: [B, N, N] pairwise distance/similarity matrix
        """
        DistanceFunction._call_count += 1
        t0 = time.perf_counter()
        
        B, L, N, _ = X.shape
        
        # Node modality: cosine similarity between projected embeddings
        node_feat = self.node_proj(E_d)  # [N, H]
        node_feat_norm = F.normalize(node_feat, dim=-1)
        node_sim = torch.mm(node_feat_norm, node_feat_norm.T)  # [N, N] cosine sim
        node_sim = node_sim.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        
        # Temporal modality: aggregate time features across sequence
        time_cat = torch.cat([T_D, D_W], dim=-1)  # [B, L, N, 2*D_time]
        time_feat = self.time_proj(time_cat).mean(dim=1)  # [B, N, H]
        time_feat_norm = F.normalize(time_feat, dim=-1)
        time_sim = torch.bmm(time_feat_norm, time_feat_norm.transpose(1, 2))  # [B, N, N]
        
        # Data modality: recent pattern similarity
        data_feat = self.data_proj(X[:, -self.k_t:, :, :].mean(dim=1))  # [B, N, H]
        data_feat_norm = F.normalize(data_feat, dim=-1)
        data_sim = torch.bmm(data_feat_norm, data_feat_norm.transpose(1, 2))  # [B, N, N]
        
        # Combine modalities: stack and project
        combined = torch.stack([node_sim, time_sim, data_sim], dim=-1)  # [B, N, N, 3]
        # Simple weighted average (learnable via upstream training)
        dist_mx = combined.mean(dim=-1)  # [B, N, N]
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        if self._debug_on and DistanceFunction._call_count % 200 == 1:
            print(f"        [Distance #{DistanceFunction._call_count}] {elapsed:.2f}ms | "
                  f"node_sim μ={node_sim.mean().item():.4f} "
                  f"time_sim μ={time_sim.mean().item():.4f} "
                  f"data_sim μ={data_sim.mean().item():.4f}")
        
        return dist_mx
