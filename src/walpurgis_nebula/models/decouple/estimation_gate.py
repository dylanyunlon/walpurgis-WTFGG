"""Nebula gate: Capsule routing gate with dynamic routing-by-agreement."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

class CapsuleRoutingGate(nn.Module):
    """Capsule-style routing gate: projects features into capsule space,
    performs iterative routing-by-agreement to compute gating coefficients.
    Replaces: upstream's 2-layer FC + sigmoid gate."""
    def __init__(self, input_dim, capsule_dim=8, num_capsules=4, routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        # Project input into capsule space
        self.capsule_proj = nn.Linear(input_dim, num_capsules * capsule_dim)
        # Agreement-to-gate: map routed capsules to scalar gate
        self.gate_fc = nn.Linear(num_capsules * capsule_dim, 1)

    def _squash(self, s):
        """Squash activation for capsules: v = ||s||^2 / (1 + ||s||^2) * s / ||s||."""
        sq_norm = (s ** 2).sum(dim=-1, keepdim=True)
        scale = sq_norm / (1.0 + sq_norm) / (sq_norm.sqrt() + 1e-8)
        return scale * s

    def forward(self, features):
        """features: [..., input_dim] -> gate: [..., 1] in (0, 1)."""
        batch_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        B = flat.shape[0]
        # Project to capsules: [B, num_capsules, capsule_dim]
        caps = self.capsule_proj(flat).view(B, self.num_capsules, self.capsule_dim)
        # Routing by agreement
        logits = torch.zeros(B, self.num_capsules, device=flat.device)
        for _ in range(self.routing_iters):
            c = F.softmax(logits, dim=-1).unsqueeze(-1)  # [B, num_capsules, 1]
            s = (c * caps).sum(dim=1)  # [B, capsule_dim]
            v = self._squash(s)  # [B, capsule_dim]
            if _ < self.routing_iters - 1:
                agreement = (caps * v.unsqueeze(1)).sum(dim=-1)  # [B, num_capsules]
                logits = logits + agreement
        # Routed output: concatenate all capsule contributions
        routed = caps * F.softmax(logits, dim=-1).unsqueeze(-1)  # [B, C, D]
        routed_flat = routed.reshape(B, -1)  # [B, C*D]
        gate = torch.sigmoid(self.gate_fc(routed_flat))  # [B, 1]
        gate = gate.view(*batch_shape, 1)
        if _NEB_DBG:
            print(f"[NEB:capsule_gate@estimation_gate] gate_mean={gate.mean().item():.4f} gate_std={gate.std().item():.4f}", file=sys.stderr)
        return gate


class EstimationGate(nn.Module):
    """Nebula estimation gate using capsule routing mechanism."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        total_dim = 2 * node_emb_dim + time_emb_dim * 2
        self.capsule_gate = CapsuleRoutingGate(
            input_dim=total_dim, capsule_dim=max(hidden_dim // 4, 4),
            num_capsules=4, routing_iters=3)

    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        """Generate capsule-routed gate values to estimate hidden time series proportions."""
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        gate_feat = torch.cat([
            time_in_day_feat, day_in_week_feat,
            node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1),
            node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1, -1)
        ], dim=-1)
        estimation_gate = self.capsule_gate(gate_feat)[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        if _NEB_DBG:
            print(f"[NEB:output@estimation_gate] gated_norm={history_data.norm().item():.4f}", file=sys.stderr)
        return history_data
