"""Nebula mask: top-k sparse mask with straight-through estimator."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class TopKStraightThrough(torch.autograd.Function):
    """Straight-through estimator for top-k: forward uses hard mask,
    backward passes gradients through as if mask were identity."""
    @staticmethod
    def forward(ctx, scores, k):
        # scores: [B, N, N]
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        ctx.save_for_backward(mask)
        return scores * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # Straight-through: pass gradient as-is (ignoring mask in backward)
        return grad_output, None


class Mask(nn.Module):
    """Nebula top-k sparse mask: retains only top-k connections per node.
    Uses straight-through estimator for differentiable training.
    Replaces upstream's predefined adjacency mask."""
    def __init__(self, **model_args):
        super().__init__()
        self.adj_templates = model_args['adjs']
        num_nodes = self.adj_templates[0].shape[0]
        # k = max(3, 30% of nodes) -- adaptive sparsity
        self.k = max(3, int(num_nodes * 0.3))
        # Learnable logit bias per adjacency
        self.logit_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in self.adj_templates
        ])

    def _sparse_mask(self, index, adj):
        """Apply top-k sparse mask with straight-through gradient."""
        template = self.adj_templates[index]
        # Combine adjacency prior with learnable bias
        prior = (template + 1e-7).to(adj.device)
        biased_adj = adj + self.logit_bias[index] * prior.unsqueeze(0)
        # Top-k sparse selection with straight-through
        if biased_adj.dim() == 2:
            biased_adj = biased_adj.unsqueeze(0)
        k = min(self.k, biased_adj.shape[-1])
        masked = TopKStraightThrough.apply(biased_adj, k)
        if _NEB_DBG:
            sparsity = (masked == 0).float().mean().item()
            print(f"[NEB:topk_mask@mask] k={k} sparsity={sparsity:.2%} bias={self.logit_bias[index].item():.4f}", file=sys.stderr)
        return masked.squeeze(0) if adj.dim() == 2 else masked

    def forward(self, adj):
        result = []
        for index, a in enumerate(adj):
            result.append(self._sparse_mask(index % len(self.adj_templates), a))
        return result
