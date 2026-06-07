"""Tempest mask: straight-through Bernoulli (hard in forward, soft in backward).
Unlike upstream (fixed binary mask from adj) and eclipse (sigmoid soft-gating),
Tempest uses a straight-through estimator with Bernoulli sampling: the forward pass
gets discrete 0/1 masks, but gradients flow through the soft sigmoid probabilities.
This encourages sparse graph structure while remaining differentiable."""
import torch, torch.nn as nn, sys, os
_TEM_DBG = os.environ.get('TEMPEST_DEBUG', '0') == '1'

class StraightThroughBernoulli(torch.autograd.Function):
    """Straight-through estimator: hard Bernoulli in forward, soft sigmoid in backward."""
    @staticmethod
    def forward(ctx, logits):
        probs = torch.sigmoid(logits)
        # Hard sample: Bernoulli(probs)
        hard = torch.bernoulli(probs)
        ctx.save_for_backward(probs)
        return hard

    @staticmethod
    def backward(ctx, grad_output):
        probs, = ctx.saved_tensors
        # Straight-through: gradient flows through sigmoid probabilities
        return grad_output * probs * (1 - probs)

class Mask(nn.Module):
    """Tempest mask: straight-through Bernoulli.
    Each edge has a learnable logit; during forward, we sample discrete masks
    via Bernoulli, but backprop through the soft probabilities."""
    def __init__(self, **model_args):
        super().__init__()
        self.adj_templates = model_args['adjs']
        # Initialize logits so initial prob ~ 0.8 (slightly above threshold)
        self.logits = nn.ParameterList([
            nn.Parameter(torch.ones_like(a) * 1.4)  # sigmoid(1.4) ~ 0.8
            for a in self.adj_templates
        ])

    def _st_mask(self, idx, adj):
        if self.training:
            # Straight-through Bernoulli: hard in forward, soft in backward
            hard_mask = StraightThroughBernoulli.apply(self.logits[idx])
        else:
            # Deterministic at eval: threshold at 0.5
            hard_mask = (torch.sigmoid(self.logits[idx]) > 0.5).float()
        masked = hard_mask.to(adj.device) * adj
        return masked

    def forward(self, adj):
        result = [self._st_mask(i, a) for i, a in enumerate(adj)]
        if _TEM_DBG:
            probs = torch.sigmoid(self.logits[0])
            print(f"[TEM:mask@mask] sparsity={(result[0]==0).float().mean().item():.2%} "
                  f"prob_mean={probs.mean().item():.4f} prob_std={probs.std().item():.4f}", file=sys.stderr)
        return result
