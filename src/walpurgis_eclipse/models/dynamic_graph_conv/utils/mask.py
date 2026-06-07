"""Eclipse mask: sigmoid soft-gating with nn.Parameter logits."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.adj_templates = model_args['adjs']
        self.logits = nn.ParameterList([nn.Parameter(torch.zeros_like(a)) for a in self.adj_templates])

    def _soft_mask(self, idx, adj):
        gate = torch.sigmoid(self.logits[idx]).to(adj.device)
        return gate * adj

    def forward(self, adj):
        result = [self._soft_mask(i, a) for i, a in enumerate(adj)]
        if _ECL_DBG: print(f"[ECL:mask] sparsity={(result[0]==0).float().mean().item():.2%} gate_mean={torch.sigmoid(self.logits[0]).mean().item():.4f}", file=sys.stderr)
        return result
