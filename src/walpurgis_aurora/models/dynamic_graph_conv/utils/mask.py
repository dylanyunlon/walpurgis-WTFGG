import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('AURORA_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        nnz = (val.abs() > 1e-6).float().mean().item()
        print(f"[AUR:mask:{tag}] sparsity={1-nnz:.2%}", file=sys.stderr)

class Mask(nn.Module):
    """upstream: 硬mask乘adj
    aurora: sigmoid soft-gating, 可学习logits控制哪些边保留"""
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # aurora: 可学习soft gate logits
        N = self.mask[0].shape[0]
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(N, N)) for _ in self.mask
        ])

    def _mask(self, index, adj):
        # Handle when adj list is longer than pre-defined masks
        mask_idx = index % len(self.mask)
        gate_idx = index % len(self.gate_logits)
        base_mask = self.mask[mask_idx].to(adj.device)
        soft_gate = torch.sigmoid(self.gate_logits[gate_idx].to(adj.device))
        combined = (base_mask + 1e-7) * soft_gate
        result = combined.unsqueeze(0) * adj if adj.dim() == 3 else combined * adj
        _adbg(f"soft_mask_{index}", result)
        return result

    def forward(self, adj):
        return [self._mask(i, a) for i, a in enumerate(adj)]
