import torch
import torch.nn as nn
import sys, os

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        nnz = (val.abs() > 1e-6).float().mean().item()
        print(f"[SOL:mask:{tag}] sparsity={1-nnz:.2%}", file=sys.stderr)

class Mask(nn.Module):
    """upstream: 硬mask乘adj
    solstice: Gumbel-sigmoid soft-gating — 训练时加Gumbel噪声探索边结构"""
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # solstice: 可学习soft gate logits
        N = self.mask[0].shape[0]
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(N, N)) for _ in self.mask
        ])
        # solstice: Gumbel噪声温度
        self._gumbel_tau = 0.5

    def _gumbel_sigmoid(self, logits):
        """solstice: Gumbel-Sigmoid — 训练时加噪声, 推理时纯sigmoid"""
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            return torch.sigmoid((logits + gumbel) / self._gumbel_tau)
        return torch.sigmoid(logits)

    def _mask(self, index, adj):
        mask_idx = index % len(self.mask)
        gate_idx = index % len(self.gate_logits)
        base_mask = self.mask[mask_idx].to(adj.device)
        soft_gate = self._gumbel_sigmoid(self.gate_logits[gate_idx].to(adj.device))
        combined = (base_mask + 1e-7) * soft_gate
        result = combined.unsqueeze(0) * adj if adj.dim() == 3 else combined * adj
        _adbg(f"gumbel_mask_{index}", result)
        return result

    def forward(self, adj):
        return [self._mask(i, a) for i, a in enumerate(adj)]
