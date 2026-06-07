import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

def _edbg(tag, val):
    if os.environ.get('EQUINOX_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        nnz = (val.abs() > 1e-6).float().mean().item()
        print(f"[EQX:mask:{tag}] sparsity={1-nnz:.2%}", file=sys.stderr)

class Mask(nn.Module):
    """upstream: 硬mask乘adj
    equinox: Gumbel-Softmax离散图采样, 可学习logits控制边保留概率"""
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        # equinox: Gumbel-Softmax logits (2-class: keep/drop)
        N = self.mask[0].shape[0]
        self.edge_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(N, N, 2)) for _ in self.mask
        ])
        # equinox: 温度退火参数
        self.tau = 1.0

    def set_tau(self, tau):
        """外部可调温度: 训练后期降低tau趋向硬采样"""
        self.tau = max(0.1, tau)

    def _mask(self, index, adj):
        mask_idx = index % len(self.mask)
        logit_idx = index % len(self.edge_logits)
        base_mask = self.mask[mask_idx].to(adj.device)
        # equinox: Gumbel-Softmax采样 — 可微分离散边选择
        logits = self.edge_logits[logit_idx].to(adj.device)
        if self.training:
            soft_mask = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)[..., 0]
        else:
            # 推理时用argmax硬选择
            soft_mask = (logits[..., 0] > logits[..., 1]).float()
        combined = (base_mask + 1e-7) * soft_mask
        result = combined.unsqueeze(0) * adj if adj.dim() == 3 else combined * adj
        _edbg(f"gumbel_mask_{index}", result)
        return result

    def forward(self, adj):
        return [self._mask(i, a) for i, a in enumerate(adj)]
