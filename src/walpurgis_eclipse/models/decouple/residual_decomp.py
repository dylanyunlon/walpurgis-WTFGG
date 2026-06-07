"""Eclipse residual: RMSNorm + Mish + learnable alpha."""
import torch, torch.nn as nn, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.norm = RMSNorm(input_shape[-1])
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, y):
        residual = x - mish(y) * self.alpha
        out = self.norm(residual)
        if _ECL_DBG: print(f"[ECL:resdecomp] res_norm={residual.norm().item():.4f} alpha={self.alpha.item():.4f}", file=sys.stderr)
        return out
