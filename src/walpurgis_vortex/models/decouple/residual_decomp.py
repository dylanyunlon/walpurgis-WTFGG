"""Vortex residual decomposition: PowerNorm + Swish + exponential scaling.
Unlike upstream (LayerNorm+ReLU) and eclipse (RMSNorm+Mish+alpha),
Vortex uses PowerNorm (power-mean normalization with learnable exponent p),
Swish activation, and exponential scaling factor exp(learnable_log_scale)."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class PowerNorm(nn.Module):
    """Power-mean normalization: norm(x) = ((mean(|x|^p))^(1/p)) with learnable p.
    Generalizes LayerNorm (p=2) and RMSNorm, allowing the network to learn
    the optimal normalization power for each layer."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.log_p = nn.Parameter(torch.tensor(0.6931))  # init p~2.0 (log(2))
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        p = self.log_p.exp().clamp(min=0.5, max=4.0)  # constrain p
        x_abs = x.abs().clamp(min=self.eps)
        # Power mean: (E[|x|^p])^(1/p)
        power_mean = x_abs.pow(p).mean(dim=-1, keepdim=True).pow(1.0 / p)
        x_norm = x / power_mean.clamp(min=self.eps)
        return x_norm * self.scale + self.bias

def swish(x):
    """Swish: x * sigmoid(x). Smoother than ReLU/Mish, self-gating."""
    return x * torch.sigmoid(x)

class ResidualDecomp(nn.Module):
    """Vortex residual decomposition.
    residual = x - swish(y) * exp(log_scale)
    Exponential scaling allows the decomposition to learn the relative
    magnitude of the component to subtract, avoiding gradient issues
    from direct multiplication with alpha (eclipse)."""
    def __init__(self, input_shape):
        super().__init__()
        self.norm = PowerNorm(input_shape[-1])
        self.log_scale = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0 init

    def forward(self, x, y):
        scale = torch.exp(self.log_scale).clamp(max=5.0)
        residual = x - swish(y) * scale
        out = self.norm(residual)
        if _VX_DBG:
            print(f"[VX:resdecomp@residual_decomp] res_norm={residual.norm().item():.4f} "
                  f"scale={scale.item():.4f} p={self.norm.log_p.exp().item():.4f}", file=sys.stderr)
        return out
