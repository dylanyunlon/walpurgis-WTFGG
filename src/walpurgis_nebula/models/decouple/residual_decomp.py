"""Nebula residual: BatchNorm + ELU + learnable affine transform."""
import torch, torch.nn as nn, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'


class ResidualDecomp(nn.Module):
    """Nebula residual decomposition with BatchNorm + ELU + affine.
    Replaces upstream LayerNorm + ReLU with:
    - BatchNorm1d for cross-sample normalization
    - ELU activation (smooth, non-zero gradient for negatives)
    - Learnable affine scale+shift for adaptive residual weighting."""

    def __init__(self, input_shape):
        super().__init__()
        dim = input_shape[-1]
        self.bn = nn.BatchNorm1d(dim)
        self.ac = nn.ELU(alpha=1.0)
        # Learnable affine: scale and shift for residual blending
        self.affine_scale = nn.Parameter(torch.ones(dim))
        self.affine_shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x, y):
        """x: original signal, y: learned component to subtract."""
        residual = x - self.ac(y)
        # Apply BatchNorm: need to reshape for BN1d [*, D] -> [N, D]
        orig_shape = residual.shape
        flat = residual.reshape(-1, orig_shape[-1])
        normed = self.bn(flat)
        normed = normed.view(orig_shape)
        # Affine transform
        out = normed * self.affine_scale + self.affine_shift
        if _NEB_DBG:
            print(f"[NEB:decomp@residual_decomp] residual_norm={residual.norm().item():.4f} "
                  f"scale_mean={self.affine_scale.mean().item():.4f} out_norm={out.norm().item():.4f}", file=sys.stderr)
        return out
