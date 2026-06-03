import torch
import torch.nn as nn
import sys

_DBG_DECOMP = ("--dbg-decomp" in sys.argv)


class ResidualDecomp(nn.Module):
    """Residual decomposition — 算法改动:
    用 ELU (alpha=1.0) 替代 ReLU。
    ReLU 在负半轴梯度为零会杀死信号; ELU 允许负值有小梯度,
    在减法残差中保留更多信息。
    """

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ELU(alpha=1.0)  # 替换 ReLU

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        if _DBG_DECOMP:
            with torch.no_grad():
                diff_ratio = (x - y).abs().mean().item()
                residual_norm = u.norm(dim=-1).mean().item()
                print(f"[DBG-DECOMP] |x-y|_mean={diff_ratio:.5f}  "
                      f"|residual|_mean={residual_norm:.5f}  "
                      f"y_activated_range=[{self.ac(y).min().item():.4f}, "
                      f"{self.ac(y).max().item():.4f}]")
        return u
