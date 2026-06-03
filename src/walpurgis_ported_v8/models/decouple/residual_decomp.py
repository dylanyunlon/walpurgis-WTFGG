import torch
import torch.nn as nn
import sys

_DBG = ("--dbg" in sys.argv)


class ResidualDecomp(nn.Module):
    """算法改动: EMA-style residual decomposition
    原版: u = LayerNorm(x - ReLU(y))
    改为: 引入可学习 momentum alpha (sigmoid 约束到 [0,1])
          u = LayerNorm(alpha * x - (1 - alpha) * ReLU(y))
    当 alpha -> 1 时退化为原版 (忽略 y 的贡献),
    当 alpha -> 0.5 时是等权混合,
    让网络自己学 residual 分解的激进程度
    """

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()
        # 可学习 momentum, 初始化为 logit(0.8)≈1.386 使得初始 alpha≈0.8 接近原版行为
        self._raw_alpha = nn.Parameter(torch.tensor(1.386))

    def forward(self, x, y):
        alpha = torch.sigmoid(self._raw_alpha)
        u = alpha * x - (1.0 - alpha) * self.ac(y)
        u = self.ln(u)
        if _DBG:
            with torch.no_grad():
                print(f"[DBG][ResidualDecomp] alpha={alpha.item():.4f}  "
                      f"u_mean={u.mean().item():.5f}  "
                      f"u_std={u.std().item():.5f}", flush=True)
        return u
