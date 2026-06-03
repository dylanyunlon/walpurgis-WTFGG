import torch.nn as nn

# Delta vs upstream:
#   1. Activation: ReLU → GELU (smoother gradient around zero)

class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.GELU()       # delta 1: GELU replaces ReLU

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u
