from .dif_block import DifBlock

__all__ = ["DifBlock"]

# Walpurgis: diffusion block — spatial graph diffusion with ST convolution
# Tier affinity: HBM (graph conv is compute-bound, AR loop is latency-sensitive)
