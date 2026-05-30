"""Walpurgis model package — D2STGNN adapted for heterogeneous-memory
temporal-subgraph engine.

Module tier placement summary:
    HBM:   model.D2STGNN (forward), DifBlock (ST conv), InhBlock (AR loop),
           DynamicGraphConstructor.DistanceFunction (O(N^2) attention)
    GDDR:  MultiOrder (graph powers), DifBlock (forecast branch)
    DRAM:  EstimationGate, ResidualDecomp, Mask, Normalizer, PositionalEncoding (buffer)
"""

from .model import D2STGNN

__all__ = ["D2STGNN"]
