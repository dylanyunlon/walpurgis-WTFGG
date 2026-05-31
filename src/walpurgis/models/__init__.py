"""Walpurgis model package — D2STGNN adapted for heterogeneous-memory research.

Tier placement guide:
    HBM:   D2STGNN forward, DifBlock (ST conv), InhBlock (AR loop),
           DynamicGraphConstructor.DistanceFunction (O(N²) pairwise)
    GDDR:  MultiOrder (graph powers), forecast branches
    DRAM:  EstimationGate, ResidualDecomp, Mask, Normalizer, PE buffers
"""

from .model import D2STGNN

__all__ = ["D2STGNN"]
