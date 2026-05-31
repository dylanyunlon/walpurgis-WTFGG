"""
Walpurgis — Heterogeneous-Memory Temporal-Subgraph Engine
==========================================================
D2STGNN adaptation with tier-aware debug instrumentation for profiling
spatial-temporal graph neural networks across HBM/GDDR/DRAM hierarchies.

Packages:
    models/      D2STGNN with per-component TensorProbe profiling
    utils/       Data loading, adjacency computation, training infra
    dataloader/  Batch iteration with memory footprint tracking
    datasets/    Raw data preprocessing scripts
    configs/     YAML hyperparameter configurations per dataset
"""
