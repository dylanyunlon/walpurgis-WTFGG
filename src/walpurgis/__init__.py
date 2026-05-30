"""Walpurgis — heterogeneous-memory temporal-subgraph engine.

D2STGNN adaptation with tier-aware debug instrumentation for
profiling spatial-temporal graph neural networks across HBM/GDDR/DRAM
memory hierarchies.

Modules:
    models/         — D2STGNN model with per-component profiling
    utils/          — data loading, adjacency computation, training infra
    dataloader/     — batch iteration with memory footprint tracking
    datasets/       — raw data preprocessing scripts
    configs/        — YAML hyperparameter configurations per dataset
"""
