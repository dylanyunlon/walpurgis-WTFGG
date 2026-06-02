"""Walpurgis v4 datasets — raw data preprocessing and adjacency generation.

Each subdirectory under raw_data/ contains scripts to convert raw
traffic sensor recordings into the train/val/test .npz splits
consumed by the Walpurgis data loader.

Tier note: preprocessing is CPU-only; no GPU memory placement applies.
"""
