"""
Walpurgis v4 — Fourth-pass rewrite from the Walpurgis/D2STGNN lineage.

≈20% algorithmic delta vs v3 across all modules.  Key changes:
  - TensorProbe: added gradient-flow tracing & auto-bisect anomaly localiser
  - Model aggregation: attention-pool → *Mixture-of-Experts gating* with load-balancing
  - Embedding warmup: cosine-annealing → *inverse-sqrt schedule* (Vaswani et al.)
  - Skip gate: hard-concrete → *straight-through Gumbel-softmax* with annealable τ
  - Loss: adaptive-δ Huber → *log-cosh* smooth loss with horizon-aware Cauchy weighting
  - Trainer clip: Welford → *percentile-tracked AGC* (adaptive gradient clipping per-param)
  - DataLoader shuffle: stratified block → *reservoir-sampled block* with online quantile
  - DynGraph cache: Frobenius norm → *spectral hash* via top-k singular values
  - Full breakpoint/print instrumentation on every data structure at every pipeline stage
"""
