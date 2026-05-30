import time

import torch
import torch.nn as nn


class Mask(nn.Module):
    """Apply static adjacency mask to dynamic graph weights.

    Walpurgis notes:
    - The mask tensors are static (loaded once from adj data) and can
      safely reside in DRAM — they are only read during forward.
    - Epsilon (1e-7) is added to avoid exact zeros which would break
      gradient flow through masked positions.
    """

    _call_count = 0

    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']
        print(f"[Walpurgis::Mask] init with {len(self.mask)} adjacency masks")
        for i, m in enumerate(self.mask):
            nnz = (m.abs() > 1e-7).sum().item() if hasattr(m, 'abs') else 'N/A'
            print(f"  mask[{i}] shape={list(m.shape)} nonzero={nnz}")

    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj

    def forward(self, adj):
        Mask._call_count += 1
        _verbose = (Mask._call_count <= 3 or Mask._call_count % 500 == 0)

        t0 = time.perf_counter()
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if _verbose:
            print(f"[Walpurgis::Mask::forward] call#{Mask._call_count} "
                  f"num_adj={len(adj)} elapsed={elapsed_ms:.3f}ms")

        return result
