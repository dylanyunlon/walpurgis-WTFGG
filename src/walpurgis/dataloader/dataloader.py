"""
Walpurgis v2 DataLoader — Batch Iterator with Prefetch Hint
==============================================================
Delta: adds `.prefetch_hint(device)` that pre-allocates a pinned-memory
buffer for the next batch, enabling overlap of data transfer and compute
on systems with CUDA async copy.
"""
import time
import numpy as np
from collections import deque


class DataLoader:
    """Batch iterator for spatial-temporal graph data.

    Debug: loader.batch_timing_stats()
    """

    _n_instances = 0

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        DataLoader._n_instances += 1
        self._id = DataLoader._n_instances
        self.batch_size = batch_size
        self._n_yields = 0
        self._batch_times = deque(maxlen=200)

        orig_len = len(xs)
        if pad_with_last_sample:
            n_pad = (batch_size - (len(xs) % batch_size)) % batch_size
            if n_pad > 0:
                xs = np.concatenate([xs, np.repeat(xs[-1:], n_pad, axis=0)], axis=0)
                ys = np.concatenate([ys, np.repeat(ys[-1:], n_pad, axis=0)], axis=0)
                print(f"[DataLoader#{self._id}] padded {n_pad} ({orig_len}→{len(xs)})")

        self.size = len(xs)
        self.num_batch = self.size // self.batch_size
        self.xs = xs
        self.ys = ys
        self._pinned = None

        mem_mb = (xs.nbytes + ys.nbytes) / (1024 * 1024)
        print(
            f"[DataLoader#{self._id}] xs={list(xs.shape)} ys={list(ys.shape)} "
            f"bs={batch_size} batches={self.num_batch} mem={mem_mb:.2f}MB"
        )
        if shuffle:
            self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]

    def __len__(self):
        return self.num_batch

    def prefetch_hint(self, device):
        """Pre-allocate pinned buffer for async H2D transfer."""
        try:
            import torch
            if torch.cuda.is_available() and "cuda" in str(device):
                x_shape = (self.batch_size, *self.xs.shape[1:])
                self._pinned = torch.empty(x_shape, pin_memory=True)
                print(f"[DataLoader#{self._id}] pinned buffer allocated for {device}")
        except Exception:
            pass

    def get_iterator(self):
        self._cursor = 0
        epoch_t0 = time.perf_counter()

        def _gen():
            while self._cursor < self.num_batch:
                t0 = time.perf_counter()
                start = self.batch_size * self._cursor
                end = min(self.size, self.batch_size * (self._cursor + 1))
                x_batch = self.xs[start:end]
                y_batch = self.ys[start:end]
                self._n_yields += 1
                self._batch_times.append((time.perf_counter() - t0) * 1000)
                yield (x_batch, y_batch)
                self._cursor += 1

            if self._n_yields % (self.num_batch * 5) < self.num_batch:
                elapsed = time.perf_counter() - epoch_t0
                print(
                    f"[DataLoader#{self._id}] epoch: {self.num_batch} batches "
                    f"in {elapsed:.3f}s"
                )

        return _gen()

    def batch_timing_stats(self):
        if self._batch_times:
            arr = np.array(self._batch_times)
            print(
                f"[DataLoader#{self._id}] batch timing: "
                f"μ={arr.mean():.2f}ms p99={np.percentile(arr, 99):.2f}ms"
            )
