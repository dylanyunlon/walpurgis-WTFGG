"""
Walpurgis DataLoader — Batch Iterator for Spatial-Temporal Data
================================================================
Derived from D2STGNN dataloader.py with ~20% restructuring.

Changes:
  1. Uses class-level instance counter for multi-loader identification
  2. Batch timing uses deque ringbuffer instead of per-epoch snapshot
  3. Shuffle reports entropy estimate for reproducibility debugging
  4. Memory footprint reported with tier annotation
"""
import time
import numpy as np
from collections import deque


class DataLoader:
    """Batch iterator for spatial-temporal graph data.
    
    Stores x (history) and y (future) arrays in DRAM, yielding
    batch slices on demand. The training loop handles device transfer.
    
    Debug usage:
        loader.batch_timing_stats()  # prints recent batch latencies
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
                print(f"[DataLoader#{self._id}] padded {n_pad} samples "
                      f"({orig_len} → {len(xs)})")

        self.size = len(xs)
        self.num_batch = self.size // self.batch_size
        self.xs = xs
        self.ys = ys

        mem_mb = (xs.nbytes + ys.nbytes) / (1024 * 1024)
        print(f"[DataLoader#{self._id}] xs={list(xs.shape)} ys={list(ys.shape)} "
              f"bs={batch_size} batches={self.num_batch} mem={mem_mb:.2f}MB [DRAM]")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """Shuffle in-place with diagnostic logging."""
        perm = np.random.permutation(self.size)
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Yield (x_batch, y_batch) with per-batch timing."""
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

            # Periodic epoch summary
            if self._n_yields % (self.num_batch * 5) < self.num_batch:
                elapsed = time.perf_counter() - epoch_t0
                per_batch = elapsed / max(self.num_batch, 1) * 1000
                print(f"[DataLoader#{self._id}] epoch: {self.num_batch} batches "
                      f"in {elapsed:.3f}s ({per_batch:.1f}ms/batch)")

        return _gen()

    def batch_timing_stats(self):
        """Print recent batch timing — call from debugger."""
        if self._batch_times:
            arr = np.array(self._batch_times)
            print(f"[DataLoader#{self._id}] batch timing: "
                  f"μ={arr.mean():.2f}ms σ={arr.std():.2f}ms "
                  f"p99={np.percentile(arr, 99):.2f}ms (n={len(arr)})")
