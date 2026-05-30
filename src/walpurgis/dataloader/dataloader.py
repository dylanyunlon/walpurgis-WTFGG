"""Walpurgis DataLoader — batch iterator for spatial-temporal data.

Walpurgis adaptations:
- Initialization prints full dataset statistics (shape, padding, memory)
- Shuffle operations are logged with seed info
- Iterator tracks batch-level timing for data pipeline profiling
- Memory tier annotation: data arrays are CPU-resident (DRAM) and
  transferred to device per-batch in the training loop
"""
import time

import numpy as np


class DataLoader(object):
    """Load train/val/test data and provide a batch iterator.

    Ref: https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    _instance_count = 0

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """Initialize DataLoader with Walpurgis diagnostics.

        Args:
            xs (np.array): history sequence [num_samples, history_len, num_nodes, num_feats]
            ys (np.array): future sequence  [num_samples, future_len, num_nodes, num_feats]
            batch_size (int): batch size
            pad_with_last_sample (bool): pad to make divisible by batch_size
            shuffle (bool): shuffle dataset
        """
        DataLoader._instance_count += 1
        self._id = DataLoader._instance_count

        self.batch_size = batch_size
        self.current_ind = 0
        self._total_yields = 0  # Walpurgis: track total batches yielded

        original_len = len(xs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            if num_padding > 0:
                print(f"[Walpurgis::DataLoader#{self._id}] padded {num_padding} samples "
                      f"({original_len} → {len(xs)})")

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

        # Walpurgis: memory footprint
        mem_mb = (xs.nbytes + ys.nbytes) / (1024 * 1024)
        print(f"[Walpurgis::DataLoader#{self._id}] init "
              f"xs={list(xs.shape)} ys={list(ys.shape)} "
              f"batch_size={batch_size} num_batch={self.num_batch} "
              f"mem={mem_mb:.2f}MB tier=DRAM")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """Shuffle dataset in-place."""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Fetch batches as a generator with Walpurgis timing."""
        self.current_ind = 0
        epoch_t0 = time.perf_counter()

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                self._total_yields += 1
                yield (x_i, y_i)
                self.current_ind += 1

            # Walpurgis: epoch-level summary (print every 5th epoch worth)
            if self._total_yields % (self.num_batch * 5) < self.num_batch:
                elapsed = time.perf_counter() - epoch_t0
                print(f"[Walpurgis::DataLoader#{self._id}] epoch iteration complete "
                      f"{self.num_batch} batches in {elapsed:.3f}s "
                      f"({elapsed / max(self.num_batch, 1) * 1000:.1f}ms/batch)")

        return _wrapper()
