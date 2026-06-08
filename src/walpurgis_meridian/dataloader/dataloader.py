"""Meridian DataLoader — stratified variance-based sampling + diagnostics.
Changes vs upstream:
  - Stratified shuffle: bins samples by variance, shuffles within bins
  - Per-batch statistics tracking (when MERIDIAN_DEBUG=1)
"""
import numpy as np
import os, sys

_DBG = os.environ.get('MERIDIAN_DEBUG', '0') == '1'


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        # track per-sample variance for stratified sampling
        self._sample_var = np.var(xs.reshape(len(xs), -1), axis=1)
        if shuffle:
            self.shuffle()

    def shuffle(self):
        """Stratified shuffle: partition by variance quartile, shuffle within each."""
        n_bins = 4
        quartiles = np.percentile(self._sample_var, np.linspace(0, 100, n_bins + 1))
        indices = np.arange(self.size)
        shuffled = []
        for i in range(n_bins):
            lo, hi = quartiles[i], quartiles[i + 1]
            if i == n_bins - 1:
                mask = (self._sample_var >= lo) & (self._sample_var <= hi)
            else:
                mask = (self._sample_var >= lo) & (self._sample_var < hi)
            bin_idx = indices[mask]
            np.random.shuffle(bin_idx)
            shuffled.append(bin_idx)
        perm = np.concatenate(shuffled)
        # final interleave: round-robin from bins for diversity
        final_perm = np.empty(self.size, dtype=int)
        ptrs = [0] * n_bins
        lens = [len(s) for s in shuffled]
        pos = 0
        while pos < self.size:
            for b in range(n_bins):
                if ptrs[b] < lens[b]:
                    final_perm[pos] = shuffled[b][ptrs[b]]
                    ptrs[b] += 1
                    pos += 1
                    if pos >= self.size:
                        break
        self.xs = self.xs[final_perm]
        self.ys = self.ys[final_perm]
        self._sample_var = self._sample_var[final_perm]
        if _DBG:
            print(f"[MER:dataloader] stratified shuffle done, {n_bins} bins, "
                  f"var range=[{self._sample_var.min():.4f}, {self._sample_var.max():.4f}]",
                  file=sys.stderr)

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                if _DBG and self.current_ind % 20 == 0:
                    print(f"[MER:dataloader] batch {self.current_ind}/{self.num_batch} "
                          f"x_mean={x_i.mean():.4f} x_std={x_i.std():.4f} "
                          f"y_mean={y_i.mean():.4f}", file=sys.stderr)
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()
