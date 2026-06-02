"""
Minimal batch DataLoader for time-series forecasting.
Ref: Graph-WaveNet util.py pattern.
"""

import numpy as np
import sys

_DBG_DL = ("--debug-dl" in sys.argv) or False


class DataLoader:
    """
    Load train/val/test arrays and iterate in fixed-size batches.

    Parameters
    ----------
    xs : np.ndarray   – history window  [N_samples, L_in,  N_nodes, D]
    ys : np.ndarray   – target window   [N_samples, L_out, N_nodes, D]
    batch_size : int
    pad_last : bool   – pad to make sample count divisible by batch_size
    shuffle : bool    – shuffle on construction
    """

    def __init__(self, xs, ys, batch_size, pad_last=True, shuffle=False):
        self.batch_size = batch_size
        self._cursor = 0

        if pad_last:
            n_pad = (batch_size - len(xs) % batch_size) % batch_size
            if n_pad > 0:
                xs = np.concatenate([xs, np.repeat(xs[-1:], n_pad, axis=0)], axis=0)
                ys = np.concatenate([ys, np.repeat(ys[-1:], n_pad, axis=0)], axis=0)

        self.total_samples = len(xs)
        self.num_batch = self.total_samples // self.batch_size
        self.xs = xs
        self.ys = ys

        if shuffle:
            self.shuffle()

        if _DBG_DL:
            print(f"[DBG:dataloader] init  samples={self.total_samples}  "
                  f"batches={self.num_batch}  x_shape={xs.shape}  y_shape={ys.shape}")

    def shuffle(self):
        perm = np.random.permutation(self.total_samples)
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Yield (x_batch, y_batch) tuples."""
        self._cursor = 0

        def _generate():
            while self._cursor < self.num_batch:
                lo = self.batch_size * self._cursor
                hi = min(self.total_samples, self.batch_size * (self._cursor + 1))
                xb = self.xs[lo:hi]
                yb = self.ys[lo:hi]
                if _DBG_DL and self._cursor % max(1, self.num_batch // 5) == 0:
                    print(f"[DBG:dataloader] batch {self._cursor}/{self.num_batch}  "
                          f"x_range=[{xb.min():.4g},{xb.max():.4g}]  "
                          f"y_range=[{yb.min():.4g},{yb.max():.4g}]")
                yield (xb, yb)
                self._cursor += 1

        return _generate()
