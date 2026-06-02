"""
Batch data loader with padding, shuffle, and debug hooks.
"""
import sys
import numpy as np

_DBG = ("--debug-loader" in sys.argv)


class DataLoader:
    """Iterable batch feeder for train / val / test splits."""

    def __init__(self, xs, ys, batch_size,
                 pad_last=True, shuffle=False):
        self.batch_size = batch_size
        self._cursor = 0

        # pad so total is divisible by batch_size
        if pad_last:
            n_pad = (batch_size - len(xs) % batch_size) % batch_size
            if n_pad > 0:
                xs = np.concatenate([xs, np.repeat(xs[-1:], n_pad, axis=0)], 0)
                ys = np.concatenate([ys, np.repeat(ys[-1:], n_pad, axis=0)], 0)

        self.n_samples = len(xs)
        self.num_batch = self.n_samples // batch_size
        self.xs = xs
        self.ys = ys

        if shuffle:
            self.shuffle()

        if _DBG:
            print(f"[DBG:loader] DataLoader  samples={self.n_samples}  "
                  f"batches={self.num_batch}  bs={batch_size}  "
                  f"x.shape={xs.shape}  y.shape={ys.shape}")

    def __len__(self):
        return self.num_batch

    def shuffle(self):
        perm = np.random.permutation(self.n_samples)
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]
        if _DBG:
            print(f"[DBG:loader] shuffled {self.n_samples} samples")

    def get_iterator(self):
        self._cursor = 0

        def _gen():
            while self._cursor < self.num_batch:
                lo = self.batch_size * self._cursor
                hi = min(self.n_samples, self.batch_size * (self._cursor + 1))
                xb = self.xs[lo:hi]
                yb = self.ys[lo:hi]
                if _DBG and self._cursor < 2:
                    print(f"[DBG:loader] batch {self._cursor}  "
                          f"x=[{lo}:{hi}]  x_range=[{xb.min():.3f},{xb.max():.3f}]")
                yield (xb, yb)
                self._cursor += 1

        return _gen()
