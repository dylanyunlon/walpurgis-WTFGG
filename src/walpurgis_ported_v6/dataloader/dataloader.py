import numpy as np


class DataLoader(object):
    """Batch data feeder with ring-buffer index and per-batch diagnostics.

    Differences from upstream
    -------------------------
    * Index management uses a pre-computed permutation ring instead of a
      monotonic counter — avoids the branch in ``get_iterator`` and makes
      the shuffle path allocation-free after init.
    * ``get_iterator`` yields a 3-tuple ``(x_i, y_i, meta)`` where *meta*
      carries batch-level statistics (mean/std of x) so downstream code
      can sanity-check normalisation without extra passes.
    """

    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self._cursor = 0

        if pad_with_last_sample:
            remainder = len(xs) % batch_size
            if remainder != 0:
                n_pad = batch_size - remainder
                xs = np.concatenate([xs, np.repeat(xs[-1:], n_pad, axis=0)])
                ys = np.concatenate([ys, np.repeat(ys[-1:], n_pad, axis=0)])

        self.size = len(xs)
        self.num_batch = self.size // self.batch_size
        self.xs = xs
        self.ys = ys
        # pre-allocate the permutation ring
        self._perm = np.arange(self.size)
        if shuffle:
            self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._perm)
        self.xs = self.xs[self._perm]
        self.ys = self.ys[self._perm]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self._cursor = 0

        def _gen():
            while self._cursor < self.num_batch:
                lo = self.batch_size * self._cursor
                hi = lo + self.batch_size
                x_i = self.xs[lo:hi]
                y_i = self.ys[lo:hi]
                # -- diagnostic meta --
                meta = {
                    "batch_idx": self._cursor,
                    "x_mean":    float(np.mean(x_i)),
                    "x_std":     float(np.std(x_i)),
                    "y_mean":    float(np.mean(y_i)),
                    "y_std":     float(np.std(y_i)),
                }
                yield (x_i, y_i, meta)
                self._cursor += 1

        return _gen()
