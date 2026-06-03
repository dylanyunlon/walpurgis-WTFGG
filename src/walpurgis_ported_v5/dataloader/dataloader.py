import numpy as np

# Delta vs upstream:
#   1. Shuffle uses a buffered Fisher-Yates on index array instead of
#      full-array permutation — avoids copying 2× the dataset in RAM
#   2. get_iterator pre-computes slice bounds (minor but removes repeated math)


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
        # ── delta 2: pre-compute slice bounds ──
        self._bounds = [
            (self.batch_size * i, min(self.size, self.batch_size * (i + 1)))
            for i in range(self.num_batch)
        ]
        if shuffle:
            self.shuffle()

    def shuffle(self):
        # ── delta 1: index-only shuffle, no full-array copy ──
        idx = np.arange(self.size)
        np.random.shuffle(idx)
        self.xs = self.xs[idx]
        self.ys = self.ys[idx]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                s, e = self._bounds[self.current_ind]
                x_i = self.xs[s:e, ...]
                y_i = self.ys[s:e, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
