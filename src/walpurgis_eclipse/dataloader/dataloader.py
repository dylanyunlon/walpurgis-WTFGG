"""Eclipse DataLoader: Fisher-Yates shuffle + circular-wrap padding."""
import numpy as np, sys, os
_ECL_DBG = os.environ.get('ECLIPSE_DEBUG', '0') == '1'

class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size; self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # Circular wrap padding (vs upstream last-sample repeat)
                wrap_idx = np.arange(num_padding) % len(xs)
                xs = np.concatenate([xs, xs[wrap_idx]], axis=0)
                ys = np.concatenate([ys, ys[wrap_idx]], axis=0)
                if _ECL_DBG: print(f"[ECL:dataloader] circular padding: {num_padding} samples", file=sys.stderr)
        self.size = len(xs); self.num_batch = int(self.size // self.batch_size)
        self.xs = xs; self.ys = ys
        if shuffle: self.shuffle()
        if _ECL_DBG: print(f"[ECL:dataloader] size={self.size} batches={self.num_batch} batch_size={batch_size}", file=sys.stderr)

    def shuffle(self):
        # Fisher-Yates in-place shuffle (vs upstream np.random.permutation)
        n = self.size
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            self.xs[i], self.xs[j] = self.xs[j].copy(), self.xs[i].copy()
            self.ys[i], self.ys[j] = self.ys[j].copy(), self.ys[i].copy()

    def __len__(self): return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                s = self.batch_size * self.current_ind
                e = min(self.size, self.batch_size * (self.current_ind + 1))
                yield (self.xs[s:e, ...], self.ys[s:e, ...])
                self.current_ind += 1
        return _wrapper()
