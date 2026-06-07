"""Nebula DataLoader: stride-based sub-sampling + modular-arithmetic padding."""
import numpy as np, sys, os
_NEB_DBG = os.environ.get('NEBULA_DEBUG', '0') == '1'

class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size; self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # Modular-arithmetic padding: mirror-reflect indices
                pad_idx = np.array([len(xs) - 1 - (i % len(xs)) for i in range(num_padding)])
                xs = np.concatenate([xs, xs[pad_idx]], axis=0)
                ys = np.concatenate([ys, ys[pad_idx]], axis=0)
                if _NEB_DBG: print(f"[NEB:dataloader] mirror padding: {num_padding} samples", file=sys.stderr)
        self.size = len(xs); self.num_batch = int(self.size // self.batch_size)
        self.xs = xs; self.ys = ys
        if shuffle: self.shuffle()
        if _NEB_DBG: print(f"[NEB:dataloader] size={self.size} batches={self.num_batch} batch_size={batch_size}", file=sys.stderr)

    def shuffle(self):
        # Stride-based sub-sampling shuffle: split into strides, shuffle within each
        n = self.size
        stride = max(1, n // 8)
        perm = np.arange(n)
        for start in range(0, n, stride):
            end = min(start + stride, n)
            sub = perm[start:end]
            np.random.shuffle(sub)
            perm[start:end] = sub
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]

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
