"""Helix DataLoader: spiral-interleave shuffle + cosine-weighted padding.
Unlike upstream (np.random.permutation) and vortex (stratified shuffle + mirror padding),
Helix uses spiral-interleave shuffling: samples are arranged in a spiral pattern
where nearby temporal samples are spread across batches following a golden-ratio stride.
Cosine-weighted padding blends boundary samples with their neighbors."""
import numpy as np, sys, os
_HX_DBG = os.environ.get('HELIX_DEBUG', '0') == '1'

class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size; self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # Helix: cosine-weighted padding — blend last samples with neighbors
                pad_xs = []
                pad_ys = []
                for i in range(num_padding):
                    # Weight blends from last sample toward earlier ones
                    alpha = 0.5 * (1 + np.cos(np.pi * i / max(num_padding, 1)))
                    idx_a = len(xs) - 1
                    idx_b = max(0, len(xs) - 2 - i)
                    pad_xs.append(alpha * xs[idx_a] + (1 - alpha) * xs[idx_b])
                    pad_ys.append(alpha * ys[idx_a] + (1 - alpha) * ys[idx_b])
                xs = np.concatenate([xs, np.stack(pad_xs)], axis=0)
                ys = np.concatenate([ys, np.stack(pad_ys)], axis=0)
                if _HX_DBG:
                    print(f"[HX:dataloader] cosine padding: {num_padding} samples", file=sys.stderr)
        self.size = len(xs); self.num_batch = int(self.size // self.batch_size)
        self.xs = xs; self.ys = ys
        if shuffle: self.shuffle()
        if _HX_DBG:
            print(f"[HX:dataloader] size={self.size} batches={self.num_batch} "
                  f"batch_size={batch_size}", file=sys.stderr)

    def shuffle(self):
        """Helix spiral-interleave shuffle: use golden-ratio stride to spread
        temporally nearby samples across the dataset, achieving good temporal
        diversity per batch without explicit stratification."""
        n = self.size
        golden_ratio = (1 + np.sqrt(5)) / 2
        stride = int(np.round(n / golden_ratio))
        if stride == 0: stride = 1
        # Generate spiral order using golden-ratio stepping
        spiral_order = []
        current = 0
        visited = set()
        for _ in range(n):
            while current in visited:
                current = (current + 1) % n
            spiral_order.append(current)
            visited.add(current)
            current = (current + stride) % n
        order = np.array(spiral_order[:n])
        self.xs = self.xs[order]
        self.ys = self.ys[order]

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
