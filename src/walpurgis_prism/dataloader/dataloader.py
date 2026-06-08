"""Prism DataLoader: block shuffle + reflection padding.
Unlike upstream (np.random.permutation) and vortex (stratified shuffle + mirror padding),
Prism uses block-based shuffling: consecutive samples are grouped into blocks of size K,
blocks are shuffled while preserving intra-block order. This maintains short-range
temporal coherence (useful for contrastive learning) while randomizing long-range order.
Reflection padding extends the dataset by reflecting from both ends."""
import numpy as np, sys, os
_PR_DBG = os.environ.get('PRISM_DEBUG', '0') == '1'


class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size; self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # Reflection padding: alternate front/back reflection
                reflect_idx = []
                for i in range(num_padding):
                    if i % 2 == 0:
                        # Reflect from end
                        reflect_idx.append(len(xs) - 1 - (i // 2) % len(xs))
                    else:
                        # Reflect from start
                        reflect_idx.append((i // 2) % len(xs))
                xs = np.concatenate([xs, xs[reflect_idx]], axis=0)
                ys = np.concatenate([ys, ys[reflect_idx]], axis=0)
                if _PR_DBG:
                    print(f"[PR:dataloader] reflection padding: {num_padding} samples", file=sys.stderr)
        self.size = len(xs); self.num_batch = int(self.size // self.batch_size)
        self.xs = xs; self.ys = ys
        # Block size for block shuffle (4-8 consecutive samples per block)
        self._block_size = min(8, max(4, batch_size // 2))
        if shuffle: self.shuffle()
        if _PR_DBG:
            print(f"[PR:dataloader] size={self.size} batches={self.num_batch} "
                  f"batch_size={batch_size} block_size={self._block_size}", file=sys.stderr)

    def shuffle(self):
        """Block shuffle: group consecutive samples into blocks, shuffle blocks.
        Preserves short-range temporal coherence within each block (good for
        contrastive learning where nearby samples share temporal patterns)."""
        n = self.size
        num_blocks = n // self._block_size
        # Create block indices
        block_order = np.random.permutation(num_blocks)
        indices = []
        for b in block_order:
            start = b * self._block_size
            end = min(start + self._block_size, n)
            indices.extend(range(start, end))
        # Append remainder
        remainder_start = num_blocks * self._block_size
        if remainder_start < n:
            indices.extend(range(remainder_start, n))
        order = np.array(indices[:n])
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
