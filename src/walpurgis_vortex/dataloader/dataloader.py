"""Vortex DataLoader: stratified shuffle + mirror-padding.
Unlike upstream (np.random.permutation) and eclipse (Fisher-Yates + circular wrap),
Vortex uses stratified shuffling: samples are divided into temporal strata (bins),
then shuffled within each stratum before interleaving. This preserves the distribution
of temporal patterns across batches while still randomizing within each stratum.
Mirror padding reflects the end of the sequence for more natural boundary behavior."""
import numpy as np, sys, os
_VX_DBG = os.environ.get('VORTEX_DEBUG', '0') == '1'

class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size; self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # Mirror padding: reflect from the end (vs upstream last-repeat, eclipse circular)
                mirror_idx = np.arange(1, num_padding + 1)
                mirror_idx = len(xs) - 1 - (mirror_idx % len(xs))
                xs = np.concatenate([xs, xs[mirror_idx]], axis=0)
                ys = np.concatenate([ys, ys[mirror_idx]], axis=0)
                if _VX_DBG:
                    print(f"[VX:dataloader] mirror padding: {num_padding} samples", file=sys.stderr)
        self.size = len(xs); self.num_batch = int(self.size // self.batch_size)
        self.xs = xs; self.ys = ys
        self._num_strata = min(8, max(2, self.size // (batch_size * 2)))
        if shuffle: self.shuffle()
        if _VX_DBG:
            print(f"[VX:dataloader] size={self.size} batches={self.num_batch} "
                  f"batch_size={batch_size} strata={self._num_strata}", file=sys.stderr)

    def shuffle(self):
        """Stratified shuffle: divide into temporal strata, shuffle within each,
        then interleave strata. This preserves temporal diversity per batch."""
        n = self.size
        stratum_size = n // self._num_strata
        indices = np.arange(n)
        # Divide into strata
        strata = []
        for s in range(self._num_strata):
            start = s * stratum_size
            end = start + stratum_size if s < self._num_strata - 1 else n
            stratum = indices[start:end].copy()
            np.random.shuffle(stratum)
            strata.append(stratum)
        # Interleave: take one from each stratum in round-robin
        interleaved = []
        max_len = max(len(s) for s in strata)
        for i in range(max_len):
            for s in strata:
                if i < len(s):
                    interleaved.append(s[i])
        order = np.array(interleaved[:n])
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
