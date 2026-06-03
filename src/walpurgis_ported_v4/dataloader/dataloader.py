"""
DataLoader — walpurgis_ported_v4
Modifications:
  - shuffle(): prints permutation checksum for reproducibility auditing
  - get_iterator(): counts total batches yielded across all epochs (v4 debug)
  - __init__: prints padding info
"""
import numpy as np
import sys


_V4_DEBUG = True


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """Dataloader with debug instrumentation.

        Args:
            xs: history sequence [num_samples, history_len, num_nodes, num_feats]
            ys: future sequence  [num_samples, future_len, num_nodes, num_feats]
            batch_size: batch size
            pad_with_last_sample: pad to make divisible by batch_size
            shuffle: shuffle on init
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self._total_batches_yielded = 0  # v4: lifetime counter

        num_padding = 0
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

        if _V4_DEBUG:
            print(f"[v4-DBG][DataLoader.__init__] samples={self.size} "
                  f"batches={self.num_batch} batch_size={batch_size} "
                  f"padded={num_padding} x_shape={xs.shape} y_shape={ys.shape}",
                  file=sys.stderr)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        if _V4_DEBUG:
            # v4: checksum for reproducibility verification
            cksum = int(np.sum(permutation[:8]))  # lightweight fingerprint
            print(f"[v4-DBG][DataLoader.shuffle] perm_checksum(first8)={cksum}",
                  file=sys.stderr)

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Batch generator with per-epoch debug summary (v4)."""
        self.current_ind = 0
        epoch_batch_count = 0

        def _wrapper():
            nonlocal epoch_batch_count
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
                epoch_batch_count += 1
                self._total_batches_yielded += 1

            if _V4_DEBUG and epoch_batch_count > 0:
                print(f"[v4-DBG][DataLoader.get_iterator] epoch done: "
                      f"{epoch_batch_count} batches, "
                      f"lifetime total={self._total_batches_yielded}",
                      file=sys.stderr)

        return _wrapper()
