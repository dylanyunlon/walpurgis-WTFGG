"""
D2STGNN CardGame variant — dataloader.py
Algorithm changes vs upstream:
  1. Mixup data augmentation: randomly mix pairs of samples (x and y)
     with a Beta-distributed ratio during training
  2. Circular padding: instead of repeating last sample for batch padding,
     wrap around to the beginning of the dataset
"""

import os
import sys
import numpy as np

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module=""):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min():.6f} max={tensor.max():.6f} "
               f"mean={tensor.mean():.6f} std={tensor.std():.6f}")
    else:
        msg = f"[CG-DBG:{tag}] value={tensor}"
    print(msg, file=sys.stderr)


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False,
                 mixup_alpha=0.2, enable_mixup=False):
        """Load train/val/test data and get a dataloader.

        Args:
            xs (np.array): history sequence, [num_samples, history_len, num_nodes, num_feats].
            ys (np.array): future sequence, [num_samples, future_len, num_nodes, num_feats].
            batch_size (int): batch size
            pad_with_last_sample (bool): pad with the last sample to make number of samples divisible to batch_size.
            shuffle (bool): shuffle dataset.
            mixup_alpha (float): Beta distribution parameter for mixup (CARDGAME).
            enable_mixup (bool): whether to enable mixup augmentation (CARDGAME).
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.mixup_alpha = mixup_alpha
        self.enable_mixup = enable_mixup

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # --- CARDGAME: circular padding instead of repeating last sample ---
                x_padding = xs[:num_padding]
                y_padding = ys[:num_padding]
                _dbg("dataloader.circular_pad", f"num_padding={num_padding}", "dataloader")
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def _mixup_batch(self, x_batch, y_batch):
        """Apply mixup augmentation to a batch.

        Randomly interpolate pairs of samples using a Beta-distributed lambda.

        Args:
            x_batch: np.array, shape [B, L, N, F]
            y_batch: np.array, shape [B, ...]

        Returns:
            mixed_x, mixed_y: np.arrays with same shapes
        """
        if not self.enable_mixup or self.mixup_alpha <= 0:
            return x_batch, y_batch
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x_batch.shape[0]
        perm = np.random.permutation(batch_size)
        mixed_x = lam * x_batch + (1 - lam) * x_batch[perm]
        mixed_y = lam * y_batch + (1 - lam) * y_batch[perm]
        _dbg("mixup.lambda", f"{lam:.4f}", "dataloader")
        return mixed_x, mixed_y

    def get_iterator(self):
        """Fetch a batch of data."""
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                # --- CARDGAME: optional mixup augmentation ---
                x_i, y_i = self._mixup_batch(x_i, y_i)
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
