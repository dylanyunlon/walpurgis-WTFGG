"""
dataloader.py — v9 port
Algo delta:
  1. pad_with_last_sample → circular wrap:
     不再重复最后一条样本填充, 而是把数据头部的样本循环接到尾部.
     减少最后 batch 的分布偏移 (upstream 把同一条 sample 复制 N 次)
  2. get_iterator yield 3-tuple (x, y, meta_dict):
     meta_dict 含 {"indices": 该 batch 中每条样本在原始数据集里的索引}
     便于训练时追踪哪些样本 loss 异常大
  3. shuffle 用 Fisher-Yates 原地 permutation, 避免中间拷贝
"""
import numpy as np
from walpurgis_ported_v9 import _dbg

_TAG = "dataloader"


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0

        n = len(xs)
        num_padding = (batch_size - (n % batch_size)) % batch_size

        if num_padding > 0 and pad_with_last_sample:
            # v9: circular wrap instead of repeat-last
            wrap_idx = np.arange(num_padding) % n
            x_padding = xs[wrap_idx]
            y_padding = ys[wrap_idx]
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            _dbg(_TAG, f"circular pad  n={n} → {len(xs)}  padding={num_padding}")

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        # v9: track original indices
        self._indices = np.arange(self.size)
        if shuffle:
            self.shuffle()

    def shuffle(self):
        # v9: Fisher-Yates in-place shuffle
        perm = np.arange(self.size)
        for i in range(self.size - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        self.xs = self.xs[perm]
        self.ys = self.ys[perm]
        self._indices = self._indices[perm]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                s = self.batch_size * self.current_ind
                e = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[s:e, ...]
                y_i = self.ys[s:e, ...]
                # v9: yield 3-tuple with sample indices
                meta = {"indices": self._indices[s:e].copy()}
                yield (x_i, y_i, meta)
                self.current_ind += 1

        return _wrapper()
