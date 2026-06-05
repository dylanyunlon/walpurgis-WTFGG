"""
DataLoader — Nightfall变体
算法改写: padding策略从'重复最后一个样本'改为'随机采样填充'
这减少了尾部样本的过度表示
"""
import numpy as np
from .. import _dbg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # 随机采样填充 (替代重复最后一个)
                pad_indices = np.random.choice(len(xs), num_padding, replace=True)
                x_padding = xs[pad_indices]
                y_padding = ys[pad_indices]
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        _dbg("dataloader", f"size={self.size} num_batch={self.num_batch} batch_size={batch_size}", "data")
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()
