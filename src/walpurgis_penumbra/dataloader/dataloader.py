"""
DataLoader — Penumbra变体
改动: 添加batch统计诊断, 渐进式shuffle (先粗粒度再细粒度)
"""
import numpy as np
from .. import _dbg, _is_debug


class DataLoader(object):
    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        self._shuffle_epoch = 0

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

        if _is_debug():
            _dbg("dataloader.init",
                 f"samples={self.size} batches={self.num_batch} "
                 f"x_shape={xs.shape} y_shape={ys.shape}")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """渐进式shuffle: 前几个epoch按块shuffle, 后面完全随机"""
        self._shuffle_epoch += 1
        if self._shuffle_epoch <= 2:
            # 粗粒度: 按chunk(每4个batch)打乱
            chunk = self.batch_size * 4
            n_chunks = max(1, self.size // chunk)
            chunk_idx = np.random.permutation(n_chunks)
            new_xs, new_ys = [], []
            for ci in chunk_idx:
                s, e = ci * chunk, min((ci + 1) * chunk, self.size)
                new_xs.append(self.xs[s:e])
                new_ys.append(self.ys[s:e])
            # 残余
            remainder = n_chunks * chunk
            if remainder < self.size:
                new_xs.append(self.xs[remainder:])
                new_ys.append(self.ys[remainder:])
            self.xs = np.concatenate(new_xs, axis=0)
            self.ys = np.concatenate(new_ys, axis=0)
            _dbg("dataloader.shuffle",
                 f"coarse (epoch {self._shuffle_epoch})")
        else:
            permutation = np.random.permutation(self.size)
            self.xs = self.xs[permutation]
            self.ys = self.ys[permutation]
            _dbg("dataloader.shuffle",
                 f"fine (epoch {self._shuffle_epoch})")

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                if _is_debug() and self.current_ind == 0:
                    _dbg("dataloader.first_batch",
                         f"x range=[{x_i[...,0].min():.2f},"
                         f"{x_i[...,0].max():.2f}] "
                         f"y range=[{y_i[...,0].min():.2f},"
                         f"{y_i[...,0].max():.2f}]")
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
