"""
DataLoader — Transit变体
改动: 添加batch统计诊断, 按时间窗口分层采样(前期按时间块, 后期全局随机)
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
        """分层采样: 初期按时间窗口分块打乱, 后期全局随机"""
        self._shuffle_epoch += 1
        if self._shuffle_epoch <= 2:
            window = self.batch_size * 4
            n_windows = max(1, self.size // window)
            win_idx = np.random.permutation(n_windows)
            new_xs, new_ys = [], []
            for wi in win_idx:
                lo, hi = wi * window, min((wi + 1) * window, self.size)
                new_xs.append(self.xs[lo:hi])
                new_ys.append(self.ys[lo:hi])
            tail = n_windows * window
            if tail < self.size:
                new_xs.append(self.xs[tail:])
                new_ys.append(self.ys[tail:])
            self.xs = np.concatenate(new_xs, axis=0)
            self.ys = np.concatenate(new_ys, axis=0)
            _dbg("dataloader.shuffle",
                 f"stratified-window (epoch {self._shuffle_epoch})")
        else:
            perm = np.random.permutation(self.size)
            self.xs = self.xs[perm]
            self.ys = self.ys[perm]
            _dbg("dataloader.shuffle",
                 f"global-random (epoch {self._shuffle_epoch})")

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
