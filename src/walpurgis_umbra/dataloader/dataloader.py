"""
DataLoader — Umbra变体
改动: 分层采样shuffle (按时间戳分桶后桶内+桶间shuffle)
     添加batch统计诊断, 数据完整性校验
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
            # 数据完整性检查
            if np.isnan(xs).any():
                _dbg("dataloader.WARN", "NaN detected in input xs!")
            if np.isnan(ys).any():
                _dbg("dataloader.WARN", "NaN detected in input ys!")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """分层采样shuffle: 将数据按时间窗口分桶, 桶间打乱, 桶内也打乱
        保证相邻时间步有一定概率在同一batch, 但又不完全顺序"""
        self._shuffle_epoch += 1
        num_buckets = max(1, self.size // (self.batch_size * 3))
        bucket_size = self.size // num_buckets

        bucket_order = np.random.permutation(num_buckets)
        new_xs_parts, new_ys_parts = [], []
        for bi in bucket_order:
            s = bi * bucket_size
            e = min(s + bucket_size, self.size)
            inner_perm = np.random.permutation(e - s)
            new_xs_parts.append(self.xs[s:e][inner_perm])
            new_ys_parts.append(self.ys[s:e][inner_perm])
        # 残余样本
        remainder_start = num_buckets * bucket_size
        if remainder_start < self.size:
            new_xs_parts.append(self.xs[remainder_start:])
            new_ys_parts.append(self.ys[remainder_start:])

        self.xs = np.concatenate(new_xs_parts, axis=0)
        self.ys = np.concatenate(new_ys_parts, axis=0)

        _dbg("dataloader.shuffle",
             f"stratified epoch={self._shuffle_epoch} "
             f"buckets={num_buckets}")

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
                         f"x range=[{x_i[..., 0].min():.2f},"
                         f"{x_i[..., 0].max():.2f}] "
                         f"y range=[{y_i[..., 0].min():.2f},"
                         f"{y_i[..., 0].max():.2f}]")
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
