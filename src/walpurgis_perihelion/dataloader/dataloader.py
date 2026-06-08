"""
DataLoader — Perihelion变体
改动: 带频域感知的shuffle策略(先按信号能量分层, 再内部随机)
      配合FFT Band-Pass分解的数据预处理思路
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

        # 计算每个样本的频谱能量指纹用于分层采样
        self._energy_bins = None
        if self.size > 0 and xs.ndim >= 3:
            try:
                signal = xs[:, :, 0, 0] if xs.ndim == 4 else xs[:, :, 0]
                fft_mag = np.abs(np.fft.rfft(signal, axis=1))
                self._energy_bins = fft_mag.sum(axis=1)
            except Exception:
                self._energy_bins = None

        if _is_debug():
            _dbg("dataloader.init",
                 f"samples={self.size} batches={self.num_batch} "
                 f"x_shape={xs.shape} y_shape={ys.shape} "
                 f"energy_aware={'yes' if self._energy_bins is not None else 'no'}")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """频域感知shuffle: 前期按能量分层shuffle, 后期完全随机"""
        self._shuffle_epoch += 1
        if (self._shuffle_epoch <= 2
                and self._energy_bins is not None):
            # 按频谱能量分为4层, 层内shuffle, 层间保序
            n_strata = 4
            sorted_idx = np.argsort(self._energy_bins)
            stratum_size = max(1, self.size // n_strata)
            new_idx = []
            for s in range(n_strata):
                start = s * stratum_size
                end = min((s + 1) * stratum_size, self.size)
                chunk = sorted_idx[start:end]
                np.random.shuffle(chunk)
                new_idx.append(chunk)
            # 余数
            remainder_start = n_strata * stratum_size
            if remainder_start < self.size:
                tail = sorted_idx[remainder_start:]
                np.random.shuffle(tail)
                new_idx.append(tail)
            perm = np.concatenate(new_idx)
            self.xs = self.xs[perm]
            self.ys = self.ys[perm]
            self._energy_bins = self._energy_bins[perm]
            _dbg("dataloader.shuffle",
                 f"stratified (epoch {self._shuffle_epoch})")
        else:
            permutation = np.random.permutation(self.size)
            self.xs = self.xs[permutation]
            self.ys = self.ys[permutation]
            if self._energy_bins is not None:
                self._energy_bins = self._energy_bins[permutation]
            _dbg("dataloader.shuffle",
                 f"random (epoch {self._shuffle_epoch})")

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
