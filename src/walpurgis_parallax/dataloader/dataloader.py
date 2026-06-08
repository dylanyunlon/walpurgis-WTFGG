"""
DataLoader — Parallax变体
改动: 分层采样shuffle (按时间段分桶再抽样), 滑窗统计监控
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

        # 滑窗均值/方差追踪器 — 监控数据分布漂移
        self._window_stats = []

        if _is_debug():
            _dbg("dataloader.init",
                 f"samples={self.size} batches={self.num_batch} "
                 f"x_shape={xs.shape} y_shape={ys.shape}")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """分层采样shuffle: 将时间轴分段, 每段内随机打乱再交错"""
        self._shuffle_epoch += 1
        n_strata = max(1, min(8, self.size // (self.batch_size * 2)))
        stratum_size = self.size // n_strata
        new_indices = []
        for s in range(n_strata):
            start = s * stratum_size
            end = start + stratum_size if s < n_strata - 1 else self.size
            local_idx = np.arange(start, end)
            np.random.shuffle(local_idx)
            new_indices.append(local_idx)
        # 交错合并: 从每个stratum轮流取样
        merged = []
        max_len = max(len(idx) for idx in new_indices)
        for pos in range(max_len):
            for s_idx in new_indices:
                if pos < len(s_idx):
                    merged.append(s_idx[pos])
        permutation = np.array(merged)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]
        _dbg("dataloader.shuffle",
             f"stratified epoch={self._shuffle_epoch} "
             f"n_strata={n_strata}")

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
                    x_mu = x_i[..., 0].mean()
                    x_sig = x_i[..., 0].std()
                    self._window_stats.append((x_mu, x_sig))
                    _dbg("dataloader.first_batch",
                         f"x range=[{x_i[...,0].min():.2f},"
                         f"{x_i[...,0].max():.2f}] "
                         f"μ={x_mu:.3f} σ={x_sig:.3f}")
                    if len(self._window_stats) > 2:
                        prev_mu = self._window_stats[-2][0]
                        drift = abs(x_mu - prev_mu)
                        if drift > 1.0:
                            _dbg("dataloader.DRIFT_ALERT",
                                 f"Δμ={drift:.3f}")
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
