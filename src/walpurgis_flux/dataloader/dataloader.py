"""Flux DataLoader: 滑动窗口重叠采样 + 渐进式shuffle.
与upstream(np.random.permutation)和vortex(stratified shuffle + mirror-padding)不同,
Flux使用渐进式shuffle: 前N轮shuffle粒度较粗(保留更多时序局部性),
后续轮逐渐细化到完全随机. 这模拟了流式推理中数据到达的渐进性.
Padding使用滑动窗口重叠: 最后几个sample与前面重叠而非简单复制."""
import numpy as np
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class DataLoader:
    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True,
                 shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        self._shuffle_granularity = 4  # 初始粗粒度
        self._shuffle_epoch = 0
        if pad_with_last_sample:
            num_padding = (
                batch_size - (len(xs) % batch_size)
            ) % batch_size
            if num_padding > 0:
                # Flux: 滑动窗口重叠padding
                # 取最后几个样本做overlap而非最后一个重复
                overlap_start = max(
                    0, len(xs) - num_padding)
                pad_xs = xs[overlap_start:
                            overlap_start + num_padding]
                pad_ys = ys[overlap_start:
                            overlap_start + num_padding]
                xs = np.concatenate(
                    [xs, pad_xs], axis=0)
                ys = np.concatenate(
                    [ys, pad_ys], axis=0)
                if _FX_DBG:
                    print(f"[FX:dataloader] overlap "
                          f"padding: {num_padding} "
                          f"samples from idx "
                          f"{overlap_start}",
                          file=sys.stderr)
        self.size = len(xs)
        self.num_batch = int(
            self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()
        if _FX_DBG:
            print(f"[FX:dataloader] size={self.size} "
                  f"batches={self.num_batch} "
                  f"batch_size={batch_size}",
                  file=sys.stderr)

    def shuffle(self):
        """Flux: 渐进式shuffle — 前几轮保留更多时序局部性,
        后续轮逐渐趋向完全随机.
        粒度 = max(1, init_granularity - epoch)"""
        n = self.size
        granularity = max(
            1, self._shuffle_granularity -
            self._shuffle_epoch)
        if granularity <= 1:
            # 完全随机
            permutation = np.random.permutation(n)
        else:
            # 块内shuffle: 把序列分成大小为granularity的块
            n_blocks = (n + granularity - 1) // granularity
            blocks = []
            for b in range(n_blocks):
                start = b * granularity
                end = min(start + granularity, n)
                block = np.arange(start, end)
                np.random.shuffle(block)
                blocks.append(block)
            # 随机排列块的顺序
            block_order = np.random.permutation(n_blocks)
            permutation = np.concatenate(
                [blocks[i] for i in block_order])
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]
        self._shuffle_epoch += 1
        if _FX_DBG:
            print(f"[FX:dataloader.shuffle] "
                  f"granularity={granularity} "
                  f"epoch={self._shuffle_epoch}",
                  file=sys.stderr)

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                s = self.batch_size * self.current_ind
                e = min(self.size,
                        self.batch_size *
                        (self.current_ind + 1))
                yield (self.xs[s:e, ...],
                       self.ys[s:e, ...])
                self.current_ind += 1
        return _wrapper()
