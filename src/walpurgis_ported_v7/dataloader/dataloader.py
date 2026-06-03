import numpy as np
import sys

_DBG_DL = ("--dbg-dl" in sys.argv)


class DataLoader(object):
    """算法改动:
    1. shuffle 策略改为 block-shuffle: 先把样本按连续 block 分组,
       然后 shuffle blocks 而非单条样本。这保留了短期时序连续性
       (对 seq2seq 任务有益), 同时还是打乱了不同时段。
    2. get_iterator 加 prefetch: 预先把下一个 batch 转为 contiguous array,
       减少训练 loop 里的内存拷贝延迟。
    """

    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        self._shuffle_block_size = 8  # 每个 block 内部保持顺序

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

        if _DBG_DL:
            print(f"[DBG-DL] DataLoader init  samples={self.size}  "
                  f"batches={self.num_batch}  batch_size={batch_size}  "
                  f"x_shape={xs.shape}  y_shape={ys.shape}")

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """算法改动: block-shuffle —
        将数据按 block_size 分成连续块, shuffle 块的顺序"""
        n = self.size
        bs = self._shuffle_block_size
        num_blocks = n // bs
        remainder = n % bs

        block_indices = np.arange(num_blocks)
        np.random.shuffle(block_indices)

        new_order = []
        for bi in block_indices:
            new_order.extend(range(bi * bs, (bi + 1) * bs))
        # 剩余不足一个 block 的放末尾
        if remainder > 0:
            new_order.extend(range(num_blocks * bs, n))

        new_order = np.array(new_order)
        self.xs = self.xs[new_order]
        self.ys = self.ys[new_order]

        if _DBG_DL:
            print(f"[DBG-DL] block-shuffle  blocks={num_blocks}  "
                  f"block_size={bs}  remainder={remainder}")

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                # 算法改动: 确保 contiguous 减少下游 to_tensor 开销
                x_i = np.ascontiguousarray(self.xs[start_ind:end_ind])
                y_i = np.ascontiguousarray(self.ys[start_ind:end_ind])
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
