import numpy as np
from walpurgis_ported_v10 import _dbg

_TAG = "loader"


class DataLoader(object):
    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                # 改动1: 环形 wrap padding — upstream 用最后一个 sample 重复
                # 环形: 从头部取 sample 拼到尾部, 比重复末尾更多样
                wrap_idx = np.arange(num_padding) % len(xs)
                x_padding = xs[wrap_idx]
                y_padding = ys[wrap_idx]
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

        # 改动3: 样本权重 — 均匀初始化, 可被外部修改
        self.sample_weights = np.ones(self.size, dtype=np.float32)

        if shuffle:
            self.shuffle()

        print(f"[v10 DataLoader] size={self.size}, "
              f"num_batch={self.num_batch}, bs={batch_size}, "
              f"pad_mode=wrap_ring")

    def shuffle(self):
        # 改动2: Fisher-Yates (Knuth) 原地 shuffle
        # upstream: np.random.permutation(self.size) 新分配index数组
        # Knuth shuffle: O(n) 原地, 无额外分配
        n = self.size
        indices = np.arange(n)
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        self.xs = self.xs[indices]
        self.ys = self.ys[indices]
        self.sample_weights = self.sample_weights[indices]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        # 改动4: prefetch — 预取下一 batch
        def _prefetch_gen():
            # 先加载第一个 batch
            next_x = None
            next_y = None
            next_w = None

            for batch_i in range(self.num_batch):
                start = self.batch_size * batch_i
                end = min(self.size, self.batch_size * (batch_i + 1))

                if next_x is not None:
                    # 返回上一轮预取的
                    cur_x, cur_y, cur_w = next_x, next_y, next_w
                else:
                    cur_x = self.xs[start:end]
                    cur_y = self.ys[start:end]
                    cur_w = self.sample_weights[start:end]

                # 预取下一 batch (如果有的话)
                next_start = self.batch_size * (batch_i + 1)
                next_end = min(self.size, self.batch_size * (batch_i + 2))
                if batch_i + 1 < self.num_batch:
                    next_x = self.xs[next_start:next_end].copy()
                    next_y = self.ys[next_start:next_end].copy()
                    next_w = self.sample_weights[next_start:next_end].copy()
                else:
                    next_x = None

                # 改动3: yield 3-tuple — upstream 只 yield (x, y)
                yield (cur_x, cur_y, cur_w)
                self.current_ind += 1

        return _prefetch_gen()
