import numpy as np
from walpurgis_walking import _dbg

_TAG = "loader"


class DataLoader(object):
    """内存友好的DataLoader.

    改动 vs upstream:
      1) wrap-ring padding (而非重复末尾sample)
      2) Knuth shuffle (原地, 只shuffle索引不拷贝数据)
      3) 3-tuple yield (x, y, weight)
      4) lazy slice — 不做 .copy(), 直接返回 view

    vs 原walpurgis v10:
      - 去掉了 prefetch 里的 .copy() — 在内存受限环境下那是致命的
      - shuffle 改为只 shuffle 索引数组, 不移动底层数据
      - padding 用索引而非拷贝数据
    """

    def __init__(self, xs, ys, batch_size,
                 pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        self._raw_size = len(xs)

        # 不拷贝数据, 只记录原始引用
        self.xs = xs
        self.ys = ys

        # wrap-ring padding: 只计算需要多少padding, 用索引表达
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        else:
            num_padding = 0
        self._num_padding = num_padding

        self.size = self._raw_size + num_padding
        self.num_batch = int(self.size // self.batch_size)

        # 索引数组: 初始为 [0, 1, ..., raw_size-1, 0, 1, ...] (wrap部分)
        self._indices = np.arange(self.size, dtype=np.int32)
        if num_padding > 0:
            self._indices[self._raw_size:] = np.arange(num_padding) % self._raw_size

        # 样本权重
        self.sample_weights = np.ones(self.size, dtype=np.float32)

        if shuffle:
            self.shuffle()

        mem_mb = (xs.nbytes + ys.nbytes) / 1e6
        print(f"[v10 DataLoader] size={self.size}, "
              f"num_batch={self.num_batch}, bs={batch_size}, "
              f"pad={num_padding}, mem={mem_mb:.0f}MB")

    def shuffle(self):
        # Knuth shuffle 只操作索引, 不移动底层数据
        n = self.size
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            self._indices[i], self._indices[j] = self._indices[j], self._indices[i]

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _gen():
            for batch_i in range(self.num_batch):
                start = self.batch_size * batch_i
                end = self.batch_size * (batch_i + 1)
                idx = self._indices[start:end]

                # 不 copy, 直接 fancy indexing (返回的是新数组但不重复整个dataset)
                yield (self.xs[idx], self.ys[idx], self.sample_weights[idx])
                self.current_ind += 1

        return _gen()
