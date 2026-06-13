"""
双缓冲图通信 — 从Neuron_SP/deepspeed/compile/custom_ops/double_buffer_a2a.py鲁迅拿法
改写点 (~20%):
  1. 去除all_to_all序列并行依赖, 改为图卷积消息传递的双缓冲
  2. 新增memory_guard: 分配前检查显存水位, 超过阈值自动降级到单缓冲
  3. 新增profile_swap: 记录每次swap的耗时, 用于调优prefetch策略
  4. 全链路_dbg() + dump_struct_state()断点调试
  5. 新增GraphMessageBuffer子类: 适配[N, K, D]图消息张量
鲁迅: 拿来主义——可以拿的就拿, 但要经过挑选。
"""
import threading
import time
from typing import Optional, Dict

import torch

from .. import _dbg, _is_debug, dump_struct_state

_MODULE = "double_buffer"


class DoubleBuffer:
    """双缓冲器 — 用于图卷积消息传递的流水线隐藏

    改写 vs Neuron_SP:
      - 去除sp_dp_registry.track_buffer_event依赖
      - 新增memory_guard (显存安全阈值检查)
      - 新增profile_swap (swap耗时追踪)
    """

    def __init__(self, dtype=torch.float32, device=None,
                 memory_guard_ratio=0.3):
        self._dtype = dtype
        self._device = device or (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available() else torch.device("cpu"))
        self.selector = 0
        self._data = [None, None]
        self._valid = [False, False]
        self._allocated = False
        self._lock = threading.Lock()
        self._swap_count = 0
        self._memory_guard_ratio = memory_guard_ratio
        # 改写: swap耗时追踪
        self._swap_times = []
        self._alloc_bytes = 0

    def allocate(self, shape, dtype=None):
        """分配双缓冲 (改写: 新增memory_guard)"""
        with self._lock:
            if dtype is not None and dtype != self._dtype:
                self._dtype = dtype
                if self._allocated:
                    self._free_unlocked()
            if self._allocated:
                if (self._data[0] is not None
                        and self._data[0].shape == shape):
                    return
                self._free_unlocked()

            numel = 1
            for s in shape:
                numel *= s
            elem_bytes = torch.tensor([], dtype=self._dtype).element_size()
            buf_bytes = numel * elem_bytes * 2  # 双缓冲=2倍

            # 改写: memory_guard — 检查显存安全
            if torch.cuda.is_available() and self._device.type == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info(self._device)
                pressure = 1.0 - (free_mem / total_mem)
                if buf_bytes > free_mem * self._memory_guard_ratio:
                    _dbg(f"{_MODULE}.memory_guard",
                         f"WARN: buffer {buf_bytes/(1024**2):.0f}MB > "
                         f"{self._memory_guard_ratio*100:.0f}% of free "
                         f"{free_mem/(1024**2):.0f}MB. "
                         f"Falling back to single buffer.", _MODULE)
                    # 降级: 只分配一个buffer
                    self._data[0] = torch.empty(
                        shape, dtype=self._dtype, device=self._device)
                    self._data[1] = self._data[0]  # 指向同一个
                    self._allocated = True
                    self._alloc_bytes = buf_bytes // 2
                    return

            for i in range(2):
                self._data[i] = torch.empty(
                    shape, dtype=self._dtype, device=self._device)
            self._allocated = True
            self._alloc_bytes = buf_bytes

            _dbg(f"{_MODULE}.alloc",
                 f"shape={list(shape)} dtype={self._dtype} "
                 f"total={buf_bytes/(1024**2):.1f}MB", _MODULE)

    def current(self) -> Optional[torch.Tensor]:
        return self._data[self.selector]

    def alternate(self) -> Optional[torch.Tensor]:
        return self._data[self.selector ^ 1]

    def swap(self):
        """切换前后buffer (改写: 新增耗时追踪)"""
        t0 = time.perf_counter()
        with self._lock:
            self.selector ^= 1
            self._swap_count += 1
        dt = time.perf_counter() - t0
        self._swap_times.append(dt)
        if len(self._swap_times) > 100:
            self._swap_times = self._swap_times[-50:]

    def swap_count(self) -> int:
        return self._swap_count

    def avg_swap_time_us(self) -> float:
        """平均swap耗时 (改写: 新增诊断接口)"""
        if not self._swap_times:
            return 0.0
        return sum(self._swap_times) / len(self._swap_times) * 1e6

    def mark_valid(self, slot=-1):
        if slot < 0:
            slot = self.selector
        self._valid[slot] = True

    def is_valid(self, slot=-1) -> bool:
        if slot < 0:
            slot = self.selector
        return self._valid[slot]

    def invalidate(self, slot=-1):
        if slot < 0:
            slot = self.selector
        self._valid[slot] = False

    def _free_unlocked(self):
        for i in range(2):
            self._data[i] = None
            self._valid[i] = False
        self._allocated = False
        self.selector = 0
        self._swap_count = 0
        self._alloc_bytes = 0

    def free(self):
        with self._lock:
            self._free_unlocked()
        _dbg(f"{_MODULE}.free", "buffer freed", _MODULE)

    @property
    def allocated(self) -> bool:
        return self._allocated

    def diagnostics(self) -> dict:
        """全量诊断信息 (改写: 新增)"""
        return {
            "allocated": self._allocated,
            "selector": self.selector,
            "swap_count": self._swap_count,
            "alloc_bytes": self._alloc_bytes,
            "avg_swap_us": self.avg_swap_time_us(),
            "valid": list(self._valid),
            "shape": (list(self._data[0].shape)
                      if self._data[0] is not None else None),
        }


class GraphMessageBuffer(DoubleBuffer):
    """图消息传递专用双缓冲 (改写: 新增, Neuron_SP无此类)

    用于图卷积中的邻域聚合:
      - 前buffer存当前层的聚合消息 [N, K, D]
      - 后buffer预取下一层的邻域特征
      - swap()在层间切换, 隐藏数据搬运延迟
    """

    def __init__(self, num_nodes: int, k_hops: int, hidden_dim: int,
                 dtype=torch.float32, device=None):
        super().__init__(dtype=dtype, device=device)
        self._num_nodes = num_nodes
        self._k_hops = k_hops
        self._hidden_dim = hidden_dim
        self._layer_idx = 0

    def allocate_for_graph(self):
        """按图参数分配"""
        shape = (self._num_nodes, self._k_hops, self._hidden_dim)
        self.allocate(shape, self._dtype)
        _dbg(f"{_MODULE}.graph_alloc",
             f"N={self._num_nodes} K={self._k_hops} D={self._hidden_dim}",
             _MODULE)

    def advance_layer(self):
        """推进到下一层图卷积"""
        self._layer_idx += 1
        self.swap()
        self.invalidate()  # 新buffer待填充

    @property
    def layer_idx(self) -> int:
        return self._layer_idx


# ═══ 全局缓冲池 (从Neuron_SP移植, 改写: 新增diagnostics_all) ═══
class BufferPool:
    def __init__(self):
        self._buffers: Dict[str, DoubleBuffer] = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: str, dtype=torch.float32,
                      device=None) -> DoubleBuffer:
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = DoubleBuffer(dtype=dtype, device=device)
                _dbg(f"{_MODULE}.pool.create", f"key={key}", _MODULE)
            return self._buffers[key]

    def swap_all(self):
        with self._lock:
            for buf in self._buffers.values():
                buf.swap()

    def free_all(self):
        with self._lock:
            for buf in self._buffers.values():
                buf.free()
            self._buffers.clear()
        _dbg(f"{_MODULE}.pool.free_all", "all freed", _MODULE)

    def diagnostics_all(self) -> dict:
        """所有buffer的诊断汇总 (改写: 新增)"""
        with self._lock:
            return {k: b.diagnostics() for k, b in self._buffers.items()}

    def __len__(self):
        return len(self._buffers)


_GLOBAL_POOL: Optional[BufferPool] = None


def get_buffer_pool() -> BufferPool:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = BufferPool()
    return _GLOBAL_POOL


# ═══ 自检 ═══
def self_check():
    pool = get_buffer_pool()
    buf = pool.get_or_create("__selftest", dtype=torch.float32)
    buf.allocate((4, 8))
    assert buf.allocated
    assert buf.current() is not None
    buf.swap()
    assert buf.swap_count() == 1
    buf.free()
    assert not buf.allocated
    _dbg(f"{_MODULE}.self_check", "PASSED", _MODULE)
    return True
