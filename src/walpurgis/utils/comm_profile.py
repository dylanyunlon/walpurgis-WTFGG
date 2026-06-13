"""
图通信性能分析器 — 从Neuron_SP/deepspeed/compile/profilers/comm_profile.py鲁迅拿法
改写点 (~20%):
  1. 去除DeepSpeed dist/accelerator依赖, 改用torch原生API
  2. 从all_gather profiling改为graph message passing profiling
  3. 新增per_layer_profile: 逐层图卷积通信耗时记录
  4. 新增memory_timeline: 记录训练过程中显存使用轨迹
  5. 全链路_dbg() + dump_struct_state()断点调试
  6. 集成walpurgis PerfTimer
鲁迅: 拿来的东西, 不是给自己充门面, 要能运用才好。
"""
import os
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np

from .. import _dbg, _is_debug, dump_struct_state, PerfTimer

_MODULE = "comm_profile"


class GraphCommProfiler:
    """图通信性能分析器

    记录图卷积各阶段的通信量和耗时:
      - neighbor_sample: 邻域采样
      - feature_gather: 特征收集
      - message_pass: 消息传递
      - gradient_sync: 梯度同步
    """

    def __init__(self):
        self._records: Dict[str, List[dict]] = defaultdict(list)
        self._active_timers: Dict[str, float] = {}
        self._perf = PerfTimer()
        self._memory_timeline: List[dict] = []
        self._step = 0

    def start(self, op_name: str):
        """开始计时一个通信操作"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._active_timers[op_name] = time.perf_counter()
        self._perf.start(op_name)

    def stop(self, op_name: str, bytes_transferred: int = 0):
        """结束计时, 记录结果"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._perf.stop(op_name)

        if op_name not in self._active_timers:
            return

        duration = time.perf_counter() - self._active_timers.pop(op_name)

        record = {
            "step": self._step,
            "duration_ms": duration * 1000,
            "bytes": bytes_transferred,
            "bandwidth_gbps": (bytes_transferred / duration / 1e9
                               if duration > 0 else 0.0),
        }
        self._records[op_name].append(record)

        _dbg(f"{_MODULE}.{op_name}",
             f"step={self._step} {duration*1000:.2f}ms "
             f"{bytes_transferred/(1024**2):.1f}MB "
             f"{record['bandwidth_gbps']:.1f}GB/s", _MODULE)

    def tick(self):
        """推进到下一步"""
        self._step += 1

    def snapshot_memory(self, label: str = ""):
        """记录当前显存使用情况 (改写: 新增, 用于显存时间线)"""
        if not torch.cuda.is_available():
            return

        mem_info = {
            "step": self._step,
            "label": label,
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }

        try:
            free, total = torch.cuda.mem_get_info()
            mem_info["free_mb"] = free / (1024**2)
            mem_info["total_mb"] = total / (1024**2)
            mem_info["pressure"] = 1.0 - (free / total)
        except Exception:
            pass

        self._memory_timeline.append(mem_info)

        _dbg(f"{_MODULE}.memory.{label}",
             f"alloc={mem_info['allocated_mb']:.0f}MB "
             f"reserved={mem_info['reserved_mb']:.0f}MB",
             _MODULE)

    def get_bandwidth(self, op_name: str, size: int,
                      duration: float) -> Tuple[float, float]:
        """计算吞吐率和bus带宽 (改写: 适配图通信模式)

        图通信不同于all_gather/all_reduce:
          - scatter: 每个GPU发送N/P个节点的特征给邻居
          - gather: 每个GPU接收来自邻域的聚合消息
          - 带宽模型: 类似all_to_all但稀疏
        """
        if duration <= 0:
            return 0.0, 0.0

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if op_name in ("message_pass", "feature_gather"):
            # 图消息传递: 稀疏通信, 有效带宽约为all_to_all的50-80%
            tput = size / duration
            busbw = tput * (num_gpus - 1) / num_gpus * 0.7  # 稀疏系数
        elif op_name == "gradient_sync":
            # all_reduce类型
            tput = size * 2 / duration
            busbw = (size / duration) * (2 * (num_gpus - 1) / num_gpus)
        else:
            tput = size / duration
            busbw = tput

        return tput, busbw

    def report(self, top_k: int = 10):
        """打印性能报告"""
        if not _is_debug() and not self._records:
            return

        print(f"\n{'='*60}")
        print(f"[GRAPH-COMM-PROFILE] Step {self._step}")
        print(f"{'='*60}")

        for op_name, records in self._records.items():
            if not records:
                continue
            durations = [r["duration_ms"] for r in records]
            bandwidths = [r["bandwidth_gbps"] for r in records]
            total_bytes = sum(r["bytes"] for r in records)

            print(f"\n  {op_name}:")
            print(f"    calls:     {len(records)}")
            print(f"    avg_ms:    {np.mean(durations):.2f}")
            print(f"    p50_ms:    {np.median(durations):.2f}")
            print(f"    p99_ms:    {np.percentile(durations, 99):.2f}")
            print(f"    avg_bw:    {np.mean(bandwidths):.1f} GB/s")
            print(f"    total_MB:  {total_bytes/(1024**2):.1f}")

        if self._memory_timeline:
            peaks = max(m["allocated_mb"] for m in self._memory_timeline)
            print(f"\n  Memory peak: {peaks:.0f} MB")

        print(f"{'='*60}\n")
        self._perf.report()

    def per_layer_summary(self) -> Dict[str, Dict]:
        """逐层通信耗时汇总 (改写: 新增)"""
        summary = {}
        for op_name, records in self._records.items():
            if records:
                durations = [r["duration_ms"] for r in records]
                summary[op_name] = {
                    "count": len(records),
                    "total_ms": sum(durations),
                    "avg_ms": np.mean(durations),
                    "max_ms": max(durations),
                }
        return summary

    def reset(self):
        self._records.clear()
        self._active_timers.clear()
        self._memory_timeline.clear()
        self._step = 0


# ═══ 全局单例 ═══
_PROFILER: Optional[GraphCommProfiler] = None


def get_profiler() -> GraphCommProfiler:
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = GraphCommProfiler()
    return _PROFILER


# ═══ 便捷装饰器: 自动profile任意函数 ═══
def profile_comm(op_name: str):
    """装饰器: 自动记录函数的通信耗时

    用法:
        @profile_comm("graph_conv_L0")
        def run_graph_conv(x, adj):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profiler.start(op_name)
            result = func(*args, **kwargs)
            # 估算传输量: 如果返回tensor, 用其字节数
            bytes_transferred = 0
            if isinstance(result, torch.Tensor):
                bytes_transferred = result.nelement() * result.element_size()
            profiler.stop(op_name, bytes_transferred)
            return result
        return wrapper
    return decorator


# ═══ 自检 ═══
def self_check():
    p = GraphCommProfiler()
    p.start("test_op")
    time.sleep(0.001)
    p.stop("test_op", bytes_transferred=1024)
    p.tick()
    assert len(p._records["test_op"]) == 1
    assert p._records["test_op"][0]["bytes"] == 1024
    summary = p.per_layer_summary()
    assert "test_op" in summary
    _dbg(f"{_MODULE}.self_check", "PASSED", _MODULE)
    return True
