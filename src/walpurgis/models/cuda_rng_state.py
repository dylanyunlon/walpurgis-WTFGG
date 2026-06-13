"""
cuda_rng_state.py — 2ef236753 迁移: Support latest PyTorch RNG state API. (#8)

migrate 2ef236753: Support latest PyTorch RNG state API. (#8)

上游变化 (2ef236753 — mpu/random.py):
  _set_cuda_rng_state(new_state, device=-1) 函数内部实现重构：

  before (旧版 PyTorch):
    def cb():
        with device_ctx_manager(device):
            _C._cuda_setRNGState(new_state)

  after (新旧 PyTorch 双路径):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)

核心设计:
  - 旧版 PyTorch: 通过 torch._C._cuda_setRNGState (C++ 内部 API) 直接设置 RNG 状态
  - 新版 PyTorch: _cuda_setRNGState 被移除, 改用 torch.cuda.default_generators[idx].set_state()
  - device 参数归一化: -1 → torch.device('cuda'), str → torch.device(str), int → torch.device('cuda', int)
  - _lazy_call 机制保持不变: callback 推迟到 CUDA 初始化后执行

Walpurgis 改写20%(鲁迅拿法):
  上游改动的本质是一次「新旧交替的双轨迁移」——旧 API 是一道厚重的铁门,
  新 API 是另辟的侧门。鲁迅写《药》里人血馒头,是说旧物件里藏着时代的断层;
  _C._cuda_setRNGState 的消失同理: 一个内部 C++ 符号悄然退场,
  无声无息,只留下 hasattr 的探针和 default_generators 的新路。
  Walpurgis 将这次双轨迁移结构化为:

  1. RngBackend 枚举: 明示 LEGACY / MODERN 两条路径, 替代上游的 hasattr 内联判断
  2. DeviceNormalizer: 封装 device=-1/str/int → torch.device 的归一化逻辑,
     替代上游 if/elif 散落在 else 分支里的三段式转换 (Python 无复用)
  3. CudaRngStateSetter: 封装整个 _set_cuda_rng_state 逻辑, 持有 backend 探测结果,
     替代上游裸函数 + 闭包 + _lazy_call 的平铺结构
  4. LazyCudaCallable: 封装 _lazy_call 语义, 使 callback 推迟执行可显式表达和调试
  5. 断点调试: WALPURGIS_DEBUG=1 开启全链路 print
     - backend 探测: 打印 _C._cuda_setRNGState 可用性
     - device 归一化: 打印 input device → torch.device 转换结果
     - cb() 执行路径: 打印走 LEGACY 还是 MODERN 路径 + device index
     - _lazy_call 调度: 打印 callback 注册时机

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
from enum import Enum
from typing import Any, Callable, Optional, Union

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg(tag: str, msg: str) -> None:
    """断点调试: cuda_rng_state 专用 print"""
    if _DBG:
        print(f"[DEBUG 2ef236753 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── RngBackend: 明示旧版/新版 PyTorch RNG API 路径 ──────────────────────────
# 上游 (2ef236753 mpu/random.py):
#   if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
#       # older PyTorch → _C._cuda_setRNGState
#   else:
#       # newer PyTorch → default_generator.set_state()
#
# 改写: 枚举显式建模两条路径, 替代上游内联 hasattr 判断
class RngBackend(Enum):
    LEGACY = "legacy"   # older PyTorch: torch._C._cuda_setRNGState
    MODERN = "modern"   # newer PyTorch: torch.cuda.default_generators[idx].set_state()


# ─── DeviceNormalizer: 封装 device 参数归一化逻辑 ────────────────────────────
# 上游 (2ef236753 mpu/random.py, newer PyTorch 分支):
#   if device == -1:
#       device = torch.device('cuda')
#   elif isinstance(device, str):
#       device = torch.device(device)
#   elif isinstance(device, int):
#       device = torch.device('cuda', device)
#
# 改写: 封装为独立类, 逻辑复用 + 调试输出 (上游内联在 else 分支, 无复用)
class DeviceNormalizer:
    """
    归一化 _set_cuda_rng_state 的 device 参数为 torch.device。

    上游 (2ef236753): if/elif 散落在 newer PyTorch else 分支内, 不可复用。
    改写: 独立封装, 三种输入类型均有调试输出, 便于排查 device 归一化问题。

    输入类型:
      -1    → torch.device('cuda')           (当前 CUDA 设备)
      str   → torch.device(device)           (e.g. 'cuda:0')
      int   → torch.device('cuda', device)   (e.g. cuda:2)
      torch.device → 原样返回 (改写新增: 防御性处理)
    """

    def normalize(self, device: Any, torch_module: Any) -> Any:
        """
        对应上游 2ef236753 newer PyTorch 分支的 device 归一化三段式。

        断点调试: 打印 input_device + normalized_device + 走哪条路径
        """
        _dbg(
            "DeviceNormalizer.normalize",
            f"input device={device!r} type={type(device).__name__}"
        )

        # 已是 torch.device: 改写新增防御路径 (上游无此处理)
        if hasattr(torch_module, 'device') and isinstance(device, torch_module.device):
            _dbg("DeviceNormalizer.normalize", f"already torch.device → {device}")
            return device

        # 上游 (2ef236753): if device == -1 → torch.device('cuda')
        if device == -1:
            result = torch_module.device('cuda')
            _dbg(
                "DeviceNormalizer.normalize",
                f"device==-1 → torch.device('cuda') idx=None (current device)"
            )
            return result

        # 上游 (2ef236753): elif isinstance(device, str) → torch.device(device)
        if isinstance(device, str):
            result = torch_module.device(device)
            _dbg(
                "DeviceNormalizer.normalize",
                f"str device={device!r} → {result}"
            )
            return result

        # 上游 (2ef236753): elif isinstance(device, int) → torch.device('cuda', device)
        if isinstance(device, int):
            result = torch_module.device('cuda', device)
            _dbg(
                "DeviceNormalizer.normalize",
                f"int device={device} → torch.device('cuda', {device})"
            )
            return result

        # 改写: 兜底路径 (上游无; 防止未知类型静默失败)
        _dbg(
            "DeviceNormalizer.normalize",
            f"WARN: unknown device type {type(device).__name__!r}, "
            f"attempting torch.device({device!r})"
        )
        return torch_module.device(device)


# ─── LazyCudaCallable: 封装 _lazy_call 调度语义 ─────────────────────────────
# 上游 (2ef236753 mpu/random.py):
#   _lazy_call(cb)
#
# _lazy_call 是 Megatron-LM 内部机制: CUDA 未初始化时将 cb 推入队列,
# 初始化后立即执行。Walpurgis 无 _lazy_call, 改写为封装类显式表达此语义。
#
# 改写: LazyCudaCallable 持有 callback, 提供 schedule(lazy_call_fn) 接口,
#       兼容注入真实 _lazy_call 或在单机环境直接执行
class LazyCudaCallable:
    """
    封装 _lazy_call 的调度语义。

    上游 (2ef236753): _lazy_call(cb) — 一行裸调用, 无调试输出。
    改写: 持有 callback 对象, schedule() 方法接受 lazy_call_fn 并调用,
          支持 WALPURGIS_DEBUG=1 打印调度时机。
    若 lazy_call_fn 为 None, 直接执行 callback (Walpurgis 单机降级)。
    """

    def __init__(self, cb: Callable, tag: str = ""):
        self._cb = cb
        self._tag = tag
        _dbg(
            "LazyCudaCallable.__init__",
            f"callback registered tag={tag!r}"
        )

    def schedule(self, lazy_call_fn: Optional[Callable] = None) -> None:
        """
        对应上游 _lazy_call(cb)。

        Args:
            lazy_call_fn: 注入 _lazy_call 函数 (None → 直接执行, 单机降级)
        """
        if lazy_call_fn is not None:
            _dbg(
                "LazyCudaCallable.schedule",
                f"tag={self._tag!r} → delegating to _lazy_call"
            )
            lazy_call_fn(self._cb)
        else:
            # 改写: 单机 / 无 _lazy_call 环境直接执行
            _dbg(
                "LazyCudaCallable.schedule",
                f"tag={self._tag!r} → no lazy_call_fn, executing cb() directly"
            )
            self._cb()


# ─── CudaRngStateSetter: 封装 _set_cuda_rng_state 完整逻辑 ──────────────────
# 上游 (2ef236753 mpu/random.py):
#   def _set_cuda_rng_state(new_state, device=-1):
#       if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
#           def cb():
#               with device_ctx_manager(device):
#                   _C._cuda_setRNGState(new_state)
#       else:
#           if device == -1: ...  # 归一化
#           def cb():
#               idx = device.index
#               if idx is None:
#                   idx = torch.cuda.current_device()
#               default_generator = torch.cuda.default_generators[idx]
#               default_generator.set_state(new_state)
#       _lazy_call(cb)
#
# 改写: 封装为 CudaRngStateSetter 类, backend 探测在构造时完成 (上游每次调用都探测)
class CudaRngStateSetter:
    """
    对应 2ef236753 _set_cuda_rng_state(new_state, device=-1) 完整逻辑。

    上游: 裸函数, 每次调用都内联 hasattr 探测 + closure + _lazy_call。
    改写: 类封装, 构造时探测 backend (缓存结果避免重复 hasattr),
          set_state() 对应上游函数调用点, schedule() 对应 _lazy_call(cb)。

    两条路径:
      LEGACY (older PyTorch):
        torch._C._cuda_setRNGState 存在且可调用 →
        with device_ctx_manager(device): _C._cuda_setRNGState(new_state)

      MODERN (newer PyTorch):
        torch._C._cuda_setRNGState 不可用 →
        device 归一化 → default_generators[idx].set_state(new_state)

    断点调试: WALPURGIS_DEBUG=1 打印 backend 探测 + device 归一化 + cb 执行路径
    """

    def __init__(self, torch_module: Any = None):
        """
        构造时探测 backend (对应上游 hasattr(_C, '_cuda_setRNGState') 检查)。

        Args:
            torch_module: 注入 torch (None → 运行时 import)
        """
        self._torch = torch_module
        self._device_normalizer = DeviceNormalizer()
        self._backend: Optional[RngBackend] = None  # 延迟探测

        _dbg(
            "CudaRngStateSetter.__init__",
            "CudaRngStateSetter created, backend probe deferred to first use"
        )

    def _get_torch(self) -> Any:
        """延迟 import torch"""
        if self._torch is None:
            try:
                import torch as _t
                self._torch = _t
            except ImportError:
                raise RuntimeError(
                    "[2ef236753 CudaRngStateSetter] torch not available."
                )
        return self._torch

    def _probe_backend(self) -> RngBackend:
        """
        对应上游:
            if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
                # older PyTorch
            else:
                # newer PyTorch

        改写: 封装为独立方法, 结果缓存 (上游每次调用都重新探测)
        断点调试: 打印 _C._cuda_setRNGState 可用性 + 最终 backend
        """
        if self._backend is not None:
            _dbg("CudaRngStateSetter._probe_backend", f"cached backend={self._backend.value}")
            return self._backend

        torch = self._get_torch()
        _C = torch._C

        # 上游 (2ef236753): hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState)
        has_legacy = hasattr(_C, '_cuda_setRNGState') and callable(
            getattr(_C, '_cuda_setRNGState', None)
        )

        _dbg(
            "CudaRngStateSetter._probe_backend",
            f"torch._C._cuda_setRNGState exists={has_legacy}"
        )

        self._backend = RngBackend.LEGACY if has_legacy else RngBackend.MODERN

        _dbg(
            "CudaRngStateSetter._probe_backend",
            f"→ backend={self._backend.value} "
            f"({'_C._cuda_setRNGState' if has_legacy else 'default_generators.set_state'})"
        )
        return self._backend

    def _build_legacy_cb(
        self,
        new_state: Any,
        device: Any,
        device_ctx_manager: Callable,
    ) -> Callable:
        """
        构造 LEGACY 路径 callback。

        对应上游 (2ef236753 older PyTorch):
            def cb():
                with device_ctx_manager(device):
                    _C._cuda_setRNGState(new_state)

        改写: 独立方法返回 cb, 加调试输出; 上游是内联 def cb()
        """
        torch = self._get_torch()
        _C = torch._C

        _dbg(
            "CudaRngStateSetter._build_legacy_cb",
            f"building LEGACY cb: device={device!r} using _C._cuda_setRNGState"
        )

        def cb():
            _dbg(
                "CudaRngStateSetter._legacy_cb",
                f"executing: device_ctx_manager({device!r}) + _C._cuda_setRNGState"
            )
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

        return cb

    def _build_modern_cb(
        self,
        new_state: Any,
        device: Any,
    ) -> Callable:
        """
        构造 MODERN 路径 callback。

        对应上游 (2ef236753 newer PyTorch):
            def cb():
                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                default_generator = torch.cuda.default_generators[idx]
                default_generator.set_state(new_state)

        注意: device 在此之前已经由 DeviceNormalizer 归一化为 torch.device

        改写: 独立方法返回 cb, 加调试输出 + idx 解析日志
        """
        torch = self._get_torch()

        # 上游 (2ef236753): device 归一化在 _set_cuda_rng_state 函数体内完成
        # 改写: 归一化提前在 set_state() 中完成, 此处 device 已是 torch.device
        _dbg(
            "CudaRngStateSetter._build_modern_cb",
            f"building MODERN cb: normalized_device={device!r}"
        )

        def cb():
            # 上游 (2ef236753):
            #   idx = device.index
            #   if idx is None:
            #       idx = torch.cuda.current_device()
            idx = device.index
            _dbg(
                "CudaRngStateSetter._modern_cb",
                f"device.index={idx!r}"
            )
            if idx is None:
                idx = torch.cuda.current_device()
                _dbg(
                    "CudaRngStateSetter._modern_cb",
                    f"idx was None → torch.cuda.current_device()={idx}"
                )

            # 上游 (2ef236753):
            #   default_generator = torch.cuda.default_generators[idx]
            #   default_generator.set_state(new_state)
            default_generator = torch.cuda.default_generators[idx]
            _dbg(
                "CudaRngStateSetter._modern_cb",
                f"default_generators[{idx}].set_state(new_state)"
            )
            default_generator.set_state(new_state)

        return cb

    def set_state(
        self,
        new_state: Any,
        device: Any = -1,
        device_ctx_manager: Optional[Callable] = None,
        lazy_call_fn: Optional[Callable] = None,
    ) -> None:
        """
        对应上游 2ef236753 _set_cuda_rng_state(new_state, device=-1) 完整调用。

        Args:
            new_state: RNG 状态 tensor (对应上游 new_state)
            device: CUDA device, 默认 -1 (对应上游 device=-1)
            device_ctx_manager: 设备上下文管理器 (LEGACY 路径需要; None → 跳过)
            lazy_call_fn: 注入 _lazy_call (None → 直接执行 cb)

        断点调试: 打印 backend + device + _lazy_call 调度
        """
        _dbg(
            "CudaRngStateSetter.set_state",
            f"new_state={type(new_state).__name__} device={device!r}"
        )

        backend = self._probe_backend()

        if backend == RngBackend.LEGACY:
            # 上游 (2ef236753 older PyTorch): _lazy_call 接受带 device_ctx_manager 的 cb
            # LEGACY 路径 device 不做归一化 (上游如此)
            _dbg(
                "CudaRngStateSetter.set_state",
                f"LEGACY path: device={device!r} (no normalization, uses device_ctx_manager)"
            )
            if device_ctx_manager is None:
                # 改写: 提供 noop ctx manager 作为降级 (上游必须有 device_ctx_manager)
                from contextlib import contextmanager
                @contextmanager
                def _noop_ctx(d):
                    _dbg(
                        "CudaRngStateSetter._noop_ctx",
                        f"noop device_ctx_manager for device={d!r} "
                        "(no device_ctx_manager injected)"
                    )
                    yield
                device_ctx_manager = _noop_ctx

            cb = self._build_legacy_cb(new_state, device, device_ctx_manager)

        else:
            # 上游 (2ef236753 newer PyTorch): 先归一化 device, 再构造 cb
            _dbg(
                "CudaRngStateSetter.set_state",
                f"MODERN path: normalizing device={device!r}"
            )
            torch = self._get_torch()
            normalized_device = self._device_normalizer.normalize(device, torch)
            cb = self._build_modern_cb(new_state, normalized_device)

        # 上游 (2ef236753): _lazy_call(cb)
        lazy_callable = LazyCudaCallable(cb, tag=f"2ef236753:{backend.value}")
        lazy_callable.schedule(lazy_call_fn)

        _dbg(
            "CudaRngStateSetter.set_state",
            f"→ cb scheduled via {'_lazy_call' if lazy_call_fn else 'direct exec'}"
        )


# ─── 模块级默认 setter (对应上游函数级调用点) ────────────────────────────────
# 上游: _set_cuda_rng_state(new_state, device=-1) 是模块级裸函数
# 改写: 提供模块级单例 + 便捷函数, 保持与上游调用点兼容
_DEFAULT_SETTER = CudaRngStateSetter()


def set_cuda_rng_state(
    new_state: Any,
    device: Any = -1,
    device_ctx_manager: Optional[Callable] = None,
    lazy_call_fn: Optional[Callable] = None,
) -> None:
    """
    对应 2ef236753 mpu/random.py _set_cuda_rng_state(new_state, device=-1)。

    改写: 封装为便捷函数, 委托给 _DEFAULT_SETTER.set_state()。
    WALPURGIS_DEBUG=1 开启全链路断点输出。

    Args:
        new_state: RNG 状态 tensor
        device: CUDA device (-1=当前, str='cuda:0', int=index, torch.device)
        device_ctx_manager: 上下文管理器 (LEGACY 路径使用)
        lazy_call_fn: 注入 _lazy_call (None → 直接执行)

    Usage:
        # 对应上游: _set_cuda_rng_state(state)
        set_cuda_rng_state(state)

        # 对应上游: _set_cuda_rng_state(state, device=2)
        set_cuda_rng_state(state, device=2)

        # 注入 _lazy_call (Megatron-LM 分布式环境)
        set_cuda_rng_state(state, lazy_call_fn=_lazy_call)
    """
    _dbg(
        "set_cuda_rng_state",
        f"new_state={type(new_state).__name__} device={device!r} "
        f"has_ctx_mgr={device_ctx_manager is not None} "
        f"has_lazy_call={lazy_call_fn is not None}"
    )
    _DEFAULT_SETTER.set_state(
        new_state=new_state,
        device=device,
        device_ctx_manager=device_ctx_manager,
        lazy_call_fn=lazy_call_fn,
    )
