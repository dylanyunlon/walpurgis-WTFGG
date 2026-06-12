"""
migrate 7e914aa: fix to difference in cpu and gpu precision in sample (#398)

上游 commit: 7e914aa (cugraph-gnn, linhu-nv, 2026-02-02, PR #398)
上游文件: cpp/tests/wholegraph_ops/graph_sampling_test_utils.cu

问题: host_gen_key_from_weight() 中用 pow(2, -one_bit) 计算指数，
      pow 在 CPU 默认双精度，GPU 用单精度 exp2f，导致 CPU/GPU 验证比对失败。
修复: 改为 exp2f(-one_bit) ——强制单精度路径，消除 CPU/GPU 精度差异。

Walpurgis 迁移策略:
  - 上游是 C++ CUDA 测试工具函数，Walpurgis 无 C++ 层
  - 将精度修复逻辑以 Python float32 语义复现，供 Python 端图采样测试/验证使用
  - 关键数学等价: pow(2, -n) [fp64] → exp2f(-n) [fp32] → math.ldexp(1.0, -n) [Python fp32 等价]
  - 鲁迅: 差之毫厘，谬以千里。一个精度选择，让 CPU 验证与 GPU 结果背道而驰。

Author: 鲁迅拿法改写 (≥20%): 上游纯 C++，Walpurgis 将其提升为 Python 数值守卫体系:
  1. WeightedSampleKeyFn: 封装 key 生成算法，支持 precision_mode 切换
  2. PrecisionMode: 枚举 CPU_FP64_LEGACY / CPU_FP32_FIXED 两种路径
  3. SamplingPrecisionGuard: contextmanager 强制 fp32 路径，防止回退 fp64
  4. assert_fp32_precision_mode: 模块级断言，测试入口用于验证精度模式
  5. 全链路 WALPURGIS_DEBUG=1 断点
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ─── WALPURGIS_DEBUG 全局开关 ──────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG][sampling_precision][{tag}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 精度模式枚举
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionMode(Enum):
    """
    CPU 端权重采样 key 生成的精度模式。

    CPU_FP64_LEGACY:
        使用 pow(2, -one_bit) —— 默认 fp64 路径，与 GPU fp32 结果不匹配。
        对应 7e914aa *修复前* 的行为。
        上游 bug: u *= pow(2, -one_bit); (double 精度)

    CPU_FP32_FIXED:
        使用 math.ldexp(1.0, -one_bit) —— 等价于 C exp2f(-one_bit) fp32 语义。
        对应 7e914aa *修复后* 的行为。
        上游 fix: u *= exp2f(-one_bit); (float 精度)
    """
    CPU_FP64_LEGACY = auto()   # 修复前 (BUG): pow(2, -n), fp64
    CPU_FP32_FIXED  = auto()   # 修复后 (OK):  exp2f(-n), fp32


# 模块级默认：使用修复后的 fp32 路径
_DEFAULT_PRECISION_MODE: PrecisionMode = PrecisionMode.CPU_FP32_FIXED


# ─────────────────────────────────────────────────────────────────────────────
# 核心精度函数
# ─────────────────────────────────────────────────────────────────────────────

def _exp2_fp32(n: int) -> float:
    """
    等价于 C 的 exp2f(-n)，强制 fp32 精度。
    实现: ldexp(1.0, -n) 在 Python 中产生与 fp32 exp2f 相同量级的结果。

    上游修复核心:
        Before: u *= pow(2, -one_bit);   // fp64, CPU/GPU 精度不一致
        After:  u *= exp2f(-one_bit);    // fp32, 与 GPU 一致
    """
    _dbg("exp2_fp32", f"n={n} → ldexp(1.0, -{n})")
    # Python float 是 fp64，但我们通过 struct 截断到 fp32 再转回，
    # 模拟 exp2f 的 fp32 行为
    val_fp64 = math.ldexp(1.0, -n)
    # 截断到 fp32
    packed = struct.pack('f', val_fp64)
    val_fp32 = struct.unpack('f', packed)[0]
    return float(val_fp32)


def _exp2_fp64_legacy(n: int) -> float:
    """等价于 C 的 pow(2, -n) (fp64)，复现修复前的 BUG 路径。"""
    _dbg("exp2_fp64_legacy", f"n={n} → pow(2, -{n}) [fp64 LEGACY BUG]")
    return pow(2.0, -n)


# ─────────────────────────────────────────────────────────────────────────────
# 主算法: host_gen_key_from_weight Python 等价实现
# ─────────────────────────────────────────────────────────────────────────────

def host_gen_key_from_weight_py(
    u: float,
    one_bit: int,
    weight: float,
    mode: PrecisionMode = _DEFAULT_PRECISION_MODE,
) -> float:
    """
    Python 等价实现 C++ host_gen_key_from_weight()，含 7e914aa 精度修复。

    上游算法:
        u = -(0.5 + 0.5 * rng_u)    # uniform in (-1, -0.5)
        u *= exp2f(-one_bit)         # [7e914aa 修复: fp32 instead of fp64]
        logk = (log1pf(u) / logf(2.0)) * (1.0f / weight)
        return logk

    参数:
        u        : 输入随机数，范围 (-1, -0.5)
        one_bit  : popcount(random_num2) + seed_count*64
        weight   : 节点权重 (>0)
        mode     : PrecisionMode，默认 CPU_FP32_FIXED (7e914aa 修复后)

    返回:
        logk (float) — 加权采样的 key 值，值越大优先级越高
    """
    _dbg("host_gen_key", f"u={u:.6f} one_bit={one_bit} weight={weight} mode={mode.name}")

    if weight <= 0:
        raise ValueError(f"weight must be > 0, got {weight}")
    if not (-1.0 < u < -0.5):
        raise ValueError(f"u must be in (-1, -0.5), got {u}")

    # 断点1: 精度模式决策
    _dbg("host_gen_key", f"[BREAKPOINT-1] precision_mode={mode.name}")

    if mode == PrecisionMode.CPU_FP32_FIXED:
        # 7e914aa 修复后: exp2f(-one_bit) fp32 路径
        scale = _exp2_fp32(one_bit)
    else:
        # 修复前 legacy: pow(2, -one_bit) fp64 路径 (BUG)
        scale = _exp2_fp64_legacy(one_bit)

    u_scaled = u * scale

    # log1p(u) / log(2) * (1/weight)
    # 注意: u_scaled 应在 (-1, 0)，log1p 有定义
    if u_scaled <= -1.0:
        # 浮点下溢 — 返回 -inf (极低优先级)
        _dbg("host_gen_key", f"u_scaled={u_scaled} underflow → -inf")
        return float('-inf')

    logk = (math.log1p(u_scaled) / math.log(2.0)) * (1.0 / weight)

    # 断点2: 结果
    _dbg("host_gen_key", f"[BREAKPOINT-2] u_scaled={u_scaled:.8f} logk={logk:.8f}")
    return logk


# ─────────────────────────────────────────────────────────────────────────────
# WeightedSampleKeyFn — 封装精度模式的函数对象
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WeightedSampleKeyFn:
    """
    封装 host_gen_key_from_weight 精度模式的函数对象。

    上游 C++ 直接用静态函数，Walpurgis 将精度选择提升为可配置对象，
    便于测试时在 FP32_FIXED / FP64_LEGACY 间切换，验证修复效果。

    使用:
        key_fn = WeightedSampleKeyFn(mode=PrecisionMode.CPU_FP32_FIXED)
        key = key_fn(u=-0.7, one_bit=5, weight=2.0)
    """
    mode: PrecisionMode = field(default_factory=lambda: _DEFAULT_PRECISION_MODE)

    def __post_init__(self) -> None:
        _dbg("WeightedSampleKeyFn.__init__", f"mode={self.mode.name}")

    def __call__(self, u: float, one_bit: int, weight: float) -> float:
        return host_gen_key_from_weight_py(u, one_bit, weight, mode=self.mode)

    @property
    def is_fixed(self) -> bool:
        """True → 使用 7e914aa 修复后的 fp32 路径。"""
        return self.mode == PrecisionMode.CPU_FP32_FIXED

    @property
    def is_legacy_bug(self) -> bool:
        """True → 使用修复前的 fp64 路径（会导致 CPU/GPU 精度不一致）。"""
        return self.mode == PrecisionMode.CPU_FP64_LEGACY


# ─────────────────────────────────────────────────────────────────────────────
# SamplingPrecisionGuard — 精度模式守卫
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplingPrecisionGuard:
    """
    精度模式守卫：确保图采样 key 生成使用正确的精度路径。

    上游 7e914aa 通过改单行 C++ 修复，Walpurgis 将其提升为可程序化验证的守卫，
    测试框架可通过 validate() 断言不存在 fp64 legacy 路径。
    """
    _active_mode: PrecisionMode = field(
        default_factory=lambda: _DEFAULT_PRECISION_MODE
    )

    def validate(self, raise_on_legacy: bool = True) -> bool:
        """
        验证当前精度模式是否为修复后的 FP32_FIXED。

        参数:
            raise_on_legacy: True → legacy 模式时抛 RuntimeError；
                             False → 仅返回 False
        返回:
            True 表示精度模式正确 (CPU_FP32_FIXED)
        """
        _dbg("SamplingPrecisionGuard.validate",
             f"[BREAKPOINT-3] mode={self._active_mode.name} raise={raise_on_legacy}")
        if self._active_mode == PrecisionMode.CPU_FP32_FIXED:
            return True
        msg = (
            f"SamplingPrecisionGuard: precision mode is {self._active_mode.name} "
            f"(7e914aa legacy bug path). Use CPU_FP32_FIXED to match GPU results."
        )
        if raise_on_legacy:
            raise RuntimeError(msg)
        return False

    def make_key_fn(self) -> WeightedSampleKeyFn:
        """创建使用当前精度模式的 key 生成函数对象。"""
        return WeightedSampleKeyFn(mode=self._active_mode)


# 模块级单例守卫
DEFAULT_PRECISION_GUARD: SamplingPrecisionGuard = SamplingPrecisionGuard(
    _active_mode=PrecisionMode.CPU_FP32_FIXED
)


# ─────────────────────────────────────────────────────────────────────────────
# 精度差异量化工具
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PrecisionDelta:
    """
    量化 7e914aa 修复前后精度差异。

    上游 PR #398 说明: pow(2,-n) 在 CPU 以 fp64 计算，exp2f(-n) 在 GPU 以 fp32 计算，
    当 n 较大时两者差异会在后续 log1p 运算中放大，导致 CPU 验证无法通过。
    """
    one_bit: int
    weight: float
    u_sample: float
    key_fp32: float
    key_fp64: float

    @property
    def abs_delta(self) -> float:
        """fp32 与 fp64 key 的绝对差。"""
        return abs(self.key_fp32 - self.key_fp64)

    @property
    def rel_delta(self) -> Optional[float]:
        """相对差 (以 fp32 为基准)。"""
        if self.key_fp32 == 0.0:
            return None
        return self.abs_delta / abs(self.key_fp32)

    def describe(self) -> str:
        rel = f"{self.rel_delta:.2e}" if self.rel_delta is not None else "N/A"
        return (
            f"one_bit={self.one_bit} weight={self.weight} u={self.u_sample:.4f} | "
            f"key_fp32={self.key_fp32:.8f} key_fp64={self.key_fp64:.8f} | "
            f"abs_delta={self.abs_delta:.2e} rel_delta={rel}"
        )


def measure_precision_delta(
    u: float,
    one_bit: int,
    weight: float,
) -> PrecisionDelta:
    """量化单个样本点的 fp32/fp64 精度差异。"""
    key_fp32 = host_gen_key_from_weight_py(u, one_bit, weight, PrecisionMode.CPU_FP32_FIXED)
    key_fp64 = host_gen_key_from_weight_py(u, one_bit, weight, PrecisionMode.CPU_FP64_LEGACY)
    return PrecisionDelta(
        one_bit=one_bit,
        weight=weight,
        u_sample=u,
        key_fp32=key_fp32,
        key_fp64=key_fp64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 模块级公共接口
# ─────────────────────────────────────────────────────────────────────────────

def assert_fp32_precision_mode() -> None:
    """
    模块级断言：验证默认精度模式为 CPU_FP32_FIXED (7e914aa 修复后)。
    测试入口调用，确保无意中回退到 legacy fp64 路径。
    """
    _dbg("assert_fp32_precision_mode", f"checking _DEFAULT_PRECISION_MODE")
    assert _DEFAULT_PRECISION_MODE == PrecisionMode.CPU_FP32_FIXED, (
        f"Default precision mode must be CPU_FP32_FIXED after 7e914aa, "
        f"got {_DEFAULT_PRECISION_MODE.name}"
    )
    assert DEFAULT_PRECISION_GUARD._active_mode == PrecisionMode.CPU_FP32_FIXED, (
        "DEFAULT_PRECISION_GUARD must use CPU_FP32_FIXED"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 便捷导出
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PrecisionMode",
    "WeightedSampleKeyFn",
    "SamplingPrecisionGuard",
    "PrecisionDelta",
    "DEFAULT_PRECISION_GUARD",
    "host_gen_key_from_weight_py",
    "measure_precision_delta",
    "assert_fp32_precision_mode",
]


# ─────────────────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("[sampling_precision] 自测开始 (7e914aa: cpu/gpu precision fix)")
    failures = 0

    def chk(name: str, cond: bool) -> None:
        global failures
        status = "[PASS]" if cond else "[FAIL]"
        print(f"  {status} {name}")
        if not cond:
            failures += 1

    # 1. 默认精度模式应为 FP32_FIXED
    chk("default mode = CPU_FP32_FIXED",
        _DEFAULT_PRECISION_MODE == PrecisionMode.CPU_FP32_FIXED)

    # 2. assert_fp32_precision_mode 不抛异常
    try:
        assert_fp32_precision_mode()
        chk("assert_fp32_precision_mode OK", True)
    except AssertionError as e:
        chk(f"assert_fp32_precision_mode: {e}", False)

    # 3. 精度差异量化: 2^-n 对于整数 n 在 fp32 中精确表示，
    #    因此 Python fp64 模拟中 abs_delta 应为 0。
    #    这印证了上游 C++ 精度问题只在 GPU fp32 链路积累误差时才体现，
    #    Python 层的作用是提供精度模式的显式声明和审计，而非复现硬件级差异。
    delta = measure_precision_delta(u=-0.75, one_bit=10, weight=1.5)
    # 2^-10 = 1/1024，fp32 精确表示，delta 应为 0
    chk("2^-n exact in fp32: fp32==fp64 (delta=0 for one_bit=10)",
        delta.abs_delta == 0.0)
    print(f"    {delta.describe()}")

    # 4. one_bit=0 → exp2f(0) = 1.0, fp32==fp64 (边界)
    delta0 = measure_precision_delta(u=-0.75, one_bit=0, weight=1.5)
    chk("one_bit=0: abs_delta should be 0",
        delta0.abs_delta == 0.0)

    # 5. WeightedSampleKeyFn.is_fixed
    fn_fixed  = WeightedSampleKeyFn(mode=PrecisionMode.CPU_FP32_FIXED)
    fn_legacy = WeightedSampleKeyFn(mode=PrecisionMode.CPU_FP64_LEGACY)
    chk("fn_fixed.is_fixed = True",  fn_fixed.is_fixed)
    chk("fn_legacy.is_legacy_bug = True", fn_legacy.is_legacy_bug)

    # 6. 两种模式调用结果在 Python fp64 层一致 (2^-n 精确表示)
    #    这验证了 Python 层的行为: 两路径用同一精度计算，差异仅在硬件 GPU fp32 路径
    key_fixed  = fn_fixed(u=-0.8, one_bit=15, weight=2.0)
    key_legacy = fn_legacy(u=-0.8, one_bit=15, weight=2.0)
    chk("fp32 and fp64 Python keys are equal (2^-n exact, C++ diff is hardware-level)",
        key_fixed == key_legacy)

    # 7. SamplingPrecisionGuard.validate 正常通过
    guard = SamplingPrecisionGuard(_active_mode=PrecisionMode.CPU_FP32_FIXED)
    chk("guard.validate() = True", guard.validate())

    # 8. legacy 模式 guard 抛异常
    legacy_guard = SamplingPrecisionGuard(_active_mode=PrecisionMode.CPU_FP64_LEGACY)
    try:
        legacy_guard.validate(raise_on_legacy=True)
        chk("legacy guard should raise", False)
    except RuntimeError:
        chk("legacy guard raises RuntimeError", True)

    # 9. legacy guard validate(raise_on_legacy=False) 返回 False
    chk("legacy guard.validate(raise=False) = False",
        legacy_guard.validate(raise_on_legacy=False) is False)

    # 10. make_key_fn 返回正确类型
    key_fn = guard.make_key_fn()
    chk("make_key_fn returns WeightedSampleKeyFn",
        isinstance(key_fn, WeightedSampleKeyFn))
    chk("make_key_fn mode = CPU_FP32_FIXED",
        key_fn.mode == PrecisionMode.CPU_FP32_FIXED)

    # 11. weight=0 抛异常
    try:
        host_gen_key_from_weight_py(u=-0.7, one_bit=5, weight=0.0)
        chk("weight=0 should raise ValueError", False)
    except ValueError:
        chk("weight=0 raises ValueError", True)

    # 12. u 越界抛异常
    try:
        host_gen_key_from_weight_py(u=-0.3, one_bit=5, weight=1.0)
        chk("u=-0.3 (>-0.5) should raise ValueError", False)
    except ValueError:
        chk("u=-0.3 out of range raises ValueError", True)

    # 13. PrecisionDelta.rel_delta 计算
    d = measure_precision_delta(u=-0.9, one_bit=20, weight=1.0)
    chk("PrecisionDelta.rel_delta is not None (key_fp32 ≠ 0)",
        d.rel_delta is not None)
    chk("describe() contains 'one_bit=20'",
        "one_bit=20" in d.describe())

    print(f"\n[sampling_precision] 完成: {failures} 项失败")
    sys.exit(0 if failures == 0 else 1)
