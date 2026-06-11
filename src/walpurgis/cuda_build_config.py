"""
cuda_build_config.py — b89f57d 迁移: CUDA 设备代码压缩编译 Flag 配置层

上游来源: cpp/CMakeLists.txt
commit: b89f57d (Enable device code compression, Robert Maynard, 2025-05-13, PR #202)

上游逻辑（CMake，9行）:
  list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
     AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
          AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0))
    list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=--compress-level=3)
  endif()

Walpurgis 改写20%（鲁迅拿法）:
1. _NvccProbe dataclass 封装 nvcc 探测结果，替代 CMake 散落变量
   （上游: CMAKE_CUDA_COMPILER_VERSION / CMAKE_CUDA_COMPILER_ID 分散）
2. _FatbinPolicy dataclass 封装压缩策略决策，使"为何选此 flag"可审计
3. get_fatbin_flags() 惰性单例 + 显式缓存，避免重复 subprocess 调用
4. _parse_nvcc_version() 独立出来，避免上游 CMake 正则 magic 不可测试
5. 全链路 WALPURGIS_DEBUG=1 断点 print:
   - nvcc 探测入口/结果
   - 版本解析详情
   - 策略决策路径（base / tune / final）
   - get_fatbin_flags() 命中缓存 vs 计算路径
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 调试开关（与整个 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS cuda_build_config b89f57d]", *args, **kwargs)


# ─────────────────────────────────────────────────────────────
# _NvccProbe — nvcc 可用性及版本探测结果
# 上游用 CMake find_package(CUDAToolkit)，这里改为 subprocess 主动探测
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _NvccProbe:
    """nvcc 编译器探测结果，不可变（frozen），便于哈希和断言。"""
    available: bool
    compiler_id: str          # 通常 "NVIDIA" 或 "" (not found)
    version_str: str          # e.g. "12.9.86"
    major: int
    minor: int
    patch: int

    def version_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        if not self.available:
            return "NvccProbe(unavailable)"
        return (
            f"NvccProbe(id={self.compiler_id!r}, "
            f"version={self.version_str}, "
            f"tuple={self.version_tuple()})"
        )


# ─────────────────────────────────────────────────────────────
# _FatbinPolicy — fatbin 压缩策略决策记录
# 上游仅 append 到 CMake 变量，无决策可见性；这里显式记录
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FatbinPolicy:
    """
    fatbin 压缩策略决策，记录 base flag、tune flag 及采用理由。

    base_flag: 适用于所有支持 nvcc 的构建环境（nvcc 可用即生效）
    tune_flag: 仅 CUDA 12.9.x 可用（--compress-level=3 是 12.9 新参数）
    reason:    决策路径说明，便于调试和审计
    """
    base_flag: str           # "-Xfatbin=-compress-all" 或 ""
    tune_flag: str           # "-Xfatbin=--compress-level=3" 或 ""
    reason: str

    def flags(self) -> List[str]:
        """返回实际追加到 NVFLAGS 的 flag 列表（过滤空串）。"""
        return [f for f in (self.base_flag, self.tune_flag) if f]

    def __str__(self) -> str:
        return (
            f"FatbinPolicy("
            f"base={self.base_flag!r}, "
            f"tune={self.tune_flag!r}, "
            f"flags={self.flags()}, "
            f"reason={self.reason!r})"
        )


# ─────────────────────────────────────────────────────────────
# 内部: nvcc 版本解析
# 上游依赖 CMake 正则；这里改为 Python re，显式测试友好
# ─────────────────────────────────────────────────────────────

def _parse_nvcc_version(raw: str) -> Tuple[int, int, int]:
    """
    从 `nvcc --version` 输出中提取 (major, minor, patch)。

    支持格式:
      release 12.9, V12.9.86
      release 11.8, V11.8.89
    失败时返回 (0, 0, 0)。
    """
    _dbg(f"_parse_nvcc_version: raw={raw!r}")

    # 优先匹配 "V<major>.<minor>.<patch>"
    m = re.search(r"V(\d+)\.(\d+)\.(\d+)", raw)
    if m:
        t = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        _dbg(f"_parse_nvcc_version: matched V-pattern → {t}")
        return t

    # 备选: "release <major>.<minor>"（无 patch）
    m2 = re.search(r"release\s+(\d+)\.(\d+)", raw)
    if m2:
        t = (int(m2.group(1)), int(m2.group(2)), 0)
        _dbg(f"_parse_nvcc_version: matched release-pattern → {t}")
        return t

    _dbg("_parse_nvcc_version: no match, returning (0,0,0)")
    return (0, 0, 0)


# ─────────────────────────────────────────────────────────────
# 内部: 探测 nvcc
# ─────────────────────────────────────────────────────────────

def _probe_nvcc() -> _NvccProbe:
    """运行 nvcc --version，返回 _NvccProbe。失败时返回 unavailable 实例。"""
    _dbg("_probe_nvcc: 开始探测 nvcc --version")

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        raw = result.stdout + result.stderr
        _dbg(f"_probe_nvcc: nvcc returncode={result.returncode}")
        _dbg(f"_probe_nvcc: stdout={result.stdout!r}")

        if result.returncode != 0:
            _dbg("_probe_nvcc: returncode!=0, 视为不可用")
            return _NvccProbe(
                available=False, compiler_id="", version_str="",
                major=0, minor=0, patch=0,
            )

        major, minor, patch = _parse_nvcc_version(raw)
        version_str = f"{major}.{minor}.{patch}" if major > 0 else ""

        # nvcc 存在且来自 NVIDIA toolkit，compiler_id 固定为 "NVIDIA"
        compiler_id = "NVIDIA" if major > 0 else ""

        probe = _NvccProbe(
            available=True,
            compiler_id=compiler_id,
            version_str=version_str,
            major=major, minor=minor, patch=patch,
        )
        _dbg(f"_probe_nvcc: 探测完成 → {probe}")
        return probe

    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        _dbg(f"_probe_nvcc: 异常 {type(exc).__name__}: {exc}")
        return _NvccProbe(
            available=False, compiler_id="", version_str="",
            major=0, minor=0, patch=0,
        )


# ─────────────────────────────────────────────────────────────
# 内部: 根据探测结果决策 fatbin 策略
# 直接对应上游 CMakeLists.txt b89f57d 逻辑
# ─────────────────────────────────────────────────────────────

def _decide_fatbin_policy(probe: _NvccProbe) -> _FatbinPolicy:
    """
    根据 _NvccProbe 决策 fatbin 压缩 flag。

    对应上游 CMake 逻辑:
      - 无条件追加 -Xfatbin=-compress-all（nvcc 可用时）
      - CUDA 12.9.x 额外追加 -Xfatbin=--compress-level=3
    """
    _dbg(f"_decide_fatbin_policy: probe={probe}")

    # Case 1: nvcc 不可用 → 无 flag
    if not probe.available:
        policy = _FatbinPolicy(
            base_flag="",
            tune_flag="",
            reason="nvcc unavailable — no fatbin flags applied",
        )
        _dbg(f"_decide_fatbin_policy: → {policy}")
        return policy

    # Case 2: nvcc 可用，compiler_id != "NVIDIA" (理论上不会发生，防御性检查)
    if probe.compiler_id != "NVIDIA":
        policy = _FatbinPolicy(
            base_flag="-Xfatbin=-compress-all",
            tune_flag="",
            reason=(
                f"nvcc available but compiler_id={probe.compiler_id!r} "
                f"(not 'NVIDIA') — base flag only, tune skipped"
            ),
        )
        _dbg(f"_decide_fatbin_policy: → {policy}")
        return policy

    # Case 3: NVIDIA nvcc，判断是否 12.9.x
    # 上游: VERSION_GREATER_EQUAL 12.9 AND VERSION_LESS 13.0
    # 即 major==12 且 minor==9 (patch 任意)
    is_cuda_129x = (probe.major == 12 and probe.minor == 9)
    _dbg(f"_decide_fatbin_policy: is_cuda_129x={is_cuda_129x} "
         f"(major={probe.major}, minor={probe.minor})")

    tune_flag = "-Xfatbin=--compress-level=3" if is_cuda_129x else ""
    reason = (
        f"NVIDIA nvcc {probe.version_str}: base=-compress-all"
        + (", tune=--compress-level=3 (CUDA 12.9.x)" if is_cuda_129x
           else f", tune=skipped (not 12.9.x)")
    )

    policy = _FatbinPolicy(
        base_flag="-Xfatbin=-compress-all",
        tune_flag=tune_flag,
        reason=reason,
    )
    _dbg(f"_decide_fatbin_policy: → {policy}")
    return policy


# ─────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────

# 模块级缓存（等价 CMake configure-once 语义）
_cached_probe: Optional[_NvccProbe] = None
_cached_policy: Optional[_FatbinPolicy] = None


def get_nvcc_probe(force: bool = False) -> _NvccProbe:
    """
    返回（或重新探测）nvcc 编译器信息。

    Args:
        force: True 时绕过缓存重新运行 nvcc --version。
    """
    global _cached_probe
    _dbg(f"get_nvcc_probe: force={force}, cached={_cached_probe is not None}")

    if _cached_probe is None or force:
        _dbg("get_nvcc_probe: 执行探测")
        _cached_probe = _probe_nvcc()
        _dbg(f"get_nvcc_probe: 探测结果 → {_cached_probe}")
    else:
        _dbg(f"get_nvcc_probe: 命中缓存 → {_cached_probe}")

    return _cached_probe


def get_fatbin_flags(force: bool = False) -> List[str]:
    """
    返回应追加到 NVCC 编译命令的 fatbin 压缩 flag 列表。

    对应上游 b89f57d: WHOLEGRAPH_CUDA_FLAGS 中的 fatbin 压缩部分。
    结果被模块级缓存，保证多次调用幂等（等价 CMake configure-once）。

    Args:
        force: True 时绕过缓存，重新探测 nvcc 并重新决策。

    Returns:
        flag 列表，例如:
          []                                           # nvcc 不可用
          ["-Xfatbin=-compress-all"]                   # CUDA < 12.9 或 >= 13
          ["-Xfatbin=-compress-all",
           "-Xfatbin=--compress-level=3"]              # CUDA 12.9.x

    Example::

        import subprocess, shlex
        flags = get_fatbin_flags()
        cmd = ["nvcc", "-std=c++17", "-O2"] + flags + ["-o", "out", "kernel.cu"]
        subprocess.run(cmd)
    """
    global _cached_policy
    _dbg(f"get_fatbin_flags: force={force}, cached={_cached_policy is not None}")

    if _cached_policy is None or force:
        _dbg("get_fatbin_flags: 未命中缓存，执行决策")
        probe = get_nvcc_probe(force=force)
        _cached_policy = _decide_fatbin_policy(probe)
        _dbg(f"get_fatbin_flags: 决策完成 → {_cached_policy}")
    else:
        _dbg(f"get_fatbin_flags: 命中缓存 → {_cached_policy}")

    flags = _cached_policy.flags()
    _dbg(f"get_fatbin_flags: 返回 flags={flags}")
    return flags


def get_fatbin_policy(force: bool = False) -> _FatbinPolicy:
    """
    返回完整的 _FatbinPolicy 决策记录（含 reason）。

    供调试、日志、单元测试使用。
    """
    _dbg(f"get_fatbin_policy: force={force}")
    get_fatbin_flags(force=force)  # 触发缓存填充
    assert _cached_policy is not None
    _dbg(f"get_fatbin_policy: → {_cached_policy}")
    return _cached_policy


def nvflags_with_fatbin(base_flags: Optional[List[str]] = None) -> List[str]:
    """
    在给定 base_flags 基础上追加 fatbin 压缩 flag，返回合并后的完整 NVFLAGS。

    等价上游: list(APPEND WHOLEGRAPH_CUDA_FLAGS ...)

    Args:
        base_flags: 已有 nvcc flag 列表。None 时使用 Walpurgis 默认值。

    Returns:
        合并后的 flag 列表（原列表不修改）。

    Example::

        flags = nvflags_with_fatbin(["-std=c++17", "-O2", "-arch=sm_86"])
        # → ["-std=c++17", "-O2", "-arch=sm_86",
        #    "-Xfatbin=-compress-all", "-Xfatbin=--compress-level=3"]
    """
    if base_flags is None:
        # Walpurgis Makefile 默认 NVFLAGS（无 fatbin 部分）
        base_flags = [
            "-std=c++14", "-O2",
            "-Xcompiler", "-pthread,-fopenmp,-Wall",
            "-lineinfo",
        ]

    _dbg(f"nvflags_with_fatbin: base_flags={base_flags}")
    fatbin = get_fatbin_flags()
    result = list(base_flags) + fatbin
    _dbg(f"nvflags_with_fatbin: result={result}")
    return result


# ─────────────────────────────────────────────────────────────
# CLI 自检（WALPURGIS_DEBUG=1 python cuda_build_config.py）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("[cuda_build_config b89f57d] 自检启动")
    print(f"  WALPURGIS_DEBUG={os.environ.get('WALPURGIS_DEBUG', '0')}")

    probe = get_nvcc_probe()
    print(f"  nvcc probe     : {probe}")

    policy = get_fatbin_policy()
    print(f"  fatbin policy  : {policy}")

    flags = get_fatbin_flags()
    print(f"  fatbin flags   : {flags}")

    full_flags = nvflags_with_fatbin()
    print(f"  full NVFLAGS   : {full_flags}")

    # 幂等性验证: 二次调用应命中缓存，结果不变
    flags2 = get_fatbin_flags()
    assert flags == flags2, f"幂等性失败: {flags} != {flags2}"
    print("  幂等性检查     : PASS")

    # force 刷新验证
    flags3 = get_fatbin_flags(force=True)
    assert flags == flags3, f"force 刷新后结果变化: {flags} != {flags3}"
    print("  force 刷新检查 : PASS")

    print("[cuda_build_config b89f57d] 自检完成")
    sys.exit(0)
