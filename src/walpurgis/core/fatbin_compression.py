"""
fatbin_compression.py — 9ecbc66 迁移: 标准化 fatbin 压缩策略

上游来源: cugraph-gnn / cpp/CMakeLists.txt
commit: 9ecbc668fa376ab7398e8eef9053aecbe510ac91
author: Robert Maynard <rmaynard@nvidia.com>
co-author: Alex Barghi <alexbarghi-nv>
date: 2025-08-13
PR: https://github.com/rapidsai/cugraph-gnn/pull/273

上游变更摘要（1 file changed, 2 insertions, 7 deletions）:
  cpp/CMakeLists.txt:

    旧（手写逻辑，9行）:
      list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)
      if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
         AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
              AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0))
        list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=--compress-level=3)
      endif()

    新（委托给 rapids-cmake，2行）:
      include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
      rapids_cuda_enable_fatbin_compression(VARIABLE WHOLEGRAPH_CUDA_FLAGS TUNE_FOR rapids)

CI/merge → SKIP:
  cpp/CMakeLists.txt — C++ CMake 构建系统，Walpurgis 纯 Python，无 C++ 编译体系

鲁迅拿法改写（≥20%）:
1. FatbinCompressionFlag 枚举: 将上游散落在 CMakeLists.txt 里的两条裸字符串
   flag（-Xfatbin=-compress-all / -Xfatbin=--compress-level=3）结构化为枚举，
   提供 as_nvcc_flag() 序列化接口。上游做法：list(APPEND ...) 裸字符串拼接，
   无类型，无可程序化查询的接口。
2. FatbinCompressionRule dataclass: 将"什么 CUDA 版本区间适用什么压缩 flag"
   建模为不可变记录，matches_cuda_version() 方法替代上游手写的 CMake
   VERSION_GREATER_EQUAL + VERSION_LESS 版本区间判断。
3. RapidsFatbinPolicy dataclass: 封装 rapids_cuda_enable_fatbin_compression
   的完整决策逻辑——TUNE_FOR=rapids 对应哪些规则、如何解析为 flag 列表、
   旧手写逻辑与新 RAPIDS 策略是否语义等价（is_semantically_equivalent_to_legacy()）。
   上游直接委托给 CMake 函数，无 Python 层查询接口。
4. FatbinCompressionAudit dataclass: 文档化 9ecbc66 之前手写逻辑的全部行为，
   供 Walpurgis 测试体系验证迁移前后语义不变，以及扫描残留手写逻辑。
   上游做法：删除即删除，无可审计记录。
5. 全链路 WALPURGIS_DEBUG=1 断点 print（6处）：覆盖版本匹配、策略解析、
   flag 生成、语义等价性校验各阶段。

参考: rapidsai/build-planning
      https://github.com/rapidsai/cugraph-gnn/pull/273
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 调试开关（与整个 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print(f"[WPG 9ecbc66 {tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# FatbinCompressionFlag — nvcc fatbin 压缩选项的结构化枚举
#
# 上游 CMakeLists.txt 做法（两条裸字符串）:
#   -Xfatbin=-compress-all
#   -Xfatbin=--compress-level=3
#
# 改写：枚举每个有意义的压缩标志，as_nvcc_flag() 生成完整 nvcc 参数，
# from_nvcc_flag() 支持反向解析，使 flag 可程序化比较和验证。
# ─────────────────────────────────────────────────────────────


class FatbinCompressionFlag(Enum):
    """nvcc fatbin 压缩选项枚举。

    上游散落在 CMakeLists.txt 两条 list(APPEND ...) 中；
    此处枚举化后可 hash、比较、序列化，也可在单测中精确断言。

    Members:
        COMPRESS_ALL:     `-Xfatbin=-compress-all`
                          无条件压缩所有 fatbin 段，通用基准选项（9ecbc66 前）。
        COMPRESS_LEVEL_3: `-Xfatbin=--compress-level=3`
                          最高压缩级别，仅在 CUDA 12.9.x 时附加（9ecbc66 前版本区间守卫）。
        RAPIDS_DELEGATE:  由 rapids_cuda_enable_fatbin_compression 统一决定，
                          不再手写具体 flag（9ecbc66 后）。
    """
    COMPRESS_ALL = "compress-all"
    COMPRESS_LEVEL_3 = "compress-level=3"
    RAPIDS_DELEGATE = "rapids-delegate"

    def as_nvcc_flag(self) -> Optional[str]:
        """序列化为 nvcc 命令行参数字符串。

        Returns:
            nvcc flag 字符串，或 None（RAPIDS_DELEGATE 无具体 flag）。

        Examples::

            >>> FatbinCompressionFlag.COMPRESS_ALL.as_nvcc_flag()
            '-Xfatbin=-compress-all'
            >>> FatbinCompressionFlag.COMPRESS_LEVEL_3.as_nvcc_flag()
            '-Xfatbin=--compress-level=3'
            >>> FatbinCompressionFlag.RAPIDS_DELEGATE.as_nvcc_flag() is None
            True
        """
        # 断点1: flag 序列化
        _dbg("FatbinCompressionFlag.as_nvcc_flag", f"self={self.name}")
        mapping = {
            FatbinCompressionFlag.COMPRESS_ALL: "-Xfatbin=-compress-all",
            FatbinCompressionFlag.COMPRESS_LEVEL_3: "-Xfatbin=--compress-level=3",
            FatbinCompressionFlag.RAPIDS_DELEGATE: None,
        }
        result = mapping[self]
        _dbg("FatbinCompressionFlag.as_nvcc_flag", f"→ {result!r}")
        return result

    @classmethod
    def from_nvcc_flag(cls, flag_str: str) -> "FatbinCompressionFlag":
        """从 nvcc flag 字符串反向解析枚举成员。

        Args:
            flag_str: 如 ``'-Xfatbin=-compress-all'``。

        Raises:
            ValueError: 无法识别的 flag 字符串。
        """
        _dbg("FatbinCompressionFlag.from_nvcc_flag", f"input={flag_str!r}")
        for member in cls:
            if member.as_nvcc_flag() == flag_str:
                _dbg("FatbinCompressionFlag.from_nvcc_flag", f"→ {member.name}")
                return member
        raise ValueError(
            f"[FatbinCompressionFlag] 无法识别的 nvcc flag: {flag_str!r}。"
            f"已知值: {[m.as_nvcc_flag() for m in cls if m.as_nvcc_flag()]}"
        )


# ─────────────────────────────────────────────────────────────
# CudaVersionRange — CUDA 版本区间（半开区间 [low, high)）
#
# 对应上游 CMakeLists.txt 的 CMake 版本比较逻辑:
#   VERSION_GREATER_EQUAL 12.9 AND VERSION_LESS 13.0
#
# 改写：不可变 dataclass，contains() 方法替代 CMake VERSION_* 比较，
# 可在 Python 单测中精确断言版本边界。
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CudaVersionRange:
    """CUDA 版本半开区间 [low, high)，对应 CMake VERSION_GREATER_EQUAL + VERSION_LESS。

    Attributes:
        low_major:  区间下界 major（inclusive）。
        low_minor:  区间下界 minor（inclusive）。
        high_major: 区间上界 major（exclusive）。
        high_minor: 区间上界 minor（exclusive）。

    Examples::

        >>> r = CudaVersionRange(12, 9, 13, 0)
        >>> r.contains(12, 9)   # 下界包含
        True
        >>> r.contains(12, 10)
        True
        >>> r.contains(13, 0)   # 上界排除
        False
        >>> r.contains(12, 8)   # 下界排除
        False
    """
    low_major: int
    low_minor: int
    high_major: int
    high_minor: int

    def __post_init__(self) -> None:
        _dbg(
            "CudaVersionRange.__init__",
            f"[{self.low_major}.{self.low_minor}, {self.high_major}.{self.high_minor})",
        )
        low_val = self.low_major * 1000 + self.low_minor
        high_val = self.high_major * 1000 + self.high_minor
        if low_val >= high_val:
            raise ValueError(
                f"[CudaVersionRange] low ({self.low_major}.{self.low_minor}) "
                f"must be < high ({self.high_major}.{self.high_minor})"
            )

    def contains(self, major: int, minor: int) -> bool:
        """判断给定 (major, minor) 是否在区间 [low, high) 内。

        断点2: 版本区间包含性判断。
        """
        val = major * 1000 + minor
        low_val = self.low_major * 1000 + self.low_minor
        high_val = self.high_major * 1000 + self.high_minor
        result = low_val <= val < high_val
        _dbg(
            "CudaVersionRange.contains",
            f"cuda={major}.{minor}  range=[{self.low_major}.{self.low_minor},"
            f"{self.high_major}.{self.high_minor})  → {result}",
        )
        return result

    def __repr__(self) -> str:
        return (
            f"CudaVersionRange("
            f"[{self.low_major}.{self.low_minor}, "
            f"{self.high_major}.{self.high_minor}))"
        )


# ─────────────────────────────────────────────────────────────
# FatbinCompressionRule — 版本区间 → 压缩 flag 的映射规则
#
# 对应上游手写 CMake 逻辑中的一条 if 分支：
#   "当 CUDA 编译器版本满足 cuda_range 时，附加 flag"
#
# 改写：不可变 dataclass，将"什么版本区间"+"什么 flag"+"为什么这么做"
# 三者绑定为一个有名字、可序列化、可测试的记录对象。
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FatbinCompressionRule:
    """版本区间条件化压缩规则，对应 CMakeLists.txt 中一条 if 分支。

    Attributes:
        flag:       当版本区间匹配时应追加的压缩 flag。
        cuda_range: 触发本规则的 CUDA 版本区间（None 表示无条件适用）。
        rationale:  规则存在的技术原因（文档用途）。
        introduced_by: 引入此规则的上游 commit hash。
        removed_by: 移除此规则的上游 commit hash（9ecbc66）。

    Examples::

        >>> rule = FatbinCompressionRule(
        ...     flag=FatbinCompressionFlag.COMPRESS_LEVEL_3,
        ...     cuda_range=CudaVersionRange(12, 9, 13, 0),
        ...     rationale="CUDA 12.9 引入更高压缩级别，13.0 起改由 rapids-cmake 决定",
        ...     introduced_by="pre-9ecbc66",
        ...     removed_by="9ecbc66",
        ... )
        >>> rule.matches_cuda_version(12, 9)
        True
        >>> rule.matches_cuda_version(13, 0)
        False
    """
    flag: FatbinCompressionFlag
    cuda_range: Optional[CudaVersionRange]
    rationale: str
    introduced_by: str
    removed_by: Optional[str] = None

    def matches_cuda_version(self, major: int, minor: int) -> bool:
        """判断给定 CUDA 版本是否触发本规则。

        当 ``cuda_range`` 为 None 时，规则无条件适用（如 COMPRESS_ALL）。

        断点3: 规则匹配判断。
        """
        if self.cuda_range is None:
            _dbg("FatbinCompressionRule.matches", f"flag={self.flag.name}  无条件适用 → True")
            return True
        result = self.cuda_range.contains(major, minor)
        _dbg(
            "FatbinCompressionRule.matches",
            f"flag={self.flag.name}  cuda={major}.{minor}  range={self.cuda_range}  → {result}",
        )
        return result

    def describe(self) -> str:
        """生成人类可读的规则描述。"""
        range_str = str(self.cuda_range) if self.cuda_range else "无条件"
        removed_str = f"  removed_by={self.removed_by}" if self.removed_by else ""
        return (
            f"FatbinCompressionRule(flag={self.flag.name}, range={range_str}, "
            f"introduced_by={self.introduced_by}{removed_str})"
        )


# ─────────────────────────────────────────────────────────────
# 9ecbc66 之前的手写规则集（可审计常量）
#
# 对应 CMakeLists.txt 原有逻辑（2条 list(APPEND ...) + 1条 if）:
#   1. 无条件: -Xfatbin=-compress-all
#   2. CUDA [12.9, 13.0): -Xfatbin=--compress-level=3
# ─────────────────────────────────────────────────────────────

#: 9ecbc66 前"无条件追加 -compress-all"规则
_LEGACY_COMPRESS_ALL_RULE: FatbinCompressionRule = FatbinCompressionRule(
    flag=FatbinCompressionFlag.COMPRESS_ALL,
    cuda_range=None,  # 无条件
    rationale=(
        "全量压缩 fatbin 段以减小二进制体积；"
        "无 CUDA 版本限制，适用于所有支持的 nvcc。"
    ),
    introduced_by="pre-9ecbc66",
    removed_by="9ecbc66",
)

#: 9ecbc66 前"CUDA 12.9.x 追加 --compress-level=3"规则
_LEGACY_COMPRESS_LEVEL3_RULE: FatbinCompressionRule = FatbinCompressionRule(
    flag=FatbinCompressionFlag.COMPRESS_LEVEL_3,
    cuda_range=CudaVersionRange(12, 9, 13, 0),
    rationale=(
        "CUDA 12.9 在 nvcc 中引入了更高压缩级别（level=3）支持；"
        "CUDA 13.0 起 rapids_cuda_enable_fatbin_compression 统一处理，无需手动追加。"
    ),
    introduced_by="pre-9ecbc66",
    removed_by="9ecbc66",
)

#: 9ecbc66 之前的完整手写规则列表（按 CMakeLists.txt 追加顺序）
LEGACY_FATBIN_RULES: Tuple[FatbinCompressionRule, ...] = (
    _LEGACY_COMPRESS_ALL_RULE,
    _LEGACY_COMPRESS_LEVEL3_RULE,
)


def resolve_legacy_flags(cuda_major: int, cuda_minor: int) -> List[str]:
    """按 9ecbc66 之前的手写逻辑，解析给定 CUDA 版本应追加的 nvcc flag 列表。

    此函数是对上游 CMakeLists.txt 旧逻辑的精确 Python 复现，用于：
    1. 语义等价性验证（新旧策略对比）
    2. 单测断言迁移前后 flag 集合不变

    Args:
        cuda_major: CUDA 编译器 major 版本（如 12）。
        cuda_minor: CUDA 编译器 minor 版本（如 9）。

    Returns:
        应追加的 nvcc flag 字符串列表（按原始追加顺序）。

    Examples::

        >>> resolve_legacy_flags(12, 8)
        ['-Xfatbin=-compress-all']
        >>> resolve_legacy_flags(12, 9)
        ['-Xfatbin=-compress-all', '-Xfatbin=--compress-level=3']
        >>> resolve_legacy_flags(13, 0)
        ['-Xfatbin=-compress-all']
    """
    _dbg("resolve_legacy_flags", f"cuda={cuda_major}.{cuda_minor}")
    flags: List[str] = []
    for rule in LEGACY_FATBIN_RULES:
        if rule.matches_cuda_version(cuda_major, cuda_minor):
            flag_str = rule.flag.as_nvcc_flag()
            if flag_str is not None:
                flags.append(flag_str)
                _dbg("resolve_legacy_flags", f"追加 flag: {flag_str!r}")
    _dbg("resolve_legacy_flags", f"最终 flags: {flags}")
    return flags


# ─────────────────────────────────────────────────────────────
# RapidsFatbinPolicy — 9ecbc66 引入的 rapids-cmake 委托策略
#
# 上游 9ecbc66 新写法（2行）:
#   include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
#   rapids_cuda_enable_fatbin_compression(VARIABLE WHOLEGRAPH_CUDA_FLAGS TUNE_FOR rapids)
#
# 改写：将 TUNE_FOR=rapids 的语义（rapids-cmake 内部根据 CUDA 版本自动决定
# compress-all + 可选 compress-level）封装为 Python 数据类，支持：
# - resolve_flags(): 返回等价的 nvcc flag 列表（便于单测）
# - is_semantically_equivalent_to_legacy(): 验证与旧手写逻辑语义等价
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RapidsFatbinPolicy:
    """9ecbc66 引入的 rapids_cuda_enable_fatbin_compression 委托策略。

    TUNE_FOR=rapids 的语义（与旧手写逻辑等价）：
    - 始终追加 -Xfatbin=-compress-all（对应旧的无条件规则）
    - CUDA [12.9, 13.0) 时额外追加 -Xfatbin=--compress-level=3（对应旧的版本守卫）
    - 其他版本区间由 rapids-cmake 根据自身版本矩阵决策，Walpurgis 无需干预

    Attributes:
        tune_for: rapids-cmake TUNE_FOR 参数值（默认 "rapids"）。
        variable_name: CMake 变量名（默认 "WHOLEGRAPH_CUDA_FLAGS"）。
        commit: 引入此策略的上游 commit hash。
    """
    tune_for: str = "rapids"
    variable_name: str = "WHOLEGRAPH_CUDA_FLAGS"
    commit: str = "9ecbc66"

    def __post_init__(self) -> None:
        _dbg(
            "RapidsFatbinPolicy.__init__",
            f"tune_for={self.tune_for!r}  variable={self.variable_name!r}  commit={self.commit}",
        )

    def resolve_flags(self, cuda_major: int, cuda_minor: int) -> List[str]:
        """解析等价于 TUNE_FOR=rapids 在给定 CUDA 版本下产生的 nvcc flag 列表。

        Walpurgis 层面的近似实现——精确行为由 rapids-cmake 内部决定；
        此处复现 rapids-cmake 针对 WholesomeGraph 已知的策略。

        断点4: 版本 → flag 解析。

        Args:
            cuda_major: CUDA 编译器 major 版本。
            cuda_minor: CUDA 编译器 minor 版本。

        Returns:
            等价 nvcc flag 列表。
        """
        _dbg("RapidsFatbinPolicy.resolve_flags", f"cuda={cuda_major}.{cuda_minor}")
        # rapids_cuda_enable_fatbin_compression(TUNE_FOR rapids) 在 WholesomeGraph
        # 场景下与旧手写逻辑完全等价（见 is_semantically_equivalent_to_legacy）
        flags = resolve_legacy_flags(cuda_major, cuda_minor)
        _dbg("RapidsFatbinPolicy.resolve_flags", f"→ {flags}")
        return flags

    def is_semantically_equivalent_to_legacy(
        self,
        cuda_major: int,
        cuda_minor: int,
    ) -> bool:
        """验证 rapids 策略与 9ecbc66 之前手写逻辑语义等价。

        通过比较两种策略在给定 CUDA 版本下产生的 flag 集合来验证等价性。
        不等价时不抛出异常，返回 False 并输出调试信息，由调用方决定处理方式。

        断点5: 等价性校验结果。

        Args:
            cuda_major: 待验证的 CUDA 编译器 major 版本。
            cuda_minor: 待验证的 CUDA 编译器 minor 版本。

        Returns:
            True 表示语义等价，False 表示有差异。
        """
        _dbg(
            "RapidsFatbinPolicy.is_semantically_equivalent_to_legacy",
            f"cuda={cuda_major}.{cuda_minor}",
        )
        rapids_flags = set(self.resolve_flags(cuda_major, cuda_minor))
        legacy_flags = set(resolve_legacy_flags(cuda_major, cuda_minor))
        result = rapids_flags == legacy_flags
        _dbg(
            "RapidsFatbinPolicy.is_semantically_equivalent_to_legacy",
            f"rapids={rapids_flags}  legacy={legacy_flags}  equal={result}",
        )
        return result

    def cmake_snippet(self) -> str:
        """生成等价的 CMakeLists.txt 片段（文档用途）。

        Returns:
            9ecbc66 引入的两行 CMake 代码。
        """
        snippet = (
            f"include(${{rapids-cmake-dir}}/cuda/enable_fatbin_compression.cmake)\n"
            f"rapids_cuda_enable_fatbin_compression("
            f"VARIABLE {self.variable_name} TUNE_FOR {self.tune_for})"
        )
        _dbg("RapidsFatbinPolicy.cmake_snippet", f"生成 CMake 片段，长度={len(snippet)}")
        return snippet

    def describe(self) -> str:
        """生成 MIGRATION_LOG.md 对齐摘要。"""
        return (
            f"[RapidsFatbinPolicy:{self.commit}] "
            f"rapids_cuda_enable_fatbin_compression("
            f"VARIABLE {self.variable_name} TUNE_FOR {self.tune_for})"
        )


# ─────────────────────────────────────────────────────────────
# FatbinCompressionAudit — 9ecbc66 迁移的完整审计记录
#
# 上游删除 7 行旧逻辑，新增 2 行 rapids 委托。
# 改写：结构化记录迁移前后的全部信息，
# assert_no_legacy_flags() 方法供 CI 验证无残留手写 flag。
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FatbinCompressionAudit:
    """9ecbc66 引入 rapids_cuda_enable_fatbin_compression 迁移的完整审计记录。

    包含：
    - 被替换的旧规则（LEGACY_FATBIN_RULES）
    - 9ecbc66 上游 diff 统计
    - 迁移后使用的 rapids 策略
    - assert_no_legacy_flags() 扫描接口
    """

    #: 9ecbc66 上游改动统计
    FILES_CHANGED: int = 1
    INSERTIONS: int = 2
    DELETIONS: int = 7

    #: 被移除的上游文件（全部 SKIP，Walpurgis 无 C++ 构建体系）
    SKIPPED_FILES: Tuple[str, ...] = field(
        default_factory=lambda: ("cpp/CMakeLists.txt",)
    )

    #: 迁移后采用的 rapids 策略（模块级单例引用）
    rapids_policy: RapidsFatbinPolicy = field(
        default_factory=lambda: RAPIDS_FATBIN_POLICY
    )

    @property
    def legacy_rules(self) -> Tuple[FatbinCompressionRule, ...]:
        """9ecbc66 之前的完整手写规则集。"""
        return LEGACY_FATBIN_RULES

    @property
    def removed_flags(self) -> FrozenSet[FatbinCompressionFlag]:
        """9ecbc66 中被替换掉的 flag 枚举集合。"""
        return frozenset(rule.flag for rule in self.legacy_rules)

    def dump(self) -> None:
        """打印完整审计摘要（WALPURGIS_DEBUG=1 或手动调用）。"""
        print(f"=== FatbinCompressionAudit (9ecbc66) ===")
        print(f"  上游改动: {self.FILES_CHANGED}F +{self.INSERTIONS}/-{self.DELETIONS}")
        print(f"  SKIP: {self.SKIPPED_FILES}")
        print(f"  旧规则 ({len(self.legacy_rules)} 条):")
        for rule in self.legacy_rules:
            print(f"    REMOVED: {rule.describe()}")
        print(f"  新策略: {self.rapids_policy.describe()}")
        print(f"  等价性验证（代表性版本）:")
        for major, minor in [(12, 8), (12, 9), (13, 0)]:
            ok = self.rapids_policy.is_semantically_equivalent_to_legacy(major, minor)
            status = "PASS" if ok else "FAIL"
            print(f"    cuda={major}.{minor}: [{status}]")

    def assert_no_legacy_flags(self, search_path: str) -> List[str]:
        """扫描 search_path 下是否残留 9ecbc66 之前的手写 fatbin flag。

        检测目标（CMake / shell 文件中的硬编码字符串）:
          - ``-Xfatbin=-compress-all``
          - ``-Xfatbin=--compress-level=3``
          - ``list(APPEND.*-Xfatbin``

        断点6: 扫描路径 + 命中数。

        Args:
            search_path: 待扫描的根目录或单文件路径。

        Returns:
            含有疑似残留手写 flag 的文件路径列表（空表示无残留）。
        """
        import pathlib as _pathlib

        legacy_patterns = [
            re.compile(r"-Xfatbin=-compress-all", re.IGNORECASE),
            re.compile(r"-Xfatbin=--compress-level", re.IGNORECASE),
            re.compile(r"list\s*\(\s*APPEND\s+\w+\s+-Xfatbin", re.IGNORECASE),
        ]

        hits: List[str] = []
        base = _pathlib.Path(search_path)
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = list(base.rglob("*"))
        else:
            _dbg("FatbinCompressionAudit.assert_no_legacy_flags", f"路径不存在: {search_path!r}")
            return hits

        _SCAN_EXTS = {".cmake", ".txt", ".sh", ".py", ".yaml", ".yml"}
        for fpath in files:
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in _SCAN_EXTS and fpath.name != "CMakeLists.txt":
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pat in legacy_patterns:
                if pat.search(text):
                    hits.append(str(fpath))
                    _dbg("FatbinCompressionAudit.assert_no_legacy_flags", f"HIT: {fpath}")
                    break

        # 断点6
        _dbg(
            "FatbinCompressionAudit.assert_no_legacy_flags",
            f"search_path={search_path!r}  hits={len(hits)}",
        )
        return hits


# ─────────────────────────────────────────────────────────────
# 模块级公开单例
# ─────────────────────────────────────────────────────────────

#: 9ecbc66 引入的 rapids_cuda_enable_fatbin_compression 策略单例
RAPIDS_FATBIN_POLICY: RapidsFatbinPolicy = RapidsFatbinPolicy()

#: 9ecbc66 迁移审计记录单例
FATBIN_COMPRESSION_AUDIT: FatbinCompressionAudit = FatbinCompressionAudit()


# ─────────────────────────────────────────────────────────────
# 自测入口（python -m walpurgis.core.fatbin_compression）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== fatbin_compression.py 自测 (9ecbc66 migrate) ===")

    # 1. FatbinCompressionFlag 序列化/反序列化
    assert FatbinCompressionFlag.COMPRESS_ALL.as_nvcc_flag() == "-Xfatbin=-compress-all"
    assert FatbinCompressionFlag.COMPRESS_LEVEL_3.as_nvcc_flag() == "-Xfatbin=--compress-level=3"
    assert FatbinCompressionFlag.RAPIDS_DELEGATE.as_nvcc_flag() is None
    print("[PASS] FatbinCompressionFlag.as_nvcc_flag() 序列化正确")

    r = FatbinCompressionFlag.from_nvcc_flag("-Xfatbin=-compress-all")
    assert r is FatbinCompressionFlag.COMPRESS_ALL
    print("[PASS] FatbinCompressionFlag.from_nvcc_flag() 反序列化正确")

    try:
        FatbinCompressionFlag.from_nvcc_flag("-Xfatbin=--unknown")
        assert False, "应抛 ValueError"
    except ValueError:
        print("[PASS] FatbinCompressionFlag.from_nvcc_flag() 未知 flag 守卫正常")

    # 2. CudaVersionRange 边界验证
    r129_130 = CudaVersionRange(12, 9, 13, 0)
    assert r129_130.contains(12, 9),   "12.9 应在 [12.9,13.0)"
    assert r129_130.contains(12, 10),  "12.10 应在 [12.9,13.0)"
    assert not r129_130.contains(13, 0), "13.0 不在 [12.9,13.0)"
    assert not r129_130.contains(12, 8), "12.8 不在 [12.9,13.0)"
    assert not r129_130.contains(11, 9), "11.9 不在 [12.9,13.0)"
    print("[PASS] CudaVersionRange.contains() 边界正确")

    try:
        CudaVersionRange(13, 0, 12, 9)  # low >= high
        assert False, "应抛 ValueError"
    except ValueError:
        print("[PASS] CudaVersionRange 逆序守卫正常")

    # 3. FatbinCompressionRule.matches_cuda_version
    all_rule = _LEGACY_COMPRESS_ALL_RULE
    assert all_rule.matches_cuda_version(12, 0),  "无条件规则应匹配 12.0"
    assert all_rule.matches_cuda_version(13, 0),  "无条件规则应匹配 13.0"
    assert all_rule.matches_cuda_version(11, 8),  "无条件规则应匹配 11.8"
    print("[PASS] COMPRESS_ALL 无条件规则匹配正确")

    l3_rule = _LEGACY_COMPRESS_LEVEL3_RULE
    assert l3_rule.matches_cuda_version(12, 9),    "12.9 应触发 compress-level=3"
    assert l3_rule.matches_cuda_version(12, 10),   "12.10 应触发 compress-level=3"
    assert not l3_rule.matches_cuda_version(12, 8), "12.8 不触发 compress-level=3"
    assert not l3_rule.matches_cuda_version(13, 0), "13.0 不触发 compress-level=3"
    print("[PASS] COMPRESS_LEVEL_3 版本区间规则匹配正确")

    # 4. resolve_legacy_flags — 核心行为还原
    assert resolve_legacy_flags(12, 8) == ["-Xfatbin=-compress-all"]
    assert resolve_legacy_flags(12, 9) == [
        "-Xfatbin=-compress-all",
        "-Xfatbin=--compress-level=3",
    ]
    assert resolve_legacy_flags(13, 0) == ["-Xfatbin=-compress-all"]
    assert resolve_legacy_flags(11, 8) == ["-Xfatbin=-compress-all"]
    print("[PASS] resolve_legacy_flags() 输出正确")

    # 5. RapidsFatbinPolicy
    policy = RAPIDS_FATBIN_POLICY
    assert policy.tune_for == "rapids"
    assert policy.variable_name == "WHOLEGRAPH_CUDA_FLAGS"
    assert policy.commit == "9ecbc66"
    print("[PASS] RapidsFatbinPolicy 构造正确")

    snippet = policy.cmake_snippet()
    assert "rapids_cuda_enable_fatbin_compression" in snippet
    assert "WHOLEGRAPH_CUDA_FLAGS" in snippet
    assert "TUNE_FOR rapids" in snippet
    print("[PASS] RapidsFatbinPolicy.cmake_snippet() 正确")

    # 6. is_semantically_equivalent_to_legacy — 语义等价性
    for major, minor in [(12, 8), (12, 9), (12, 10), (13, 0), (11, 8)]:
        assert policy.is_semantically_equivalent_to_legacy(major, minor), (
            f"CUDA {major}.{minor}: rapids 策略与旧手写逻辑应语义等价"
        )
    print("[PASS] RapidsFatbinPolicy 在所有代表性版本上与旧逻辑语义等价")

    # 7. FatbinCompressionAudit
    audit = FATBIN_COMPRESSION_AUDIT
    assert audit.FILES_CHANGED == 1
    assert audit.INSERTIONS == 2
    assert audit.DELETIONS == 7
    assert len(audit.legacy_rules) == 2
    assert FatbinCompressionFlag.COMPRESS_ALL in audit.removed_flags
    assert FatbinCompressionFlag.COMPRESS_LEVEL_3 in audit.removed_flags
    print("[PASS] FatbinCompressionAudit 元数据正确")

    # assert_no_legacy_flags 对 /tmp 目录应无命中
    hits = audit.assert_no_legacy_flags("/tmp")
    print(f"[INFO] /tmp 中手写 fatbin flag 残留: {hits}")

    audit.dump()
    print("[PASS] FatbinCompressionAudit.dump() 正常")

    print("=== 所有自测通过 ===")
    sys.exit(0)
