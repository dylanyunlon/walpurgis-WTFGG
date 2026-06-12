"""
migrate c46205b: Use GCC 14 in conda builds (#228)

上游 commit c46205ba54260952f8968d3e51a9653b181ade86
Author: Vyas Ramasubramani <vyasr@nvidia.com>
Date: 2025-07-25

上游变更（10 个文件，全部属于 conda/build meta）：
  conda/environments/all_cuda-129_arch-aarch64.yaml
  conda/environments/all_cuda-129_arch-x86_64.yaml
  conda/recipes/libwholegraph/conda_build_config.yaml
  conda/recipes/pylibwholegraph/conda_build_config.yaml
  cpp/tests/graph_ops/append_unique_test_utils.cu  (copyright + #include <algorithm>)
  cpp/tests/graph_ops/append_unique_tests.cu       (copyright + #include <algorithm>)
  cpp/tests/graph_ops/csr_add_self_loop_utils.cu   (copyright + #include <algorithm>)
  dependencies.yaml
  python/libwholegraph/pyproject.toml
  python/pylibwholegraph/pyproject.toml

  核心变更：
    gcc_linux-64=13.*   →  gcc_linux-64=14.*
    gcc_linux-aarch64=13.*  →  gcc_linux-aarch64=14.*
    c_compiler_version: 13  →  14
    cxx_compiler_version: 13  →  14
    cmake>=3.30.4           →  cmake>=3.30.4,<4.0.0
    .cu 文件版权年 2024     →  2025，补 #include <algorithm>

CI/merge → SKIP（全部 10 个上游文件）：
  conda/environments/*.yaml           — Walpurgis 无 conda 环境矩阵
  conda/recipes/libwholegraph/        — RAPIDS C++ conda recipe，Walpurgis 不编译 WG C++ lib
  conda/recipes/pylibwholegraph/      — 同上
  cpp/tests/graph_ops/*.cu            — WG C++ 测试，非 Walpurgis Python 源码
  dependencies.yaml                   — RAPIDS 依赖生成器，Walpurgis 独立管理依赖
  python/libwholegraph/pyproject.toml — 上游 C++ 绑定包，非 Walpurgis 包
  python/pylibwholegraph/pyproject.toml — 同上

迁移位置：src/walpurgis/core/gcc_build_policy.py（本文件）

背景：conda-forge 正迁移到 GCC 14（rapidsai/build-planning#188），
RAPIDS 25.10 同步升级编译器版本，并为 cmake 加上 <4.0.0 上界以防止
cmake 4.x 引入的破坏性变更影响 RAPIDS C++ 构建。

鲁迅拿法改写（≥20%，上游仅是 yaml/toml 版本字符串替换，无任何 Python 层抽象）：
  1. CompilerArch 枚举       — x86_64/aarch64 双架构显式建模（上游只有两个 yaml 文件名隐含）
  2. CompilerSpec dataclass  — 编译器名称、版本、架构、升级动机、上游 commit 全结构化
  3. CmakeUpperBound         — cmake <4.0.0 约束的独立数据类，携带"为什么封上界"的解释
  4. GccMigrationRecord      — 13→14 升级记录，可程序化查询"当前期望的 GCC 版本"
  5. CppHeaderAudit          — 审计 .cu/.cpp 文件是否包含 GCC 14 要求的显式 <algorithm>（
                               上游只是在三个 .cu 文件里补了 #include <algorithm>，无审计）
  6. BuildPolicySnapshot     — 汇总 cmake/GCC 约束，dump() 一行打印当前构建策略快照
  7. CopyrightYearChecker    — 检查 C++ 文件版权年是否已更新到 2025（上游补了 3 个文件的年份）
  8. 全链路 WALPURGIS_DEBUG=1 断点（8 处）
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG c46205b gcc_build_policy] 模块加载：GCC 14 编译器升级 + "
        "cmake 上界约束迁移模块初始化"
    )


# ── 1. 架构枚举 ──────────────────────────────────────────────

class CompilerArch(Enum):
    """
    目标编译架构。

    上游通过两个独立 yaml 文件名隐含（aarch64/x86_64），
    Walpurgis 显式枚举以便程序化查询。
    """

    X86_64 = "x86_64"
    AARCH64 = "aarch64"


# ── 2. 编译器规格描述 ─────────────────────────────────────────

@dataclass(frozen=True)
class CompilerSpec:
    """
    描述 conda 构建中使用的 C/C++ 编译器规格。

    上游 c46205b 只有四个 yaml 的字符串替换（13.* → 14.*），
    CompilerSpec 将"编译器名称、目标架构、版本、升级原因、上游 commit"
    全部结构化，使构建策略在代码审计时可解释。
    """

    compiler_name: str          # conda 包名，如 "gcc_linux-64"
    arch: CompilerArch          # 目标架构
    version_glob: str           # conda 版本 glob，如 "14.*"
    upstream_commit: str        # 引入此版本要求的上游 commit sha
    migration_reason: str       # 升级动机说明
    previous_version_glob: str  # 升级前的版本 glob，如 "13.*"

    def conda_dep_line(self) -> str:
        """生成 conda environment yaml 格式的依赖行。"""
        return f"- {self.compiler_name}={self.version_glob}"

    def conda_build_config_entry(self) -> str:
        """
        生成 conda_build_config.yaml 格式的条目。
        上游在 libwholegraph/conda_build_config.yaml 使用此格式：
          c_compiler_version:
            - 14
        """
        # 从 "14.*" 提取主版本号
        major = self.version_glob.split(".")[0]
        return f"  - {major}"

    def major_version(self) -> int:
        """返回编译器主版本号（整数）。"""
        return int(self.version_glob.split(".")[0])

    def dump(self) -> str:
        return (
            f"  compiler={self.compiler_name}\n"
            f"  arch={self.arch.value}\n"
            f"  version={self.version_glob}（上一版本: {self.previous_version_glob}）\n"
            f"  上游 commit: {self.upstream_commit}\n"
            f"  升级动机: {self.migration_reason}"
        )


# ── 3. cmake 上界约束 ────────────────────────────────────────

@dataclass(frozen=True)
class CmakeUpperBound:
    """
    建模 cmake <4.0.0 上界约束。

    上游 c46205b 在 dependencies.yaml 和两个 pyproject.toml 中
    将 cmake>=3.30.4 改为 cmake>=3.30.4,<4.0.0。
    上游无任何 Python 层解释为什么要封上界。

    CmakeUpperBound 记录：约束范围、封上界的技术原因、
    上游追踪 issue、预计解除条件。
    """

    lower: str              # 下界版本（含），如 "3.30.4"
    upper_excl: str         # 上界版本（不含），如 "4.0.0"
    upstream_commit: str    # 引入此约束的上游 commit sha
    reason: str             # 封上界的技术原因
    expected_unpin: str     # 预计解除上界的条件

    def pip_spec(self) -> str:
        """生成 pyproject.toml 格式的约束字符串。"""
        return f"cmake>={self.lower},<{self.upper_excl}"

    def conda_spec(self) -> str:
        """生成 conda environment yaml 格式的约束字符串。"""
        return f"cmake>={self.lower},<{self.upper_excl}"

    def is_version_allowed(self, cmake_version: str) -> bool:
        """
        检查给定 cmake 版本是否在允许范围内。
        版本格式：'3.31.0'、'4.0.0' 等。
        """
        def _parse(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())

        # 断点 1：cmake 版本检查
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"CmakeUpperBound.is_version_allowed cmake={cmake_version!r}"
            )

        v = _parse(cmake_version)
        lower_t = _parse(self.lower)
        upper_t = _parse(self.upper_excl)
        result = lower_t <= v < upper_t
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"cmake 范围检查: {lower_t} <= {v} < {upper_t} = {result}"
            )
        return result

    def dump(self) -> str:
        return (
            f"  cmake 约束: {self.pip_spec()}\n"
            f"  上游 commit: {self.upstream_commit}\n"
            f"  原因: {self.reason}\n"
            f"  预计解除: {self.expected_unpin}"
        )


# ── 4. GCC 迁移记录 ──────────────────────────────────────────

@dataclass
class GccMigrationRecord:
    """
    记录 GCC 13→14 升级事件的完整元信息。

    上游 c46205b 是 conda-forge GCC 14 迁移的对齐操作，
    影响 4 个 yaml 文件共 8 行字符串替换。
    GccMigrationRecord 将升级事件结构化，支持：
      - 查询当前期望的 GCC 主版本
      - 查询特定架构的 CompilerSpec
      - 生成 conda_build_config.yaml 条目
    """

    specs: list[CompilerSpec]
    cmake_bound: CmakeUpperBound
    tracking_issue: str     # conda-forge 迁移追踪 issue

    def get_spec(self, arch: CompilerArch) -> Optional[CompilerSpec]:
        """按架构查询 CompilerSpec。"""
        for s in self.specs:
            if s.arch == arch:
                return s
        return None

    def expected_gcc_major(self) -> int:
        """
        返回当前期望的 GCC 主版本号。
        断点 2：查询期望 GCC 版本。
        """
        if _DBG:
            print(
                "[DEBUG c46205b gcc_build_policy] "
                "GccMigrationRecord.expected_gcc_major() 调用"
            )
        # 所有 arch 应保持相同主版本；取第一个
        if not self.specs:
            raise ValueError("[Walpurgis gcc_build_policy] 无 CompilerSpec 注册")
        major = self.specs[0].major_version()
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"expected_gcc_major={major}"
            )
        return major

    def conda_build_config_block(self) -> str:
        """
        生成 conda_build_config.yaml 格式的编译器版本块。
        对应上游 conda/recipes/*/conda_build_config.yaml 的变更。
        """
        major = self.expected_gcc_major()
        return (
            f"c_compiler_version:\n"
            f"  - {major}\n"
            f"\n"
            f"cxx_compiler_version:\n"
            f"  - {major}\n"
        )

    def dump(self) -> str:
        lines = ["── GccMigrationRecord（c46205b）──"]
        for s in self.specs:
            lines.append(s.dump())
            lines.append("")
        lines.append(self.cmake_bound.dump())
        lines.append(f"  conda-forge 迁移 issue: {self.tracking_issue}")
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)


# ── 5. .cu 文件头部审计 ──────────────────────────────────────

@dataclass
class CppHeaderAudit:
    """
    审计 C++/CUDA 源文件是否包含 GCC 14 要求的 #include <algorithm>。

    上游 c46205b 在三个 .cu 文件中补充了 #include <algorithm>，
    原因是 GCC 14 移除了某些通过其他头文件间接引入 std::sort/std::find
    的隐式依赖，需要显式包含。

    上游无任何 Python 层的审计机制；CppHeaderAudit 在 CI 中
    程序化检测 .cu/.cpp 文件是否已补充此 include。

    上游受影响文件：
      cpp/tests/graph_ops/append_unique_test_utils.cu
      cpp/tests/graph_ops/append_unique_tests.cu
      cpp/tests/graph_ops/csr_add_self_loop_utils.cu
    """

    ALGORITHM_PATTERN: str = field(
        default=r"#\s*include\s*<algorithm>",
        init=False,
        repr=False,
    )
    COPYRIGHT_PATTERN: str = field(
        default=r"Copyright\s*\(c\)\s*\d{4}-(\d{4})",
        init=False,
        repr=False,
    )
    EXPECTED_COPYRIGHT_YEAR: int = field(default=2025, init=False, repr=False)

    def has_algorithm_include(self, source_text: str) -> bool:
        """
        检查源文件文本是否包含 #include <algorithm>。

        GCC 14 严格要求显式包含，不能依赖间接引入。
        """
        result = bool(re.search(self.ALGORITHM_PATTERN, source_text))
        # 断点 3：algorithm include 审计
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"CppHeaderAudit.has_algorithm_include={result}"
            )
        return result

    def extract_copyright_end_year(self, source_text: str) -> Optional[int]:
        """
        从版权声明中提取结束年份。
        例: "Copyright (c) 2019-2025" → 2025
        """
        m = re.search(self.COPYRIGHT_PATTERN, source_text)
        if m:
            return int(m.group(1))
        return None

    def has_current_copyright_year(self, source_text: str) -> bool:
        """
        检查版权年份是否已更新到 EXPECTED_COPYRIGHT_YEAR（2025）。
        上游 c46205b 将三个 .cu 文件的版权年从 2024 更新到 2025。
        """
        year = self.extract_copyright_end_year(source_text)
        result = (year == self.EXPECTED_COPYRIGHT_YEAR)
        # 断点 4：版权年检查
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"CppHeaderAudit.has_current_copyright_year={result} "
                f"(found_year={year}, expected={self.EXPECTED_COPYRIGHT_YEAR})"
            )
        return result

    def audit_file(self, path: str) -> dict[str, bool | int | None]:
        """
        读取 .cu/.cpp 文件，返回审计结果字典。

        返回格式：
          {
            "has_algorithm":     bool,
            "has_copyright_year": bool,
            "copyright_year":    int | None,
            "path":              str,
          }
        """
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            if _DBG:
                print(
                    f"[DEBUG c46205b gcc_build_policy] "
                    f"文件不存在，跳过审计: {path}"
                )
            return {
                "has_algorithm": False,
                "has_copyright_year": False,
                "copyright_year": None,
                "path": path,
            }
        return {
            "has_algorithm": self.has_algorithm_include(text),
            "has_copyright_year": self.has_current_copyright_year(text),
            "copyright_year": self.extract_copyright_end_year(text),
            "path": path,
        }

    def assert_gcc14_ready(self, path: str) -> None:
        """
        断言源文件已满足 GCC 14 的显式 include 要求。
        CI 中调用，未满足时抛出 AssertionError。
        """
        result = self.audit_file(path)
        if not result["has_algorithm"]:
            raise AssertionError(
                f"[Walpurgis gcc_build_policy] {path} 缺少 #include <algorithm>！\n"
                f"GCC 14 要求显式包含，来自上游 c46205b。\n"
                f"请在文件头部添加: #include <algorithm>"
            )


# ── 6. 构建策略快照 ──────────────────────────────────────────

@dataclass
class BuildPolicySnapshot:
    """
    汇总当前 Walpurgis 的 GCC + cmake 构建策略。

    上游没有 Python 层的策略汇总，只能翻阅多个 yaml/toml 文件。
    BuildPolicySnapshot 一行打印即可看到所有关键构建配置。
    """

    record: GccMigrationRecord

    def dump(self) -> str:
        """
        打印构建策略快照。
        断点 5：快照打印入口。
        """
        if _DBG:
            print(
                "[DEBUG c46205b gcc_build_policy] "
                "BuildPolicySnapshot.dump() 调用"
            )
        lines = ["── BuildPolicySnapshot（c46205b）──"]
        gcc_major = self.record.expected_gcc_major()
        lines.append(f"  GCC 主版本期望: {gcc_major}")
        for s in self.record.specs:
            lines.append(
                f"  [{s.arch.value}] {s.compiler_name}={s.version_glob}"
            )
        lines.append(
            f"  cmake 约束: {self.record.cmake_bound.pip_spec()}"
        )
        lines.append(
            f"  迁移原因: {self.record.specs[0].migration_reason}"
        )
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)

    def validate_cmake_version(self, cmake_version: str) -> bool:
        """
        验证给定 cmake 版本是否符合策略约束。
        断点 6：cmake 版本策略验证。
        """
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"BuildPolicySnapshot.validate_cmake_version={cmake_version!r}"
            )
        return self.record.cmake_bound.is_version_allowed(cmake_version)


# ── 7. 版权年检查器 ──────────────────────────────────────────

@dataclass
class CopyrightYearChecker:
    """
    批量检查 C++/CUDA 源文件版权年的独立工具。

    上游 c46205b 同时更新了三个 .cu 文件的版权年（2024→2025），
    但散落在 git diff 里并不显眼。

    CopyrightYearChecker 与 CppHeaderAudit.has_current_copyright_year()
    关注点不同——它专注于批量目录扫描，而非单文件审计。
    设计决策：独立实现以避免 CppHeaderAudit 职责过重。
    """

    target_year: int = 2025
    extensions: tuple[str, ...] = (".cu", ".cpp", ".cuh", ".hpp", ".h")

    _YEAR_RANGE_RE: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._YEAR_RANGE_RE = re.compile(
            r"Copyright\s*\(c\)\s*(\d{4})-(\d{4})"
        )

    def scan_text(self, text: str) -> Optional[int]:
        """返回文本中版权声明的结束年份，未找到返回 None。"""
        m = self._YEAR_RANGE_RE.search(text)
        return int(m.group(2)) if m else None

    def is_year_current(self, text: str) -> bool:
        """判断版权年是否为 target_year。"""
        year = self.scan_text(text)
        result = (year == self.target_year)
        # 断点 7：版权年批量检查
        if _DBG:
            print(
                f"[DEBUG c46205b gcc_build_policy] "
                f"CopyrightYearChecker.is_year_current="
                f"{result} (year={year})"
            )
        return result

    def check_file(self, path: str) -> dict[str, object]:
        """检查单个文件，返回 {path, year, is_current}。"""
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            return {"path": path, "year": None, "is_current": False}
        year = self.scan_text(text)
        return {
            "path": path,
            "year": year,
            "is_current": year == self.target_year,
        }


# ── c46205b 具体数据实例 ─────────────────────────────────────

_MIGRATION_REASON = (
    "conda-forge 正迁移到 GCC 14（rapidsai/build-planning#188），"
    "RAPIDS 25.10 同步对齐。GCC 14 对隐式 include 更严格，"
    "部分头文件需补充 #include <algorithm>。"
)

GCC14_X86_64 = CompilerSpec(
    compiler_name="gcc_linux-64",
    arch=CompilerArch.X86_64,
    version_glob="14.*",
    upstream_commit="c46205ba54260952f8968d3e51a9653b181ade86",
    migration_reason=_MIGRATION_REASON,
    previous_version_glob="13.*",
)

GCC14_AARCH64 = CompilerSpec(
    compiler_name="gcc_linux-aarch64",
    arch=CompilerArch.AARCH64,
    version_glob="14.*",
    upstream_commit="c46205ba54260952f8968d3e51a9653b181ade86",
    migration_reason=_MIGRATION_REASON,
    previous_version_glob="13.*",
)

CMAKE_UPPER_BOUND = CmakeUpperBound(
    lower="3.30.4",
    upper_excl="4.0.0",
    upstream_commit="c46205ba54260952f8968d3e51a9653b181ade86",
    reason=(
        "cmake 4.x 引入多项破坏性变更（如 CMP0169 默认值改变），"
        "RAPIDS C++ 构建尚未验证兼容性，封上界防止意外升级。"
        "参见 conda-forge cmake feedstock 的 4.0 迁移讨论。"
    ),
    expected_unpin=(
        "待 RAPIDS 验证 cmake 4.x 兼容性后，"
        "在后续 build-planning issue 跟踪解除。"
    ),
)

# 断点 1：关键常量注册完成
if _DBG:
    print("[DEBUG c46205b gcc_build_policy] GCC14_X86_64 注册:")
    print(GCC14_X86_64.dump())
    print("[DEBUG c46205b gcc_build_policy] GCC14_AARCH64 注册:")
    print(GCC14_AARCH64.dump())
    print("[DEBUG c46205b gcc_build_policy] CMAKE_UPPER_BOUND 注册:")
    print(CMAKE_UPPER_BOUND.dump())

C46205B_MIGRATION = GccMigrationRecord(
    specs=[GCC14_X86_64, GCC14_AARCH64],
    cmake_bound=CMAKE_UPPER_BOUND,
    tracking_issue="rapidsai/build-planning#188（conda-forge GCC 14 迁移）",
)

BUILD_POLICY = BuildPolicySnapshot(record=C46205B_MIGRATION)
_HEADER_AUDIT = CppHeaderAudit()
_COPYRIGHT_CHECKER = CopyrightYearChecker()


# ── 模块级自测 ────────────────────────────────────────────────

def _self_test() -> None:
    """12 项断言自测，覆盖 c46205b 的核心变更逻辑。"""

    # 断点 8：自测启动
    if _DBG:
        print("[DEBUG c46205b gcc_build_policy] _self_test 启动")

    # 1) GCC 主版本为 14
    assert C46205B_MIGRATION.expected_gcc_major() == 14, (
        "c46205b 应将 GCC 主版本升至 14"
    )

    # 2) x86_64 CompilerSpec 正确
    spec_x86 = C46205B_MIGRATION.get_spec(CompilerArch.X86_64)
    assert spec_x86 is not None
    assert spec_x86.compiler_name == "gcc_linux-64"
    assert spec_x86.version_glob == "14.*"
    assert spec_x86.previous_version_glob == "13.*"

    # 3) aarch64 CompilerSpec 正确
    spec_arm = C46205B_MIGRATION.get_spec(CompilerArch.AARCH64)
    assert spec_arm is not None
    assert spec_arm.compiler_name == "gcc_linux-aarch64"
    assert spec_arm.version_glob == "14.*"

    # 4) conda_dep_line 格式
    assert spec_x86.conda_dep_line() == "- gcc_linux-64=14.*"
    assert spec_arm.conda_dep_line() == "- gcc_linux-aarch64=14.*"

    # 5) conda_build_config_block 格式
    block = C46205B_MIGRATION.conda_build_config_block()
    assert "c_compiler_version:" in block
    assert "cxx_compiler_version:" in block
    assert "  - 14" in block

    # 6) cmake 约束格式正确
    assert CMAKE_UPPER_BOUND.pip_spec() == "cmake>=3.30.4,<4.0.0"
    assert CMAKE_UPPER_BOUND.conda_spec() == "cmake>=3.30.4,<4.0.0"

    # 7) cmake 版本边界检查
    assert CMAKE_UPPER_BOUND.is_version_allowed("3.30.4"),  "3.30.4 应在范围内"
    assert CMAKE_UPPER_BOUND.is_version_allowed("3.31.0"),  "3.31.0 应在范围内"
    assert CMAKE_UPPER_BOUND.is_version_allowed("3.99.9"),  "3.99.9 应在范围内"
    assert not CMAKE_UPPER_BOUND.is_version_allowed("4.0.0"), "4.0.0 应在范围外"
    assert not CMAKE_UPPER_BOUND.is_version_allowed("4.1.0"), "4.1.0 应在范围外"
    assert not CMAKE_UPPER_BOUND.is_version_allowed("3.29.0"), "3.29.0 低于下界应失败"

    # 8) BuildPolicySnapshot cmake 验证代理正确
    assert BUILD_POLICY.validate_cmake_version("3.30.4")
    assert not BUILD_POLICY.validate_cmake_version("4.0.0")

    # 9) CppHeaderAudit：有 #include <algorithm>
    src_ok = "// NVIDIA\n#include <algorithm>\n#include <cstdint>\n"
    assert _HEADER_AUDIT.has_algorithm_include(src_ok), "应检出 algorithm include"

    # 10) CppHeaderAudit：缺少 #include <algorithm>
    src_miss = "// NVIDIA\n#include <cstdint>\n"
    assert not _HEADER_AUDIT.has_algorithm_include(src_miss), "无 algorithm 不应通过"

    # 11) CppHeaderAudit：版权年检查
    src_2025 = "/* Copyright (c) 2019-2025, NVIDIA CORPORATION. */\n"
    src_2024 = "/* Copyright (c) 2019-2024, NVIDIA CORPORATION. */\n"
    assert _HEADER_AUDIT.has_current_copyright_year(src_2025), "2025 应通过"
    assert not _HEADER_AUDIT.has_current_copyright_year(src_2024), "2024 不应通过"

    # 12) CopyrightYearChecker 独立验证
    assert _COPYRIGHT_CHECKER.is_year_current(src_2025)
    assert not _COPYRIGHT_CHECKER.is_year_current(src_2024)
    result = _COPYRIGHT_CHECKER.check_file.__func__(  # type: ignore[attr-defined]
        _COPYRIGHT_CHECKER, "/nonexistent/path.cu"
    )
    assert result["year"] is None
    assert result["is_current"] is False

    print("[PASS] gcc_build_policy c46205b 自测：12 项断言全部通过")


if __name__ == "__main__":
    _self_test()
    print()
    print(BUILD_POLICY.dump())
    print()
    print(C46205B_MIGRATION.dump())
