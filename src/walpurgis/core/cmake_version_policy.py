"""
cmake_version_policy.py
=======================
迁移自 cugraph-gnn upstream commit 525ca06
（Allow CMake 4, PR #307, Author: Bradley Dice）

上游变更摘要：
    7 个构建配置文件统一将 ``cmake>=3.30.4,<4.0.0`` 改为 ``cmake>=3.30.4``，
    解除对 CMake 4.x 的版本上界封锁。

迁移策略：
    上游改动全部属于 conda 环境矩阵 YAML、RAPIDS dependencies.yaml、
    以及 libwholegraph/pylibwholegraph 两个上游包的 pyproject.toml —— 无一
    属于 Walpurgis 源码逻辑层。全部 7 个文件均 CI/merge → SKIP（见 MIGRATION_LOG）。

    本模块以"鲁迅拿法"将上游的版本约束结论提升为：
      1. 结构化版本区间语义（``VersionBound`` dataclass）
      2. 可调用的约束验证器（``CMakeVersionPolicy``）
      3. 历史约束变更台账（``ConstraintRevision`` + ``REVISION_HISTORY``）
      4. 跨工具链兼容性矩阵（``ToolchainCompatMatrix``）
      5. 运行时 cmake 探测与版本报告（``probe_cmake_version()``）

    上游只有裸字符串 diff；本模块将其转化为可程序化查询与审计的知识结构。

鲁迅语录注脚：
    "不在沉默中爆发，就在沉默中灭亡。"
    ——上界约束的沉默封锁终于在 CMake 4 面前爆发；本模块记录这次解封的全貌。

调试：
    export WALPURGIS_DEBUG=1  激活全链路断点（pdb.set_trace / breakpoint）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 全局调试开关
# ---------------------------------------------------------------------------
_DEBUG: bool = os.environ.get("WALPURGIS_DEBUG", "").strip() == "1"


def _bp(label: str) -> None:
    """条件断点：WALPURGIS_DEBUG=1 时暂停，供 pdb 单步检查。"""
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG] 断点 ▶ {label}")
        breakpoint()  # noqa: T100  # 调试断点，生产环境由 _DEBUG 守卫


# ---------------------------------------------------------------------------
# 枚举：版本约束类型
# ---------------------------------------------------------------------------
class BoundKind(Enum):
    """版本区间边界类型。

    上游 diff 只呈现两种形态的字符串；本枚举将其语义化。
    """

    LOWER_INCLUSIVE = auto()   # >=x.y.z
    UPPER_EXCLUSIVE = auto()   # <x.y.z  （已被 525ca06 移除）
    UPPER_INCLUSIVE = auto()   # <=x.y.z （保留供未来扩展）
    EXACT = auto()             # ==x.y.z


# ---------------------------------------------------------------------------
# dataclass：单个版本边界
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VersionBound:
    """一条版本约束边界的结构化表示。

    Attributes
    ----------
    kind:
        边界类型（见 ``BoundKind``）。
    version:
        三元组 (major, minor, patch)。
    raw:
        原始字符串，如 ``">=3.30.4"`` 或 ``"<4.0.0"``。
    """

    kind: BoundKind
    version: Tuple[int, int, int]
    raw: str

    # ---- 断点 #1 ----
    def __post_init__(self) -> None:
        _bp("VersionBound.__post_init__ — 检查 version 三元组格式")
        if len(self.version) != 3:
            raise ValueError(
                f"version 必须为三元组 (major, minor, patch)，得到: {self.version!r}"
            )

    @staticmethod
    def parse(constraint: str) -> "VersionBound":
        """将约束字符串解析为 ``VersionBound``。

        Examples
        --------
        >>> VersionBound.parse(">=3.30.4")
        VersionBound(kind=BoundKind.LOWER_INCLUSIVE, version=(3, 30, 4), raw='>=3.30.4')
        >>> VersionBound.parse("<4.0.0")
        VersionBound(kind=BoundKind.UPPER_EXCLUSIVE, version=(4, 0, 0), raw='<4.0.0')
        """
        _bp(f"VersionBound.parse — 解析约束字符串 {constraint!r}")
        _PATTERN = re.compile(
            r"^(?P<op>>=|<=|>|<|==)\s*(?P<ver>\d+\.\d+\.\d+)$"
        )
        m = _PATTERN.match(constraint.strip())
        if not m:
            raise ValueError(f"无法解析约束字符串: {constraint!r}")
        op, ver_str = m.group("op"), m.group("ver")
        ver = tuple(int(x) for x in ver_str.split("."))  # type: ignore[assignment]
        _KIND_MAP = {
            ">=": BoundKind.LOWER_INCLUSIVE,
            "<=": BoundKind.UPPER_INCLUSIVE,
            "<":  BoundKind.UPPER_EXCLUSIVE,
            "==": BoundKind.EXACT,
        }
        kind = _KIND_MAP.get(op)
        if kind is None:
            raise ValueError(f"不支持的操作符: {op!r}")
        return VersionBound(kind=kind, version=ver, raw=constraint.strip())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# dataclass：版本区间（下界 + 可选上界）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VersionRange:
    """由一或两个 ``VersionBound`` 组成的版本区间。

    上游 before 状态：lower=``>=3.30.4``, upper=``<4.0.0``
    上游 after  状态：lower=``>=3.30.4``, upper=None
    """

    lower: VersionBound
    upper: Optional[VersionBound] = None

    def satisfies(self, version: Tuple[int, int, int]) -> bool:
        """检查给定版本是否满足本区间约束。

        Parameters
        ----------
        version:
            待检测版本三元组，如 ``(4, 0, 0)``。
        """
        _bp(f"VersionRange.satisfies — 检测版本 {version}")
        lo = self.lower
        if lo.kind == BoundKind.LOWER_INCLUSIVE and not (version >= lo.version):
            return False
        if self.upper is not None:
            up = self.upper
            if up.kind == BoundKind.UPPER_EXCLUSIVE and not (version < up.version):
                return False
            if up.kind == BoundKind.UPPER_INCLUSIVE and not (version <= up.version):
                return False
        return True

    def is_open_upper(self) -> bool:
        """是否已移除上界（即 525ca06 之后的状态）。"""
        return self.upper is None

    @staticmethod
    def parse(constraint: str) -> "VersionRange":
        """将逗号分隔的约束字符串解析为 ``VersionRange``。

        Examples
        --------
        >>> VersionRange.parse(">=3.30.4,<4.0.0").is_open_upper()
        False
        >>> VersionRange.parse(">=3.30.4").is_open_upper()
        True
        """
        _bp(f"VersionRange.parse — 输入: {constraint!r}")
        parts = [p.strip() for p in constraint.split(",") if p.strip()]
        bounds = [VersionBound.parse(p) for p in parts]
        lowers = [b for b in bounds if b.kind == BoundKind.LOWER_INCLUSIVE]
        uppers = [
            b for b in bounds
            if b.kind in (BoundKind.UPPER_EXCLUSIVE, BoundKind.UPPER_INCLUSIVE)
        ]
        if len(lowers) != 1:
            raise ValueError(f"期望恰好一个下界约束，得到: {[b.raw for b in lowers]}")
        return VersionRange(
            lower=lowers[0],
            upper=uppers[0] if uppers else None,
        )


# ---------------------------------------------------------------------------
# dataclass：约束变更记录（历史台账）
# ---------------------------------------------------------------------------
@dataclass
class ConstraintRevision:
    """单次约束变更的完整历史记录。

    上游只有 diff 减号/加号行；本 dataclass 将其提升为可查询的变更台账条目。

    Attributes
    ----------
    upstream_commit:
        触发本次变更的上游 commit SHA（短）。
    upstream_pr:
        对应 PR 编号，如 ``"#307"``。
    upstream_author:
        提交人 GitHub handle。
    source_file:
        上游被修改的文件路径（相对于 cugraph-gnn 根目录）。
    before:
        变更前的版本区间。
    after:
        变更后的版本区间。
    rationale:
        变更原因的人工摘要。
    """

    upstream_commit: str
    upstream_pr: str
    upstream_author: str
    source_file: str
    before: VersionRange
    after: VersionRange
    rationale: str

    def is_upper_bound_removal(self) -> bool:
        """检测本次变更是否为纯粹的上界移除（下界不变）。"""
        return (
            not self.before.is_open_upper()
            and self.after.is_open_upper()
            and self.before.lower == self.after.lower
        )


# ---------------------------------------------------------------------------
# 历史台账：REVISION_HISTORY
# 上游 7 个文件统一变更，此处结构化记录（上游无此汇总层）
# ---------------------------------------------------------------------------
_BEFORE_RANGE = VersionRange.parse(">=3.30.4,<4.0.0")
_AFTER_RANGE  = VersionRange.parse(">=3.30.4")

REVISION_HISTORY: List[ConstraintRevision] = [
    ConstraintRevision(
        upstream_commit="525ca06",
        upstream_pr="#307",
        upstream_author="bdice",
        source_file=src,
        before=_BEFORE_RANGE,
        after=_AFTER_RANGE,
        rationale="CMake 4.0 发布后 <4.0.0 上界导致 CI 构建失败；移除上界以允许 CMake 4.x",
    )
    for src in [
        "conda/environments/all_cuda-129_arch-aarch64.yaml",
        "conda/environments/all_cuda-129_arch-x86_64.yaml",
        "conda/environments/all_cuda-130_arch-aarch64.yaml",
        "conda/environments/all_cuda-130_arch-x86_64.yaml",
        "dependencies.yaml",
        "python/libwholegraph/pyproject.toml",
        "python/pylibwholegraph/pyproject.toml",
    ]
]


# ---------------------------------------------------------------------------
# dataclass：工具链兼容性矩阵条目
# 上游无此层；本模块原创，捕捉"为何 CMake 4 解封对 GNN 训练无实质影响"的推理
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolchainCompatEntry:
    """单条工具链兼容性声明。

    Attributes
    ----------
    cmake_range:
        适用的 cmake 版本区间。
    cuda_version:
        适用的 CUDA 版本字符串，如 ``"12.9"``。
    arch:
        目标架构，如 ``"x86_64"`` 或 ``"aarch64"``。
    walpurgis_affected:
        Walpurgis 训练流程是否受影响（本 commit 全部为 False）。
    note:
        补充说明。
    """

    cmake_range: VersionRange
    cuda_version: str
    arch: str
    walpurgis_affected: bool
    note: str


TOOLCHAIN_COMPAT_MATRIX: List[ToolchainCompatEntry] = [
    ToolchainCompatEntry(
        cmake_range=_AFTER_RANGE,
        cuda_version="12.9",
        arch="x86_64",
        walpurgis_affected=False,
        note="Walpurgis 以 Python wheel 分发，cmake 仅在 C 扩展编译时介入，训练推理路径不感知",
    ),
    ToolchainCompatEntry(
        cmake_range=_AFTER_RANGE,
        cuda_version="12.9",
        arch="aarch64",
        walpurgis_affected=False,
        note="同上；aarch64 arm server 场景 cmake 4 解封无训练侧回归风险",
    ),
    ToolchainCompatEntry(
        cmake_range=_AFTER_RANGE,
        cuda_version="13.0",
        arch="x86_64",
        walpurgis_affected=False,
        note="CUDA 13.0 尚未正式发布，保留占位；结论同 12.9",
    ),
    ToolchainCompatEntry(
        cmake_range=_AFTER_RANGE,
        cuda_version="13.0",
        arch="aarch64",
        walpurgis_affected=False,
        note="同上",
    ),
]


# ---------------------------------------------------------------------------
# CMakeVersionPolicy：核心策略对象
# ---------------------------------------------------------------------------
class CMakeVersionPolicy:
    """CMake 版本约束策略的可调用验证器。

    上游结论是"移除 `<4.0.0`"；本类将其转化为程序化可查询的策略对象，
    支持：版本合规检查、历史变更检索、工具链矩阵过滤。

    Usage
    -----
    >>> policy = CMakeVersionPolicy()
    >>> policy.check_version((4, 0, 0))
    True
    >>> policy.check_version((3, 29, 0))
    False
    """

    def __init__(self) -> None:
        _bp("CMakeVersionPolicy.__init__ — 初始化策略对象")
        self._current_range: VersionRange = _AFTER_RANGE
        self._history: List[ConstraintRevision] = list(REVISION_HISTORY)
        self._compat_matrix: List[ToolchainCompatEntry] = list(TOOLCHAIN_COMPAT_MATRIX)

    # ---- 断点 #2 ----
    def check_version(self, version: Tuple[int, int, int]) -> bool:
        """检查给定 cmake 版本是否满足当前策略。

        Parameters
        ----------
        version:
            cmake 版本三元组，如 ``(4, 0, 0)``。

        Returns
        -------
        bool
            满足返回 ``True``，否则 ``False``。
        """
        _bp(f"CMakeVersionPolicy.check_version — 待检版本: {version}")
        result = self._current_range.satisfies(version)
        if _DEBUG:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  [DEBUG] cmake {'.'.join(str(x) for x in version)} → {status}")
        return result

    def upper_bound_removals(self) -> List[ConstraintRevision]:
        """返回所有"纯粹移除上界"的历史变更记录。"""
        _bp("CMakeVersionPolicy.upper_bound_removals — 检索上界移除记录")
        return [r for r in self._history if r.is_upper_bound_removal()]

    def compat_entries_for(
        self, cuda_version: Optional[str] = None, arch: Optional[str] = None
    ) -> List[ToolchainCompatEntry]:
        """过滤工具链兼容性矩阵。

        Parameters
        ----------
        cuda_version:
            如果指定，只返回匹配 CUDA 版本的条目。
        arch:
            如果指定，只返回匹配架构的条目。
        """
        _bp(f"CMakeVersionPolicy.compat_entries_for — cuda={cuda_version}, arch={arch}")
        entries = self._compat_matrix
        if cuda_version is not None:
            entries = [e for e in entries if e.cuda_version == cuda_version]
        if arch is not None:
            entries = [e for e in entries if e.arch == arch]
        return entries

    def walpurgis_impact_summary(self) -> str:
        """生成 Walpurgis 影响摘要字符串（用于 MIGRATION_LOG 自动填充）。"""
        _bp("CMakeVersionPolicy.walpurgis_impact_summary — 生成摘要")
        affected = [e for e in self._compat_matrix if e.walpurgis_affected]
        if not affected:
            return (
                "Walpurgis 训练/推理流程：零影响。"
                "cmake 版本约束仅作用于 C 扩展编译阶段，"
                "Python GNN 训练路径不感知 cmake 版本。"
            )
        return f"受影响条目数: {len(affected)}"


# ---------------------------------------------------------------------------
# probe_cmake_version：运行时探测本机 cmake 版本
# ---------------------------------------------------------------------------
def probe_cmake_version() -> Optional[Tuple[int, int, int]]:
    """探测当前环境中 cmake 的实际版本。

    Returns
    -------
    Optional[Tuple[int, int, int]]
        解析成功返回三元组，cmake 未安装返回 ``None``。
    """
    _bp("probe_cmake_version — 调用 cmake --version")
    cmake_bin = shutil.which("cmake")
    if cmake_bin is None:
        if _DEBUG:
            print("  [DEBUG] cmake 未找到，返回 None")
        return None
    try:
        out = subprocess.check_output(
            [cmake_bin, "--version"], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    # 典型输出: "cmake version 3.30.4\n..."
    m = re.search(r"cmake version\s+(\d+)\.(\d+)\.(\d+)", out, re.IGNORECASE)
    if not m:
        return None
    ver = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if _DEBUG:
        print(f"  [DEBUG] 探测到 cmake 版本: {ver}")
    return ver


# ---------------------------------------------------------------------------
# verify_environment：一键环境验证
# ---------------------------------------------------------------------------
def verify_environment(strict: bool = False) -> bool:
    """验证当前环境中的 cmake 版本是否满足策略。

    Parameters
    ----------
    strict:
        ``True`` 时 cmake 未安装视为失败；``False`` 时跳过检测返回 ``True``。

    Returns
    -------
    bool
        满足策略返回 ``True``。
    """
    _bp("verify_environment — 开始环境验证")
    policy = CMakeVersionPolicy()
    ver = probe_cmake_version()
    if ver is None:
        if strict:
            print("[cmake_version_policy] WARN: cmake 未安装，strict 模式视为失败")
            return False
        print("[cmake_version_policy] INFO: cmake 未安装，宽松模式跳过检测")
        return True
    ok = policy.check_version(ver)
    ver_str = ".".join(str(x) for x in ver)
    status = "PASS" if ok else "FAIL"
    print(f"[cmake_version_policy] cmake {ver_str} 版本检测: {status}")
    return ok


# ---------------------------------------------------------------------------
# 自测（python -m walpurgis.core.cmake_version_policy）
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """内置自测，覆盖上游 7 条 diff 的核心语义。"""
    _bp("_self_test — 开始自测")
    tests_passed = 0
    tests_total = 0

    def _assert(cond: bool, msg: str) -> None:
        nonlocal tests_passed, tests_total
        tests_total += 1
        if cond:
            tests_passed += 1
            print(f"  [PASS] {msg}")
        else:
            print(f"  [FAIL] {msg}")

    print("=" * 60)
    print("cmake_version_policy 自测（迁移 525ca06）")
    print("=" * 60)

    # T1: 旧约束拒绝 CMake 4
    old_range = VersionRange.parse(">=3.30.4,<4.0.0")
    _assert(not old_range.satisfies((4, 0, 0)), "旧约束拒绝 cmake 4.0.0")

    # T2: 新约束接受 CMake 4
    new_range = VersionRange.parse(">=3.30.4")
    _assert(new_range.satisfies((4, 0, 0)), "新约束接受 cmake 4.0.0")

    # T3: 新约束接受 CMake 3.30.4（下界边值）
    _assert(new_range.satisfies((3, 30, 4)), "新约束接受 cmake 3.30.4（下界边值）")

    # T4: 新约束拒绝 CMake 3.29
    _assert(not new_range.satisfies((3, 29, 0)), "新约束拒绝 cmake 3.29.0（低于下界）")

    # T5: 上界开放检测
    _assert(new_range.is_open_upper(), "新约束 is_open_upper() == True")
    _assert(not old_range.is_open_upper(), "旧约束 is_open_upper() == False")

    # T6: REVISION_HISTORY 覆盖 7 个文件
    _assert(len(REVISION_HISTORY) == 7, "REVISION_HISTORY 包含 7 条记录")

    # T7: 全部记录均为上界移除
    policy = CMakeVersionPolicy()
    removals = policy.upper_bound_removals()
    _assert(len(removals) == 7, "upper_bound_removals() 返回 7 条")

    # T8: Walpurgis 无受影响条目
    affected = [e for e in TOOLCHAIN_COMPAT_MATRIX if e.walpurgis_affected]
    _assert(len(affected) == 0, "TOOLCHAIN_COMPAT_MATRIX 无 Walpurgis 受影响条目")

    # T9: check_version cmake 4.0.0 → True
    _assert(policy.check_version((4, 0, 0)), "CMakeVersionPolicy.check_version(4,0,0) == True")

    # T10: check_version cmake 3.29.0 → False
    _assert(not policy.check_version((3, 29, 0)), "CMakeVersionPolicy.check_version(3,29,0) == False")

    # T11: compat_entries_for 过滤
    entries_129 = policy.compat_entries_for(cuda_version="12.9")
    _assert(len(entries_129) == 2, "compat_entries_for(cuda_version='12.9') 返回 2 条")

    print("=" * 60)
    print(f"自测完成: {tests_passed}/{tests_total} PASS")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
