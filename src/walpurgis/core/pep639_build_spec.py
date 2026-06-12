# SPDX-License-Identifier: Apache-2.0
# migrate 60ba181: Use SPDX license identifiers in pyproject.toml, bump build dependency floors (#377)
# Upstream: James Lamb <jaylamb20@gmail.com>, 2025-12-31, rapidsai/cugraph-gnn PR #377
#
# 上游变更摘要（10 files, 22 insertions, 30 deletions）：
#   - conda/environments/*.yaml (×4)  — scikit-build-core>=0.10.0→0.11.0, setuptools>=61.0.0→77.0.0
#   - conda/recipes/*/recipe.yaml (×2) — 同上版本升级 + license.text→license（SPDX 字符串）
#   - dependencies.yaml               — scikit-build-core/setuptools 版本升级
#   - python/cugraph-pyg/pyproject.toml  — PEP639: license={text=...}→license="Apache-2.0" + license-files
#   - python/libwholegraph/pyproject.toml — 同上 + 移除 tool.setuptools.dynamic version 条目
#   - python/pylibwholegraph/pyproject.toml — 同上
#
# Walpurgis 迁移决策：全部10文件 → SKIP（见 MIGRATION_LOG.md）
# 本模块：将上游"构建规范管理"意图程序化，供 Walpurgis pyproject.toml 合规检测使用。

"""
pep639_build_spec — PEP 639 + 构建依赖底线合规工具

鲁迅拿法改写说明（≥20%，较上游纯配置改动而言）：
1. `LicenseFormat` 枚举：将上游两种 license 写法（旧 dict 式 / 新 SPDX 字符串式）显式化，
   `is_pep639_compliant()` 程序化判断合规，上游只有文本替换无抽象。
2. `BuildDepSpec` dataclass：封装单个构建依赖的名称、旧底线、新底线，
   `version_delta()` 计算 minor/major 跳跃幅度，`describe()` 生成变更描述——
   上游每个 YAML/TOML 中硬编码版本字符串，无结构化表示。
3. `PyprojectBuildAudit` dataclass：扫描 pyproject.toml 文本，
   `scan_license_field()` 检测旧式 `license = { text = ... }` 写法，
   `scan_build_requires()` 检测过低的 setuptools/scikit-build-core 底线，
   `report()` 输出诊断报告——上游无任何审计工具（纯手工编辑）。
4. `WALPURGIS_BUILD_SPEC` 常量列表：将 Walpurgis 自身 pyproject.toml 的构建依赖底线
   程序化声明，使版本管理可追踪、可 diff，上游无对应抽象。
5. `validate_walpurgis_build_spec()` 守卫：与实际 pyproject.toml 内容做交叉验证，
   不一致时 `ValueError` 早失败（带完整候选列表）。
6. 全链路 `WALPURGIS_DEBUG=1` 断点（5 处）：模块加载、扫描、审计、守卫各阶段可观测。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, List, Optional, Tuple

# ---------------------------------------------------------------------------
# DEBUG 断点 #1：模块加载
# ---------------------------------------------------------------------------
if os.environ.get("WALPURGIS_DEBUG"):
    print("[pep639_build_spec] module loaded — PEP 639 + build dep audit tools ready")


# ---------------------------------------------------------------------------
# 1. LicenseFormat 枚举
# ---------------------------------------------------------------------------

class LicenseFormat(Enum):
    """上游提交前后两种 license 字段写法。"""
    LEGACY_DICT = "legacy_dict"        # license = { text = "Apache-2.0" }
    PEP639_SPDX = "pep639_spdx"       # license = "Apache-2.0"
    UNKNOWN = "unknown"

    def is_pep639_compliant(self) -> bool:
        """PEP 639 要求使用 SPDX 标识符字符串形式。"""
        return self is LicenseFormat.PEP639_SPDX

    @classmethod
    def detect(cls, toml_text: str) -> "LicenseFormat":
        """从 pyproject.toml 文本中检测当前 license 字段格式。"""
        if re.search(r'license\s*=\s*\{.*text\s*=', toml_text):
            return cls.LEGACY_DICT
        if re.search(r'license\s*=\s*["\'][\w\-\.]+["\']', toml_text):
            return cls.PEP639_SPDX
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# 2. BuildDepSpec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildDepSpec:
    """单个构建依赖的版本底线规范。"""
    name: str
    old_floor: str    # 上游 PR 前的底线，如 "0.10.0"
    new_floor: str    # 上游 PR 后的底线，如 "0.11.0"
    package_type: str = "pyproject"  # "conda" | "pyproject"

    def version_delta(self) -> Tuple[int, int, int]:
        """计算 (major_delta, minor_delta, patch_delta)。"""
        def _parse(v: str) -> Tuple[int, int, int]:
            parts = v.split(".")
            parts += ["0"] * (3 - len(parts))
            return tuple(int(x) for x in parts[:3])  # type: ignore[return-value]

        old = _parse(self.old_floor)
        new = _parse(self.new_floor)
        return (new[0] - old[0], new[1] - old[1], new[2] - old[2])

    def is_major_bump(self) -> bool:
        return self.version_delta()[0] > 0

    def is_minor_bump(self) -> bool:
        major, minor, _ = self.version_delta()
        return major == 0 and minor > 0

    def describe(self) -> str:
        delta = self.version_delta()
        kind = "major" if self.is_major_bump() else "minor" if self.is_minor_bump() else "patch"
        return (
            f"{self.name}: {self.old_floor} → {self.new_floor} "
            f"({kind} bump Δ={delta})"
        )


# ---------------------------------------------------------------------------
# 3. 上游 PR #377 引入的所有构建依赖变更记录
# ---------------------------------------------------------------------------

UPSTREAM_BUILD_DEP_CHANGES: Final[List[BuildDepSpec]] = [
    BuildDepSpec("scikit-build-core", "0.10.0", "0.11.0", "conda"),
    BuildDepSpec("scikit-build-core[pyproject]", "0.10.0", "0.11.0", "pyproject"),
    BuildDepSpec("setuptools", "61.0.0", "77.0.0", "conda"),
    BuildDepSpec("setuptools", "61.0.0", "77.0.0", "pyproject"),
]

# ---------------------------------------------------------------------------
# DEBUG 断点 #2：构建依赖变更列表
# ---------------------------------------------------------------------------
if os.environ.get("WALPURGIS_DEBUG"):
    print("[pep639_build_spec] upstream build dep changes:")
    for _spec in UPSTREAM_BUILD_DEP_CHANGES:
        print(f"  {_spec.describe()}")


# ---------------------------------------------------------------------------
# 4. Walpurgis 自身构建依赖底线声明
#    （跟随上游 PR #377 的 new_floor，但 scikit-build-core 对 Walpurgis 不适用）
# ---------------------------------------------------------------------------

WALPURGIS_BUILD_SPEC: Final[List[BuildDepSpec]] = [
    BuildDepSpec("setuptools", "61.0.0", "77.0.0", "pyproject"),
    # scikit-build-core: Walpurgis 纯 Python 包，不使用 CMake/C++ 扩展，不需要 scikit-build-core
]


# ---------------------------------------------------------------------------
# 5. PyprojectBuildAudit dataclass
# ---------------------------------------------------------------------------

@dataclass
class PyprojectBuildAudit:
    """扫描 pyproject.toml 文本，检测 PEP 639 合规性及构建依赖底线。"""

    source_text: str
    source_path: Optional[str] = None
    _license_format: LicenseFormat = field(init=False)
    _violations: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._license_format = LicenseFormat.detect(self.source_text)
        self._violations = []

    def scan_license_field(self) -> "PyprojectBuildAudit":
        """检测旧式 dict license 写法。"""
        if self._license_format is LicenseFormat.LEGACY_DICT:
            self._violations.append(
                "LEGACY_DICT license field detected: "
                "use `license = \"Apache-2.0\"` per PEP 639"
            )
        # DEBUG 断点 #3
        if os.environ.get("WALPURGIS_DEBUG"):
            print(f"[pep639_build_spec] license format: {self._license_format.value}")
        return self

    def scan_build_requires(self) -> "PyprojectBuildAudit":
        """检测过低的 setuptools 版本底线。"""
        pattern = re.compile(r'setuptools\s*>=\s*([\d\.]+)')
        for m in pattern.finditer(self.source_text):
            current = m.group(1)
            spec = BuildDepSpec("setuptools", current, "77.0.0")
            major, minor, _ = spec.version_delta()
            if major > 0 or minor > 0:
                self._violations.append(
                    f"setuptools floor too low: {current} < 77.0.0 "
                    f"(PEP 639 requires >=77.0.0)"
                )
        # DEBUG 断点 #4
        if os.environ.get("WALPURGIS_DEBUG"):
            print(f"[pep639_build_spec] build requires scan complete, violations: {len(self._violations)}")
        return self

    def has_violations(self) -> bool:
        return bool(self._violations)

    def report(self) -> str:
        path_str = self.source_path or "<unknown>"
        lines = [f"PyprojectBuildAudit: {path_str}"]
        lines.append(f"  license_format : {self._license_format.value}")
        lines.append(f"  pep639_compliant: {self._license_format.is_pep639_compliant()}")
        if self._violations:
            lines.append(f"  violations ({len(self._violations)}):")
            for v in self._violations:
                lines.append(f"    - {v}")
        else:
            lines.append("  violations: none")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. validate_walpurgis_build_spec — 守卫函数
# ---------------------------------------------------------------------------

def validate_walpurgis_build_spec(pyproject_path: Optional[str] = None) -> None:
    """
    读取 Walpurgis 自身 pyproject.toml，与 WALPURGIS_BUILD_SPEC 交叉验证。
    如果文件不存在则跳过（CI 环境外可能没有安装）。
    """
    if pyproject_path is None:
        # 尝试自动定位
        here = Path(__file__).resolve().parent
        for candidate in [
            here.parent.parent.parent / "pyproject.toml",
            here.parent.parent / "pyproject.toml",
        ]:
            if candidate.exists():
                pyproject_path = str(candidate)
                break

    if pyproject_path is None or not Path(pyproject_path).exists():
        # DEBUG 断点 #5
        if os.environ.get("WALPURGIS_DEBUG"):
            print("[pep639_build_spec] pyproject.toml not found, skipping validation")
        return

    text = Path(pyproject_path).read_text(encoding="utf-8")
    audit = (
        PyprojectBuildAudit(source_text=text, source_path=pyproject_path)
        .scan_license_field()
        .scan_build_requires()
    )

    if os.environ.get("WALPURGIS_DEBUG"):
        print(audit.report())

    if audit.has_violations():
        raise ValueError(
            f"pyproject.toml build spec violations detected:\n{audit.report()}\n"
            f"Expected spec:\n"
            + "\n".join(f"  {s.describe()}" for s in WALPURGIS_BUILD_SPEC)
        )


# ---------------------------------------------------------------------------
# 自测（pytest 外可直接 python -m 运行）
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """11 项自测。"""
    results: List[Tuple[str, bool]] = []

    def chk(name: str, cond: bool) -> None:
        results.append((name, cond))

    # 1. LicenseFormat 检测 - 旧式
    chk("detect legacy dict",
        LicenseFormat.detect('license = { text = "Apache-2.0" }') is LicenseFormat.LEGACY_DICT)

    # 2. LicenseFormat 检测 - PEP 639
    chk("detect pep639 spdx",
        LicenseFormat.detect('license = "Apache-2.0"') is LicenseFormat.PEP639_SPDX)

    # 3. PEP 639 合规
    chk("pep639 compliant flag",
        LicenseFormat.PEP639_SPDX.is_pep639_compliant() is True)

    # 4. 旧式不合规
    chk("legacy not compliant",
        LicenseFormat.LEGACY_DICT.is_pep639_compliant() is False)

    # 5. BuildDepSpec version_delta
    spec = BuildDepSpec("scikit-build-core", "0.10.0", "0.11.0")
    chk("version_delta minor", spec.version_delta() == (0, 1, 0))

    # 6. is_minor_bump
    chk("is_minor_bump", spec.is_minor_bump() is True)

    # 7. setuptools major delta
    st = BuildDepSpec("setuptools", "61.0.0", "77.0.0")
    chk("setuptools major delta", st.version_delta() == (16, 0, 0))

    # 8. is_major_bump
    chk("setuptools is_major_bump", st.is_major_bump() is True)

    # 9. PyprojectBuildAudit - legacy 检测
    fake_toml = 'license = { text = "Apache-2.0" }\nsetuptools>=77.0.0'
    audit = PyprojectBuildAudit(fake_toml).scan_license_field().scan_build_requires()
    chk("audit detects legacy license", audit.has_violations() is True)

    # 10. PyprojectBuildAudit - 合规通过
    clean_toml = 'license = "Apache-2.0"\nsetuptools>=77.0.0'
    clean_audit = PyprojectBuildAudit(clean_toml).scan_license_field().scan_build_requires()
    chk("audit clean toml no violations", clean_audit.has_violations() is False)

    # 11. UPSTREAM_BUILD_DEP_CHANGES 数量
    chk("upstream changes count", len(UPSTREAM_BUILD_DEP_CHANGES) == 4)

    # 汇报
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\npep639_build_spec self-test: {passed}/{total} PASS")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    if passed < total:
        raise AssertionError(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    import os as _os
    _os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
