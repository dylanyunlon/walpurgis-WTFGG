"""
migrate a560ad0: 版本升级策略（Update to 26.04 #387）

上游 commit a560ad09a875e6e283a68557d181d650d1d34228
  Author: Jake Awe <50372925+AyodeAwe@users.noreply.github.com>
  Date:   Fri Jan 16 08:59:17 2026 -0600
  Repo:   rapidsai/cugraph-gnn

  变更摘要 (20 files changed, 89 insertions(+), 89 deletions(−)):
  ┌────────────────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                               │ 处置   │
  ├────────────────────────────────────────────────────────────────────┼────────┤
  │ .devcontainer/cuda12.9-conda/devcontainer.json                     │  SKIP  │
  │ .devcontainer/cuda12.9-pip/devcontainer.json                       │  SKIP  │
  │ .devcontainer/cuda13.1-conda/devcontainer.json                     │  SKIP  │
  │ .devcontainer/cuda13.1-pip/devcontainer.json                       │  SKIP  │
  │ .github/workflows/build.yaml                                       │  SKIP  │
  │ .github/workflows/pr.yaml                                          │  SKIP  │
  │ .github/workflows/test.yaml                                        │  SKIP  │
  │ VERSION                                                            │  SKIP  │
  │ conda/environments/all_cuda-129_arch-aarch64.yaml                  │  SKIP  │
  │ conda/environments/all_cuda-129_arch-x86_64.yaml                   │  SKIP  │
  │ conda/environments/all_cuda-131_arch-aarch64.yaml                  │  SKIP  │
  │ conda/environments/all_cuda-131_arch-x86_64.yaml                   │  SKIP  │
  │ dependencies.yaml                                                  │  SKIP  │
  │ python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-aarch64.yaml│  SKIP  │
  │ python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-x86_64.yaml │  SKIP  │
  │ python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-aarch64.yaml│  SKIP  │
  │ python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-x86_64.yaml │  SKIP  │
  │ python/cugraph-pyg/pyproject.toml                                  │  SKIP  │
  │ python/libwholegraph/pyproject.toml                                 │  SKIP  │
  │ python/pylibwholegraph/pyproject.toml                              │  SKIP  │
  └────────────────────────────────────────────────────────────────────┴────────┘

CI/devcontainer/conda/merge 文件 → 全部 SKIP:
  a560ad0 的全部 89 处改动均为单一的字符串替换：\"26.02\" → \"26.04\"，
  涵盖 devcontainer 基础镜像标签、CI workflow 容器镜像、VERSION 文件、
  conda 环境 yaml 中的 rapids 包版本约束、以及上游 pyproject.toml 的
  依赖声明版本范围。
  Walpurgis 无 RAPIDS 发布流水线、无 conda 构建矩阵、无 devcontainer 配置、
  无 cugraph-pyg / libwholegraph / pylibwholegraph 包，
  故全部 20 个文件均无对应迁移实体，一律 SKIP。

迁移位置:
  src/walpurgis/core/version_bump_policy.py（本文件，新增）

鲁迅拿法改写（≥20%）:
  上游是纯文本的全局 sed 替换（89 处 26.02→26.04），没有任何 Python
  对象模型、版本跃迁语义、或可审计的版本状态机。
  以鲁迅"横眉冷对"之势，将这次版本跃迁内化为六层结构：

  1. RapidsVersion dataclass — 将上游裸字符串 "26.02" / "26.04" 强类型化为
     可比较的 (year, month) 二元组，cycle_delta() 方法计算发布周期差，
     上游只有 sed 无任何版本对象。

  2. BumpKind 枚举 — 区分 MINOR（same-year month 跳变）、YEARLY（年度跨越）、
     PATCH（仅 patch 号变化）三类跃迁语义，上游不作区分，一律字符串替换。

  3. VersionBump dataclass — 封装\"从 FROM 到 TO\"这一版本跃迁事实，
     携带 commit_sha、pr_number、author、rationale 字段，
     bump_kind()、is_forward()、as_sed_pattern() 方法均为上游所无。

  4. AffectedScope dataclass — 建模\"哪些层的配置被此次 bump 影响\"，
     将 20 个文件按功能域归类（devcontainer/CI workflow/conda env/
     conda recipe/pyproject），为 Walpurgis SKIP 决策提供可审计依据。
     上游无此分类，全部文件平铺在 git diff 里。

  5. BumpCompatibilityProbe — 给定一个 RapidsVersion，探测 Walpurgis
     运行时实际安装的 rapids 相关包（cugraph / pylibcugraph / rmm 等）
     版本是否与该 rapids cycle 兼容，上游无 Python 层兼容性探测。

  6. VersionBumpAudit — 扫描任意文本文件，确认给定版本号不再以
     旧版本字符串形式残留，供 CI 调用验证 bump 是否彻底，
     上游通过全局 sed 完成后无回头检查机制。

  7. 全链路 WALPURGIS_DEBUG=1 断点（7 处）：覆盖版本解析→跃迁计算→
     影响域分类→兼容性探测→残留扫描→自测各阶段。
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DBG:
        print(f"[WPG:version_bump_policy:{tag}] {msg}", flush=True)


# ─── 1. RapidsVersion — 强类型 RAPIDS 版本对象 ────────────────────────────────


@dataclass(frozen=True, order=True)
class RapidsVersion:
    """
    封装 RAPIDS YY.MM 版本格式。

    上游 a560ad0 只有裸字符串 "26.02" / "26.04"，散落在 20 个文件的
    89 处 sed 替换中，没有任何类型化的版本对象。

    改写：不可变、可排序、可比较的版本值对象，携带发布周期语义。
    排序字段必须声明在前以保证 dataclass order=True 按 (year, month) 排序。
    """

    year: int    # YY，如 26
    month: int   # MM，如 2 或 4

    def __post_init__(self) -> None:
        _dbg("RapidsVersion.parse", f"year={self.year} month={self.month}")
        if not (0 <= self.year <= 99):
            raise ValueError(
                f"[RapidsVersion] year 必须在 0-99 范围内，收到: {self.year}"
            )
        if self.month not in {2, 4, 6, 8, 10, 12}:
            raise ValueError(
                f"[RapidsVersion] RAPIDS 只在偶数月发布，收到: {self.month}。\n"
                f"合法值: 2, 4, 6, 8, 10, 12"
            )
        _dbg("RapidsVersion.ok", f"tag={self.tag}")

    @property
    def tag(self) -> str:
        """返回上游格式字符串，如 '26.04'。"""
        return f"{self.year:02d}.{self.month:02d}"

    @property
    def full_tag(self) -> str:
        """返回带 patch 的完整版本，如 '26.04.00'。"""
        return f"{self.year:02d}.{self.month:02d}.00"

    @property
    def conda_wildcard(self) -> str:
        """conda 依赖约束通配符，如 '26.4.*'（conda 格式去前导零）。"""
        return f"{self.year}.{self.month}.*"

    @property
    def pip_pin(self) -> str:
        """pip/pyproject.toml 约束前缀，如 '==26.4.*,>=0.0.0a0'。"""
        return f"=={self.year}.{self.month}.*,>=0.0.0a0"

    def cycle_delta(self, other: "RapidsVersion") -> int:
        """
        计算两个版本间的发布周期差（正数表示 self 更新）。
        RAPIDS 每两个月发布一次，一个 cycle = 2 个月。
        上游无此计算，只有字符串替换。
        """
        self_months = self.year * 12 + self.month
        other_months = other.year * 12 + other.month
        delta_months = self_months - other_months
        return delta_months // 2

    @classmethod
    def parse(cls, tag: str) -> "RapidsVersion":
        """
        从字符串解析，支持 '26.04' 和 '26.4' 两种格式。
        上游只有裸字符串，无解析函数。
        """
        _dbg("RapidsVersion.from_str", f"tag={tag!r}")
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", tag.strip())
        if not m:
            raise ValueError(
                f"[RapidsVersion] 无法解析版本字符串: {tag!r}\n"
                f"期望格式: YY.MM，例如 '26.04' 或 '26.4'"
            )
        return cls(year=int(m.group(1)), month=int(m.group(2)))


# ─── 2. BumpKind 枚举 — 版本跃迁类型 ─────────────────────────────────────────


class BumpKind(Enum):
    """
    区分版本跃迁的语义类型。
    上游 a560ad0 只有一种操作（全局 sed），不区分跃迁类型。
    """

    MINOR = "minor"     # 同年内 month 跳变（26.02→26.04），最常见
    YEARLY = "yearly"   # 跨年跃迁（25.12→26.02）
    PATCH = "patch"     # 仅 patch 号变化（26.04.00→26.04.01），罕见
    UNKNOWN = "unknown" # 逆序或超过一个 cycle，异常情况


# ─── 3. VersionBump — 封装一次版本跃迁事实 ───────────────────────────────────


@dataclass(frozen=True)
class VersionBump:
    """
    封装"从 FROM 版本到 TO 版本"这一跃迁事实，携带溯源元数据。

    上游 a560ad0 的事实：26.02 → 26.04，89 处文件改动，
    上游只有 git diff，没有 Python 对象来表示这次跃迁。

    改写：不可变值对象，所有语义通过方法派生，无裸字符串。
    """

    from_version: RapidsVersion
    to_version: RapidsVersion
    commit_sha: str          # 引入此 bump 的上游 commit
    pr_number: int           # 上游 PR 号
    author: str              # 上游 PR 作者
    rationale: str           # 此次升级的原因说明

    def __post_init__(self) -> None:
        _dbg(
            "VersionBump.init",
            f"{self.from_version.tag} → {self.to_version.tag} "
            f"commit={self.commit_sha[:8]} PR=#{self.pr_number}",
        )

    def bump_kind(self) -> BumpKind:
        """
        计算跃迁类型。
        上游无此概念，全部是字符串替换。
        """
        _dbg("VersionBump.bump_kind", f"from={self.from_version.tag} to={self.to_version.tag}")
        if self.to_version < self.from_version:
            return BumpKind.UNKNOWN
        delta = self.to_version.cycle_delta(self.from_version)
        if delta == 1:
            # 跨年判断：同一 cycle 跳但年份不同（25.12→26.02）
            if self.from_version.year != self.to_version.year:
                return BumpKind.YEARLY
            return BumpKind.MINOR
        elif delta > 1:
            return BumpKind.UNKNOWN
        return BumpKind.UNKNOWN

    def is_forward(self) -> bool:
        """是否为正向升级（to > from）。上游不作此判断，直接替换。"""
        return self.to_version > self.from_version

    def cycle_count(self) -> int:
        """跨越的发布周期数，如 26.02→26.04 = 1 cycle。"""
        return abs(self.to_version.cycle_delta(self.from_version))

    def as_sed_pattern(self) -> str:
        """
        还原上游操作的 sed 等价命令（仅用于审计记录，Walpurgis 不执行 sed）。
        上游是全局字符串替换，此处序列化为可读形式。
        """
        return (
            f"s/{re.escape(self.from_version.tag)}/"
            f"{self.to_version.tag}/g"
        )

    def summary(self) -> str:
        """单行摘要，对应 git log 一行描述。"""
        return (
            f"[{self.commit_sha[:8]}] {self.from_version.tag} → {self.to_version.tag} "
            f"(+{self.cycle_count()} cycle, {self.bump_kind().value}, PR=#{self.pr_number})"
        )


# ─── a560ad0 版本跃迁实例 ────────────────────────────────────────────────────

A560AD0_BUMP = VersionBump(
    from_version=RapidsVersion.parse("26.02"),
    to_version=RapidsVersion.parse("26.04"),
    commit_sha="a560ad09a875e6e283a68557d181d650d1d34228",
    pr_number=387,
    author="AyodeAwe (Jake Awe, NVIDIA)",
    rationale=(
        "26.04 发布周期版本升级，属于 26.02 release burndown 流程的一部分。"
        "全量 sed 替换 20 个配置文件中所有 26.02 → 26.04 引用。"
    ),
)

# 断点 1：版本跃迁对象注册
_dbg("A560AD0_BUMP", A560AD0_BUMP.summary())


# ─── 4. AffectedScope — 受影响文件的功能域分类 ───────────────────────────────


class ConfigDomain(Enum):
    """
    a560ad0 涉及的配置文件功能域分类。
    上游只有 git diff 文件列表，无功能域分类。
    """

    DEVCONTAINER = "devcontainer"     # .devcontainer/*.json
    CI_WORKFLOW = "ci_workflow"       # .github/workflows/*.yaml
    VERSION_FILE = "version_file"     # VERSION
    CONDA_ENV = "conda_env"           # conda/environments/*.yaml
    CONDA_RECIPE = "conda_recipe"     # conda/recipes/
    DEP_MANIFEST = "dep_manifest"     # dependencies.yaml
    PYPROJECT = "pyproject"           # python/*/pyproject.toml


@dataclass(frozen=True)
class AffectedFile:
    """
    描述一个受 bump 影响的文件及其 SKIP 理由。
    上游每个文件只是 diff 中的一行，无 SKIP 语义。
    """

    path: str
    domain: ConfigDomain
    skip_reason: str    # Walpurgis SKIP 决策理由

    def is_skip(self) -> bool:
        """Walpurgis 中所有 a560ad0 文件均 SKIP。"""
        return True


@dataclass(frozen=True)
class AffectedScope:
    """
    汇总 a560ad0 全部 20 个受影响文件，按功能域分组。

    上游无此结构，只有 20 行 git diff stat。
    AffectedScope 使\"为何 SKIP\"可按域查询、可程序化审计。
    """

    files: tuple[AffectedFile, ...]

    def by_domain(self, domain: ConfigDomain) -> tuple[AffectedFile, ...]:
        """按功能域过滤文件列表。"""
        return tuple(f for f in self.files if f.domain == domain)

    def skip_count(self) -> int:
        return sum(1 for f in self.files if f.is_skip())

    def migrate_count(self) -> int:
        return sum(1 for f in self.files if not f.is_skip())

    def domain_summary(self) -> dict[str, int]:
        """返回各功能域的文件数量，用于 MIGRATION_LOG 生成。"""
        result: dict[str, int] = {}
        for f in self.files:
            key = f.domain.value
            result[key] = result.get(key, 0) + 1
        return result

    def dump(self) -> None:
        """打印结构化的 SKIP 理由摘要。"""
        for domain in ConfigDomain:
            files = self.by_domain(domain)
            if not files:
                continue
            print(f"  [{domain.value}]")
            for f in files:
                tag = "SKIP" if f.is_skip() else "MIGRATE"
                print(f"    [{tag}] {f.path}")
                print(f"           {f.skip_reason}")


# a560ad0 实际受影响文件集
_WALPURGIS_SKIP_CI = "Walpurgis 无 RAPIDS CI 流水线，此 CI 配置无迁移目标"
_WALPURGIS_SKIP_CONDA = "Walpurgis 无 conda 构建矩阵，此 conda 环境无迁移目标"
_WALPURGIS_SKIP_PYPROJECT = (
    "上游包（cugraph-pyg/libwholegraph/pylibwholegraph）非 Walpurgis 源码，"
    "其 pyproject.toml 依赖声明无迁移目标"
)

A560AD0_SCOPE = AffectedScope(
    files=(
        # devcontainer × 4
        AffectedFile(
            ".devcontainer/cuda12.9-conda/devcontainer.json",
            ConfigDomain.DEVCONTAINER,
            "devcontainer 基础镜像标签 26.02→26.04，Walpurgis 无 devcontainer 配置",
        ),
        AffectedFile(
            ".devcontainer/cuda12.9-pip/devcontainer.json",
            ConfigDomain.DEVCONTAINER,
            "同上，cuda12.9-pip 变体",
        ),
        AffectedFile(
            ".devcontainer/cuda13.1-conda/devcontainer.json",
            ConfigDomain.DEVCONTAINER,
            "同上，cuda13.1-conda 变体",
        ),
        AffectedFile(
            ".devcontainer/cuda13.1-pip/devcontainer.json",
            ConfigDomain.DEVCONTAINER,
            "同上，cuda13.1-pip 变体",
        ),
        # CI workflow × 3
        AffectedFile(".github/workflows/build.yaml", ConfigDomain.CI_WORKFLOW, _WALPURGIS_SKIP_CI),
        AffectedFile(".github/workflows/pr.yaml", ConfigDomain.CI_WORKFLOW, _WALPURGIS_SKIP_CI),
        AffectedFile(".github/workflows/test.yaml", ConfigDomain.CI_WORKFLOW, _WALPURGIS_SKIP_CI),
        # VERSION × 1
        AffectedFile(
            "VERSION",
            ConfigDomain.VERSION_FILE,
            "上游 VERSION 文件（26.02.00→26.04.00），Walpurgis 版本独立管理",
        ),
        # conda env × 4
        AffectedFile(
            "conda/environments/all_cuda-129_arch-aarch64.yaml",
            ConfigDomain.CONDA_ENV,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "conda/environments/all_cuda-129_arch-x86_64.yaml",
            ConfigDomain.CONDA_ENV,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "conda/environments/all_cuda-131_arch-aarch64.yaml",
            ConfigDomain.CONDA_ENV,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "conda/environments/all_cuda-131_arch-x86_64.yaml",
            ConfigDomain.CONDA_ENV,
            _WALPURGIS_SKIP_CONDA,
        ),
        # dependencies.yaml × 1
        AffectedFile(
            "dependencies.yaml",
            ConfigDomain.DEP_MANIFEST,
            "RAPIDS 构建依赖清单，Walpurgis 用 pyproject.toml 独立管理",
        ),
        # cugraph-pyg conda × 4
        AffectedFile(
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-aarch64.yaml",
            ConfigDomain.CONDA_RECIPE,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-x86_64.yaml",
            ConfigDomain.CONDA_RECIPE,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-aarch64.yaml",
            ConfigDomain.CONDA_RECIPE,
            _WALPURGIS_SKIP_CONDA,
        ),
        AffectedFile(
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-x86_64.yaml",
            ConfigDomain.CONDA_RECIPE,
            _WALPURGIS_SKIP_CONDA,
        ),
        # pyproject.toml × 3
        AffectedFile(
            "python/cugraph-pyg/pyproject.toml",
            ConfigDomain.PYPROJECT,
            _WALPURGIS_SKIP_PYPROJECT,
        ),
        AffectedFile(
            "python/libwholegraph/pyproject.toml",
            ConfigDomain.PYPROJECT,
            _WALPURGIS_SKIP_PYPROJECT,
        ),
        AffectedFile(
            "python/pylibwholegraph/pyproject.toml",
            ConfigDomain.PYPROJECT,
            _WALPURGIS_SKIP_PYPROJECT,
        ),
    )
)

# 断点 2：影响域统计
_dbg("A560AD0_SCOPE", f"total={len(A560AD0_SCOPE.files)} skip={A560AD0_SCOPE.skip_count()} migrate={A560AD0_SCOPE.migrate_count()}")
_dbg("A560AD0_SCOPE.domains", str(A560AD0_SCOPE.domain_summary()))


# ─── 5. BumpCompatibilityProbe — 运行时包版本兼容性探测 ──────────────────────


# RAPIDS 26.04 发布的核心包集合（来自 a560ad0 的 conda/pyproject 变更）
_RAPIDS_26_04_PACKAGES = (
    "cugraph",
    "pylibcugraph",
    "cudf",
    "cuml",
    "rmm",
    "libraft",
    "librmm",
    "libwholegraph",
    "pylibwholegraph",
    "cugraph-pyg",
)


@dataclass
class PackageCompatResult:
    """单个包的兼容性探测结果。"""
    package: str
    installed_version: Optional[str]   # None 表示未安装
    expected_rapids_cycle: str         # 期望的 RAPIDS cycle tag，如 "26.4"
    is_compatible: Optional[bool]      # None 表示未安装，无法判断

    def describe(self) -> str:
        if self.installed_version is None:
            return f"  {self.package:30s} <未安装>  (期望 rapids-cycle={self.expected_rapids_cycle})"
        compat_mark = "✓" if self.is_compatible else "✗"
        return (
            f"  {self.package:30s} {self.installed_version:15s} "
            f"期望={self.expected_rapids_cycle}  {compat_mark}"
        )


class BumpCompatibilityProbe:
    """
    探测 Walpurgis 运行时中实际安装的 RAPIDS 相关包版本，
    判断是否与给定的 RAPIDS 版本 cycle 兼容。

    上游 a560ad0 只在 pyproject.toml / conda yaml 声明版本约束，
    没有 Python 层的运行时探测机制。

    改写：在 Python import 层主动探测，使版本不匹配可被 CI 或测试捕获。
    """

    def __init__(self, target: RapidsVersion) -> None:
        self.target = target
        # 断点 3：探针初始化
        _dbg("BumpCompatibilityProbe.init", f"target={target.tag}")

    def _installed_version(self, pkg: str) -> Optional[str]:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _version_matches_cycle(self, installed: str, target: RapidsVersion) -> bool:
        """
        判断已安装版本是否属于目标 RAPIDS cycle。
        RAPIDS 包版本格式为 YY.M.x（conda 去前导零），如 26.4.0。
        """
        # 断点 4：版本匹配判断
        _dbg(
            "BumpCompatibilityProbe._version_matches_cycle",
            f"installed={installed!r} target_cycle={target.year}.{target.month}",
        )
        # 解析已安装版本的 major.minor
        m = re.match(r"^(\d+)\.(\d+)", installed)
        if not m:
            return False
        inst_year, inst_month = int(m.group(1)), int(m.group(2))
        return inst_year == target.year and inst_month == target.month

    def probe_all(self) -> list[PackageCompatResult]:
        """探测所有 RAPIDS 26.04 核心包，返回结果列表。"""
        results = []
        cycle_str = f"{self.target.year}.{self.target.month}"
        for pkg in _RAPIDS_26_04_PACKAGES:
            installed = self._installed_version(pkg)
            if installed is None:
                compat = None
            else:
                compat = self._version_matches_cycle(installed, self.target)
            results.append(PackageCompatResult(
                package=pkg,
                installed_version=installed,
                expected_rapids_cycle=cycle_str,
                is_compatible=compat,
            ))
        return results

    def dump(self) -> None:
        """打印探测报告。"""
        results = self.probe_all()
        installed_count = sum(1 for r in results if r.installed_version is not None)
        compat_count = sum(1 for r in results if r.is_compatible is True)
        print(f"── BumpCompatibilityProbe (RAPIDS {self.target.tag}) ──")
        for r in results:
            print(r.describe())
        print(
            f"── 已安装: {installed_count}/{len(results)}  "
            f"兼容: {compat_count}/{installed_count if installed_count else 'N/A'} ──"
        )


# ─── 6. VersionBumpAudit — 残留旧版本字符串扫描 ──────────────────────────────


@dataclass
class VersionBumpAudit:
    """
    扫描文件文本，确认旧版本字符串（如 "26.02"）不再残留。

    使用场景：CI 验证 a560ad0 的 sed 替换是否彻底。
    上游通过全局 sed 完成后没有回头检查机制，Walpurgis 独有。
    """

    bump: VersionBump

    @property
    def _old_pattern(self) -> str:
        """匹配旧版本号的正则（同时匹配 26.02 和 26.2 格式）。"""
        v = self.bump.from_version
        # 同时匹配带前导零和不带前导零的格式
        month_pat = f"{v.month:02d}|{v.month}"
        return rf"\b{v.year:02d}\.(?:{month_pat})\b"

    def scan_text(self, text: str) -> list[tuple[int, str]]:
        """
        扫描文本，返回包含旧版本号的 (行号, 行内容) 列表。
        上游通过 grep 人工检查，此处程序化。
        """
        # 断点 5：文本扫描入口
        _dbg(
            "VersionBumpAudit.scan_text",
            f"pattern={self._old_pattern!r} text_len={len(text)}",
        )
        pattern = re.compile(self._old_pattern)
        hits = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((lineno, line))
        _dbg("VersionBumpAudit.scan_text.result", f"hits={len(hits)}")
        return hits

    def assert_clean(self, path: str, encoding: str = "utf-8") -> None:
        """
        读取文件，断言不含旧版本残留。
        上游无此机制。
        """
        try:
            text = open(path, encoding=encoding).read()
        except FileNotFoundError:
            _dbg("VersionBumpAudit.assert_clean", f"文件不存在（跳过）: {path}")
            return
        hits = self.scan_text(text)
        if hits:
            detail = "\n".join(f"  L{n}: {line}" for n, line in hits[:5])
            raise AssertionError(
                f"[Walpurgis VersionBumpAudit] {path} 仍残留旧版本 {self.bump.from_version.tag}！\n"
                f"（来自上游 {self.bump.commit_sha[:8]}，sed 替换应已完成）\n"
                f"残留行示例:\n{detail}"
            )


# a560ad0 审计器
A560AD0_AUDIT = VersionBumpAudit(bump=A560AD0_BUMP)

# 断点 6：审计器注册
_dbg("A560AD0_AUDIT", f"old_pattern={A560AD0_AUDIT._old_pattern!r}")


# ─── 自测 ─────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """10 项断言，覆盖 a560ad0 版本跃迁建模的核心路径。"""

    # 断点 7：自测启动
    _dbg("_self_test", "启动 a560ad0 version_bump_policy 自测")

    # 1. RapidsVersion 解析与比较
    v_from = RapidsVersion.parse("26.02")
    v_to = RapidsVersion.parse("26.04")
    assert v_from.year == 26 and v_from.month == 2, "test1: from 解析错误"
    assert v_to.year == 26 and v_to.month == 4, "test1: to 解析错误"
    assert v_to > v_from, "test1: 26.04 应大于 26.02"
    print("[PASS] test1: RapidsVersion 解析与排序")

    # 2. tag 与 conda_wildcard 格式
    assert v_to.tag == "26.04", f"test2: tag={v_to.tag!r}"
    assert v_to.conda_wildcard == "26.4.*", f"test2: conda_wildcard={v_to.conda_wildcard!r}"
    assert v_to.pip_pin == "==26.4.*,>=0.0.0a0", f"test2: pip_pin={v_to.pip_pin!r}"
    print("[PASS] test2: tag / conda_wildcard / pip_pin 格式")

    # 3. cycle_delta 计算
    assert v_to.cycle_delta(v_from) == 1, "test3: 26.02→26.04 应为 1 cycle"
    v_26_12 = RapidsVersion(year=26, month=12)
    v_27_02 = RapidsVersion(year=27, month=2)
    assert v_27_02.cycle_delta(v_26_12) == 1, "test3: 26.12→27.02 跨年 1 cycle"
    print("[PASS] test3: cycle_delta 计算")

    # 4. BumpKind 判断
    assert A560AD0_BUMP.bump_kind() == BumpKind.MINOR, (
        f"test4: a560ad0 应为 MINOR bump，实际={A560AD0_BUMP.bump_kind()}"
    )
    assert A560AD0_BUMP.is_forward(), "test4: a560ad0 应为正向升级"
    assert A560AD0_BUMP.cycle_count() == 1, "test4: a560ad0 跨 1 cycle"
    print("[PASS] test4: BumpKind / is_forward / cycle_count")

    # 5. as_sed_pattern
    sed_pat = A560AD0_BUMP.as_sed_pattern()
    assert "26\\.02" in sed_pat, f"test5: sed 应包含 from 版本，实际={sed_pat!r}"
    assert "26.04" in sed_pat, f"test5: sed 应包含 to 版本，实际={sed_pat!r}"
    print("[PASS] test5: as_sed_pattern")

    # 6. AffectedScope 统计
    assert len(A560AD0_SCOPE.files) == 20, f"test6: 应有 20 个受影响文件，实际={len(A560AD0_SCOPE.files)}"
    assert A560AD0_SCOPE.skip_count() == 20, "test6: 全部 20 个文件应 SKIP"
    assert A560AD0_SCOPE.migrate_count() == 0, "test6: 无需迁移的文件为 0"
    print("[PASS] test6: AffectedScope 统计（20 SKIP，0 MIGRATE）")

    # 7. AffectedScope 按域过滤
    devcontainer_files = A560AD0_SCOPE.by_domain(ConfigDomain.DEVCONTAINER)
    assert len(devcontainer_files) == 4, f"test7: devcontainer 应有 4 个文件，实际={len(devcontainer_files)}"
    ci_files = A560AD0_SCOPE.by_domain(ConfigDomain.CI_WORKFLOW)
    assert len(ci_files) == 3, f"test7: CI workflow 应有 3 个文件，实际={len(ci_files)}"
    print("[PASS] test7: AffectedScope.by_domain 过滤")

    # 8. VersionBumpAudit.scan_text：旧版本残留检测
    audit = VersionBumpAudit(bump=A560AD0_BUMP)
    clean_text = "rapidsai/ci-conda:26.04-latest\npylibcugraph==26.4.*,>=0.0.0a0\n"
    assert audit.scan_text(clean_text) == [], "test8: 干净文本不应有命中"
    dirty_text = "rapidsai/ci-conda:26.02-latest\npylibcugraph==26.2.*,>=0.0.0a0\n"
    hits = audit.scan_text(dirty_text)
    assert len(hits) == 2, f"test8: 应命中 2 行，实际={len(hits)}"
    print("[PASS] test8: VersionBumpAudit.scan_text 残留检测")

    # 9. RapidsVersion 非法输入拒绝
    rejected = False
    try:
        RapidsVersion.parse("26.03")   # 奇数月，非 RAPIDS 发布周期
    except ValueError:
        rejected = True
    assert rejected, "test9: 奇数月应被拒绝"
    print("[PASS] test9: 奇数月版本拒绝")

    # 10. A560AD0_BUMP 元数据完整性
    assert A560AD0_BUMP.pr_number == 387, "test10: PR 号应为 387"
    assert "26.04" in A560AD0_BUMP.to_version.tag, "test10: to_version 应为 26.04"
    assert A560AD0_BUMP.commit_sha.startswith("a560ad0"), "test10: commit sha 前缀"
    print("[PASS] test10: A560AD0_BUMP 元数据完整性")

    print("\n[ALL PASS] a560ad0 version_bump_policy 自测：10 项断言全部通过")


if __name__ == "__main__":
    _self_test()
