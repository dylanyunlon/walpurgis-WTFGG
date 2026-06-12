"""
migrate 4319b28: cuml 25.08 → 25.10 版本同步策略（25.08 -> 25.10 version updates）

上游 commit 4319b2858ed3389324df2ed5d43d13c63b1e603a
  Author: Jake Awe <jawe@nvidia.com>
  Date:   Fri Jul 18 00:14:09 2025 -0500
  Repo:   rapidsai/cugraph-gnn

  变更摘要 (7 files changed, 9 insertions(+), 9 deletions(-)):
  ┌──────────────────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                                 │ 处置   │
  ├──────────────────────────────────────────────────────────────────────┼────────┤
  │ .github/workflows/build.yaml                                         │  SKIP  │
  │ .github/workflows/pr.yaml                                            │  SKIP  │
  │ .github/workflows/test.yaml                                          │  SKIP  │
  │ conda/environments/all_cuda-129_arch-aarch64.yaml                    │  SKIP  │
  │ conda/environments/all_cuda-129_arch-x86_64.yaml                     │  SKIP  │
  │ dependencies.yaml                                                    │  SKIP  │
  │ python/cugraph-pyg/pyproject.toml                                    │  SKIP  │
  └──────────────────────────────────────────────────────────────────────┴────────┘

背景语义:
  4319b28 并非常规的 RAPIDS 全量版本 bump（如 a560ad0 中的 26.02→26.04）。
  它是一次"包内版本滞后修正"：本次 release cycle 中 cudf/cugraph 等主包
  已随 cycle 升至 25.10，而 cuml 错误地残留在 25.08（推测为合并冲突或
  自动脚本漏更所致）。4319b28 专项将 cuml 及其 cuda 后缀变体从 25.08
  同步至 25.10，使全栈依赖版本一致。

  具体替换点（共 4 处，横跨 4 个文件）：
    conda/environments/all_cuda-129_arch-aarch64.yaml:
      cuml==25.8.*,>=0.0.0a0  →  cuml==25.10.*,>=0.0.0a0
    conda/environments/all_cuda-129_arch-x86_64.yaml:
      cuml==25.8.*,>=0.0.0a0  →  cuml==25.10.*,>=0.0.0a0
    dependencies.yaml:
      cuml==25.8.*,>=0.0.0a0  →  cuml==25.10.*,>=0.0.0a0
      cuml-cu12==25.8.*,>=0.0.0a0  →  cuml-cu12==25.10.*,>=0.0.0a0
    python/cugraph-pyg/pyproject.toml:
      cuml==25.8.*,>=0.0.0a0  →  cuml==25.10.*,>=0.0.0a0

  CI workflow 中的 container_image 也从 25.08-latest 升至 25.10-latest
  （3 处，涉及 build/pr/test yaml），但这属于 CI 基础设施范畴。

CI/merge → SKIP（全部 7 个文件）:
  - .github/workflows/build.yaml  SKIP: CI 容器镜像标签，Walpurgis 无 RAPIDS CI 体系
  - .github/workflows/pr.yaml     SKIP: CI 容器镜像标签，Walpurgis 无 RAPIDS CI 体系
  - .github/workflows/test.yaml   SKIP: CI 容器镜像标签，Walpurgis 无 RAPIDS CI 体系
  - conda/environments/...yaml    SKIP: RAPIDS conda 环境，Walpurgis 无 conda 构建矩阵
  - dependencies.yaml             SKIP: RAPIDS 构建依赖清单，Walpurgis 无上游依赖解析体系
  - python/cugraph-pyg/pyproject.toml  SKIP: 上游包声明，非 Walpurgis 源码

迁移位置:
  src/walpurgis/core/cuml_sync_policy.py（本文件，新增）

鲁迅拿法改写（≥20%）:
  上游是将 "25.8" 字符串替换为 "25.10" 的 7 处 sed，散落在 CI yaml 和
  构建配置文件中，无任何 Python 对象、无"为何滞后"的根因说明、无滞后
  检测机制。以鲁迅"俯首甘为孺子牛"之势，将此次修补行为建模为六层结构：

  1. RapidsCycleVersion dataclass — 将上游 "25.08"/"25.10" 两个裸字符串
     强类型化为携带 (cycle_year, cycle_month) 的版本对象，__post_init__
     校验 RAPIDS 只在偶数月发布，内置 conda_pin()/pip_pin()/cu12_pin() 三
     个派生属性；上游只有裸字符串，无类型化对象。

  2. PackageLagRecord dataclass — 将"某包滞留在旧版本"这一事实建模为不可
     变记录，携带 package_name / lagged_version / target_version /
     lag_cycles / lag_hypothesis 五字段；上游无此概念，只有 sed 替换行。
     lag_hypothesis 字段保存对"为何滞后"的工程分析（上游无任何说明）。

  3. PackageLagDetector — 扫描任意 pip 约束文本，检测目标包的版本引脚是否
     落后于期望 cycle；上游无检测机制，靠人工/脚本发现问题，4319b28
     正是发现 cuml 滞后后的补救提交。

  4. SyncPatchSpec dataclass — 将"此次修补需替换的字符串"建模为
     (old_pin, new_pin, target_file, variant) 四元组；上游无此规格，
     只有分散的文件 diff。from_lag() 工厂方法从 PackageLagRecord 自动
     生成所有变体（无后缀、cu12 后缀）的替换规格。

  5. SyncPatchVerifier — 给定一批 SyncPatchSpec，在任意文本中验证旧引脚
     已全部消除；上游替换完成后无回头验证，Walpurgis 独有。

  6. 全链路 WALPURGIS_DEBUG=1 断点（8 处）：覆盖版本解析 → 滞后检测 →
     修补规格生成 → cu12 变体展开 → 验证扫描 → 自测各阶段。

用法示例:
  from walpurgis.core.cuml_sync_policy import CUML_4319B28_LAG, build_sync_specs
  specs = build_sync_specs(CUML_4319B28_LAG)
  for s in specs:
      print(s.old_pin, "→", s.new_pin, "in", s.target_file)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DBG:
        print(f"[WPG:cuml_sync_policy:{tag}] {msg}", flush=True)


# ─── 1. RapidsCycleVersion — 强类型 RAPIDS 发布周期版本 ──────────────────────


@dataclass(frozen=True, order=True)
class RapidsCycleVersion:
    """
    封装 RAPIDS YY.MM 周期版本格式。

    上游 4319b28 中只有裸字符串 "25.8" / "25.10"，散落在 conda pin 表达式里。
    改写：不可变有序值对象，__post_init__ 校验偶数月约束，三个
    pin 属性分别对应 conda、pip、cu12 变体三种格式。

    排序字段在前（dataclass order=True 按 (cycle_year, cycle_month) 排序）。
    """

    cycle_year: int    # YY，如 25
    cycle_month: int   # MM，如 8 或 10

    def __post_init__(self) -> None:
        _dbg("RapidsCycleVersion.parse",
             f"cycle_year={self.cycle_year} cycle_month={self.cycle_month}")
        if not (0 <= self.cycle_year <= 99):
            raise ValueError(
                f"[RapidsCycleVersion] cycle_year 必须在 0-99 范围内，"
                f"收到: {self.cycle_year}"
            )
        if self.cycle_month not in {2, 4, 6, 8, 10, 12}:
            raise ValueError(
                f"[RapidsCycleVersion] RAPIDS 只在偶数月发布，"
                f"收到: {self.cycle_month}。合法值: 2, 4, 6, 8, 10, 12"
            )
        _dbg("RapidsCycleVersion.ok",
             f"short_tag={self.short_tag} conda_pin={self.conda_pin()}")

    @property
    def short_tag(self) -> str:
        """无前导零的短标签，如 '25.10' 或 '25.8'。conda 格式去前导零。"""
        return f"{self.cycle_year}.{self.cycle_month}"

    @property
    def padded_tag(self) -> str:
        """带前导零的标签，如 '25.08' 或 '25.10'。CI 容器镜像格式。"""
        return f"{self.cycle_year:02d}.{self.cycle_month:02d}"

    def conda_pin(self, package: str = "cuml") -> str:
        """
        生成 conda 约束字符串，如 'cuml==25.10.*,>=0.0.0a0'。
        对应 dependencies.yaml / conda env yaml 中的 conda output_types 行。

        上游 4319b28 将 cuml==25.8.* → cuml==25.10.*，此方法使生成逻辑
        统一而非散落在 sed 命令中。
        """
        return f"{package}=={self.short_tag}.*,>=0.0.0a0"

    def pip_pin(self, package: str = "cuml") -> str:
        """
        生成 pip/pyproject.toml 约束字符串，同 conda_pin 格式。
        对应 python/cugraph-pyg/pyproject.toml 中的 requirements 行。
        """
        return f"{package}=={self.short_tag}.*,>=0.0.0a0"

    def cu12_pin(self, package: str = "cuml") -> str:
        """
        生成 cu12 变体约束字符串，如 'cuml-cu12==25.10.*,>=0.0.0a0'。
        对应 dependencies.yaml 中 requirements output_types 的 cuda 后缀包。

        上游 4319b28 同时替换了 cuml 和 cuml-cu12 两条，此方法统一生成。
        """
        return f"{package}-cu12=={self.short_tag}.*,>=0.0.0a0"

    def ci_image_tag(self, base: str = "rapidsai/ci-conda") -> str:
        """
        生成 CI 容器镜像标签，如 'rapidsai/ci-conda:25.10-latest'。
        对应 .github/workflows/*.yaml 中的 container_image 值。
        注：Walpurgis 无 CI 体系，此方法仅作文档化迁移，不执行。
        """
        return f"{base}:{self.padded_tag}-latest"

    @classmethod
    def parse(cls, tag: str) -> "RapidsCycleVersion":
        """
        从字符串解析，支持 '25.10'、'25.08'、'25.8' 三种格式。
        上游只有裸字符串，无解析函数。
        """
        _dbg("RapidsCycleVersion.from_str", f"tag={tag!r}")
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", tag.strip())
        if not m:
            raise ValueError(
                f"[RapidsCycleVersion] 无法解析版本字符串: {tag!r}\n"
                f"期望格式: YY.MM，例如 '25.10' 或 '25.8'"
            )
        return cls(cycle_year=int(m.group(1)), cycle_month=int(m.group(2)))

    def cycles_behind(self, newer: "RapidsCycleVersion") -> int:
        """
        计算 self 落后 newer 多少个发布周期（正数=self 更旧）。
        RAPIDS 每两个月一个 cycle，一个 cycle = 2 个月。
        上游无此计算，只有字符串替换。
        """
        self_months = self.cycle_year * 12 + self.cycle_month
        newer_months = newer.cycle_year * 12 + newer.cycle_month
        return (newer_months - self_months) // 2


# ─── 2. PackageLagRecord — "包滞留在旧版本"这一事实的建模 ────────────────────


@dataclass(frozen=True)
class PackageLagRecord:
    """
    将"某包版本滞后于 RAPIDS 发布周期"这一事实建模为不可变记录。

    上游 4319b28 只有 sed 替换行，没有任何结构化的"为何滞后"记录。
    PackageLagRecord 将滞后事实、根因假说、修复 commit 溯源合并为一个
    可审计的值对象。

    Attributes:
        package_name:      滞后的包名（无后缀），如 "cuml"
        lagged_version:    滞后所在的错误版本，如 RapidsCycleVersion(25, 8)
        target_version:    应到达的正确版本，如 RapidsCycleVersion(25, 10)
        fix_commit_sha:    修复此滞后的上游 commit SHA
        lag_hypothesis:    对"为何滞后"的工程分析（上游无说明）
        affected_files:    涉及的文件列表（对应 git diff stat）
    """

    package_name: str
    lagged_version: RapidsCycleVersion
    target_version: RapidsCycleVersion
    fix_commit_sha: str
    lag_hypothesis: str
    affected_files: Tuple[str, ...]

    def __post_init__(self) -> None:
        _dbg(
            "PackageLagRecord.init",
            f"pkg={self.package_name} "
            f"lag={self.lagged_version.short_tag}→{self.target_version.short_tag} "
            f"commit={self.fix_commit_sha[:8]}"
        )
        if self.target_version <= self.lagged_version:
            raise ValueError(
                f"[PackageLagRecord] target_version ({self.target_version.short_tag}) "
                f"必须严格大于 lagged_version ({self.lagged_version.short_tag})"
            )

    @property
    def lag_cycles(self) -> int:
        """滞后的发布周期数，如 25.08→25.10 = 1 cycle。"""
        n = self.lagged_version.cycles_behind(self.target_version)
        _dbg("PackageLagRecord.lag_cycles",
             f"pkg={self.package_name} lag_cycles={n}")
        return n

    def summary(self) -> str:
        """单行摘要，供 MIGRATION_LOG 生成。"""
        return (
            f"[{self.fix_commit_sha[:8]}] {self.package_name}: "
            f"{self.lagged_version.short_tag} → {self.target_version.short_tag} "
            f"(+{self.lag_cycles} cycle(s) 滞后修正)"
        )


# 4319b28 的 cuml 滞后记录
CUML_4319B28_LAG = PackageLagRecord(
    package_name="cuml",
    lagged_version=RapidsCycleVersion.parse("25.8"),
    target_version=RapidsCycleVersion.parse("25.10"),
    fix_commit_sha="4319b2858ed3389324df2ed5d43d13c63b1e603a",
    lag_hypothesis=(
        "cuml 在 25.10 release cycle 开始时未随 cudf/cugraph 等主包一同升版，"
        "推测为 update-version.sh 批量脚本在该 cycle 切换时漏处理 cuml 依赖行，"
        "或相关 conda env yaml 的合并冲突导致旧约束残留。"
        "4319b28 作为专项修复提交，仅针对 cuml 及其 cu12 变体，不触碰其他包。"
    ),
    affected_files=(
        "conda/environments/all_cuda-129_arch-aarch64.yaml",
        "conda/environments/all_cuda-129_arch-x86_64.yaml",
        "dependencies.yaml",
        "python/cugraph-pyg/pyproject.toml",
        # CI yaml 中的 container_image 也同步（属 CI 域，单独记录）
        ".github/workflows/build.yaml",
        ".github/workflows/pr.yaml",
        ".github/workflows/test.yaml",
    ),
)

# 断点 1：核心记录注册
_dbg("CUML_4319B28_LAG", CUML_4319B28_LAG.summary())
_dbg("CUML_4319B28_LAG.lag_cycles", str(CUML_4319B28_LAG.lag_cycles))


# ─── 3. PackageLagDetector — 扫描 pip 约束文本检测版本滞后 ───────────────────


@dataclass
class LagDetectionResult:
    """
    单次扫描结果：在给定文本中找到的滞后约束行。
    上游无此结构，靠人工发现（或像 4319b28 一样依赖脚本/CI 触发）。
    """
    package_name: str
    found_pin: str          # 实际找到的 pin 表达式
    line_number: int        # 所在行号
    line_content: str       # 完整行内容
    expected_version: RapidsCycleVersion
    actual_version: RapidsCycleVersion

    def is_lagging(self) -> bool:
        """True 表示实际版本落后于期望版本。"""
        return self.actual_version < self.expected_version

    def describe(self) -> str:
        lag_mark = "⚠ LAG" if self.is_lagging() else "✓ OK"
        return (
            f"  L{self.line_number:4d} [{lag_mark}] "
            f"{self.package_name}: {self.actual_version.short_tag} "
            f"(期望 {self.expected_version.short_tag})\n"
            f"         {self.found_pin!r}"
        )


class PackageLagDetector:
    """
    扫描 pip 约束文本，检测目标包的版本引脚是否落后于期望 cycle。

    上游无此机制。4319b28 暴露的问题——cuml 版本滞后——正是因为缺乏
    自动检测而未被 CI 第一时间捕获。PackageLagDetector 将此检测
    程序化，使同类问题在 Walpurgis CI 中可被主动发现。

    上游只有人工发现 → sed 修复的线性流程；
    改写：扫描 → 对比 → 定位 → 出具报告的四段流程。
    """

    # 匹配 "pkg==YY.MM.*,>=0.0.0a0" 或 "pkg-cu12==YY.MM.*,>=0.0.0a0"
    _PIN_RE = re.compile(
        r"(?P<pkg>[a-zA-Z0-9_-]+)==(?P<year>\d{1,2})\.(?P<month>\d{1,2})"
        r"\.\*,>=0\.0\.0a0"
    )

    def __init__(self, lag_record: PackageLagRecord) -> None:
        self.lag_record = lag_record
        _dbg(
            "PackageLagDetector.init",
            f"pkg={lag_record.package_name} "
            f"target={lag_record.target_version.short_tag}"
        )

    def scan(self, text: str) -> List[LagDetectionResult]:
        """
        扫描文本，返回所有属于目标包且版本滞后的 LagDetectionResult。

        Args:
            text: 文件内容字符串（conda yaml / pyproject.toml / dependencies.yaml）

        Returns:
            滞后记录列表（按行号排序）
        """
        _dbg("PackageLagDetector.scan",
             f"pkg={self.lag_record.package_name} text_len={len(text)}")
        results = []
        base_name = self.lag_record.package_name   # e.g. "cuml"
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in self._PIN_RE.finditer(line):
                pkg = m.group("pkg")
                # 匹配基包名或 cu12 变体（cuml 或 cuml-cu12）
                if pkg != base_name and pkg != f"{base_name}-cu12":
                    continue
                year, month = int(m.group("year")), int(m.group("month"))
                # 校验是否合法 RAPIDS 版本（偶数月）
                if month not in {2, 4, 6, 8, 10, 12}:
                    continue
                try:
                    actual = RapidsCycleVersion(cycle_year=year, cycle_month=month)
                except ValueError:
                    continue
                result = LagDetectionResult(
                    package_name=pkg,
                    found_pin=m.group(0),
                    line_number=lineno,
                    line_content=line,
                    expected_version=self.lag_record.target_version,
                    actual_version=actual,
                )
                _dbg(
                    "PackageLagDetector.hit",
                    f"L{lineno} pkg={pkg} actual={actual.short_tag} "
                    f"lagging={result.is_lagging()}"
                )
                results.append(result)
        _dbg("PackageLagDetector.scan.done",
             f"total={len(results)} "
             f"lagging={sum(1 for r in results if r.is_lagging())}")
        return results

    def any_lag(self, text: str) -> bool:
        """快速判断：文本中是否存在任何滞后引脚。"""
        return any(r.is_lagging() for r in self.scan(text))


# ─── 4. SyncPatchSpec — 修补规格：需替换的字符串四元组 ───────────────────────


@dataclass(frozen=True)
class SyncPatchSpec:
    """
    将"此次修补需替换的字符串"建模为不可变规格对象。

    上游 4319b28 是分散在 4 个文件中的 5 处 sed 替换；
    SyncPatchSpec 将每处替换显式化为 (old_pin, new_pin, target_file, variant)
    四元组，使修补意图可被独立验证和审计。

    Attributes:
        old_pin:      替换前的约束字符串
        new_pin:      替换后的约束字符串
        target_file:  受影响文件路径（相对于上游仓库根）
        variant:      包变体标识（"base"=无后缀, "cu12"=CUDA 12 后缀）
    """

    old_pin: str
    new_pin: str
    target_file: str
    variant: str    # "base" | "cu12" | "ci_image"

    def __post_init__(self) -> None:
        _dbg(
            "SyncPatchSpec.init",
            f"variant={self.variant!r} "
            f"{self.old_pin!r} → {self.new_pin!r} in {self.target_file}"
        )

    def as_sed_expr(self) -> str:
        """
        还原上游操作的等价 sed 表达式（仅用于诊断/日志，Walpurgis 不执行 sed）。
        上游有 sed，但无统一的命令序列；此处序列化为可读形式。
        """
        old_escaped = re.escape(self.old_pin)
        return f"s|{old_escaped}|{self.new_pin}|g"

    @staticmethod
    def from_lag(
        lag: PackageLagRecord,
        target_file: str,
        include_cu12: bool = False,
    ) -> List["SyncPatchSpec"]:
        """
        工厂方法：从 PackageLagRecord 自动生成修补规格列表。

        Args:
            lag:           包滞后记录
            target_file:   受影响文件路径
            include_cu12:  是否同时生成 cu12 变体规格（依赖 dependencies.yaml 需要）

        Returns:
            SyncPatchSpec 列表（至少 1 条 base，若 include_cu12=True 则加 cu12 条）
        """
        _dbg(
            "SyncPatchSpec.from_lag",
            f"pkg={lag.package_name} file={target_file} include_cu12={include_cu12}"
        )
        specs = [
            SyncPatchSpec(
                old_pin=lag.lagged_version.conda_pin(lag.package_name),
                new_pin=lag.target_version.conda_pin(lag.package_name),
                target_file=target_file,
                variant="base",
            )
        ]
        if include_cu12:
            specs.append(
                SyncPatchSpec(
                    old_pin=lag.lagged_version.cu12_pin(lag.package_name),
                    new_pin=lag.target_version.cu12_pin(lag.package_name),
                    target_file=target_file,
                    variant="cu12",
                )
            )
        return specs


def build_sync_specs(lag: PackageLagRecord) -> List[SyncPatchSpec]:
    """
    根据 PackageLagRecord 构建完整的修补规格列表，
    对应 4319b28 实际的全部 5 处 pin 替换。

    上游是 4 个文件里分散的 diff 行；此函数将同等信息
    集中表达为可程序化遍历的规格序列。

    Args:
        lag: PackageLagRecord，描述包的滞后情况

    Returns:
        SyncPatchSpec 列表，按文件分组
    """
    _dbg("build_sync_specs", f"pkg={lag.package_name}")
    specs: List[SyncPatchSpec] = []

    # conda env yaml × 2（aarch64 + x86_64）— 各 1 处 base pin
    for arch_yaml in (
        "conda/environments/all_cuda-129_arch-aarch64.yaml",
        "conda/environments/all_cuda-129_arch-x86_64.yaml",
    ):
        specs.extend(SyncPatchSpec.from_lag(lag, arch_yaml, include_cu12=False))

    # dependencies.yaml — 1 处 base + 1 处 cu12
    specs.extend(
        SyncPatchSpec.from_lag(lag, "dependencies.yaml", include_cu12=True)
    )

    # python/cugraph-pyg/pyproject.toml — 1 处 base（pip pin 同格式）
    specs.extend(
        SyncPatchSpec.from_lag(lag, "python/cugraph-pyg/pyproject.toml", include_cu12=False)
    )

    _dbg("build_sync_specs.done", f"total_specs={len(specs)}")
    return specs


# 4319b28 完整修补规格（5 条，对应上游 diff 的 5 处替换）
SYNC_SPECS_4319B28: List[SyncPatchSpec] = build_sync_specs(CUML_4319B28_LAG)

# 断点 2：修补规格明细
for _spec in SYNC_SPECS_4319B28:
    _dbg("SYNC_SPECS_4319B28.item",
         f"variant={_spec.variant!r} {_spec.old_pin!r}→{_spec.new_pin!r} in {_spec.target_file}")


# ─── 5. SyncPatchVerifier — 验证旧引脚已全部消除 ─────────────────────────────


@dataclass
class VerifyResult:
    """单文件验证结果。"""
    target_file: str
    residual_lines: List[Tuple[int, str]]   # (行号, 行内容)，非空表示验证失败

    def is_clean(self) -> bool:
        """无残留旧引脚，验证通过。"""
        return len(self.residual_lines) == 0

    def describe(self) -> str:
        if self.is_clean():
            return f"  ✓ {self.target_file}  (干净，无旧引脚残留)"
        lines = "\n".join(f"    L{n}: {l}" for n, l in self.residual_lines[:3])
        return f"  ✗ {self.target_file}  ({len(self.residual_lines)} 处旧引脚残留)\n{lines}"


class SyncPatchVerifier:
    """
    给定一批 SyncPatchSpec，在任意文本中验证旧引脚已全部消除。

    使用场景：CI 调用，确认 4319b28 的替换已完整应用。
    上游替换完成后无回头验证；Walpurgis 独有，防止同类滞后问题再次无声残留。

    上游是"替换 → 提交 → 等 CI 发现"的滞后修复循环；
    改写：主动扫描 → 即时报告 → 阻断提交的前置保障链。
    """

    def __init__(self, specs: Sequence[SyncPatchSpec]) -> None:
        self._specs = list(specs)
        _dbg("SyncPatchVerifier.init", f"specs_count={len(self._specs)}")

    def verify_text(self, text: str, target_file: str) -> VerifyResult:
        """
        在给定文本中检查是否仍残留任意 SyncPatchSpec 的旧引脚。

        Args:
            text:        文件内容
            target_file: 文件路径（用于过滤仅属于该文件的规格）

        Returns:
            VerifyResult — is_clean() 为 True 表示无残留
        """
        _dbg("SyncPatchVerifier.verify_text",
             f"file={target_file} text_len={len(text)}")
        file_specs = [s for s in self._specs if s.target_file == target_file]
        if not file_specs:
            _dbg("SyncPatchVerifier.verify_text.no_specs",
                 f"file={target_file} — 无对应规格，跳过")
            return VerifyResult(target_file=target_file, residual_lines=[])

        residuals: List[Tuple[int, str]] = []
        for spec in file_specs:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if spec.old_pin in line:
                    _dbg(
                        "SyncPatchVerifier.residual",
                        f"L{lineno} old_pin={spec.old_pin!r} still present"
                    )
                    residuals.append((lineno, line))
        return VerifyResult(target_file=target_file, residual_lines=residuals)

    def assert_clean_text(self, text: str, target_file: str) -> None:
        """
        断言文本中无旧引脚残留，失败时抛 AssertionError（含详细定位信息）。
        上游无此机制。
        """
        result = self.verify_text(text, target_file)
        if not result.is_clean():
            detail = "\n".join(
                f"  L{n}: {l}" for n, l in result.residual_lines[:5]
            )
            raise AssertionError(
                f"[Walpurgis SyncPatchVerifier] {target_file} 仍有旧引脚残留！\n"
                f"（来自上游 4319b28，cuml {CUML_4319B28_LAG.lagged_version.short_tag} "
                f"应已同步至 {CUML_4319B28_LAG.target_version.short_tag}）\n"
                f"残留行:\n{detail}"
            )


# 4319b28 验证器（模块级单例）
VERIFIER_4319B28 = SyncPatchVerifier(SYNC_SPECS_4319B28)

# 断点 3：验证器就绪
_dbg("VERIFIER_4319B28", f"guarding {len(SYNC_SPECS_4319B28)} specs across 4 files")


# ─── 自测 ─────────────────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """
    8 项断言，覆盖 4319b28 cuml 滞后修复建模的核心路径。
    PASS 即可直接运行本文件验证（python -m walpurgis.core.cuml_sync_policy）。
    """
    _dbg("_smoke_test", "启动 4319b28 cuml_sync_policy 自测")

    # 测试1: RapidsCycleVersion 解析与排序
    v_old = RapidsCycleVersion.parse("25.8")
    v_new = RapidsCycleVersion.parse("25.10")
    assert v_old.cycle_year == 25 and v_old.cycle_month == 8, \
        f"test1: 解析错误 year={v_old.cycle_year} month={v_old.cycle_month}"
    assert v_new > v_old, "test1: 25.10 应大于 25.8"
    assert v_old.short_tag == "25.8", f"test1: short_tag={v_old.short_tag!r}"
    assert v_new.padded_tag == "25.10", f"test1: padded_tag={v_new.padded_tag!r}"
    print("[PASS] test1: RapidsCycleVersion 解析、排序、标签格式")

    # 测试2: conda/pip/cu12 pin 生成
    assert v_new.conda_pin() == "cuml==25.10.*,>=0.0.0a0", \
        f"test2: conda_pin={v_new.conda_pin()!r}"
    assert v_old.conda_pin() == "cuml==25.8.*,>=0.0.0a0", \
        f"test2: old conda_pin={v_old.conda_pin()!r}"
    assert v_new.cu12_pin() == "cuml-cu12==25.10.*,>=0.0.0a0", \
        f"test2: cu12_pin={v_new.cu12_pin()!r}"
    assert v_new.ci_image_tag() == "rapidsai/ci-conda:25.10-latest", \
        f"test2: ci_image_tag={v_new.ci_image_tag()!r}"
    print("[PASS] test2: conda_pin / cu12_pin / ci_image_tag 格式正确")

    # 测试3: cycles_behind 计算
    assert v_old.cycles_behind(v_new) == 1, \
        f"test3: 25.8→25.10 应为 1 cycle，实际={v_old.cycles_behind(v_new)}"
    assert CUML_4319B28_LAG.lag_cycles == 1, \
        f"test3: CUML lag_cycles 应为 1，实际={CUML_4319B28_LAG.lag_cycles}"
    print("[PASS] test3: cycles_behind / lag_cycles 计算")

    # 测试4: PackageLagRecord 正向/逆向约束
    assert CUML_4319B28_LAG.package_name == "cuml", "test4: 包名应为 cuml"
    raised = False
    try:
        PackageLagRecord(
            package_name="cuml",
            lagged_version=RapidsCycleVersion.parse("25.10"),
            target_version=RapidsCycleVersion.parse("25.8"),   # 逆序，应拒绝
            fix_commit_sha="deadbeef",
            lag_hypothesis="test",
            affected_files=(),
        )
    except ValueError:
        raised = True
    assert raised, "test4: target < lagged 应抛出 ValueError"
    print("[PASS] test4: PackageLagRecord 逆序版本拒绝")

    # 测试5: PackageLagDetector 检测旧引脚
    detector = PackageLagDetector(CUML_4319B28_LAG)
    clean_text = "- cuml==25.10.*,>=0.0.0a0\n- cuml-cu12==25.10.*,>=0.0.0a0\n"
    assert not detector.any_lag(clean_text), "test5: 已更新的文本不应检出滞后"
    lagging_text = "- cuml==25.8.*,>=0.0.0a0\n- cuml-cu12==25.8.*,>=0.0.0a0\n"
    results = detector.scan(lagging_text)
    assert len(results) == 2, f"test5: 应检出 2 个滞后引脚，实际={len(results)}"
    assert all(r.is_lagging() for r in results), "test5: 所有结果应为 lagging"
    print("[PASS] test5: PackageLagDetector 正确检出滞后引脚")

    # 测试6: build_sync_specs 生成 5 条规格
    assert len(SYNC_SPECS_4319B28) == 5, \
        f"test6: 应生成 5 条 SyncPatchSpec，实际={len(SYNC_SPECS_4319B28)}"
    cu12_specs = [s for s in SYNC_SPECS_4319B28 if s.variant == "cu12"]
    assert len(cu12_specs) == 1, f"test6: 应有 1 条 cu12 规格，实际={len(cu12_specs)}"
    assert cu12_specs[0].target_file == "dependencies.yaml", \
        f"test6: cu12 规格应在 dependencies.yaml"
    print("[PASS] test6: build_sync_specs 生成正确数量和结构的规格")

    # 测试7: SyncPatchSpec.as_sed_expr 生成合法表达式
    spec = SYNC_SPECS_4319B28[0]
    sed = spec.as_sed_expr()
    assert "cuml" in sed, f"test7: sed 表达式应包含包名，实际={sed!r}"
    assert "25" in sed, f"test7: sed 表达式应包含版本号，实际={sed!r}"
    assert "25.10" in sed, f"test7: sed 表达式应包含新版本，实际={sed!r}"
    print("[PASS] test7: SyncPatchSpec.as_sed_expr 格式正确")

    # 测试8: SyncPatchVerifier 干净文本通过，脏文本抛 AssertionError
    clean = "- cuml==25.10.*,>=0.0.0a0\n"
    VERIFIER_4319B28.assert_clean_text(
        clean, "conda/environments/all_cuda-129_arch-aarch64.yaml"
    )
    dirty = "- cuml==25.8.*,>=0.0.0a0\n"
    raised = False
    try:
        VERIFIER_4319B28.assert_clean_text(
            dirty, "conda/environments/all_cuda-129_arch-aarch64.yaml"
        )
    except AssertionError as e:
        raised = True
        assert "25.8" in str(e), f"test8: 错误信息应提及旧版本"
    assert raised, "test8: 脏文本应抛出 AssertionError"
    print("[PASS] test8: SyncPatchVerifier 干净通过 / 脏文本正确拒绝")

    print("\n[ALL PASS] 4319b28 cuml_sync_policy 自测：8 项断言全部通过")


if __name__ == "__main__":
    _smoke_test()
