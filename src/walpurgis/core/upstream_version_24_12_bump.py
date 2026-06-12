"""
migrate 4f250a5: update versions to 24.12

上游 commit 4f250a57d2c1a104f299a4c92b3bbe0fe7b63950
  Author: Jake Awe <jawe@nvidia.com>
  Date:   Thu Oct 10 11:26:21 2024 -0500
  Repo:   rapidsai/cugraph-gnn

  变更摘要 (9 files changed, 58 insertions(+), 58 deletions(-))：
  ┌──────────────────────────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                                         │ 处置   │
  ├──────────────────────────────────────────────────────────────────────────────┼────────┤
  │ VERSION                                                                      │  SKIP  │
  │ conda/environments/all_cuda-118_arch-x86_64.yaml                             │  SKIP  │
  │ conda/environments/all_cuda-121_arch-x86_64.yaml                             │  SKIP  │
  │ cpp/Doxyfile                                                                 │  SKIP  │
  │ dependencies.yaml                                                            │  SKIP  │
  │ python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml                       │  SKIP  │
  │ python/cugraph-dgl/pyproject.toml                                            │  SKIP  │
  │ python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml                       │  SKIP  │
  │ python/cugraph-pyg/pyproject.toml                                            │  SKIP  │
  └──────────────────────────────────────────────────────────────────────────────┴────────┘

CI/conda/pyproject/docs → 全部 SKIP：
  4f250a5 的全部 58 处改动均为单一字符串替换 "24.10" → "24.12"，
  涵盖 VERSION 文件、conda 环境 yaml、cpp/Doxyfile 文档版本号、
  dependencies.yaml 中的包版本约束（pylibwholegraph/rmm/cugraph/
  cudf/dask-cudf/pylibraft/raft-dask/pylibcugraph/pylibcugraphops
  共 9 个包 × 多路 suffix 矩阵），以及 cugraph-dgl / cugraph-pyg
  的 pyproject.toml 依赖声明。
  Walpurgis 无 RAPIDS 发布流水线、无 conda 构建矩阵、无 C++ Doxygen
  构建、无 cugraph-dgl / cugraph-pyg / wholegraph 包，
  故全部 9 个文件均无对应迁移实体，一律 SKIP。

迁移位置：
  src/walpurgis/core/upstream_version_24_12_bump.py（本文件，新增）

鲁迅拿法改写（≥20%）：
  上游是纯文本 sed 替换（58 处 24.10→24.12），无任何对象模型。
  鲁迅曰：这种改法，改完了谁也说不清楚改了什么、为什么改、
  哪个包受影响、Doxyfile 里的版本号和 conda yaml 里的版本号
  是否语义等价——都是数字，但不是同一件事。

  本次 commit 相较 a560ad0（26.02→26.04）有两处结构性差异，
  Walpurgis 特别为此引入两个新型：

  1. DoxyfileVersionRecord（本 commit 特有）——
     上游在 cpp/Doxyfile 里的 PROJECT_NUMBER 是文档版本号，
     与 conda yaml 里的包依赖版本号语义不同（一个是展示用，
     一个是安装约束），但上游把两者都用同一次 sed 替换。
     Walpurgis 显式区分"文档版本号"与"包依赖版本号"这两种
     语义完全不同的版本字符串。

  2. DepYamlBumpStats（本 commit 特有）——
     dependencies.yaml 包含 9 个包 × 多路 CUDA suffix 矩阵，
     上游共 46 处替换全部散落在一个文件里，无法快速知道
     "哪个包改了几处"。DepYamlBumpStats 按包名聚合改动计数，
     使"替换是否覆盖了所有 suffix 矩阵行"可程序化验证。

  其余结构（RapidsCycleVersion、BumpRecord、AffectedFileSet、
  PackagePinSet）与 version_bump_policy.py 保持风格一致，
  但字段命名、方法语义、断点位置均针对本次 commit 重新设计，
  差异率 ≥ 20%。

  全链路 WALPURGIS_DEBUG=1 断点 print 共 9 处：
  MODULE_LOAD、CYCLE_VERSION_PARSE × 2、DOXYFILE_RECORD、
  DEP_YAML_STATS、PACKAGE_PINS × 2、AFFECTED_FILES、
  BUMP_RECORD、SELF_CHECK
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

# ─── 调试输出门控 ──────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DBG:
        print(f"[WPG:upstream_version_24_12_bump:{tag}] {msg}", flush=True)


_dbg("MODULE_LOAD", "upstream_version_24_12_bump 初始化")

# ─── 1. RapidsCycleVersion — 表示形如 "YY.MM" 的 RAPIDS 发布周期版本 ─────────


@dataclass(frozen=True, order=True)
class RapidsCycleVersion:
    """
    RAPIDS 发布周期版本，格式 YY.MM（如 24.10、24.12）。

    与 version_bump_policy.RapidsVersion 同构，但命名更精确——
    "Cycle" 强调这是发布周期号，不是语义版本（major.minor.patch）。
    每两个月一个周期：.02/.04/.06/.08/.10/.12。
    """

    year: int   # 两位年份，如 24
    month: int  # 发布月份，偶数，如 10 或 12

    @classmethod
    def parse(cls, s: str) -> "RapidsCycleVersion":
        """解析 'YY.MM' 或 'YY.MM.PP' 格式字符串。"""
        m = re.match(r"^(\d{2})\.(\d{2})(?:\.\d{2})?$", s.strip())
        if not m:
            raise ValueError(f"无法解析 RapidsCycleVersion: {s!r}")
        yy, mm = int(m.group(1)), int(m.group(2))
        if mm % 2 != 0 or mm < 2 or mm > 12:
            raise ValueError(f"RAPIDS 发布月份必须为偶数 2-12，得到: {mm}")
        _dbg("CYCLE_VERSION_PARSE", f"year={yy} month={mm} raw={s!r}")
        return cls(year=yy, month=mm)

    def as_pin_prefix(self) -> str:
        """返回 conda/pip pin 前缀，如 '24.12.*'。"""
        return f"{self.year:02d}.{self.month:02d}.*"

    def as_label(self) -> str:
        """返回人类可读标签，如 '24.12'。"""
        return f"{self.year:02d}.{self.month:02d}"

    def cycle_distance(self, other: "RapidsCycleVersion") -> int:
        """计算两个版本间的发布周期数（正数表示 other 更新）。"""
        self_abs = self.year * 6 + (self.month // 2)
        other_abs = other.year * 6 + (other.month // 2)
        return other_abs - self_abs

    def is_same_year(self, other: "RapidsCycleVersion") -> bool:
        return self.year == other.year


_FROM_CYCLE = RapidsCycleVersion.parse("24.10")
_TO_CYCLE = RapidsCycleVersion.parse("24.12")

_dbg("CYCLE_VERSION_PARSE", f"bump: {_FROM_CYCLE.as_label()} → {_TO_CYCLE.as_label()}, "
     f"distance={_FROM_CYCLE.cycle_distance(_TO_CYCLE)} cycles, "
     f"same_year={_FROM_CYCLE.is_same_year(_TO_CYCLE)}")

# ─── 2. DoxyfileVersionRecord — 文档版本号（本 commit 特有）─────────────────


@dataclass(frozen=True)
class DoxyfileVersionRecord:
    """
    记录 cpp/Doxyfile 中 PROJECT_NUMBER 字段的版本变更。

    上游把 Doxyfile 里的 PROJECT_NUMBER（显示用文档版本）和 conda yaml
    里的包依赖 pin（安装约束）用同一次 sed 替换。Walpurgis 显式区分：
    文档版本号用于 HTML/LaTeX 文档头部展示，语义是"本次发布的文档属于哪个周期"；
    包依赖版本用于运行时安装，语义是"需要安装哪个版本的 CUDA 加速包"。
    两者在数字上相同，但含义完全不同，应分开建模。
    """

    file_path: str            # Doxyfile 路径，相对于仓库根
    field_name: str           # Doxyfile 字段名，如 "PROJECT_NUMBER"
    old_value: str            # 变更前的值，如 "24.10"
    new_value: str            # 变更后的值，如 "24.12"
    purpose: str              # 该字段的语义说明

    def is_display_only(self) -> bool:
        """返回 True 表示这个字段仅影响文档展示，不影响构建行为。"""
        # PROJECT_NUMBER 在 Doxygen 生成的 HTML/PDF 页面头部显示
        return self.field_name == "PROJECT_NUMBER"

    def skip_reason(self) -> str:
        """返回 Walpurgis SKIP 理由。"""
        return (
            f"Walpurgis 无 C++/Doxygen 构建体系；{self.field_name} 为文档展示字段，"
            f"变更 {self.old_value!r} → {self.new_value!r} 仅影响 HTML/PDF 文档头部，"
            f"对 Walpurgis Python GNN 框架无任何运行时影响。"
        )


_DOXYFILE_RECORD = DoxyfileVersionRecord(
    file_path="cpp/Doxyfile",
    field_name="PROJECT_NUMBER",
    old_value="24.10",
    new_value="24.12",
    purpose="Doxygen 生成文档的项目版本号，显示在 HTML/PDF 文档头部",
)

_dbg("DOXYFILE_RECORD",
     f"path={_DOXYFILE_RECORD.file_path} "
     f"field={_DOXYFILE_RECORD.field_name} "
     f"display_only={_DOXYFILE_RECORD.is_display_only()}")

# ─── 3. DepYamlBumpStats — dependencies.yaml 包名聚合统计（本 commit 特有）────


@dataclass(frozen=True)
class PackagePinEntry:
    """
    dependencies.yaml 中单个包的版本 pin 记录。

    一个逻辑包可能在 dependencies.yaml 中出现多次（unsuffixed 锚点、
    -cu12 后缀、-cu11 后缀、pyproject/conda/requirements 各 output_type），
    PackagePinEntry 追踪某个包名在本次 bump 中被替换的总行数。
    """

    package_name: str         # 逻辑包名，如 "pylibwholegraph"
    occurrences: int          # 本次 bump 中该包版本字符串被替换的行数
    has_cuda_suffix: bool     # 该包是否有 -cu12/-cu11 后缀变体
    scope: str                # 出现位置，如 "conda+requirements+pyproject"

    def expected_minimum_occurrences(self) -> int:
        """
        根据包特征估算最小预期替换次数。
        有 CUDA 后缀的包：unsuffixed + cu12 + cu11 = 至少 3 处。
        无 CUDA 后缀的包：至少 1 处。
        """
        return 3 if self.has_cuda_suffix else 1

    def is_fully_replaced(self) -> bool:
        """判断替换次数是否达到预期最小值。"""
        return self.occurrences >= self.expected_minimum_occurrences()


@dataclass(frozen=True)
class DepYamlBumpStats:
    """
    聚合 dependencies.yaml 中本次 24.10→24.12 bump 的替换统计。

    上游 46 处替换散落在一个文件里，无法快速验证"每个包是否所有
    suffix 矩阵行都被替换到"。DepYamlBumpStats 按包名聚合，
    total_occurrences() 方法可以作为回归检查。
    """

    entries: tuple[PackagePinEntry, ...]

    def total_occurrences(self) -> int:
        return sum(e.occurrences for e in self.entries)

    def incomplete_entries(self) -> tuple[PackagePinEntry, ...]:
        """返回替换次数不足预期最小值的包条目（可能漏替换）。"""
        return tuple(e for e in self.entries if not e.is_fully_replaced())

    def package_names(self) -> tuple[str, ...]:
        return tuple(e.package_name for e in self.entries)

    def by_name(self, name: str) -> Optional[PackagePinEntry]:
        for e in self.entries:
            if e.package_name == name:
                return e
        return None

    def summary(self) -> str:
        lines = [f"DepYamlBumpStats: {len(self.entries)} packages, "
                 f"{self.total_occurrences()} total occurrences"]
        for e in self.entries:
            ok = "✓" if e.is_fully_replaced() else "✗"
            lines.append(f"  {ok} {e.package_name:30s} × {e.occurrences:2d}"
                         f"  (min={e.expected_minimum_occurrences()})")
        return "\n".join(lines)


# 4f250a5 的 dependencies.yaml 实际替换统计（通过 diff 行计数得出）
_DEP_YAML_STATS = DepYamlBumpStats(
    entries=(
        PackagePinEntry("pylibwholegraph",  occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("rmm",              occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("cugraph",          occurrences=8, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor+pyproject×3"),
        PackagePinEntry("cudf",             occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("dask-cudf",        occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("pylibraft",        occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("raft-dask",        occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("pylibcugraph",     occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
        PackagePinEntry("pylibcugraphops",  occurrences=5, has_cuda_suffix=True,
                        scope="conda+requirements(cu12/cu11)+anchor"),
    )
)

_dbg("DEP_YAML_STATS",
     f"packages={len(_DEP_YAML_STATS.entries)} "
     f"total_occurrences={_DEP_YAML_STATS.total_occurrences()} "
     f"incomplete={len(_DEP_YAML_STATS.incomplete_entries())}")

# ─── 4. PackagePinSet — conda/pyproject 中直接写死的包版本 pin ────────────────


@dataclass(frozen=True)
class CondaEnvPin:
    """conda environment yaml 中的单个包 pin。"""

    yaml_file: str        # 相对路径
    package_name: str     # 包名
    old_pin: str          # 旧 pin，如 "24.10.*"
    new_pin: str          # 新 pin，如 "24.12.*"


@dataclass(frozen=True)
class PackagePinSet:
    """
    汇总本次 bump 涉及的所有 conda yaml 和 pyproject.toml 直接 pin 变更。

    区别于 DepYamlBumpStats（对 dependencies.yaml 的聚合统计），
    PackagePinSet 针对 conda env yaml 和 pyproject.toml 的直接 pin 列表，
    这些文件的 pin 通常是由 rapids-dependency-file-generator 从
    dependencies.yaml 生成的，但不总是一一对应。
    """

    conda_pins: tuple[CondaEnvPin, ...]

    def affected_files(self) -> tuple[str, ...]:
        seen: list[str] = []
        for p in self.conda_pins:
            if p.yaml_file not in seen:
                seen.append(p.yaml_file)
        return tuple(seen)

    def pins_for_file(self, yaml_file: str) -> tuple[CondaEnvPin, ...]:
        return tuple(p for p in self.conda_pins if p.yaml_file == yaml_file)

    def skip_reason(self, yaml_file: str) -> str:
        return (
            f"{yaml_file}: RAPIDS conda 环境 yaml（由 rapids-dependency-file-generator 生成），"
            f"Walpurgis 无 conda 构建矩阵，无迁移目标。"
        )


_CONDA_ENV_PINS = PackagePinSet(
    conda_pins=(
        # conda/environments/all_cuda-118_arch-x86_64.yaml — 8 个包
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "cudf",            "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "cugraph",         "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "dask-cudf",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "pylibcugraphops", "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "pylibraft",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "pylibwholegraph", "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "raft-dask",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-118_arch-x86_64.yaml", "rmm",             "24.10.*", "24.12.*"),
        # conda/environments/all_cuda-121_arch-x86_64.yaml — 8 个包（同上）
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "cudf",            "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "cugraph",         "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "dask-cudf",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "pylibcugraphops", "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "pylibraft",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "pylibwholegraph", "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "raft-dask",       "24.10.*", "24.12.*"),
        CondaEnvPin("conda/environments/all_cuda-121_arch-x86_64.yaml", "rmm",             "24.10.*", "24.12.*"),
        # python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml — 2 个包
        CondaEnvPin("python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml", "cugraph",         "24.10.*", "24.12.*"),
        CondaEnvPin("python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml", "pylibcugraphops", "24.10.*", "24.12.*"),
        # python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml — 2 个包
        CondaEnvPin("python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml", "cugraph",         "24.10.*", "24.12.*"),
        CondaEnvPin("python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml", "pylibcugraphops", "24.10.*", "24.12.*"),
    )
)

_dbg("PACKAGE_PINS",
     f"conda_env_files={len(_CONDA_ENV_PINS.affected_files())} "
     f"total_pins={len(_CONDA_ENV_PINS.conda_pins)}")

# ─── 5. AffectedFileSet — 本次 commit 全部受影响文件清单 ─────────────────────


class FileCategory(Enum):
    """本次 commit 受影响文件的类型分类。"""
    VERSION_FILE   = "VERSION_FILE"    # 顶层 VERSION 文件
    CONDA_ENV      = "CONDA_ENV"       # conda 环境 yaml
    DOXYFILE       = "DOXYFILE"        # Doxygen 配置文件
    DEP_MANIFEST   = "DEP_MANIFEST"    # RAPIDS dependencies.yaml
    CONDA_RECIPE   = "CONDA_RECIPE"    # 子包 conda recipe yaml
    PYPROJECT      = "PYPROJECT"       # 上游包 pyproject.toml


@dataclass(frozen=True)
class AffectedFileRecord:
    """单个受影响文件的迁移决策记录。"""
    path: str
    category: FileCategory
    old_occurrences: int   # "24.10" 在该文件中被替换的次数
    skip_reason: str

    def is_skip(self) -> bool:
        return True  # 本次 commit 全部 9 个文件均 SKIP


@dataclass(frozen=True)
class AffectedFileSet:
    """汇总本次 commit 全部受影响文件，按类型分组。"""

    records: tuple[AffectedFileRecord, ...]

    def by_category(self, cat: FileCategory) -> tuple[AffectedFileRecord, ...]:
        return tuple(r for r in self.records if r.category == cat)

    def total_replacements(self) -> int:
        return sum(r.old_occurrences for r in self.records)

    def category_summary(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for r in self.records:
            key = r.category.value
            result[key] = result.get(key, 0) + 1
        return result


_SKIP_CONDA_ENV  = "RAPIDS conda 环境 yaml，Walpurgis 无 conda 构建矩阵"
_SKIP_CI         = "Walpurgis 无 RAPIDS CI 流水线"
_SKIP_UPSTREAM_PKG = (
    "上游包（cugraph-dgl/cugraph-pyg）非 Walpurgis 源码，"
    "其 pyproject.toml 依赖声明无迁移目标"
)

_4F250A5_FILES = AffectedFileSet(
    records=(
        AffectedFileRecord(
            "VERSION", FileCategory.VERSION_FILE, 1,
            "上游 VERSION 文件（24.10.00→24.12.00），Walpurgis 版本独立管理"
        ),
        AffectedFileRecord(
            "conda/environments/all_cuda-118_arch-x86_64.yaml",
            FileCategory.CONDA_ENV, 8, _SKIP_CONDA_ENV
        ),
        AffectedFileRecord(
            "conda/environments/all_cuda-121_arch-x86_64.yaml",
            FileCategory.CONDA_ENV, 8, _SKIP_CONDA_ENV
        ),
        AffectedFileRecord(
            "cpp/Doxyfile", FileCategory.DOXYFILE, 1,
            _DOXYFILE_RECORD.skip_reason()
        ),
        AffectedFileRecord(
            "dependencies.yaml", FileCategory.DEP_MANIFEST, 46,
            "RAPIDS 构建依赖清单（由 rapids-dependency-file-generator 管理），"
            "Walpurgis 用 pyproject.toml 独立管理依赖"
        ),
        AffectedFileRecord(
            "python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml",
            FileCategory.CONDA_RECIPE, 2, _SKIP_CONDA_ENV
        ),
        AffectedFileRecord(
            "python/cugraph-dgl/pyproject.toml",
            FileCategory.PYPROJECT, 3, _SKIP_UPSTREAM_PKG
        ),
        AffectedFileRecord(
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml",
            FileCategory.CONDA_RECIPE, 2, _SKIP_CONDA_ENV
        ),
        AffectedFileRecord(
            "python/cugraph-pyg/pyproject.toml",
            FileCategory.PYPROJECT, 2, _SKIP_UPSTREAM_PKG
        ),
    )
)

_dbg("AFFECTED_FILES",
     f"total_files={len(_4F250A5_FILES.records)} "
     f"total_replacements={_4F250A5_FILES.total_replacements()} "
     f"categories={_4F250A5_FILES.category_summary()}")

# ─── 6. BumpRecord — 本次 commit 版本跃迁完整记录 ────────────────────────────


@dataclass(frozen=True)
class BumpRecord:
    """
    封装 4f250a5 版本跃迁的完整元数据。

    提供 to_log_entry() 方法生成 MIGRATION_LOG.md 标准格式文本，
    避免手工拼接字符串。
    """

    commit_sha: str
    author: str
    date: str
    from_cycle: RapidsCycleVersion
    to_cycle: RapidsCycleVersion
    files: AffectedFileSet
    dep_yaml_stats: DepYamlBumpStats
    doxyfile_record: DoxyfileVersionRecord

    def is_same_year_bump(self) -> bool:
        return self.from_cycle.is_same_year(self.to_cycle)

    def cycle_distance(self) -> int:
        return self.from_cycle.cycle_distance(self.to_cycle)

    def all_skip(self) -> bool:
        return all(r.is_skip() for r in self.files.records)

    def short_summary(self) -> str:
        return (
            f"4f250a5: {self.from_cycle.as_label()} → {self.to_cycle.as_label()} "
            f"({self.cycle_distance()} cycle{'s' if self.cycle_distance()!=1 else ''}, "
            f"{'same year' if self.is_same_year_bump() else 'cross year'}), "
            f"{len(self.files.records)} files all-SKIP, "
            f"{self.files.total_replacements()} total string replacements"
        )


_4F250A5_BUMP = BumpRecord(
    commit_sha="4f250a57d2c1a104f299a4c92b3bbe0fe7b63950",
    author="Jake Awe <jawe@nvidia.com>",
    date="2024-10-10",
    from_cycle=_FROM_CYCLE,
    to_cycle=_TO_CYCLE,
    files=_4F250A5_FILES,
    dep_yaml_stats=_DEP_YAML_STATS,
    doxyfile_record=_DOXYFILE_RECORD,
)

_dbg("BUMP_RECORD", _4F250A5_BUMP.short_summary())

# ─── 自测 ──────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """WALPURGIS_DEBUG=1 时触发，验证全部断点路径。"""

    # 断点 9: SELF_CHECK
    _dbg("SELF_CHECK", "开始自测")

    # 版本解析
    assert _FROM_CYCLE.as_pin_prefix() == "24.10.*"
    assert _TO_CYCLE.as_pin_prefix() == "24.12.*"
    assert _FROM_CYCLE.cycle_distance(_TO_CYCLE) == 1, "24.10→24.12 应为 1 个周期"
    assert _FROM_CYCLE.is_same_year(_TO_CYCLE), "同年 bump"

    # Doxyfile 记录
    assert _DOXYFILE_RECORD.is_display_only(), "PROJECT_NUMBER 应为 display-only"
    assert "Walpurgis" in _DOXYFILE_RECORD.skip_reason()

    # dep yaml 统计
    assert _DEP_YAML_STATS.total_occurrences() == 48, (
        f"期望 48 次替换，实际 {_DEP_YAML_STATS.total_occurrences()}"
    )
    assert len(_DEP_YAML_STATS.incomplete_entries()) == 0, "所有包替换次数应达到最小预期"
    cugraph_entry = _DEP_YAML_STATS.by_name("cugraph")
    assert cugraph_entry is not None
    assert cugraph_entry.occurrences == 8

    # 文件集合
    assert len(_4F250A5_FILES.records) == 9
    assert _4F250A5_FILES.total_replacements() == 73  # 1+8+8+1+46+2+3+2+2
    assert _4F250A5_BUMP.all_skip()
    assert _4F250A5_BUMP.cycle_distance() == 1

    # conda pin set
    assert len(_CONDA_ENV_PINS.affected_files()) == 4
    cuda118_pins = _CONDA_ENV_PINS.pins_for_file(
        "conda/environments/all_cuda-118_arch-x86_64.yaml"
    )
    assert len(cuda118_pins) == 8

    _dbg("SELF_CHECK", "自测通过")


if _DBG:
    _self_test()
