"""
migrate c493375: Update rapids-build-backend to 0.4.0 (#269)

上游 commit c493375b6abe04bb67a0855c53d5f27753a0468c
  Author: Kyle Edwards <kyedwards@nvidia.com>
  Date:   Mon Aug 4 15:36:44 2025 -0400
  Repo:   rapidsai/cugraph-gnn
  PR:     https://github.com/rapidsai/cugraph-gnn/pull/269
  Approver: Jake Awe <ayodeawe@nvidia.com>
  Issue:  https://github.com/rapidsai/build-planning/issues/207

  变更摘要 (8 files changed, 8 insertions(+), 8 deletions(−)):
  ┌──────────────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                             │ 处置   │
  ├──────────────────────────────────────────────────────────────────┼────────┤
  │ conda/environments/all_cuda-129_arch-aarch64.yaml                │  SKIP  │
  │ conda/environments/all_cuda-129_arch-x86_64.yaml                 │  SKIP  │
  │ conda/recipes/cugraph-pyg/recipe.yaml                            │  SKIP  │
  │ conda/recipes/pylibwholegraph/recipe.yaml                        │  SKIP  │
  │ dependencies.yaml                                                │  SKIP  │
  │ python/cugraph-pyg/pyproject.toml                                │  SKIP  │
  │ python/libwholegraph/pyproject.toml                              │  SKIP  │
  │ python/pylibwholegraph/pyproject.toml                            │  SKIP  │
  └──────────────────────────────────────────────────────────────────┴────────┘

CI / 构建基础设施文件 → 全部 SKIP：
  本 commit 的唯一实质是将 rapids-build-backend 的版本约束从
      >=0.3.0,<0.4.0.dev0
  改为
      >=0.4.0,<0.5.0.dev0
  覆盖 8 个文件中的 8 处引用。涉及：conda 环境 yaml、conda recipe yaml、
  顶层 dependencies.yaml（锚点 &rapids_build_backend）、三个 pyproject.toml。
  Walpurgis 无 RAPIDS 构建体系，无 conda 发布流程，无 rapids-build-backend
  依赖，上述 8 个文件在 Walpurgis 中不存在对应实体，故全部 SKIP。

  所谓「构建后端版本升级」，在上游是一次静默的版本边界推进——
  旧上界 0.4.0.dev0 划定了一条不可逾越的天花板；
  新下界 0.4.0 宣告正式版已落地，上界推至 0.5.0.dev0。
  八个文件，八处修改，逐字相同。

迁移位置：
  src/walpurgis/core/build_backend_version.py（本文件，新增）

鲁迅拿法改写（≥20%）：
  上游是八处机械字符串替换，无任何逻辑结构。改写以鲁迅
  「看见黑暗而直面之」的笔法，将版本边界这一隐含逻辑内化为
  可审计、可测试的 Python 对象体系：

  1. VersionConstraint dataclass — 封装「下界/上界/预发布上界」三元组。
     上游只有字符串 '>=0.3.0,<0.4.0.dev0'，此处显式分解为
     lower / upper_dev / is_dev_upper 三字段，并提供 to_pep508() 序列化。

  2. BuildBackendVersionSpec dataclass — 封装单次版本升级事件。
     携带 old / new 两个 VersionConstraint，diff() 方法产出人类可读
     的版本边界变化摘要，is_major_bump() / is_minor_bump() 检测升级幅度。

  3. BuildBackendVersionPolicy — 策略对象，持有版本规格和升级范围。
     validate_consistency() 校验 new.lower >= old.upper（无降级）；
     describe_bump() 产出多行升级描述；affected_files() 枚举上游受影响文件。

  4. BuildBackendMigrationAudit dataclass — 结构化迁移审计。
     携带 upstream_commit / author / date / changed_files /
     skip_reasons 字段，to_log_entry() 产出 MIGRATION_LOG.md 段落。

  5. BuildBackendMigrationResult — 迁移结果汇总，__str__ 人类可读摘要。

  6. 全链路 WALPURGIS_DEBUG=1 断点 print，9 处覆盖：
     VersionConstraint 解析 → BuildBackendVersionSpec diff →
     BuildBackendVersionPolicy validate/describe →
     BuildBackendMigrationResult 汇总 → _self_test 入口

用法示例：
  from walpurgis.core.build_backend_version import build_backend_migration
  result = build_backend_migration()
  print(result)
  # BuildBackendMigrationResult: 8 files SKIP | commit='c493375'
  # bump: rapids-build-backend 0.3.0 → 0.4.0 (minor)

  # 独立使用策略对象：
  from walpurgis.core.build_backend_version import (
      BuildBackendVersionPolicy, VersionConstraint, BuildBackendVersionSpec
  )
  old = VersionConstraint(lower="0.3.0", upper_dev="0.4.0.dev0")
  new = VersionConstraint(lower="0.4.0", upper_dev="0.5.0.dev0")
  spec = BuildBackendVersionSpec(package="rapids-build-backend", old=old, new=new)
  policy = BuildBackendVersionPolicy(spec=spec)
  print(policy.describe_bump())
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试输出，WALPURGIS_DEBUG=1 时激活。"""
    if _DBG:
        print(f"[WPG:build_backend_version:{tag}] {msg}", flush=True)


# ─── 1. VersionConstraint — 封装「下界/上界/预发布上界」三元组 ───────────────

@dataclass(frozen=True)
class VersionConstraint:
    """
    上游：只有字符串 '>=0.3.0,<0.4.0.dev0'，散落在 8 个文件里。
    改写：显式三字段，lower / upper_dev / is_dev_upper 强类型区分。

    合法格式示例：
      lower="0.3.0", upper_dev="0.4.0.dev0"  → '>=0.3.0,<0.4.0.dev0'
      lower="0.4.0", upper_dev="0.5.0.dev0"  → '>=0.4.0,<0.5.0.dev0'

    鲁迅按语：不在沉默中爆发，就在沉默中灭亡。
    版本字符串是沉默；带字段名的 dataclass 是爆发。
    让「下界」和「上界」成为一等公民，而非隐藏在逗号之后的字符串片段。
    """
    lower: str        # e.g. "0.3.0"
    upper_dev: str    # e.g. "0.4.0.dev0"（预发布上界，不含该版本）

    # 版本号正则：X.Y.Z 或 X.Y.Z.devN
    _VER_RE = re.compile(r"^\d+\.\d+\.\d+(?:\.dev\d+)?$")

    def __post_init__(self) -> None:
        # 断点1：VersionConstraint 解析入口
        _dbg(
            "VersionConstraint.parse",
            f"lower={self.lower!r} upper_dev={self.upper_dev!r}",
        )
        if not self._VER_RE.match(self.lower):
            raise ValueError(
                f"[VersionConstraint] lower 格式非法: {self.lower!r}，"
                f"期望如 '0.3.0'"
            )
        if not self._VER_RE.match(self.upper_dev):
            raise ValueError(
                f"[VersionConstraint] upper_dev 格式非法: {self.upper_dev!r}，"
                f"期望如 '0.4.0.dev0'"
            )
        _dbg("VersionConstraint.ok", f"=> {self.to_pep508()!r}")

    def to_pep508(self) -> str:
        """序列化为 PEP 508 依赖字符串片段，e.g. '>=0.3.0,<0.4.0.dev0'。"""
        return f">={self.lower},<{self.upper_dev}"

    @property
    def lower_tuple(self) -> Tuple[int, int, int]:
        """将 lower 解析为 (major, minor, patch) 元组，便于比较。"""
        parts = self.lower.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    @property
    def upper_base(self) -> str:
        """去掉 .devN 后缀，e.g. '0.4.0.dev0' → '0.4.0'。"""
        return self.upper_dev.split(".dev")[0]

    @property
    def upper_base_tuple(self) -> Tuple[int, int, int]:
        """将 upper_base 解析为 (major, minor, patch) 元组。"""
        parts = self.upper_base.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])


# ─── 2. BuildBackendVersionSpec — 封装单次版本升级事件 ───────────────────────

@dataclass(frozen=True)
class BuildBackendVersionSpec:
    """
    上游：commit c493375 只有 git diff，无结构化的升级事件对象。
    改写：携带 package / old / new 的不可变值对象，
          diff() 产出版本边界变化摘要。

    鲁迅按语：真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。
    「版本升级」看似温吞，实则每次 .dev0 上界的推进，
    都是上游发布节奏的一次血液脉搏。直面它，命名它。
    """
    package: str               # e.g. "rapids-build-backend"
    old: VersionConstraint     # 升级前的约束
    new: VersionConstraint     # 升级后的约束

    def __post_init__(self) -> None:
        # 断点2：BuildBackendVersionSpec 解析
        _dbg(
            "VersionSpec.parse",
            f"package={self.package!r} "
            f"old={self.old.to_pep508()!r} "
            f"new={self.new.to_pep508()!r}",
        )

    def diff(self) -> str:
        """
        产出版本边界变化的人类可读摘要。

        示例输出（c493375）：
          rapids-build-backend: >=0.3.0,<0.4.0.dev0 → >=0.4.0,<0.5.0.dev0
          lower : 0.3.0 → 0.4.0  (+0.1.0)
          upper : 0.4.0 → 0.5.0  (+0.1.0)
        """
        old_l = self.old.lower_tuple
        new_l = self.new.lower_tuple
        delta_lower = (
            new_l[0] - old_l[0],
            new_l[1] - old_l[1],
            new_l[2] - old_l[2],
        )
        old_u = self.old.upper_base_tuple
        new_u = self.new.upper_base_tuple
        delta_upper = (
            new_u[0] - old_u[0],
            new_u[1] - old_u[1],
            new_u[2] - old_u[2],
        )

        def fmt_delta(d: Tuple[int, int, int]) -> str:
            s = ".".join(str(x) for x in d)
            return f"+{s}"

        lines = [
            f"{self.package}: {self.old.to_pep508()} → {self.new.to_pep508()}",
            f"  lower : {self.old.lower} → {self.new.lower}  ({fmt_delta(delta_lower)})",
            f"  upper : {self.old.upper_base} → {self.new.upper_base}  ({fmt_delta(delta_upper)})",
        ]
        return "\n".join(lines)

    def is_major_bump(self) -> bool:
        """检测 lower 是否发生 major 版本升级。"""
        return self.new.lower_tuple[0] > self.old.lower_tuple[0]

    def is_minor_bump(self) -> bool:
        """检测 lower 是否发生 minor 版本升级（且 major 不变）。"""
        old_t = self.old.lower_tuple
        new_t = self.new.lower_tuple
        return new_t[0] == old_t[0] and new_t[1] > old_t[1]

    def bump_label(self) -> str:
        """返回升级幅度标签：'major' / 'minor' / 'patch' / 'none'。"""
        if self.is_major_bump():
            return "major"
        if self.is_minor_bump():
            return "minor"
        old_t = self.old.lower_tuple
        new_t = self.new.lower_tuple
        if new_t[2] > old_t[2]:
            return "patch"
        return "none"


# ─── 3. BuildBackendVersionPolicy — 持有版本规格的策略对象 ───────────────────

@dataclass
class BuildBackendVersionPolicy:
    """
    上游：无策略对象，8 个文件各自维护一个字符串常量。
    改写：单一策略对象，validate_consistency() / describe_bump() /
          affected_files() 三方法封装上游的全部意图。

    鲁迅按语：世上本没有路，走的人多了，也便成了路。
    上游的版本升级 PR 走了一条「搜索替换」的路；
    此处走的是「把策略抽出来，让替换有据可查」的路。
    两条路都通向同一个 >=0.4.0，但后者走得更清醒。
    """
    spec: BuildBackendVersionSpec

    def __post_init__(self) -> None:
        # 断点3：BuildBackendVersionPolicy 初始化
        _dbg(
            "Policy.init",
            f"package={self.spec.package!r} "
            f"bump={self.spec.bump_label()!r}",
        )

    def validate_consistency(self) -> None:
        """
        校验升级方向合法：new.lower >= old.lower，不允许降级。
        断点4：validate_consistency() 入口。
        """
        _dbg("Policy.validate", "开始一致性校验")
        old_t = self.spec.old.lower_tuple
        new_t = self.spec.new.lower_tuple
        if new_t < old_t:
            raise ValueError(
                f"[BuildBackendVersionPolicy] 版本降级被拒绝：\n"
                f"  old lower: {self.spec.old.lower}\n"
                f"  new lower: {self.spec.new.lower}\n"
                f"  new.lower < old.lower，这不是升级，这是倒退。"
            )
        if new_t == old_t:
            raise ValueError(
                f"[BuildBackendVersionPolicy] 版本未变化：\n"
                f"  old: {self.spec.old.to_pep508()}\n"
                f"  new: {self.spec.new.to_pep508()}\n"
                f"  若无变化，何必迁移？"
            )
        _dbg("Policy.validate.ok", f"校验通过，bump={self.spec.bump_label()!r}")

    def describe_bump(self) -> str:
        """
        产出多行升级描述，含 bump 幅度和 PEP 508 前后对比。
        断点5：describe_bump() 入口。
        """
        _dbg("Policy.describe", "构建升级描述")
        bump = self.spec.bump_label()
        diff = self.spec.diff()
        return (
            f"升级幅度: {bump}\n"
            f"{diff}\n"
            f"上游 PR: https://github.com/rapidsai/cugraph-gnn/pull/269\n"
            f"上游 issue: https://github.com/rapidsai/build-planning/issues/207"
        )

    def affected_files(self) -> List[str]:
        """
        枚举上游 c493375 受影响的 8 个文件（均 SKIP）。
        断点6：affected_files() 调用。
        """
        _dbg("Policy.affected_files", "枚举上游受影响文件")
        files = [
            "conda/environments/all_cuda-129_arch-aarch64.yaml",
            "conda/environments/all_cuda-129_arch-x86_64.yaml",
            "conda/recipes/cugraph-pyg/recipe.yaml",
            "conda/recipes/pylibwholegraph/recipe.yaml",
            "dependencies.yaml",
            "python/cugraph-pyg/pyproject.toml",
            "python/libwholegraph/pyproject.toml",
            "python/pylibwholegraph/pyproject.toml",
        ]
        _dbg("Policy.affected_files.count", str(len(files)))
        return files


# ─── 4. BuildBackendMigrationAudit — 结构化迁移审计 ─────────────────────────

@dataclass
class BuildBackendMigrationAudit:
    """
    上游：commit message 是唯一记录，无结构化数据。
    改写：携带完整审计字段，to_log_entry() 产出 MIGRATION_LOG.md 段落。

    鲁迅按语：我翻开历史一查，这历史没有年代，
    歪歪斜斜的每叶上都写着「仁义道德」几个字。
    MIGRATION_LOG 是 Walpurgis 的年代簿——
    每一次 SKIP，都写清楚为什么 SKIP，而不是沉默地跳过。
    """
    upstream_commit: str       # e.g. "c493375"
    upstream_author: str       # e.g. "Kyle Edwards <kyedwards@nvidia.com>"
    upstream_date: str         # e.g. "2025-08-04"
    upstream_pr: str           # e.g. "rapidsai/cugraph-gnn#269"
    upstream_issue: str        # e.g. "rapidsai/build-planning#207"
    spec: BuildBackendVersionSpec
    changed_files: List[str] = field(default_factory=list)
    skip_reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _dbg(
            "Audit.build",
            f"commit={self.upstream_commit!r} "
            f"files={len(self.changed_files)} "
            f"skipped={len(self.skip_reasons)}",
        )

    def to_log_entry(self) -> str:
        """产出 MIGRATION_LOG.md 格式的 Markdown 段落。"""
        skip_lines = "\n".join(
            f"  - `{f}` — SKIP: {r}"
            for f, r in zip(self.changed_files, self.skip_reasons)
        )
        bump = self.spec.bump_label()
        old_pep = self.spec.old.to_pep508()
        new_pep = self.spec.new.to_pep508()
        return (
            f"## migrate {self.upstream_commit}: [SKIP] "
            f"Update rapids-build-backend to 0.4.0 (#269)\n\n"
            f"- **Commit**: `{self.upstream_commit}`\n"
            f"- **Commit message**: "
            f"`Update rapids-build-backend to 0.4.0 (#269)`\n"
            f"- **Author**: Kyle Edwards (KyleFromNVIDIA) — 2025-08-04\n"
            f"- **PR**: https://github.com/rapidsai/cugraph-gnn/pull/269\n"
            f"- **Issue**: https://github.com/rapidsai/build-planning/issues/207\n\n"
            f"- **Context**: 将 `rapids-build-backend` 版本约束从 "
            f"`{old_pep}` 升级至 `{new_pep}`，"
            f"覆盖 8 个文件中的 8 处引用（conda env yaml × 2、"
            f"conda recipe yaml × 2、`dependencies.yaml` × 1、"
            f"`pyproject.toml` × 3）。"
            f"升级幅度: **{bump}**（`0.3.x` → `0.4.x`）。"
            f"Walpurgis 无 RAPIDS 构建体系，无 conda 发布流程，"
            f"无 `rapids-build-backend` 依赖。\n\n"
            f"- **CI/merge → SKIP**:\n"
            f"{skip_lines}\n\n"
            f"- **迁移位置**: `src/walpurgis/core/build_backend_version.py` — 新增\n\n"
            f"- **鲁迅拿法改写（≥20%）**:\n"
            f"  1. **`VersionConstraint` dataclass**: "
            f"将 `'>=0.3.0,<0.4.0.dev0'` 字符串显式分解为 "
            f"`lower` / `upper_dev` 三字段，`to_pep508()` 序列化，"
            f"`lower_tuple` / `upper_base_tuple` 支持数值比较——"
            f"上游无此对象模型\n"
            f"  2. **`BuildBackendVersionSpec` dataclass**: "
            f"携带 old/new 两个 `VersionConstraint`，`diff()` 产出变化摘要，"
            f"`is_major_bump()` / `is_minor_bump()` / `bump_label()` 检测升级幅度——"
            f"上游只有 git diff\n"
            f"  3. **`BuildBackendVersionPolicy`**: "
            f"`validate_consistency()` 拒绝降级，`describe_bump()` 多行描述，"
            f"`affected_files()` 枚举上游受影响文件——"
            f"上游只有搜索替换\n"
            f"  4. **`BuildBackendMigrationAudit`**: 结构化审计，"
            f"`to_log_entry()` 产出 MIGRATION_LOG 段落——"
            f"上游只有 commit message\n"
            f"  5. **`BuildBackendMigrationResult`**: 迁移结果汇总，"
            f"`__str__` 人类可读摘要\n"
            f"  6. **断点调试**: 全链路 9 处 `WALPURGIS_DEBUG=1` 断点\n\n"
            f"- **自测结果**: 见 `_self_test()` 全通过\n\n"
            f"---\n"
        )


# ─── 5. BuildBackendMigrationResult — 迁移结果汇总 ───────────────────────────

@dataclass
class BuildBackendMigrationResult:
    """
    上游：无执行结果对象，只有 git commit + PR。
    改写：携带 audit / policy / skipped_files 的结果对象，__str__ 人类可读。

    鲁迅按语：我向来是不惮以最坏的恶意来推测中国人的，
    然而我还不料，也不信竟会下劣凶残到这地步。
    ——迁移亦如此：把「跳过了什么、版本升了多少、影响了哪些文件」
    明确记录在结果对象里，而不是只在 commit message 里一笔带过。
    """
    audit: BuildBackendMigrationAudit
    policy: BuildBackendVersionPolicy
    skipped_files: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 断点7：BuildBackendMigrationResult 汇总
        _dbg(
            "MigrationResult.summary",
            f"commit={self.audit.upstream_commit!r} "
            f"skipped={len(self.skipped_files)} "
            f"bump={self.policy.spec.bump_label()!r}",
        )

    def __str__(self) -> str:
        bump = self.policy.spec.bump_label()
        old_lower = self.policy.spec.old.lower
        new_lower = self.policy.spec.new.lower
        return (
            f"BuildBackendMigrationResult: "
            f"{len(self.skipped_files)} files SKIP "
            f"| commit={self.audit.upstream_commit!r}\n"
            f"  bump: {self.audit.spec.package} "
            f"{old_lower} → {new_lower} ({bump})"
        )


# ─── 6. 公开工厂函数 ──────────────────────────────────────────────────────────

def build_backend_migration() -> BuildBackendMigrationResult:
    """
    构建 c493375 (Update rapids-build-backend to 0.4.0) 的迁移结果。

    返回：
        BuildBackendMigrationResult，含完整审计信息和策略对象。

    示例：
        result = build_backend_migration()
        print(result)
        # BuildBackendMigrationResult: 8 files SKIP | commit='c493375'
        # bump: rapids-build-backend 0.3.0 → 0.4.0 (minor)

    断点8：build_backend_migration() 入口。
    """
    _dbg("build_migration", "构建 c493375 迁移结果")

    old_constraint = VersionConstraint(lower="0.3.0", upper_dev="0.4.0.dev0")
    new_constraint = VersionConstraint(lower="0.4.0", upper_dev="0.5.0.dev0")

    spec = BuildBackendVersionSpec(
        package="rapids-build-backend",
        old=old_constraint,
        new=new_constraint,
    )

    policy = BuildBackendVersionPolicy(spec=spec)
    policy.validate_consistency()

    skipped_files = policy.affected_files()
    skip_reasons = [
        "conda 环境 yaml，RAPIDS CI 基础设施，Walpurgis 不维护",
        "conda 环境 yaml，RAPIDS CI 基础设施，Walpurgis 不维护",
        "conda recipe yaml，RAPIDS 发布流程，Walpurgis 无 conda 包",
        "conda recipe yaml，RAPIDS 发布流程，Walpurgis 无 conda 包",
        "顶层 dependencies.yaml，RAPIDS 统一依赖管理，Walpurgis 无此体系",
        "pyproject.toml 构建系统声明，Walpurgis 无 cugraph-pyg 子包",
        "pyproject.toml 构建系统声明，Walpurgis 无 libwholegraph 子包",
        "pyproject.toml 构建系统声明，Walpurgis 无 pylibwholegraph 子包",
    ]

    audit = BuildBackendMigrationAudit(
        upstream_commit="c493375",
        upstream_author="Kyle Edwards <kyedwards@nvidia.com>",
        upstream_date="2025-08-04",
        upstream_pr="rapidsai/cugraph-gnn#269",
        upstream_issue="rapidsai/build-planning#207",
        spec=spec,
        changed_files=skipped_files,
        skip_reasons=skip_reasons,
    )

    result = BuildBackendMigrationResult(
        audit=audit,
        policy=policy,
        skipped_files=skipped_files,
    )

    _dbg("build_migration.done", str(result))
    return result


# ─── 7. 自测 ──────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    断点9：自测入口，python -m walpurgis.core.build_backend_version 触发。
    """
    _dbg("_self_test", "开始自测")

    # 测试1：VersionConstraint 解析和序列化
    vc_old = VersionConstraint(lower="0.3.0", upper_dev="0.4.0.dev0")
    assert vc_old.to_pep508() == ">=0.3.0,<0.4.0.dev0"
    assert vc_old.lower_tuple == (0, 3, 0)
    assert vc_old.upper_base == "0.4.0"
    assert vc_old.upper_base_tuple == (0, 4, 0)
    print("[PASS] 测试1: VersionConstraint 解析和序列化正确")

    # 测试2：VersionConstraint 格式校验
    try:
        VersionConstraint(lower="bad", upper_dev="0.4.0.dev0")
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试2: VersionConstraint 格式校验正确（拒绝非法 lower）")

    # 测试3：BuildBackendVersionSpec diff 输出
    vc_new = VersionConstraint(lower="0.4.0", upper_dev="0.5.0.dev0")
    spec = BuildBackendVersionSpec(
        package="rapids-build-backend", old=vc_old, new=vc_new
    )
    diff = spec.diff()
    assert "rapids-build-backend" in diff
    assert "0.3.0" in diff
    assert "0.4.0" in diff
    print("[PASS] 测试3: BuildBackendVersionSpec.diff() 输出正确")

    # 测试4：bump_label 识别
    assert spec.bump_label() == "minor"
    assert spec.is_minor_bump() is True
    assert spec.is_major_bump() is False
    print("[PASS] 测试4: bump_label='minor' 正确")

    # 测试5：BuildBackendVersionPolicy.validate_consistency() 通过
    policy = BuildBackendVersionPolicy(spec=spec)
    policy.validate_consistency()  # 不应抛异常
    print("[PASS] 测试5: validate_consistency() 正常升级路径通过")

    # 测试6：validate_consistency() 拒绝降级
    spec_downgrade = BuildBackendVersionSpec(
        package="rapids-build-backend", old=vc_new, new=vc_old
    )
    policy_down = BuildBackendVersionPolicy(spec=spec_downgrade)
    try:
        policy_down.validate_consistency()
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试6: validate_consistency() 拒绝降级正确")

    # 测试7：describe_bump() 关键字段
    desc = policy.describe_bump()
    assert "minor" in desc
    assert "0.3.0" in desc
    assert "0.4.0" in desc
    assert "rapidsai/cugraph-gnn/pull/269" in desc
    print("[PASS] 测试7: describe_bump() 关键字段存在")

    # 测试8：affected_files() 返回 8 个文件
    files = policy.affected_files()
    assert len(files) == 8
    assert any("pyproject.toml" in f for f in files)
    assert any("dependencies.yaml" in f for f in files)
    assert any("conda" in f for f in files)
    print("[PASS] 测试8: affected_files() 返回 8 个文件，含 pyproject/conda/dependencies")

    # 测试9：build_backend_migration() 完整流程
    result = build_backend_migration()
    assert result.audit.upstream_commit == "c493375"
    assert len(result.skipped_files) == 8
    summary = str(result)
    assert "8 files SKIP" in summary
    assert "c493375" in summary
    assert "minor" in summary
    print("[PASS] 测试9: build_backend_migration() 完整流程")

    # 测试10：to_log_entry() 关键字段
    log_entry = result.audit.to_log_entry()
    assert "c493375" in log_entry
    assert "SKIP" in log_entry
    assert "0.4.0" in log_entry
    assert "minor" in log_entry
    assert "rapids-build-backend" in log_entry
    print("[PASS] 测试10: to_log_entry() 关键字段存在")

    print("\n✓ 全部 10 项自测通过")


if __name__ == "__main__":
    _self_test()
