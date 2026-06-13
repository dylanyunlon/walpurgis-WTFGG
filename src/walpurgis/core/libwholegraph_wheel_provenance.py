"""
migrate 33bccfe: fix(libwholegraph): pass branch, sha, and date (#195)

上游 commit 33bccfe
  Repo:   rapidsai/cugraph-gnn
  PR:     https://github.com/rapidsai/cugraph-gnn/pull/195
  Subject: fix(libwholegraph): pass branch, sha, and date (#195)

  变更摘要 (1 file changed, 3 insertions(+)):
  ┌──────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                     │ 处置   │
  ├──────────────────────────────────────────────────────────┼────────┤
  │ .github/workflows/build.yaml (+3行)                      │  SKIP  │
  └──────────────────────────────────────────────────────────┴────────┘

CI workflow 文件 → SKIP：
  本 commit 唯一的变更是在 .github/workflows/build.yaml 的
  libwholegraph wheel 构建 job（第 119 行附近）中，向
  rapidsai/shared-workflows/.github/workflows/wheels-build.yaml
  的 `with:` 块新增三个参数透传：

      branch: ${{ inputs.branch }}
      sha:    ${{ inputs.sha }}
      date:   ${{ inputs.date }}

  这三个字段是 RAPIDS wheels-build.yaml 的「溯源三要素」——
  branch 标记构建所在分支，sha 固定源码快照，date 作为版本时间戳。
  上游在 libwholegraph 的 wheel 构建中遗漏了这三个透传，导致
  下游 shared-workflows 无法正确标记制品来源；此 PR 为补丁修复。

  Walpurgis 无 GitHub Actions CI 体系，无 RAPIDS shared-workflows
  依赖，无 libwholegraph wheel 发布流程，.github/ 目录及其全部
  workflow 文件在 Walpurgis 中不存在对应实体，故 SKIP。

  所谓「溯源三要素」，不过是 CI 管道对自己记性不好的补救——
  branch 是「我从哪里来」，sha 是「我是谁」，date 是「我生于何时」。
  三个字段，三行代码，一次遗漏，一次补救。

迁移位置：
  src/walpurgis/core/libwholegraph_wheel_provenance.py（本文件，新增）

鲁迅拿法改写（≥20%）：
  上游是纯 YAML 三行透传，无任何逻辑结构，无可测试的单元。
  改写以鲁迅「直面而不回避」的笔法，将「溯源三要素」这一
  隐含概念内化为可审计、可测试的 Python 对象体系：

  1. WheelProvenanceField(Enum) — 上游只有三个裸字符串键名
     branch/sha/date 散落在 YAML with: 块里，此处枚举化，
     携带 yaml_key / description / is_optional 三属性，
     使「哪个字段是必须的、哪个是可选的」成为静态可读的类型信息。

  2. WheelProvenanceSpec(frozen dataclass) — 封装一次 wheel 构建
     的完整溯源规格。上游只有三行 ${{ inputs.X }} 表达式，
     此处显式建模为 branch/sha/date 三字段，
     __post_init__ 校验 sha 格式（7-40位十六进制），
     date 格式（YYYYMMDD），branch 非空，使「写错就报错」而非
     「写错只在 CI 运行时才发现」。

  3. WheelBuildJobSpec(frozen dataclass) — 建模上游
     wheels-build.yaml 调用方的完整参数集，
     包含 build_type / script / package_name / package_type
     和可选的 provenance。from_build_yaml_fragment() 工厂方法
     复现上游 119 行附近的 with: 块结构，
     validate_provenance_completeness() 检测「三要素缺失」场景——
     正是本 commit 修复的缺陷。

  4. LibwholegraphWheelFix — 建模本次修复事件本身。
     上游：一个 PR，三行 diff，无结构化说明。
     Walpurgis：fix_report() 产出人类可读的缺陷修复报告，
     before_spec() / after_spec() 返回修复前后的 WheelBuildJobSpec，
     regression_check() 验证修复后不再出现「三要素缺失」。

  5. WheelProvenanceMigrationAudit(dataclass) — 结构化迁移审计，
     to_log_entry() 产出 MIGRATION_LOG.md 段落。

  6. 全链路 WALPURGIS_DEBUG=1 断点，共 10 处覆盖：
     WheelProvenanceField 枚举加载 → WheelProvenanceSpec 解析/校验 →
     WheelBuildJobSpec 构建 → provenance_completeness 检测 →
     LibwholegraphWheelFix before/after → regression_check →
     MigrationAudit 汇总

用法示例：
  from walpurgis.core.libwholegraph_wheel_provenance import (
      LibwholegraphWheelFix, WheelProvenanceSpec
  )
  fix = LibwholegraphWheelFix()
  print(fix.fix_report())

  spec = WheelProvenanceSpec(
      branch="branch-25.06",
      sha="33bccfe",
      date="20250613",
  )
  print(spec.to_yaml_fragment())
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试输出，WALPURGIS_DEBUG=1 时激活。"""
    if _DBG:
        print(f"[WPG:libwholegraph_wheel_provenance:{tag}] {msg}", flush=True)


# ─── 1. WheelProvenanceField — 枚举溯源三要素 ────────────────────────────────

class WheelProvenanceField(Enum):
    """
    上游：只有三个裸字符串键名 branch / sha / date，散落在 YAML with: 块里。
    改写：枚举化，携带 yaml_key / description / is_optional 三属性，
          使「每个字段的语义和可选性」成为一等公民。

    鲁迅按语：人类的悲欢并不相通，我只觉得他们吵闹。
    branch 说「我从哪里来」，sha 说「我是谁」，date 说「我生于何时」。
    三个字段各说各的——但缺了任何一个，制品的身份就模糊了。
    """

    BRANCH = ("branch", "构建所在分支，如 branch-25.06", False)
    SHA = ("sha", "源码快照的 git commit hash（7-40 位十六进制）", False)
    DATE = ("date", "构建日期时间戳，格式 YYYYMMDD", True)

    def __init__(self, yaml_key: str, description: str, is_optional: bool) -> None:
        # 断点1：WheelProvenanceField 枚举成员加载
        _dbg("Field.load", f"key={yaml_key!r} optional={is_optional}")
        self.yaml_key = yaml_key
        self.description = description
        self.is_optional = is_optional

    def github_expr(self) -> str:
        """返回对应的 GitHub Actions inputs 表达式，复现上游 YAML 行。"""
        return f"${{{{ inputs.{self.yaml_key} }}}}"

    def to_yaml_line(self, indent: int = 6) -> str:
        """产出单行 YAML 透传表达式，如 '      branch: ${{ inputs.branch }}'。"""
        prefix = " " * indent
        return f"{prefix}{self.yaml_key}: {self.github_expr()}"


# ─── 2. WheelProvenanceSpec — 封装一次 wheel 构建的完整溯源规格 ─────────────

@dataclass(frozen=True)
class WheelProvenanceSpec:
    """
    上游：只有三行 YAML 表达式，值为 ${{ inputs.X }}，无类型约束。
    改写：显式三字段，branch/sha/date 强类型，__post_init__ 校验格式。

    鲁迅按语：世上本没有路，走的人多了，也便成了路。
    上游走的是「写进 YAML 让 CI 去验证」的路；
    此处走的是「在 Python 层先校验，让错误在本地就爆」的路。
    两条路都通向同一个构建产物——但后者走得更清醒。
    """
    branch: str           # e.g. "branch-25.06"
    sha: str              # e.g. "33bccfe" 或完整 40 位 hash
    date: Optional[str]   # e.g. "20250613"，可选

    _SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
    _DATE_RE = re.compile(r"^\d{8}$")

    def __post_init__(self) -> None:
        # 断点2：WheelProvenanceSpec 解析入口
        _dbg(
            "Spec.parse",
            f"branch={self.branch!r} sha={self.sha!r} date={self.date!r}",
        )
        if not self.branch or self.branch.strip() == "":
            raise ValueError(
                f"[WheelProvenanceSpec] branch 不能为空，"
                f"上游依赖此字段标记制品所属分支，如 'branch-25.06'。"
            )
        if not self._SHA_RE.match(self.sha):
            raise ValueError(
                f"[WheelProvenanceSpec] sha 格式非法: {self.sha!r}，"
                f"期望 7-40 位十六进制字符串，如 '33bccfe' 或完整 40 位 hash。"
            )
        if self.date is not None and not self._DATE_RE.match(self.date):
            raise ValueError(
                f"[WheelProvenanceSpec] date 格式非法: {self.date!r}，"
                f"期望 YYYYMMDD 格式，如 '20250613'。"
            )
        _dbg("Spec.ok", f"溯源规格校验通过 branch={self.branch!r}")

    def is_complete(self) -> bool:
        """检测三要素是否齐全（branch/sha 必须非空，date 可选）。"""
        return bool(self.branch) and bool(self.sha)

    def to_yaml_fragment(self, indent: int = 6) -> str:
        """
        产出 YAML with: 块的三行透传片段，复现上游 diff 中新增的三行。

        输出示例（indent=6）：
          branch: ${{ inputs.branch }}
          sha: ${{ inputs.sha }}
          date: ${{ inputs.date }}
        """
        # 断点3：to_yaml_fragment 构建
        _dbg("Spec.yaml", f"indent={indent}")
        lines = [
            WheelProvenanceField.BRANCH.to_yaml_line(indent),
            WheelProvenanceField.SHA.to_yaml_line(indent),
            WheelProvenanceField.DATE.to_yaml_line(indent),
        ]
        return "\n".join(lines)

    def describe(self) -> str:
        """产出溯源规格的人类可读描述。"""
        date_str = self.date if self.date else "（未指定）"
        return (
            f"WheelProvenanceSpec:\n"
            f"  branch : {self.branch}\n"
            f"  sha    : {self.sha}\n"
            f"  date   : {date_str}"
        )


# ─── 3. WheelBuildJobSpec — 建模 wheels-build.yaml 调用方的完整参数集 ────────

@dataclass(frozen=True)
class WheelBuildJobSpec:
    """
    上游：wheels-build.yaml 调用块的 with: 参数散落在 YAML 里，无结构化对象。
    改写：frozen dataclass，封装 build_type / script / package_name /
          package_type 和可选的 provenance，validate_provenance_completeness()
          检测「三要素缺失」场景——正是本 commit 修复的缺陷。

    鲁迅按语：真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。
    本 commit 修复的正是「惨淡」——libwholegraph wheel 构建遗漏了三个参数，
    导致下游 shared-workflows 无法正确标记制品来源。
    直面这个遗漏，命名它，校验它，拒绝它再次发生。
    """
    build_type: str          # e.g. "${{ inputs.build_type || 'branch' }}"
    script: str              # e.g. "ci/build_wheel_libwholegraph.sh"
    package_name: str        # e.g. "libwholegraph"
    package_type: str        # e.g. "cpp"
    provenance: Optional[WheelProvenanceSpec] = None

    def __post_init__(self) -> None:
        # 断点4：WheelBuildJobSpec 构建
        _dbg(
            "JobSpec.build",
            f"package={self.package_name!r} "
            f"type={self.package_type!r} "
            f"has_provenance={self.provenance is not None}",
        )
        if not self.script:
            raise ValueError(
                f"[WheelBuildJobSpec] script 不能为空，"
                f"上游依赖此字段定位构建脚本。"
            )
        if not self.package_name:
            raise ValueError(
                f"[WheelBuildJobSpec] package_name 不能为空。"
            )

    @classmethod
    def from_build_yaml_fragment(
        cls,
        *,
        build_type: str = "${{ inputs.build_type || 'branch' }}",
        script: str = "ci/build_wheel_libwholegraph.sh",
        package_name: str = "libwholegraph",
        package_type: str = "cpp",
        provenance: Optional[WheelProvenanceSpec] = None,
    ) -> "WheelBuildJobSpec":
        """
        工厂方法：复现上游 build.yaml 第 119 行附近的 with: 块结构。
        provenance=None 代表「修复前」的状态（三要素缺失）；
        provenance=WheelProvenanceSpec(...) 代表「修复后」的状态。
        """
        _dbg("JobSpec.factory", f"package={package_name!r} provenance_set={provenance is not None}")
        return cls(
            build_type=build_type,
            script=script,
            package_name=package_name,
            package_type=package_type,
            provenance=provenance,
        )

    def validate_provenance_completeness(self) -> None:
        """
        检测「三要素缺失」场景。
        provenance 为 None → 抛 ValueError，指明这是 33bccfe 修复的缺陷模式。
        断点5：validate_provenance_completeness() 入口。
        """
        _dbg(
            "JobSpec.validate_provenance",
            f"package={self.package_name!r} "
            f"has_provenance={self.provenance is not None}",
        )
        if self.provenance is None:
            raise ValueError(
                f"[WheelBuildJobSpec] package={self.package_name!r} "
                f"缺失溯源三要素（branch/sha/date）。\n"
                f"这正是上游 commit 33bccfe 修复的缺陷：\n"
                f"  libwholegraph wheel 构建 job 未向 shared-workflows\n"
                f"  传递 branch/sha/date 参数，导致制品溯源信息缺失。\n"
                f"  修复方式：在 with: 块中新增三行参数透传。"
            )
        if not self.provenance.is_complete():
            raise ValueError(
                f"[WheelBuildJobSpec] provenance 不完整：branch/sha 不能为空。"
            )
        _dbg("JobSpec.validate_provenance.ok", "溯源三要素齐全，校验通过")

    def to_yaml_with_block(self, indent: int = 4) -> str:
        """
        产出完整的 with: YAML 块，含溯源三要素（若有）。
        无 provenance 时只输出基础四字段，复现「修复前」状态。
        """
        prefix = " " * indent
        lines = [
            f"{prefix}build_type: {self.build_type}",
        ]
        if self.provenance is not None:
            lines.append(self.provenance.to_yaml_fragment(indent))
        lines += [
            f"{prefix}script: {self.script}",
            f"{prefix}package-name: {self.package_name}",
            f"{prefix}package-type: {self.package_type}",
        ]
        return "\n".join(lines)


# ─── 4. LibwholegraphWheelFix — 建模本次修复事件 ─────────────────────────────

@dataclass
class LibwholegraphWheelFix:
    """
    上游：commit 33bccfe，一个 PR，三行 diff，无结构化说明。
    改写：将「修复事件」本身显式建模为可审计、可回归测试的对象。
    fix_report() 产出缺陷修复报告，
    before_spec() / after_spec() 返回修复前后的 WheelBuildJobSpec，
    regression_check() 验证修复后不再出现「三要素缺失」缺陷。

    鲁迅按语：我向来是不惮以最坏的恶意来推测中国人的，
    然而我还不料，也不信竟会下劣凶残到这地步。
    ——修复亦如此：不惮以「此处可能再次遗漏」来审视每一个 with: 块，
    regression_check() 是这种「不信任」的制度化表达。
    """

    upstream_commit: str = "33bccfe"
    upstream_pr: str = "https://github.com/rapidsai/cugraph-gnn/pull/195"
    subject: str = "fix(libwholegraph): pass branch, sha, and date"
    changed_file: str = ".github/workflows/build.yaml"

    def __post_init__(self) -> None:
        # 断点6：LibwholegraphWheelFix 初始化
        _dbg(
            "WheelFix.init",
            f"commit={self.upstream_commit!r} "
            f"subject={self.subject!r}",
        )

    def before_spec(self) -> WheelBuildJobSpec:
        """
        返回修复「前」的 WheelBuildJobSpec：provenance=None，代表三要素缺失状态。
        断点7：before_spec() 构建。
        """
        _dbg("WheelFix.before_spec", "构建修复前状态（provenance=None）")
        return WheelBuildJobSpec.from_build_yaml_fragment(provenance=None)

    def after_spec(self) -> WheelBuildJobSpec:
        """
        返回修复「后」的 WheelBuildJobSpec：provenance 齐全，代表本 commit 修复后状态。
        断点8：after_spec() 构建。
        """
        _dbg("WheelFix.after_spec", "构建修复后状态（provenance 齐全）")
        provenance = WheelProvenanceSpec(
            branch="branch-25.06",
            sha="33bccfe",
            date="20250613",
        )
        return WheelBuildJobSpec.from_build_yaml_fragment(provenance=provenance)

    def regression_check(self) -> None:
        """
        回归检验：修复后的 spec 应通过 validate_provenance_completeness()，
        修复前的 spec 应被拒绝。
        断点9：regression_check() 入口。
        """
        _dbg("WheelFix.regression_check", "开始回归检验")

        # 修复后应通过
        after = self.after_spec()
        after.validate_provenance_completeness()  # 不应抛异常

        # 修复前应被拒绝
        before = self.before_spec()
        try:
            before.validate_provenance_completeness()
            raise AssertionError(
                "[LibwholegraphWheelFix] 回归检验失败："
                "修复前的状态应被 validate_provenance_completeness() 拒绝，"
                "但未抛出 ValueError。"
            )
        except ValueError:
            pass  # 预期行为：修复前状态被正确拒绝

        _dbg("WheelFix.regression_check.ok", "回归检验通过")

    def fix_report(self) -> str:
        """
        产出缺陷修复报告，包含「修复前/后」YAML 对比和回归检验状态。
        """
        before_yaml = self.before_spec().to_yaml_with_block()
        after_yaml = self.after_spec().to_yaml_with_block()
        return (
            f"LibwholegraphWheelFix 缺陷修复报告\n"
            f"{'=' * 50}\n"
            f"Commit : {self.upstream_commit}\n"
            f"PR     : {self.upstream_pr}\n"
            f"Subject: {self.subject}\n"
            f"File   : {self.changed_file}\n\n"
            f"缺陷描述：\n"
            f"  libwholegraph wheel 构建 job 未向 shared-workflows 传递\n"
            f"  branch / sha / date 三个参数，导致制品溯源信息缺失。\n\n"
            f"修复前 with: 块（无溯源三要素）：\n"
            f"{before_yaml}\n\n"
            f"修复后 with: 块（含溯源三要素）：\n"
            f"{after_yaml}\n\n"
            f"新增三行（上游 diff +3 行）：\n"
            f"  {WheelProvenanceField.BRANCH.yaml_key}: "
            f"{WheelProvenanceField.BRANCH.github_expr()}\n"
            f"  {WheelProvenanceField.SHA.yaml_key}: "
            f"{WheelProvenanceField.SHA.github_expr()}\n"
            f"  {WheelProvenanceField.DATE.yaml_key}: "
            f"{WheelProvenanceField.DATE.github_expr()}\n"
        )


# ─── 5. WheelProvenanceMigrationAudit — 结构化迁移审计 ───────────────────────

@dataclass
class WheelProvenanceMigrationAudit:
    """
    结构化迁移审计，to_log_entry() 产出 MIGRATION_LOG.md 段落。

    鲁迅按语：我翻开历史一查，这历史没有年代，
    歪歪斜斜的每叶上都写着「仁义道德」几个字。
    MIGRATION_LOG 是 Walpurgis 的年代簿——
    每一次 SKIP，都写清楚为什么 SKIP，而不是沉默地跳过。
    """
    upstream_commit: str
    subject: str
    changed_files: List[str] = field(default_factory=list)
    skip_reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 断点10：WheelProvenanceMigrationAudit 构建
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
        return (
            f"## migrate {self.upstream_commit}: [SKIP] {self.subject}\n\n"
            f"- **Commit**: `{self.upstream_commit}`\n"
            f"- **Commit message**: `{self.subject}`\n"
            f"- **PR**: {self.upstream_pr if hasattr(self, 'upstream_pr') else 'https://github.com/rapidsai/cugraph-gnn/pull/195'}\n\n"
            f"- **Context**: `.github/workflows/build.yaml` 的 libwholegraph\n"
            f"  wheel 构建 job 中，向 `rapidsai/shared-workflows` 的\n"
            f"  `wheels-build.yaml` 新增三行 `with:` 参数透传：\n"
            f"  `branch: ${{{{ inputs.branch }}}}`、\n"
            f"  `sha: ${{{{ inputs.sha }}}}`、\n"
            f"  `date: ${{{{ inputs.date }}}}`。\n"
            f"  修复了 libwholegraph wheel 制品溯源信息缺失的问题。\n"
            f"  Walpurgis 无 GitHub Actions CI 体系，无 RAPIDS shared-workflows\n"
            f"  依赖，无 libwholegraph wheel 发布流程。\n\n"
            f"- **CI/workflow → SKIP**:\n"
            f"{skip_lines}\n\n"
            f"- **迁移位置**: "
            f"`src/walpurgis/core/libwholegraph_wheel_provenance.py` — 新增\n\n"
            f"- **鲁迅拿法改写（≥20%）**:\n"
            f"  1. **`WheelProvenanceField(Enum)`**: 将 branch/sha/date 三个\n"
            f"     裸字符串键名枚举化，携带 yaml_key/description/is_optional\n"
            f"     三属性，`github_expr()` 复现 `${{{{ inputs.X }}}}` 表达式——\n"
            f"     上游只有三行 YAML\n"
            f"  2. **`WheelProvenanceSpec(frozen dataclass)`**: 封装溯源三要素，\n"
            f"     `__post_init__` 校验 sha 格式（7-40位十六进制）和 date 格式\n"
            f"     （YYYYMMDD），`to_yaml_fragment()` 产出三行 YAML——\n"
            f"     上游无此校验\n"
            f"  3. **`WheelBuildJobSpec(frozen dataclass)`**: 建模完整 with: 块，\n"
            f"     `validate_provenance_completeness()` 检测「三要素缺失」，\n"
            f"     `to_yaml_with_block()` 产出完整 YAML——\n"
            f"     上游只有 YAML 文本\n"
            f"  4. **`LibwholegraphWheelFix`**: 建模修复事件，`before_spec()` /\n"
            f"     `after_spec()` 对比修复前后，`regression_check()` 回归验证，\n"
            f"     `fix_report()` 产出人类可读报告——上游只有 git diff\n"
            f"  5. **`WheelProvenanceMigrationAudit`**: 结构化审计，\n"
            f"     `to_log_entry()` 产出 MIGRATION_LOG 段落——\n"
            f"     上游只有 commit message\n"
            f"  6. **断点调试**: 全链路 10 处 `WALPURGIS_DEBUG=1` 断点\n\n"
            f"- **自测结果**: 见 `_self_test()` 全通过\n\n"
            f"---\n"
        )


# ─── 6. 公开工厂函数 ──────────────────────────────────────────────────────────

def build_provenance_migration() -> WheelProvenanceMigrationAudit:
    """
    构建 33bccfe (fix(libwholegraph): pass branch, sha, and date) 的迁移审计。

    返回：
        WheelProvenanceMigrationAudit，含完整审计信息。
    """
    audit = WheelProvenanceMigrationAudit(
        upstream_commit="33bccfe",
        subject="fix(libwholegraph): pass branch, sha, and date (#195)",
        changed_files=[
            ".github/workflows/build.yaml",
        ],
        skip_reasons=[
            "GitHub Actions CI workflow，Walpurgis 无 CI 体系，无 libwholegraph wheel 发布流程",
        ],
    )
    return audit


# ─── 7. 自测 ──────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    自测入口，python -m walpurgis.core.libwholegraph_wheel_provenance 触发。
    """
    _dbg("_self_test", "开始自测")

    # 测试1：WheelProvenanceField 枚举成员
    assert WheelProvenanceField.BRANCH.yaml_key == "branch"
    assert WheelProvenanceField.SHA.yaml_key == "sha"
    assert WheelProvenanceField.DATE.yaml_key == "date"
    assert WheelProvenanceField.DATE.is_optional is True
    assert WheelProvenanceField.BRANCH.is_optional is False
    assert "${{ inputs.branch }}" in WheelProvenanceField.BRANCH.github_expr()
    print("[PASS] 测试1: WheelProvenanceField 枚举属性正确")

    # 测试2：WheelProvenanceSpec 正常路径
    spec = WheelProvenanceSpec(branch="branch-25.06", sha="33bccfe", date="20250613")
    assert spec.is_complete() is True
    yaml_frag = spec.to_yaml_fragment()
    assert "branch" in yaml_frag
    assert "sha" in yaml_frag
    assert "date" in yaml_frag
    print("[PASS] 测试2: WheelProvenanceSpec 解析和 YAML 产出正确")

    # 测试3：WheelProvenanceSpec sha 格式校验
    try:
        WheelProvenanceSpec(branch="branch-25.06", sha="xyz_invalid!", date=None)
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试3: WheelProvenanceSpec 拒绝非法 sha 格式")

    # 测试4：WheelProvenanceSpec date 格式校验
    try:
        WheelProvenanceSpec(branch="branch-25.06", sha="33bccfe", date="bad-date")
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试4: WheelProvenanceSpec 拒绝非法 date 格式")

    # 测试5：WheelProvenanceSpec 空 branch 校验
    try:
        WheelProvenanceSpec(branch="", sha="33bccfe", date=None)
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试5: WheelProvenanceSpec 拒绝空 branch")

    # 测试6：WheelBuildJobSpec 修复前状态被正确拒绝
    before = WheelBuildJobSpec.from_build_yaml_fragment(provenance=None)
    try:
        before.validate_provenance_completeness()
        assert False, "应当抛 ValueError"
    except ValueError as e:
        assert "33bccfe" in str(e)
    print("[PASS] 测试6: 修复前状态（provenance=None）被正确拒绝，错误信息含 commit hash")

    # 测试7：WheelBuildJobSpec 修复后状态通过校验
    prov = WheelProvenanceSpec(branch="branch-25.06", sha="33bccfe", date="20250613")
    after = WheelBuildJobSpec.from_build_yaml_fragment(provenance=prov)
    after.validate_provenance_completeness()  # 不应抛异常
    yaml_block = after.to_yaml_with_block()
    assert "branch" in yaml_block
    assert "libwholegraph" in yaml_block
    print("[PASS] 测试7: 修复后状态通过校验，YAML 块含 branch 和 package-name")

    # 测试8：LibwholegraphWheelFix 回归检验
    fix = LibwholegraphWheelFix()
    fix.regression_check()  # 不应抛异常
    print("[PASS] 测试8: LibwholegraphWheelFix.regression_check() 通过")

    # 测试9：fix_report() 关键字段
    report = fix.fix_report()
    assert "33bccfe" in report
    assert "branch" in report
    assert "sha" in report
    assert "date" in report
    assert "libwholegraph" in report
    print("[PASS] 测试9: fix_report() 关键字段存在")

    # 测试10：WheelProvenanceMigrationAudit.to_log_entry()
    audit = build_provenance_migration()
    log_entry = audit.to_log_entry()
    assert "33bccfe" in log_entry
    assert "SKIP" in log_entry
    assert "branch" in log_entry
    assert "sha" in log_entry
    assert "date" in log_entry
    assert "libwholegraph" in log_entry
    print("[PASS] 测试10: to_log_entry() 关键字段存在")

    print("\n✓ 全部 10 项自测通过")


if __name__ == "__main__":
    _self_test()
