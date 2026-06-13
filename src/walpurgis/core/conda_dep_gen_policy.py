"""
conda_dep_gen_policy.py
=======================
迁移自 upstream cugraph-gnn e43ac6b (Kyle Edwards, 2025-04-11)
"Remove nvidia and dask channels (#231)"

上游变更摘要（6 files changed, 1 insertion(+), 40 deletions(-)）:

  .pre-commit-config.yaml
    rapids-dependency-file-generator  v1.17.0 → v1.19.0

  conda/environments/all_cuda-128_arch-aarch64.yaml
  conda/environments/all_cuda-128_arch-x86_64.yaml
    channels 块：删除 `dask/label/dev`、删除 `nvidia`

  dependencies.yaml
    channels 块：删除 `dask/label/dev`、删除 `nvidia`
    若干 files: 节下尾部空行清除（30 处 `-` 删除）

  python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-aarch64.yaml
  python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-x86_64.yaml
    channels 块：同上删除两 channel

删除这两个 channel 的理由（PR #231 正文）:
  - `nvidia` channel: 自 CUDA 11 时代遗留；RAPIDS 25.x 起所有 NVIDIA
    运行时包已通过 conda-forge 分发，`nvidia` channel 提供的包与
    conda-forge 重叠且版本落后，strict channel_priority 下会引入歧义。
  - `dask/label/dev`: dask 开发版 channel，仅在测试上游 dask 兼容性时
    临时引入，现已无需，继续保留会导致 CI 拉取未发布的 dask 开发包。

鲁迅拿法改写（≥20%）:
  上游仅是 YAML 手工删行 + pre-commit 版本号字符串替换，
  无任何策略声明，改一处必须记住改全部 6 个文件——如同鲁迅所言
  「从来如此，便对么？」六处散点改动，没有中心契约，
  下一个维护者读 git log 才知道「哦，原来这两个 channel 被删了」。

  Walpurgis 将此次删除的意图建模为可查询的策略声明：

  1. CondaChannelRisk enum: 描述一个 channel 被删除的根因类型
     (LEGACY_OVERLAP / DEV_SNAPSHOT / VENDOR_REDIRECT / DEPRECATED)
  2. DeprecatedChannelRecord dataclass: 一条 channel 废弃事实
     (name, risk, since_commit, reason, affects_files)
  3. DeprecatedChannelRegistry: 注册表，持有所有已废弃 channel 记录，
     提供 is_deprecated() / audit_yaml() / assert_clean() 接口
  4. DepFileGenVersion dataclass: 封装 rapids-dependency-file-generator
     版本跃迁（v1.17.0 → v1.19.0），记录版本号变更的契约
  5. DependenciesYamlCleaner: 检测并清除 dependencies.yaml 中
     「文件节末尾多余空行」问题（此 commit 顺手清理了 30 处）
  6. 全链路 WALPURGIS_DEBUG=1 断点（6 处）

Walpurgis 迁移位置: src/walpurgis/core/conda_dep_gen_policy.py（新建）

CI/conda YAML 文件本体 → SKIP:
  Walpurgis 无 RAPIDS conda 构建矩阵，不维护
  conda/environments/ 和 python/cugraph-pyg/conda/ 目录。
  策略语义迁移为本 Python 模块；见 MIGRATION_LOG.md。

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import pdb
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 调试开关：WALPURGIS_DEBUG=1 时各阶段设置断点
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(label: str) -> None:
    """统一调试断点入口。仅在 WALPURGIS_DEBUG=1 时触发。"""
    if _DEBUG:
        print(f"[_dbg] {label}")
        pdb.set_trace()


# ---------------------------------------------------------------------------
# 1. CondaChannelRisk  ——  channel 被废弃的根因分类
#    上游 PR #231 删除两个 channel，各有不同根因；枚举化使原因可被程序查询。
# ---------------------------------------------------------------------------

class CondaChannelRisk(Enum):
    """conda channel 被废弃的根因类型。

    上游直接删行，没有分类；Walpurgis 将「为什么删」建模为枚举，
    使未来的 channel audit 可以按根因过滤。
    """
    # 历史遗留：曾经必需，现已由其他 channel（如 conda-forge）覆盖
    LEGACY_OVERLAP = "legacy_overlap"
    # 开发快照：仅用于测试上游开发版包，不应出现在稳定 CI 环境
    DEV_SNAPSHOT = "dev_snapshot"
    # 厂商重定向：厂商包已迁移到 conda-forge，原 vendor channel 冗余
    VENDOR_REDIRECT = "vendor_redirect"
    # 正式弃用：channel 维护者已宣布废弃
    DEPRECATED = "deprecated"

    def risk_level(self) -> int:
        """返回风险等级（1=低, 3=高），用于 audit 排序。"""
        return {
            CondaChannelRisk.LEGACY_OVERLAP: 2,
            CondaChannelRisk.DEV_SNAPSHOT:   3,
            CondaChannelRisk.VENDOR_REDIRECT: 2,
            CondaChannelRisk.DEPRECATED:      3,
        }[self]

    def description(self) -> str:
        """返回人类可读的根因说明。"""
        return {
            CondaChannelRisk.LEGACY_OVERLAP:  "历史遗留，现已由 conda-forge 覆盖",
            CondaChannelRisk.DEV_SNAPSHOT:    "开发快照，包含未发布版本，不应进入稳定 CI",
            CondaChannelRisk.VENDOR_REDIRECT: "厂商包已迁移至 conda-forge，原 channel 冗余",
            CondaChannelRisk.DEPRECATED:      "channel 维护者已正式废弃",
        }[self]


# ---------------------------------------------------------------------------
# 2. DeprecatedChannelRecord  ——  一条 channel 废弃事实
#    上游删行即删，Walpurgis 将「删了什么、为什么删、影响哪些文件」写成台账。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeprecatedChannelRecord:
    """一个被从 conda channel 列表中移除的 channel 的完整记录。

    字段对应上游 PR #231 的实际变更内容，
    使「为什么这个 channel 不存在」在代码层面可溯源。
    """
    # channel 的 conda 名称（与 YAML 中的写法完全一致）
    name: str
    # 废弃根因
    risk: CondaChannelRisk
    # 引入废弃决定的上游 commit hash（短）
    since_commit: str
    # 完整原因说明（对应 PR body）
    reason: str
    # 受影响的上游文件列表（对应 diff 中的 6 files）
    affects_files: Tuple[str, ...] = ()

    def summary(self) -> str:
        """返回单行摘要，用于 audit 报告。"""
        files_str = f"({len(self.affects_files)} files)" if self.affects_files else ""
        return (
            f"[{self.risk.value.upper()}] channel={self.name!r}  "
            f"since={self.since_commit}  {files_str}  // {self.reason[:60]}"
        )


# ---------------------------------------------------------------------------
# 3. DeprecatedChannelRegistry  ——  已废弃 channel 注册表
#    核心接口：is_deprecated / audit_yaml / assert_clean
# ---------------------------------------------------------------------------

# 上游 e43ac6b 中被删除的两个 channel，对应 diff 中的实际行删除
_E43AC6B_AFFECTED_FILES: Tuple[str, ...] = (
    ".pre-commit-config.yaml",  # 版本升级（dep-file-generator）
    "conda/environments/all_cuda-128_arch-aarch64.yaml",
    "conda/environments/all_cuda-128_arch-x86_64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-aarch64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-x86_64.yaml",
    "dependencies.yaml",
)

# nvidia channel 的废弃记录：
#   CUDA 11 时代由 nvidia 官方 channel 提供 cudatoolkit 等包；
#   RAPIDS 23.x 起 conda-forge 接管，nvidia channel 提供的包与
#   conda-forge 重叠，strict priority 下产生歧义，必须移除。
_NVIDIA_CHANNEL_RECORD = DeprecatedChannelRecord(
    name="nvidia",
    risk=CondaChannelRisk.LEGACY_OVERLAP,
    since_commit="e43ac6b",
    reason=(
        "CUDA 11 时代遗留。RAPIDS 25.x 起 CUDA 运行时包已由 conda-forge 完整覆盖；"
        "保留 nvidia channel 在 strict channel_priority 下造成与 conda-forge 的版本歧义。"
        "PR #231 决定彻底移除，统一由 conda-forge 提供 NVIDIA 运行时包。"
    ),
    affects_files=_E43AC6B_AFFECTED_FILES[1:],  # 排除 .pre-commit-config.yaml
)

# dask/label/dev channel 的废弃记录：
#   曾用于测试 dask 上游开发版本与 cugraph-gnn 的兼容性；
#   PR #231 决定不再跟踪 dask 开发版，移除此快照 channel。
_DASK_DEV_CHANNEL_RECORD = DeprecatedChannelRecord(
    name="dask/label/dev",
    risk=CondaChannelRisk.DEV_SNAPSHOT,
    since_commit="e43ac6b",
    reason=(
        "dask 开发版快照 channel，曾用于测试 dask 上游兼容性。"
        "PR #231 确认不再需要跟踪 dask 开发版：稳定 CI 只依赖 conda-forge 的 dask 稳定版。"
        "继续保留会导致 CI 环境意外拉取未发布的 dask 开发包，破坏可复现性。"
    ),
    affects_files=_E43AC6B_AFFECTED_FILES[1:],
)


class DeprecatedChannelRegistry:
    """已废弃 conda channel 注册表。

    持有所有已从 channel 列表中移除的 channel 记录，提供：
    - is_deprecated(name): 查询某 channel 是否已被废弃
    - audit_yaml(yaml_text): 检测 YAML 文本中是否残留废弃 channel
    - assert_clean(yaml_text): CI 断言——有残留则抛 AssertionError
    """

    def __init__(self) -> None:
        # key: channel name, value: record
        self._registry: Dict[str, DeprecatedChannelRecord] = {}
        # 注册 e43ac6b 废弃的两个 channel
        self.register(_NVIDIA_CHANNEL_RECORD)
        self.register(_DASK_DEV_CHANNEL_RECORD)

    def register(self, record: DeprecatedChannelRecord) -> None:
        """注册一条废弃记录。"""
        _dbg(f"DeprecatedChannelRegistry.register: name={record.name!r}")  # 断点①
        self._registry[record.name] = record

    def is_deprecated(self, name: str) -> bool:
        """返回 True 表示 channel `name` 已被废弃，不应出现在 YAML 中。"""
        return name in self._registry

    def get_record(self, name: str) -> Optional[DeprecatedChannelRecord]:
        """返回废弃记录，若未注册则返回 None。"""
        return self._registry.get(name)

    def all_records(self) -> List[DeprecatedChannelRecord]:
        """返回所有废弃记录，按风险等级降序排列。"""
        return sorted(
            self._registry.values(),
            key=lambda r: r.risk.risk_level(),
            reverse=True,
        )

    def audit_yaml(self, yaml_text: str) -> List[Tuple[str, DeprecatedChannelRecord]]:
        """扫描 YAML 文本，返回残留的废弃 channel 列表。

        返回值: [(channel_name, record), ...]，空列表表示干净。

        上游 e43ac6b 删除这些 channel 后，任何新增/回滚都应被此方法捕获。
        """
        _dbg("DeprecatedChannelRegistry.audit_yaml 入口")  # 断点②
        found: List[Tuple[str, DeprecatedChannelRecord]] = []
        for name, record in self._registry.items():
            # 匹配 channel 行：`- nvidia` 或 `- dask/label/dev`
            # 行首可有空白，channel 名后可有空白/注释
            pattern = re.compile(
                r"^[\s\-]+{}[\s]*(?:#.*)?$".format(re.escape(name)),
                re.MULTILINE,
            )
            if pattern.search(yaml_text):
                found.append((name, record))
        return found

    def assert_clean(self, yaml_text: str, source_hint: str = "") -> None:
        """断言 YAML 中不含任何废弃 channel，否则抛 AssertionError。

        供 CI 调用：对新增/修改的 conda YAML 做门禁检查。
        """
        _dbg(f"DeprecatedChannelRegistry.assert_clean: source={source_hint!r}")  # 断点③
        violations = self.audit_yaml(yaml_text)
        if violations:
            lines = [f"[assert_clean] 发现废弃 channel（来源: {source_hint!r}）:"]
            for name, rec in violations:
                lines.append(f"  ✗ {rec.summary()}")
            raise AssertionError("\n".join(lines))

    def report(self) -> str:
        """生成人类可读的注册表报告。"""
        lines = [
            "=== DeprecatedChannelRegistry ===",
            f"  已废弃 channel 数: {len(self._registry)}",
        ]
        for rec in self.all_records():
            lines.append(f"  {rec.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. DepFileGenVersion  ——  rapids-dependency-file-generator 版本跃迁记录
#    对应 .pre-commit-config.yaml 中 rev: v1.17.0 → v1.19.0
#    上游直接改字符串，Walpurgis 将版本跃迁建模为有契约的数据结构。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepFileGenVersion:
    """rapids-dependency-file-generator 版本跃迁记录。

    上游 e43ac6b 将 pre-commit hook 从 v1.17.0 升至 v1.19.0，
    与删除 nvidia/dask channel 同步发布——因为 v1.19.0 的
    dependency-file-generator 已内置对这两个 channel 的清除支持。

    此 dataclass 将「版本号从哪里来、升到哪里去、为什么升」写成台账，
    替代原来隐藏在 .yaml 字符串里的隐式知识。
    """
    tool_name: str = "rapids-dependency-file-generator"
    repo: str = "https://github.com/rapidsai/dependency-file-generator"
    from_version: str = "v1.17.0"
    to_version: str = "v1.19.0"
    since_commit: str = "e43ac6b"
    # 升级理由：与 channel 清理同步，v1.19.0 引入对 nvidia/dask channel 的清除
    upgrade_reason: str = (
        "与 PR #231 channel 清理同步发布：v1.19.0 的 dependency-file-generator "
        "已处理 nvidia 和 dask/label/dev channel 的自动剔除逻辑，"
        "升级后工具生成的 YAML 不再包含这两个废弃 channel。"
    )
    # 使用场景：pre-commit hook，由 .pre-commit-config.yaml 调用
    usage_context: str = "pre-commit hook, rapids-dependency-file-generator --clean"

    def pre_commit_snippet(self) -> str:
        """生成等价于上游 e43ac6b 修改后的 pre-commit config 片段。"""
        _dbg("DepFileGenVersion.pre_commit_snippet")  # 断点④
        return (
            f"  - repo: {self.repo}\n"
            f"    rev: {self.to_version}\n"
            f"    hooks:\n"
            f"        - id: rapids-dependency-file-generator\n"
            f"          args: [\"--clean\"]\n"
        )

    def is_upgrade(self) -> bool:
        """返回 True 表示 to_version > from_version（按语义版本比较）。"""
        def _parse(v: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in v.lstrip("v").split("."))
        return _parse(self.to_version) > _parse(self.from_version)

    def changelog(self) -> str:
        """返回单行变更记录，用于 audit 报告。"""
        arrow = "↑" if self.is_upgrade() else "↓"
        return (
            f"{self.tool_name}  {self.from_version} {arrow} {self.to_version}  "
            f"(since={self.since_commit})"
        )


# ---------------------------------------------------------------------------
# 5. DependenciesYamlCleaner  ——  检测并清除 dependencies.yaml 文件节末尾空行
#    e43ac6b 同时清理了 dependencies.yaml 中 30 处「文件节末尾多余空行」
#    上游 PR #231 说明这是 dependency-file-generator v1.19.0 格式规范变更的体现：
#    新版生成器不再在每个 files: 节末尾输出空行，旧文件需手工对齐。
# ---------------------------------------------------------------------------

@dataclass
class DependenciesYamlCleaner:
    """检测并清除 dependencies.yaml 文件节末尾多余空行。

    上游 e43ac6b 在 dependencies.yaml 中删除了 30 处形如：
        includes:
          - some_dep
                          ← 这行空行被删
      next_section:

    这些空行是 v1.17.0 生成器的格式产物；v1.19.0 不再生成，
    因此手工清理使文件与新版生成器的输出格式一致。

    Walpurgis 将此清理操作实现为幂等的程序化变换，替代手工删行。
    """

    # 匹配「includes: 节末尾紧跟空行，然后是下一个键」的模式
    # 上游 diff 显示删除的空行均出现在 `includes:` 节的最后一个条目之后
    _TRAILING_BLANK_RE = re.compile(
        r"((?:^[ \t]*-[ \t]+\S.*\n)+)(\n+)([ \t]*\w)",
        re.MULTILINE,
    )

    def count_trailing_blanks(self, yaml_text: str) -> int:
        """统计 YAML 文本中「列表节末尾多余空行」的数量。"""
        _dbg("DependenciesYamlCleaner.count_trailing_blanks")  # 断点⑤
        count = 0
        for m in self._TRAILING_BLANK_RE.finditer(yaml_text):
            # 只统计超过 1 行的空行（1 行是正常分隔，>1 行是冗余）
            blank_lines = m.group(2).count("\n")
            if blank_lines > 1:
                count += blank_lines - 1
        return count

    def strip_trailing_blanks(self, yaml_text: str) -> str:
        """幂等清除列表节末尾的多余空行（保留 1 行正常分隔）。

        对应 e43ac6b 中 dependencies.yaml 的 30 处 `-` 删除。
        """
        def _replace(m: re.Match) -> str:
            # 保留列表内容 + 1 行分隔 + 下一键
            return m.group(1) + "\n" + m.group(3)

        result = yaml_text
        # 迭代至收敛（幂等）
        while True:
            new = self._TRAILING_BLANK_RE.sub(_replace, result)
            if new == result:
                break
            result = new
        return result

    def diff_summary(self, yaml_text: str) -> str:
        """返回清理前后的差异摘要（不执行修改）。"""
        stripped = self.strip_trailing_blanks(yaml_text)
        before_lines = yaml_text.count("\n")
        after_lines = stripped.count("\n")
        removed = before_lines - after_lines
        status = "已达到目标状态" if removed == 0 else f"可清除 {removed} 处多余空行"
        return f"DependenciesYamlCleaner: {status}（before={before_lines}L, after={after_lines}L）"


# ---------------------------------------------------------------------------
# 6. E43AC6BMigrationManifest  ——  本次迁移的完整台账
#    将上游 6 files changed 的全部语义集中在一个可查询对象中。
# ---------------------------------------------------------------------------

@dataclass
class E43AC6BMigrationManifest:
    """e43ac6b 'Remove nvidia and dask channels (#231)' 迁移台账。

    上游 6 files changed, 1 insertion(+), 40 deletions(-)  的语义摘要：
    - 1 insertion: .pre-commit-config.yaml 版本号 v1.17.0 → v1.19.0
    - 40 deletions: 6 YAML 文件中删除 `nvidia`、`dask/label/dev` channel 行
                   + dependencies.yaml 多处空行清理

    此台账将上述事实结构化，供 audit / 文档生成使用。
    """
    upstream_commit: str = "e43ac6b"
    upstream_pr: str = "#231"
    upstream_author: str = "Kyle Edwards"
    upstream_subject: str = "Remove nvidia and dask channels"
    files_changed: int = 6
    insertions: int = 1
    deletions: int = 40

    dep_file_gen: DepFileGenVersion = field(default_factory=DepFileGenVersion)
    channel_registry: DeprecatedChannelRegistry = field(
        default_factory=DeprecatedChannelRegistry
    )
    yaml_cleaner: DependenciesYamlCleaner = field(
        default_factory=DependenciesYamlCleaner
    )

    # Walpurgis CI/conda YAML 文件均 SKIP（无 RAPIDS 构建矩阵）
    skipped_files: Tuple[str, ...] = (
        "conda/environments/all_cuda-128_arch-aarch64.yaml",
        "conda/environments/all_cuda-128_arch-x86_64.yaml",
        "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-aarch64.yaml",
        "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-128_arch-x86_64.yaml",
        ".pre-commit-config.yaml",  # pre-commit 版本由 dep_file_gen 记录，文件本体 SKIP
        "dependencies.yaml",        # RAPIDS dep schema，Walpurgis 无此文件，语义迁入本模块
    )

    def self_check(self) -> List[str]:
        """执行 6 项自检，返回失败项列表（空列表 = 全部通过）。"""
        failures: List[str] = []

        # 1. dep-file-generator 版本确为升级
        if not self.dep_file_gen.is_upgrade():
            failures.append(
                f"DepFileGenVersion: {self.dep_file_gen.from_version} → "
                f"{self.dep_file_gen.to_version} 应为升级，实为降级"
            )

        # 2. nvidia channel 已注册为废弃
        if not self.channel_registry.is_deprecated("nvidia"):
            failures.append("nvidia channel 未在注册表中标记为废弃")

        # 3. dask/label/dev channel 已注册为废弃
        if not self.channel_registry.is_deprecated("dask/label/dev"):
            failures.append("dask/label/dev channel 未在注册表中标记为废弃")

        # 4. nvidia 的根因为 LEGACY_OVERLAP
        rec = self.channel_registry.get_record("nvidia")
        if rec and rec.risk != CondaChannelRisk.LEGACY_OVERLAP:
            failures.append(f"nvidia 废弃根因应为 LEGACY_OVERLAP，实为 {rec.risk}")

        # 5. dask/label/dev 的根因为 DEV_SNAPSHOT
        rec2 = self.channel_registry.get_record("dask/label/dev")
        if rec2 and rec2.risk != CondaChannelRisk.DEV_SNAPSHOT:
            failures.append(
                f"dask/label/dev 废弃根因应为 DEV_SNAPSHOT，实为 {rec2.risk}"
            )

        # 6. skipped_files 覆盖上游实际变更的 6 个文件
        if len(self.skipped_files) != self.files_changed:
            failures.append(
                f"skipped_files 数量 {len(self.skipped_files)} "
                f"≠ 上游 files_changed {self.files_changed}"
            )

        return failures

    def audit_report(self) -> str:
        """生成人类可读的迁移台账报告。"""
        _dbg("E43AC6BMigrationManifest.audit_report")  # 断点⑥
        lines = [
            "=" * 60,
            f"  迁移台账: {self.upstream_commit} PR {self.upstream_pr}",
            f"  主题: {self.upstream_subject}",
            f"  作者: {self.upstream_author}",
            f"  变更规模: {self.files_changed} files, "
            f"+{self.insertions}/-{self.deletions}",
            "-" * 60,
            "  [dep-file-gen 版本跃迁]",
            f"  {self.dep_file_gen.changelog()}",
            "  [废弃 channel 注册表]",
            self.channel_registry.report(),
            "  [SKIP 文件列表]",
        ]
        for f in self.skipped_files:
            lines.append(f"    SKIP: {f}")
        failures = self.self_check()
        lines.append("-" * 60)
        if failures:
            lines.append(f"  自检: {len(failures)} 项失败")
            for f in failures:
                lines.append(f"    ✗ {f}")
        else:
            lines.append(f"  自检: 全部通过（6 项）")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模块级单例：供外部直接 import 使用
# ---------------------------------------------------------------------------

#: 全局废弃 channel 注册表（含 e43ac6b 废弃的 nvidia + dask/label/dev）
DEPRECATED_CHANNELS = DeprecatedChannelRegistry()

#: e43ac6b 版本跃迁记录
DEP_FILE_GEN_V1_19 = DepFileGenVersion()

#: 本次迁移完整台账
E43AC6B_MANIFEST = E43AC6BMigrationManifest()


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """运行 6 项自测，覆盖本模块全部类。"""
    results: List[Tuple[str, str]] = []

    # ---- 测试 1: CondaChannelRisk 枚举语义 ----
    try:
        assert CondaChannelRisk.DEV_SNAPSHOT.risk_level() == 3
        assert CondaChannelRisk.LEGACY_OVERLAP.risk_level() == 2
        assert "conda-forge" in CondaChannelRisk.LEGACY_OVERLAP.description()
        results.append(("CondaChannelRisk 枚举语义", "PASS"))
    except AssertionError as e:
        results.append(("CondaChannelRisk 枚举语义", f"FAIL: {e}"))

    # ---- 测试 2: DeprecatedChannelRegistry.is_deprecated ----
    try:
        reg = DeprecatedChannelRegistry()
        assert reg.is_deprecated("nvidia")
        assert reg.is_deprecated("dask/label/dev")
        assert not reg.is_deprecated("conda-forge")
        assert not reg.is_deprecated("rapidsai")
        results.append(("DeprecatedChannelRegistry.is_deprecated", "PASS"))
    except AssertionError as e:
        results.append(("DeprecatedChannelRegistry.is_deprecated", f"FAIL: {e}"))

    # ---- 测试 3: DeprecatedChannelRegistry.audit_yaml（检出废弃 channel）----
    try:
        reg = DeprecatedChannelRegistry()
        dirty_yaml = (
            "channels:\n"
            "- rapidsai\n"
            "- rapidsai-nightly\n"
            "- dask/label/dev\n"
            "- conda-forge\n"
            "- nvidia\n"
        )
        found = reg.audit_yaml(dirty_yaml)
        found_names = {name for name, _ in found}
        assert "nvidia" in found_names, f"nvidia 未被检出，found={found_names}"
        assert "dask/label/dev" in found_names, f"dask/label/dev 未被检出"
        # assert_clean 应抛 AssertionError
        try:
            reg.assert_clean(dirty_yaml, source_hint="test_dirty.yaml")
            results.append(("DeprecatedChannelRegistry.audit_yaml+assert_clean", "FAIL: 未抛异常"))
        except AssertionError:
            results.append(("DeprecatedChannelRegistry.audit_yaml+assert_clean", "PASS"))
    except AssertionError as e:
        results.append(("DeprecatedChannelRegistry.audit_yaml+assert_clean", f"FAIL: {e}"))

    # ---- 测试 4: DeprecatedChannelRegistry.assert_clean（干净 YAML 不抛异常）----
    try:
        reg = DeprecatedChannelRegistry()
        clean_yaml = (
            "channels:\n"
            "- rapidsai\n"
            "- rapidsai-nightly\n"
            "- conda-forge\n"
        )
        reg.assert_clean(clean_yaml, source_hint="test_clean.yaml")  # 不应抛
        results.append(("DeprecatedChannelRegistry.assert_clean（干净 YAML）", "PASS"))
    except AssertionError as e:
        results.append(("DeprecatedChannelRegistry.assert_clean（干净 YAML）", f"FAIL: {e}"))

    # ---- 测试 5: DepFileGenVersion 版本跃迁校验 ----
    try:
        ver = DepFileGenVersion()
        assert ver.is_upgrade(), "v1.17.0 → v1.19.0 应为升级"
        snippet = ver.pre_commit_snippet()
        assert "v1.19.0" in snippet
        assert "rapids-dependency-file-generator" in snippet
        assert "--clean" in snippet
        assert "FAIL" not in ver.changelog()
        results.append(("DepFileGenVersion 版本跃迁", "PASS"))
    except AssertionError as e:
        results.append(("DepFileGenVersion 版本跃迁", f"FAIL: {e}"))

    # ---- 测试 6: E43AC6BMigrationManifest.self_check（全部通过）----
    try:
        manifest = E43AC6BMigrationManifest()
        failures = manifest.self_check()
        assert failures == [], f"self_check 失败项: {failures}"
        results.append(("E43AC6BMigrationManifest.self_check", "PASS"))
    except AssertionError as e:
        results.append(("E43AC6BMigrationManifest.self_check", f"FAIL: {e}"))

    # ---- 汇报 ----
    print("\n=== conda_dep_gen_policy.py 自测 ===")
    all_pass = True
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] {name}")
        if status != "PASS":
            all_pass = False
    total = len(results)
    print(f"\n{'全部通过' if all_pass else '存在失败项'} ({total} 项)\n")


if __name__ == "__main__":
    _selftest()
    print(E43AC6B_MANIFEST.audit_report())
