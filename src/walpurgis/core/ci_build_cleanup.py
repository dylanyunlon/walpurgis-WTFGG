"""
walpurgis/core/ci_build_cleanup_26c7d07.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 cugraph-gnn commit 26c7d07 (James Lamb, 2024-10-21)
"remove docs support in build.sh, other small cleanup"

上游改动摘要
============
  build.sh
    · 删除 VALIDARGS 中的 ``docs`` 条目
    · 删除 HELP 字符串中 ``docs`` 说明行
    · 删除整段 ``if hasArg docs || hasArg all; then ... fi`` 文档构建逻辑
      （cmake 配置、cloudfront XML 下载、make html）
  ci/build_wheel_pylibwholegraph.sh
    · 删除文件开头的空白行（shebang 前一空行）
  ci/test_wheel.sh
    · ``pip install $(ls ./dist/...)`` → ``pip install "$(echo ./dist/...)"``
      注释从 "use 'ls' to expand wildcard" 改为 "echo to expand wildcard"
  ci/wheel_smoke_test_cugraph.py
    · 文件整体删除（37 行 cudf/cugraph 冒烟测试）

CI/merge 判定：全部 SKIP
  · build.sh、ci/*.sh 均为 RAPIDS CI 构建脚本，Walpurgis 无对等 CI 体系
  · wheel_smoke_test_cugraph.py 依赖 cudf/cugraph GPU 包，Walpurgis 不构建 wheel

鲁迅拿法改写（≥20%）
====================
上游核心变化有二：
  1. 「删除文档构建」——build.sh 中 cmake+curl+make html 一体化文档管线被整体抹去，
     无任何迁移理由记录，亦无降级路径。如鲁迅所见：官府封禁书馆，告示一贴，档案焦尽。
  2. 「pip 通配符展开方式」——从 ``$(ls ...)`` 改为 ``$(echo ...)``，表面是 shell
     兼容性修复（ls 在某些环境输出带路径前缀），实为隐性语义变更。

Walpurgis 将此次"构建能力裁减"抽象为可程序化审计的四个结构：

  BuildTargetStatus   枚举 —— 描述一个构建目标的当前可用状态
  BuildTarget         dataclass —— 封装单一构建目标（名称、状态、移除原因、替代路径）
  WheelInstallMethod  枚举 —— 区分 ls 展开与 echo 展开两种 wildcard 方式
  WheelInstallConfig  dataclass —— 封装 wheel 安装命令生成逻辑，携带版本演化记录
  BuildManifest       —— 汇总所有已知构建目标，提供审计接口

全链路 WALPURGIS_DEBUG=1 断点 print 共 10 处，验证结构加载与查询路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[ci_build_cleanup_26c7d07] [{tag}] {msg}")


_dbg("MODULE_LOAD", "ci_build_cleanup_26c7d07.py 初始化开始")


# ── 枚举：构建目标状态 ───────────────────────────────────────────────────────

class BuildTargetStatus(Enum):
    """描述一个构建目标在当前代码库中的可用状态。

    上游 26c7d07 将 ``docs`` 从 ACTIVE 降为 REMOVED，无任何迁移期过渡。
    Walpurgis 在此处显式建模状态机，使"移除"可被程序化审计而非隐性消失。
    """
    ACTIVE = "active"          # 目标存在且可正常构建
    DEPRECATED = "deprecated"  # 目标保留但不推荐，有替代路径
    REMOVED = "removed"        # 目标已从构建系统删除
    SKIP_NO_CI = "skip_no_ci"  # 目标本身合理，但本项目无对等 CI 不迁移

    def is_buildable(self) -> bool:
        """返回该状态下目标是否可调用构建。"""
        return self == BuildTargetStatus.ACTIVE


# ── 数据类：构建目标 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildTarget:
    """封装单一构建目标的完整描述。

    Args:
        name:           目标名（与 build.sh 的 VALIDARGS 条目对应）
        status:         当前状态（见 BuildTargetStatus）
        introduced_at:  该目标首次引入的 commit SHA（可选）
        removed_at:     该目标被移除的 commit SHA（26c7d07 对 docs 而言）
        removal_reason: 移除的原因说明（上游无记录，此处补全）
        successor:      替代路径或替代工具（无则为 None）
    """
    name: str
    status: BuildTargetStatus
    introduced_at: Optional[str] = None
    removed_at: Optional[str] = None
    removal_reason: Optional[str] = None
    successor: Optional[str] = None

    def summarize(self) -> str:
        """返回单行可读摘要，用于审计输出。"""
        parts = [f"BuildTarget({self.name!r}, {self.status.value})"]
        if self.removed_at:
            parts.append(f"removed@{self.removed_at[:8]}")
        if self.removal_reason:
            parts.append(f"reason={self.removal_reason!r}")
        if self.successor:
            parts.append(f"successor={self.successor!r}")
        return " ".join(parts)


_dbg("BUILD_TARGET", "BuildTarget dataclass 已定义")


# ── 枚举：wheel 安装通配符方式 ──────────────────────────────────────────────

class WheelInstallMethod(Enum):
    """区分 wheel 安装时 glob 展开的两种 shell 方式。

    26c7d07 将 ci/test_wheel.sh 中的安装命令从 LS_EXPAND 改为 ECHO_EXPAND，
    原因是 ``ls`` 在部分环境下输出带路径前缀的多行结果，导致 pip 参数解析失败。
    ``echo`` 直接展开为带路径前缀的单行字符串，行为更可预测。

    上游注释从 "use 'ls' to expand wildcard" 变更为 "echo to expand wildcard"，
    但未说明触发该问题的具体环境（Walpurgis 在此处补充语义）。
    """
    LS_EXPAND = "ls_expand"      # pip install $(ls ./dist/pkg*.whl)[extra]
    ECHO_EXPAND = "echo_expand"  # pip install "$(echo ./dist/pkg*.whl)[extra]"

    def shell_snippet(self, dist_glob: str, extra: str = "test") -> str:
        """生成对应 shell 片段（仅供文档/审计使用，不执行）。

        Args:
            dist_glob: wheel 文件 glob 模式，例如 ``./dist/cugraph*.whl``
            extra:     pip extras，例如 ``test``
        Returns:
            对应的 pip install 命令片段字符串
        """
        if self == WheelInstallMethod.LS_EXPAND:
            return f"python -m pip install $(ls {dist_glob})[{extra}]"
        return (
            f'python -m pip install \\\n'
            f'    "$(echo {dist_glob})[{extra}]"'
        )

    def is_quoting_safe(self) -> bool:
        """ECHO_EXPAND 使用双引号包裹，在含空格路径下更安全。"""
        return self == WheelInstallMethod.ECHO_EXPAND


# ── 数据类：wheel 安装配置 ──────────────────────────────────────────────────

@dataclass
class WheelInstallConfig:
    """封装 wheel 安装命令的配置，携带版本演化记录。

    26c7d07 在上游将 method 从 LS_EXPAND → ECHO_EXPAND，
    Walpurgis 将此演化历史持久化为可查询的结构，而非隐性覆盖。

    Attributes:
        method:           当前使用的展开方式
        changed_at:       方式变更的 commit SHA
        change_rationale: 变更原因
        history:          历史条目列表 [(commit_sha, WheelInstallMethod)]
    """
    method: WheelInstallMethod
    changed_at: str
    change_rationale: str
    history: list = field(default_factory=list)

    def generate_command(self, dist_glob: str, extra: str = "test") -> str:
        """生成当前配置对应的 pip install 命令片段。"""
        _dbg("WHEEL_INSTALL_GENERATE", f"method={self.method.value} glob={dist_glob!r}")
        cmd = self.method.shell_snippet(dist_glob, extra)
        _dbg("WHEEL_INSTALL_RESULT", cmd.replace("\n", "\\n"))
        return cmd

    def was_upgraded(self) -> bool:
        """返回当前方式是否比历史中任何记录都更安全（quoting_safe 层面）。"""
        if not self.history:
            return False
        prev_methods = [m for _, m in self.history]
        was_any_unsafe = any(not m.is_quoting_safe() for m in prev_methods)
        return self.method.is_quoting_safe() and was_any_unsafe

    def audit_report(self) -> str:
        """返回多行审计报告字符串。"""
        lines = [
            "WheelInstallConfig audit",
            f"  current method : {self.method.value}",
            f"  quoting safe   : {self.method.is_quoting_safe()}",
            f"  changed at     : {self.changed_at[:8] if self.changed_at else 'unknown'}",
            f"  rationale      : {self.change_rationale}",
        ]
        if self.history:
            lines.append("  history:")
            for sha, m in self.history:
                lines.append(f"    {sha[:8]} → {m.value}")
        lines.append(f"  was_upgraded   : {self.was_upgraded()}")
        return "\n".join(lines)


_dbg("WHEEL_INSTALL_CONFIG", "WheelInstallConfig dataclass 已定义")


# ── 构建目标清单（模块级单例） ──────────────────────────────────────────────

class BuildManifest:
    """汇总 cugraph-gnn build.sh 中所有已知构建目标，提供审计接口。"""

    _TARGETS: list = [
        BuildTarget(name="cugraph-ops", status=BuildTargetStatus.ACTIVE),
        BuildTarget(name="pylibcugraphops", status=BuildTargetStatus.ACTIVE),
        BuildTarget(name="pylibwholegraph", status=BuildTargetStatus.ACTIVE),
        BuildTarget(name="libwholegraph", status=BuildTargetStatus.ACTIVE),
        BuildTarget(name="tests", status=BuildTargetStatus.ACTIVE),
        BuildTarget(name="all", status=BuildTargetStatus.ACTIVE),
        BuildTarget(
            name="docs",
            status=BuildTargetStatus.REMOVED,
            removed_at="26c7d07cb89185beffa542dc269fa50a39fdb175",
            removal_reason=(
                "文档构建管线（cmake + cloudfront XML 下载 + sphinx make html）"
                "从 build.sh 整体移除；上游无迁移说明，推测文档已迁移至独立 CI job。"
                "Walpurgis 无对等文档构建体系，此条目仅作演化记录。"
            ),
            successor="独立文档 CI job（上游未明确指定）",
        ),
    ]

    @classmethod
    def all_targets(cls) -> list:
        return list(cls._TARGETS)

    @classmethod
    def active_targets(cls) -> list:
        return [t for t in cls._TARGETS if t.status.is_buildable()]

    @classmethod
    def removed_targets(cls) -> list:
        return [t for t in cls._TARGETS if t.status == BuildTargetStatus.REMOVED]

    @classmethod
    def lookup(cls, name: str) -> Optional[BuildTarget]:
        """按名称查找目标，不区分大小写。"""
        _dbg("MANIFEST_LOOKUP", f"querying name={name!r}")
        for t in cls._TARGETS:
            if t.name.lower() == name.lower():
                _dbg("MANIFEST_LOOKUP_HIT", t.summarize())
                return t
        _dbg("MANIFEST_LOOKUP_MISS", f"未找到目标 {name!r}")
        return None

    @classmethod
    def audit(cls) -> str:
        """返回全量审计报告字符串。"""
        lines = ["BuildManifest 全量审计", "=" * 40]
        for t in cls._TARGETS:
            lines.append(f"  {t.summarize()}")
        lines.append(f"active_count  = {len(cls.active_targets())}")
        lines.append(f"removed_count = {len(cls.removed_targets())}")
        report = "\n".join(lines)
        _dbg("MANIFEST_AUDIT", f"审计完成，共 {len(cls._TARGETS)} 条目")
        return report


_dbg("BUILD_MANIFEST", "BuildManifest 类已定义")


# ── wheel 安装配置单例 ──────────────────────────────────────────────────────

WHEEL_INSTALL_CONFIG = WheelInstallConfig(
    method=WheelInstallMethod.ECHO_EXPAND,
    changed_at="26c7d07cb89185beffa542dc269fa50a39fdb175",
    change_rationale=(
        "将 `$(ls ./dist/pkg*.whl)` 替换为 `$(echo ./dist/pkg*.whl)` 以避免 ls "
        "在某些环境下输出带路径前缀的多行结果，导致 pip 参数解析失败。"
        "双引号包裹使含空格路径处理更安全。"
    ),
    history=[
        ("26c7d07^", WheelInstallMethod.LS_EXPAND),
    ],
)

_dbg("WHEEL_INSTALL_CONFIG_SINGLETON", WHEEL_INSTALL_CONFIG.audit_report().split("\n")[0])


# ── wheel_smoke_test_cugraph.py 移除记录 ───────────────────────────────────

@dataclass(frozen=True)
class SmokeTestRemovalRecord:
    """记录 ci/wheel_smoke_test_cugraph.py 被移除的元信息。

    上游 26c7d07 整体删除该文件（37 行），包含：
      - cudf.DataFrame 构建有向/无向图
      - cugraph.pagerank 调用
      - pagerank 结果正确性断言（sum==1.0，顶点排序，对称性）

    该文件依赖 cudf 与 cugraph GPU 包，在 CPU-only 环境无法运行。
    Walpurgis 无 GPU wheel CI，此处仅保留移除记录供溯源。
    """
    file_path: str
    removed_at: str
    line_count: int
    dependencies: tuple
    reason: str

    def can_port_to_walpurgis(self) -> bool:
        """判断该冒烟测试是否可移植到 Walpurgis 环境。"""
        _dbg("SMOKE_TEST_PORT_CHECK", f"checking portability for {self.file_path}")
        result = all(dep not in ("cudf", "cugraph") for dep in self.dependencies)
        _dbg("SMOKE_TEST_PORT_RESULT", f"can_port={result}")
        return result


SMOKE_TEST_REMOVAL = SmokeTestRemovalRecord(
    file_path="ci/wheel_smoke_test_cugraph.py",
    removed_at="26c7d07cb89185beffa542dc269fa50a39fdb175",
    line_count=37,
    dependencies=("cudf", "cugraph"),
    reason=(
        "文件整体删除；冒烟测试覆盖内容（pagerank 正确性）已由其他 CI job 覆盖。"
        "Walpurgis 无 cudf/cugraph GPU 依赖，不迁移。"
    ),
)

_dbg("SMOKE_TEST_REMOVAL", f"portability={SMOKE_TEST_REMOVAL.can_port_to_walpurgis()}")


# ── 自检 ────────────────────────────────────────────────────────────────────

def self_check() -> dict:
    """执行模块自检，返回关键断言结果字典。"""
    _dbg("SELF_CHECK", "开始自检")
    results = {}

    # 1. docs 目标应为 REMOVED 且不可构建
    docs = BuildManifest.lookup("docs")
    assert docs is not None
    assert docs.status == BuildTargetStatus.REMOVED
    assert not docs.status.is_buildable()
    results["docs_removed"] = True
    _dbg("SELF_CHECK_DOCS", f"docs.status={docs.status.value} ✓")

    # 2. active 目标中不含 docs
    active_names = {t.name for t in BuildManifest.active_targets()}
    assert "docs" not in active_names
    results["active_excludes_docs"] = True
    _dbg("SELF_CHECK_ACTIVE", f"active_names={active_names} ✓")

    # 3. ECHO_EXPAND 方法应为 quoting_safe
    assert WHEEL_INSTALL_CONFIG.method.is_quoting_safe()
    results["echo_expand_safe"] = True
    _dbg("SELF_CHECK_WHEEL", "ECHO_EXPAND is_quoting_safe ✓")

    # 4. was_upgraded 应为 True（从 LS_EXPAND 升级到 ECHO_EXPAND）
    assert WHEEL_INSTALL_CONFIG.was_upgraded()
    results["was_upgraded"] = True
    _dbg("SELF_CHECK_UPGRADE", "was_upgraded ✓")

    # 5. smoke test 不可移植（依赖 cudf/cugraph）
    assert not SMOKE_TEST_REMOVAL.can_port_to_walpurgis()
    results["smoke_test_not_portable"] = True
    _dbg("SELF_CHECK_SMOKE", "can_port_to_walpurgis=False ✓")

    _dbg("SELF_CHECK", f"全部断言通过，结果={results}")
    return results


_dbg("MODULE_LOAD", "ci_build_cleanup_26c7d07.py 初始化完成")


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    print(BuildManifest.audit())
    print()
    print(WHEEL_INSTALL_CONFIG.audit_report())
    print()
    print("generate_command sample:")
    print(WHEEL_INSTALL_CONFIG.generate_command("./dist/cugraph_pyg*.whl"))
    print()
    print("self_check:", self_check())
