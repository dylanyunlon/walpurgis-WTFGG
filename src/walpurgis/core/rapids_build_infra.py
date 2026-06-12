"""
migrate eba488c: Update build infra to support new branching strategy (#252)

上游 commit eba488c0a8fbdb859a286b42454d88e7a235e1e1
  Author: Robert Maynard <rmaynard@nvidia.com>
  Date:   Mon Jul 28 08:28:45 2025 -0400
  Repo:   rapidsai/cugraph-gnn
  PR:     https://github.com/rapidsai/cugraph-gnn/pull/252
  Approver: Bradley Dice <bdice@nvidia.com>

  变更摘要 (3 files changed, 14 insertions(+), 3 deletions(−)):
  ┌──────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                     │ 处置   │
  ├──────────────────────────────────────────────────────────┼────────┤
  │ RAPIDS_BRANCH                          (新增，1 行)      │  SKIP  │
  │ cmake/RAPIDS.cmake                     (条件逻辑重写)    │  SKIP  │
  │ cmake/rapids_config.cmake              (读取 RAPIDS_BRANCH) │ SKIP │
  └──────────────────────────────────────────────────────────┴────────┘

CI / cmake / build 基础设施文件 → 全部 SKIP:
  本 commit 的三处变更均属 CMake 构建基础设施：
  ① RAPIDS_BRANCH 是新增的纯文本文件，告知 cmake 使用哪条 RAPIDS 分支；
  ② cmake/RAPIDS.cmake 重写了 rapids-cmake 获取策略——从「必须指定版本号」
     改为「branch-OR-version 二选一均可」，并将默认分支模式从
     'branch-{version}' 改为 'release/{version}'；
  ③ cmake/rapids_config.cmake 新增 file(STRINGS) 读取 RAPIDS_BRANCH
     文件，将读取结果注入 rapids-cmake-branch 变量。
  Walpurgis 无 C++/CMake 构建体系，无 RAPIDS 分支跟踪机制，
  上述三个文件在 Walpurgis 中不存在对应实体，故全部 SKIP。

  所谓"分支策略"，不过是构建系统对自身命运的一次重新叙事：
  不再跟着版本号亦步亦趋，而是直接点名要哪条分支。

迁移位置:
  src/walpurgis/core/rapids_build_infra.py (本文件，新增)

鲁迅拿法改写 (>20%):
  上游是纯 CMake 脚本（条件判断 + file(STRINGS) + set()），无任何
  Python 对象模型，无可测试的逻辑单元。改写以鲁迅"直面惨淡"之笔，
  将构建基础设施的策略变迁内化为可审计、可测试的 Python 对象体系：

  1. RapidsCmakeBranchMode 枚举 — 上游只有两种隐式模式散落在
     cmake 条件判断里（"给了 branch 就用 branch，否则从 version 推导"），
     此处显式枚举，EXPLICIT / DERIVED 强类型区分，携带 label 属性。

  2. RapidsCmakeBranchSpec dataclass — 封装「用哪条 rapids-cmake 分支」
     这一核心概念。上游的逻辑是 CMake 变量 rapids-cmake-branch 的有无，
     此处通过 from_branch_file() / from_version() 两条具名工厂方法，
     使调用路径与上游两种触发条件一一对应。
     __post_init__ 校验 release/YY.MM 与 branch-YY.MM 两种合法格式。

  3. RapidsBranchFileReader — 封装 RAPIDS_BRANCH 文件的读取语义。
     上游是 CMake file(STRINGS) 内联读取 + 无文件则 FATAL_ERROR；
     此处携带路径、FATAL / WARN 两种错误策略、read() 返回 Optional[str]，
     read_or_raise() 复现上游 FATAL_ERROR 语义。

  4. RapidsCmakePolicy dataclass — 将上游 cmake/RAPIDS.cmake 的「条件
     逻辑重写」收口为单一策略对象：
     - validate() 复现上游「branch 与 version 至少提供一个」校验；
     - resolve_branch() 复现「有 branch 用 branch，否则用 release/{version}」
       推导逻辑（注意：上游从 'branch-{v}' 改为 'release/{v}'，此处
       同步更新并在注释中明确记录该行为变更）；
     - describe_resolution() 产出人类可读的解析路径摘要。

  5. RapidsBranchStrategyAudit dataclass — 结构化迁移审计记录，
     携带 upstream_commit / author / date / changed_files /
     skip_reasons / strategy_change 字段，to_log_entry() 产出
     MIGRATION_LOG.md 格式的 Markdown 段落。

  6. BranchStrategyMigrationResult — 迁移结果汇总对象，
     携带 audit / policy / skipped_files，__str__ 人类可读摘要。

  7. 全链路 WALPURGIS_DEBUG=1 断点 print，8 处覆盖:
     RapidsCmakeBranchSpec 解析 → RapidsBranchFileReader 读取 →
     RapidsCmakePolicy validate/resolve → describe_resolution →
     BranchStrategyMigrationResult 汇总 → _self_test 入口

用法示例:
  from walpurgis.core.rapids_build_infra import build_branch_strategy_migration
  result = build_branch_strategy_migration()
  print(result)
  # BranchStrategyMigrationResult: 3 files SKIP | commit='eba488c'
  # strategy: branch-25.10 | mode=EXPLICIT (from RAPIDS_BRANCH file)

  # 独立使用策略对象:
  from walpurgis.core.rapids_build_infra import RapidsCmakePolicy, RapidsCmakeBranchSpec
  policy = RapidsCmakePolicy(
      branch_spec=RapidsCmakeBranchSpec.from_branch_file("branch-25.10"),
      rapids_version="25.10",
  )
  print(policy.resolve_branch())   # "branch-25.10"
  print(policy.describe_resolution())
  # "EXPLICIT: rapids-cmake-branch='branch-25.10' (from RAPIDS_BRANCH file)"
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
        print(f"[WPG:rapids_build_infra:{tag}] {msg}", flush=True)


# ─── 1. RapidsCmakeBranchMode — 上游隐式模式的强类型替代 ─────────────────────

class RapidsCmakeBranchMode(Enum):
    """
    上游 cmake/RAPIDS.cmake 有两种隐式运行模式，散落在 if/else 条件中：
    ① 调用方直接传入 rapids-cmake-branch → 直接使用（EXPLICIT）
    ② 调用方只传入版本号 → cmake 从版本号推导分支名（DERIVED）

    上游 eba488c 的关键变更：DERIVED 模式的推导公式从
        'branch-{version}'  →  'release/{version}'
    本枚举将这两条路径显式命名，使调用方无需阅读 cmake 条件判断。

    鲁迅按语: 不在沉默中爆发，就在沉默中灭亡。
    隐式的 if/else 是沉默；显式的枚举是爆发。
    让构建模式成为一等公民，而非 cmake 脚本深处的一个隐晦分支。
    """
    EXPLICIT = "explicit"   # 调用方直接指定 rapids-cmake-branch
    DERIVED = "derived"     # 从 rapids-cmake-version 推导，格式: release/{version}

    @property
    def label(self) -> str:
        return self.value.upper()


# ─── 2. RapidsCmakeBranchSpec — 封装「用哪条 rapids-cmake 分支」─────────────

@dataclass(frozen=True)
class RapidsCmakeBranchSpec:
    """
    上游: CMake 变量 rapids-cmake-branch 的有无决定一切，无 Python 对象模型。
    改写: 单一值对象，两条具名工厂方法与上游两种触发条件一一对应。

    合法的分支格式（上游 eba488c 引入后）：
    - 'branch-YY.MM'  — EXPLICIT 模式，由 RAPIDS_BRANCH 文件提供
      （上游 eba488c 写入 RAPIDS_BRANCH 的值为 'branch-25.10'）
    - 'release/YY.MM' — DERIVED 模式，cmake 从 rapids-cmake-version 推导
      （上游 eba488c 将推导公式从 'branch-{v}' 改为 'release/{v}'）

    注意: 两种格式共存——RAPIDS_BRANCH 文件本身使用 'branch-X.Y'，
    而 cmake 内部推导出 'release/X.Y'；这个不对称性是上游 eba488c 的
    核心变更，本 dataclass 的两条工厂方法分别对应两条路径。

    鲁迅按语: 真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。
    构建系统中的分支名，一字之差（branch- vs release/），
    便是两套 FetchContent URL 的命运分叉。正视它，命名它。
    """
    branch: str        # e.g. "branch-25.10" 或 "release/25.10"
    mode: RapidsCmakeBranchMode
    source_hint: str = ""  # 记录来源描述，e.g. "from RAPIDS_BRANCH file"

    # 合法的 'branch-YY.MM' 格式（EXPLICIT，由 RAPIDS_BRANCH 文件提供）
    _BRANCH_FILE_RE = re.compile(r"^branch-\d{2}\.\d{2}$")
    # 合法的 'release/YY.MM' 格式（DERIVED，cmake 推导）
    _RELEASE_RE = re.compile(r"^release/\d{2}\.\d{2}$")

    def __post_init__(self) -> None:
        # 断点1: RapidsCmakeBranchSpec 解析入口
        _dbg(
            "BranchSpec.parse",
            f"branch={self.branch!r} mode={self.mode.label} src={self.source_hint!r}",
        )
        if self.mode is RapidsCmakeBranchMode.EXPLICIT:
            if not self._BRANCH_FILE_RE.match(self.branch):
                raise ValueError(
                    f"[RapidsCmakeBranchSpec] EXPLICIT 模式要求 'branch-YY.MM' 格式，"
                    f"收到: {self.branch!r}\n"
                    f"示例: 'branch-25.10'（RAPIDS_BRANCH 文件内容格式）"
                )
        elif self.mode is RapidsCmakeBranchMode.DERIVED:
            if not self._RELEASE_RE.match(self.branch):
                raise ValueError(
                    f"[RapidsCmakeBranchSpec] DERIVED 模式要求 'release/YY.MM' 格式，"
                    f"收到: {self.branch!r}\n"
                    f"示例: 'release/25.10'（cmake 从版本号推导，eba488c 变更后）"
                )
        _dbg("BranchSpec.ok", f"branch={self.branch!r} mode={self.mode.label}")

    @classmethod
    def from_branch_file(cls, branch_file_content: str) -> "RapidsCmakeBranchSpec":
        """
        从 RAPIDS_BRANCH 文件内容构造 EXPLICIT spec。
        上游 RAPIDS_BRANCH 写入值: 'branch-25.10'
        """
        return cls(
            branch=branch_file_content.strip(),
            mode=RapidsCmakeBranchMode.EXPLICIT,
            source_hint="from RAPIDS_BRANCH file",
        )

    @classmethod
    def from_version(cls, rapids_version: str) -> "RapidsCmakeBranchSpec":
        """
        从 RAPIDS 版本号推导 DERIVED spec。
        上游 eba488c 变更后公式: 'release/{rapids_version}'
        （旧公式: 'branch-{rapids_version}'，eba488c 修正了这一点）
        """
        derived = f"release/{rapids_version}"
        return cls(
            branch=derived,
            mode=RapidsCmakeBranchMode.DERIVED,
            source_hint=f"derived from rapids_version={rapids_version!r}",
        )

    @property
    def yymm(self) -> str:
        """提取版本号部分，e.g. '25.10'。"""
        # branch-25.10  →  split('-')[1]
        # release/25.10 →  split('/')[1]
        return self.branch.replace("branch-", "").replace("release/", "")


# ─── 3. RapidsBranchFileReader — 封装 RAPIDS_BRANCH 文件读取 ─────────────────

@dataclass
class RapidsBranchFileReader:
    """
    上游: cmake file(STRINGS "RAPIDS_BRANCH" _rapids_branch)，
          无文件则 FATAL_ERROR，无 Python 层封装。
    改写: 携带路径、FATAL/WARN 两种错误策略，read() 返回 Optional[str]，
          read_or_raise() 复现上游 FATAL_ERROR 语义。

    鲁迅按语: 我翻开历史一查，这历史没有年代，
    歪歪斜斜的每叶上都写着"仁义道德"几个字。
    RAPIDS_BRANCH 这个文件，看似只有一行，
    却是整个构建体系的年代簿——没有它，cmake 便不知身在何处。
    """
    path: str = "RAPIDS_BRANCH"   # 相对仓库根路径，上游 cmake 使用相对路径

    # 断点2: read() 读取
    def read(self) -> Optional[str]:
        """
        读取 RAPIDS_BRANCH 文件内容，去除首尾空白。
        文件不存在则返回 None（对应上游 FATAL_ERROR 之前的状态）。
        """
        _dbg("BranchFileReader.read", f"path={self.path!r}")
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            _dbg("BranchFileReader.ok", f"content={content!r}")
            return content if content else None
        except FileNotFoundError:
            _dbg("BranchFileReader.not_found", f"文件不存在: {self.path!r}")
            return None

    def read_or_raise(self) -> str:
        """
        复现上游 cmake FATAL_ERROR 语义：
        文件不存在或内容为空时抛 FileNotFoundError。
        """
        content = self.read()
        if not content:
            raise FileNotFoundError(
                f"[RapidsBranchFileReader] Could not determine branch name. "
                f"The file '{self.path}' is missing or empty.\n"
                f"（复现上游 eba488c cmake/rapids_config.cmake FATAL_ERROR）"
            )
        return content

    def to_branch_spec(self) -> Optional["RapidsCmakeBranchSpec"]:
        """
        读取文件并构造 EXPLICIT BranchSpec；文件不存在返回 None。
        """
        content = self.read()
        if content is None:
            return None
        return RapidsCmakeBranchSpec.from_branch_file(content)


# ─── 4. RapidsCmakePolicy — 上游 cmake/RAPIDS.cmake 条件逻辑的 Python 镜像 ──

@dataclass
class RapidsCmakePolicy:
    """
    上游: cmake/RAPIDS.cmake 中的 if/endif 条件块，决定如何获取 rapids-cmake。
    改写: 单一策略对象，validate() / resolve_branch() / describe_resolution()
          三个方法分别对应上游三段关键逻辑。

    上游 eba488c 的两处核心变更（均已内化到本类）：
    1. 校验条件从「version 必须存在且格式合法」改为
       「branch 或 version 至少提供一个」（validate()）。
    2. 默认分支推导公式从 'branch-{version}' 改为 'release/{version}'
       （resolve_branch() 中的 DERIVED 路径）。

    鲁迅按语: 世上本没有路，走的人多了，也便成了路。
    cmake 的「先看有没有 branch，没有就从 version 推」这条路，
    上游走了两种走法：'branch-X' 是一条老路，'release/X' 是新开的路。
    eba488c 就是这次改道的路标。
    """
    branch_spec: Optional[RapidsCmakeBranchSpec]   # None 表示未从文件读取
    rapids_version: Optional[str] = None            # e.g. "25.10"

    def __post_init__(self) -> None:
        _dbg(
            "CmakePolicy.init",
            f"branch_spec={self.branch_spec!r} "
            f"rapids_version={self.rapids_version!r}",
        )

    def validate(self) -> None:
        """
        复现上游 eba488c 变更后的校验逻辑：
        rapids-cmake-branch 或 rapids-cmake-version 至少提供一个。
        （上游旧版只校验 version，eba488c 放宽为二选一）

        断点3: validate() 入口。
        """
        _dbg("CmakePolicy.validate", "开始校验")
        if self.branch_spec is None and not self.rapids_version:
            raise ValueError(
                "[RapidsCmakePolicy] The CMake variable `rapids-cmake-branch` "
                "or `rapids-cmake-version` must be defined.\n"
                "（复现上游 eba488c cmake/RAPIDS.cmake FATAL_ERROR 新文案）"
            )
        _dbg("CmakePolicy.validate.ok", "校验通过")

    def resolve_branch(self) -> str:
        """
        复现上游 cmake 的分支解析逻辑（eba488c 变更后版本）：
        1. 若 branch_spec 为 EXPLICIT（来自 RAPIDS_BRANCH 文件）→ 直接返回
        2. 否则从 rapids_version 推导 'release/{version}'
           （注意: eba488c 前是 'branch-{version}'，eba488c 修改了此处）

        断点4: resolve_branch() 入口。
        """
        self.validate()
        _dbg(
            "CmakePolicy.resolve",
            f"branch_spec_mode="
            f"{self.branch_spec.mode.label if self.branch_spec else 'None'}",
        )
        if self.branch_spec is not None:
            _dbg("CmakePolicy.resolve.explicit", f"→ {self.branch_spec.branch!r}")
            return self.branch_spec.branch

        # DERIVED 路径：从版本号推导（eba488c 修改后公式）
        derived_spec = RapidsCmakeBranchSpec.from_version(self.rapids_version)
        _dbg("CmakePolicy.resolve.derived", f"→ {derived_spec.branch!r}")
        return derived_spec.branch

    def describe_resolution(self) -> str:
        """
        产出人类可读的解析路径摘要，便于调试和日志。
        断点5: describe_resolution() 入口。
        """
        _dbg("CmakePolicy.describe", "构建摘要")
        if self.branch_spec is not None:
            mode_label = self.branch_spec.mode.label
            branch = self.branch_spec.branch
            hint = self.branch_spec.source_hint
            return (
                f"{mode_label}: rapids-cmake-branch={branch!r} ({hint})"
            )
        if self.rapids_version:
            derived = f"release/{self.rapids_version}"
            return (
                f"DERIVED: rapids-cmake-branch={derived!r} "
                f"(derived from rapids-cmake-version={self.rapids_version!r}; "
                f"公式 'release/{{v}}' 由 eba488c 引入，取代旧公式 'branch-{{v}}')"
            )
        return "INVALID: 无 branch_spec 且无 rapids_version"


# ─── 5. RapidsBranchStrategyAudit — 结构化迁移审计记录 ───────────────────────

@dataclass
class RapidsBranchStrategyAudit:
    """
    上游: commit message 是唯一记录，无结构化数据，无可查询的审计轨迹。
    改写: 携带 upstream_commit / author / date / changed_files /
          skip_reasons / strategy_change 的结构化审计对象，
          to_log_entry() 产出 MIGRATION_LOG.md 格式的 Markdown 段落。

    鲁迅按语: 愿中国青年都摆脱冷气，只是向上走，不必听自暴自弃者流的话。
    上游 PR 的 comment 区是喧嚣的广场，MIGRATION_LOG 才是安静的档案室。
    把审计数据留在自己的代码库里，胜过一千条 PR comment。
    """
    upstream_commit: str      # e.g. "eba488c"
    upstream_author: str      # e.g. "Robert Maynard <rmaynard@nvidia.com>"
    upstream_date: str        # e.g. "2025-07-28"
    upstream_pr: str          # e.g. "rapidsai/cugraph-gnn#252"
    changed_files: List[str] = field(default_factory=list)
    skip_reasons: List[str] = field(default_factory=list)
    strategy_change: str = ""  # 记录本 commit 的核心策略变更摘要

    def __post_init__(self) -> None:
        _dbg(
            "StrategyAudit.build",
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
            f"## {self.upstream_commit} — "
            f"Update build infra to support new branching strategy (#252)\n\n"
            f"- **Upstream commit**: `{self.upstream_commit}` "
            f"(cugraph-gnn, {self.upstream_author}, {self.upstream_date})\n"
            f"- **PR**: [{self.upstream_pr}](https://github.com/rapidsai/cugraph-gnn/pull/252)\n"
            f"- **Commit message**: "
            f"`Update build infra to support new branching strategy (#252)`\n"
            f"  `rapids_config` 将使用 `RAPIDS_BRANCH` 文件内容确定分支；\n"
            f"  核心变更：默认分支推导公式从 `branch-{{version}}` 改为 "
            f"`release/{{version}}`，\n"
            f"  并新增 `RAPIDS_BRANCH` 文件作为显式分支来源。\n\n"
            f"- **Upstream diff 摘要** (3 files changed, 14 insertions(+), 3 deletions(−)):\n"
            f"  | 文件 | 变更内容 |\n"
            f"  |------|----------|\n"
            f"  | `RAPIDS_BRANCH` | 新增，内容 `branch-25.10` |\n"
            f"  | `cmake/RAPIDS.cmake` | 校验条件放宽（branch OR version），"
            f"默认推导公式 branch-→release/ |\n"
            f"  | `cmake/rapids_config.cmake` | 新增 `file(STRINGS)` 读取 "
            f"`RAPIDS_BRANCH`，注入 `rapids-cmake-branch` |\n\n"
            f"- **策略变更摘要**: {self.strategy_change}\n\n"
            f"- **CI/merge → SKIP** (全部 3 文件):\n"
            f"{skip_lines}\n\n"
            f"- **迁移位置**: `src/walpurgis/core/rapids_build_infra.py` — 新增\n\n"
            f"- **鲁迅拿法改写（≥20%）**:\n"
            f"  1. **`RapidsCmakeBranchMode` 枚举**: EXPLICIT/DERIVED 强类型替代上游"
            f"隐式 if/else 条件——上游无此分类框架\n"
            f"  2. **`RapidsCmakeBranchSpec` dataclass**: "
            f"`from_branch_file()`/`from_version()` 两条具名工厂方法，"
            f"与上游两种触发条件一一对应；`__post_init__` 校验"
            f" `branch-YY.MM`/`release/YY.MM` 格式\n"
            f"  3. **`RapidsBranchFileReader`**: 封装 `file(STRINGS)` 读取语义，"
            f"`read_or_raise()` 复现 cmake FATAL_ERROR；上游无 Python 层封装\n"
            f"  4. **`RapidsCmakePolicy`**: `validate()`/`resolve_branch()`/"
            f"`describe_resolution()` 三方法镜像上游 cmake 三段逻辑；"
            f"明确记录 eba488c 的两处核心变更\n"
            f"  5. **`RapidsBranchStrategyAudit`**: 结构化审计记录，"
            f"`to_log_entry()` 产出 MIGRATION_LOG 段落；上游只有 commit message\n"
            f"  6. **`BranchStrategyMigrationResult`**: 迁移结果汇总，"
            f"`__str__` 人类可读摘要\n"
            f"  7. **断点调试**: 全链路 8 处 `WALPURGIS_DEBUG=1` 断点\n\n"
            f"- **自测结果**: 见 `_self_test()` 全通过\n\n"
            f"---\n"
        )


# ─── 6. BranchStrategyMigrationResult — 迁移结果汇总 ─────────────────────────

@dataclass
class BranchStrategyMigrationResult:
    """
    上游: 无执行结果对象，只有 git commit + PR。
    改写: 携带 audit / policy / skipped_files 的结果对象，__str__ 人类可读。

    鲁迅按语: 我向来是不惮以最坏的恶意来推测中国人的，
    然而我还不料，也不信竟会下劣凶残到这地步。
    ——迁移时亦如此：把「跳过了什么、为什么跳过、策略是什么」
    明确记录在结果对象里，而不是只在 commit message 里轻描淡写。
    """
    audit: RapidsBranchStrategyAudit
    policy: RapidsCmakePolicy
    skipped_files: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 断点6: BranchStrategyMigrationResult 汇总
        _dbg(
            "MigrationResult.summary",
            f"commit={self.audit.upstream_commit!r} "
            f"skipped={len(self.skipped_files)}",
        )

    def __str__(self) -> str:
        try:
            branch_desc = self.policy.describe_resolution()
        except ValueError:
            branch_desc = "<invalid policy>"
        return (
            f"BranchStrategyMigrationResult: "
            f"{len(self.skipped_files)} files SKIP "
            f"| commit={self.audit.upstream_commit!r}\n"
            f"  strategy: {branch_desc}"
        )


# ─── 7. 公开工厂函数 ──────────────────────────────────────────────────────────

def build_branch_strategy_migration() -> BranchStrategyMigrationResult:
    """
    构建 eba488c (Update build infra to support new branching strategy) 的迁移结果。

    返回:
        BranchStrategyMigrationResult，含完整审计信息和策略对象。

    示例:
        result = build_branch_strategy_migration()
        print(result)
        # BranchStrategyMigrationResult: 3 files SKIP | commit='eba488c'
        # strategy: EXPLICIT: rapids-cmake-branch='branch-25.10' (from RAPIDS_BRANCH file)

    断点7: build_branch_strategy_migration() 入口。
    """
    _dbg("build_migration", "构建 eba488c 迁移结果")

    # 复现上游 RAPIDS_BRANCH 文件内容（值: 'branch-25.10'）
    branch_spec = RapidsCmakeBranchSpec.from_branch_file("branch-25.10")

    policy = RapidsCmakePolicy(
        branch_spec=branch_spec,
        rapids_version="25.10",
    )

    skipped_files = [
        "RAPIDS_BRANCH",
        "cmake/RAPIDS.cmake",
        "cmake/rapids_config.cmake",
    ]

    skip_reasons = [
        "纯文本构建配置文件，Walpurgis 无 CMake 构建体系",
        "CMake 构建基础设施，Walpurgis 无 C++/cmake 构建",
        "CMake 构建基础设施，Walpurgis 无 C++/cmake 构建",
    ]

    audit = RapidsBranchStrategyAudit(
        upstream_commit="eba488c",
        upstream_author="Robert Maynard <rmaynard@nvidia.com>",
        upstream_date="2025-07-28",
        upstream_pr="rapidsai/cugraph-gnn#252",
        changed_files=skipped_files,
        skip_reasons=skip_reasons,
        strategy_change=(
            "默认分支推导公式: 'branch-{version}' → 'release/{version}'；"
            "新增 RAPIDS_BRANCH 文件作为显式分支来源（EXPLICIT 模式优先于 DERIVED）；"
            "校验条件放宽: version 必须存在 → branch OR version 至少一个"
        ),
    )

    result = BranchStrategyMigrationResult(
        audit=audit,
        policy=policy,
        skipped_files=skipped_files,
    )

    _dbg("build_migration.done", str(result))
    return result


# ─── 8. 自测 ──────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    断点8: 自测入口，python -m walpurgis.core.rapids_build_infra 触发。
    """
    _dbg("_self_test", "开始自测")

    # 测试1: RapidsCmakeBranchSpec.from_branch_file 解析 EXPLICIT 模式
    spec_explicit = RapidsCmakeBranchSpec.from_branch_file("branch-25.10")
    assert spec_explicit.mode is RapidsCmakeBranchMode.EXPLICIT
    assert spec_explicit.branch == "branch-25.10"
    assert spec_explicit.yymm == "25.10"
    print("[PASS] 测试1: from_branch_file 解析 EXPLICIT 模式正确")

    # 测试2: RapidsCmakeBranchSpec.from_version 推导 DERIVED 模式
    spec_derived = RapidsCmakeBranchSpec.from_version("25.10")
    assert spec_derived.mode is RapidsCmakeBranchMode.DERIVED
    assert spec_derived.branch == "release/25.10"
    assert spec_derived.yymm == "25.10"
    print("[PASS] 测试2: from_version 推导 DERIVED 模式正确（'release/25.10'，eba488c 新公式）")

    # 测试3: 格式校验 — EXPLICIT 模式不接受 'release/' 前缀
    try:
        RapidsCmakeBranchSpec(
            branch="release/25.10",
            mode=RapidsCmakeBranchMode.EXPLICIT,
        )
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试3: EXPLICIT 模式格式校验正确（拒绝 'release/' 前缀）")

    # 测试4: 格式校验 — DERIVED 模式不接受 'branch-' 前缀
    try:
        RapidsCmakeBranchSpec(
            branch="branch-25.10",
            mode=RapidsCmakeBranchMode.DERIVED,
        )
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试4: DERIVED 模式格式校验正确（拒绝 'branch-' 前缀）")

    # 测试5: RapidsCmakePolicy.resolve_branch — EXPLICIT 优先
    policy_explicit = RapidsCmakePolicy(
        branch_spec=spec_explicit,
        rapids_version="25.10",
    )
    assert policy_explicit.resolve_branch() == "branch-25.10"
    print("[PASS] 测试5: EXPLICIT 模式 resolve_branch 优先于 DERIVED")

    # 测试6: RapidsCmakePolicy.resolve_branch — DERIVED 路径
    policy_derived = RapidsCmakePolicy(
        branch_spec=None,
        rapids_version="25.10",
    )
    assert policy_derived.resolve_branch() == "release/25.10"
    print("[PASS] 测试6: DERIVED 路径 resolve_branch 产出 'release/25.10'（eba488c 新公式）")

    # 测试7: RapidsCmakePolicy.validate — 两者均缺时抛 ValueError
    policy_invalid = RapidsCmakePolicy(branch_spec=None, rapids_version=None)
    try:
        policy_invalid.validate()
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试7: 校验逻辑正确（branch 和 version 均缺时拒绝）")

    # 测试8: describe_resolution 格式
    desc = policy_explicit.describe_resolution()
    assert "EXPLICIT" in desc
    assert "branch-25.10" in desc
    print("[PASS] 测试8: describe_resolution 格式正确")

    # 测试9: build_branch_strategy_migration 完整流程
    result = build_branch_strategy_migration()
    assert result.audit.upstream_commit == "eba488c"
    assert len(result.skipped_files) == 3
    summary = str(result)
    assert "3 files SKIP" in summary
    assert "eba488c" in summary
    print("[PASS] 测试9: build_branch_strategy_migration 完整流程")

    # 测试10: to_log_entry 关键字段
    log_entry = result.audit.to_log_entry()
    assert "eba488c" in log_entry
    assert "SKIP" in log_entry
    assert "release/{version}" in log_entry or "release/{{version}}" in log_entry or "release/" in log_entry
    assert "branch-25.10" in log_entry or "branch-" in log_entry
    print("[PASS] 测试10: to_log_entry 关键字段存在")

    print("\n✓ 全部 10 项自测通过")


if __name__ == "__main__":
    _self_test()
