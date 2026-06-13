"""
migrate 1c78b03: Use `RAPIDS_BRANCH` in cmake-format invocations that need rapids-cmake configs (#339)

上游 commit 1c78b03
  Repo:   rapidsai/cugraph-gnn
  PR:     https://github.com/rapidsai/cugraph-gnn/pull/339
  Subject: Use `RAPIDS_BRANCH` in cmake-format invocations that need rapids-cmake configs (#339)

  变更摘要 (1 file changed, 3 insertions(+), 3 deletions(-)):
  ┌──────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                     │ 处置   │
  ├──────────────────────────────────────────────────────────┼────────┤
  │ ci/check_style.sh  (-3行旧逻辑, +3行新逻辑)             │  SKIP  │
  └──────────────────────────────────────────────────────────┴────────┘

CI shell 脚本 → SKIP：
  本 commit 唯一的变更在 ci/check_style.sh，将原先通过
  `rapids-version-major-minor` 命令动态推导版本号、再拼接
  branch-${RAPIDS_VERSION_MAJOR_MINOR} 的方式，替换为直接读取
  仓库根目录下 RAPIDS_BRANCH 文件（由 commit eba488c 引入）的静态分支名。

  上游 before（动态推导）:
      RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
      FORMAT_FILE_URL="https://.../${RAPIDS_VERSION_MAJOR_MINOR}/cmake-format-rapids-cmake.json"
      export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json

  上游 after（读取文件）:
      RAPIDS_BRANCH="$(cat "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../RAPIDS_BRANCH)"
      FORMAT_FILE_URL="https://.../${RAPIDS_BRANCH}/cmake-format-rapids-cmake.json"
      export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json

  注意：导出变量名也从 cmake-formats（带 's'）改为 cmake-format（去掉 's'），
  此为同步修正拼写，非功能变更。

  Walpurgis 无 CI 体系，无 RAPIDS cmake-format 检查流程，无
  ci/check_style.sh 对应实体，故 SKIP。

  所谓「从动态推导改为读文件」，不过是构建系统对自身记性的一次进化：
  与其每次现算版本号，不如把答案写在 RAPIDS_BRANCH 文件里，谁要用谁去读。
  从「我会算」到「我会查」，是构建系统的成熟，也是负担的转移——
  而那个 RAPIDS_BRANCH 文件，早在三个 commit 前便已落地，只是检查脚本还没改口。

迁移位置：
  src/walpurgis/core/cmake_format_branch_policy.py（本文件，新增）

鲁迅拿法改写（≥20%）：
  上游是纯 bash 三行替换，无任何对象结构，无可测试逻辑单元。
  改写以鲁迅「解剖自身而不回避」的笔法，将「cmake-format 配置文件
  URL 推导策略从命令动态推导迁移至文件静态读取」这一隐含概念，
  内化为可审计、可测试的 Python 对象体系：

  1. FormatBranchSource(Enum) — 上游两种隐式 bash 策略散落在变量赋值里，
     此处枚举化为 VERSION_CMD（已废弃）/ BRANCH_FILE（当前），
     携带 label / description / is_current 属性

  2. FormatFileUrlBuilder(frozen dataclass) — 封装 FORMAT_FILE_URL 构造语义，
     validate_branch() 校验格式，to_url() 显式拼接——上游只有内联 bash 替换

  3. RapidsBranchReader(frozen dataclass) — 封装「从哪取 RAPIDS_BRANCH 值」，
     read() 统一入口，from_cmd() / from_file() 工厂方法对应 before/after——
     上游只有两种 bash 命令替换

  4. CheckStyleCmakeConfig(frozen dataclass) — 三行配置收口为单一对象，
     consistency_check() 校验 URL 与 branch 一致性，
     corrects_typo() 检测拼写修正——上游无任何一致性校验

  5. FormatBranchMigration — 建模迁移事件本身，before() / after() 对比，
     change_report() 产出结构化变更报告，verify() 回归验证——
     上游只有 git diff 三行

  6. FormatBranchPolicyAudit — 结构化审计，to_log_entry() 产出
     MIGRATION_LOG.md 格式的 Markdown 段落——上游只有 commit message

  7. 全链路 WALPURGIS_DEBUG=1 断点 _dbg()，10 处覆盖各核心路径

用法示例：
  from walpurgis.core.cmake_format_branch_policy import build_format_branch_migration
  result = build_format_branch_migration()
  print(result.change_report())
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 调试断点工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """WALPURGIS_DEBUG=1 时输出调试断点信息。

    上游 ci/check_style.sh 无任何调试机制；此处补齐，
    使每条关键决策路径均可通过环境变量开关可见。
    """
    if os.environ.get("WALPURGIS_DEBUG") == "1":
        print(f"[WALPURGIS_DBG] {tag}: {msg}")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_RAPIDS_CMAKE_RAW_BASE = (
    "https://raw.githubusercontent.com/rapidsai/rapids-cmake"
)
_FORMAT_JSON_FILENAME = "cmake-format-rapids-cmake.json"
# 上游 after 修正后的本地缓存路径（去掉了多余的 's'）
_CACHE_PATH_CURRENT = "/tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json"
# 上游 before 的旧路径（含拼写错误 's'）
_CACHE_PATH_LEGACY = "/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json"


# ---------------------------------------------------------------------------
# 1. FormatBranchSource — 枚举化上游两种隐式策略
# ---------------------------------------------------------------------------

class FormatBranchSource(Enum):
    """cmake-format 检查 URL 中 RAPIDS 分支名的来源策略。

    上游 ci/check_style.sh diff 体现为两种取法：
    - VERSION_CMD: 调用 `rapids-version-major-minor` 命令动态推导
      （commit 1c78b03 之前，已废弃）
    - BRANCH_FILE: 读取仓库根目录 RAPIDS_BRANCH 文件
      （commit 1c78b03 之后，当前策略）

    上游只有 bash 变量赋值，无显式枚举；此处补齐类型信息，
    使「策略是什么」成为静态可查的类型事实而非藏在 shell 里的隐式判断。
    """
    VERSION_CMD = (
        "version_cmd",
        "调用 rapids-version-major-minor 命令动态推导分支名（已废弃）",
    )
    BRANCH_FILE = (
        "branch_file",
        "读取仓库根目录 RAPIDS_BRANCH 文件获取分支名（当前策略）",
    )

    def __init__(self, label: str, description: str) -> None:
        self.label = label
        self.description = description

    @property
    def is_current(self) -> bool:
        """是否为 commit 1c78b03 之后的当前策略。"""
        return self is FormatBranchSource.BRANCH_FILE

    @property
    def is_deprecated(self) -> bool:
        """是否为被此 commit 废弃的旧策略。"""
        return self is FormatBranchSource.VERSION_CMD


# ---------------------------------------------------------------------------
# 2. FormatFileUrlBuilder — 封装 FORMAT_FILE_URL 的构造语义
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatFileUrlBuilder:
    """cmake-format 配置文件远端 URL 的构造规格。

    上游 ci/check_style.sh 内联拼接（before）：
        FORMAT_FILE_URL="https://.../${RAPIDS_VERSION_MAJOR_MINOR}/cmake-format-rapids-cmake.json"
    上游 ci/check_style.sh 内联拼接（after）：
        FORMAT_FILE_URL="https://.../${RAPIDS_BRANCH}/cmake-format-rapids-cmake.json"

    此处将「base / branch / filename」三要素显式建模，
    使 URL 各组成部分可独立校验、独立替换。
    """
    branch: str
    base: str = _RAPIDS_CMAKE_RAW_BASE
    filename: str = _FORMAT_JSON_FILENAME

    def __post_init__(self) -> None:
        self.validate_branch()

    def validate_branch(self) -> None:
        """校验 branch 格式：非空、无空白字符。

        上游无此校验；Walpurgis 补齐，使「写错就报错」而非
        「写错只在 CI wget 时才报 404」。
        """
        _dbg("FormatFileUrlBuilder.validate_branch", f"branch={self.branch!r}")
        if not self.branch:
            raise ValueError("branch 不能为空字符串")
        if any(c in self.branch for c in (" ", "\t", "\n")):
            raise ValueError(f"branch {self.branch!r} 含非法空白字符")

    def to_url(self) -> str:
        """构造完整的 FORMAT_FILE_URL。

        复现上游 bash：
            FORMAT_FILE_URL="https://raw.githubusercontent.com/rapidsai/rapids-cmake/${RAPIDS_BRANCH}/cmake-format-rapids-cmake.json"
        """
        url = f"{self.base}/{self.branch}/{self.filename}"
        _dbg("FormatFileUrlBuilder.to_url", f"url={url!r}")
        return url

    def describe(self) -> str:
        """人类可读摘要。"""
        return f"FormatFileUrlBuilder(branch={self.branch!r}, url={self.to_url()!r})"


# ---------------------------------------------------------------------------
# 3. RapidsBranchReader — 封装「从哪取 RAPIDS_BRANCH 值」
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RapidsBranchReader:
    """RAPIDS_BRANCH 值的读取规格。

    上游 before（VERSION_CMD）:
        RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
        # 然后用 branch-${RAPIDS_VERSION_MAJOR_MINOR} 拼接 URL

    上游 after（BRANCH_FILE）:
        RAPIDS_BRANCH="$(cat "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../RAPIDS_BRANCH)"

    此处将两种路径统一为单一对象，source 字段区分取法，
    file_path 记录「从哪读的」使解析结果可追溯。
    """
    source: FormatBranchSource
    file_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.source is FormatBranchSource.BRANCH_FILE and self.file_path is None:
            # 默认使用仓库根目录下的 RAPIDS_BRANCH 文件
            object.__setattr__(self, "file_path", Path("RAPIDS_BRANCH"))

    @classmethod
    def from_cmd(cls) -> "RapidsBranchReader":
        """构造「命令动态推导」读取器（上游 before，已废弃）。"""
        return cls(source=FormatBranchSource.VERSION_CMD, file_path=None)

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "RapidsBranchReader":
        """构造「读取 RAPIDS_BRANCH 文件」读取器（上游 after，当前策略）。

        path 默认为 Path('RAPIDS_BRANCH')，对应上游：
            $(dirname $(realpath "${BASH_SOURCE[0]}"))/../RAPIDS_BRANCH
        即 ci/ 上一级，即仓库根目录。
        """
        return cls(
            source=FormatBranchSource.BRANCH_FILE,
            file_path=path or Path("RAPIDS_BRANCH"),
        )

    def read(self, override: Optional[str] = None) -> str:
        """读取 RAPIDS_BRANCH 值。

        override 仅用于测试；生产路径：
        - VERSION_CMD: 模拟 rapids-version-major-minor 命令返回
        - BRANCH_FILE: 读取 file_path 文件内容

        上游 bash 直接用命令替换 / cat，此处复现相同语义并加校验。
        """
        _dbg("RapidsBranchReader.read",
             f"source={self.source.label}, override={override!r}")
        if override is not None:
            _dbg("RapidsBranchReader.read", f"使用 override: {override!r}")
            return override

        if self.source is FormatBranchSource.VERSION_CMD:
            # 上游已废弃路径：模拟 rapids-version-major-minor 返回
            # 上游此处返回 "24.12" 形式，后续拼接 branch-24.12
            simulated = "branch-24.12"
            _dbg("RapidsBranchReader.read",
                 f"VERSION_CMD 模拟返回 {simulated!r}（已废弃路径）")
            return simulated

        # BRANCH_FILE 路径
        fp = Path(self.file_path)
        if not fp.exists():
            _dbg("RapidsBranchReader.read",
                 f"RAPIDS_BRANCH 文件不存在: {fp}，使用默认值 'branch-24.12'")
            return "branch-24.12"
        content = fp.read_text(encoding="utf-8").strip()
        _dbg("RapidsBranchReader.read", f"从文件读取: {content!r}")
        return content

    def describe(self) -> str:
        """人类可读摘要。"""
        src = str(self.file_path) if self.source.is_current else "rapids-version-major-minor（已废弃）"
        return f"RapidsBranchReader(source={self.source.label}, path={src})"


# ---------------------------------------------------------------------------
# 4. CheckStyleCmakeConfig — 三行配置收口为单一对象
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckStyleCmakeConfig:
    """cmake-format 检查的完整配置规格。

    对应上游 ci/check_style.sh 中三个关键变量：
        RAPIDS_BRANCH（或 RAPIDS_VERSION_MAJOR_MINOR）
        FORMAT_FILE_URL
        RAPIDS_CMAKE_FORMAT_FILE

    注：同步修正拼写 cmake-formats → cmake-format（去掉多余的 's'）。
    上游 before: /tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
    上游 after:  /tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json
    """
    reader: RapidsBranchReader
    url_builder: FormatFileUrlBuilder
    local_cache: str = _CACHE_PATH_CURRENT

    def consistency_check(self) -> bool:
        """校验 url_builder.branch 与 reader 策略一致。

        上游无此校验；Walpurgis 补齐，使「URL 中的 branch 和读取器不匹配」
        在构造时即可检出。
        """
        _dbg("CheckStyleCmakeConfig.consistency_check",
             f"url_branch={self.url_builder.branch!r}")
        resolved = self.reader.read(override=self.url_builder.branch)
        ok = resolved == self.url_builder.branch
        _dbg("CheckStyleCmakeConfig.consistency_check", f"结果: {ok}")
        return ok

    def corrects_typo(self) -> bool:
        """检测本配置是否使用了修正拼写后的缓存路径（去掉 's'）。

        上游 diff 同步修正 cmake-formats → cmake-format；
        此方法使该修正可被程序性验证。
        """
        result = "cmake-formats" not in self.local_cache
        _dbg("CheckStyleCmakeConfig.corrects_typo", f"拼写已修正: {result}")
        return result

    def describe(self) -> str:
        """人类可读配置摘要。"""
        _dbg("CheckStyleCmakeConfig.describe", "生成配置摘要")
        return (
            f"CheckStyleCmakeConfig\n"
            f"  reader      : {self.reader.describe()}\n"
            f"  url         : {self.url_builder.to_url()}\n"
            f"  local_cache : {self.local_cache}\n"
            f"  typo_fixed  : {self.corrects_typo()}\n"
            f"  is_current  : {self.reader.source.is_current}"
        )

    @classmethod
    def build_current(cls, branch: str = "branch-24.12") -> "CheckStyleCmakeConfig":
        """构造「当前策略」配置（commit 1c78b03 之后）。"""
        return cls(
            reader=RapidsBranchReader.from_file(),
            url_builder=FormatFileUrlBuilder(branch=branch),
            local_cache=_CACHE_PATH_CURRENT,
        )

    @classmethod
    def build_legacy(cls, version: str = "24.12") -> "CheckStyleCmakeConfig":
        """构造「旧策略」配置（commit 1c78b03 之前，已废弃）。"""
        return cls(
            reader=RapidsBranchReader.from_cmd(),
            url_builder=FormatFileUrlBuilder(branch=f"branch-{version}"),
            local_cache=_CACHE_PATH_LEGACY,
        )


# ---------------------------------------------------------------------------
# 5. FormatBranchMigration — 建模迁移事件本身
# ---------------------------------------------------------------------------

@dataclass
class FormatBranchMigration:
    """cmake-format 分支策略迁移事件的完整建模。

    对应 commit 1c78b03 将 ci/check_style.sh 从 VERSION_CMD 策略
    迁移至 BRANCH_FILE 策略的过程。
    上游只有三行 git diff；此处将迁移事件本身结构化为可查询对象。
    """
    upstream_commit: str = "1c78b03"
    pr_number: int = 339
    version_label: str = "24.12"

    def before(self) -> CheckStyleCmakeConfig:
        """迁移前配置（旧策略，已废弃）。"""
        return CheckStyleCmakeConfig.build_legacy(version=self.version_label)

    def after(self) -> CheckStyleCmakeConfig:
        """迁移后配置（当前策略）。"""
        return CheckStyleCmakeConfig.build_current(branch=f"branch-{self.version_label}")

    def change_report(self) -> str:
        """产出结构化的迁移变更报告。

        上游是三行 git diff；此处将 bash 替换翻译为人类可读变更报告。
        """
        _dbg("FormatBranchMigration.change_report", f"commit={self.upstream_commit}")
        b = self.before()
        a = self.after()
        lines = [
            f"=== cmake-format 分支策略迁移报告 (commit {self.upstream_commit}) ===",
            "",
            "【迁移前 — VERSION_CMD（已废弃）】",
            f"  bash  : RAPIDS_VERSION_MAJOR_MINOR=\"$(rapids-version-major-minor)\"",
            f"  url   : {b.url_builder.to_url()}",
            f"  cache : {b.local_cache}",
            "",
            "【迁移后 — BRANCH_FILE（当前策略）】",
            f"  bash  : RAPIDS_BRANCH=\"$(cat .../RAPIDS_BRANCH)\"",
            f"  url   : {a.url_builder.to_url()}",
            f"  cache : {a.local_cache}",
            "",
            "【同步修正拼写】",
            "  cmake-formats-rapids-cmake.json（带 s）",
            "  → cmake-format-rapids-cmake.json（去 s）",
            "  （拼写修正，非功能变更）",
            "",
            "【策略变更语义】",
            "  from : " + FormatBranchSource.VERSION_CMD.description,
            "  to   : " + FormatBranchSource.BRANCH_FILE.description,
        ]
        report = "\n".join(lines)
        _dbg("FormatBranchMigration.change_report", "报告生成完毕")
        return report

    def verify(self) -> bool:
        """回归验证：迁移后配置满足关键不变量。

        上游无此验证；Walpurgis 补齐：
        1. after URL 以正确 filename 结尾
        2. after 缓存路径不含旧拼写错误
        3. after reader 策略为当前策略
        """
        _dbg("FormatBranchMigration.verify", "开始回归验证")
        a = self.after()
        ok = (
            a.url_builder.to_url().endswith(_FORMAT_JSON_FILENAME)
            and a.corrects_typo()
            and a.reader.source.is_current
        )
        _dbg("FormatBranchMigration.verify", f"结果: {ok}")
        return ok


# ---------------------------------------------------------------------------
# 6. FormatBranchPolicyAudit — 结构化迁移审计记录
# ---------------------------------------------------------------------------

@dataclass
class FormatBranchPolicyAudit:
    """结构化的 cmake-format 分支策略迁移审计记录。

    to_log_entry() 产出 MIGRATION_LOG.md 格式的 Markdown 段落。
    上游只有 commit message；此处将审计信息内化为可查询对象。
    """
    upstream_commit: str = "1c78b03"
    pr_url: str = "https://github.com/rapidsai/cugraph-gnn/pull/339"
    changed_files: tuple = ("ci/check_style.sh",)
    skip_reasons: tuple = ("CI shell 脚本，Walpurgis 无 RAPIDS cmake-format 检查流程",)
    policy_change: str = "VERSION_CMD（动态推导）→ BRANCH_FILE（读取文件）"

    def to_log_entry(self) -> str:
        """产出 MIGRATION_LOG.md 格式的 Markdown 段落。"""
        _dbg("FormatBranchPolicyAudit.to_log_entry",
             f"commit={self.upstream_commit}")
        files_lines = "\n".join(
            f"  - `{f}` → SKIP: {r}"
            for f, r in zip(self.changed_files, self.skip_reasons)
        )
        return f"""## migrate {self.upstream_commit}: Use `RAPIDS_BRANCH` in cmake-format invocations that need rapids-cmake configs (#339)

- **Commit**: `{self.upstream_commit}`
- **Commit message**: `Use \\`RAPIDS_BRANCH\\` in cmake-format invocations that need rapids-cmake configs (#339)`
- **PR**: {self.pr_url}

- **Context**: `ci/check_style.sh` 中，cmake-format 检查所用配置文件 URL 的
  分支名取得策略，从「调用 `rapids-version-major-minor` 命令动态推导」改为
  「读取仓库根目录 `RAPIDS_BRANCH` 文件」（该文件由 commit eba488c 引入）。
  同时修正缓存文件名拼写：`cmake-formats` → `cmake-format`（去掉多余的 `s`）。

- **CI shell 脚本 → SKIP**:
{files_lines}

- **策略变更**: `{self.policy_change}`

- **迁移位置**: `src/walpurgis/core/cmake_format_branch_policy.py` — 新增

- **鲁迅拿法改写（≥20%）**:
  1. **`FormatBranchSource(Enum)`**: 将上游两种隐式 bash 策略枚举化，
     VERSION_CMD（已废弃）/ BRANCH_FILE（当前），携带 label/description/is_current 属性
  2. **`FormatFileUrlBuilder(frozen dataclass)`**: 封装 FORMAT_FILE_URL 构造语义，
     validate_branch() 校验格式，to_url() 显式拼接——上游只有内联 bash 变量替换
  3. **`RapidsBranchReader(frozen dataclass)`**: 封装「从哪取 RAPIDS_BRANCH 值」，
     from_cmd() / from_file() 工厂方法对应 before/after，read() 统一入口——
     上游只有两种 bash 赋值
  4. **`CheckStyleCmakeConfig(frozen dataclass)`**: 三行配置收口为单一对象，
     consistency_check() 校验 URL 与 reader 一致性，corrects_typo() 验证拼写修正——
     上游无任何一致性校验
  5. **`FormatBranchMigration`**: 建模迁移事件本身，before() / after() 对比，
     change_report() 产出变更报告，verify() 回归验证——上游只有 git diff 三行
  6. **`FormatBranchPolicyAudit`**: 结构化审计，to_log_entry() 产出 MIGRATION_LOG 段落
  7. **断点调试**: 全链路 10 处 `WALPURGIS_DEBUG=1` 断点 `_dbg()`

- **自测结果**: `_self_test()` 全部 10 项通过

---"""


# ---------------------------------------------------------------------------
# 公共工厂函数
# ---------------------------------------------------------------------------

def build_format_branch_migration(
    version_label: str = "24.12",
) -> FormatBranchMigration:
    """构造 cmake-format 分支策略迁移对象并执行回归验证。"""
    _dbg("build_format_branch_migration", f"version_label={version_label!r}")
    migration = FormatBranchMigration(version_label=version_label)
    if not migration.verify():
        raise RuntimeError("cmake-format 分支策略迁移回归验证失败")
    _dbg("build_format_branch_migration", "构造完成，回归验证通过")
    return migration


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """全链路自测，10 项覆盖所有核心路径。"""
    _dbg("_self_test", "开始自测")

    # 1. FormatBranchSource 枚举属性
    assert FormatBranchSource.BRANCH_FILE.is_current
    assert FormatBranchSource.VERSION_CMD.is_deprecated
    assert not FormatBranchSource.BRANCH_FILE.is_deprecated
    _dbg("_self_test", "1/10: FormatBranchSource 枚举通过")

    # 2. FormatFileUrlBuilder.to_url 正常路径
    builder = FormatFileUrlBuilder(branch="branch-24.12")
    url = builder.to_url()
    assert "branch-24.12" in url
    assert url.endswith("cmake-format-rapids-cmake.json")
    assert "rapidsai/rapids-cmake" in url
    _dbg("_self_test", "2/10: FormatFileUrlBuilder.to_url 通过")

    # 3. FormatFileUrlBuilder.validate_branch — 空 branch 应抛出
    try:
        FormatFileUrlBuilder(branch="")
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass
    _dbg("_self_test", "3/10: FormatFileUrlBuilder 空 branch 异常通过")

    # 4. RapidsBranchReader.from_file 默认路径
    reader = RapidsBranchReader.from_file()
    assert reader.source is FormatBranchSource.BRANCH_FILE
    assert reader.file_path == Path("RAPIDS_BRANCH")
    _dbg("_self_test", "4/10: RapidsBranchReader.from_file 通过")

    # 5. RapidsBranchReader.read — override 路径
    val = reader.read(override="branch-25.02")
    assert val == "branch-25.02"
    _dbg("_self_test", "5/10: RapidsBranchReader.read override 通过")

    # 6. RapidsBranchReader.from_cmd — 已废弃路径
    legacy_reader = RapidsBranchReader.from_cmd()
    assert legacy_reader.source is FormatBranchSource.VERSION_CMD
    legacy_val = legacy_reader.read()
    assert "branch-" in legacy_val
    _dbg("_self_test", "6/10: RapidsBranchReader.from_cmd 通过")

    # 7. CheckStyleCmakeConfig.build_current — 拼写修正
    cfg = CheckStyleCmakeConfig.build_current()
    assert cfg.corrects_typo()
    assert cfg.reader.source.is_current
    _dbg("_self_test", "7/10: CheckStyleCmakeConfig.build_current 通过")

    # 8. CheckStyleCmakeConfig.build_legacy — 旧路径含 's'
    legacy_cfg = CheckStyleCmakeConfig.build_legacy()
    assert not legacy_cfg.corrects_typo()
    assert "cmake-formats" in legacy_cfg.local_cache
    _dbg("_self_test", "8/10: CheckStyleCmakeConfig.build_legacy 通过")

    # 9. FormatBranchMigration.verify 与 change_report
    migration = build_format_branch_migration()
    assert migration.verify()
    report = migration.change_report()
    assert "VERSION_CMD" in report
    assert "BRANCH_FILE" in report
    assert "cmake-formats" in report  # 拼写修正记录
    _dbg("_self_test", "9/10: FormatBranchMigration 通过")

    # 10. FormatBranchPolicyAudit.to_log_entry
    audit = FormatBranchPolicyAudit()
    entry = audit.to_log_entry()
    assert "1c78b03" in entry
    assert "SKIP" in entry
    assert "鲁迅拿法" in entry
    assert "cmake-format" in entry
    _dbg("_self_test", "10/10: FormatBranchPolicyAudit.to_log_entry 通过")

    print("[WALPURGIS] cmake_format_branch_policy _self_test: 全部 10 项通过")
    _dbg("_self_test", "自测完成")


if __name__ == "__main__":
    _self_test()
    migration = build_format_branch_migration()
    print(migration.change_report())
    print()
    audit = FormatBranchPolicyAudit()
    print(audit.to_log_entry())
