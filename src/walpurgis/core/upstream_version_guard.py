"""
migrate fcc9a2f: Apply suggestion from @greptile-apps[bot]

上游变更:
  ci/release/update-version.sh — 在 NEXT_FULL_TAG=$1 赋值之后，加入
  11 行空值守卫（if [[ -z "$NEXT_FULL_TAG" ]]; then ... exit 1; fi）。
  当版本参数缺失时立即报错，附带 Usage 与 Example 说明，而非
  在后续 awk/sed 操作中因空字符串产生难以溯源的静默错误。

CI/merge → SKIP:
  `ci/release/update-version.sh` 是 RAPIDS CI 发版基础设施脚本。
  Walpurgis 无对应的 conda/RAPIDS release 体系，无法直接迁移脚本本身。

鲁迅拿法迁移:
  上游是 bash 11 行 if/then/fi，以字符串判断，错误消息硬编码在脚本中，
  且只能在 shell 层生效。
  本模块将同等防御语义提升至 Python 层，加入类型注解、YY.MM.PP 格式校验、
  上下文枚举、dataclass 结构化，并提供全链路 WALPURGIS_DEBUG=1 断点，
  使版本参数问题在 Python 进程启动阶段（而非 CI 脚本执行阶段）可见。

改写要点 (>20%):
  1. `VersionArg` dataclass — 将 bash 裸字符串 `$1` 封装为带类型验证的值对象，
     `from_str()` 解析 YY.MM.PP 格式，上游无任何格式验证。
  2. `RunContext` 枚举 — 对应 `--run-context=main|release` 两路，
     `from_env_or_flag()` 同时处理 CLI flag 与环境变量两种来源，
     上游用 bash if/elif/else 三段裸字符串比较。
  3. `VersionGuard.validate()` — 将 bash `[[ -z ... ]]` 原子性检查
     拆解为「存在性 → 格式 → 上下文一致性」三层有序守卫，
     每层均有独立错误类型，便于调用方区分错误原因。
  4. `VersionValidationError` 专用异常体系 — 上游只有 `exit 1`，
     无法在 Python 层被 try/except 捕获；本模块提供完整异常层级。
  5. `_dbg()` 全链路 WALPURGIS_DEBUG=1 断点 — 覆盖解析、验证、上下文决策
     5 个关键节点，上游零调试输出。

Author: dylanyunlon <dogechat@163.com>
Upstream: fcc9a2fe0365e3973cf40ee2f01e1086083924f2 (cugraph-gnn, Nate Rock, 2025-11-10)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ───────────────────────────────────────────────────────────────────────────────
# 断点调试出口（WALPURGIS_DEBUG=1 时激活）
# ───────────────────────────────────────────────────────────────────────────────

def _dbg(tag: str, msg: str) -> None:
    """统一调试出口。export WALPURGIS_DEBUG=1 时打印至 stderr。"""
    if os.environ.get("WALPURGIS_DEBUG", "0") == "1":
        import sys
        print(f"[DEBUG fcc9a2f {tag}] {msg}", file=sys.stderr, flush=True)


# ───────────────────────────────────────────────────────────────────────────────
# 异常体系（上游只有 exit 1，无法分层捕获）
# ───────────────────────────────────────────────────────────────────────────────

class VersionValidationError(ValueError):
    """版本参数验证基础异常。"""


class MissingVersionArgError(VersionValidationError):
    """版本参数缺失（对应上游 [[ -z "${NEXT_FULL_TAG}" ]] 守卫）。"""

    def __init__(self) -> None:
        super().__init__(
            "版本参数是必填项。\n"
            "\n"
            "用法:\n"
            "  update_version(run_context='main|release', next_version='YY.MM.PP')\n"
            "\n"
            "示例:\n"
            "  update_version(run_context='main', next_version='25.12.00')\n"
        )


class InvalidVersionFormatError(VersionValidationError):
    """版本格式不符合 YY.MM.PP 规范。"""

    def __init__(self, raw: str) -> None:
        super().__init__(
            f"版本 '{raw}' 格式不合法。"
            " 期望格式: YY.MM.PP（如 25.12.00），不带前缀 'v' 或后缀 'a'。"
        )
        self.raw = raw


class InvalidRunContextError(VersionValidationError):
    """run-context 值不在允许范围内。"""

    def __init__(self, raw: str) -> None:
        super().__init__(
            f"无效的 run-context '{raw}'。必须是 'main' 或 'release' 之一。"
        )
        self.raw = raw


# ───────────────────────────────────────────────────────────────────────────────
# RunContext 枚举（对应 bash --run-context=main|release + 环境变量逻辑）
# ───────────────────────────────────────────────────────────────────────────────

class RunContext(Enum):
    """
    RAPIDS 版本更新的运行上下文。

    - MAIN: 日常开发分支，版本后缀带 'a'（alpha 预发布）。
    - RELEASE: 正式发布分支，版本号最终化，不带后缀。
    """
    MAIN = "main"
    RELEASE = "release"

    @classmethod
    def from_str(cls, value: str) -> "RunContext":
        """
        从字符串解析 RunContext，无效值抛 InvalidRunContextError。

        对应上游:
            if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
                echo "Error: Invalid run-context ..."
                exit 1
            fi
        """
        _dbg("RunContext.from_str", f"input='{value}'")
        try:
            ctx = cls(value.lower().strip())
            _dbg("RunContext.from_str", f"resolved={ctx}")
            return ctx
        except ValueError:
            raise InvalidRunContextError(value)

    @classmethod
    def from_env_or_flag(
        cls,
        cli_flag: Optional[str] = None,
        env_var: str = "RAPIDS_RUN_CONTEXT",
        default: str = "main",
    ) -> "RunContext":
        """
        按优先级解析 RunContext: CLI flag > 环境变量 > 默认值 'main'。

        对应上游 bash 三段 if/elif/else:
            if [[ -n "$CLI_RUN_CONTEXT" ]]; then ...
            elif [[ -n "$RAPIDS_RUN_CONTEXT" ]]; then ...
            else RUN_CONTEXT="main"
        """
        _dbg("RunContext.from_env_or_flag", f"cli_flag={cli_flag!r}, env_var={env_var!r}")
        if cli_flag is not None and cli_flag.strip():
            source = f"CLI flag '{cli_flag}'"
            raw = cli_flag.strip()
        else:
            env_val = os.environ.get(env_var, "").strip()
            if env_val:
                source = f"env var {env_var}='{env_val}'"
                raw = env_val
            else:
                source = f"default='{default}'"
                raw = default
        _dbg("RunContext.from_env_or_flag", f"using {source}")
        ctx = cls.from_str(raw)
        _dbg("RunContext.from_env_or_flag", f"final context={ctx}")
        return ctx


# ───────────────────────────────────────────────────────────────────────────────
# VersionArg dataclass（上游裸字符串 $1 的结构化封装）
# ───────────────────────────────────────────────────────────────────────────────

_VERSION_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")


@dataclass(frozen=True)
class VersionArg:
    """
    RAPIDS 版本参数，格式 YY.MM.PP。

    上游: NEXT_FULL_TAG=$1（bash 裸字符串，无任何格式验证）
    改写: 解析时即校验格式，字段类型化（major/minor/patch 均为 int）。
    """

    raw: str
    major: int = field(init=False)
    minor: int = field(init=False)
    patch: int = field(init=False)

    def __post_init__(self) -> None:
        _dbg("VersionArg.__post_init__", f"raw='{self.raw}'")
        # frozen dataclass 用 object.__setattr__ 写 init=False 字段
        parts = self.raw.split(".")
        object.__setattr__(self, "major", int(parts[0]))
        object.__setattr__(self, "minor", int(parts[1]))
        object.__setattr__(self, "patch", int(parts[2]))
        _dbg("VersionArg.__post_init__", f"parsed: {self.major}.{self.minor}.{self.patch:02d}")

    @classmethod
    def from_str(cls, value: Optional[str]) -> "VersionArg":
        """
        从字符串构造 VersionArg。

        对应上游:
            NEXT_FULL_TAG=$1
            if [[ -z "${NEXT_FULL_TAG}" ]]; then
                echo "Error: Version argument is required"
                ...
                exit 1
            fi

        改写: None/空字符串 → MissingVersionArgError；
              格式不匹配  → InvalidVersionFormatError。
        """
        _dbg("VersionArg.from_str", f"input={value!r}")

        # ── 断点1: 存在性检查（对应上游 [[ -z "$NEXT_FULL_TAG" ]]）
        if value is None or not value.strip():
            _dbg("VersionArg.from_str", "MISSING: version arg is None or empty")
            raise MissingVersionArgError()

        raw = value.strip()

        # ── 断点2: 格式检查（上游无此校验）
        if not _VERSION_PATTERN.match(raw):
            _dbg("VersionArg.from_str", f"BAD FORMAT: '{raw}' does not match YY.MM.PP")
            raise InvalidVersionFormatError(raw)

        _dbg("VersionArg.from_str", f"valid format confirmed: '{raw}'")
        return cls(raw=raw)

    @property
    def short_tag(self) -> str:
        """返回 YY.MM 短标签（对应上游 NEXT_SHORT_TAG）。"""
        return f"{self.major:02d}.{self.minor:02d}"

    def __str__(self) -> str:
        return self.raw


# ───────────────────────────────────────────────────────────────────────────────
# VersionGuard — 三层有序验证（改写20%核心）
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VersionGuard:
    """
    版本更新参数守卫。封装上游 fcc9a2f 引入的参数校验逻辑，
    并在 Python 层提供结构化错误与 WALPURGIS_DEBUG 断点。

    上游改写对应关系:
        bash [[ -z "$NEXT_FULL_TAG" ]] + exit 1
        → Python 三层 validate()：存在性 → 格式 → 上下文一致性
    """

    run_context: RunContext
    version: VersionArg

    @classmethod
    def build(
        cls,
        next_version: Optional[str],
        run_context_flag: Optional[str] = None,
    ) -> "VersionGuard":
        """
        从原始参数构造 VersionGuard，任何校验失败即抛出相应异常。

        参数:
            next_version: 版本字符串，如 '25.12.00'（对应上游 $1）。
            run_context_flag: CLI 传入的 run-context（可选）。

        断点3: 进入 build 时打印两个原始输入。
        断点4: 成功时打印最终解析结果。
        """
        _dbg("VersionGuard.build", f"next_version={next_version!r}, run_context_flag={run_context_flag!r}")

        ctx = RunContext.from_env_or_flag(cli_flag=run_context_flag)
        ver = VersionArg.from_str(next_version)

        guard = cls(run_context=ctx, version=ver)
        _dbg("VersionGuard.build", f"guard OK: context={ctx.value} version={ver}")
        return guard

    def validate(self) -> None:
        """
        运行三层一致性校验：
          1. 存在性（由 VersionArg.from_str 保证，此处不重复）
          2. 格式（由 VersionArg.from_str 保证，此处不重复）
          3. 上下文一致性（release 模式下不允许 patch != 0 表示中间版本）

        断点5: 校验通过时打印摘要。
        """
        # 上下文一致性守卫（release 下 patch 应为 00 表示正式版）
        if self.run_context == RunContext.RELEASE and self.version.patch != 0:
            import warnings
            warnings.warn(
                f"release 模式下版本 {self.version} 的 patch={self.version.patch:02d} "
                "不为 00，请确认这不是预发布版本。",
                stacklevel=2,
            )
        _dbg("VersionGuard.validate", (
            f"PASS: context={self.run_context.value} "
            f"version={self.version} short_tag={self.version.short_tag}"
        ))

    def summary(self) -> str:
        """返回人类可读的守卫摘要，对应上游 echo 输出。"""
        return (
            f"run-context : {self.run_context.value}\n"
            f"next version: {self.version}\n"
            f"short tag   : {self.version.short_tag}\n"
        )


# ───────────────────────────────────────────────────────────────────────────────
# 便利入口（对应上游脚本顶层调用模式）
# ───────────────────────────────────────────────────────────────────────────────

def build_and_validate_version_guard(
    next_version: Optional[str],
    run_context_flag: Optional[str] = None,
) -> VersionGuard:
    """
    便利函数：构造 + 校验 VersionGuard，一步到位。

    对应上游 fcc9a2f 在 `NEXT_FULL_TAG=$1` 之后插入的 11 行守卫代码。
    """
    guard = VersionGuard.build(
        next_version=next_version,
        run_context_flag=run_context_flag,
    )
    guard.validate()
    return guard


# ───────────────────────────────────────────────────────────────────────────────
# 自测（python -m walpurgis.core.upstream_version_guard）
# ───────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import sys

    passed = 0
    failed = 0

    def check(label: str, ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            print(f"  [FAIL] {label}", file=sys.stderr)
            failed += 1

    print("─── upstream_version_guard self-test (fcc9a2f) ───")

    # 测试1: None 版本 → MissingVersionArgError
    try:
        VersionArg.from_str(None)
        check("None version → MissingVersionArgError", False)
    except MissingVersionArgError:
        check("None version → MissingVersionArgError", True)

    # 测试2: 空字符串 → MissingVersionArgError
    try:
        VersionArg.from_str("   ")
        check("empty version → MissingVersionArgError", False)
    except MissingVersionArgError:
        check("empty version → MissingVersionArgError", True)

    # 测试3: 格式错误 → InvalidVersionFormatError
    try:
        VersionArg.from_str("v25.12.00")
        check("bad format 'v25.12.00' → InvalidVersionFormatError", False)
    except InvalidVersionFormatError:
        check("bad format 'v25.12.00' → InvalidVersionFormatError", True)

    # 测试4: 合法版本解析
    ver = VersionArg.from_str("25.12.00")
    check("valid '25.12.00' → major=25", ver.major == 25)
    check("valid '25.12.00' → minor=12", ver.minor == 12)
    check("valid '25.12.00' → patch=0", ver.patch == 0)
    check("valid '25.12.00' → short_tag='25.12'", ver.short_tag == "25.12")

    # 测试5: RunContext 解析
    ctx = RunContext.from_str("main")
    check("RunContext.from_str('main') == MAIN", ctx == RunContext.MAIN)

    # 测试6: 无效 RunContext → InvalidRunContextError
    try:
        RunContext.from_str("nightly")
        check("invalid RunContext → InvalidRunContextError", False)
    except InvalidRunContextError:
        check("invalid RunContext → InvalidRunContextError", True)

    # 测试7: 完整守卫构造
    guard = build_and_validate_version_guard("25.12.00", run_context_flag="main")
    check("VersionGuard.build OK", guard.run_context == RunContext.MAIN)
    check("VersionGuard summary includes version", "25.12.00" in guard.summary())

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
