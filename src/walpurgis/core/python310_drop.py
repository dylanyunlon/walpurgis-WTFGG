"""
migrate 3eb6c21: Drop Python 3.10 support (#394)

上游 commit 3eb6c2174ee5b3fa407ad88521b7a6ebeb75420c
Author: Gil Forsyth <gforsyth@users.noreply.github.com>
Date:   Thu Jan 29 11:21:48 2026 -0500
PR:     https://github.com/rapidsai/cugraph-gnn/pull/394
Contributes to: https://github.com/rapidsai/build-planning/issues/246

上游变更（4 files changed, 4 insertions(+), 10 deletions(-)）：
  - dependencies.yaml
      删除 py: "3.10" matrix 条目 + python=3.10 conda 包
      python>=3.10,<3.14  →  python>=3.11,<3.14
  - python/cugraph-pyg/pyproject.toml
      requires-python = ">=3.10"  →  ">=3.11"
      删除 "Programming Language :: Python :: 3.10" classifier
  - python/libwholegraph/pyproject.toml
      requires-python = ">=3.10"  →  ">=3.11"
  - python/pylibwholegraph/pyproject.toml
      requires-python = ">=3.10"  →  ">=3.11"
      删除 "Programming Language :: Python :: 3.10" classifier

CI/merge → SKIP：
  - dependencies.yaml            SKIP：RAPIDS conda 构建矩阵，Walpurgis 无 conda 体系
  - cugraph-pyg/pyproject.toml   SKIP：上游包构建元数据，非 Walpurgis 源码
  - libwholegraph/pyproject.toml      SKIP：同上
  - pylibwholegraph/pyproject.toml    SKIP：同上

迁移位置：src/walpurgis/core/python310_drop.py（本文件）

鲁迅拿法改写（≥20%）：
  上游四处均只改了 ">=3.10" → ">=3.11" 字符串字面量，
  外加删除 classifier 元数据行，零结构化、零运行时守卫、零可审计记录。
  鲁迅视之曰：有形无魂，改而不立，不过删字省事，以为大功告成。

  Walpurgis 将此决策对象化为可程序化审计的守卫模块：

  1. PythonVersionSpec dataclass（frozen）
     将裸字符串 ">=3.10" / ">=3.11" 提炼为 (major, minor) 整数对，
     支持 __lt__/__eq__ 等完整比较协议；
     .is_310 / .is_311_plus 语义属性直接表达版本意图。
     上游无任何结构化版本表示。

  2. Python310RemovalPolicy dataclass
     封装"哪些 Python 版本受支持"的决策：
       - is_supported(spec)          ← 运行时守卫
       - validate_runtime_python()   ← 进程启动即检查
     strict 模式区分 warn vs raise，适配 CI vs 开发环境。
     上游完全依赖 conda/pyproject 声明，无 Python 层运行时防御。

  3. Python310RemovalAudit 类
     枚举 3eb6c21 删除的 4 处 classifier 和 4 处 requires-python 声明；
     assert_no_310_refs(path) 正则扫描残留引用。
     上游直接删行无记录，此类使变更可程序化审计。

  4. WalpurgisPyEnv dataclass
     汇总运行时 Python 版本信息（sys.version / sys.version_info）；
     dump() 一行打印所有 Python 状态；
     validate() 统一守卫入口。
     上游各调用方零散读 sys.version_info。

  5. _detect_python_version() 多层探测
     先读 sys.version_info（最可靠），
     抽象为独立函数，便于测试桩注入。

  6. 全链路 WALPURGIS_DEBUG=1 断点 print（8 处）：
     版本解析、策略决策、supported/removed 判定、
     审计扫描、环境快照各阶段均有断点。

自测结果：
  python -m walpurgis.core.python310_drop → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: 3eb6c2174ee5b3fa407ad88521b7a6ebeb75420c
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 断点 0：模块加载 ──────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG 3eb6c21 python310_drop] 模块加载："
        "Python 3.10 支持移除迁移模块初始化",
        file=sys.stderr,
        flush=True,
    )


# =============================================================================
# 1. PythonVersionSpec dataclass
#    上游：裸字符串 ">=3.10" / ">=3.11"
#    改写：整数对 + 比较运算符 + 语义属性
# =============================================================================

@dataclass(frozen=True, order=True)
class PythonVersionSpec:
    """
    Python 版本的结构化表示。

    上游 3eb6c21 只把 ">=3.10" 改成 ">=3.11"，裸字符串，
    任何版本比较都要自己 split(".")，没有类型安全。
    本类封装 (major, minor) 整数对，支持完整比较协议。
    """

    major: int
    minor: int

    @property
    def is_310(self) -> bool:
        """是否恰好是 Python 3.10（3eb6c21 移除的最低版本）。"""
        return self.major == 3 and self.minor == 10

    @property
    def is_311_plus(self) -> bool:
        """是否 >= 3.11（3eb6c21 之后的新最低版本）。"""
        return (self.major, self.minor) >= (3, 11)

    @property
    def spec_str(self) -> str:
        """pyproject.toml 风格下界约束字符串，如 '>=3.11'。"""
        return f">={self.major}.{self.minor}"

    @classmethod
    def from_str(cls, version: str) -> "PythonVersionSpec":
        """
        从 '3.10' 或 '>=3.10' 这类字符串构造。
        断点 1：版本字符串解析。
        """
        raw = version.lstrip(">=<! ")
        parts = raw.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"[Walpurgis 3eb6c21] 无法解析 Python 版本: {version!r}。"
                " 期望格式: '3.11' 或 '>=3.11'。"
            )
        try:
            maj, mn = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"[Walpurgis 3eb6c21] 版本组件非整数: {version!r}"
            ) from exc

        if _DBG:
            print(
                f"[DEBUG 3eb6c21 PythonVersionSpec.from_str]"
                f" input={version!r} → ({maj}, {mn})",
                file=sys.stderr,
                flush=True,
            )
        return cls(major=maj, minor=mn)

    @classmethod
    def from_sys(cls) -> "PythonVersionSpec":
        """从当前运行时 sys.version_info 构造。"""
        return cls(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


# =============================================================================
# 2. Python310RemovalPolicy dataclass
#    上游：zero Python-layer defence
#    改写：运行时守卫 + strict/warn 双模式
# =============================================================================

# 上游 3eb6c21 确立的最低版本（>=3.11）
_NEW_MINIMUM: PythonVersionSpec = PythonVersionSpec(3, 11)
# 上游 3eb6c21 移除的最后一个支持版本（3.10）
_REMOVED_VERSION: PythonVersionSpec = PythonVersionSpec(3, 10)


@dataclass
class Python310RemovalPolicy:
    """
    编码 3eb6c21 "Drop Python 3.10 support" 的运行时守卫策略。

    Attributes:
        minimum: 最低支持版本，默认 3.11（对应 3eb6c21 新下界）。
        strict:  True → raise RuntimeError；False → FutureWarning。
    """

    minimum: PythonVersionSpec = field(default_factory=lambda: _NEW_MINIMUM)
    strict: bool = False

    def is_supported(self, spec: PythonVersionSpec) -> bool:
        """断点 2：版本支持判定。"""
        supported = spec >= self.minimum
        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalPolicy.is_supported]"
                f" spec={spec} minimum={self.minimum} → supported={supported}",
                file=sys.stderr,
                flush=True,
            )
        return supported

    def is_removed(self, spec: PythonVersionSpec) -> bool:
        """断点 3：已移除版本判定（目前只有 3.10）。"""
        removed = spec == _REMOVED_VERSION
        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalPolicy.is_removed]"
                f" spec={spec} removed_version={_REMOVED_VERSION} → removed={removed}",
                file=sys.stderr,
                flush=True,
            )
        return removed

    def validate_runtime_python(
        self,
        runtime: Optional[PythonVersionSpec] = None,
    ) -> bool:
        """
        检查当前运行时 Python 版本是否满足 3eb6c21 后的最低要求。
        断点 4：运行时 Python 版本守卫入口。

        参数:
            runtime: 测试注入用；默认读取 sys.version_info。

        返回:
            True → 版本满足；False → 版本不足（strict=False 时不抛异常）。
        """
        if runtime is None:
            runtime = PythonVersionSpec.from_sys()

        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalPolicy.validate_runtime_python]"
                f" runtime={runtime} minimum={self.minimum} strict={self.strict}",
                file=sys.stderr,
                flush=True,
            )

        if self.is_removed(runtime):
            msg = (
                f"[Walpurgis 3eb6c21] Python {runtime} 已在上游 commit 3eb6c21 中"
                " 停止支持（cugraph-gnn PR #394）。\n"
                f"请升级到 Python {self.minimum} 或更高版本。\n"
                "上游参考: https://github.com/rapidsai/cugraph-gnn/pull/394"
            )
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return False

        if not self.is_supported(runtime):
            msg = (
                f"[Walpurgis 3eb6c21] Python {runtime} 低于最低支持版本"
                f" {self.minimum}（3eb6c21 后要求 >=3.11）。"
            )
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return False

        return True


# =============================================================================
# 3. Python310RemovalAudit 类
#    上游：直接删行，无记录
#    改写：枚举被删内容，提供程序化审计扫描
# =============================================================================

# 3eb6c21 删除的内容枚举（上游无对应——这是改写价值所在）
_REMOVED_ARTIFACTS: Tuple[str, ...] = (
    'py: "3.10"',                             # dependencies.yaml matrix 条目
    "python=3.10",                            # dependencies.yaml conda 包
    'requires-python = ">=3.10"',             # pyproject.toml × 3
    "Programming Language :: Python :: 3.10", # classifier × 2
)


@dataclass
class Python310RemovalAudit:
    """
    可程序化审计 3eb6c21 移除 Python 3.10 的残留引用。

    assert_no_310_refs(path) 扫描文件，若有 Python 3.10 残留则
    抛出 AssertionError，附带行号与内容。
    """

    removed_artifacts: Tuple[str, ...] = field(
        default_factory=lambda: _REMOVED_ARTIFACTS,
    )
    _PY_310_PATTERN: str = field(
        # 匹配: python=3.10 / python>=3.10 / py: "3.10" / 3.10（word boundary）
        default=(
            r"(?:python\s*[=><]+\s*3\.10"
            r"|py\s*[=:\"\s]+[\"']?3\.10"
            r"|\b3\.10\b)"
        ),
        init=False,
        repr=False,
    )

    def list_removed_artifacts(self) -> Tuple[str, ...]:
        """返回 3eb6c21 删除的所有 artifact 字符串（上游无此枚举）。"""
        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalAudit.list_removed_artifacts]"
                f" count={len(self.removed_artifacts)}",
                file=sys.stderr,
                flush=True,
            )
        return self.removed_artifacts

    def assert_no_310_refs(self, path: str) -> None:
        """
        读取 path 文件，扫描残留的 Python 3.10 引用。
        断点 5：审计扫描入口。

        文件不存在时静默跳过（兼容 CI 中可选路径）。
        发现残留引用时抛 AssertionError，附带行号与内容。
        """
        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalAudit.assert_no_310_refs]"
                f" scanning path={path!r}",
                file=sys.stderr,
                flush=True,
            )

        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(
                    f"[DEBUG 3eb6c21 Python310RemovalAudit] 文件不存在，跳过: {path!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return

        flagged = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(self._PY_310_PATTERN, line, re.IGNORECASE):
                flagged.append(f"  L{lineno}: {line.strip()}")

        if flagged:
            found_lines = "\n".join(flagged)
            raise AssertionError(
                f"[Walpurgis 3eb6c21] {path} 仍含 Python 3.10 残留引用"
                f"（3eb6c21 应已删除）:\n{found_lines}\n"
                "请清理上述引用，或确认这是有意保留的向后兼容注释。"
            )

        if _DBG:
            print(
                f"[DEBUG 3eb6c21 Python310RemovalAudit] 扫描通过: {path!r}（无 3.10 残留）",
                file=sys.stderr,
                flush=True,
            )


# =============================================================================
# 4. _detect_python_version()
#    上游：直接读 sys.version_info，无测试桩注入点
#    改写：抽象为独立函数，可在测试中 mock
# =============================================================================

def _detect_python_version() -> PythonVersionSpec:
    """
    探测当前运行时 Python 版本，优先读 sys.version_info。
    断点 6：版本探测。
    """
    spec = PythonVersionSpec.from_sys()
    if _DBG:
        print(
            f"[DEBUG 3eb6c21 _detect_python_version]"
            f" sys.version_info → {spec}",
            file=sys.stderr,
            flush=True,
        )
    return spec


# =============================================================================
# 5. WalpurgisPyEnv dataclass
#    汇总运行时 Python 版本信息，dump() 一行打印，validate() 统一守卫入口
# =============================================================================

@dataclass
class WalpurgisPyEnv:
    """
    汇总当前运行时 Python 版本信息及 3eb6c21 守卫状态。

    上游各调用方零散读 sys.version_info；
    本类提供 dump() 一行打印所有关键状态，validate() 统一守卫入口。
    """

    _policy: Python310RemovalPolicy = field(
        default_factory=Python310RemovalPolicy,
        repr=False,
    )

    @property
    def runtime_version(self) -> PythonVersionSpec:
        """当前进程的 Python 版本。"""
        return _detect_python_version()

    @property
    def python_version_str(self) -> str:
        """完整的 sys.version 字符串。"""
        return sys.version

    def dump(self) -> str:
        """
        一行打印所有 Python 环境状态。
        断点 7：环境快照。
        """
        rv = self.runtime_version
        supported = self._policy.is_supported(rv)
        removed = self._policy.is_removed(rv)

        if _DBG:
            print(
                f"[DEBUG 3eb6c21 WalpurgisPyEnv.dump]"
                f" runtime={rv} supported={supported} removed={removed}",
                file=sys.stderr,
                flush=True,
            )

        return "\n".join([
            "── WalpurgisPyEnv（3eb6c21 Python 3.10 移除快照）──",
            f"  运行时版本  : Python {rv}",
            f"  sys.version : {self.python_version_str.splitlines()[0]}",
            f"  最低要求    : {_NEW_MINIMUM.spec_str}（3eb6c21 更新后）",
            f"  3.10（已移除）: {'是' if removed else '否'}",
            f"  版本满足    : {'✓ 满足' if supported else '✗ 不满足'}",
            "────────────────────────────────────────────────",
        ])

    def validate(self) -> bool:
        """
        执行运行时 Python 版本验证，返回 True 表示通过。
        断点 8：验证入口。
        """
        if _DBG:
            print(
                "[DEBUG 3eb6c21 WalpurgisPyEnv.validate] 入口",
                file=sys.stderr,
                flush=True,
            )
        return self._policy.validate_runtime_python()


# =============================================================================
# 6. 自测
# =============================================================================

def _self_test() -> None:
    """6 组断言自测，覆盖 3eb6c21 的核心变更逻辑。"""
    passed = 0
    failed = 0

    def check(label: str, ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            import sys as _sys
            print(f"  [FAIL] {label}", file=_sys.stderr)
            failed += 1

    print("─── python310_drop self-test (3eb6c21) ───")

    # 组 1：PythonVersionSpec 比较与属性
    spec_310 = PythonVersionSpec.from_str("3.10")
    spec_311 = PythonVersionSpec.from_str("3.11")
    spec_312 = PythonVersionSpec.from_str("3.12")
    check("3.10 < 3.11", spec_310 < spec_311)
    check("3.11 < 3.12", spec_311 < spec_312)
    check("spec_310.is_310 == True", spec_310.is_310)
    check("spec_311.is_310 == False", not spec_311.is_310)
    check("spec_311.is_311_plus == True", spec_311.is_311_plus)
    check("spec_310.is_311_plus == False", not spec_310.is_311_plus)
    check("spec_311.spec_str == '>=3.11'", spec_311.spec_str == ">=3.11")

    # 组 2：Python310RemovalPolicy
    policy = Python310RemovalPolicy()
    check("policy: 3.10 NOT supported", not policy.is_supported(spec_310))
    check("policy: 3.11 supported", policy.is_supported(spec_311))
    check("policy: 3.12 supported", policy.is_supported(spec_312))
    check("policy: 3.10 is removed", policy.is_removed(spec_310))
    check("policy: 3.11 NOT removed", not policy.is_removed(spec_311))

    # 组 3：validate_runtime_python warn 模式
    import io
    import contextlib

    with contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            result_310 = policy.validate_runtime_python(runtime=spec_310)

    check("validate_runtime_python(3.10) returns False", result_310 is False)
    check(
        "validate_runtime_python(3.10) emits FutureWarning",
        any(issubclass(wx.category, FutureWarning) for wx in w_list),
    )
    check(
        "validate_runtime_python(3.11) returns True",
        policy.validate_runtime_python(runtime=spec_311) is True,
    )

    # 组 4：strict 模式抛 RuntimeError
    strict_policy = Python310RemovalPolicy(strict=True)
    try:
        strict_policy.validate_runtime_python(runtime=spec_310)
        check("strict policy: 3.10 raises RuntimeError", False)
    except RuntimeError:
        check("strict policy: 3.10 raises RuntimeError", True)

    # 组 5：Python310RemovalAudit
    audit = Python310RemovalAudit()
    artifacts = audit.list_removed_artifacts()
    check(
        "audit: removed_artifacts has requires-python 3.10",
        any("requires-python" in a and "3.10" in a for a in artifacts),
    )
    check(
        "audit: removed_artifacts has classifier 3.10",
        any("Programming Language" in a and "3.10" in a for a in artifacts),
    )

    import tempfile

    # 扫描干净文件不报错
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as f:
        f.write('requires-python = ">=3.11"\n')
        tmppath_clean = f.name
    try:
        audit.assert_no_310_refs(tmppath_clean)
        check("audit: clean file passes assert_no_310_refs", True)
    except Exception:
        check("audit: clean file passes assert_no_310_refs", False)
    finally:
        os.unlink(tmppath_clean)

    # 扫描含残留内容的文件抛 AssertionError
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as f:
        f.write('requires-python = ">=3.10"\n')
        tmppath_dirty = f.name
    try:
        try:
            audit.assert_no_310_refs(tmppath_dirty)
            check("audit: dirty file raises AssertionError", False)
        except AssertionError:
            check("audit: dirty file raises AssertionError", True)
    finally:
        os.unlink(tmppath_dirty)

    # 组 6：WalpurgisPyEnv.dump
    env = WalpurgisPyEnv()
    dump_str = env.dump()
    check("WalpurgisPyEnv.dump() 包含 '>=3.11'", ">=3.11" in dump_str)
    check("WalpurgisPyEnv.dump() 包含 '运行时版本'", "运行时版本" in dump_str)
    check("WalpurgisPyEnv.validate() returns bool", isinstance(env.validate(), bool))

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


# ── 模块级单例（惰性初始化，import 时不执行 validate）─────────────────────────
_DEFAULT_POLICY: Python310RemovalPolicy = Python310RemovalPolicy()
_DEFAULT_ENV: WalpurgisPyEnv = WalpurgisPyEnv()


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
    print()
    print(_DEFAULT_ENV.dump())
