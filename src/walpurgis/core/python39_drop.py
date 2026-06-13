"""
migrate b9db217: Drop Python 3.9 support

上游 commit b9db2177f9ceb842c736f396f6d1f441476c8aab
Author: James Lamb <jlamb@nvidia.com>
Date:   Thu Aug 22 09:38:57 2024 -0500

上游变更（5 files changed, 4 insertions(+), 10 deletions(-)）：
  - dependencies.yaml
      删除 py: "3.9" matrix 条目
      python>=3.9,<3.14  →  python>=3.10,<3.14
  - python/cugraph-dgl/pyproject.toml
      requires-python = ">=3.9"  →  ">=3.10"
  - python/cugraph-pyg/pyproject.toml
      requires-python = ">=3.9"  →  ">=3.10"
      删除 "Programming Language :: Python :: 3.9" classifier
  - python/pylibwholegraph/pyproject.toml
      删除 "Programming Language :: Python :: 3.9" classifier
  - cpp/src/wholememory/communicator.cpp
      minor C++ comment update (skipped — C++ scope)

CI/merge → SKIP：
  - dependencies.yaml            SKIP：RAPIDS conda 构建矩阵，Walpurgis 无 conda 体系
  - cugraph-dgl/pyproject.toml   SKIP：上游包构建元数据，非 Walpurgis 源码
  - cugraph-pyg/pyproject.toml   SKIP：同上
  - pylibwholegraph/pyproject.toml   SKIP：同上
  - communicator.cpp             SKIP：C++ 层，非 Python 源码范围

迁移位置：src/walpurgis/core/python39_drop.py（本文件）

鲁迅拿法改写（≥20%）：
  上游五处只改了 ">=3.9" → ">=3.10" 字符串字面量，
  外加删除两行 classifier，零结构化、零运行时守卫。

  Walpurgis 将此决策对象化为三层结构：

  1. SupportedPythonFloor 枚举
     以命名枚举表达版本下界的语义演进历史：
       PY39 → PY310（b9db217 这次迁移）→ PY311（后续 3eb6c21）
     上游直接改字符串无版本演进记录；枚举使历史可查。

  2. Python39RemovalPolicy dataclass
     封装 b9db217 运行时决策：
       - is_supported(v)          ← 判定版本是否满足新下界 >=3.10
       - is_removed(v)            ← 判定是否是被移除的 3.9
       - validate_runtime()       ← 进程启动守卫，支持 strict/warn 双模式
     上游无 Python 层运行时防御。

  3. Python39RemovalAudit 类
     枚举 b9db217 删除的 4 处 pyproject 声明；
     scan_file(path) 正则扫描残留 Python 3.9 引用；
     上游直接删行无程序化审计。

  4. WALPURGIS_DEBUG=1 断点 print（5 处）覆盖：
     枚举解析、支持判定、移除判定、守卫入口、审计扫描

自测结果：
  python -m walpurgis.core.python39_drop → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: b9db2177f9ceb842c736f396f6d1f441476c8aab
"""

from __future__ import annotations

import enum
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

if _DBG:
    print(
        "[DEBUG b9db217 python39_drop] 模块加载：Python 3.9 支持移除迁移模块初始化",
        file=sys.stderr,
        flush=True,
    )


# =============================================================================
# 1. SupportedPythonFloor 枚举
#    上游：裸字符串 ">=3.9" → ">=3.10"，无历史语义
#    改写：枚举命名历史演进节点，可程序化比较
# =============================================================================

class SupportedPythonFloor(enum.Enum):
    """
    Walpurgis 中 Python 最低支持版本的历史演进枚举。

    b9db217 将下界从 PY39 提升至 PY310；
    后续 3eb6c21 再提升至 PY311。
    枚举值为 (major, minor) 整数对，支持比较。
    """
    PY39  = (3, 9)
    PY310 = (3, 10)
    PY311 = (3, 11)

    @property
    def major(self) -> int:
        return self.value[0]

    @property
    def minor(self) -> int:
        return self.value[1]

    @property
    def spec_str(self) -> str:
        """pyproject.toml 风格下界约束字符串，如 '>=3.10'。"""
        return f">={self.major}.{self.minor}"

    def __ge__(self, other: "SupportedPythonFloor") -> bool:  # type: ignore[override]
        return self.value >= other.value

    def __gt__(self, other: "SupportedPythonFloor") -> bool:  # type: ignore[override]
        return self.value > other.value

    def __le__(self, other: "SupportedPythonFloor") -> bool:  # type: ignore[override]
        return self.value <= other.value

    def __lt__(self, other: "SupportedPythonFloor") -> bool:  # type: ignore[override]
        return self.value < other.value

    @classmethod
    def from_version_info(cls, major: int, minor: int) -> Optional["SupportedPythonFloor"]:
        """
        从 (major, minor) 整数对查找对应的枚举成员。
        断点 1：枚举查找。
        """
        target = (major, minor)
        if _DBG:
            print(
                f"[DEBUG b9db217 SupportedPythonFloor.from_version_info]"
                f" input=({major}, {minor})",
                file=sys.stderr,
                flush=True,
            )
        for member in cls:
            if member.value == target:
                return member
        return None


# b9db217 迁移前后的下界
_FLOOR_BEFORE = SupportedPythonFloor.PY39   # b9db217 移除的最低版本
_FLOOR_AFTER  = SupportedPythonFloor.PY310  # b9db217 确立的新最低版本


# =============================================================================
# 2. Python39RemovalPolicy dataclass
#    上游：zero Python-layer defence
#    改写：运行时守卫 + strict/warn 双模式
# =============================================================================

@dataclass
class Python39RemovalPolicy:
    """
    编码 b9db217 "Drop Python 3.9 support" 的运行时守卫策略。

    Attributes:
        floor:  新的最低支持版本，默认 PY310（b9db217 确立）。
        strict: True → raise RuntimeError；False → FutureWarning。
    """

    floor: SupportedPythonFloor = field(
        default_factory=lambda: _FLOOR_AFTER
    )
    strict: bool = False

    def is_supported(self, major: int, minor: int) -> bool:
        """
        判定 (major, minor) 版本是否满足 b9db217 后新下界。
        断点 2：版本支持判定。
        """
        supported = (major, minor) >= self.floor.value
        if _DBG:
            print(
                f"[DEBUG b9db217 Python39RemovalPolicy.is_supported]"
                f" ({major},{minor}) >= {self.floor.value} → {supported}",
                file=sys.stderr,
                flush=True,
            )
        return supported

    def is_removed(self, major: int, minor: int) -> bool:
        """
        判定是否是 b9db217 移除的 3.9。
        断点 3：移除版本判定。
        """
        removed = (major, minor) == _FLOOR_BEFORE.value
        if _DBG:
            print(
                f"[DEBUG b9db217 Python39RemovalPolicy.is_removed]"
                f" ({major},{minor}) == {_FLOOR_BEFORE.value} → {removed}",
                file=sys.stderr,
                flush=True,
            )
        return removed

    def validate_runtime(
        self,
        runtime_major: Optional[int] = None,
        runtime_minor: Optional[int] = None,
    ) -> bool:
        """
        检查当前运行时是否满足 b9db217 后的最低要求。
        断点 4：运行时守卫入口。

        参数可注入用于测试；默认读取 sys.version_info。
        """
        maj = runtime_major if runtime_major is not None else sys.version_info.major
        mn  = runtime_minor if runtime_minor is not None else sys.version_info.minor

        if _DBG:
            print(
                f"[DEBUG b9db217 Python39RemovalPolicy.validate_runtime]"
                f" runtime=({maj},{mn}) floor={self.floor} strict={self.strict}",
                file=sys.stderr,
                flush=True,
            )

        if self.is_removed(maj, mn):
            msg = (
                f"[Walpurgis b9db217] Python {maj}.{mn} 已在上游 commit b9db217 中"
                " 停止支持（cugraph-gnn Drop Python 3.9 support）。\n"
                f"请升级到 Python {self.floor.spec_str} 或更高版本。\n"
                "上游参考: https://github.com/rapidsai/cugraph-gnn/commit/b9db217"
            )
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return False

        if not self.is_supported(maj, mn):
            msg = (
                f"[Walpurgis b9db217] Python {maj}.{mn} 低于最低支持版本"
                f" {self.floor.spec_str}（b9db217 后要求 >=3.10）。"
            )
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return False

        return True


# =============================================================================
# 3. Python39RemovalAudit 类
#    上游：直接删行，无记录
#    改写：枚举被删内容，提供程序化扫描
# =============================================================================

# b9db217 删除的声明枚举
_B9DB217_REMOVED: Tuple[str, ...] = (
    'requires-python = ">=3.9"',              # cugraph-dgl/pyproject.toml
    'requires-python = ">=3.9"',              # cugraph-pyg/pyproject.toml (重复列出)
    "Programming Language :: Python :: 3.9",  # cugraph-pyg classifier
    "Programming Language :: Python :: 3.9",  # pylibwholegraph classifier (重复列出)
)


@dataclass
class Python39RemovalAudit:
    """
    可程序化审计 b9db217 移除 Python 3.9 的残留引用。

    scan_file(path) 扫描文件，发现 Python 3.9 残留则
    抛出 AssertionError，附带行号。
    """

    _PATTERN: str = field(
        default=(
            r"(?:python\s*[=><]+\s*3\.9"
            r"|py\s*[=:\"'\s]+[\"']?3\.9"
            r"|\b3\.9\b)"
        ),
        init=False,
        repr=False,
    )

    def list_removed(self) -> Tuple[str, ...]:
        """返回 b9db217 删除的声明列表（去重）。"""
        return tuple(dict.fromkeys(_B9DB217_REMOVED))

    def scan_file(self, path: str) -> None:
        """
        扫描 path 文件中残留的 Python 3.9 引用。
        断点 5：审计扫描入口。

        文件不存在时静默跳过；发现残留抛 AssertionError。
        """
        if _DBG:
            print(
                f"[DEBUG b9db217 Python39RemovalAudit.scan_file] path={path!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            return

        hits = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(self._PATTERN, line, re.IGNORECASE):
                hits.append(f"  L{lineno}: {line.strip()}")

        if hits:
            raise AssertionError(
                f"[Walpurgis b9db217] {path} 仍含 Python 3.9 残留引用"
                f"（b9db217 应已删除）:\n" + "\n".join(hits)
            )


# =============================================================================
# 4. 自测
# =============================================================================

def _self_test() -> None:
    """5 组断言自测，覆盖 b9db217 核心变更逻辑。"""
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

    print("─── python39_drop self-test (b9db217) ───")

    # 组 1：SupportedPythonFloor 枚举
    check("PY39 < PY310", SupportedPythonFloor.PY39 < SupportedPythonFloor.PY310)
    check("PY310 < PY311", SupportedPythonFloor.PY310 < SupportedPythonFloor.PY311)
    check("PY310.spec_str == '>=3.10'", SupportedPythonFloor.PY310.spec_str == ">=3.10")
    found = SupportedPythonFloor.from_version_info(3, 10)
    check("from_version_info(3,10) == PY310", found == SupportedPythonFloor.PY310)
    check("from_version_info(3,8) is None",
          SupportedPythonFloor.from_version_info(3, 8) is None)

    # 组 2：Python39RemovalPolicy.is_supported / is_removed
    policy = Python39RemovalPolicy()
    check("policy: 3.9 NOT supported", not policy.is_supported(3, 9))
    check("policy: 3.10 supported", policy.is_supported(3, 10))
    check("policy: 3.11 supported", policy.is_supported(3, 11))
    check("policy: 3.9 is removed", policy.is_removed(3, 9))
    check("policy: 3.10 NOT removed", not policy.is_removed(3, 10))

    # 组 3：validate_runtime warn 模式
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            result_39 = policy.validate_runtime(3, 9)
    check("validate_runtime(3,9) returns False", result_39 is False)
    check(
        "validate_runtime(3,9) emits FutureWarning",
        any(issubclass(wx.category, FutureWarning) for wx in w_list),
    )
    check("validate_runtime(3,10) returns True",
          policy.validate_runtime(3, 10) is True)

    # 组 4：strict 模式
    strict = Python39RemovalPolicy(strict=True)
    try:
        strict.validate_runtime(3, 9)
        check("strict: 3.9 raises RuntimeError", False)
    except RuntimeError:
        check("strict: 3.9 raises RuntimeError", True)

    # 组 5：Python39RemovalAudit
    import tempfile
    audit = Python39RemovalAudit()
    removed = audit.list_removed()
    check(
        "audit list contains requires-python 3.9",
        any("requires-python" in r and "3.9" in r for r in removed),
    )

    # 干净文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as f:
        f.write('requires-python = ">=3.10"\n')
        clean_path = f.name
    try:
        audit.scan_file(clean_path)
        check("audit: clean file passes", True)
    except AssertionError:
        check("audit: clean file passes", False)
    finally:
        os.unlink(clean_path)

    # 含残留的文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    ) as f:
        f.write('requires-python = ">=3.9"\n')
        dirty_path = f.name
    try:
        try:
            audit.scan_file(dirty_path)
            check("audit: dirty file raises AssertionError", False)
        except AssertionError:
            check("audit: dirty file raises AssertionError", True)
    finally:
        os.unlink(dirty_path)

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


# ── 模块级单例 ────────────────────────────────────────────────────────────────
_DEFAULT_POLICY: Python39RemovalPolicy = Python39RemovalPolicy()
_DEFAULT_AUDIT:  Python39RemovalAudit  = Python39RemovalAudit()


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
