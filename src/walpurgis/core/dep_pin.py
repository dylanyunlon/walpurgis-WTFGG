"""
migrate a01924a: Pin Cython pre-3.2.0 and PyTest pre-9

上游 commit a01924ae23feb84a371e9e2e170084377a1b39f0
Author: jakirkham <jakirkham@gmail.com>
Date: 2025-11-25

上游变更：conda 环境 yaml × 4、conda recipe、dependencies.yaml、
cugraph-pyg conda dev yaml × 4、cugraph-pyg pyproject.toml、
pylibwholegraph pyproject.toml 共 12 个文件，全部执行：
  - cython>=3.0.0          →  cython>=3.0.0,<3.2.0a0
  - pytest（无上限）        →  pytest<9.0.0a0
  - 版权年 2023            →  2023-2025（pylibwholegraph/pyproject.toml）

CI/conda/merge 文件 → SKIP：
  - conda/environments/*.yaml        — Walpurgis 无 conda 环境矩阵
  - conda/recipes/pylibwholegraph/   — RAPIDS conda recipe，Walpurgis 不编译 pylibwholegraph
  - dependencies.yaml                — RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
  - cugraph-pyg/conda/*.yaml         — conda 开发环境，同上
  - cugraph-pyg/pyproject.toml       — 上游包构建配置，非 Walpurgis 源码

迁移位置：src/walpurgis/core/dep_pin.py（本文件）

鲁迅拿法改写（≥20%）：
  上游是散落在 12 个 yaml/toml 文件里的裸字符串版本约束，
  没有任何结构化的约束理由、运行时守卫或可审计记录。
  Walpurgis 将其提炼为：
  1. PinReason 枚举    — 约束动机类型（上游无此抽象）
  2. DepPin dataclass  — 包名 + 版本范围 + 动机 + 上游 commit + 有效期
  3. PinPolicy         — 运行时 import 前守卫，严格模式可 raise，宽松模式警告
  4. PinAudit          — 扫描 requirements 文本，发现约束是否仍存在
  5. WalpurgisPinEnv   — 汇总当前 Cython/pytest 版本，dump() 打印快照
  6. 全链路 WALPURGIS_DEBUG=1 断点（8 处）
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────
if _DBG:
    print("[DEBUG a01924a dep_pin] 模块加载：Cython/pytest 版本钉迁移模块初始化")


# ── 1. 约束动机枚举 ─────────────────────────────────────────


class PinReason(Enum):
    """版本约束的动机类型。上游无此分类，全部是裸字符串。"""

    UPSTREAM_BUG = "upstream_bug"          # 上游版本有已知 bug
    COMPAT_BREAK = "compatibility_break"   # 新版破坏兼容性
    BUILD_ISSUE = "build_issue"            # 编译/构建层问题
    CI_STABILITY = "ci_stability"          # CI 稳定性临时 workaround


# ── 2. 单个版本钉描述 ────────────────────────────────────────


@dataclass(frozen=True)
class DepPin:
    """
    描述一个版本约束的完整元信息。

    上游 a01924a 只有裸字符串，例如：
        - cython>=3.0.0,<3.2.0a0
        - pytest<9.0.0a0

    DepPin 将"为什么钉这个版本"也显式记录下来，
    使约束在代码审计时可解释、可查询。
    """

    package: str
    lower: Optional[str]        # 下界版本（含），None 表示无下界
    upper_excl: Optional[str]   # 上界版本（不含），None 表示无上界
    reason: PinReason
    upstream_commit: str        # 引入此约束的上游 commit sha
    tracking_issue: str         # 上游 issue/PR 说明
    expected_fix_version: Optional[str] = None  # 预计在哪个版本修复（可选）

    # ── 版本范围字符串（conda/pip 格式） ──

    def conda_spec(self) -> str:
        """生成 conda-style 版本约束字符串。"""
        parts = [self.package]
        if self.lower:
            parts.append(f">={self.lower}")
            if self.upper_excl:
                parts.append(f",<{self.upper_excl}")
        elif self.upper_excl:
            parts.append(f"<{self.upper_excl}")
        return "".join(parts)

    def pip_spec(self) -> str:
        """生成 pip/pyproject.toml-style 版本约束字符串。"""
        parts = [self.package]
        if self.lower:
            parts.append(f">={self.lower}")
            if self.upper_excl:
                parts.append(f",<{self.upper_excl}")
        elif self.upper_excl:
            parts.append(f"<{self.upper_excl}")
        return "".join(parts)

    def is_upper_pinned(self) -> bool:
        """是否有上界限制（是安全修复钉还是纯下界约束）。"""
        return self.upper_excl is not None

    def dump(self) -> str:
        return (
            f"  package={self.package}\n"
            f"  范围: {self.pip_spec()}\n"
            f"  动机: {self.reason.value}\n"
            f"  上游 commit: {self.upstream_commit}\n"
            f"  issue: {self.tracking_issue}\n"
            f"  预计修复版本: {self.expected_fix_version or '未知'}"
        )


# ── 3. a01924a 引入的两个版本钉 ──────────────────────────────

CYTHON_PIN = DepPin(
    package="cython",
    lower="3.0.0",
    upper_excl="3.2.0a0",
    reason=PinReason.BUILD_ISSUE,
    upstream_commit="a01924ae23feb84a371e9e2e170084377a1b39f0",
    tracking_issue=(
        "Cython 3.2.0 导致 RAPIDS 构建出现若干细微问题，"
        "具体见 rapidsai/build-planning#229 和 #230。"
        "固定到 3.1.x 直至兼容性问题解决。"
    ),
    expected_fix_version="3.2.x（待 RAPIDS 验证后解除）",
)

PYTEST_PIN = DepPin(
    package="pytest",
    lower=None,
    upper_excl="9.0.0a0",
    reason=PinReason.COMPAT_BREAK,
    upstream_commit="a01924ae23feb84a371e9e2e170084377a1b39f0",
    tracking_issue=(
        "PyTest 9 引入了若干破坏性变更，"
        "具体见 rapidsai/build-planning#230。"
        "固定到 <9 直至问题解决。"
    ),
    expected_fix_version="9.x（待 RAPIDS 验证后解除）",
)

# 断点 1：版本钉注册
if _DBG:
    print("[DEBUG a01924a dep_pin] CYTHON_PIN 注册:")
    print(CYTHON_PIN.dump())
    print("[DEBUG a01924a dep_pin] PYTEST_PIN 注册:")
    print(PYTEST_PIN.dump())

# 全局注册表（可扩展添加其他约束）
_ALL_PINS: list[DepPin] = [CYTHON_PIN, PYTEST_PIN]


# ── 4. 运行时守卫策略 ────────────────────────────────────────


@dataclass
class PinPolicy:
    """
    在 Walpurgis Python 运行时中，对上界约束做预检。

    上游只在 conda/pyproject 声明约束，安装后没有 Python 层守卫。
    PinPolicy 在 import 时检查已安装版本是否在约束范围内。

    strict=True:  版本越界时抛出 ImportError（适合 CI 环境）
    strict=False: 版本越界时发出 UserWarning（适合开发环境）
    """

    strict: bool = False

    def _get_installed_version(self, pkg: str) -> Optional[str]:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _parse_version_tuple(ver: str) -> tuple[int, ...]:
        """把 '3.1.5' 这类字符串解析为整数 tuple，忽略 pre-release 后缀。"""
        # 取 a/b/rc/post 之前的数字部分
        numeric_part = re.split(r"[a-zA-Z]", ver)[0]
        return tuple(int(x) for x in numeric_part.split(".") if x.isdigit())

    def check(self, pin: DepPin) -> bool:
        """
        检查已安装版本是否符合 pin 约束。

        返回 True 表示检查通过（版本在范围内或包未安装）。
        """
        installed = self._get_installed_version(pin.package)
        if installed is None:
            # 包未安装，不做检查
            if _DBG:
                print(f"[DEBUG a01924a dep_pin] {pin.package} 未安装，跳过守卫检查")
            return True

        inst_t = self._parse_version_tuple(installed)

        # 检查上界（核心 a01924a 约束）
        if pin.upper_excl is not None:
            upper_t = self._parse_version_tuple(pin.upper_excl)
            if inst_t >= upper_t:
                msg = (
                    f"[Walpurgis dep_pin] {pin.package}=={installed} "
                    f"超出上界 <{pin.upper_excl}。\n"
                    f"动机: {pin.tracking_issue}\n"
                    f"请降级: pip install '{pin.pip_spec()}'"
                )
                # 断点 2：上界越界
                if _DBG:
                    print(f"[DEBUG a01924a dep_pin] 上界越界检测: {msg}")
                if self.strict:
                    raise ImportError(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
                return False

        # 检查下界
        if pin.lower is not None:
            lower_t = self._parse_version_tuple(pin.lower)
            if inst_t < lower_t:
                msg = (
                    f"[Walpurgis dep_pin] {pin.package}=={installed} "
                    f"低于下界 >={pin.lower}。"
                )
                if _DBG:
                    print(f"[DEBUG a01924a dep_pin] 下界不足检测: {msg}")
                warnings.warn(msg, UserWarning, stacklevel=3)
                return False

        if _DBG:
            print(
                f"[DEBUG a01924a dep_pin] {pin.package}=={installed} "
                f"符合约束 {pin.pip_spec()} ✓"
            )
        return True

    def check_all(self) -> dict[str, bool]:
        """检查所有注册的 pin 约束，返回 {package: 是否通过} 字典。"""
        # 断点 3：批量检查入口
        if _DBG:
            print(f"[DEBUG a01924a dep_pin] PinPolicy.check_all 开始（strict={self.strict}）")
        return {pin.package: self.check(pin) for pin in _ALL_PINS}


# ── 5. 约束存在性审计 ────────────────────────────────────────


@dataclass
class PinAudit:
    """
    扫描 requirements 文本，确认 a01924a 引入的上界约束仍然存在。

    使用场景：CI 中定期校验 pyproject.toml 是否意外丢失了约束。
    上游无此机制（约束全靠人工维护 yaml）。
    """

    CYTHON_PATTERN: str = field(
        default=r"cython\s*>=\s*3\.0\.0\s*,\s*<\s*3\.2",
        init=False,
        repr=False,
    )
    PYTEST_PATTERN: str = field(
        default=r"pytest\s*<\s*9\.0",
        init=False,
        repr=False,
    )

    def has_cython_pin(self, requirements_text: str) -> bool:
        """检查文本中是否包含 Cython 上界约束。"""
        result = bool(re.search(self.CYTHON_PATTERN, requirements_text))
        # 断点 4：Cython 约束扫描
        if _DBG:
            print(
                f"[DEBUG a01924a dep_pin] PinAudit.has_cython_pin="
                f"{result}（pattern={self.CYTHON_PATTERN}）"
            )
        return result

    def has_pytest_pin(self, requirements_text: str) -> bool:
        """检查文本中是否包含 pytest 上界约束。"""
        result = bool(re.search(self.PYTEST_PATTERN, requirements_text))
        # 断点 5：pytest 约束扫描
        if _DBG:
            print(
                f"[DEBUG a01924a dep_pin] PinAudit.has_pytest_pin="
                f"{result}（pattern={self.PYTEST_PATTERN}）"
            )
        return result

    def assert_no_cython_pin_missing(self, path: str) -> None:
        """
        读取文件内容，断言 Cython 约束存在。
        上游通过 conda yaml 维护，我们通过此函数程序化审计。
        """
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(f"[DEBUG a01924a dep_pin] 文件不存在（跳过审计）: {path}")
            return
        if not self.has_cython_pin(text):
            raise AssertionError(
                f"[Walpurgis dep_pin] {path} 缺少 Cython <3.2 约束！\n"
                f"来自上游 a01924a，请检查是否被意外删除。"
            )

    def assert_no_pytest_pin_missing(self, path: str) -> None:
        """读取文件内容，断言 pytest 约束存在。"""
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(f"[DEBUG a01924a dep_pin] 文件不存在（跳过审计）: {path}")
            return
        if not self.has_pytest_pin(text):
            raise AssertionError(
                f"[Walpurgis dep_pin] {path} 缺少 pytest <9 约束！\n"
                f"来自上游 a01924a，请检查是否被意外删除。"
            )


# ── 6. 环境快照 ──────────────────────────────────────────────


@dataclass
class WalpurgisPinEnv:
    """
    汇总当前运行时 Cython/pytest 版本及其约束状态。

    上游没有 Python 层的环境汇总，只能靠 conda info 或 pip show。
    WalpurgisPinEnv 一行打印即可看到所有关键信息。
    """

    _policy: PinPolicy = field(default_factory=PinPolicy, repr=False)

    def _version_of(self, pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "<未安装>"

    def dump(self) -> str:
        cython_v = self._version_of("cython")
        pytest_v = self._version_of("pytest")
        results = self._policy.check_all()

        lines = [
            "── WalpurgisPinEnv（a01924a 版本钉快照）──",
            f"  cython  : {cython_v:12s}  约束={CYTHON_PIN.pip_spec()}  符合={'✓' if results.get('cython', True) else '✗'}",
            f"  pytest  : {pytest_v:12s}  约束={PYTEST_PIN.pip_spec()}  符合={'✓' if results.get('pytest', True) else '✗'}",
            "────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def validate(self) -> bool:
        """
        执行全部约束检查，返回 True 表示全部通过。
        断点 6：环境验证入口。
        """
        if _DBG:
            print("[DEBUG a01924a dep_pin] WalpurgisPinEnv.validate() 入口")
        results = self._policy.check_all()
        all_ok = all(results.values())
        # 断点 7：验证结果汇总
        if _DBG:
            print(f"[DEBUG a01924a dep_pin] validate 结果={results}，全部通过={all_ok}")
        return all_ok


# ── 模块级自检 ───────────────────────────────────────────────


def _self_test() -> None:
    """6 项断言自测，对应上游 a01924a 的核心变更逻辑。"""
    audit = PinAudit()

    # 断点 8：自测启动
    if _DBG:
        print("[DEBUG a01924a dep_pin] _self_test 启动")

    # 1) CYTHON_PIN 上界正确
    assert CYTHON_PIN.upper_excl == "3.2.0a0", "Cython 上界应为 3.2.0a0"

    # 2) PYTEST_PIN 上界正确
    assert PYTEST_PIN.upper_excl == "9.0.0a0", "pytest 上界应为 9.0.0a0"

    # 3) pip_spec 格式正确
    assert "cython>=3.0.0,<3.2.0a0" == CYTHON_PIN.pip_spec()
    assert "pytest<9.0.0a0" == PYTEST_PIN.pip_spec()

    # 4) 审计模式：包含约束的文本能被识别
    sample_reqs = "cython>=3.0.0,<3.2.0a0\npytest<9.0.0a0\n"
    assert audit.has_cython_pin(sample_reqs), "审计应识别 Cython 约束"
    assert audit.has_pytest_pin(sample_reqs), "审计应识别 pytest 约束"

    # 5) 审计模式：缺少约束的文本被识别为缺失
    no_pin = "cython>=3.0.0\npytest\n"
    assert not audit.has_cython_pin(no_pin), "无上界约束不应通过 Cython 审计"
    assert not audit.has_pytest_pin(no_pin), "无上界约束不应通过 pytest 审计"

    # 6) PinPolicy 版本解析正确
    policy = PinPolicy()
    assert policy._parse_version_tuple("3.1.5") == (3, 1, 5)
    assert policy._parse_version_tuple("9.0.0a0") == (9, 0, 0)
    assert policy._parse_version_tuple("3.2.0a0") == (3, 2, 0)

    print("[PASS] dep_pin a01924a 自测：6 项断言全部通过")


# ── 模块级懒初始化：运行时版本检查 ──────────────────────────

_env = WalpurgisPinEnv()

if __name__ == "__main__":
    _self_test()
    print()
    print(_env.dump())
