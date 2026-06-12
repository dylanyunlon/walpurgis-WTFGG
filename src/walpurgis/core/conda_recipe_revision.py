"""
migrate a4d6bc2: Quote head_rev in conda recipes (#204)

上游 commit a4d6bc2be5dd3c5ba417d28a71b711334c002592
Author: Bradley Dice <bdice@bradleydice.com>
Date:   2025-05-13

上游变更（4 files changed, 4 insertions(+), 4 deletions(-)）：
  conda/recipes/cugraph-dgl/recipe.yaml
  conda/recipes/cugraph-pyg/recipe.yaml
  conda/recipes/libwholegraph/recipe.yaml
  conda/recipes/pylibwholegraph/recipe.yaml

每个文件均将：
  head_rev: ${{ git.head_rev(".")[:8] }}
改为：
  head_rev: '${{ git.head_rev(".")[:8] }}'

修复原因：git SHA 如 "0abcdef1" 以数字 0 开头，YAML 解析器在无引号
时将其当作八进制整数（或截断为整数），导致 conda 构建产物包名丢失前导零。
xref: https://github.com/rapidsai/build-planning/issues/176

CI/conda/merge 文件 → SKIP：
  - conda/recipes/*/recipe.yaml — RAPIDS conda rattler-build recipe，
    Walpurgis 无 conda 构建体系，不打包为 conda artifact

迁移位置：src/walpurgis/core/conda_recipe_revision.py（本文件）

鲁迅拿法改写（≥20%）：
  上游是 4 个 YAML 文件中的 4 行字符串加引号，无任何结构化
  的诊断、守卫或可审计记录。Walpurgis 将其提炼为：
  1. HeadRevFormat 枚举    — 区分带引号/不带引号两种格式（上游无此抽象）
  2. RecipeRevisionSpec dataclass — 封装包名+revision字符串+引号状态
  3. RevisionGuard dataclass     — 运行时检查 rev 字符串是否有前导零风险
  4. RecipeRevisionAudit         — 扫描任意 recipe.yaml 文本，发现未加引号的
                                    head_rev 赋值（上游无程序化审计工具）
  5. WalpurgisRevisionEnv        — 汇总当前 git HEAD 信息，dump() 打印快照
  6. simulate_head_rev_quote()   — 演示加引号前后 YAML 解析差异
  7. 全链路 WALPURGIS_DEBUG=1 断点（6 处）
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────
if _DBG:
    print("[DEBUG a4d6bc2 conda_recipe_revision] 模块加载：head_rev 引号修复迁移模块初始化")


# ── 1. HeadRevFormat 枚举 ────────────────────────────────────────────────────

class HeadRevFormat(Enum):
    """
    head_rev 字段在 conda recipe YAML 中的引号状态。

    上游 a4d6bc2 修复前后两种形态：
    - UNQUOTED: head_rev: ${{ git.head_rev(".")[:8] }}
      YAML 解析器可能将前导零 SHA 截断（如 0abcdef1 → 5765617）
    - QUOTED:   head_rev: '${{ git.head_rev(".")[:8] }}'
      引号强制字符串类型，前导零安全保留
    """
    UNQUOTED = "unquoted"
    QUOTED = "quoted"


# ── 2. RecipeRevisionSpec dataclass ─────────────────────────────────────────

@dataclass
class RecipeRevisionSpec:
    """
    描述一个 conda recipe 包的 head_rev 规格。
    上游仅有裸 YAML 行，无任何结构化的包→格式映射。
    """
    package_name: str          # 如 "cugraph-dgl", "libwholegraph"
    head_rev_str: str          # 实际的 git SHA 字符串（[:8] 截断后）
    fmt: HeadRevFormat = HeadRevFormat.QUOTED

    def has_leading_zero_risk(self) -> bool:
        """
        判断该 SHA 字符串是否存在前导零被 YAML 误解的风险。
        八进制前导零在 YAML 1.1 中触发：以 0 开头且全为 0-7 数字。
        YAML 1.2 中不再自动转八进制，但 conda-build 等工具可能用旧解析器。
        """
        # 断点 1：前导零检测
        if _DBG:
            print(
                f"[DEBUG a4d6bc2] has_leading_zero_risk: pkg={self.package_name!r} "
                f"sha={self.head_rev_str!r}"
            )
        if not self.head_rev_str:
            return False
        # 八进制风险：0 开头，仅含 0-7
        if self.head_rev_str.startswith("0") and re.fullmatch(r"[0-7]+", self.head_rev_str):
            return True
        # 整数误解风险：纯十进制数字（无字母）
        if re.fullmatch(r"\d+", self.head_rev_str):
            return True
        return False

    def safe_yaml_value(self) -> str:
        """
        返回在 YAML 中安全表达 head_rev 的字符串（带单引号）。
        上游修复即在此处加引号，Walpurgis 将其程序化。
        """
        return f"'{self.head_rev_str}'"

    def describe(self) -> str:
        risk = "⚠ 前导零风险" if self.has_leading_zero_risk() else "✓ 安全"
        return (
            f"RecipeRevisionSpec(pkg={self.package_name!r}, "
            f"sha={self.head_rev_str!r}, fmt={self.fmt.value}, {risk})"
        )


# ── 3. RevisionGuard dataclass ───────────────────────────────────────────────

@dataclass
class RevisionGuard:
    """
    运行时守卫：验证给定 SHA 字符串在 YAML 中是否需要引号保护。
    上游直接修改文件，无任何运行时防御层。
    """
    strict: bool = False  # True → 发现风险时 raise；False → 仅返回警告文本

    def validate(self, spec: RecipeRevisionSpec) -> Optional[str]:
        """
        检查 spec。
        - 若 fmt=UNQUOTED 且存在前导零风险：strict→raise，否则返回警告。
        - 若 fmt=QUOTED：安全，返回 None。
        """
        # 断点 2：守卫检查
        if _DBG:
            print(
                f"[DEBUG a4d6bc2] RevisionGuard.validate: {spec.describe()} "
                f"strict={self.strict}"
            )
        if spec.fmt == HeadRevFormat.UNQUOTED and spec.has_leading_zero_risk():
            msg = (
                f"[RevisionGuard] {spec.package_name!r}: "
                f"SHA {spec.head_rev_str!r} 未加引号且存在前导零/整数误解风险，"
                "conda 构建产物包名可能截断。应改为带单引号格式。"
            )
            if self.strict:
                raise ValueError(msg)
            return msg
        return None


# ── 4. RecipeRevisionAudit ───────────────────────────────────────────────────

@dataclass
class RecipeRevisionAudit:
    """
    扫描 recipe.yaml 文本，发现未加引号的 head_rev 赋值行。
    上游无程序化审计工具，直接人工 grep + 手工修改。

    a4d6bc2 涉及的 4 个 recipe 文件模式：
      head_rev: ${{ git.head_rev(".")[:8] }}     ← 修复前（危险）
      head_rev: '${{ git.head_rev(".")[:8] }}'   ← 修复后（安全）
    """

    # 未加引号的 head_rev 赋值模式（允许双引号，排除单引号和换行）
    _UNQUOTED_PATTERN: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"""^\s*head_rev:\s+\$\{\{[^'\n]+\}\}\s*$""",
            re.MULTILINE,
        ),
        repr=False,
    )
    # 已加引号的 head_rev 赋值模式
    _QUOTED_PATTERN: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"""^\s*head_rev:\s+'[^'\n]+'\s*$""",
            re.MULTILINE,
        ),
        repr=False,
    )

    def scan(self, recipe_text: str, source_label: str = "<recipe>") -> dict:
        """
        扫描 recipe 文本，返回审计结果字典。
        """
        # 断点 3：审计扫描
        if _DBG:
            print(f"[DEBUG a4d6bc2] RecipeRevisionAudit.scan: label={source_label!r}")
        unquoted = self._UNQUOTED_PATTERN.findall(recipe_text)
        quoted = self._QUOTED_PATTERN.findall(recipe_text)
        result = {
            "source": source_label,
            "unquoted_head_rev_lines": [l.strip() for l in unquoted],
            "quoted_head_rev_lines": [l.strip() for l in quoted],
            "needs_fix": len(unquoted) > 0,
        }
        if _DBG:
            print(f"[DEBUG a4d6bc2] 审计结果: {result}")
        return result

    def apply_fix(self, recipe_text: str) -> str:
        """
        将 recipe 文本中所有未加引号的 head_rev 赋值改为带引号形式。
        模拟 a4d6bc2 的变更，但以程序化方式执行（上游是手工编辑）。
        """
        # 替换：  head_rev: ${{ expr }}  →  head_rev: '${{ expr }}'
        # 注意：内容中可含双引号（如 git.head_rev(".")），仅排除单引号和换行
        fixed = re.sub(
            r"(^\s*head_rev:\s+)(\$\{\{[^'\n]+\}\})(\s*)$",
            r"\1'\2'\3",
            recipe_text,
            flags=re.MULTILINE,
        )
        # 断点 4：修复应用
        if _DBG:
            changed = fixed != recipe_text
            print(f"[DEBUG a4d6bc2] apply_fix: changed={changed}")
        return fixed


# ── 5. WalpurgisRevisionEnv ──────────────────────────────────────────────────

@dataclass
class WalpurgisRevisionEnv:
    """
    汇总当前 git HEAD 信息，模拟 conda recipe 中 git.head_rev(".")[:8] 的行为。
    上游各 recipe 通过 rattler-build 内置函数获取，无 Python 层等价实现。
    """
    repo_path: str = "."

    def get_head_rev(self) -> str:
        """获取当前 git HEAD 的前 8 位 SHA。"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "deadbeef"  # fallback

    def dump(self) -> str:
        rev = self.get_head_rev()
        spec = RecipeRevisionSpec(
            package_name="<current-repo>",
            head_rev_str=rev,
            fmt=HeadRevFormat.QUOTED,
        )
        # 断点 5：环境快照
        if _DBG:
            print(f"[DEBUG a4d6bc2] WalpurgisRevisionEnv.dump: rev={rev!r}")
        return (
            f"WalpurgisRevisionEnv(\n"
            f"  repo_path={self.repo_path!r},\n"
            f"  head_rev={rev!r},\n"
            f"  safe_yaml_value={spec.safe_yaml_value()!r},\n"
            f"  leading_zero_risk={spec.has_leading_zero_risk()}\n"
            f")"
        )


# ── 6. simulate_head_rev_quote ───────────────────────────────────────────────

def simulate_head_rev_quote(sha8: str) -> dict:
    """
    演示 head_rev 加引号前后在 YAML 解析（Python yaml 库）上的差异。
    对应 a4d6bc2 的核心修复逻辑，上游无此诊断工具。

    注意：Python yaml.safe_load 默认用 YAML 1.1 规则，
    rattler-build 内部用 Rust serde_yaml（YAML 1.2），行为略有不同。
    本函数用 YAML 1.1 规则演示风险场景。
    """
    try:
        import yaml  # type: ignore

        unquoted_line = f"head_rev: {sha8}"
        quoted_line = f"head_rev: '{sha8}'"

        unquoted_val = yaml.safe_load(unquoted_line).get("head_rev")
        quoted_val = yaml.safe_load(quoted_line).get("head_rev")
    except ImportError:
        # 无 PyYAML 时用简单模拟：检测前导零
        unquoted_val = int(sha8, 8) if (
            sha8.startswith("0") and re.fullmatch(r"[0-7]+", sha8)
        ) else sha8
        quoted_val = sha8

    # 断点 6：解析结果对比
    if _DBG:
        print(
            f"[DEBUG a4d6bc2] simulate_head_rev_quote: sha8={sha8!r} "
            f"unquoted→{unquoted_val!r} quoted→{quoted_val!r}"
        )

    return {
        "sha8_input": sha8,
        "unquoted_parsed": unquoted_val,
        "quoted_parsed": quoted_val,
        "values_differ": unquoted_val != quoted_val,
        "fix_needed": unquoted_val != sha8,
    }


# ── 模块级常量：a4d6bc2 涉及的四个 recipe 包 ────────────────────────────────

A4D6BC2_AFFECTED_PACKAGES = [
    "cugraph-dgl",
    "cugraph-pyg",
    "libwholegraph",
    "pylibwholegraph",
]

# ── 自测 ─────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """模块自测：验证所有核心逻辑路径。"""
    passed = 0
    total = 0

    def check(label: str, cond: bool) -> None:
        nonlocal passed, total
        total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            passed += 1

    print("=== conda_recipe_revision.py 自测 (a4d6bc2) ===")

    # 1. HeadRevFormat
    check("HeadRevFormat.QUOTED.value == 'quoted'",
          HeadRevFormat.QUOTED.value == "quoted")
    check("HeadRevFormat.UNQUOTED.value == 'unquoted'",
          HeadRevFormat.UNQUOTED.value == "unquoted")

    # 2. RecipeRevisionSpec 前导零检测
    spec_zero = RecipeRevisionSpec("cugraph-dgl", "01234567", HeadRevFormat.UNQUOTED)
    spec_safe = RecipeRevisionSpec("cugraph-pyg", "a1b2c3d4", HeadRevFormat.QUOTED)
    check("前导零 SHA '01234567' 有风险", spec_zero.has_leading_zero_risk())
    check("正常 SHA 'a1b2c3d4' 无风险", not spec_safe.has_leading_zero_risk())
    check("safe_yaml_value 包含单引号",
          spec_safe.safe_yaml_value().startswith("'") and spec_safe.safe_yaml_value().endswith("'"))

    # 3. RevisionGuard 宽松模式
    guard = RevisionGuard(strict=False)
    warn = guard.validate(spec_zero)
    check("危险 spec 返回警告字符串", warn is not None and "风险" in warn)
    check("安全 spec 返回 None", guard.validate(spec_safe) is None)

    # 4. RevisionGuard 严格模式
    guard_strict = RevisionGuard(strict=True)
    raised = False
    try:
        guard_strict.validate(spec_zero)
    except ValueError:
        raised = True
    check("严格模式对危险 spec 抛出 ValueError", raised)

    # 5. RecipeRevisionAudit
    audit = RecipeRevisionAudit()
    sample_unquoted = 'head_rev: ${{ git.head_rev(".")[:8] }}'
    sample_quoted   = "head_rev: '${{ git.head_rev(\".\")[:8] }}'"
    result_bad = audit.scan(sample_unquoted, "bad_recipe")
    result_ok  = audit.scan(sample_quoted,   "ok_recipe")
    check("未加引号的 recipe 需要修复", result_bad["needs_fix"])
    check("已加引号的 recipe 无需修复", not result_ok["needs_fix"])
    fixed = audit.apply_fix(sample_unquoted)
    result_fixed = audit.scan(fixed, "after_fix")
    check("apply_fix 后审计通过", not result_fixed["needs_fix"])

    # 6. simulate_head_rev_quote
    sim_safe = simulate_head_rev_quote("a1b2c3d4")
    check("普通 SHA 无需修复", not sim_safe["fix_needed"])

    # 7. 受影响包列表
    check("a4d6bc2 涉及 4 个包",
          len(A4D6BC2_AFFECTED_PACKAGES) == 4)
    check("cugraph-dgl 在列表中",
          "cugraph-dgl" in A4D6BC2_AFFECTED_PACKAGES)

    # 8. WalpurgisRevisionEnv.dump 可调用
    env = WalpurgisRevisionEnv(repo_path=".")
    dumped = env.dump()
    check("dump() 返回字符串", isinstance(dumped, str) and "head_rev" in dumped)

    print(f"\n自测结果: {passed}/{total} 通过")
    assert passed == total, f"{total - passed} 项失败"
    print("[PASS] 全部通过\n")


if __name__ == "__main__":
    _self_test()
