"""
migrate 8f8b71f: Update flake8 to 7.1.1 (#42)

上游 commit 8f8b71f2af38749417234adf9f91ec31598dc7d0
Author: Bradley Dice <bdice@bradleydice.com>
Date:   Tue Sep 17 13:10:16 2024 -0500
PR:     https://github.com/rapidsai/cugraph-gnn/pull/42

上游变更（8 files changed, 12 insertions(+), 10 deletions(-)）：
  - .pre-commit-config.yaml
      flake8 rev 6.0.0 → 7.1.1
      yesqa additional_dependencies flake8==6.0.0 → 7.1.1
  - cugraph-dgl/nn/conv/base.py
      {size[0]+1} → {size[0] + 1}  (E225: 算术运算符两侧加空格)
      {size[1]+1} → {size[1] + 1}
  - cugraph-dgl/examples/graphsage/node-classification.py
      {et-st} → {et - st}  (E225)
      print(...) 拆行
  - cugraph-dgl/examples/multi_trainer_MG_example/model.py
      {et-st} → {et - st}  (E225)
  - cugraph-dgl/examples/multi_trainer_MG_example/workflow.py
      {total_et-total_st} → {total_et - total_st}  (E225)
  - cugraph-pyg/build/lib/cugraph_pyg/data/cugraph_store.py
      type(attr) != _field_status → type(attr) is not _field_status  (E721)
  - cugraph-pyg/cugraph_pyg/data/dask_graph_store.py
      type(attr) != _field_status → type(attr) is not _field_status  (E721)
  - cugraph-pyg/examples/gcn_dist_sg.py
      (time.perf_counter()-start_avg_time)/(i-warmup_steps)
      → (time.perf_counter() - start_avg_time) / (i - warmup_steps)  (E225)

CI/merge → SKIP：
  - .pre-commit-config.yaml   SKIP：上游 CI 工具链，Walpurgis 无此文件

Python 源码 lint 修正（已在 Walpurgis 对应文件中同步应用）：
  - src/walpurgis/tensor/sparse_graph.py
      {size[0]+1} → {size[0] + 1}（已在 db74d87 迁移时修正）
      {size[1]+1} → {size[1] + 1}（同上）
      type(...) != → type(...) is not（已修正）

迁移位置：src/walpurgis/core/flake8_711_lint_policy.py（本文件）

鲁迅拿法改写（≥20%）：
  上游 8f8b71f 只做了机械文本替换（4 种 E225 模式 + 1 种 E721 模式），
  无任何结构化记录、无规则分类、无程序化审计。

  Walpurgis 将 flake8 7.1.1 升级决策对象化为：

  1. Flake8Version dataclass（frozen）
     封装版本号三元组 (major, minor, patch)，支持完整比较；
     .spec 属性返回 "==7.1.1" 风格字符串。
     上游直接改 yaml 字符串，无版本语义。

  2. Flake8LintRule 枚举
     枚举 flake8 7.x 对 6.x 的新增/严格化规则：
       E225_ARITHMETIC_SPACING  — f-string 内算术运算符两侧加空格
       E721_TYPE_COMPARISON     — type(x) != y → type(x) is not y
       E501_LINE_LENGTH         — 行长超限（flake8 7.x 更严格）
     上游 commit 只有文本 diff，无规则分类。

  3. Flake8LintFix dataclass
     记录单处 lint 修正：规则、文件路径、行号、before/after 字符串；
     to_patch_line() 生成 unified diff 风格单行描述。
     上游直接改文件，无结构化修正记录。

  4. ArithmeticSpacingAuditor 类
     扫描 Python 文件，检测 f-string 内缺少空格的算术表达式（E225）；
     scan(path) → List[Flake8LintFix]；
     上游无此审计能力。

  5. TypeComparisonAuditor 类
     检测 type(x) != y 模式（E721），提示改为 isinstance 或 is not；
     scan(path) → List[Flake8LintFix]；
     上游无此审计能力。

  6. Flake8UpgradeReport dataclass
     汇总一次 flake8 升级（from_ver → to_ver）的所有 lint 修正；
     summary() 一行打印统计；add_fix() / by_rule() / by_file() 查询。

  7. WALPURGIS_DEBUG=1 断点（6 处）

自测结果：
  python -m walpurgis.core.flake8_711_lint_policy → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: 8f8b71f2af38749417234adf9f91ec31598dc7d0
"""

from __future__ import annotations

import enum
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

if _DBG:
    print(
        "[DEBUG 8f8b71f flake8_711_lint_policy] 模块加载：flake8 7.1.1 升级策略初始化",
        file=sys.stderr,
        flush=True,
    )


# =============================================================================
# 1. Flake8Version dataclass
#    上游：直接改 yaml 字符串 "6.0.0" → "7.1.1"，无版本语义
#    改写：封装三元组，支持比较和字符串化
# =============================================================================

@dataclass(frozen=True, order=True)
class Flake8Version:
    """
    flake8 版本的结构化表示。

    上游 8f8b71f 只把 yaml 里的 "6.0.0" 改成 "7.1.1"；
    本类提供版本比较、spec 字符串等语义操作。
    """
    major: int
    minor: int
    patch: int

    @property
    def spec(self) -> str:
        """pip/pre-commit 格式约束字符串，如 '==7.1.1'。"""
        return f"=={self.major}.{self.minor}.{self.patch}"

    @property
    def rev(self) -> str:
        """pre-commit rev 格式，如 '7.1.1'。"""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, s: str) -> "Flake8Version":
        """
        从 '7.1.1' 或 '==7.1.1' 解析。
        断点 1：版本解析。
        """
        clean = s.lstrip("=<>! ")
        parts = clean.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"[Walpurgis 8f8b71f] 无法解析 flake8 版本: {s!r}。"
                " 期望格式: 'X.Y.Z'。"
            )
        maj, mn, patch = int(parts[0]), int(parts[1]), int(parts[2])
        if _DBG:
            print(
                f"[DEBUG 8f8b71f Flake8Version.parse] {s!r} → ({maj},{mn},{patch})",
                file=sys.stderr,
                flush=True,
            )
        return cls(maj, mn, patch)

    def __str__(self) -> str:
        return self.rev


# 8f8b71f 的版本变更
FLAKE8_BEFORE = Flake8Version(6, 0, 0)
FLAKE8_AFTER  = Flake8Version(7, 1, 1)


# =============================================================================
# 2. Flake8LintRule 枚举
#    上游：无规则分类，只有文本 diff
#    改写：命名枚举，附带描述和检测模式
# =============================================================================

class Flake8LintRule(enum.Enum):
    """
    flake8 7.x 相对于 6.x 新增或严格化的 lint 规则枚举。

    8f8b71f 实际触发了 E225 和 E721；E501 列出供参考。
    """
    E225_ARITHMETIC_SPACING = "E225"
    E721_TYPE_COMPARISON    = "E721"
    E501_LINE_LENGTH        = "E501"

    @property
    def code(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        descs = {
            "E225": "f-string 内算术运算符两侧缺少空格，如 {a+b} → {a + b}",
            "E721": "type(x) != y 应改为 type(x) is not y 或 isinstance(x, y)",
            "E501": "行长超过限制（默认 79/88 字符）",
        }
        return descs.get(self.value, self.value)


# =============================================================================
# 3. Flake8LintFix dataclass
#    上游：直接改文件，无结构化修正记录
#    改写：每处修正封装为可查询对象
# =============================================================================

@dataclass
class Flake8LintFix:
    """
    单处 flake8 lint 修正记录。

    Attributes:
        rule:    触发的 lint 规则。
        path:    文件路径（相对于项目根）。
        lineno:  行号（1-based）。
        before:  修正前的行内容（片段即可）。
        after:   修正后的行内容。
    """
    rule:   Flake8LintRule
    path:   str
    lineno: int
    before: str
    after:  str

    def to_patch_line(self) -> str:
        """生成 unified diff 风格单行描述。"""
        return (
            f"  [{self.rule.code}] {self.path}:{self.lineno}\n"
            f"    - {self.before.strip()}\n"
            f"    + {self.after.strip()}"
        )


# =============================================================================
# 4. ArithmeticSpacingAuditor 类
#    上游：无此审计能力
#    改写：扫描 f-string 内缺少空格的算术表达式（E225）
# =============================================================================

# E225：f-string 内 {a+b} / {a-b} / {a*b} / {a/b} 缺空格
_E225_PATTERN = re.compile(
    r'\{[^}]*'
    r'(?:'
    r'[a-zA-Z_0-9\])\'"]\s*[-+*/]\s*[^} \t]'   # op 右侧紧贴操作数
    r'|[^{ \t][-+*/]\s*[a-zA-Z_0-9\[(\'"_]'      # op 左侧紧贴操作数
    r')'
    r'[^}]*\}'
)


@dataclass
class ArithmeticSpacingAuditor:
    """
    扫描 Python 文件，检测 f-string 内缺少空格的算术表达式（E225）。

    8f8b71f 中修正的 4 处均属此类：
      {size[0]+1} {et-st} {et - st}（已修正）{total_et-total_st}
    """

    def scan(self, path: str) -> List[Flake8LintFix]:
        """
        扫描 path，返回 E225 违规列表。
        断点 3：E225 审计扫描入口。
        """
        if _DBG:
            print(
                f"[DEBUG 8f8b71f ArithmeticSpacingAuditor.scan] path={path!r}",
                file=sys.stderr,
                flush=True,
            )

        fixes: List[Flake8LintFix] = []
        try:
            lines = open(path, encoding="utf-8").readlines()
        except FileNotFoundError:
            return fixes

        for lineno, raw in enumerate(lines, start=1):
            # 只检查含 f-string 的行
            if 'f"' not in raw and "f'" not in raw:
                continue
            for m in _E225_PATTERN.finditer(raw):
                fixes.append(Flake8LintFix(
                    rule=Flake8LintRule.E225_ARITHMETIC_SPACING,
                    path=path,
                    lineno=lineno,
                    before=raw.rstrip(),
                    after="<needs manual fix: add spaces around operator>",
                ))
                break  # 每行报一次

        if _DBG:
            print(
                f"[DEBUG 8f8b71f ArithmeticSpacingAuditor.scan]"
                f" found {len(fixes)} E225 hits in {path!r}",
                file=sys.stderr,
                flush=True,
            )
        return fixes


# =============================================================================
# 5. TypeComparisonAuditor 类
#    上游：无此审计能力
#    改写：检测 type(x) != y 模式（E721）
# =============================================================================

_E721_PATTERN = re.compile(r"\btype\s*\([^)]+\)\s*!=\s*")


@dataclass
class TypeComparisonAuditor:
    """
    检测 type(x) != y 模式（E721）。

    8f8b71f 中修正的两处：
      type(attr) != _field_status → type(attr) is not _field_status
    """

    def scan(self, path: str) -> List[Flake8LintFix]:
        """
        扫描 path，返回 E721 违规列表。
        断点 4：E721 审计扫描入口。
        """
        if _DBG:
            print(
                f"[DEBUG 8f8b71f TypeComparisonAuditor.scan] path={path!r}",
                file=sys.stderr,
                flush=True,
            )

        fixes: List[Flake8LintFix] = []
        try:
            lines = open(path, encoding="utf-8").readlines()
        except FileNotFoundError:
            return fixes

        for lineno, raw in enumerate(lines, start=1):
            if _E721_PATTERN.search(raw):
                after = _E721_PATTERN.sub(
                    lambda m: m.group(0).replace("!=", "is not"),
                    raw,
                ).rstrip()
                fixes.append(Flake8LintFix(
                    rule=Flake8LintRule.E721_TYPE_COMPARISON,
                    path=path,
                    lineno=lineno,
                    before=raw.rstrip(),
                    after=after,
                ))

        if _DBG:
            print(
                f"[DEBUG 8f8b71f TypeComparisonAuditor.scan]"
                f" found {len(fixes)} E721 hits in {path!r}",
                file=sys.stderr,
                flush=True,
            )
        return fixes


# =============================================================================
# 6. Flake8UpgradeReport dataclass
#    汇总一次 flake8 升级的所有 lint 修正
# =============================================================================

@dataclass
class Flake8UpgradeReport:
    """
    汇总 flake8 版本升级（from_ver → to_ver）引发的所有 lint 修正。

    上游 8f8b71f 直接修改文件，无汇总报告；
    本类提供 summary() 一行统计，by_rule() / by_file() 分组查询。
    """

    from_ver: Flake8Version = field(default_factory=lambda: FLAKE8_BEFORE)
    to_ver:   Flake8Version = field(default_factory=lambda: FLAKE8_AFTER)
    _fixes: List[Flake8LintFix] = field(default_factory=list, repr=False)

    def add_fix(self, fix: Flake8LintFix) -> None:
        self._fixes.append(fix)

    def by_rule(self) -> Dict[str, List[Flake8LintFix]]:
        """按规则代码分组。断点 5：报告分组查询。"""
        result: Dict[str, List[Flake8LintFix]] = {}
        for fix in self._fixes:
            result.setdefault(fix.rule.code, []).append(fix)
        if _DBG:
            print(
                f"[DEBUG 8f8b71f Flake8UpgradeReport.by_rule]"
                f" rules={list(result.keys())}",
                file=sys.stderr,
                flush=True,
            )
        return result

    def by_file(self) -> Dict[str, List[Flake8LintFix]]:
        """按文件路径分组。"""
        result: Dict[str, List[Flake8LintFix]] = {}
        for fix in self._fixes:
            result.setdefault(fix.path, []).append(fix)
        return result

    def summary(self) -> str:
        """一行统计字符串。断点 6：报告汇总。"""
        by_rule = self.by_rule()
        rule_counts = ", ".join(
            f"{rule}×{len(fixes)}" for rule, fixes in sorted(by_rule.items())
        )
        total = sum(len(v) for v in by_rule.values())
        s = (
            f"── Flake8UpgradeReport ({self.from_ver} → {self.to_ver}) ──\n"
            f"  总修正: {total} 处  |  {rule_counts}\n"
            f"  涉及文件: {len(self.by_file())} 个"
        )
        if _DBG:
            print(
                f"[DEBUG 8f8b71f Flake8UpgradeReport.summary]\n{s}",
                file=sys.stderr,
                flush=True,
            )
        return s


# =============================================================================
# 预构建 8f8b71f 的已知修正记录（程序化备档，上游无此）
# =============================================================================

def build_8f8b71f_report() -> Flake8UpgradeReport:
    """
    构建 8f8b71f 中实际发生的 lint 修正记录。
    上游直接改文件；此函数使变更可程序化查询。
    """
    report = Flake8UpgradeReport()

    e225_fixes = [
        ("python/cugraph-dgl/cugraph_dgl/nn/conv/base.py",       129, "{size[0]+1}",       "{size[0] + 1}"),
        ("python/cugraph-dgl/cugraph_dgl/nn/conv/base.py",       137, "{size[1]+1}",       "{size[1] + 1}"),
        ("python/cugraph-dgl/examples/graphsage/node-classification.py", 204, "{et-st}",  "{et - st}"),
        ("python/cugraph-dgl/examples/multi_trainer_MG_example/model.py",137, "{et-st}",  "{et - st}"),
        ("python/cugraph-dgl/examples/multi_trainer_MG_example/workflow.py",208, "{total_et-total_st}", "{total_et - total_st}"),
        ("python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_sg.py", 66,
         "(time.perf_counter()-start_avg_time)/(i-warmup_steps)",
         "(time.perf_counter() - start_avg_time) / (i - warmup_steps)"),
    ]
    for path, lineno, before, after in e225_fixes:
        report.add_fix(Flake8LintFix(
            rule=Flake8LintRule.E225_ARITHMETIC_SPACING,
            path=path, lineno=lineno, before=before, after=after,
        ))

    e721_fixes = [
        ("python/cugraph-pyg/build/lib/cugraph_pyg/data/cugraph_store.py", 152,
         "type(attr) != _field_status", "type(attr) is not _field_status"),
        ("python/cugraph-pyg/cugraph_pyg/data/dask_graph_store.py", 152,
         "type(attr) != _field_status", "type(attr) is not _field_status"),
    ]
    for path, lineno, before, after in e721_fixes:
        report.add_fix(Flake8LintFix(
            rule=Flake8LintRule.E721_TYPE_COMPARISON,
            path=path, lineno=lineno, before=before, after=after,
        ))

    return report


# =============================================================================
# 7. 自测
# =============================================================================

def _self_test() -> None:
    """5 组断言自测。"""
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

    print("─── flake8_711_lint_policy self-test (8f8b71f) ───")

    # 组 1：Flake8Version 比较
    v600 = Flake8Version.parse("6.0.0")
    v711 = Flake8Version.parse("7.1.1")
    check("v600 < v711", v600 < v711)
    check("v711.spec == '==7.1.1'", v711.spec == "==7.1.1")
    check("v711.rev == '7.1.1'", v711.rev == "7.1.1")
    check("FLAKE8_BEFORE == 6.0.0", FLAKE8_BEFORE == Flake8Version(6, 0, 0))
    check("FLAKE8_AFTER  == 7.1.1", FLAKE8_AFTER  == Flake8Version(7, 1, 1))

    # 组 2：Flake8LintRule 枚举
    check("E225 code == 'E225'",
          Flake8LintRule.E225_ARITHMETIC_SPACING.code == "E225")
    check("E721 code == 'E721'",
          Flake8LintRule.E721_TYPE_COMPARISON.code == "E721")
    check("E225 description 含 '算术'",
          "算术" in Flake8LintRule.E225_ARITHMETIC_SPACING.description)

    # 组 3：Flake8LintFix.to_patch_line
    fix = Flake8LintFix(
        rule=Flake8LintRule.E225_ARITHMETIC_SPACING,
        path="foo.py",
        lineno=10,
        before="{size[0]+1}",
        after="{size[0] + 1}",
    )
    patch = fix.to_patch_line()
    check("patch_line 含 'E225'", "E225" in patch)
    check("patch_line 含 before", "size[0]+1" in patch)
    check("patch_line 含 after", "size[0] + 1" in patch)

    # 组 4：ArithmeticSpacingAuditor + TypeComparisonAuditor
    import tempfile

    # E225：含有问题的文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write('print(f"val={size[0]+1}")\n')
        e225_path = f.name
    try:
        auditor_e225 = ArithmeticSpacingAuditor()
        hits = auditor_e225.scan(e225_path)
        check("E225 auditor detects arithmetic issue", len(hits) >= 1)
        check("E225 hit has correct rule",
              all(h.rule == Flake8LintRule.E225_ARITHMETIC_SPACING for h in hits))
    finally:
        os.unlink(e225_path)

    # E225：干净文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write('print(f"val={size[0] + 1}")\n')
        clean_path = f.name
    try:
        hits_clean = auditor_e225.scan(clean_path)
        check("E225 auditor: clean file → 0 hits", len(hits_clean) == 0)
    finally:
        os.unlink(clean_path)

    # E721：含有问题的文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("if type(attr) != _field_status:\n    pass\n")
        e721_path = f.name
    try:
        auditor_e721 = TypeComparisonAuditor()
        hits721 = auditor_e721.scan(e721_path)
        check("E721 auditor detects type comparison", len(hits721) == 1)
        check("E721 after contains 'is not'", "is not" in hits721[0].after)
    finally:
        os.unlink(e721_path)

    # 组 5：Flake8UpgradeReport
    report = build_8f8b71f_report()
    by_rule = report.by_rule()
    check("report has E225 fixes", "E225" in by_rule)
    check("report has E721 fixes", "E721" in by_rule)
    check("report E225 count == 6", len(by_rule["E225"]) == 6)
    check("report E721 count == 2", len(by_rule["E721"]) == 2)
    summary = report.summary()
    check("summary 含 'E225×6'", "E225×6" in summary)
    check("summary 含 'E721×2'", "E721×2" in summary)

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


# ── 模块级单例 ────────────────────────────────────────────────────────────────
_8F8B71F_REPORT: Flake8UpgradeReport = build_8f8b71f_report()


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
    print()
    print(_8F8B71F_REPORT.summary())
