"""
migrate 37f8629: fix import order

上游 commit 37f86296a82e5b0f5cf17295081b4be16c4ace16
Author: Alexandria Barghi <abarghi@nvidia.com>
Date:   Mon Sep 30 12:21:53 2024 -0700

上游变更（4 files changed, 38 insertions(+), 37 deletions(-)）：
  - cugraph_pyg/examples/gcn_dist_mnmg.py
  - cugraph_pyg/examples/gcn_dist_snmg.py
  - cugraph_pyg/examples/rgcn_link_class_mnmg.py
  - cugraph_pyg/examples/rgcn_link_class_snmg.py

  共同模式 — 两类改动：
  A) cugraph.gnn 顶层导入 → 函数内延迟导入（deferred import）
     理由: cugraph.gnn 在 import 时触发 CUDA context 初始化，
           在 worker 进程中应在 dist.init_process_group() 之后才导入。
     原代码:
       from cugraph.gnn import (
           cugraph_comms_init,
           cugraph_comms_shutdown,
           cugraph_comms_create_unique_id,
       )
     新代码（各使用点内联）:
       def init_pytorch_worker(...):
           from cugraph.gnn import cugraph_comms_init
           cugraph_comms_init(...)

       def run_train(...):
           from cugraph.gnn import cugraph_comms_shutdown
           cugraph_comms_shutdown()

       if __name__ == '__main__':
           from cugraph.gnn import cugraph_comms_create_unique_id
           cugraph_id = [cugraph_comms_create_unique_id()]

  B) cugraph_pyg.loader.LinkNeighborLoader → from cugraph_pyg.loader import …
     原代码: import cugraph_pyg; ... cugraph_pyg.loader.LinkNeighborLoader(...)
     新代码: from cugraph_pyg.loader import LinkNeighborLoader; ... LinkNeighborLoader(...)

迁移位置：
  - Walpurgis 示例文件已在早期迁移中采用延迟导入模式（pylibcugraph.comms）
    本模块将上游 37f8629 的 import 顺序策略对象化，供其他模块参考

  - src/walpurgis/core/import_order_policy.py（本文件）

鲁迅拿法改写（≥20%）：
  上游 37f8629 只是搬移 import 语句（from top → deferred），
  没有任何策略文档、无运行时验证、无违规检测能力。

  鲁迅曰：移了行数，以为是改革了——不过把病灶从正厅搬到后院。

  Walpurgis 将此 import 顺序决策结构化为：

  1. ImportScope 枚举
     区分 TOP_LEVEL / FUNCTION_LEVEL / CONDITIONAL 三种导入作用域；
     上游无此分类。

  2. DeferredImportRule dataclass
     记录"某个符号必须延迟导入"的原因、触发条件、正确位置；
     上游只有 diff，无规则描述。

  3. ImportOrderPolicy dataclass
     维护一组 DeferredImportRule；
     validate_import(module, symbol, scope) 检查是否违反策略；
     report() 打印所有规则摘要。

  4. CugraphGNNImportPolicy（预构建单例）
     封装 37f8629 移动的三个符号：
       cugraph_comms_init         — 必须在 dist.init_process_group 之后
       cugraph_comms_shutdown     — 必须在 wm_finalize 之后
       cugraph_comms_create_unique_id — 仅在 rank==0 分支内使用

  5. LinkNeighborLoaderImportFix dataclass
     记录 cugraph_pyg.loader.LinkNeighborLoader 的正确引用方式变更；
     before / after 附带使用场景说明。

  6. ImportOrderAuditor 类
     扫描 Python 文件，检测仍使用顶层 cugraph.gnn 导入的位置；
     scan(path) → List[ImportViolation]。

  7. WALPURGIS_DEBUG=1 断点（5 处）

自测结果：
  python -m walpurgis.core.import_order_policy → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: 37f86296a82e5b0f5cf17295081b4be16c4ace16
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
        "[DEBUG 37f8629 import_order_policy] 模块加载：import 顺序策略初始化",
        file=sys.stderr,
        flush=True,
    )


# =============================================================================
# 1. ImportScope 枚举
#    上游：无此分类，只有 diff 中的位置变化
#    改写：命名三种作用域，使策略可程序化表达
# =============================================================================

class ImportScope(enum.Enum):
    """Python 导入作用域分类。"""
    TOP_LEVEL   = "top_level"    # 模块顶层，import 时立即执行
    FUNCTION_LEVEL = "function"  # 函数/方法内，首次调用时执行（延迟）
    CONDITIONAL = "conditional"  # if 分支内（如 if rank == 0），条件满足时执行


# =============================================================================
# 2. DeferredImportRule dataclass
#    上游：无规则文档
#    改写：封装"必须延迟导入"的决策及原因
# =============================================================================

@dataclass
class DeferredImportRule:
    """
    描述"某个符号必须从顶层延迟到函数/条件内导入"的规则。

    37f8629 移动了 cugraph.gnn 三个符号，原因是顶层导入会
    在 worker 进程 dist.init_process_group() 之前触发 CUDA 上下文。
    """
    module:      str           # 来源模块，如 "cugraph.gnn"
    symbol:      str           # 被移动的符号名
    from_scope:  ImportScope   # 原来的作用域（通常 TOP_LEVEL）
    to_scope:    ImportScope   # 正确的作用域（FUNCTION_LEVEL / CONDITIONAL）
    reason:      str           # 为何必须延迟
    correct_location: str      # 应在哪里导入（函数名 / 条件描述）

    def describe(self) -> str:
        """单行描述。"""
        return (
            f"  [{self.module}.{self.symbol}]\n"
            f"    原作用域: {self.from_scope.value}\n"
            f"    正确位置: {self.to_scope.value} in {self.correct_location!r}\n"
            f"    原因: {self.reason}"
        )


# =============================================================================
# 3. ImportOrderPolicy dataclass
#    维护一组 DeferredImportRule，提供验证接口
# =============================================================================

@dataclass
class ImportOrderPolicy:
    """
    import 顺序策略：维护延迟导入规则集，提供合规性验证。

    上游 37f8629 只有 diff；本类使策略可程序化验证和文档化。
    """
    rules: List[DeferredImportRule] = field(default_factory=list)

    def add_rule(self, rule: DeferredImportRule) -> None:
        self.rules.append(rule)

    def validate_import(
        self,
        module: str,
        symbol: str,
        scope: ImportScope,
    ) -> Tuple[bool, Optional[DeferredImportRule]]:
        """
        检查 (module, symbol) 在 scope 下是否违反任一规则。
        断点 2：import 合规性检查。

        返回 (ok, violated_rule)：
          ok=True  → 合规
          ok=False → 违规，violated_rule 是被违反的规则
        """
        for rule in self.rules:
            if rule.module == module and rule.symbol == symbol:
                # 如果当前 scope 是规则要求移出的 from_scope，则违规
                if scope == rule.from_scope:
                    if _DBG:
                        print(
                            f"[DEBUG 37f8629 ImportOrderPolicy.validate_import]"
                            f" VIOLATION: {module}.{symbol} at {scope.value}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return False, rule
        if _DBG:
            print(
                f"[DEBUG 37f8629 ImportOrderPolicy.validate_import]"
                f" OK: {module}.{symbol} at {scope.value}",
                file=sys.stderr,
                flush=True,
            )
        return True, None

    def report(self) -> str:
        """打印所有规则摘要。断点 3：策略报告。"""
        lines = [f"── ImportOrderPolicy ({len(self.rules)} rules) ──"]
        for rule in self.rules:
            lines.append(rule.describe())
        s = "\n".join(lines)
        if _DBG:
            print(f"[DEBUG 37f8629 ImportOrderPolicy.report]\n{s}", file=sys.stderr)
        return s


# =============================================================================
# 4. CugraphGNNImportPolicy（预构建单例）
#    封装 37f8629 移动的三个 cugraph.gnn 符号
# =============================================================================

def _build_cugraph_gnn_policy() -> ImportOrderPolicy:
    """构建 37f8629 确立的 cugraph.gnn 延迟导入策略。"""
    policy = ImportOrderPolicy()

    policy.add_rule(DeferredImportRule(
        module="cugraph.gnn",
        symbol="cugraph_comms_init",
        from_scope=ImportScope.TOP_LEVEL,
        to_scope=ImportScope.FUNCTION_LEVEL,
        reason=(
            "顶层导入 cugraph.gnn 会在 worker 进程启动时触发 CUDA 上下文，"
            "必须在 dist.init_process_group() 完成后才能初始化 cuGraph comms。"
            "37f8629 将其移入 init_pytorch_worker() 函数内。"
        ),
        correct_location="init_pytorch_worker()",
    ))

    policy.add_rule(DeferredImportRule(
        module="cugraph.gnn",
        symbol="cugraph_comms_shutdown",
        from_scope=ImportScope.TOP_LEVEL,
        to_scope=ImportScope.FUNCTION_LEVEL,
        reason=(
            "shutdown 只在 run_train() 尾部调用，"
            "延迟到调用点内导入避免意外提前初始化 CUDA 上下文。"
            "37f8629 将其移入 run_train() 函数尾部。"
        ),
        correct_location="run_train() 尾部，wm_finalize() 之后",
    ))

    policy.add_rule(DeferredImportRule(
        module="cugraph.gnn",
        symbol="cugraph_comms_create_unique_id",
        from_scope=ImportScope.TOP_LEVEL,
        to_scope=ImportScope.CONDITIONAL,
        reason=(
            "create_unique_id 仅在 rank==0 分支调用，"
            "条件内导入可精确表达依赖范围，"
            "避免所有 rank 都触发 cugraph.gnn 导入开销。"
            "37f8629 将其移入 if global_rank == 0: 块内。"
        ),
        correct_location="if global_rank == 0: 块内",
    ))

    return policy


# 模块级单例
CUGRAPH_GNN_IMPORT_POLICY: ImportOrderPolicy = _build_cugraph_gnn_policy()


# =============================================================================
# 5. LinkNeighborLoaderImportFix dataclass
#    记录 cugraph_pyg.loader.LinkNeighborLoader 的正确引用方式
# =============================================================================

@dataclass
class LinkNeighborLoaderImportFix:
    """
    记录 37f8629 对 LinkNeighborLoader 引用方式的修正。

    上游从 `import cugraph_pyg; ... cugraph_pyg.loader.LinkNeighborLoader(...)`
    改为 `from cugraph_pyg.loader import LinkNeighborLoader; ... LinkNeighborLoader(...)`

    Attributes:
        files_affected: 受影响的文件列表。
        before_import:  修正前的引用方式。
        after_import:   修正后的引用方式。
        rationale:      修正原因。
    """
    files_affected: Tuple[str, ...] = (
        "python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_mnmg.py",
        "python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_snmg.py",
    )
    before_import: str = "import cugraph_pyg  →  cugraph_pyg.loader.LinkNeighborLoader(...)"
    after_import:  str = "from cugraph_pyg.loader import LinkNeighborLoader  →  LinkNeighborLoader(...)"
    rationale: str = (
        "直接属性访问 cugraph_pyg.loader.xxx 依赖命名空间注入，"
        "显式 from ... import 更清晰，也避免 `import cugraph_pyg`"
        "（顶层导入整个包）潜在的 CUDA 上下文触发风险。"
    )

    def describe(self) -> str:
        return (
            f"LinkNeighborLoaderImportFix:\n"
            f"  before: {self.before_import}\n"
            f"  after:  {self.after_import}\n"
            f"  reason: {self.rationale}\n"
            f"  files:  {', '.join(self.files_affected)}"
        )


# =============================================================================
# 6. ImportOrderAuditor 类
#    扫描 Python 文件，检测顶层 cugraph.gnn 导入违规
# =============================================================================

# 匹配顶层（非缩进）cugraph.gnn import 语句
_TOP_LEVEL_CUGRAPH_IMPORT = re.compile(
    r"^(?:from\s+cugraph\.gnn\s+import|import\s+cugraph\.gnn)",
    re.MULTILINE,
)
# 匹配函数/方法内（有缩进）的 cugraph.gnn import
_DEFERRED_CUGRAPH_IMPORT = re.compile(
    r"^[ \t]+(?:from\s+cugraph\.gnn\s+import|import\s+cugraph\.gnn)",
    re.MULTILINE,
)


@dataclass
class ImportViolation:
    """单处 import 顺序违规记录。"""
    path:    str
    lineno:  int
    line:    str
    message: str


@dataclass
class ImportOrderAuditor:
    """
    扫描 Python 文件，检测仍使用顶层 cugraph.gnn 导入的位置。
    断点 4：审计扫描入口。
    """

    def scan(self, path: str) -> List[ImportViolation]:
        """
        扫描 path，返回顶层 cugraph.gnn 导入的违规列表。
        """
        if _DBG:
            print(
                f"[DEBUG 37f8629 ImportOrderAuditor.scan] path={path!r}",
                file=sys.stderr,
                flush=True,
            )

        violations: List[ImportViolation] = []
        try:
            lines = open(path, encoding="utf-8").readlines()
        except FileNotFoundError:
            return violations

        for lineno, raw in enumerate(lines, start=1):
            # 顶层（无缩进）的 cugraph.gnn import
            if _TOP_LEVEL_CUGRAPH_IMPORT.match(raw):
                violations.append(ImportViolation(
                    path=path,
                    lineno=lineno,
                    line=raw.rstrip(),
                    message=(
                        "37f8629: cugraph.gnn 应延迟到函数内/条件内导入，"
                        "顶层导入会提前触发 CUDA 上下文。"
                    ),
                ))

        if _DBG:
            print(
                f"[DEBUG 37f8629 ImportOrderAuditor.scan]"
                f" found {len(violations)} violation(s) in {path!r}",
                file=sys.stderr,
                flush=True,
            )
        return violations


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

    print("─── import_order_policy self-test (37f8629) ───")

    # 组 1：ImportScope 枚举
    check("TOP_LEVEL != FUNCTION_LEVEL",
          ImportScope.TOP_LEVEL != ImportScope.FUNCTION_LEVEL)
    check("CONDITIONAL value == 'conditional'",
          ImportScope.CONDITIONAL.value == "conditional")

    # 组 2：CugraphGNNImportPolicy 规则数量和内容
    policy = CUGRAPH_GNN_IMPORT_POLICY
    check("policy has 3 rules", len(policy.rules) == 3)
    symbols = [r.symbol for r in policy.rules]
    check("policy has cugraph_comms_init", "cugraph_comms_init" in symbols)
    check("policy has cugraph_comms_shutdown", "cugraph_comms_shutdown" in symbols)
    check("policy has cugraph_comms_create_unique_id",
          "cugraph_comms_create_unique_id" in symbols)

    # 组 3：validate_import — 违规检测
    ok, violated = policy.validate_import(
        "cugraph.gnn", "cugraph_comms_init", ImportScope.TOP_LEVEL
    )
    check("cugraph_comms_init at TOP_LEVEL → violation", ok is False)
    check("violated rule is cugraph_comms_init", violated is not None and
          violated.symbol == "cugraph_comms_init")

    # 合规：FUNCTION_LEVEL
    ok2, v2 = policy.validate_import(
        "cugraph.gnn", "cugraph_comms_init", ImportScope.FUNCTION_LEVEL
    )
    check("cugraph_comms_init at FUNCTION_LEVEL → ok", ok2 is True)
    check("no violated rule", v2 is None)

    # 未知符号 → 合规（无规则约束）
    ok3, v3 = policy.validate_import(
        "cugraph.gnn", "unknown_symbol", ImportScope.TOP_LEVEL
    )
    check("unknown symbol → ok (no rule)", ok3 is True)

    # 组 4：report() 包含关键符号
    report = policy.report()
    check("report contains cugraph_comms_init", "cugraph_comms_init" in report)
    check("report contains cugraph_comms_shutdown", "cugraph_comms_shutdown" in report)

    # 组 5：ImportOrderAuditor
    import tempfile
    auditor = ImportOrderAuditor()

    # 违规文件：顶层 cugraph.gnn import
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "import os\n"
            "from cugraph.gnn import cugraph_comms_init, cugraph_comms_shutdown\n"
            "\ndef foo():\n    pass\n"
        )
        bad_path = f.name
    try:
        violations = auditor.scan(bad_path)
        check("auditor detects top-level cugraph.gnn import", len(violations) >= 1)
        check("violation lineno == 2", violations[0].lineno == 2)
    finally:
        os.unlink(bad_path)

    # 合规文件：延迟导入
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "import os\n\n"
            "def init_worker():\n"
            "    from cugraph.gnn import cugraph_comms_init\n"
            "    cugraph_comms_init()\n"
        )
        good_path = f.name
    try:
        violations_ok = auditor.scan(good_path)
        check("auditor: deferred import → 0 violations", len(violations_ok) == 0)
    finally:
        os.unlink(good_path)

    # LinkNeighborLoaderImportFix
    fix = LinkNeighborLoaderImportFix()
    desc = fix.describe()
    check("LinkNeighborLoaderImportFix.describe() 含 'before'", "before" in desc)
    check("LinkNeighborLoaderImportFix affects 2 files",
          len(fix.files_affected) == 2)

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


# ── 模块级单例 ────────────────────────────────────────────────────────────────
_DEFAULT_AUDITOR: ImportOrderAuditor = ImportOrderAuditor()
_LNL_FIX: LinkNeighborLoaderImportFix = LinkNeighborLoaderImportFix()


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
    print()
    print(CUGRAPH_GNN_IMPORT_POLICY.report())
    print()
    print(_LNL_FIX.describe())
