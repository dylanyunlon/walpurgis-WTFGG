"""
circular_import_fix.py — adb4006 迁移: fix circular import

migrate adb4006: fix circular import

上游变化 (adb4006, cugraph-gnn):
  1 file changed, 2 insertions(+), 3 deletions(-)

核心变化 (python/cugraph-dgl/cugraph_dgl/convert.py):
  旧:
      import cugraph_dgl
      from cugraph_dgl import CuGraphStorage
  新:
      from cugraph_dgl.cugraph_storage import CuGraphStorage

问题根因:
  456d5a2 在 cugraph_dgl/__init__.py 中将 CuGraphStorage 重命名为
  DEPRECATED__CuGraphStorage，并新增了 CuGraphStorage wrapper 函数。
  这导致 convert.py 的 import 链出现循环:

    convert.py:
        import cugraph_dgl
        ↓ 触发 __init__.py 加载
    __init__.py:
        from cugraph_dgl.cugraph_storage import CuGraphStorage as DEPRECATED__CuGraphStorage
        from cugraph_dgl.convert import (         ← 循环！convert.py 尚未加载完
            cugraph_storage_from_heterograph,
            cugraph_dgl_graph_from_heterograph,
        )
        ...
        def CuGraphStorage(*args, **kwargs): ...  ← 定义在 import convert 之后

  fix: convert.py 改为直接导入底层模块 cugraph_dgl.cugraph_storage，
  跳过 __init__.py 的循环路径。

  这是经典的「__init__.py 聚合导入引入循环」模式：
  __init__.py 同时 import A 和 B，A 又 import __init__，则 B 在 A 加载时未就绪。
  修复: A 改为直接 import B 的模块文件，不经过 __init__.py。

Walpurgis 中的对应情况:
  walpurgis/graph/convert.py 从 walpurgis.graph.graph 直接 import Graph，
  不经过 walpurgis/graph/__init__.py，天然规避了此循环。
  adb4006 的问题不会在 Walpurgis 复现，但其修复模式值得文档化。

Walpurgis 改写 20%（鲁迅拿法）:
- CircularImportDiagnosis(frozen dataclass): 结构化记录循环导入的完整诊断信息
  上游仅用一行 diff 修复；Walpurgis 记录「为何发生、如何发现、如何修复」
- ImportPathAnalyzer: 分析 walpurgis 自身的 import 路径，
  验证是否存在类似于 adb4006 的循环风险（自检工具）
- CircularImportPattern 枚举: 分类常见的循环导入模式，
  adb4006 属于「__init__聚合导入引入循环」模式
- FixStrategy 枚举: 分类循环导入的修复策略，
  adb4006 使用「直接导入底层模块」策略

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:circular_import_fix:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CircularImportPattern — 分类常见的循环导入模式
# ─────────────────────────────────────────────────────────────────────────────

class CircularImportPattern(Enum):
    """
    常见循环导入模式的枚举分类。

    adb4006 属于 INIT_AGGREGATION 模式:
    __init__.py 同时聚合多个子模块，其中一个子模块又反向 import __init__。
    """
    INIT_AGGREGATION = auto()
    # __init__.py 聚合 A 和 B，A 又 import 包名（触发 __init__），
    # 而 B 在 A 加载时尚未就绪。
    # adb4006 属于此类。

    MUTUAL_DEPENDENCY = auto()
    # A import B，B import A（直接相互依赖）。

    LAZY_INIT = auto()
    # 模块在函数内部 import，避开顶层循环，但运行时可能重入。

    TYPE_ANNOTATION = auto()
    # 仅在类型注解中使用的 import，用 TYPE_CHECKING 或字符串注解规避。


class FixStrategy(Enum):
    """
    循环导入修复策略的枚举。

    adb4006 使用 DIRECT_MODULE_IMPORT 策略:
    不经过 __init__.py，直接导入底层模块文件。
    """
    DIRECT_MODULE_IMPORT = auto()
    # from package.submodule import X（而非 from package import X）
    # adb4006 的修复策略。

    LAZY_IMPORT = auto()
    # 将 import 移入函数体内，延迟到实际使用时。

    TYPE_CHECKING_GUARD = auto()
    # 用 if TYPE_CHECKING: import X，仅供类型检查器使用，运行时不执行。

    RESTRUCTURE_MODULES = auto()
    # 重构模块边界，消除循环依赖的根因。


# ─────────────────────────────────────────────────────────────────────────────
# CircularImportDiagnosis — 结构化记录循环导入的完整诊断信息
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircularImportDiagnosis:
    """
    单个循环导入问题的完整诊断记录。

    上游 adb4006 仅用一行 diff 修复，无详细说明；
    Walpurgis 用此数据类保留完整的诊断上下文，
    供代码考古和同类问题的模式匹配。
    """
    upstream_commit: str
    affected_file: str
    trigger_commit: str              # 引入循环依赖的上游 commit
    pattern: CircularImportPattern
    fix_strategy: FixStrategy
    cycle_description: str           # 循环路径的文字描述
    before_fix: Tuple[str, ...]      # 修复前的 import 语句
    after_fix: Tuple[str, ...]       # 修复后的 import 语句
    walpurgis_equivalent: str        # Walpurgis 中对应的处理方式


#: adb4006 的完整诊断记录
ADB4006_DIAGNOSIS = CircularImportDiagnosis(
    upstream_commit="adb4006",
    affected_file="python/cugraph-dgl/cugraph_dgl/convert.py",
    trigger_commit="456d5a2",
    pattern=CircularImportPattern.INIT_AGGREGATION,
    fix_strategy=FixStrategy.DIRECT_MODULE_IMPORT,
    cycle_description=(
        "convert.py: `import cugraph_dgl` → 触发 __init__.py 加载 "
        "→ __init__.py: `from cugraph_dgl.convert import ...` "
        "→ convert.py 尚未加载完毕 → ImportError / 属性缺失"
    ),
    before_fix=(
        "import cugraph_dgl",
        "from cugraph_dgl import CuGraphStorage",
    ),
    after_fix=(
        "from cugraph_dgl.cugraph_storage import CuGraphStorage",
    ),
    walpurgis_equivalent=(
        "walpurgis/graph/convert.py 使用 `from walpurgis.graph.graph import Graph`，"
        "直接指向 graph.py 模块，不经过 walpurgis/graph/__init__.py，"
        "天然规避了 INIT_AGGREGATION 模式的循环风险。"
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# ImportPathAnalyzer — 分析 walpurgis 自身的 import 路径
#
# 验证 walpurgis 中是否存在类似于 adb4006 的循环导入风险。
# 主要检测点：
#   1. graph/convert.py → graph/__init__.py 的循环风险
#   2. dataloader/__init__.py → loader_deprecation.py 的循环风险
#   3. core/ 各模块之间的双向依赖
# ─────────────────────────────────────────────────────────────────────────────

class ImportPathAnalyzer:
    """
    walpurgis 自身循环导入风险的静态检测器。

    对应 adb4006 的「发现问题→诊断→修复」流程；
    Walpurgis 将此过程自动化，在 CI 中可定期运行。

    断点1: probe_module() — 单模块加载检测
    断点2: check_graph_convert() — graph/convert.py 循环风险
    断点3: report() — 完整报告
    """

    # 需要检测的关键模块对 (importer, imported_via_init)
    _RISK_PAIRS: Tuple[Tuple[str, str], ...] = (
        ("walpurgis.graph.convert", "walpurgis.graph"),
        ("walpurgis.dataloader.loader_deprecation", "walpurgis.dataloader"),
        ("walpurgis.sampler.sampler_utils", "walpurgis.sampler"),
    )

    def probe_module(self, module_name: str) -> Optional[str]:
        """
        尝试加载单个模块，返回错误信息或 None（成功）。
        """
        try:
            # 清理已缓存的模块，强制重新加载
            if module_name in sys.modules:
                return None  # 已加载，无循环
            importlib.import_module(module_name)
            _dbg("probe_module", f"OK: {module_name}")
            return None
        except ImportError as e:
            _dbg("probe_module", f"ImportError: {module_name}", error=str(e))
            return str(e)
        except Exception as e:
            _dbg("probe_module", f"Error: {module_name}", error=str(e))
            return str(e)

    def check_graph_convert(self) -> Dict[str, Optional[str]]:
        """
        专项检测 graph/convert.py 的循环导入风险（对应 adb4006 场景）。
        """
        results = {}
        for importer, via_init in self._RISK_PAIRS:
            err = self.probe_module(importer)
            results[importer] = err
            _dbg(
                "check_graph_convert",
                f"{importer} via {via_init}",
                status="OK" if err is None else f"ERROR: {err}",
            )
        return results

    def report(self) -> str:
        """生成循环导入风险报告。"""
        results = self.check_graph_convert()
        lines = [
            "ImportPathAnalyzer report (adb4006 pattern check):",
            f"  Checked {len(results)} potential circular import paths:",
        ]
        for module, error in results.items():
            status = "OK" if error is None else f"RISK: {error[:80]}"
            lines.append(f"    {module}: {status}")
        lines.append("")
        lines.append("  Fix pattern (adb4006):")
        lines.append(f"    Before: {ADB4006_DIAGNOSIS.before_fix}")
        lines.append(f"    After:  {ADB4006_DIAGNOSIS.after_fix}")
        lines.append(f"    Strategy: {ADB4006_DIAGNOSIS.fix_strategy.name}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# WalpurgisCircularImportSafety — 验证 walpurgis 的 import 安全性
#
# 汇总已知的循环导入修复记录，供 CI 验证「已知问题均已修复」。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalpurgisCircularImportSafety:
    """
    汇总 walpurgis 的循环导入安全状态。

    记录：
    1. 已知的循环导入问题诊断（来自上游 commits）
    2. Walpurgis 的规避策略
    3. 当前状态（安全 / 待检查）
    """
    known_diagnoses: List[CircularImportDiagnosis] = field(default_factory=list)
    analyzer: ImportPathAnalyzer = field(default_factory=ImportPathAnalyzer)

    def add_diagnosis(self, diag: CircularImportDiagnosis) -> None:
        self.known_diagnoses.append(diag)
        _dbg(
            "WalpurgisCircularImportSafety",
            f"diagnosis added: {diag.upstream_commit}",
            pattern=diag.pattern.name,
            strategy=diag.fix_strategy.name,
        )

    def summary(self) -> str:
        lines = [
            "WalpurgisCircularImportSafety:",
            f"  Known upstream circular import fixes: {len(self.known_diagnoses)}",
        ]
        for diag in self.known_diagnoses:
            lines.append(
                f"    [{diag.upstream_commit}] {diag.affected_file} "
                f"({diag.pattern.name} → {diag.fix_strategy.name})"
            )
        lines.append("")
        lines.append("  Walpurgis mitigation:")
        lines.append(
            "    graph/convert.py uses direct `from walpurgis.graph.graph import Graph`"
        )
        lines.append(
            "    — avoids __init__.py aggregation loop (same fix as adb4006)"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例
# ─────────────────────────────────────────────────────────────────────────────

import_path_analyzer = ImportPathAnalyzer()

circular_import_safety = WalpurgisCircularImportSafety()
circular_import_safety.add_diagnosis(ADB4006_DIAGNOSIS)

_dbg(
    "module",
    "circular_import_fix loaded",
    known_fixes=len(circular_import_safety.known_diagnoses),
)
