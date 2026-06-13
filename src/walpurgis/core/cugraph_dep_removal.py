# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit b10f279
# 原标题: Remove cugraph Python library as a dependency (#271)
# 上游 PR: https://github.com/rapidsai/cugraph-gnn/pull/271
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「革命尚未成功，同志仍须努力。」— 此处借为: 解耦尚未彻底，依赖仍需清理。
# —— 鲁迅《热风·随感录六十六》（化用语境）
#
# b10f279 是一次依赖图的外科手术：将 cugraph（Python 高层库）从
# cugraph-pyg 的 *运行时* 依赖降级为 *测试时可选* 依赖。
#
# 背景：cugraph Python 库体积庞大且引入额外的 RAPIDS 传递依赖，
#       而 cugraph-pyg 实际在运行时只需要 pylibcugraph（C 扩展层）
#       的少量低层接口（采样、图存储读写）。
#       #249 提出解耦目标，b10f279 完成核心替换。
#
# 主要变更（36 个文件，234 行增加，163 行删除）：
#
#   pyproject.toml（cugraph-pyg）：
#     - 从 `dependencies` 移除 "cugraph==25.10.*"
#     - 新增 "pylibcugraph==25.10.*" 到 `dependencies`
#     - 将 "cugraph==25.10.*" 移至 `[project.optional-dependencies] test`（仅测试需要）
#
#   python/cugraph-pyg/cugraph_pyg/utils/imports.py（核心新增）：
#     - 新增 MissingModule 类：属性访问时抛出 RuntimeError，
#       令"未安装可选依赖"的代码路径给出明确错误而非裸 ImportError
#     - 新增 import_optional(mod, default_mod_class) 函数：
#       模块不可用时返回 MissingModule 实例，而非直接失败
#
#   python/cugraph-pyg/cugraph_pyg/sampler/io/reader.py：
#     - 将 `from cugraph.utilities.utils import import_optional, MissingModule`
#       替换为本地 `from cugraph_pyg.utils.imports import import_optional, MissingModule`
#     - 新增 DistSampleReader 类（原本依赖 cugraph 提供，现在内联实现）：
#       基于 parquet 文件名正则解析 batch 索引，支持 rank 过滤，
#       消除对 cugraph 高层 IO 路径的依赖
#
#   其余文件（feature_store, graph_store, loaders, sampler, tensor）：
#     - 将 `from cugraph.xxx import import_optional` 替换为本地版本
#     - 依赖 cugraph 的示例文件（bitcoin_mnmg, gcn_dist_mnmg 等）：
#       更新 import 路径，使用 pylibcugraph 直接接口
#
# Walpurgis 20% 改写要点：
#   1. DependencyTier 枚举 — 将"运行时依赖 vs 测试依赖 vs 已移除"
#      显式化为三档枚举，对应 pyproject.toml 的 dependencies/test/已删除
#   2. ImportGuard 类 — 扩展上游 MissingModule：
#      额外记录 install_hint（告知如何补装）和 removed_in（版本标记）
#   3. OptionalImportRegistry — 全局注册表，追踪当前运行时哪些可选模块
#      已成功加载、哪些降级为 ImportGuard，便于诊断
#   4. try_import() 函数 — 封装 import_optional()，
#      同时向 OptionalImportRegistry 注册结果
#   5. WALPURGIS_DEBUG=1 全链路输出：注册表初始化、每次 import 尝试结果、
#      MissingModule 属性访问触发时的调用栈摘要

from __future__ import annotations

import enum
import importlib
import os as _os
import sys as _sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


_dbg("cugraph_dep_removal", "module init — 迁移自 cugraph-gnn b10f279")

# ---------------------------------------------------------------------------
# 枚举：依赖层级
# ---------------------------------------------------------------------------

class DependencyTier(enum.Enum):
    """
    描述一个依赖包在 pyproject.toml 中所处的层级。

    b10f279 将 cugraph 从 RUNTIME 降级到 TEST_ONLY，
    同时将 pylibcugraph 提升为 RUNTIME。
    Walpurgis 将此变化语义化为枚举，方便自动化依赖审计工具查询。
    """

    RUNTIME = "runtime"        # pyproject.toml [project.dependencies]
    TEST_ONLY = "test_only"    # pyproject.toml [project.optional-dependencies] test
    REMOVED = "removed"        # 从依赖树中完全移除（历史标记）

    def is_available_at_runtime(self) -> bool:
        return self == DependencyTier.RUNTIME


# ---------------------------------------------------------------------------
# 数据类：带安装提示的 ImportGuard
# ---------------------------------------------------------------------------

@dataclass
class ImportGuard:
    """
    扩展上游 MissingModule：属性访问时抛出 RuntimeError，
    同时附加 install_hint 和 removed_in 字段以改善错误信息。

    上游 MissingModule.__getattr__ 只输出 "This feature requires the {name} package/module"，
    Walpurgis 版本额外说明如何补装以及该依赖从哪个版本起变为可选。
    """

    name: str
    install_hint: str = ""          # 例如 "pip install cugraph==25.10.*"
    removed_in: str = ""            # 例如 "cugraph-gnn b10f279 (PR #271)"
    tier: DependencyTier = DependencyTier.TEST_ONLY

    def __getattr__(self, attr: str) -> Any:
        # 避免 dataclass 字段递归触发
        if attr in ("name", "install_hint", "removed_in", "tier"):
            raise AttributeError(attr)
        hint = f"  安装提示: {self.install_hint}" if self.install_hint else ""
        removed = f"  变更记录: {self.removed_in}" if self.removed_in else ""
        _dbg("ImportGuard.__getattr__", f"模块 {self.name!r} 属性 {attr!r} 被访问但模块不可用")
        raise RuntimeError(
            f"[Walpurgis:ImportGuard] 此功能需要 {self.name!r} 包。\n"
            f"  当前依赖层级: {self.tier.value}\n"
            f"{hint}\n{removed}".strip()
        )


# ---------------------------------------------------------------------------
# 全局注册表：追踪可选模块加载状态
# ---------------------------------------------------------------------------

@dataclass
class _OptionalImportRegistry:
    """
    记录当前进程中所有通过 try_import() 加载的可选模块状态。

    上游 import_optional() 每次调用都是独立的，无全局状态，
    Walpurgis 增加注册表以支持审计（WALPURGIS_DEBUG=1 时可 dump 完整状态）。
    """

    _registry: Dict[str, bool] = field(default_factory=dict)  # mod_name → 是否成功加载

    def record(self, mod_name: str, success: bool) -> None:
        self._registry[mod_name] = success
        _dbg("OptionalImportRegistry", f"记录 {mod_name!r}: {'✓ 可用' if success else '✗ 不可用'}")

    def is_available(self, mod_name: str) -> Optional[bool]:
        """返回 True/False/None（None 表示尚未尝试加载）"""
        return self._registry.get(mod_name, None)

    def dump(self) -> Dict[str, bool]:
        return dict(self._registry)


_registry = _OptionalImportRegistry()

_dbg("cugraph_dep_removal", f"OptionalImportRegistry 初始化完成")


# ---------------------------------------------------------------------------
# 核心函数：try_import / import_optional
# ---------------------------------------------------------------------------

def import_optional(mod: str, default_mod_class: Type = ImportGuard) -> Any:
    """
    尝试 import mod，失败时返回 default_mod_class 的实例。

    与上游 cugraph.utilities.utils.import_optional 等价，
    但使用本地 ImportGuard 替代 MissingModule，并注册到全局 _registry。

    Parameters
    ----------
    mod:
        要导入的模块名（支持点分路径，如 "torch.cuda"）
    default_mod_class:
        模块不可用时实例化的替代类；默认 ImportGuard
    """
    _dbg("import_optional", f"尝试 import {mod!r}")
    try:
        result = importlib.import_module(mod)
        _registry.record(mod, True)
        _dbg("import_optional", f"{mod!r} 加载成功")
        return result
    except ModuleNotFoundError:
        _registry.record(mod, False)
        _dbg("import_optional", f"{mod!r} 不可用，返回 {default_mod_class.__name__} 实例")
        if default_mod_class is ImportGuard:
            return ImportGuard(
                name=mod,
                install_hint=f"pip install {mod.split('.')[0]}",
                removed_in="cugraph-gnn b10f279 (PR #271) — 降为可选依赖",
            )
        return default_mod_class(mod)


def try_import(
    mod: str,
    install_hint: str = "",
    tier: DependencyTier = DependencyTier.TEST_ONLY,
) -> Any:
    """
    import_optional() 的 Walpurgis 扩展版本：
    支持传入 install_hint 和 DependencyTier，返回更丰富的 ImportGuard。

    推荐在 Walpurgis 内部代码中使用此函数而非裸 import_optional()。
    """
    _dbg("try_import", f"mod={mod!r} tier={tier.value}")
    try:
        result = importlib.import_module(mod)
        _registry.record(mod, True)
        return result
    except ModuleNotFoundError:
        _registry.record(mod, False)
        return ImportGuard(
            name=mod,
            install_hint=install_hint or f"pip install {mod.split('.')[0]}",
            removed_in="cugraph-gnn b10f279 (PR #271)",
            tier=tier,
        )


def package_available(requirement: str) -> bool:
    """
    检查给定 requirement 字符串对应的包是否可用。

    上游 cugraph_pyg/utils/imports.py 中已有同名函数，
    此处保持接口一致。
    """
    import importlib.metadata

    try:
        pkg_name = requirement.split(">=")[0].split("==")[0].strip()
        importlib.metadata.version(pkg_name)
        _dbg("package_available", f"{pkg_name!r} → True")
        return True
    except Exception:
        _dbg("package_available", f"{requirement!r} → False")
        return False


# ---------------------------------------------------------------------------
# b10f279 核心变更摘要（供审计工具读取）
# ---------------------------------------------------------------------------

#: 上游迁移信息，供审计工具查询
UPSTREAM_COMMIT = "b10f279"
UPSTREAM_REPO = "rapidsai/cugraph-gnn"
UPSTREAM_PR = 271
UPSTREAM_TITLE = "Remove cugraph Python library as a dependency"
MIGRATION_TARGET = "walpurgis/core/cugraph_dep_removal.py"

#: b10f279 依赖层级变更记录
DEPENDENCY_CHANGES: Dict[str, DependencyTier] = {
    "cugraph": DependencyTier.TEST_ONLY,       # 运行时 → 测试可选
    "pylibcugraph": DependencyTier.RUNTIME,    # 新增为运行时依赖
    "pylibwholegraph": DependencyTier.RUNTIME, # 保持运行时依赖
}
