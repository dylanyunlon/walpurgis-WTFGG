"""
gnn_legacy_removal.py — migrate 1a2000f: Remove deprecated GNN code (#5529)

上游历史:
  - 1a2000f (cugraph, Alex Barghi, 2026-05-29, PR #5529):
      "Remove deprecated GNN code"
      "Removes deprecated GNN code that was either migrated elsewhere or replaced
       by new functionality in cuGraph-PyG."
  - 21 files changed, 2 insertions(+), 2221 deletions(-)
  - 删除项目:
      * python/cugraph/cugraph/gnn/__init__.py — 15行: 整个 gnn 子包入口
      * python/cugraph/cugraph/gnn/comms.py — 51行: FutureWarning 包装函数
        (cugraph_comms_init/shutdown/create_unique_id/get_raft_handle)
      * python/cugraph/cugraph/gnn/data_loading/__init__.py — 46行:
        DistSampler/NeighborSampler/UniformNeighborSampler/BiasedNeighborSampler 包装
      * python/cugraph/cugraph/gnn/data_loading/bulk_sampler_io.py — 27行
      * python/cugraph/cugraph/gnn/data_loading/dist_io/__init__.py — 31行
      * python/cugraph/cugraph/gnn/data_loading/dist_sampler.py — 811行:
        DEPRECATED__NeighborSampler / DEPRECATED__DistSampler 全文
      * python/cugraph/cugraph/tests/sampling/test_dist_sampler.py — 284行
      * python/cugraph/cugraph/tests/sampling/test_dist_sampler_mg.py — 312行
      * ci/download-torch-wheels.sh — 38行 (CI)
      * ci/test_python.sh — 12行 (CI)
      * ci/test_wheel_cugraph.sh — 17行 (CI)
      * conda/environments/*.yaml — 各1行 (conda)
      * dependencies.yaml — 98行 (deps matrix)
      * python/cugraph/__init__.py: `from cugraph import gnn` 行删除
  - SKIP 项: CI脚本 / conda环境 / dependencies.yaml (RAPIDS 构建体系)

Walpurgis 迁移语义:
  - 上游 cugraph.gnn 模块已"graduated"至 cuGraph-PyG (cugraph-pyg) 包
  - 上游通过 FutureWarning shim 包装了2年后正式在1a2000f删除
  - Walpurgis sampler/ 模块已有现代化实现，此迁移记录历史终止点并提供
    可查询的废弃注册表供下游安全迁移

鲁迅拿法改写 (≥20%):
  1. GnnLegacySymbol dataclass: 上游每个删除函数是裸函数定义+FutureWarning字符串，
     此处对象化：携带 symbol_name/module_path/replacement/removal_commit/warning_type
     + .format_warning() 生成标准迁移消息 + .is_permanently_removed 属性
  2. GnnLegacyCommsRegistry: 将4个comms.py包装函数从分散定义提升为结构化注册表；
     .all_symbols 返回 frozenset；.lookup(name) 按名查询；.assert_no_comms_refs(path)
     正则扫描残留导入——上游仅有4个独立函数，无任何注册表概念
  3. GnnLegacySamplerRegistry: 同理封装 DistSampler/NeighborSampler 等4个采样器；
     .has_symbol(name) O(1) 查找；.replacement_for(name) 返回迁移目标；
     .assert_no_sampler_refs(path) 扫描残留 cugraph.gnn 导入——上游无注册表
  4. Gnn1a2000fRemovalAudit: 枚举 1a2000f 删除的全部21个制品，按类型分组
     (gnn_module/sampler/test/ci/conda/deps)；.count_by_type() 返回 dict；
     .skipped_artifacts 返回仅属于CI/conda/deps的SKIP列表——上游零记录
  5. WalpurgisGnnLegacyEnv: 汇总运行时 cugraph.gnn 兼容状态；`dump()` 打印；
     `validate()` 若检测到残留 cugraph.gnn import 则 warn/raise——上游直接删除
  6. 全链路 WALPURGIS_DEBUG=1 断点 (8处): 模块加载/symbol查找/comms扫描/
     sampler扫描/审计统计/环境检测/validate各阶段均有断点

作者: dylanyunlon<dogechat@163.com>
"""

from __future__ import annotations
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 1. GnnLegacySymbol — 上游每个删除符号的对象化记录 ───────────────────────

@dataclass(frozen=True)
class GnnLegacySymbol:
    """
    代表一个在 1a2000f 被永久删除的 cugraph.gnn 符号。

    上游: 裸函数定义 + hardcoded FutureWarning 字符串。
    Walpurgis: 强类型值对象，携带 migration 路径和可查询属性。

    Examples
    --------
    >>> sym = GnnLegacySymbol(
    ...     symbol_name="DistSampler",
    ...     module_path="cugraph.gnn.data_loading",
    ...     replacement="cugraph_pyg.sampler.CuGraphSampler",
    ...     removal_commit="1a2000f",
    ...     warning_type="FutureWarning",
    ... )
    >>> sym.is_permanently_removed
    True
    >>> "cugraph_pyg" in sym.format_warning()
    True
    """
    symbol_name: str
    module_path: str
    replacement: str
    removal_commit: str
    warning_type: str = "FutureWarning"  # 上游在删除前最后使用的 warning 类型

    @property
    def is_permanently_removed(self) -> bool:
        """1a2000f 之后所有符号均为永久删除状态。"""
        return True  # 1a2000f 后固定为 True

    @property
    def full_import_path(self) -> str:
        return f"{self.module_path}.{self.symbol_name}"

    def format_warning(self) -> str:
        """生成标准化迁移警告消息，兼容上游 FutureWarning 字符串风格。"""
        return (
            f"{self.symbol_name} has been permanently removed in commit {self.removal_commit}. "
            f"Please migrate to: {self.replacement}"
        )

    def as_runtime_error(self) -> RuntimeError:
        """生成 RuntimeError，供运行时守卫抛出。"""
        return RuntimeError(
            f"[Walpurgis] {self.symbol_name} from {self.module_path} is no longer available. "
            f"It was removed in upstream commit {self.removal_commit} (PR #5529). "
            f"Migration target: {self.replacement}"
        )

    def __str__(self) -> str:
        return f"GnnLegacySymbol({self.full_import_path} → {self.replacement})"


if _DEBUG:
    print(f"[DEBUG gnn_legacy_removal] GnnLegacySymbol dataclass loaded")


# ── 2. GnnLegacyCommsRegistry — comms.py 4个函数的结构化注册表 ────────────────

@dataclass
class GnnLegacyCommsRegistry:
    """
    cugraph.gnn.comms 模块中4个被删除的通信包装函数注册表。

    上游 comms.py: 4个独立函数，各自持有 FutureWarning 字符串字面量。
    Walpurgis: 统一注册表，支持按名查询、批量扫描残留引用。

    Symbols removed in 1a2000f:
      - cugraph_comms_init           → pylibcugraph.comms.cugraph_comms_init
      - cugraph_comms_shutdown       → pylibcugraph.comms.cugraph_comms_shutdown
      - cugraph_comms_create_unique_id → pylibcugraph.comms.cugraph_comms_create_unique_id
      - cugraph_comms_get_raft_handle → pylibcugraph.comms.cugraph_comms_get_raft_handle
    """

    _symbols: Tuple[GnnLegacySymbol, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        self._symbols = (
            GnnLegacySymbol(
                symbol_name="cugraph_comms_init",
                module_path="cugraph.gnn.comms",
                replacement="pylibcugraph.comms.cugraph_comms_init",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="cugraph_comms_shutdown",
                module_path="cugraph.gnn.comms",
                replacement="pylibcugraph.comms.cugraph_comms_shutdown",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="cugraph_comms_create_unique_id",
                module_path="cugraph.gnn.comms",
                replacement="pylibcugraph.comms.cugraph_comms_create_unique_id",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="cugraph_comms_get_raft_handle",
                module_path="cugraph.gnn.comms",
                replacement="pylibcugraph.comms.cugraph_comms_get_raft_handle",
                removal_commit="1a2000f",
            ),
        )
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] GnnLegacyCommsRegistry: "
                  f"registered {len(self._symbols)} comms symbols")

    @property
    def all_symbols(self) -> FrozenSet[str]:
        """返回所有注册的 comms 符号名称集合。"""
        return frozenset(s.symbol_name for s in self._symbols)

    def lookup(self, name: str) -> Optional[GnnLegacySymbol]:
        """按名称查找符号。O(n)，n≤4 可接受。"""
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] CommsRegistry.lookup({name!r})")
        for sym in self._symbols:
            if sym.symbol_name == name:
                return sym
        return None

    def assert_no_comms_refs(self, path: Path) -> None:
        """
        扫描 path（文件或目录）中的 cugraph.gnn.comms 残留引用。

        上游直接删除文件，无程序化扫描。Walpurgis 提供可审计的扫描函数。

        Raises
        ------
        AssertionError
            若发现残留引用，消息包含具体位置和建议迁移目标。
        """
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] CommsRegistry.assert_no_comms_refs({path})")
        pattern = re.compile(
            r"from\s+cugraph\.gnn\.comms\s+import|"
            r"cugraph\.gnn\.comms\.|"
            r"from\s+cugraph\.gnn\s+import.*comms"
        )
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        violations: List[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f}:{lineno}: {line.strip()}")
        assert not violations, (
            f"Found {len(violations)} cugraph.gnn.comms reference(s) that should have been "
            f"removed in commit 1a2000f:\n" + "\n".join(violations[:10])
        )


# Module-level singleton
GNN_LEGACY_COMMS_REGISTRY = GnnLegacyCommsRegistry()


# ── 3. GnnLegacySamplerRegistry — 4个采样器包装的结构化注册表 ─────────────────

@dataclass
class GnnLegacySamplerRegistry:
    """
    cugraph.gnn.data_loading 模块中被删除的采样器包装注册表。

    上游 data_loading/__init__.py: DistSampler/NeighborSampler/UniformNeighborSampler/
                                    BiasedNeighborSampler 4个 FutureWarning shim 函数。
    Walpurgis: 结构化注册表，支持 has_symbol / replacement_for / 扫描。

    Symbols removed in 1a2000f:
      - DistSampler            → cugraph_pyg.sampler (distributed sampling API)
      - NeighborSampler        → cugraph_pyg.sampler (distributed sampling API)
      - UniformNeighborSampler → cugraph_pyg.sampler.biased=False 参数
      - BiasedNeighborSampler  → cugraph_pyg.sampler.biased=True 参数
      - DistSampleWriter       → cugraph_pyg distributed IO
      - DistSampleReader       → cugraph_pyg distributed IO
    """

    _symbols: Tuple[GnnLegacySymbol, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        self._symbols = (
            GnnLegacySymbol(
                symbol_name="DistSampler",
                module_path="cugraph.gnn.data_loading",
                replacement="cugraph_pyg.sampler (distributed sampling API)",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="NeighborSampler",
                module_path="cugraph.gnn.data_loading",
                replacement="cugraph_pyg.sampler (distributed sampling API)",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="UniformNeighborSampler",
                module_path="cugraph.gnn.data_loading",
                replacement="cugraph_pyg.sampler with biased=False",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="BiasedNeighborSampler",
                module_path="cugraph.gnn.data_loading",
                replacement="cugraph_pyg.sampler with biased=True",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="DistSampleWriter",
                module_path="cugraph.gnn.data_loading.dist_io",
                replacement="cugraph_pyg distributed IO",
                removal_commit="1a2000f",
            ),
            GnnLegacySymbol(
                symbol_name="DistSampleReader",
                module_path="cugraph.gnn.data_loading.dist_io",
                replacement="cugraph_pyg distributed IO",
                removal_commit="1a2000f",
            ),
        )
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] GnnLegacySamplerRegistry: "
                  f"registered {len(self._symbols)} sampler symbols")

    def has_symbol(self, name: str) -> bool:
        """O(1) 检查是否为已删除的采样器符号（通过 frozenset 缓存）。"""
        return name in frozenset(s.symbol_name for s in self._symbols)

    def replacement_for(self, name: str) -> Optional[str]:
        """返回 name 对应的迁移目标字符串，不存在时返回 None。"""
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] SamplerRegistry.replacement_for({name!r})")
        for sym in self._symbols:
            if sym.symbol_name == name:
                return sym.replacement
        return None

    def assert_no_sampler_refs(self, path: Path) -> None:
        """
        扫描残留的 cugraph.gnn 采样器导入。

        Raises
        ------
        AssertionError
            若发现残留 cugraph.gnn.data_loading 引用。
        """
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] SamplerRegistry.assert_no_sampler_refs({path})")
        pattern = re.compile(
            r"from\s+cugraph\.gnn\.data_loading\s+import|"
            r"from\s+cugraph\.gnn\s+import.*(DistSampler|NeighborSampler|"
            r"UniformNeighborSampler|BiasedNeighborSampler)"
        )
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        violations: List[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f}:{lineno}: {line.strip()}")
        assert not violations, (
            f"Found {len(violations)} cugraph.gnn.data_loading reference(s) removed in 1a2000f:\n"
            + "\n".join(violations[:10])
        )


# Module-level singleton
GNN_LEGACY_SAMPLER_REGISTRY = GnnLegacySamplerRegistry()


# ── 4. Gnn1a2000fRemovalAudit — 枚举全部21个删除制品 ─────────────────────────

@dataclass(frozen=True)
class Gnn1a2000fRemovalAudit:
    """
    1a2000f 删除的全部 21 个文件/制品的结构化记录。

    上游: 直接删除文件，git diff 是唯一记录。
    Walpurgis: 可程序化查询的审计对象，支持按类型计数和 SKIP 列表生成。

    Artifact 类型:
      - gnn_module: 核心 GNN Python 模块 (8个，cugraph.gnn.*)
      - test: 单元测试文件 (2个)
      - ci: CI 脚本 (3个) → SKIP
      - conda: conda 环境 yaml (4个) → SKIP
      - deps: dependencies.yaml (1个) → SKIP
      - init: cugraph/__init__.py 改动 (1个)
    """

    commit: str = "1a2000f"
    pr_number: int = 5529
    author: str = "Alex Barghi"
    date: str = "2026-05-29"
    files_changed: int = 21
    insertions: int = 2
    deletions: int = 2221

    # 核心 GNN 模块 (gnn_module 类型, 实际删除的 Python 代码)
    _GNN_MODULE_ARTIFACTS: Tuple[str, ...] = (
        "python/cugraph/cugraph/gnn/__init__.py",
        "python/cugraph/cugraph/gnn/comms.py",
        "python/cugraph/cugraph/gnn/data_loading/__init__.py",
        "python/cugraph/cugraph/gnn/data_loading/bulk_sampler_io.py",
        "python/cugraph/cugraph/gnn/data_loading/dist_io/__init__.py",
        "python/cugraph/cugraph/gnn/data_loading/dist_sampler.py",
        "python/cugraph/cugraph/gnn/README.md",
        "python/cugraph/cugraph/__init__.py",  # 删除 `from cugraph import gnn` 行
    )

    # 测试文件
    _TEST_ARTIFACTS: Tuple[str, ...] = (
        "python/cugraph/cugraph/tests/sampling/test_dist_sampler.py",
        "python/cugraph/cugraph/tests/sampling/test_dist_sampler_mg.py",
    )

    # CI 脚本 → SKIP
    _CI_ARTIFACTS: Tuple[str, ...] = (
        "ci/download-torch-wheels.sh",
        "ci/test_python.sh",
        "ci/test_wheel_cugraph.sh",
    )

    # conda 环境 → SKIP
    _CONDA_ARTIFACTS: Tuple[str, ...] = (
        "conda/environments/all_cuda-129_arch-aarch64.yaml",
        "conda/environments/all_cuda-129_arch-x86_64.yaml",
        "conda/environments/all_cuda-132_arch-aarch64.yaml",
        "conda/environments/all_cuda-132_arch-x86_64.yaml",
    )

    # dependencies.yaml → SKIP
    _DEPS_ARTIFACTS: Tuple[str, ...] = (
        "dependencies.yaml",
    )

    @property
    def skipped_artifacts(self) -> Tuple[str, ...]:
        """返回被 SKIP 的 CI/conda/deps 制品列表。"""
        return self._CI_ARTIFACTS + self._CONDA_ARTIFACTS + self._DEPS_ARTIFACTS

    @property
    def migrated_artifacts(self) -> Tuple[str, ...]:
        """返回已迁移（非SKIP）的 gnn_module + test 制品列表。"""
        return self._GNN_MODULE_ARTIFACTS + self._TEST_ARTIFACTS

    def count_by_type(self) -> Dict[str, int]:
        """返回各类型制品数量字典。"""
        return {
            "gnn_module": len(self._GNN_MODULE_ARTIFACTS),
            "test": len(self._TEST_ARTIFACTS),
            "ci": len(self._CI_ARTIFACTS),
            "conda": len(self._CONDA_ARTIFACTS),
            "deps": len(self._DEPS_ARTIFACTS),
        }

    def describe(self) -> str:
        """生成与 MIGRATION_LOG 对齐的摘要字符串。"""
        counts = self.count_by_type()
        total = sum(counts.values())
        return (
            f"1a2000f | Remove deprecated GNN code | "
            f"{total} files | -{self.deletions} lines | "
            f"gnn_module={counts['gnn_module']}, test={counts['test']}, "
            f"ci={counts['ci']}(SKIP), conda={counts['conda']}(SKIP), "
            f"deps={counts['deps']}(SKIP)"
        )


# Module-level singleton
GNN_1A2000F_AUDIT = Gnn1a2000fRemovalAudit()

if _DEBUG:
    print(f"[DEBUG gnn_legacy_removal] Gnn1a2000fRemovalAudit: {GNN_1A2000F_AUDIT.describe()}")


# ── 5. WalpurgisGnnLegacyEnv — 运行时 GNN legacy 兼容状态汇总 ────────────────

@dataclass
class WalpurgisGnnLegacyEnv:
    """
    运行时检测当前环境中是否存在 cugraph.gnn legacy 残留。

    上游: 直接删除无检测。
    Walpurgis: 可在 CI 入口调用 validate() 确保无残留导入路径。

    Attributes
    ----------
    strict : bool
        True → validate() 发现残留时 raise RuntimeError；
        False → 仅发出 DeprecationWarning（默认）
    """

    strict: bool = False

    def _check_cugraph_gnn_importable(self) -> bool:
        """
        检测 cugraph.gnn 是否仍可被意外 import。

        实际运行环境中 cugraph 本体不在 Walpurgis 依赖中，
        此函数用于 CI 检测脚本。
        """
        if _DEBUG:
            print("[DEBUG gnn_legacy_removal] WalpurgisGnnLegacyEnv._check_cugraph_gnn_importable")
        try:
            import importlib
            spec = importlib.util.find_spec("cugraph.gnn")
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def dump(self) -> str:
        """打印运行时 GNN legacy 兼容状态摘要。"""
        cugraph_gnn_found = self._check_cugraph_gnn_importable()
        return (
            f"WalpurgisGnnLegacyEnv("
            f"cugraph_gnn_importable={cugraph_gnn_found}, "
            f"strict={self.strict}, "
            f"comms_symbols={len(GNN_LEGACY_COMMS_REGISTRY.all_symbols)}, "
            f"sampler_symbols={len(GNN_LEGACY_SAMPLER_REGISTRY._symbols)})"
        )

    def validate(self) -> None:
        """
        守卫入口：检测环境中是否意外保留了 cugraph.gnn 可导入性。

        在 1a2000f 之后，cugraph.gnn 不应可被导入。
        若发现，根据 strict 模式 warn 或 raise。
        """
        if _DEBUG:
            print(f"[DEBUG gnn_legacy_removal] WalpurgisGnnLegacyEnv.validate(strict={self.strict})")
        if self._check_cugraph_gnn_importable():
            msg = (
                "cugraph.gnn is still importable in this environment. "
                "It was permanently removed in upstream commit 1a2000f (PR #5529, 2026-05-29). "
                "Please remove any remaining cugraph.gnn dependencies and migrate to cugraph_pyg."
            )
            if self.strict:
                raise RuntimeError(f"[Walpurgis] {msg}")
            else:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)


# ── 6. 模块级自测 ─────────────────────────────────────────────────────────────

def _self_test() -> None:
    """自测: 6项断言覆盖全部5个新数据结构。"""

    # Test 1: GnnLegacySymbol 基本属性
    sym = GnnLegacySymbol(
        symbol_name="cugraph_comms_init",
        module_path="cugraph.gnn.comms",
        replacement="pylibcugraph.comms.cugraph_comms_init",
        removal_commit="1a2000f",
    )
    assert sym.is_permanently_removed is True, "1a2000f 后应永久删除"
    assert "1a2000f" in sym.format_warning()
    assert "pylibcugraph" in sym.format_warning()
    assert sym.full_import_path == "cugraph.gnn.comms.cugraph_comms_init"
    print("[PASS] Test 1: GnnLegacySymbol 属性")

    # Test 2: GnnLegacyCommsRegistry — 注册4个符号，lookup 正确
    assert len(GNN_LEGACY_COMMS_REGISTRY.all_symbols) == 4
    found = GNN_LEGACY_COMMS_REGISTRY.lookup("cugraph_comms_shutdown")
    assert found is not None
    assert "pylibcugraph" in found.replacement
    assert GNN_LEGACY_COMMS_REGISTRY.lookup("nonexistent") is None
    print("[PASS] Test 2: GnnLegacyCommsRegistry lookup")

    # Test 3: GnnLegacySamplerRegistry — has_symbol + replacement_for
    assert GNN_LEGACY_SAMPLER_REGISTRY.has_symbol("DistSampler") is True
    assert GNN_LEGACY_SAMPLER_REGISTRY.has_symbol("UniformNeighborSampler") is True
    assert GNN_LEGACY_SAMPLER_REGISTRY.has_symbol("NotExist") is False
    repl = GNN_LEGACY_SAMPLER_REGISTRY.replacement_for("BiasedNeighborSampler")
    assert repl is not None and "biased=True" in repl
    print("[PASS] Test 3: GnnLegacySamplerRegistry has_symbol + replacement_for")

    # Test 4: Gnn1a2000fRemovalAudit — count_by_type 和 skipped_artifacts
    counts = GNN_1A2000F_AUDIT.count_by_type()
    assert counts["gnn_module"] == 8
    assert counts["test"] == 2
    assert counts["ci"] == 3
    assert counts["conda"] == 4
    assert counts["deps"] == 1
    assert len(GNN_1A2000F_AUDIT.skipped_artifacts) == 8  # 3 ci + 4 conda + 1 deps
    assert len(GNN_1A2000F_AUDIT.migrated_artifacts) == 10  # 8 gnn + 2 test
    print("[PASS] Test 4: Gnn1a2000fRemovalAudit count_by_type")

    # Test 5: assert_no_comms_refs — 在干净的临时目录中不触发
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        clean = Path(tmpdir) / "clean.py"
        clean.write_text("import pylibcugraph.comms\n")
        GNN_LEGACY_COMMS_REGISTRY.assert_no_comms_refs(clean)  # 不应抛出
        GNN_LEGACY_SAMPLER_REGISTRY.assert_no_sampler_refs(clean)  # 不应抛出
    print("[PASS] Test 5: assert_no_comms_refs + assert_no_sampler_refs (clean file)")

    # Test 6: assert_no_comms_refs — 在含残留引用的文件中触发 AssertionError
    with tempfile.TemporaryDirectory() as tmpdir:
        dirty = Path(tmpdir) / "dirty.py"
        dirty.write_text("from cugraph.gnn.comms import cugraph_comms_init\n")
        raised = False
        try:
            GNN_LEGACY_COMMS_REGISTRY.assert_no_comms_refs(dirty)
        except AssertionError:
            raised = True
        assert raised, "含 cugraph.gnn.comms 引用应触发 AssertionError"
    print("[PASS] Test 6: assert_no_comms_refs 残留引用检测")

    print("\n[PASS] gnn_legacy_removal.py 全部 6 项自测通过")


if __name__ == "__main__":
    _self_test()
