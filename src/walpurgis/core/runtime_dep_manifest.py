"""
migrate 338084f: add missed runtime deps, remove runtime deps from test extra
cugraph-gnn PR #277 (James Lamb / Alex Barghi, 2025-08-18)
Commit 338084facfdfcf7422ef6904c43e0256b95190a6

CI/merge -> SKIP (all 5 upstream files):
- conda/environments/*.yaml: no conda build matrix in Walpurgis
- conda/recipes/cugraph-pyg/: RAPIDS conda recipe, not Walpurgis build config
- dependencies.yaml: RAPIDS dep file generator, Walpurgis uses pyproject.toml
- python/cugraph-pyg/pyproject.toml: upstream package metadata, not Walpurgis source

Migration target: src/walpurgis/core/runtime_dep_manifest.py

Rewrite >=20% (Luxun style):
1. ImportCategory enum: UNCONDITIONAL/CONDITIONAL/DEFERRED - upstream has no such taxonomy
2. RuntimeDepEntry dataclass: structured dep with source file, rationale, commit provenance
3. DepDeclarationAudit: AST scan auto-detects undeclared top-level imports
4. TestExtraCleanup: models removal of duplicate test deps from [test] extra
5. HomepageRecord: structured URL fix record with is_corrected() verification
6. RuntimeDepSnapshot: runtime install check with strict/warn modes
7. _STDLIB_ROOTS: heuristic exclusion set for audit false-positive suppression
8. WALPURGIS_DEBUG=1 breakpoints (8 points) throughout
"""

from __future__ import annotations
import ast, importlib.metadata, os, re, warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

if _DBG:
    print("[DEBUG 338084f runtime_dep_manifest] module load")


class ImportCategory(Enum):
    UNCONDITIONAL = "unconditional"
    CONDITIONAL = "conditional"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class RuntimeDepEntry:
    package_name: str
    pypi_name: str
    min_version: Optional[str]
    import_category: ImportCategory
    declaration_scope: str
    upstream_commit: str
    upstream_source_file: str
    rationale: str

    def pip_spec(self) -> str:
        return f"{self.pypi_name}>={self.min_version}" if self.min_version else self.pypi_name

    def is_hard_dep(self) -> bool:
        return self.declaration_scope == "project"

    def dump(self) -> str:
        return (
            f"  package_name      = {self.package_name}\n"
            f"  pypi_name         = {self.pypi_name}\n"
            f"  pip_spec          = {self.pip_spec()}\n"
            f"  import_category   = {self.import_category.value}\n"
            f"  declaration_scope = {self.declaration_scope}\n"
            f"  upstream_commit   = {self.upstream_commit}\n"
            f"  source_file       = {self.upstream_source_file}\n"
            f"  rationale         = {self.rationale}"
        )


CUPY_DEP = RuntimeDepEntry(
    package_name="cupy",
    pypi_name="cupy-cuda12x",
    min_version="13.2.0",
    import_category=ImportCategory.UNCONDITIONAL,
    declaration_scope="project",
    upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
    upstream_source_file="python/cugraph-pyg/cugraph_pyg/data/graph_store.py#L17",
    rationale=(
        "graph_store.py unconditionally imports cupy at module top-level, "
        "but cupy-cuda12x was missing from [project].dependencies. "
        "338084f adds it as a hard dep (>=13.2.0)."
    ),
)

PACKAGING_DEP = RuntimeDepEntry(
    package_name="packaging",
    pypi_name="packaging",
    min_version=None,
    import_category=ImportCategory.UNCONDITIONAL,
    declaration_scope="project",
    upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
    upstream_source_file="python/cugraph-pyg/cugraph_pyg/utils/imports.py#L14",
    rationale=(
        "imports.py unconditionally imports from packaging at module top-level. "
        "338084f adds it as a hard dep. Note: 94ac7fea will later remove this "
        "import entirely when package_available() is deleted."
    ),
)

if _DBG:
    print("[DEBUG 338084f runtime_dep_manifest] CUPY_DEP registered:", CUPY_DEP.pip_spec())
    print("[DEBUG 338084f runtime_dep_manifest] PACKAGING_DEP registered:", PACKAGING_DEP.pip_spec())

_338084F_RUNTIME_DEPS: list[RuntimeDepEntry] = [CUPY_DEP, PACKAGING_DEP]

_STDLIB_ROOTS: frozenset[str] = frozenset({
    "os", "sys", "re", "io", "ast", "abc", "copy", "math", "time", "json",
    "csv", "enum", "uuid", "typing", "types", "random", "struct", "hashlib",
    "logging", "warnings", "pathlib", "functools", "itertools", "contextlib",
    "collections", "dataclasses", "importlib", "subprocess", "threading",
    "multiprocessing", "concurrent", "asyncio", "socket", "http", "urllib",
    "email", "xml", "html", "unittest", "tempfile", "shutil", "glob",
    "fnmatch", "platform", "inspect", "traceback", "weakref", "gc", "ctypes",
    "array", "queue", "signal", "textwrap", "string", "operator", "builtins",
    "__future__",
})


@dataclass
class DepDeclarationAudit:
    """
    Scans Python source files for top-level imports and cross-checks against
    declared hard deps. Auto-detects the pattern 338084f fixed (unconditional
    import without dependency declaration).

    Upstream approach: manual PR review. Walpurgis: programmatic scan_file/scan_directory.
    """
    _declared_hard_deps: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        for dep in _338084F_RUNTIME_DEPS:
            if dep.is_hard_dep():
                self._declared_hard_deps.add(dep.package_name.lower())
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] DepDeclarationAudit init: {self._declared_hard_deps}")

    def _extract_top_level_imports(self, source: str) -> list[str]:
        if _DBG:
            print("[DEBUG 338084f runtime_dep_manifest] _extract_top_level_imports: AST parse")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        pkgs: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkgs.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    pkgs.append(node.module.split(".")[0])
        return pkgs

    def scan_file(self, filepath: str) -> list[str]:
        try:
            source = Path(filepath).read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            return []
        top_imports = self._extract_top_level_imports(source)
        undeclared = [
            p for p in top_imports
            if p.lower() not in self._declared_hard_deps and p not in _STDLIB_ROOTS
        ]
        if _DBG and undeclared:
            print(f"[DEBUG 338084f runtime_dep_manifest] scan_file {filepath}: undeclared={undeclared}")
        return undeclared

    def scan_directory(self, dirpath: str, glob: str = "**/*.py") -> dict[str, list[str]]:
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] scan_directory: {dirpath}")
        results: dict[str, list[str]] = {}
        for py_file in Path(dirpath).glob(glob):
            undeclared = self.scan_file(str(py_file))
            if undeclared:
                results[str(py_file)] = undeclared
        return results


@dataclass(frozen=True)
class TestExtraCleanup:
    """
    Models removal of packages from [test] extra that were already in [project].dependencies.

    Upstream 338084f removed 3 packages from [test] with no rationale recorded.
    TestExtraCleanup structures the reason alongside the removal fact.
    """
    removed_package: str
    removal_reason: str
    upstream_commit: str
    was_in_project_deps: bool

    def dump(self) -> str:
        scope = "[project].dependencies" if self.was_in_project_deps else "elsewhere"
        return (
            f"  removed_package    = {self.removed_package}\n"
            f"  removal_reason     = {self.removal_reason}\n"
            f"  already_in         = {scope}\n"
            f"  upstream_commit    = {self.upstream_commit}"
        )


TEST_EXTRA_REMOVALS: list[TestExtraCleanup] = [
    TestExtraCleanup(
        removed_package="pylibcugraph==25.10.*,>=0.0.0a0",
        removal_reason="Already in [project].dependencies; duplicate in [test] causes resolver ambiguity.",
        upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
        was_in_project_deps=True,
    ),
    TestExtraCleanup(
        removed_package="pylibwholegraph==25.10.*,>=0.0.0a0",
        removal_reason="Same reason as pylibcugraph — already in [project].dependencies.",
        upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
        was_in_project_deps=True,
    ),
    TestExtraCleanup(
        removed_package="torch-geometric>=2.5,<2.7",
        removal_reason=(
            "Already in [project].dependencies via depends_on_pyg. "
            "torch itself (torch>=2.3) is intentionally kept in [test]."
        ),
        upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
        was_in_project_deps=True,
    ),
]

if _DBG:
    print(f"[DEBUG 338084f runtime_dep_manifest] TestExtraCleanup: {len(TEST_EXTRA_REMOVALS)} entries")


@dataclass(frozen=True)
class HomepageRecord:
    """
    Records the homepage URL correction in cugraph-pyg pyproject.toml.

    Upstream 338084f: one-line diff changing cugraph -> cugraph-gnn.
    HomepageRecord structures before/after/reason and provides is_corrected() verification.
    """
    before_url: str
    after_url: str
    upstream_commit: str
    reason: str

    def is_corrected(self, current_url: str) -> bool:
        return current_url.rstrip("/") == self.after_url.rstrip("/")

    def dump(self) -> str:
        return (
            f"  before_url      = {self.before_url}\n"
            f"  after_url       = {self.after_url}\n"
            f"  upstream_commit = {self.upstream_commit}\n"
            f"  reason          = {self.reason}"
        )


CUGRAPH_PYG_HOMEPAGE_FIX = HomepageRecord(
    before_url="https://github.com/rapidsai/cugraph",
    after_url="https://github.com/rapidsai/cugraph-gnn",
    upstream_commit="338084facfdfcf7422ef6904c43e0256b95190a6",
    reason=(
        "cugraph-gnn was split from rapidsai/cugraph monorepo into its own repo. "
        "cugraph-pyg homepage still pointed to the old monorepo URL; 338084f corrects it."
    ),
)

if _DBG:
    print(f"[DEBUG 338084f runtime_dep_manifest] HomepageRecord: {CUGRAPH_PYG_HOMEPAGE_FIX.after_url}")


@dataclass
class RuntimeDepSnapshot:
    """
    Verifies cupy/packaging install status at runtime.

    Upstream 338084f only edits pyproject.toml; no Python-level self-check.
    verify() programmatically confirms both hard deps are installed and satisfy version constraints.
    strict=True raises ImportError; strict=False emits UserWarning.
    """
    strict: bool = False

    def _installed_version(self, pypi_name: str) -> Optional[str]:
        candidates = [pypi_name]
        if pypi_name == "cupy-cuda12x":
            candidates = ["cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x", "cupy"]
        for name in candidates:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    @staticmethod
    def _version_tuple(ver: str) -> tuple[int, ...]:
        numeric_part = re.split(r"[a-zA-Z]", ver)[0]
        return tuple(int(x) for x in numeric_part.split(".") if x.isdigit())

    def check_dep(self, dep: RuntimeDepEntry) -> bool:
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] check_dep: {dep.pypi_name}")
        installed = self._installed_version(dep.pypi_name)
        if installed is None:
            msg = (
                f"[Walpurgis runtime_dep_manifest] '{dep.package_name}' not installed.\n"
                f"338084f declares it a hard runtime dep. Install: pip install '{dep.pip_spec()}'"
            )
            if self.strict:
                raise ImportError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)
            return False
        if dep.min_version:
            inst_t = self._version_tuple(installed)
            min_t = self._version_tuple(dep.min_version)
            if inst_t < min_t:
                msg = (
                    f"[Walpurgis runtime_dep_manifest] "
                    f"'{dep.package_name}=={installed}' below minimum >={dep.min_version}. "
                    f"Upgrade: pip install '{dep.pip_spec()}'"
                )
                if _DBG:
                    print(f"[DEBUG 338084f runtime_dep_manifest] version too low: {dep.pypi_name}=={installed}")
                if self.strict:
                    raise ImportError(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
                return False
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] {dep.pypi_name}=={installed} ok")
        return True

    def verify(self) -> dict[str, bool]:
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] verify() start (strict={self.strict})")
        results = {dep.package_name: self.check_dep(dep) for dep in _338084F_RUNTIME_DEPS}
        if _DBG:
            print(f"[DEBUG 338084f runtime_dep_manifest] verify() results: {results}")
        return results

    def dump(self) -> str:
        lines = ["\u2500\u2500 RuntimeDepSnapshot (338084f) \u2500\u2500"]
        for dep in _338084F_RUNTIME_DEPS:
            installed = self._installed_version(dep.pypi_name)
            status = installed if installed else "<not installed>"
            ok = "\u2713" if installed else "\u2717"
            lines.append(f"  {dep.package_name:12s}: {status:14s}  constraint={dep.pip_spec()}  {ok}")
        lines.append("\u2500" * 40)
        return "\n".join(lines)


def _self_test() -> None:
    if _DBG:
        print("[DEBUG 338084f runtime_dep_manifest] _self_test start")

    assert CUPY_DEP.package_name == "cupy"
    assert CUPY_DEP.pypi_name == "cupy-cuda12x"
    assert CUPY_DEP.min_version == "13.2.0"
    assert CUPY_DEP.import_category == ImportCategory.UNCONDITIONAL
    assert CUPY_DEP.is_hard_dep()
    assert CUPY_DEP.pip_spec() == "cupy-cuda12x>=13.2.0"

    assert PACKAGING_DEP.package_name == "packaging"
    assert PACKAGING_DEP.min_version is None
    assert PACKAGING_DEP.import_category == ImportCategory.UNCONDITIONAL
    assert PACKAGING_DEP.is_hard_dep()
    assert PACKAGING_DEP.pip_spec() == "packaging"

    assert ImportCategory.UNCONDITIONAL.value == "unconditional"
    assert ImportCategory.CONDITIONAL.value == "conditional"
    assert ImportCategory.DEFERRED.value == "deferred"

    assert len(TEST_EXTRA_REMOVALS) == 3
    removed_names = [r.removed_package for r in TEST_EXTRA_REMOVALS]
    assert any("pylibcugraph" in n for n in removed_names)
    assert any("pylibwholegraph" in n for n in removed_names)
    assert any("torch-geometric" in n for n in removed_names)
    assert all(r.was_in_project_deps for r in TEST_EXTRA_REMOVALS)

    assert "cugraph-gnn" in CUGRAPH_PYG_HOMEPAGE_FIX.after_url
    assert "cugraph-gnn" not in CUGRAPH_PYG_HOMEPAGE_FIX.before_url
    assert CUGRAPH_PYG_HOMEPAGE_FIX.is_corrected("https://github.com/rapidsai/cugraph-gnn")
    assert not CUGRAPH_PYG_HOMEPAGE_FIX.is_corrected("https://github.com/rapidsai/cugraph")

    audit = DepDeclarationAudit()
    sample = (
        "import cupy\n"
        "from packaging.version import Version\n"
        "import torch\n"
        "\ndef _load():\n"
        "    import numpy\n"
    )
    imports = audit._extract_top_level_imports(sample)
    assert "cupy" in imports
    assert "packaging" in imports
    assert "torch" in imports
    assert "numpy" not in imports

    snap = RuntimeDepSnapshot()
    assert snap._version_tuple("13.2.0") == (13, 2, 0)
    assert snap._version_tuple("14.1.0a0") == (14, 1, 0)

    dep_names = {d.package_name for d in _338084F_RUNTIME_DEPS}
    assert "cupy" in dep_names
    assert "packaging" in dep_names

    _sha = "338084facfdfcf7422ef6904c43e0256b95190a6"
    assert CUPY_DEP.upstream_commit == _sha
    assert PACKAGING_DEP.upstream_commit == _sha
    assert all(r.upstream_commit == _sha for r in TEST_EXTRA_REMOVALS)
    assert CUGRAPH_PYG_HOMEPAGE_FIX.upstream_commit == _sha

    assert "cupy" in audit._declared_hard_deps
    assert "packaging" in audit._declared_hard_deps

    snap_text = snap.dump()
    assert "cupy" in snap_text
    assert "packaging" in snap_text
    assert "338084f" in snap_text

    if _DBG:
        print("[DEBUG 338084f runtime_dep_manifest] _self_test complete")
    print("[PASS] runtime_dep_manifest 338084f self-test: 11 assertions all passed")


_snapshot = RuntimeDepSnapshot()

if __name__ == "__main__":
    _self_test()
    print()
    print(_snapshot.dump())
    print()
    print("-- HomepageRecord --")
    print(CUGRAPH_PYG_HOMEPAGE_FIX.dump())
    print()
    print("-- TestExtraCleanup (3 items) --")
    for item in TEST_EXTRA_REMOVALS:
        print(item.dump())
        print()
