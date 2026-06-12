"""
conda_build_policy.py — migrate 6a80350: Use conda-build instead of conda-mambabuild (PR #150)

上游历史:
  - 6a80350: Use conda-build instead of conda-mambabuild (#150)
      * Author: Bradley Dice (bdice@bradleydice.com)
      * Date: 2025-02-26
      * Context: conda 已内置 mamba solver，`conda mambabuild` (boa) 不再提供性能优势；
        废弃 boa，统一用 `conda build` 代替 `rapids-conda-retry mambabuild`。
        同步更新 test_notebooks.sh / test_python.sh 中 DGL channel：
          dglteam/label/th23_cu118 → th24_cu118
          dglteam/label/th23_cu121 → th24_cu124
        并更新 python/cugraph-dgl/README.md 文档。
      * 后续计划: migrate to rattler-build
      * 受影响文件 (5):
          ci/build_cpp.sh              mambabuild → build (1行)
          ci/build_python.sh           mambabuild → build (3处) + copyright 2024→2025
          ci/test_notebooks.sh         th23_cu118 → th24_cu118
          ci/test_python.sh            th23_cu118→th24_cu118, th23_cu121→th24_cu124
          python/cugraph-dgl/README.md th23 channel → th24 channel

Walpurgis 迁移语义（CI/merge → SKIP）:
  - ci/ 目录: Walpurgis 无 RAPIDS conda 构建矩阵，无 rapids-conda-retry 工具链
  - python/cugraph-dgl/README.md: channel 跃迁已由 dgl_package_policy.py 覆盖
  - 迁移位置: src/walpurgis/core/conda_build_policy.py — 新建，记录构建工具链跃迁语义

20% 改写（鲁迅拿法）:
  上游仅有 sed 替换，无任何工具链抽象。Walpurgis 改写：
  1. CondaBuildTool enum: MAMBABUILD / CONDA_BUILD / RATTLER_BUILD 三阶段类型化
  2. RapidsCondaRetryCommand dataclass: 封装 rapids-conda-retry <tool> 调用规格
  3. CondaBuildMigration dataclass: 封装 6a80350 工具切换事实
  4. BoaDeprecationProbe: 主动探测 boa/conda-mambabuild 残留
  5. CondaBuildAudit: 扫描脚本 mambabuild 残留，assert_clean() 供 CI
  6. 全链路 WALPURGIS_DEBUG=1 断点（6处）

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS conda_build_policy] {msg}", file=sys.stderr, flush=True)


# ── 断点1: CondaBuildTool ────────────────────────────────────────────────────

class CondaBuildTool(Enum):
    """
    RAPIDS conda 构建工具三阶段演进。6a80350: MAMBABUILD → CONDA_BUILD。
    断点1: from_verb() / cli_verb 调用时。
    """
    MAMBABUILD = auto()
    CONDA_BUILD = auto()
    RATTLER_BUILD = auto()

    @property
    def cli_verb(self) -> str:
        _dbg(f"CondaBuildTool.cli_verb: self={self.name}")
        return {
            CondaBuildTool.MAMBABUILD: "mambabuild",
            CondaBuildTool.CONDA_BUILD: "build",
            CondaBuildTool.RATTLER_BUILD: "rattler-build",
        }[self]

    @property
    def backend(self) -> str:
        return {
            CondaBuildTool.MAMBABUILD: "boa",
            CondaBuildTool.CONDA_BUILD: "conda-build",
            CondaBuildTool.RATTLER_BUILD: "rattler-build",
        }[self]

    @property
    def deprecated(self) -> bool:
        return self is CondaBuildTool.MAMBABUILD

    @property
    def notes(self) -> str:
        return {
            CondaBuildTool.MAMBABUILD: (
                "Requires boa; dropped to unblock Python 3.13 migration. "
                "EOL as of 6a80350 (2025-02-26, PR #150)."
            ),
            CondaBuildTool.CONDA_BUILD: (
                "Interim step; conda now ships mamba solver natively. "
                "Will migrate to rattler-build (build-planning#149)."
            ),
            CondaBuildTool.RATTLER_BUILD: (
                "Planned future migration. Tracked in rapidsai/build-planning#149."
            ),
        }[self]

    @classmethod
    def from_verb(cls, verb: str) -> "CondaBuildTool":
        verb = verb.strip().lower()
        _dbg(f"CondaBuildTool.from_verb(verb={verb!r})")
        mapping = {
            "mambabuild": cls.MAMBABUILD,
            "build": cls.CONDA_BUILD,
            "rattler-build": cls.RATTLER_BUILD,
        }
        if verb not in mapping:
            raise ValueError(f"CondaBuildTool.from_verb: 未知动词 {verb!r}")
        return mapping[verb]

    def __repr__(self) -> str:
        dep = " [DEPRECATED]" if self.deprecated else ""
        return f"CondaBuildTool.{self.name}{dep}(verb={self.cli_verb!r})"


# ── 断点2: RapidsCondaRetryCommand ───────────────────────────────────────────

@dataclass
class RapidsCondaRetryCommand:
    """
    封装 rapids-conda-retry <tool> [flags] <recipe> 调用规格。
    断点2: build_cmd() / migrate() 调用时。
    """
    tool: CondaBuildTool
    env_vars: List[Tuple[str, str]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    recipe: str = ""

    def build_cmd(self) -> str:
        _dbg(f"RapidsCondaRetryCommand.build_cmd(): tool={self.tool.cli_verb!r} recipe={self.recipe!r}")
        env_prefix = " ".join(f"{k}={v}" for k, v in self.env_vars)
        flags_str = " \\\n  ".join(self.flags)
        parts = []
        if env_prefix:
            parts.append(env_prefix)
        parts.append(f"rapids-conda-retry {self.tool.cli_verb}")
        if flags_str:
            parts.append(f"\\\n  {flags_str}")
        if self.recipe:
            parts.append(self.recipe)
        return " ".join(parts)

    def migrate(self, to_tool: CondaBuildTool) -> "RapidsCondaRetryCommand":
        _dbg(f"RapidsCondaRetryCommand.migrate(): {self.tool.cli_verb!r} → {to_tool.cli_verb!r}")
        import dataclasses
        return dataclasses.replace(self, tool=to_tool)

    def __repr__(self) -> str:
        return f"RapidsCondaRetryCommand(tool={self.tool.cli_verb!r}, recipe={self.recipe!r})"


_BUILD_CPP_CMD = RapidsCondaRetryCommand(
    tool=CondaBuildTool.MAMBABUILD,
    env_vars=[("RAPIDS_PACKAGE_VERSION", "$(rapids-generate-version)")],
    recipe="conda/recipes/libwholegraph",
)
_BUILD_PY_CMDS = [
    RapidsCondaRetryCommand(
        tool=CondaBuildTool.MAMBABUILD,
        env_vars=[("RAPIDS_PACKAGE_VERSION", "$(head -1 ./VERSION)")],
        flags=["--no-test", "--channel", '"${CPP_CHANNEL}"'],
        recipe="conda/recipes/pylibwholegraph",
    ),
    RapidsCondaRetryCommand(
        tool=CondaBuildTool.MAMBABUILD,
        env_vars=[("RAPIDS_PACKAGE_VERSION", "$(head -1 ./VERSION)")],
        flags=["--no-test", "--channel", '"${CPP_CHANNEL}"'],
        recipe="conda/recipes/cugraph-pyg",
    ),
    RapidsCondaRetryCommand(
        tool=CondaBuildTool.MAMBABUILD,
        env_vars=[("RAPIDS_PACKAGE_VERSION", "$(head -1 ./VERSION)")],
        flags=["--no-test", "--channel", '"${CPP_CHANNEL}"'],
        recipe="conda/recipes/cugraph-dgl",
    ),
]
_BUILD_CPP_CMD_MIGRATED = _BUILD_CPP_CMD.migrate(CondaBuildTool.CONDA_BUILD)
_BUILD_PY_CMDS_MIGRATED = [cmd.migrate(CondaBuildTool.CONDA_BUILD) for cmd in _BUILD_PY_CMDS]


# ── 断点3: CondaBuildMigration ───────────────────────────────────────────────

@dataclass(frozen=True)
class CondaBuildMigration:
    """
    封装 6a80350 mambabuild → conda-build 工具切换事实。
    断点3: as_shell_diff() 调用时。
    """
    from_tool: CondaBuildTool
    to_tool: CondaBuildTool
    upstream_commit: str = "6a8035000e618dc537b9dd9353f0a247a4114e59"
    pr_number: int = 150
    author: str = "Bradley Dice (bdice@bradleydice.com)"
    rationale: str = (
        "conda now ships mamba solver natively; boa no longer needed. "
        "Temporary: plan to migrate to rattler-build (build-planning#149). "
        "Also unblocks Python 3.13 by dropping boa dependency."
    )
    affected_files: Tuple[str, ...] = (
        "ci/build_cpp.sh",
        "ci/build_python.sh",
        "ci/test_notebooks.sh",
        "ci/test_python.sh",
        "python/cugraph-dgl/README.md",
    )

    def as_shell_diff(self) -> str:
        _dbg(f"CondaBuildMigration.as_shell_diff(): {self.from_tool.cli_verb!r} → {self.to_tool.cli_verb!r}")
        lines = [
            f"# 6a80350 ({self.author}): mambabuild → conda build",
            f"# PR #{self.pr_number}: {self.rationale}",
            "",
            "# ci/build_cpp.sh:",
            f"- {_BUILD_CPP_CMD.build_cmd()}",
            f"+ {_BUILD_CPP_CMD_MIGRATED.build_cmd()}",
            "",
        ]
        for i, (before, after) in enumerate(zip(_BUILD_PY_CMDS, _BUILD_PY_CMDS_MIGRATED), start=1):
            lines.append(f"# ci/build_python.sh (命令 {i}/{len(_BUILD_PY_CMDS)}):")
            lines.append(f"- {before.build_cmd()}")
            lines.append(f"+ {after.build_cmd()}")
            lines.append("")
        return "\n".join(lines)

    def total_substitutions(self) -> int:
        return 1 + len(_BUILD_PY_CMDS)

    def __repr__(self) -> str:
        return (
            f"CondaBuildMigration("
            f"commit={self.upstream_commit[:8]!r}, pr=#{self.pr_number}, "
            f"from={self.from_tool.cli_verb!r} → to={self.to_tool.cli_verb!r})"
        )


THE_6A80350_MIGRATION = CondaBuildMigration(
    from_tool=CondaBuildTool.MAMBABUILD,
    to_tool=CondaBuildTool.CONDA_BUILD,
)


# ── 断点4: BoaDeprecationProbe ───────────────────────────────────────────────

class BoaDeprecationProbe:
    """
    主动探测 boa/conda-mambabuild 是否仍存在。
    断点4: probe() 调用时。
    """

    def probe(self) -> dict:
        _dbg("BoaDeprecationProbe.probe(): 开始探测")
        result = {
            "boa_importable": False,
            "mambabuild_on_path": False,
            "warning_issued": False,
        }
        try:
            import importlib.util as _ilu
            result["boa_importable"] = _ilu.find_spec("boa") is not None
        except Exception:
            pass
        _dbg(f"  boa_importable={result['boa_importable']}")

        conda_bin = shutil.which("conda")
        if conda_bin:
            try:
                proc = subprocess.run(
                    [conda_bin, "mambabuild", "--help"],
                    capture_output=True, timeout=5,
                )
                result["mambabuild_on_path"] = proc.returncode == 0
            except Exception:
                pass
        _dbg(f"  mambabuild_on_path={result['mambabuild_on_path']}")

        if result["boa_importable"] or result["mambabuild_on_path"]:
            warnings.warn(
                "[Walpurgis:BoaDeprecationProbe] 检测到 boa/conda-mambabuild 仍存在。"
                "上游 6a80350 (PR #150) 已废弃，请切换到 rapids-conda-retry build。",
                DeprecationWarning,
                stacklevel=2,
            )
            result["warning_issued"] = True
        _dbg(f"BoaDeprecationProbe.probe() → {result}")
        return result

    def __repr__(self) -> str:
        return "BoaDeprecationProbe()"


BOA_DEPRECATION_PROBE = BoaDeprecationProbe()


# ── 断点5: CondaBuildAudit ───────────────────────────────────────────────────

class CondaBuildAudit:
    """
    扫描 shell 脚本中残留 mambabuild 调用。
    断点5: scan() / assert_clean() 调用时。
    """
    _MAMBABUILD_PATTERN = re.compile(r"\bmambabuild\b")

    def scan(self, text: str) -> List[str]:
        _dbg(f"CondaBuildAudit.scan(): text length={len(text)}")
        matches = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if self._MAMBABUILD_PATTERN.search(line):
                matches.append(f"L{lineno}: {line.strip()}")
        _dbg(f"CondaBuildAudit.scan(): 找到 {len(matches)} 处")
        return matches

    def scan_file(self, filepath: str) -> List[str]:
        _dbg(f"CondaBuildAudit.scan_file({filepath!r})")
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return self.scan(f.read())
        except OSError as exc:
            _dbg(f"scan_file: 无法读取 {filepath!r}: {exc}")
            return []

    def assert_clean(self, text: str, label: str = "") -> None:
        matches = self.scan(text)
        if matches:
            loc = f" in {label!r}" if label else ""
            raise AssertionError(
                f"CondaBuildAudit.assert_clean(): {len(matches)} 处 mambabuild 残留{loc}:\n"
                + "\n".join(f"  {m}" for m in matches)
                + "\n请参见 6a80350 (PR #150)。"
            )
        _dbg(f"assert_clean({label!r}): CLEAN")

    def __repr__(self) -> str:
        return "CondaBuildAudit()"


CONDA_BUILD_AUDIT = CondaBuildAudit()


# ── 断点6: 自测 ───────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """断点6: 此处。"""
    _dbg("_run_self_tests(): 开始")
    results = []

    def _check(name: str, cond: bool) -> None:
        status = "PASS" if cond else "FAIL"
        results.append((name, status))
        _dbg(f"  [{status}] {name}")

    _check("MAMBABUILD.cli_verb='mambabuild'", CondaBuildTool.MAMBABUILD.cli_verb == "mambabuild")
    _check("CONDA_BUILD.cli_verb='build'", CondaBuildTool.CONDA_BUILD.cli_verb == "build")
    _check("MAMBABUILD.deprecated=True", CondaBuildTool.MAMBABUILD.deprecated is True)
    _check("CONDA_BUILD.deprecated=False", CondaBuildTool.CONDA_BUILD.deprecated is False)
    _check("from_verb('mambabuild')=MAMBABUILD", CondaBuildTool.from_verb("mambabuild") is CondaBuildTool.MAMBABUILD)
    _check("from_verb('build')=CONDA_BUILD", CondaBuildTool.from_verb("build") is CondaBuildTool.CONDA_BUILD)

    try:
        CondaBuildTool.from_verb("unknown")
        _check("from_verb(unknown) 应抛 ValueError", False)
    except ValueError:
        _check("from_verb(unknown) 正确抛 ValueError", True)

    _check("build_cpp mambabuild cmd 含 mambabuild", "mambabuild" in _BUILD_CPP_CMD.build_cmd())
    _check("build_cpp migrated 不含 mambabuild", "mambabuild" not in _BUILD_CPP_CMD_MIGRATED.build_cmd())
    migrated = _BUILD_CPP_CMD.migrate(CondaBuildTool.CONDA_BUILD)
    _check("migrate() 原实例不变", _BUILD_CPP_CMD.tool is CondaBuildTool.MAMBABUILD)
    _check("migrate() 新 tool=CONDA_BUILD", migrated.tool is CondaBuildTool.CONDA_BUILD)

    m = THE_6A80350_MIGRATION
    _check("commit 前8位='6a803500'", m.upstream_commit[:8] == "6a803500")
    _check("total_substitutions=4", m.total_substitutions() == 4)
    diff_str = m.as_shell_diff()
    _check("diff 含 mambabuild", "mambabuild" in diff_str)
    _check("diff 含 rapids-conda-retry build", "rapids-conda-retry build" in diff_str)

    dirty = "rapids-conda-retry mambabuild --no-test recipe"
    clean = "rapids-conda-retry build --no-test recipe"
    _check("dirty 检出 mambabuild", len(CONDA_BUILD_AUDIT.scan(dirty)) > 0)
    _check("clean 无 mambabuild", len(CONDA_BUILD_AUDIT.scan(clean)) == 0)
    try:
        CONDA_BUILD_AUDIT.assert_clean(dirty, "test_dirty")
        _check("assert_clean(dirty) 应抛", False)
    except AssertionError:
        _check("assert_clean(dirty) 正确抛", True)
    CONDA_BUILD_AUDIT.assert_clean(clean, "test_clean")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    _dbg(f"_run_self_tests(): {passed}/{total} 通过")
    if passed < total:
        failed = [n for n, s in results if s == "FAIL"]
        raise RuntimeError(f"conda_build_policy 自测失败 ({total-passed}/{total}): {failed}")


_run_self_tests()
_dbg(f"conda_build_policy module loaded: migration={THE_6A80350_MIGRATION!r}")


__all__ = [
    "CondaBuildTool",
    "RapidsCondaRetryCommand",
    "CondaBuildMigration",
    "THE_6A80350_MIGRATION",
    "BoaDeprecationProbe",
    "BOA_DEPRECATION_PROBE",
    "CondaBuildAudit",
    "CONDA_BUILD_AUDIT",
    "_BUILD_CPP_CMD",
    "_BUILD_CPP_CMD_MIGRATED",
    "_BUILD_PY_CMDS",
    "_BUILD_PY_CMDS_MIGRATED",
]
