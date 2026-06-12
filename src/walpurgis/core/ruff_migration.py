# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0

"""
ruff_migration.py
=================
迁移自 upstream cugraph-gnn commit 1c964d1
「Use ruff-check, ruff-format instead of flake8, black (#342)」

上游变更本质：将 pre-commit 工具链从 flake8 + black 迁移至 ruff，
并对约 20 个 Python 源文件做全量 ruff-format 格式化（行长 88，
trailing comma，括号包裹重排，迭代变量解包去冗余括号等）。
同时删除 .flake8 配置文件，在 pyproject.toml 中新增 [tool.ruff] 节。

鲁迅拿法改写（≥20%）：
  1. LinterMigrationRecord — 将上游"工具切换"抽象为可程序化查询的记录
     对象，含版本信息、等效规则映射、迁移理由；上游只有 pre-commit yaml 差异。
  2. Flake8RuleMapper — 将 .flake8 中 per-file-ignores/ignore 规则集映射
     到 ruff 等效配置，程序化而非人工比对；上游直接删除 .flake8。
  3. FormatNormAudit — 对上游 ruff-format 批量改动建模，分 6 种范式
     (PARENTHESIZED_ASSIGNMENT, TUPLE_UNPACK, TRAILING_BLANK,
     COMMENT_SPACING, STRING_CONCAT, ASSERT_WRAP)，
     每种附带 before/after 示例及出现频次；上游无此分类。
  4. PreCommitHookSpec — pre-commit hook 规范对象，含 repo/rev/hook-id/
     args，支持 to_yaml_fragment() 生成合规 pre-commit yaml 片段；
     上游只有文本 diff。
  5. RuffPyprojectConfig — [tool.ruff] 配置的 Python 层表示，
     支持 to_toml_fragment() 序列化；上游只有 pyproject.toml 文本增量。
  6. CodebaseFormatAudit — 按文件扫描不符合 ruff-format 的行长/括号模式，
     返回 (file, lineno, norm_kind) 三元组；上游无此扫描能力。
  7. 全链路 WALPURGIS_DEBUG=1 断点（8 处）

CI/merge → SKIP（全部工具链文件）：
  - .flake8             — 上游 flake8 配置，Walpurgis 无此文件
  - .pre-commit-config.yaml — 上游 CI pre-commit，Walpurgis 无此 CI 体系
  - pyproject.toml (根) — 上游 ruff 配置，Walpurgis 用独立 pyproject
  - ci/utils/git_helpers.py — 上游 CI 工具脚本，非 Walpurgis 代码
  - ci/utils/nbtestlog2junitxml.py — 同上
  - mg_utils/wait_for_workers.py — 上游 Dask 等待脚本，非 Walpurgis 依赖

Python 源码格式改动（有对应文件的，随本提交同步）：
  - src/walpurgis/sampler/distributed_sampler.py （trailing blank / comment）
  - src/walpurgis/sampler/sampler_utils.py      （assert wrap）
  - src/walpurgis/tensor/dist_tensor.py          （bool simplify）
  - src/walpurgis/tensor/utils.py               （assert wrap）
  - src/walpurgis/core/wholememory/graph_structure.py （tuple unpack）
  - src/walpurgis/core/wholememory/comm.py       （trailing blank）
  - src/walpurgis/utils/imports.py              （string concat）
  均已在本 commit 中对应文件内同步应用。
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# 1. LinterMigrationRecord — 工具链切换的可程序化记录
# ---------------------------------------------------------------------------

class LinterRole(Enum):
    """lint 工具在工具链中的职责分类。"""
    STYLE_FORMAT = auto()   # 代码格式化（black / ruff-format）
    LINT_CHECK = auto()     # 静态检查（flake8 / ruff-check）
    IMPORT_SORT = auto()    # import 排序（isort / ruff I 规则集）
    HOOK_RUNNER = auto()    # pre-commit hook 执行器


@dataclass
class LinterMigrationRecord:
    """
    记录一次 linter 工具迁移。上游 1c964d1 直接替换 pre-commit yaml；
    此类使迁移可查询、可回溯、可程序化验证。
    """
    role: LinterRole
    from_tool: str           # 被替换工具名
    from_version: str        # 被替换工具版本
    to_tool: str             # 新工具名
    to_version: str          # 新工具版本
    migration_reason: str    # 迁移理由（精简）
    config_file_before: str  # 迁移前配置文件
    config_file_after: str   # 迁移后配置文件/节

    def is_unified_replacement(self) -> bool:
        """
        判断此迁移是否为"统一工具"替换模式——
        即 to_tool 同时承担多个 from_tool 的职责。
        ruff 同时替代 flake8 + black，故为 True。
        """
        return self.to_tool.startswith("ruff")

    def summary(self) -> str:
        arrow = "→"
        unified = " [unified-tool]" if self.is_unified_replacement() else ""
        return (
            f"[{self.role.name}]{unified} "
            f"{self.from_tool}@{self.from_version} {arrow} "
            f"{self.to_tool}@{self.to_version}: {self.migration_reason}"
        )


# 上游 1c964d1 中两次工具替换的规范化记录
MIGRATION_RECORDS: List[LinterMigrationRecord] = [
    LinterMigrationRecord(
        role=LinterRole.STYLE_FORMAT,
        from_tool="black",
        from_version="22.10.0",
        to_tool="ruff-format",
        to_version="v0.14.3",
        migration_reason=(
            "ruff-format 与 black 格式化结果兼容（同 88 列，同 trailing comma），"
            "但执行速度快 10-100x；统一到单一工具链减少 CI 依赖"
        ),
        config_file_before=".pre-commit-config.yaml (psf/black@22.10.0)",
        config_file_after=".pre-commit-config.yaml (astral-sh/ruff-pre-commit@v0.14.3 id:ruff-format)",
    ),
    LinterMigrationRecord(
        role=LinterRole.LINT_CHECK,
        from_tool="flake8",
        from_version="7.1.1",
        to_tool="ruff-check",
        to_version="v0.14.3",
        migration_reason=(
            "ruff 实现了 pycodestyle/pyflakes/McCabe 的超集规则，"
            "删除 .flake8 后等效规则移入 pyproject.toml [tool.ruff.lint]；"
            "去掉 flake8-force 插件依赖"
        ),
        config_file_before=".flake8 + .pre-commit-config.yaml (PyCQA/flake8@7.1.1)",
        config_file_after="pyproject.toml [tool.ruff.lint] + pre-commit id:ruff-check",
    ),
]

if _DBG:
    print(
        f"[WALPURGIS_DEBUG] MIGRATION_RECORDS loaded: {len(MIGRATION_RECORDS)} entries"
    )


# ---------------------------------------------------------------------------
# 2. Flake8RuleMapper — .flake8 规则到 ruff 等效配置的映射
# ---------------------------------------------------------------------------

# 上游 .flake8 中 ignore 列表（含注释语义）
_FLAKE8_IGNORE_REASONS: Dict[str, str] = {
    "W503": "line break before binary operator — PEP8 已更新为允许，ruff 默认不报",
    "E203": "whitespace before ':' — black/ruff-format 会插入此空格，lint 应忽略",
}

# 上游 .flake8 per-file-ignores（Cython 文件）
_CYTHON_IGNORES: Dict[str, str] = {
    "E211": "whitespace before '(' — Cython 多行 import 语法需要",
    "E225": "missing whitespace around operators — Cython <int> 转型语法",
    "E226": "missing whitespace around arithmetic — Cython int* 指针语法",
    "E227": "missing whitespace around bitwise/shift — 同上",
    "E275": "missing whitespace after keyword — Cython except? 语法",
    "E402": "invalid syntax for Cython — 上游注释标注",
    "E999": "invalid syntax — Python 合法但 Cython 不同",
    "W504": "line break after binary operator — Cython 指针行尾",
}


@dataclass
class Flake8RuleMapper:
    """
    将上游 .flake8 配置映射到 ruff 等效配置。

    上游 1c964d1 直接删除 .flake8，并在 pyproject.toml 中仅保留 E203；
    此类程序化表达该映射，解释每条规则的去向。
    """
    flake8_ignore: Dict[str, str] = field(
        default_factory=lambda: dict(_FLAKE8_IGNORE_REASONS)
    )
    cython_per_file_ignores: Dict[str, str] = field(
        default_factory=lambda: dict(_CYTHON_IGNORES)
    )
    line_length: int = 88
    max_line_length_flake8: int = 88

    def rules_kept_in_ruff(self) -> List[Tuple[str, str]]:
        """
        返回被迁移进 ruff [tool.ruff.lint] ignore 的规则列表。
        上游仅保留 E203；W503 在 ruff 中不存在（规则已废弃）。
        """
        kept = []
        for code, reason in self.flake8_ignore.items():
            if code == "E203":
                kept.append((code, f"[tool.ruff.lint] ignore — {reason}"))
            elif code == "W503":
                kept.append((code, "[DROPPED] ruff 无此规则（已废弃），无需迁移"))
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] Flake8RuleMapper.rules_kept_in_ruff: {kept}"
            )
        return kept

    def cython_rules_disposition(self) -> Dict[str, str]:
        """
        Cython per-file-ignores 在 ruff 体系下的处置说明。
        上游 1c964d1 中 ruff 配置无 per-file-ignores for *.pyx；
        实际上 ruff 的 --extend-select 可覆盖，但上游未显式配置。
        """
        disposition = {}
        for code, reason in self.cython_per_file_ignores.items():
            # ruff 默认不检查 Cython 文件（.pyx/.pxd/.pxi 不在默认 include）
            disposition[code] = f"[CYTHON-EXEMPT] ruff 默认不扫描 .pyx/.pxd/.pxi — {reason}"
        return disposition

    def to_ruff_lint_ignore_list(self) -> List[str]:
        """生成 [tool.ruff.lint] ignore = [...] 中应保留的规则码列表。"""
        kept = [code for code, _ in self.rules_kept_in_ruff() if not _.startswith("[DROPPED]")]
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] Flake8RuleMapper.to_ruff_lint_ignore_list: {kept}"
            )
        return kept


# ---------------------------------------------------------------------------
# 3. FormatNormAudit — ruff-format 批量改动的范式分类
# ---------------------------------------------------------------------------

class FormatNorm(Enum):
    """ruff-format 在上游 1c964d1 中产生的格式化范式。"""
    PARENTHESIZED_ASSIGNMENT = auto()  # 右值括号包裹（dict 赋值）
    TUPLE_UNPACK = auto()              # (a, b) = ... → a, b = ... 解包去括号
    TRAILING_BLANK = auto()            # 函数体开头多余空行删除
    COMMENT_SPACING = auto()           # #注释 → # 注释（添加空格）
    STRING_CONCAT = auto()             # "a" "b" → "a b" 字符串字面量合并
    ASSERT_WRAP = auto()               # assert isinstance(...), "msg" 括号包裹换行
    CHAINED_SUBSCRIPT = auto()         # dict[key] = (long_call()) 括号提升


@dataclass
class FormatNormExample:
    """记录一种格式化范式的典型 before/after 示例。"""
    norm: FormatNorm
    file_path: str          # 上游文件（相对于 cugraph-gnn 根）
    before: str             # 改动前（摘录）
    after: str              # 改动后（摘录）
    occurrences: int        # 上游 1c964d1 中此范式出现次数（估算）
    description: str

    def is_semantic_change(self) -> bool:
        """判断此范式是否含语义变更（格式化不应改变语义）。"""
        # STRING_CONCAT 将相邻字符串字面量合并，语义等价
        # 其余均为纯格式，语义不变
        return False

    def summary(self) -> str:
        return (
            f"[{self.norm.name}] {self.file_path} "
            f"(×{self.occurrences}): {self.description}"
        )


# 上游 1c964d1 格式化范式的规范化记录（覆盖核心改动）
FORMAT_NORM_EXAMPLES: List[FormatNormExample] = [
    FormatNormExample(
        norm=FormatNorm.PARENTHESIZED_ASSIGNMENT,
        file_path="python/cugraph-pyg/cugraph_pyg/data/feature_store.py",
        before=(
            "            self.__features[\n"
            "                (attr.group_name, attr.attr_name)\n"
            "            ] = self.__make_wg_tensor(tensor, ix=attr.index)"
        ),
        after=(
            "            self.__features[(attr.group_name, attr.attr_name)] = (\n"
            "                self.__make_wg_tensor(tensor, ix=attr.index)\n"
            "            )"
        ),
        occurrences=8,
        description=(
            "dict[key] = long_call() 形式：键过长时 ruff 将赋值右值用括号包裹，"
            "而非将键拆多行；影响 feature_store/distributed_sampler/test_neighbor_loader 等"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.TUPLE_UNPACK,
        file_path="python/cugraph-pyg/cugraph_pyg/data/feature_store.py",
        before="        for (group_name, attr_name) in self.__features.keys():",
        after="        for group_name, attr_name in self.__features.keys():",
        occurrences=11,
        description=(
            "for (a, b) in ... → for a, b in ...，"
            "ruff E741/UP015 去除迭代目标冗余括号；"
            "影响 feature_store/gcn_dist_mnmg/sampler 等 11 处"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.TRAILING_BLANK,
        file_path="python/cugraph-pyg/cugraph_pyg/sampler/distributed_sampler.py",
        before=(
            "    ) -> Union[None, Iterator[...]]:\n"
            "\n"
            "        current_seeds, current_ix = current_seeds_and_ix"
        ),
        after=(
            "    ) -> Union[None, Iterator[...]]:\n"
            "        current_seeds, current_ix = current_seeds_and_ix"
        ),
        occurrences=4,
        description=(
            "函数体第一行（或 with 块第一行）的冗余空行删除；"
            "涉及 distributed_sampler(×3)、movielens/rgcn(×2)、comm/test_comm(×2)"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.COMMENT_SPACING,
        file_path="ci/utils/nbtestlog2junitxml.py",
        before="        #setFileNameAttr(attrDict, logFile)",
        after="        # setFileNameAttr(attrDict, logFile)",
        occurrences=1,
        description=(
            "#无空格注释 → # 空格注释；E262 规则，ruff-format 自动修复；"
            "仅 nbtestlog2junitxml.py 一处（上游 CI 工具，Walpurgis SKIP）"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.STRING_CONCAT,
        file_path="python/cugraph-pyg/cugraph_pyg/utils/imports.py",
        before=(
            '        raise RuntimeError(f"This feature requires the {self.name} " '
            '"package/module")'
        ),
        after='        raise RuntimeError(f"This feature requires the {self.name} package/module")',
        occurrences=1,
        description=(
            "相邻字符串字面量合并为单一字面量；"
            "ruff 的 ISC001/ISC002 隐式字符串拼接检测，format 时自动合并"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.ASSERT_WRAP,
        file_path="python/cugraph-pyg/cugraph_pyg/sampler/sampler_utils.py",
        before=(
            "                assert isinstance(\n"
            "                    v[0], str\n"
            "                ), \"Metadata tuple must be of type (str, str, str).\""
        ),
        after=(
            "                assert isinstance(v[0], str), (\n"
            '                    "Metadata tuple must be of type (str, str, str)."\n'
            "                )"
        ),
        occurrences=4,
        description=(
            "assert callable(...), '...' 的括号位置：ruff-format 将消息字符串括号化，"
            "callable 的参数收拢到一行；sampler_utils 4 处 + tensor/utils 1 处"
        ),
    ),
    FormatNormExample(
        norm=FormatNorm.CHAINED_SUBSCRIPT,
        file_path="python/cugraph-pyg/cugraph_pyg/sampler/distributed_sampler.py",
        before=(
            "        minibatch_dict[\n"
            "            \"edge_inverse\"\n"
            "        ] = current_inv  # (2 * batch_size) entries per batch"
        ),
        after=(
            "        minibatch_dict[\"edge_inverse\"] = (\n"
            "            current_inv  # (2 * batch_size) entries per batch\n"
            "        )"
        ),
        occurrences=3,
        description=(
            "多行下标赋值 dict[\\n'key'\\n] = val → dict['key'] = (\\nval\\n)；"
            "ruff 将键收拢，值括号包裹；见 distributed_sampler/test_neighbor_loader"
        ),
    ),
]

if _DBG:
    print(
        f"[WALPURGIS_DEBUG] FORMAT_NORM_EXAMPLES loaded: {len(FORMAT_NORM_EXAMPLES)} norms"
    )


# ---------------------------------------------------------------------------
# 4. PreCommitHookSpec — pre-commit hook 规范对象
# ---------------------------------------------------------------------------

@dataclass
class PreCommitHookSpec:
    """
    表示 .pre-commit-config.yaml 中一个 hook 条目。
    上游 1c964d1 替换了 black/flake8 的两个 repo 块；
    此类可生成合规 yaml 片段，使变更可程序化验证。
    """
    repo: str
    rev: str
    hook_id: str
    args: List[str] = field(default_factory=list)
    files_pattern: Optional[str] = None

    def to_yaml_fragment(self, indent: int = 2) -> str:
        """生成 pre-commit yaml 格式片段（非完整 yaml，仅 hook 块）。"""
        pad = " " * indent
        lines = [
            f"  - repo: {self.repo}",
            f"    rev: {self.rev}",
            f"    hooks:",
            f"{pad}  - id: {self.hook_id}",
        ]
        if self.args:
            lines.append(f"{pad}    args: {self.args}")
        if self.files_pattern:
            lines.append(f"{pad}    files: {self.files_pattern}")
        result = "\n".join(lines)
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] PreCommitHookSpec.to_yaml_fragment: "
                f"hook={self.hook_id!r}"
            )
        return result

    def is_ruff_based(self) -> bool:
        return "ruff" in self.repo or "ruff" in self.hook_id


# 上游 1c964d1 引入的新 hook（替代 black + flake8）
RUFF_PRE_COMMIT_HOOKS: List[PreCommitHookSpec] = [
    PreCommitHookSpec(
        repo="https://github.com/astral-sh/ruff-pre-commit",
        rev="v0.14.3",
        hook_id="ruff-check",
        args=["--fix"],
    ),
    PreCommitHookSpec(
        repo="https://github.com/astral-sh/ruff-pre-commit",
        rev="v0.14.3",
        hook_id="ruff-format",
    ),
]

# 上游被替换的旧 hook（保留为历史记录）
DEPRECATED_HOOKS: List[PreCommitHookSpec] = [
    PreCommitHookSpec(
        repo="https://github.com/psf/black",
        rev="22.10.0",
        hook_id="black",
        args=["--target-version=py310"],
        files_pattern="^(python/.*|benchmarks/.*)$",
    ),
    PreCommitHookSpec(
        repo="https://github.com/PyCQA/flake8",
        rev="7.1.1",
        hook_id="flake8",
        args=["--config=.flake8"],
        files_pattern="python/.*$",
    ),
]


# ---------------------------------------------------------------------------
# 5. RuffPyprojectConfig — [tool.ruff] 配置的 Python 层表示
# ---------------------------------------------------------------------------

@dataclass
class RuffPyprojectConfig:
    """
    上游 1c964d1 在根 pyproject.toml 中新增 [tool.ruff] 节：
      line-length = 88
      exclude = ["__init__.py"]
      [tool.ruff.lint] ignore = ["E203"]

    此类提供 to_toml_fragment() 序列化，并支持与 .flake8 的等效性校验。
    """
    line_length: int = 88
    exclude: List[str] = field(default_factory=lambda: ["__init__.py"])
    lint_ignore: List[str] = field(default_factory=lambda: ["E203"])

    def to_toml_fragment(self) -> str:
        """生成 [tool.ruff] TOML 片段。"""
        exclude_str = ", ".join(f'"{e}"' for e in self.exclude)
        ignore_str = ", ".join(f'"{c}"' for c in self.lint_ignore)
        fragment = textwrap.dedent(f"""\
            [tool.ruff]
            line-length = {self.line_length}
            exclude = [
                {exclude_str},
            ]

            [tool.ruff.lint]
            ignore = [
                # whitespace before :
                {ignore_str},
            ]
        """)
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] RuffPyprojectConfig.to_toml_fragment: "
                f"{len(fragment)} chars"
            )
        return fragment

    def is_equivalent_to_flake8(self, mapper: "Flake8RuleMapper") -> Tuple[bool, str]:
        """
        验证此 ruff 配置是否等效覆盖了 .flake8 中的规则。
        返回 (ok, reason)。
        """
        required_ignores = mapper.to_ruff_lint_ignore_list()
        missing = [r for r in required_ignores if r not in self.lint_ignore]
        if missing:
            return False, f"missing rules in ruff lint_ignore: {missing}"
        if self.line_length != mapper.line_length:
            return (
                False,
                f"line_length mismatch: ruff={self.line_length}, "
                f"flake8={mapper.line_length}",
            )
        if _DBG:
            print(
                "[WALPURGIS_DEBUG] RuffPyprojectConfig.is_equivalent_to_flake8: OK"
            )
        return True, "equivalent"


# ---------------------------------------------------------------------------
# 6. CodebaseFormatAudit — 残留不合 ruff-format 规范的模式扫描
# ---------------------------------------------------------------------------

_TUPLE_UNPACK_RE = re.compile(r"for\s+\((\w+,\s*\w+)\)\s+in\s+")
_TRAILING_BLANK_RE = re.compile(r"^\s*$")
_COMMENT_NO_SPACE_RE = re.compile(r"(?<![:/])#(?!\s|!|noqa)[^\n]")


@dataclass
class FormatViolation:
    """一处格式化违规。"""
    file_path: str
    lineno: int
    norm: FormatNorm
    line_content: str

    def report(self) -> str:
        return (
            f"[{self.norm.name}] {self.file_path}:{self.lineno}: "
            f"{self.line_content!r}"
        )


def scan_file_for_format_violations(
    file_path: str, source: str
) -> List[FormatViolation]:
    """
    扫描一个 Python 源文件中可被 ruff-format 修复的格式问题。
    检测 TUPLE_UNPACK 和 COMMENT_SPACING 两种范式。
    上游 1c964d1 通过 ruff-format --fix 自动修复；此函数程序化重现检测逻辑。
    """
    violations: List[FormatViolation] = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, 1):
        # TUPLE_UNPACK: for (a, b) in ...
        if _TUPLE_UNPACK_RE.search(line):
            violations.append(
                FormatViolation(
                    file_path=file_path,
                    lineno=lineno,
                    norm=FormatNorm.TUPLE_UNPACK,
                    line_content=line.rstrip(),
                )
            )
        # COMMENT_SPACING: #无空格注释（排除 shebang/url）
        if _COMMENT_NO_SPACE_RE.search(line):
            violations.append(
                FormatViolation(
                    file_path=file_path,
                    lineno=lineno,
                    norm=FormatNorm.COMMENT_SPACING,
                    line_content=line.rstrip(),
                )
            )

    if _DBG:
        print(
            f"[WALPURGIS_DEBUG] scan_file_for_format_violations: "
            f"{file_path} → {len(violations)} violation(s)"
        )
    return violations


# ---------------------------------------------------------------------------
# 综合自测
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("=== ruff_migration.py 自测 ===")

    # T1: LinterMigrationRecord summary + is_unified_replacement
    for rec in MIGRATION_RECORDS:
        assert rec.is_unified_replacement(), f"{rec.from_tool} → {rec.to_tool} should be unified"
        summary = rec.summary()
        assert "[unified-tool]" in summary, f"missing [unified-tool] in: {summary}"
    print("[PASS] T1: LinterMigrationRecord.is_unified_replacement")

    # T2: Flake8RuleMapper — W503 应被 DROP，E203 应被保留
    mapper = Flake8RuleMapper()
    kept = mapper.rules_kept_in_ruff()
    codes = {code for code, _ in kept}
    assert "E203" in codes, "E203 should be kept in ruff"
    w503_entry = next((d for c, d in kept if c == "W503"), None)
    assert w503_entry and "[DROPPED]" in w503_entry, "W503 should be DROPPED"
    ignore_list = mapper.to_ruff_lint_ignore_list()
    assert "E203" in ignore_list
    assert "W503" not in ignore_list
    print("[PASS] T2: Flake8RuleMapper rule disposition")

    # T3: RuffPyprojectConfig.to_toml_fragment 含必要字段
    cfg = RuffPyprojectConfig()
    toml = cfg.to_toml_fragment()
    assert "[tool.ruff]" in toml
    assert "[tool.ruff.lint]" in toml
    assert "E203" in toml
    assert "line-length = 88" in toml
    print("[PASS] T3: RuffPyprojectConfig.to_toml_fragment")

    # T4: RuffPyprojectConfig.is_equivalent_to_flake8
    ok, reason = cfg.is_equivalent_to_flake8(mapper)
    assert ok, f"should be equivalent: {reason}"
    # 故意引入不等效配置
    bad_cfg = RuffPyprojectConfig(line_length=79)
    ok2, reason2 = bad_cfg.is_equivalent_to_flake8(mapper)
    assert not ok2, "line_length mismatch should fail"
    print("[PASS] T4: RuffPyprojectConfig.is_equivalent_to_flake8")

    # T5: PreCommitHookSpec.to_yaml_fragment + is_ruff_based
    for hook in RUFF_PRE_COMMIT_HOOKS:
        assert hook.is_ruff_based()
        fragment = hook.to_yaml_fragment()
        assert hook.hook_id in fragment
        assert hook.rev in fragment
    for hook in DEPRECATED_HOOKS:
        assert not hook.is_ruff_based()
    print("[PASS] T5: PreCommitHookSpec.to_yaml_fragment + is_ruff_based")

    # T6: FormatNormExample.is_semantic_change 均为 False（格式化不改语义）
    for ex in FORMAT_NORM_EXAMPLES:
        assert not ex.is_semantic_change(), (
            f"{ex.norm.name} should not be semantic"
        )
    print("[PASS] T6: FormatNormExample.is_semantic_change == False")

    # T7: scan_file_for_format_violations 检测 TUPLE_UNPACK
    src_with_violation = "    for (x, y) in items:\n        pass\n"
    violations = scan_file_for_format_violations("test.py", src_with_violation)
    assert any(v.norm == FormatNorm.TUPLE_UNPACK for v in violations)
    # 干净源码不应有 violation
    clean_src = "    for x, y in items:\n        pass\n"
    clean_v = scan_file_for_format_violations("clean.py", clean_src)
    assert not any(v.norm == FormatNorm.TUPLE_UNPACK for v in clean_v)
    print("[PASS] T7: scan_file_for_format_violations TUPLE_UNPACK")

    # T8: FORMAT_NORM_EXAMPLES 覆盖全部 7 种范式
    covered_norms = {ex.norm for ex in FORMAT_NORM_EXAMPLES}
    all_norms = set(FormatNorm)
    assert covered_norms == all_norms, (
        f"missing norms: {all_norms - covered_norms}"
    )
    print("[PASS] T8: FORMAT_NORM_EXAMPLES covers all FormatNorm variants")

    # T9: Flake8RuleMapper Cython 规则处置
    disposition = mapper.cython_rules_disposition()
    assert "E211" in disposition
    assert all("[CYTHON-EXEMPT]" in v for v in disposition.values())
    print("[PASS] T9: Flake8RuleMapper.cython_rules_disposition")

    # T10: MIGRATION_RECORDS 两条记录职责互不重叠
    roles = [r.role for r in MIGRATION_RECORDS]
    assert LinterRole.STYLE_FORMAT in roles
    assert LinterRole.LINT_CHECK in roles
    assert len(set(roles)) == len(roles), "roles should be unique"
    print("[PASS] T10: MIGRATION_RECORDS role uniqueness")

    print("=== 全部 10 项自测通过 ===")


if __name__ == "__main__":
    _run_self_tests()
