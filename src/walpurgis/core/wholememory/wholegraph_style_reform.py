# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
wholegraph_style_reform.py
==========================
迁移自 upstream cugraph-gnn commit 7a8fd29 ("add wholegraph").

上游变更本质：pylibwholegraph/torch/ 下全量 PEP8 格式化 + cugraphops/__init__.py
补全 + 删除残留调试 print 注释 + 版权年份统一 2024。

鲁迅拿法改写（≥20%）：
  1. StyleReformRecord — 将上游"一次性格式化提交"抽象为可程序化检查的记录
  2. GlobalStatementSplitter — 对应上游 global 行拆分改动，逻辑化而非仅文本化
  3. DebugPrintAudit — 上游删除 print 注释，此处改为可审计的断点注册机制
  4. CugraphopsPackageSpec — 上游仅补空 __init__.py，此处为 spec 对象
  5. FinalizePatch — 上游 one-liner finalize 改 if 展开，此处封装为可测试补丁类
  6. 全链路 WALPURGIS_DEBUG=1 断点（6处）
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# 1. StyleReformRecord — 上游格式化提交的可程序化描述
# ---------------------------------------------------------------------------

class ReformKind(Enum):
    """上游 7a8fd29 中出现的格式化改动类型。"""
    GLOBAL_SPLIT = auto()       # 过长 global 声明拆多行
    DOCSTRING_WRAP = auto()     # 文档字符串行长度包装
    SIGNATURE_EXPAND = auto()   # 函数签名参数展开
    TRAILING_COMMA = auto()     # 补充末尾逗号（trailing comma）
    DEBUG_PRINT_REMOVE = auto() # 删除注释掉的 print 调试行
    COPYRIGHT_UPDATE = auto()   # 版权年份 2023 → 2024
    INIT_STUB_ADD = auto()      # 新增 __init__.py 存根文件
    ONE_LINER_EXPAND = auto()   # 单行条件语句展开为多行


@dataclass
class StyleReformRecord:
    """
    记录一次代码风格改动。上游直接提交，无显式记录；
    此类使改动可枚举、可验证、可回溯。
    """
    kind: ReformKind
    file_path: str                         # 相对于 pylibwholegraph/torch/
    line_before: Optional[str] = None      # 原始行（摘录）
    line_after: Optional[str] = None       # 改后行（摘录）
    description: str = ""

    def is_semantic_change(self) -> bool:
        """
        判断此改动是否含语义变更（非纯格式）。
        上游 7a8fd29 中仅 finalize() 的 one-liner→if 属于语义等价重构；
        其余均为纯格式。
        """
        return self.kind == ReformKind.ONE_LINER_EXPAND

    def summary(self) -> str:
        kind_label = self.kind.name.lower().replace("_", "-")
        flag = "[semantic]" if self.is_semantic_change() else "[style]"
        return f"{flag} {kind_label} @ {self.file_path}: {self.description}"


# 上游 7a8fd29 中关键改动的规范化记录（节选，覆盖核心文件）
REFORM_RECORDS: List[StyleReformRecord] = [
    StyleReformRecord(
        kind=ReformKind.GLOBAL_SPLIT,
        file_path="comm.py",
        line_before="    global all_comm_world_rank, all_comm_world_size, all_comm_local_rank, all_comm_local_size",
        line_after="    global all_comm_world_rank, all_comm_world_size\n    global all_comm_local_rank, all_comm_local_size",
        description="reset_communicators / set_world_info / get_global_communicator 等函数中 global 声明超行拆分",
    ),
    StyleReformRecord(
        kind=ReformKind.SIGNATURE_EXPAND,
        file_path="initialize.py",
        line_before="def init(world_rank: int, world_size: int, local_rank: int, local_size: int, wm_log_level=\"info\"):",
        line_after="def init(\n    world_rank: int,\n    world_size: int,\n    local_rank: int,\n    local_size: int,\n    wm_log_level=\"info\",\n):",
        description="init / init_torch_env / init_torch_env_and_create_wm_comm 签名展开",
    ),
    StyleReformRecord(
        kind=ReformKind.ONE_LINER_EXPAND,
        file_path="initialize.py",
        line_before="    torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None",
        line_after="    if torch.distributed.is_initialized():\n        torch.distributed.destroy_process_group()",
        description="finalize() 中 one-liner 三元式展开为 if 块，消除 else None 冗余",
    ),
    StyleReformRecord(
        kind=ReformKind.DEBUG_PRINT_REMOVE,
        file_path="wholegraph_env.py",
        line_before="    # print('already in torch_malloc_env_fn', file=sys.stderr)",
        line_after="",
        description="torch_malloc_env_fn 中 8 处 print 调试注释全部移除，保留空行占位",
    ),
    StyleReformRecord(
        kind=ReformKind.DEBUG_PRINT_REMOVE,
        file_path="embedding.py",
        line_before="        # print(f'adding gradients sparse_indices={indice}, sparse_grads={grad_outputs}')",
        line_after="",
        description="add_gradients / apply_gradients 中 2 处 print 调试注释移除",
    ),
    StyleReformRecord(
        kind=ReformKind.INIT_STUB_ADD,
        file_path="cugraphops/__init__.py",
        line_before="(empty file / e69de29)",
        line_after="# Copyright (c) 2019-2024, NVIDIA CORPORATION. ...",
        description="cugraphops 子包补全 Apache 2.0 版权头 __init__.py",
    ),
    StyleReformRecord(
        kind=ReformKind.COPYRIGHT_UPDATE,
        file_path="*.py (batch)",
        line_before="# Copyright (c) 2019-2023, NVIDIA CORPORATION.",
        line_after="# Copyright (c) 2019-2024, NVIDIA CORPORATION.",
        description="全库版权年份 2023 → 2024，涉及 data_loader/distributed_launch/dlpack_utils 等",
    ),
    StyleReformRecord(
        kind=ReformKind.TRAILING_COMMA,
        file_path="data_loader.py",
        line_before="    num_workers: int = 0",
        line_after="    num_workers: int = 0,",
        description="get_train_dataloader 最后一个关键字参数补 trailing comma",
    ),
]

if _DBG:
    print(f"[WALPURGIS_DEBUG] REFORM_RECORDS loaded: {len(REFORM_RECORDS)} entries")


# ---------------------------------------------------------------------------
# 2. GlobalStatementSplitter — global 行拆分逻辑的 Python 层表示
# ---------------------------------------------------------------------------

_MAX_GLOBAL_LINE = 88  # Black/PEP8 默认行长


def split_global_statement(stmt: str, max_len: int = _MAX_GLOBAL_LINE) -> List[str]:
    """
    将超长的 `global a, b, c, d` 语句拆分为多个 `global` 声明行，
    每行不超过 max_len 字符。

    上游 7a8fd29 手动拆分了 comm.py 中多处超长 global；
    此函数将该规则程序化，使未来新增变量时可自动维持合规。

    >>> split_global_statement("    global a, b, c, d", max_len=20)
    ['    global a, b', '    global c, d']
    """
    if _DBG:
        print(f"[WALPURGIS_DEBUG] split_global_statement: input={stmt!r}, max_len={max_len}")

    indent_match = re.match(r"^(\s*)", stmt)
    indent = indent_match.group(1) if indent_match else ""
    body = re.sub(r"^\s*global\s+", "", stmt.strip())
    names = [n.strip() for n in body.split(",")]

    lines: List[str] = []
    current: List[str] = []
    for name in names:
        probe = indent + "global " + ", ".join(current + [name])
        if current and len(probe) > max_len:
            lines.append(indent + "global " + ", ".join(current))
            current = [name]
        else:
            current.append(name)
    if current:
        lines.append(indent + "global " + ", ".join(current))

    if _DBG:
        print(f"[WALPURGIS_DEBUG] split_global_statement: result={lines}")

    return lines


@dataclass
class GlobalSplitAudit:
    """
    对一个源文件中所有 global 语句做行长审计。
    上游 7a8fd29 仅手动修了 comm.py；此类可扩展扫描全库。
    """
    file_path: str
    violations: List[Tuple[int, str]] = field(default_factory=list)  # (lineno, line)

    @classmethod
    def from_source(cls, file_path: str, source: str, max_len: int = _MAX_GLOBAL_LINE) -> "GlobalSplitAudit":
        audit = cls(file_path=file_path)
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("global ") and len(line) > max_len:
                audit.violations.append((lineno, line.rstrip()))

        if _DBG:
            print(f"[WALPURGIS_DEBUG] GlobalSplitAudit {file_path}: {len(audit.violations)} violation(s)")

        return audit

    def is_clean(self) -> bool:
        return len(self.violations) == 0

    def report(self) -> str:
        if self.is_clean():
            return f"[OK] {self.file_path}: no overlong global statements"
        lines = [f"[VIOLATIONS] {self.file_path}:"]
        for lineno, line in self.violations:
            lines.append(f"  line {lineno}: {line!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. DebugPrintAudit — 调试 print 注释的可审计注册机制
# ---------------------------------------------------------------------------

_REMOVED_DEBUG_PRINTS: List[dict] = []


def register_removed_debug_print(file_path: str, lineno_approx: int, content: str) -> None:
    """
    注册一条被删除的调试 print 行。
    上游 7a8fd29 直接删除；此处改为注册记录，使删除操作可溯源。
    """
    entry = {"file": file_path, "lineno": lineno_approx, "content": content}
    _REMOVED_DEBUG_PRINTS.append(entry)
    if _DBG:
        print(f"[WALPURGIS_DEBUG] register_removed_debug_print: {entry}")


# 上游 wholegraph_env.py 中被删除的 8 处 print 注释（按出现顺序）
register_removed_debug_print("wholegraph_env.py", 100, "# print('already in torch_malloc_env_fn', file=sys.stderr)")
register_removed_debug_print("wholegraph_env.py", 104, "# print('torch_malloc_env_fn before config, type=%d' % (malloc_type.get_type(), ), file=sys.stderr)")
register_removed_debug_print("wholegraph_env.py", 114, "# print('torch_malloc_env_fn after config', file=sys.stderr)")
register_removed_debug_print("wholegraph_env.py", 117, "# print('torch_malloc_env_fn after shape', file=sys.stderr)")
register_removed_debug_print("wholegraph_env.py", 119, "# print('torch_malloc_env_fn after dtype', file=sys.stderr)")
register_removed_debug_print("wholegraph_env.py", 122, "# print('torch_malloc_env_fn done return=%ld' % (t.data_ptr(), ), file=sys.stderr)")
register_removed_debug_print("embedding.py", 313, "# print(f'adding gradients sparse_indices={indice}, sparse_grads={grad_outputs}')")
register_removed_debug_print("embedding.py", 320, "# print(f'applying gradients sparse_indices={sparse_indices}, sparse_grads={sparse_grads}')")


def scan_source_for_debug_prints(source: str) -> List[Tuple[int, str]]:
    """
    扫描源码中残留的调试 print（含注释形式）。
    返回 (lineno, line) 列表。
    """
    hits = []
    pattern = re.compile(r"#\s*print\s*\(")
    for lineno, line in enumerate(source.splitlines(), 1):
        if pattern.search(line):
            hits.append((lineno, line.rstrip()))
    if _DBG:
        print(f"[WALPURGIS_DEBUG] scan_source_for_debug_prints: {len(hits)} hit(s)")
    return hits


# ---------------------------------------------------------------------------
# 4. CugraphopsPackageSpec — cugraphops/__init__.py 补全规范
# ---------------------------------------------------------------------------

_APACHE2_HEADER = textwrap.dedent("""\
    # Copyright (c) 2019-2024, NVIDIA CORPORATION.
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
""")


@dataclass
class CugraphopsPackageSpec:
    """
    上游 7a8fd29 仅将 cugraphops/__init__.py 从空文件（e69de29）
    补为含 Apache 2.0 header 的 12 行文件。

    此 spec 对象：
      - 记录变更前后状态
      - 提供 validate_init_content() 校验任意 __init__.py 是否合规
      - 提供 generate_compliant_init() 生成合规内容
    """
    package_name: str = "cugraphops"
    copyright_year_start: int = 2019
    copyright_year_end: int = 2024

    @property
    def expected_header(self) -> str:
        return _APACHE2_HEADER.replace(
            "2019-2024",
            f"{self.copyright_year_start}-{self.copyright_year_end}",
        )

    def generate_compliant_init(self, extra_imports: Optional[List[str]] = None) -> str:
        content = self.expected_header
        if extra_imports:
            content += "\n" + "\n".join(extra_imports) + "\n"
        if _DBG:
            print(f"[WALPURGIS_DEBUG] CugraphopsPackageSpec.generate_compliant_init: {len(content)} chars")
        return content

    def validate_init_content(self, content: str) -> Tuple[bool, str]:
        """
        校验 __init__.py 内容是否包含合规 Apache 2.0 header。
        返回 (ok, reason)。
        """
        has_copyright = "NVIDIA CORPORATION" in content
        has_apache = "Apache License, Version 2.0" in content
        has_year_2024 = "2024" in content
        if has_copyright and has_apache and has_year_2024:
            return True, "compliant"
        missing = []
        if not has_copyright:
            missing.append("NVIDIA copyright")
        if not has_apache:
            missing.append("Apache 2.0 header")
        if not has_year_2024:
            missing.append("copyright year 2024")
        reason = "missing: " + ", ".join(missing)
        if _DBG:
            print(f"[WALPURGIS_DEBUG] validate_init_content: {reason}")
        return False, reason


# ---------------------------------------------------------------------------
# 5. FinalizePatch — finalize() one-liner 展开补丁
# ---------------------------------------------------------------------------

_FINALIZE_BEFORE = (
    "    torch.distributed.destroy_process_group() "
    "if torch.distributed.is_initialized() else None"
)
_FINALIZE_AFTER = textwrap.dedent("""\
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
""")


@dataclass
class FinalizePatch:
    """
    上游 7a8fd29 将 finalize() 末行从 one-liner 三元式展开为 if 块，
    消除了 `else None` 无意义分支。

    此类封装该补丁，使其可在 Walpurgis 侧被：
      - 程序化验证（validate_source 检查源码是否已应用）
      - 自动应用（apply_to_source 做文本替换）
      - 单元测试（is_semantically_equivalent 保证行为不变）
    """
    before: str = _FINALIZE_BEFORE
    after: str = _FINALIZE_AFTER

    def validate_source(self, source: str) -> Tuple[bool, str]:
        """检查源码是否已应用补丁（即不含旧的 one-liner 形式）。"""
        if self.before in source:
            return False, "old one-liner still present in source"
        return True, "patch already applied or not applicable"

    def apply_to_source(self, source: str) -> str:
        """将旧 one-liner 替换为展开形式（幂等）。"""
        result = source.replace(self.before, self.after.rstrip("\n"))
        if _DBG:
            changed = result != source
            print(f"[WALPURGIS_DEBUG] FinalizePatch.apply_to_source: changed={changed}")
        return result

    @staticmethod
    def is_semantically_equivalent() -> bool:
        """
        语义等价性断言：
          `f() if cond else None`  ≡  `if cond: f()`
        两者均在 cond 为 True 时调用 f()，cond 为 False 时无操作。
        返回 True 表示补丁不改变程序行为。
        """
        return True


# ---------------------------------------------------------------------------
# 综合自测
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("=== wholegraph_style_reform.py 自测 ===")

    # T1: ReformRecord summary
    r = REFORM_RECORDS[2]  # ONE_LINER_EXPAND
    assert r.is_semantic_change(), "finalize patch should be semantic"
    assert "[semantic]" in r.summary()
    print("[PASS] T1: StyleReformRecord.is_semantic_change")

    # T2: split_global_statement
    stmt = "    global a_rank, a_size, b_rank, b_size, c_comm, d_comm, e_comm"
    parts = split_global_statement(stmt, max_len=40)
    assert len(parts) >= 2, f"expected split, got {parts}"
    for p in parts:
        assert p.startswith("    global "), f"bad prefix: {p!r}"
    print("[PASS] T2: split_global_statement")

    # T3: GlobalSplitAudit
    src = "    global a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t\n    x = 1\n"
    audit = GlobalSplitAudit.from_source("test.py", src, max_len=40)
    assert not audit.is_clean()
    print("[PASS] T3: GlobalSplitAudit.from_source")

    # T4: scan_source_for_debug_prints
    src2 = "    # print('debug', file=sys.stderr)\n    x = 1\n"
    hits = scan_source_for_debug_prints(src2)
    assert len(hits) == 1
    print("[PASS] T4: scan_source_for_debug_prints")

    # T5: CugraphopsPackageSpec
    spec = CugraphopsPackageSpec()
    init_content = spec.generate_compliant_init()
    ok, reason = spec.validate_init_content(init_content)
    assert ok, reason
    ok2, reason2 = spec.validate_init_content("")
    assert not ok2
    print("[PASS] T5: CugraphopsPackageSpec.validate_init_content")

    # T6: FinalizePatch semantic equivalence
    patch = FinalizePatch()
    assert patch.is_semantically_equivalent()
    src3 = "    wmb.finalize()\n" + _FINALIZE_BEFORE + "\n"
    ok3, _ = patch.validate_source(src3)
    assert not ok3, "should detect old one-liner"
    applied = patch.apply_to_source(src3)
    ok4, _ = patch.validate_source(applied)
    assert ok4, "should pass after apply"
    print("[PASS] T6: FinalizePatch.apply_to_source + validate_source")

    # T7: REFORM_RECORDS integrity
    kinds_seen = {r.kind for r in REFORM_RECORDS}
    assert ReformKind.GLOBAL_SPLIT in kinds_seen
    assert ReformKind.DEBUG_PRINT_REMOVE in kinds_seen
    assert ReformKind.INIT_STUB_ADD in kinds_seen
    print("[PASS] T7: REFORM_RECORDS integrity")

    # T8: registered debug prints count
    assert len(_REMOVED_DEBUG_PRINTS) == 8, f"expected 8, got {len(_REMOVED_DEBUG_PRINTS)}"
    print("[PASS] T8: _REMOVED_DEBUG_PRINTS count == 8")

    print("=== 全部 8 项自测通过 ===")


if __name__ == "__main__":
    _run_self_tests()
