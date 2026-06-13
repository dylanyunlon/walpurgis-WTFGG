# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 70c33af
# 原标题: [IMP] Remove SG and SNMG Examples (#171)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「我独自远行，不但没有你，并且再没有别的影在黑暗里。
#   只有我被黑暗沉没，那世界全属于我自己。」
# —— 鲁迅《野草·影的告别》
#
# 上游 70c33af 删除了 5 个 SG/SNMG example 文件（共 1167 行）：
#   gcn_dist_sg.py         (225行)
#   gcn_dist_snmg.py       (318行)
#   pylibcugraph_sg.py     ( 66行)
#   rgcn_link_class_sg.py  (228行)
#   rgcn_link_class_snmg.py(330行)
#
# 删除理由（PR #171）：
#   - 统一 API PR（unified API）将统一用 torchrun 运行，包括 SG/SNMG 工作流
#   - MNMG examples 将通过 unified API 处理 SG/SNMG/MNMG 三种场景
#   - SNMG 和 MNMG 都将数据直接存入 WholeGraph，而非主机内存
#     → SNMG/MNMG 工作流本质上等价，SG example 也可统一到 MNMG 路径
#
# Walpurgis 20% 改写（鲁迅拿法）：
#   上游是「直接 git rm」，Walpurgis 是「留下遗言」——
#   将此文件（原 gcn_dist_sg.py 的完整 SG 实现）替换为墓碑，
#   记录为什么删除、删什么、如何迁移，而不是让人莫名其妙地发现 example 消失了。
#   ExampleRemovalSpec 数据类结构化记录删除事件，便于未来审计。

from __future__ import annotations

import os as _os
import sys as _sys
import time as _time
from dataclasses import dataclass, field
from typing import List

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-EXAMPLE-TOMBSTONE:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ── 断点1: 墓碑模块加载检测 ──────────────────────────────────────────────────
_dbg(
    "module_load",
    "gcn_dist_sg.py 墓碑模块被 import。"
    "此文件在 70c33af ([IMP] Remove SG and SNMG Examples #171) 中已删除。"
    "SG 工作流请迁移至 gcn_dist_mnmg.py（通过 torchrun 以 world_size=1 运行）。",
)


@dataclass(frozen=True)
class ExampleRemovalSpec:
    """SG/SNMG Example 删除规格（migrate 70c33af Walpurgis 改写）。

    记录 70c33af 删除 SG/SNMG examples 的结构化信息：
    - 删除动机（统一 API + torchrun 迁移）
    - 受影响的文件清单
    - Walpurgis 对应的处置方式
    - 迁移路径（SG → MNMG with torchrun world_size=1）

    上游直接 git rm 5 个文件，Walpurgis 留下结构化墓碑。
    """

    commit_hash: str = "70c33af"
    pr_number: int = 171
    pr_title: str = "[IMP] Remove SG and SNMG Examples"

    # 删除动机
    removal_reason: str = (
        "统一 API PR 将使用 torchrun 统一运行 SG/SNMG/MNMG 工作流。"
        "SNMG/MNMG 都将数据存入 WholeGraph，两者本质等价。"
        "SG example 通过 world_size=1 + torchrun 可复用 MNMG 路径。"
    )

    # 上游删除的文件
    deleted_files: List[str] = field(default_factory=lambda: [
        "python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_sg.py",        # 225行 SG GCN
        "python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_snmg.py",      # 318行 SNMG GCN
        "python/cugraph-pyg/cugraph_pyg/examples/pylibcugraph_sg.py",    #  66行 SG degree
        "python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_sg.py", # 228行 SG RGCN
        "python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_snmg.py", # 330行 SNMG RGCN
    ])

    total_lines_deleted: int = 1167

    # Walpurgis 处置
    walpurgis_actions: List[str] = field(default_factory=lambda: [
        "src/walpurgis/examples/gcn/gcn_dist_sg.py → 替换为此墓碑文件",
        "src/walpurgis/examples/gcn/gcn_dist_snmg.py → 从未迁移（SKIP），不需要墓碑",
        "src/walpurgis/examples/plc/pylibcugraph_sg.py → 从未迁移（SKIP），不需要墓碑",
        "src/walpurgis/examples/rgcn/rgcn_link_class_sg.py → 从未迁移（SKIP），不需要墓碑",
        "src/walpurgis/examples/rgcn/rgcn_link_class_snmg.py → 从未迁移（SKIP），不需要墓碑",
    ])

    # 迁移路径
    migration_path: str = (
        "SG 工作流请使用 gcn_dist_mnmg.py，以 torchrun --nproc_per_node=1 运行。\n"
        "SNMG 工作流同样迁移至 gcn_dist_mnmg.py，以 torchrun --nproc_per_node=N 运行。\n"
        "数据存储从主机内存迁移至 WholeGraph（pylibwholegraph 硬依赖，见 e01196b）。"
    )

    def self_check(self) -> bool:
        """验证删除规格完整性。"""
        _dbg("ExampleRemovalSpec.self_check", f"验证 commit={self.commit_hash}")
        assert self.commit_hash == "70c33af", "commit hash 不匹配"
        assert len(self.deleted_files) == 5, f"上游删除了5个文件，实际记录{len(self.deleted_files)}"
        assert self.total_lines_deleted == 1167, (
            f"上游删除了1167行，实际记录{self.total_lines_deleted}"
        )
        _dbg("ExampleRemovalSpec.self_check", "ALL PASS")
        return True


# ── 公开的删除规格实例 ────────────────────────────────────────────────────────
SG_SNMG_REMOVAL_SPEC = ExampleRemovalSpec()


def _raise_sg_removed_error(*args, **kwargs):
    """gcn_dist_sg.py 已删除，任何作为脚本的调用均抛出此错误。

    断点2: gcn_dist_sg 作为脚本直接调用时
    """
    _dbg(
        "gcn_dist_sg_call",
        f"检测到对已删除的 gcn_dist_sg 的调用。"
        f"迁移路径: torchrun --nproc_per_node=1 gcn_dist_mnmg.py",
    )
    raise RuntimeError(
        "\n"
        "═══════════════════════════════════════════════════════════════╗\n"
        "  gcn_dist_sg.py 已在 cugraph-gnn commit 70c33af 中删除       ║\n"
        "  ([IMP] Remove SG and SNMG Examples, PR #171)                ║\n"
        "═══════════════════════════════════════════════════════════════╝\n"
        "\n"
        "迁移路径:\n"
        "  单 GPU 训练 (SG):\n"
        "    torchrun --nproc_per_node=1 gcn_dist_mnmg.py [args]\n"
        "\n"
        "  单机多 GPU (SNMG):\n"
        "    torchrun --nproc_per_node=N gcn_dist_mnmg.py [args]\n"
        "\n"
        "  多机多 GPU (MNMG):\n"
        "    torchrun --nnodes=M --nproc_per_node=N gcn_dist_mnmg.py [args]\n"
        "\n"
        "注意: pylibwholegraph 现在是硬依赖（commit e01196b）。\n"
        "     数据存储使用 WholeGraph 而非主机内存。\n"
        "\n"
        f"删除详情: {SG_SNMG_REMOVAL_SPEC.pr_title}\n"
        f"删除行数: {SG_SNMG_REMOVAL_SPEC.total_lines_deleted} 行（5个文件）\n"
    )


# ── 如果此文件被作为脚本直接运行，给出友好的错误提示 ──────────────────────────
if __name__ == "__main__":
    _raise_sg_removed_error()


__all__ = [
    "ExampleRemovalSpec",
    "SG_SNMG_REMOVAL_SPEC",
]
