"""
upstream_compat_notice.py — migrate 3f8dddf: resolve merge conflict
(cugraph-gnn, Alexandria Barghi, 2025-06-09)

上游 3f8dddf 实质内容（merge commit d491fae ← 78d3f72）:

  README.md (+5行, branch-25.06 侧):
    * cuGraph-DGL supports the Deep Graph Library (DGL) and offers duck-typed
      versions of DGL's native graph objects, samplers, and loaders.
    ** cuGraph-DGL is slated for removal after release 25.06.
       We strongly recommend migrating to cuGraph-PyG.

  datasets/karate.csv: 删除 (156行 Zachary Karate Club edge-list，
    已由 43a80e8 在 walpurgis 中恢复，此处记 SKIP)

迁移语义:
  上游 README merge 引入了 DGL 移除时间线（slated after 25.06 →
  实际在 25.08 完成移除，见上游 NEWS）。walpurgis 的
  dgl_deprecation.py (from 456d5a2) 已有废弃 warning 机制，
  但缺少对应的「版本时间线」与「合并解析来源」可查询接口。

  本模块补充:
    CompatNotice      — 单条上游兼容性公告的数据结构
    CompatNoticeBoard — 公告注册表，可按 component / release 查询
    DGL_REMOVAL_NOTICE — 预置：cuGraph-DGL 移除时间线（源自 3f8dddf）
    KARATE_DELETION_NOTICE — 预置：karate.csv 删除公告（SKIP，已由 43a80e8 覆盖）

鲁迅拿法改写 (>20%，对比上游 README 纯文本):
  1. 数据结构化: 上游仅有散文 README 段落；
     此处用 CompatNotice dataclass 将来源、组件、版本、消息、状态
     全部显式字段化，可程序化查询。
  2. 注册表模式: CompatNoticeBoard 替代静态文本，
     支持 filter_by_component() / filter_by_status() / as_warning()。
  3. 状态机: NoticeStatus 枚举 (SLATED/REMOVED/RESTORED/SKIP)，
     对应上游 commit 序列的实际演变。
  4. 断点1: CompatNoticeBoard.__init__ 加载公告时。
  5. 断点2: CompatNoticeBoard.as_warning() 发出运行时警告时。
  6. 断点3: check_dgl_removal() 被调用时（供 dgl_deprecation 集成）。
  7. 版本比较: _release_le() 辅助函数，按 YY.MM 格式比较版本，
     用于判断当前版本是否超过 slated release（上游无此逻辑）。

作者: dylanyunlon<dogechat@163.com>
"""

from __future__ import annotations

import enum
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

# ── 调试基础设施 ──────────────────────────────────────────────────────────────

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """WALPURGIS_DEBUG=1 时向 stderr 输出带标签的调试行。"""
    if _DEBUG:
        print(f"[WALPURGIS upstream_compat_notice | {tag}] {msg}",
              file=sys.stderr, flush=True)


# ── 版本比较辅助 ──────────────────────────────────────────────────────────────

def _release_le(ver_a: str, ver_b: str) -> bool:
    """
    判断 YY.MM 格式版本号 ver_a <= ver_b。

    例: _release_le("25.06", "25.08") → True
        _release_le("25.08", "25.06") → False
        _release_le("25.06", "25.06") → True

    上游 README 仅以散文说明版本，无程序化比较；
    此函数为鲁迅拿法新增，用于 CompatNotice.is_past_slated()。
    """
    try:
        a_parts = tuple(int(x) for x in ver_a.split("."))
        b_parts = tuple(int(x) for x in ver_b.split("."))
        return a_parts <= b_parts
    except (ValueError, AttributeError):
        # 无法解析时保守返回 False（不误判为已过期）
        _dbg("_release_le", f"无法解析版本号 {ver_a!r} / {ver_b!r}，返回 False")
        return False


# ── 状态枚举 ──────────────────────────────────────────────────────────────────

class NoticeStatus(enum.Enum):
    """
    上游兼容性公告的生命周期状态。

    对应 cugraph-gnn 的实际 commit 序列：
      SLATED  — 上游公告「计划在某版本后移除」(branch-25.06 侧, 3f8dddf)
      REMOVED — 上游已执行移除（如 25.08 实际删除 cuGraph-DGL）
      RESTORED — 被后续 commit 恢复（如 karate.csv 被 43a80e8 恢复）
      SKIP    — walpurgis 判定无需迁移（devcontainer 配置等）
    """
    SLATED = "slated"
    REMOVED = "removed"
    RESTORED = "restored"
    SKIP = "skip"


# ── 核心数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class CompatNotice:
    """
    单条上游兼容性公告。

    对应上游 3f8dddf README 段落（鲁迅拿法：散文 → 结构化字段）：

      原文: "cuGraph-DGL is slated for removal after release 25.06.
             We strongly recommend migrating to cuGraph-PyG."

      结构化后:
        component       = "cuGraph-DGL"
        slated_release  = "25.06"
        removed_release = "25.08"
        status          = NoticeStatus.REMOVED
        message         = <原文>
        migrate_to      = "cuGraph-PyG / walpurgis.dataloader.DataLoader"
        upstream_commit = "3f8dddf"
    """
    component: str
    """受影响的组件名称（如 "cuGraph-DGL", "datasets/karate.csv"）。"""

    status: NoticeStatus
    """当前生命周期状态。"""

    message: str
    """公告正文（原始或 walpurgis 改写后）。"""

    upstream_commit: str
    """引入此公告的上游 commit SHA。"""

    slated_release: Optional[str] = None
    """计划移除的版本号 (YY.MM)，对应上游 "slated for removal after X"。"""

    removed_release: Optional[str] = None
    """实际执行移除的版本号 (YY.MM)。"""

    migrate_to: Optional[str] = None
    """推荐迁移路径。"""

    extra: dict = field(default_factory=dict)
    """扩展字段，供子系统附加元数据。"""

    def is_past_slated(self, current_release: str) -> bool:
        """
        判断当前版本是否已超过「slated for removal」时间点。

        参数
        ----
        current_release: YY.MM 格式的当前版本字符串。

        返回
        ----
        True 当且仅当 slated_release 已设置且 slated_release <= current_release。
        """
        if self.slated_release is None:
            return False
        return _release_le(self.slated_release, current_release)

    def as_warning_text(self) -> str:
        """生成运行时 warnings.warn 的消息文本。"""
        parts = [f"[{self.component}] {self.message}"]
        if self.slated_release:
            parts.append(f"(slated for removal after {self.slated_release})")
        if self.removed_release:
            parts.append(f"(removed in {self.removed_release})")
        if self.migrate_to:
            parts.append(f"Migrate to: {self.migrate_to}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"CompatNotice(component={self.component!r}, "
            f"status={self.status.value!r}, "
            f"slated={self.slated_release!r}, "
            f"removed={self.removed_release!r})"
        )


# ── 注册表 ────────────────────────────────────────────────────────────────────

class CompatNoticeBoard:
    """
    上游兼容性公告注册表。

    鲁迅拿法核心改写：上游 3f8dddf 的变更是静态 README 文本段落；
    此处实现为可程序化查询的注册表，支持按组件、状态、版本过滤，
    并可在运行时发出 DeprecationWarning / FutureWarning。

    用法::

        board = CompatNoticeBoard()
        board.register(DGL_REMOVAL_NOTICE)

        # 查询
        dgl_notices = board.filter_by_component("cuGraph-DGL")

        # 发出运行时警告
        board.as_warning("cuGraph-DGL")
    """

    def __init__(self) -> None:
        self._notices: List[CompatNotice] = []
        self._warn_issued: dict[str, bool] = {}  # component → 是否已发出过 warning

        # ── 断点1 ────────────────────────────────────────────────────────────
        _dbg("CompatNoticeBoard.__init__", "注册表初始化完成，notices=[]")

    def register(self, notice: CompatNotice) -> None:
        """注册一条兼容性公告。"""
        self._notices.append(notice)
        _dbg(
            "CompatNoticeBoard.register",
            f"注册公告: {notice!r}",
        )

    def filter_by_component(self, component: str) -> List[CompatNotice]:
        """返回指定组件的所有公告列表。"""
        return [n for n in self._notices if n.component == component]

    def filter_by_status(self, status: NoticeStatus) -> List[CompatNotice]:
        """返回指定状态的所有公告列表。"""
        return [n for n in self._notices if n.status == status]

    def filter_past_slated(self, current_release: str) -> List[CompatNotice]:
        """
        返回已超过「slated for removal」时间点的公告列表。

        参数
        ----
        current_release: 当前版本 (YY.MM)。
        """
        result = [n for n in self._notices if n.is_past_slated(current_release)]
        _dbg(
            "filter_past_slated",
            f"current_release={current_release!r}, "
            f"past_slated={[r.component for r in result]}",
        )
        return result

    def as_warning(
        self,
        component: str,
        stacklevel: int = 2,
        force: bool = False,
    ) -> None:
        """
        对指定组件的所有非 SKIP 公告发出运行时 warning（默认去重）。

        参数
        ----
        component:  组件名（同 CompatNotice.component）。
        stacklevel: warnings.warn stacklevel 参数。
        force:      True → 强制重新发出，无视去重标志（测试用）。

        # ── 断点2 ────────────────────────────────────────────────────────
        """
        notices = [
            n for n in self.filter_by_component(component)
            if n.status != NoticeStatus.SKIP
        ]
        if not notices:
            _dbg("as_warning", f"component={component!r}: 无非SKIP公告，跳过")
            return

        key = component
        if not force and self._warn_issued.get(key, False):
            _dbg("as_warning", f"component={component!r}: 已发出，去重跳过")
            return

        self._warn_issued[key] = True

        for notice in notices:
            text = notice.as_warning_text()
            # ── 断点2 ────────────────────────────────────────────────────
            _dbg(
                "as_warning",
                f"发出 warning: component={component!r} "
                f"status={notice.status.value!r} text={text!r}",
            )
            category = (
                FutureWarning
                if notice.status == NoticeStatus.SLATED
                else DeprecationWarning
            )
            warnings.warn(text, category, stacklevel=stacklevel)

    def reset_warn_issued(self, component: Optional[str] = None) -> None:
        """重置 warning 去重标志（测试用）。"""
        if component is None:
            self._warn_issued.clear()
            _dbg("reset_warn_issued", "全部重置")
        else:
            self._warn_issued.pop(component, None)
            _dbg("reset_warn_issued", f"重置 {component!r}")

    def summary(self) -> str:
        """返回所有已注册公告的摘要字符串。"""
        lines = [f"CompatNoticeBoard ({len(self._notices)} notices):"]
        for n in self._notices:
            lines.append(f"  {n!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CompatNoticeBoard(notices={len(self._notices)})"


# ── 预置公告：来自 3f8dddf ────────────────────────────────────────────────────

#: cuGraph-DGL 移除公告
#: 来源: 3f8dddf branch-25.06 侧 README 段落
#: "cuGraph-DGL is slated for removal after release 25.06.
#:  We strongly recommend migrating to cuGraph-PyG."
DGL_REMOVAL_NOTICE = CompatNotice(
    component="cuGraph-DGL",
    status=NoticeStatus.REMOVED,  # 25.08 已实际移除，升级 SLATED → REMOVED
    message=(
        "cuGraph-DGL was slated for removal after release 25.06 and has been "
        "removed in release 25.08. "
        "We strongly recommend migrating to cuGraph-PyG / "
        "walpurgis.dataloader.DataLoader."
    ),
    upstream_commit="3f8dddf",
    slated_release="25.06",
    removed_release="25.08",
    migrate_to=(
        "cuGraph-PyG or walpurgis.dataloader.DataLoader / "
        "walpurgis.sampler.DistributedNeighborSampler"
    ),
    extra={
        "source": "README.md merge conflict resolution (branch-25.06 → main)",
        "merge_parents": ["d491fae", "78d3f72"],
        "note": (
            "The upstream diff retained git conflict markers verbatim "
            "(<<<<< HEAD / ======= / >>>>>>>) — the actual substantive "
            "content is the branch-25.06 side: DGL removal timeline."
        ),
    },
)

#: karate.csv 删除公告 (SKIP — 已由 43a80e8 在 walpurgis 中恢复)
KARATE_DELETION_NOTICE = CompatNotice(
    component="datasets/karate.csv",
    status=NoticeStatus.RESTORED,
    message=(
        "datasets/karate.csv was deleted in upstream 3f8dddf "
        "but restored by upstream 43a80e8. "
        "In walpurgis, the file lives at "
        "src/walpurgis/datasets/benchmark_graphs/karate.csv "
        "and was migrated by commit 43a80e8."
    ),
    upstream_commit="3f8dddf",
    slated_release=None,
    removed_release=None,
    migrate_to=None,
    extra={
        "walpurgis_path": "src/walpurgis/datasets/benchmark_graphs/karate.csv",
        "restored_by": "43a80e8",
        "skip_reason": (
            "karate.csv 已由 43a80e8 迁移，3f8dddf 的删除无需在 "
            "walpurgis 中重现。"
        ),
    },
)


# ── 全局注册表单例 ────────────────────────────────────────────────────────────

#: 全局兼容性公告注册表（模块加载时预填充）
NOTICE_BOARD = CompatNoticeBoard()
NOTICE_BOARD.register(DGL_REMOVAL_NOTICE)
NOTICE_BOARD.register(KARATE_DELETION_NOTICE)

_dbg(
    "module_load",
    f"upstream_compat_notice 加载完成: {NOTICE_BOARD.summary()}",
)


# ── 集成接口 ──────────────────────────────────────────────────────────────────

def check_dgl_removal(current_release: str = "25.08") -> None:
    """
    检查 cuGraph-DGL 移除状态并按需发出运行时警告。

    供 walpurgis.core.dgl_deprecation 集成调用，
    在 DGL 兼容接口被访问时补充版本时间线信息。

    参数
    ----
    current_release: 当前 walpurgis 版本 (YY.MM)，
                     默认 "25.08"（对应 DGL 实际移除版本）。

    # ── 断点3 ────────────────────────────────────────────────────────────────
    """
    # ── 断点3 ────────────────────────────────────────────────────────────────
    _dbg(
        "check_dgl_removal",
        f"current_release={current_release!r}, "
        f"DGL_REMOVAL_NOTICE={DGL_REMOVAL_NOTICE!r}",
    )

    past_slated = DGL_REMOVAL_NOTICE.is_past_slated(current_release)
    _dbg(
        "check_dgl_removal",
        f"is_past_slated({current_release!r}) = {past_slated}",
    )

    if past_slated:
        NOTICE_BOARD.as_warning("cuGraph-DGL", stacklevel=3)
    else:
        _dbg(
            "check_dgl_removal",
            f"current_release {current_release!r} < slated "
            f"{DGL_REMOVAL_NOTICE.slated_release!r}，不发出 warning",
        )


# ── 公开 API ──────────────────────────────────────────────────────────────────

__all__ = [
    # 枚举
    "NoticeStatus",
    # 数据结构
    "CompatNotice",
    # 注册表
    "CompatNoticeBoard",
    # 预置公告
    "DGL_REMOVAL_NOTICE",
    "KARATE_DELETION_NOTICE",
    # 全局单例
    "NOTICE_BOARD",
    # 集成接口
    "check_dgl_removal",
]
