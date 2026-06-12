# SPDX-FileCopyrightText: Copyright (c) 2025 Walpurgis-WTFGG contributors
# SPDX-License-Identifier: Apache-2.0
#
# Migrated from upstream cugraph-gnn commit 3d4c449
# (rapidsai/cugraph-gnn: Update rapids-dependency-file-generator)
#
# 上游语境：libwholegraph/load.py 原先在动态加载 libwholegraph 前
# 依次 import libraft + rapids_logger 并调用各自的 load_library()。
# 3d4c449 将 rapids_logger 整行删除——鲁迅曾云：
#   「不在沉默中爆发，就在沉默中灭亡。」
# rapids_logger 无声地被从依赖树上斩断，既无讣告，亦无遗嘱，
# 只剩 pyproject.toml 里一行被划掉的墓志铭。
#
# 本文件将该「斩断」行为抽象为可复用、可审计的运行时策略，
# 并附 WALPURGIS_DEBUG=1 断点供调试。

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_DEBUG = os.getenv("WALPURGIS_DEBUG", "0") == "1"


def _bp(tag: str, **ctx) -> None:
    """条件断点——鲁迅式：「沉默到此为止」。
    WALPURGIS_DEBUG=1 时打印上下文并触发 pdb。
    """
    if not _DEBUG:
        return
    print(f"\n[WALPURGIS_DEBUG] ── 断点 {tag} ──", file=sys.stderr)
    for k, v in ctx.items():
        print(f"  {k} = {v!r}", file=sys.stderr)
    import pdb  # noqa: T100
    pdb.set_trace()  # noqa: T100


# ──────────────────────────────────────────────
# 1. 依赖探针状态枚举
# ──────────────────────────────────────────────
class ProbeStatus(Enum):
    """上游只有 try/except ModuleNotFoundError，我们把结果语义化。

    鲁迅：「凡事都有两面，不肯定，不否定，是滑头的做法。」
    此处一律明确。
    """
    PRESENT = auto()       # import 成功且 load_library() 可调
    ABSENT = auto()        # ModuleNotFoundError
    BROKEN = auto()        # import 成功但 load_library() 抛出异常
    SUPERSEDED = auto()    # 曾经存在，已被上游主动移除（本 commit 的核心）


# ──────────────────────────────────────────────
# 2. 依赖探针结果
# ──────────────────────────────────────────────
@dataclass
class ProbeResult:
    """单个运行时依赖的探测结果。

    上游只 import + call，无任何结构化记录。
    鲁迅：「我向来不惮以最坏的恶意来推测中国人」——此处则以最坏的可能
    推测每个依赖：它随时可能消失。ProbeResult 将消失记录在案。
    """
    name: str
    status: ProbeStatus
    version: Optional[str] = None
    error: Optional[str] = None
    # 若 status == SUPERSEDED，记录是哪个上游 commit 决定的
    superseded_by_commit: Optional[str] = None

    def is_loadable(self) -> bool:
        """可安全调用 load_library() 的充要条件。"""
        return self.status == ProbeStatus.PRESENT

    def emit_warning(self) -> None:
        """向 logging 发出人类可读警告。"""
        if self.status == ProbeStatus.ABSENT:
            logger.debug("%s 未安装（conda 路径可能已满足，跳过）", self.name)
        elif self.status == ProbeStatus.BROKEN:
            logger.warning("%s import 成功但 load_library() 失败: %s", self.name, self.error)
        elif self.status == ProbeStatus.SUPERSEDED:
            logger.info(
                "%s 已被上游 commit %s 从依赖树中主动移除，本运行不加载",
                self.name,
                self.superseded_by_commit or "unknown",
            )


# ──────────────────────────────────────────────
# 3. 探针工厂
# ──────────────────────────────────────────────
def _probe_module(
    name: str,
    *,
    superseded_by: Optional[str] = None,
    load_attr: str = "load_library",
) -> ProbeResult:
    """动态探测一个 Python 依赖包是否可用并可调用其加载函数。

    鲁迅：「真的勇士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。」
    此处的「惨淡」是 ModuleNotFoundError；「鲜血」是 OSError from ctypes.

    参数
    ----
    name          : 包名（e.g. "libraft", "rapids_logger"）
    superseded_by : 若非 None，表示此包已被该 commit SHA 移除，
                    直接返回 SUPERSEDED，不做任何 import
    load_attr     : 包内负责加载 DSO 的函数名（默认 "load_library"）
    """
    _bp("probe_start", name=name, superseded_by=superseded_by)  # 断点 ①

    # 3a. 上游主动移除：直接标记，不尝试 import
    if superseded_by is not None:
        result = ProbeResult(
            name=name,
            status=ProbeStatus.SUPERSEDED,
            superseded_by_commit=superseded_by,
        )
        result.emit_warning()
        _bp("probe_superseded", result=result)  # 断点 ②
        return result

    # 3b. 尝试 import
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError:
        result = ProbeResult(name=name, status=ProbeStatus.ABSENT)
        result.emit_warning()
        return result

    # 3c. 获取版本（best-effort）
    version = getattr(mod, "__version__", None) or getattr(mod, "version", None)

    # 3d. 尝试调用 load_library()
    loader: Optional[Callable] = getattr(mod, load_attr, None)
    if loader is None:
        # 有 module 但无 load_library — 视为 BROKEN
        return ProbeResult(
            name=name,
            status=ProbeStatus.BROKEN,
            version=version,
            error=f"attribute '{load_attr}' not found",
        )

    try:
        loader()
        _bp("probe_loaded", name=name, version=version)  # 断点 ③
        return ProbeResult(name=name, status=ProbeStatus.PRESENT, version=version)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            name=name,
            status=ProbeStatus.BROKEN,
            version=version,
            error=str(exc),
        )


# ──────────────────────────────────────────────
# 4. 3d4c449 移除策略：依赖加载清单
# ──────────────────────────────────────────────
@dataclass
class WholegraphDepPolicy:
    """libwholegraph 前置依赖加载策略，编码 3d4c449 的决策。

    上游原始逻辑（3d4c449 之前）：
        import libraft; libraft.load_library()
        import rapids_logger; rapids_logger.load_library()   # ← 3d4c449 删除

    鲁迅：「中国人失掉自信力了吗」——rapids_logger 失掉了它在依赖图中的位置。
    本类将「失掉」这件事明确记录，而非让它在 diff 里悄悄消失。
    """
    # 仍需加载的依赖：libraft（上游保留）
    required: list[str] = field(default_factory=lambda: ["libraft"])
    # 已被 3d4c449 移除的依赖：不再加载，但记录在案
    superseded: dict[str, str] = field(
        default_factory=lambda: {"rapids_logger": "3d4c449"}
    )

    def probe_all(self) -> list[ProbeResult]:
        """按策略探测所有依赖，返回完整探测报告。"""
        _bp("policy_probe_all", required=self.required, superseded=list(self.superseded))  # 断点 ④
        results: list[ProbeResult] = []

        for name in self.required:
            results.append(_probe_module(name))

        for name, commit in self.superseded.items():
            results.append(_probe_module(name, superseded_by=commit))

        _bp("policy_probe_done", results=[(r.name, r.status.name) for r in results])  # 断点 ⑤
        return results

    def load_required(self) -> list[ProbeResult]:
        """仅加载 required 列表，superseded 仅记录不加载。

        对应 load_library() 入口：调用此方法即可完整复现 3d4c449 后行为。
        """
        return self.probe_all()


# ──────────────────────────────────────────────
# 5. 审计器：检查残留引用
# ──────────────────────────────────────────────
@dataclass
class RapidsLoggerAudit:
    """扫描源文件，确认 rapids_logger 已彻底清除。

    鲁迅：「横眉冷对千夫指」——对任何残留的 rapids_logger import 横眉冷对。
    """
    target_string: str = "rapids_logger"

    def scan_file(self, path: str) -> list[tuple[int, str]]:
        """返回含 target_string 的 (行号, 行内容) 列表。"""
        _bp("audit_scan", path=path, target=self.target_string)  # 断点 ⑥
        hits: list[tuple[int, str]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if self.target_string in line:
                        hits.append((lineno, line.rstrip()))
        except (OSError, UnicodeDecodeError):
            pass
        return hits

    def assert_clean(self, path: str) -> None:
        """若 path 中仍有 rapids_logger 引用则 raise AssertionError。"""
        hits = self.scan_file(path)
        if hits:
            detail = "\n".join(f"  L{ln}: {txt}" for ln, txt in hits)
            raise AssertionError(
                f"3d4c449 移除的 '{self.target_string}' 在 {path} 中仍有残留:\n{detail}"
            )


# ──────────────────────────────────────────────
# 6. 自测（python -m 运行时执行）
# ──────────────────────────────────────────────
def _self_test() -> None:
    """鲁迅：「真的猛士，将更奋然而前行。」——自测是前行前的确认。"""
    _bp("self_test_start")  # 断点 ⑦

    policy = WholegraphDepPolicy()
    assert policy.required == ["libraft"], "required list mismatch"
    assert "rapids_logger" in policy.superseded, "superseded entry missing"
    assert policy.superseded["rapids_logger"] == "3d4c449", "commit SHA mismatch"

    # ProbeStatus 语义验证
    r_superseded = _probe_module("rapids_logger", superseded_by="3d4c449")
    assert r_superseded.status == ProbeStatus.SUPERSEDED
    assert not r_superseded.is_loadable()
    assert r_superseded.superseded_by_commit == "3d4c449"

    r_absent = _probe_module("__nonexistent_pkg_xyz__")
    assert r_absent.status == ProbeStatus.ABSENT
    assert not r_absent.is_loadable()

    # 审计器验证
    import tempfile, os  # noqa: E401
    audit = RapidsLoggerAudit()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write("import rapids_logger\nrapids_logger.load_library()\n")
        tmppath = tf.name
    try:
        hits = audit.scan_file(tmppath)
        assert len(hits) == 2, f"expected 2 hits, got {hits}"
        try:
            audit.assert_clean(tmppath)
            raise AssertionError("assert_clean should have raised")
        except AssertionError as exc:
            assert "残留" in str(exc), "wrong error message"
    finally:
        os.unlink(tmppath)

    print("[3d4c449 self_test] ALL PASS ✓", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _self_test()
