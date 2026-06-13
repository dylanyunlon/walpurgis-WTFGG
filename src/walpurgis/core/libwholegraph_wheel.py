# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 2dd02f9
# 原标题: feat: add libwholegraph wheel (#182)
# 上游 PR: https://github.com/rapidsai/cugraph-gnn/pull/182
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「愿中国青年都摆脱冷气，只是向上走，不必听自暴自弃者流的话。」
# —— 鲁迅《热风·题辞》
#
# 2dd02f9 做的事，说来简单，实则是一次工程结构上的松绑：
# 将 libwholegraph（.so 共享库）从"编译时绑死在 pylibwholegraph 里"
# 解耦成一个独立的、可单独分发的 Python wheel。
#
# 主要新增内容（21 个文件，482 行增加，66 行删除）：
#   - python/libwholegraph/libwholegraph/__init__.py
#       暴露 __version__、__git_commit__、load_library() 三件套
#   - python/libwholegraph/libwholegraph/load.py
#       实现 load_library()：优先从 wheel 内 lib64/ 目录加载 .so，
#       fallback 到系统路径；加载前先 import libraft / rapids_logger
#       并调用其 load_library()，保证符号依赖顺序正确。
#       使用 RTLD_LOCAL 避免全局符号污染。
#       支持 RAPIDS_LIBWHOLEGRAPH_PREFER_SYSTEM_LIBRARY 环境变量覆盖。
#   - python/libwholegraph/libwholegraph/_version.py
#       自动生成版本字符串（25.10.x 系列）
#   - python/libwholegraph/pyproject.toml
#       独立 wheel 的构建描述，依赖 libraft、rapids-logger
#   - python/libwholegraph/CMakeLists.txt
#       将 wholegraph 编译为共享库 (.so)，不再静态链入 pylibwholegraph
#   - ci/build_wheel_libwholegraph.sh
#       新增独立的 CI 构建脚本
#   - dependencies.yaml
#       新增 libwholegraph wheel 的依赖条目
#
# 上游动机（PR 描述）：
#   避免 RAFT 和 RMM 被重复编译——这两者已分别有各自的 wheel，
#   libwholegraph wheel 可以直接复用，不必再在 pylibwholegraph
#   的构建过程中重走一遍。节省 CI 时间，简化依赖图。
#
# Walpurgis 20% 改写要点：
#   1. WheelLoadStrategy 枚举 — 将"优先系统路径 vs 优先 wheel 路径"
#      的选择逻辑提升为枚举，替代上游裸字符串环境变量比较
#   2. LibraryHandle dataclass — 封装 ctypes.CDLL 句柄 + soname + 来源标记，
#      便于调试时识别库是从 wheel 还是系统路径加载的
#   3. DependencyLoadError 自定义异常 — 替代上游在 load_library() 里
#      裸抛的 OSError，附加"当前已尝试路径列表"字段
#   4. load_with_strategy() 工厂函数 — 统一入口，接受 WheelLoadStrategy，
#      返回 LibraryHandle；WALPURGIS_DEBUG=1 时打印加载路径
#   5. WALPURGIS_DEBUG=1 全链路输出：模块初始化、依赖项加载尝试、
#      最终加载路径确认

from __future__ import annotations

import ctypes
import enum
import os as _os
import sys as _sys
from dataclasses import dataclass, field
from typing import List, Optional

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


_dbg("libwholegraph_wheel", "module init — 迁移自 cugraph-gnn 2dd02f9")

# ---------------------------------------------------------------------------
# 枚举：库加载策略
# ---------------------------------------------------------------------------

class WheelLoadStrategy(enum.Enum):
    """
    控制 libwholegraph.so 的加载路径优先级。

    上游 load.py 通过环境变量 RAPIDS_LIBWHOLEGRAPH_PREFER_SYSTEM_LIBRARY
    的真/假值决定优先级，Walpurgis 将其提升为枚举，避免字符串比较散落各处。
    """

    PREFER_WHEEL = "wheel"       # 默认：优先从 wheel 内 lib64/ 加载
    PREFER_SYSTEM = "system"     # 环境变量置真时：优先从系统路径加载

    @classmethod
    def from_env(cls) -> "WheelLoadStrategy":
        val = _os.getenv("RAPIDS_LIBWHOLEGRAPH_PREFER_SYSTEM_LIBRARY", "false")
        strategy = cls.PREFER_SYSTEM if val.lower() != "false" else cls.PREFER_WHEEL
        _dbg("WheelLoadStrategy", f"env={val!r} → strategy={strategy.value}")
        return strategy


# ---------------------------------------------------------------------------
# 数据类：封装已加载的库句柄
# ---------------------------------------------------------------------------

@dataclass
class LibraryHandle:
    """
    封装 ctypes.CDLL 句柄及其元数据。

    上游直接返回裸 CDLL 对象或 None，调试时难以追溯来源路径。
    Walpurgis 统一包装为 LibraryHandle，source_path 字段标记实际加载路径。
    """

    soname: str
    handle: ctypes.CDLL
    source_path: str                   # 实际加载路径（"<system>" 或绝对路径）
    strategy_used: WheelLoadStrategy

    def __post_init__(self) -> None:
        _dbg(
            "LibraryHandle",
            f"loaded {self.soname!r} from {self.source_path!r} "
            f"via strategy={self.strategy_used.value}",
        )


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class DependencyLoadError(OSError):
    """
    加载 libwholegraph 依赖项失败时抛出。

    上游在 _load_system_installation() 里直接 raise OSError（ctypes.CDLL 失败时），
    Walpurgis 附加"已尝试路径列表"以便快速定位。
    """

    def __init__(self, soname: str, tried_paths: List[str], cause: Optional[Exception] = None):
        self.soname = soname
        self.tried_paths = tried_paths
        msg = (
            f"[Walpurgis:DependencyLoadError] 无法加载 {soname!r}。\n"
            f"已尝试路径: {tried_paths}\n"
            f"原始错误: {cause}"
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# 核心：带策略的加载入口
# ---------------------------------------------------------------------------

#: 上游 load.py 使用 RTLD_LOCAL 避免全局符号污染，此处保持一致
_PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _try_system(soname: str) -> Optional[ctypes.CDLL]:
    """尝试从系统路径 dlopen()；失败返回 None（不抛出）。"""
    try:
        handle = ctypes.CDLL(soname, _PREFERRED_LOAD_FLAG)
        _dbg("_try_system", f"成功: {soname}")
        return handle
    except OSError as exc:
        _dbg("_try_system", f"失败: {soname} — {exc}")
        return None


def _try_wheel(soname: str) -> Optional[ctypes.CDLL]:
    """
    尝试从 wheel 内 lib64/ 目录 dlopen()。

    上游路径拼接逻辑：
        os.path.join(os.path.dirname(__file__), "lib64", soname)
    此处等价实现。
    """
    candidate = _os.path.join(_os.path.dirname(__file__), "lib64", soname)
    if _os.path.isfile(candidate):
        try:
            handle = ctypes.CDLL(candidate, _PREFERRED_LOAD_FLAG)
            _dbg("_try_wheel", f"成功: {candidate}")
            return handle
        except OSError as exc:
            _dbg("_try_wheel", f"文件存在但加载失败: {candidate} — {exc}")
    else:
        _dbg("_try_wheel", f"文件不存在: {candidate}")
    return None


def _load_upstream_deps() -> None:
    """
    加载 libwholegraph 的上游运行时依赖（libraft、rapids_logger）。

    上游 load_library() 在 try/except ModuleNotFoundError 内处理：
    若依赖由 conda 包满足（无 Python 模块），静默跳过；
    若依赖由 wheel 满足，则必须先调用其 load_library()。
    """
    _dbg("_load_upstream_deps", "尝试加载 libraft / rapids_logger")
    try:
        import libraft
        import rapids_logger  # type: ignore[import]

        libraft.load_library()
        rapids_logger.load_library()
        _dbg("_load_upstream_deps", "libraft + rapids_logger 加载完成（wheel 路径）")
    except ModuleNotFoundError:
        _dbg(
            "_load_upstream_deps",
            "ModuleNotFoundError — 假设依赖由 conda 包满足，继续加载",
        )


def load_with_strategy(
    soname: str = "libwholegraph.so",
    strategy: Optional[WheelLoadStrategy] = None,
) -> LibraryHandle:
    """
    按指定策略加载 libwholegraph 共享库，返回 LibraryHandle。

    上游 load_library() 不返回任何值（副作用式加载），
    Walpurgis 改为返回 LibraryHandle，便于调试与审计。

    加载顺序：
      PREFER_WHEEL  → 先尝试 wheel lib64/，再 fallback 系统路径
      PREFER_SYSTEM → 先尝试系统路径，再 fallback wheel lib64/

    Parameters
    ----------
    soname:
        共享库文件名，默认 "libwholegraph.so"
    strategy:
        加载策略；None 时从环境变量自动推断
    """
    if strategy is None:
        strategy = WheelLoadStrategy.from_env()

    _load_upstream_deps()

    tried: List[str] = []

    if strategy == WheelLoadStrategy.PREFER_WHEEL:
        order = [("wheel", _try_wheel), ("system", _try_system)]
    else:
        order = [("system", _try_system), ("wheel", _try_wheel)]

    for label, loader in order:
        handle = loader(soname)
        tried.append(label)
        if handle is not None:
            source = (
                _os.path.join(_os.path.dirname(__file__), "lib64", soname)
                if label == "wheel"
                else "<system>"
            )
            return LibraryHandle(
                soname=soname,
                handle=handle,
                source_path=source,
                strategy_used=strategy,
            )

    raise DependencyLoadError(
        soname=soname,
        tried_paths=tried,
        cause=OSError(f"ctypes.CDLL('{soname}') 在所有路径均失败"),
    )


# ---------------------------------------------------------------------------
# 兼容上游 API：load_library() 无返回值版本
# ---------------------------------------------------------------------------

def load_library() -> None:
    """
    兼容上游 libwholegraph.load.load_library() 接口。

    上游不返回任何值；Walpurgis 内部调用 load_with_strategy()
    但丢弃返回值，保持对上游调用方的接口兼容。
    """
    _dbg("load_library", "调用 load_with_strategy() (兼容模式)")
    try:
        load_with_strategy()
    except DependencyLoadError as exc:
        _dbg("load_library", f"加载失败: {exc}")
        # 上游遇到系统安装失败时直接让 OSError 向上冒泡
        raise OSError(str(exc)) from exc


# ---------------------------------------------------------------------------
# 模块级信息导出
# ---------------------------------------------------------------------------

#: 上游迁移信息，供审计工具查询
UPSTREAM_COMMIT = "2dd02f9"
UPSTREAM_REPO = "rapidsai/cugraph-gnn"
UPSTREAM_PR = 182
UPSTREAM_TITLE = "feat: add libwholegraph wheel"
MIGRATION_TARGET = "walpurgis/core/libwholegraph_wheel.py"
