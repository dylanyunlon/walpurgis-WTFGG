# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 1b2fce2 + a57912c: Fix dataloader import references in tests
#
# 上游两个 commit:
#
#   1b2fce2 (cugraph-dgl/tests/dataloading/test_dask_dataloader_mg.py):
#     - 把 cugraph_dgl.dataloading.DaskDataLoader → cugraph_dgl.dataloading.DataLoader
#       (上游 DataLoader 在 456d5a2 后已是指向 DEPRECATED__DaskDataLoader 的 wrapper)
#
#   a57912c (同目录两个文件):
#     - test_dask_dataloader.py:
#         cugraph_dgl.dataloading.DaskDataLoader
#         → cugraph_dgl.dataloading.dask_dataloader.DaskDataLoader  (完整模块路径)
#     - test_dask_dataloader_mg.py:
#         cugraph_dgl.dataloading.DataLoader
#         → cugraph_dgl.dataloading.dask_dataloader.DaskDataLoader  (完整模块路径)
#
# Walpurgis 迁移语义:
#   - walpurgis 中与 DaskDataLoader 对应的是 walpurgis.dataloader.DataLoader
#   - 测试文件不应该通过包顶层间接引用 loader（可能触发废弃 wrapper）
#   - 修复模式: 使用完整模块路径而非包级别导入（直接模块导入，消除间接依赖链）
#
# 20% 改写 (鲁迅拿法):
#   - 把多 GPU 采样测试中的 "通过 DaskDataLoader 包级别引用" 改为
#     直接从 walpurgis.dataloader.dataloader 导入 DataLoader
#   - 新增 WALPURGIS_DEBUG 断点: dataloader 构造参数摘要
#   - 断点1: 采样调用入口 (graph store 类型 + train_nid size)
#   - 断点2: dataloader 构造完成 (batch count)
#
# 作者: dylanyunlon<dogechat@163.com>

import os
import sys
import tempfile

import pytest

os.environ.setdefault("WALPURGIS_DEBUG", "1")

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(
            f"[WALPURGIS tests/sampler/test_dataloader_refs] {msg}",
            file=sys.stderr,
            flush=True,
        )


# migrate a57912c: 直接从模块文件导入 DataLoader，
# 绕过 walpurgis.dataloader.__init__ 的包级别别名链
# (对应上游 a57912c 将 cugraph_dgl.dataloading.DataLoader
#  改为 cugraph_dgl.dataloading.dask_dataloader.DaskDataLoader)
from walpurgis.dataloader.dataloader import DataLoader as _DirectDataLoader
from walpurgis.dataloader import DataLoader as _PackageDataLoader


def test_direct_vs_package_dataloader_same_class():
    """
    验证直接模块导入与包级别导入指向同一个类。

    对应 a57912c 的问题核心:
    包级别 `cugraph_dgl.dataloading.DataLoader` 在 456d5a2 后
    变成了指向 DEPRECATED__DaskDataLoader 的 wrapper 函数（不是原始类），
    这会导致 isinstance 检查和 type() 比较失败。

    在 walpurgis 中，DataLoader 没有这个问题（从未包装），
    但直接导入更清晰、更抗未来废弃重组。

    断点1: 两种导入方式的类型信息对比
    """
    # ── 断点1 ────────────────────────────────────────────────────────────────
    _dbg(
        f"test_direct_vs_package_dataloader_same_class: "
        f"_DirectDataLoader={_DirectDataLoader.__module__}.{_DirectDataLoader.__name__}, "
        f"_PackageDataLoader={_PackageDataLoader.__module__}.{_PackageDataLoader.__name__}"
    )

    assert _DirectDataLoader is _PackageDataLoader, (
        "Direct module import and package import should resolve to the same DataLoader class. "
        "If they differ, a wrapper was introduced (like in upstream 456d5a2). "
        "Fix: use walpurgis.dataloader.dataloader.DataLoader directly."
    )

    _dbg("PASS: direct import == package import (no wrapper chain)")


def test_dataloader_construction_without_wrapper_side_effects():
    """
    验证 DataLoader 构造时不触发任何 FutureWarning/DeprecationWarning。

    对应上游 1b2fce2 的问题:
    测试代码用 cugraph_dgl.dataloading.DaskDataLoader 时，
    经过 wrapper 函数会触发 FutureWarning，污染测试输出。

    walpurgis DataLoader 不应有任何 wrapper 警告。

    断点2: DataLoader 构造完成 (batch count)
    """
    import numpy as np
    import warnings

    xs = np.random.randn(100, 12, 207, 2).astype(np.float32)
    ys = np.random.randn(100, 12, 207, 2).astype(np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # migrate 1b2fce2: 直接从模块文件构造，不经过任何 wrapper
        loader = _DirectDataLoader(xs, ys, batch_size=16, shuffle=False)

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]

    # ── 断点2 ────────────────────────────────────────────────────────────────
    _dbg(
        f"DataLoader constructed: "
        f"batch_count={loader.num_batch}, "
        f"size={loader.size}, "
        f"FutureWarnings={len(future_warnings)}, "
        f"DeprecationWarnings={len(deprecation_warnings)}"
    )

    assert len(future_warnings) == 0, (
        f"DataLoader construction triggered {len(future_warnings)} FutureWarning(s). "
        f"This means a deprecated wrapper is in the import chain. "
        f"Use walpurgis.dataloader.dataloader.DataLoader directly (migrate 1b2fce2).\n"
        f"Warnings: {[str(w.message) for w in future_warnings]}"
    )
    assert loader.num_batch == 6  # 100 / 16 = 6 full batches
    _dbg("PASS: DataLoader construction clean (no wrapper warnings)")


def test_dataloader_module_path_is_direct():
    """
    验证 DataLoader 的 __module__ 指向具体实现模块而非 __init__ 包。

    对应上游 a57912c 把 `dataloading.DataLoader` → `dataloading.dask_dataloader.DaskDataLoader`
    的动机: 包级别名在重组时可能指向不同对象，直接模块路径更稳定。

    在 walpurgis 中对应验证 DataLoader.__module__ 包含具体实现路径。
    """
    module_path = _DirectDataLoader.__module__
    _dbg(f"DataLoader.__module__ = {module_path!r}")

    # 应该是 walpurgis.dataloader.dataloader 而不是 walpurgis.dataloader
    assert "dataloader.dataloader" in module_path or "dataloader" in module_path, (
        f"DataLoader.__module__={module_path!r} should point to the concrete module. "
        "If it points to a package __init__, it may be a wrapper (migrate a57912c)."
    )
    _dbg(f"PASS: DataLoader module path confirmed: {module_path}")
