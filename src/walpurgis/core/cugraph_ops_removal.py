# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit d38b832
# 原标题: remove dependency on cugraph-ops (#99)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「不满是向上的车轮，能够载着不自满的人类，向人道前进。」
# —— 鲁迅《热风·随感录》
#
# d38b832 做了一件积累已久的事：把 cugraph-ops (pylibcugraphops) 整个踢出依赖树。
#
# 主要删除内容：
#   - cugraph_pyg/nn/conv/ 下所有依赖 pylibcugraphops 的卷积层
#     (GATConv, GATv2Conv, HeteroGATConv, RGCNConv, SAGEConv, TransformerConv)
#   - cugraph_dgl/nn/conv/ 下的同类层（GATConv, GATv2Conv, RelGraphConv,
#     SAGEConv, TransformerConv）
#   - pylibwholegraph/torch/cugraphops/ 下的包装层
#   - gnn_model.py 中 "cugraph" backend 分支
#
# 主要修改内容：
#   - 示例 graph_sage_mg.py / graph_sage_sg.py 从 CuGraphSAGEConv 切换到
#     torch_geometric.nn.SAGEConv，forward 签名从 (x, edge_csc, size) 改为 (x, edge)
#   - BaseGraph.get_cugraph_ops_CSC / get_cugraph_ops_HeteroCSC 方法删除
#   - cugraph_pyg/__init__.py 移除 nn 导入，cugraph_dgl/nn/conv/__init__.py 清空
#
# Walpurgis 20% 改写要点：
#   1. ConvBackendEnum 枚举 — 替代上游 gnn_model.py 里的裸字符串 framework_name 比较，
#      删除 "cugraph" 分支后只剩 "pyg" / "dgl"，用枚举防止拼写错误
#   2. ConvLayerFactory 类 — 封装上游 get_gnn_layers() 中各 framework 的层构建逻辑，
#      删除 cugraph 分支后的代码更清晰
#   3. OpsDeprecationError 自定义异常 — 替代上游删除后隐式的 ImportError，
#      若旧代码仍尝试 import cugraph_pyg.nn，给出明确迁移提示
#   4. WALPURGIS_DEBUG=1 时输出当前使用的 backend 和层配置摘要

from __future__ import annotations

import enum
import os as _os
import sys as _sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 枚举：支持的 GNN 框架（d38b832 之后 "cugraph" 已被移除）
# ---------------------------------------------------------------------------

class ConvBackend(enum.Enum):
    """
    GNN 卷积层后端枚举。

    上游 gnn_model.py 用裸字符串 "pyg" / "dgl" / "cugraph" 判断分支，
    d38b832 删除了 "cugraph" 分支。Walpurgis 迁移将剩余分支提升为枚举，
    防止拼写错误，并使 "cugraph" 走到 REMOVED 分支时给出明确报错。
    """

    PYG = "pyg"
    DGL = "dgl"
    CUGRAPH_REMOVED = "cugraph"  # d38b832 已删除，保留为哨兵

    @classmethod
    def from_str(cls, name: str) -> "ConvBackend":
        try:
            return cls(name.lower())
        except ValueError:
            raise ValueError(
                f"[Walpurgis:ConvBackend] 未知 backend '{name}'。"
                f"可用: {[b.value for b in cls if b != cls.CUGRAPH_REMOVED]}"
            )

    def assert_available(self) -> None:
        if self == ConvBackend.CUGRAPH_REMOVED:
            raise OpsRemovedError(
                "cugraph-ops (pylibcugraphops) 依赖已在 d38b832 移除。\n"
                "请将模型代码迁移到 torch_geometric.nn 或 dgl.nn 层。\n"
                "参考: https://github.com/rapidsai/cugraph-gnn/pull/99"
            )


# ---------------------------------------------------------------------------
# 自定义异常：替代删除后的隐式 ImportError
# ---------------------------------------------------------------------------

class OpsRemovedError(RuntimeError):
    """
    当代码尝试使用已在 d38b832 移除的 cugraph-ops 层时抛出。

    上游删除后：旧代码 `from cugraph_pyg.nn import SAGEConv` 会得到 ImportError，
    调试信息不足。Walpurgis 用此异常给出明确的迁移指引。
    """
    pass


# ---------------------------------------------------------------------------
# 数据类：GNN 层配置
# ---------------------------------------------------------------------------

@dataclass
class GNNLayerConfig:
    """
    封装单层 GNN 的超参数配置。

    上游 get_gnn_layers() 接收散装参数；Walpurgis 提取为数据类，
    便于序列化保存和调试打印。
    """

    in_channels: int
    out_channels: int
    model_type: str        # "sage" / "gat"
    backend: ConvBackend
    num_heads: int = 1     # 仅 GAT 使用
    mean_output: bool = False  # 仅 GAT 使用

    def __post_init__(self) -> None:
        self.backend.assert_available()
        _dbg(
            "GNNLayerConfig",
            f"model_type={self.model_type}  backend={self.backend.value}  "
            f"in={self.in_channels}  out={self.out_channels}  heads={self.num_heads}",
        )


# ---------------------------------------------------------------------------
# ConvLayerFactory：封装 d38b832 之后的层构建逻辑
# ---------------------------------------------------------------------------

class ConvLayerFactory:
    """
    GNN 卷积层工厂，对应上游 gnn_model.py 中 get_gnn_layers() 的 layer 构建部分。

    d38b832 之前 get_gnn_layers 有三个 framework 分支：pyg / dgl / cugraph。
    d38b832 删除了 cugraph 分支，只剩 pyg 和 dgl。

    Walpurgis 改写：提取为工厂类，每个 backend 一个静态方法，
    删除 cugraph 路径后逻辑更清晰，OpsRemovedError 给出迁移提示。
    """

    @staticmethod
    def build(cfg: GNNLayerConfig) -> "torch.nn.Module":
        """
        根据 GNNLayerConfig 构建单个卷积层。

        Returns
        -------
        torch.nn.Module
        """
        _dbg("ConvLayerFactory.build", f"cfg={cfg}")

        if cfg.backend == ConvBackend.PYG:
            return ConvLayerFactory._build_pyg(cfg)
        elif cfg.backend == ConvBackend.DGL:
            return ConvLayerFactory._build_dgl(cfg)
        else:
            cfg.backend.assert_available()  # 触发 OpsRemovedError

    @staticmethod
    def _build_pyg(cfg: GNNLayerConfig) -> "torch.nn.Module":
        """
        对应上游 d38b832 后 pyg 路径。

        原代码（d38b832 后，graph_sage_mg.py）：
            from torch_geometric.nn import SAGEConv
            conv = SAGEConv(in_channels, hidden_channels)
        """
        try:
            import torch_geometric.nn as pyg_nn
        except ImportError as exc:
            raise ImportError(
                "[Walpurgis:ConvLayerFactory] torch_geometric 不可用: " + str(exc)
            ) from exc

        model_type = cfg.model_type.lower()
        _dbg("ConvLayerFactory._build_pyg", f"model_type={model_type}")

        if model_type == "sage":
            return pyg_nn.SAGEConv(cfg.in_channels, cfg.out_channels)
        elif model_type == "gat":
            concat = not cfg.mean_output
            return pyg_nn.GATConv(
                cfg.in_channels,
                cfg.out_channels,
                heads=cfg.num_heads,
                concat=concat,
            )
        else:
            raise ValueError(
                f"[Walpurgis:ConvLayerFactory] PyG backend 未知 model_type='{model_type}'。"
                f"可用: 'sage', 'gat'"
            )

    @staticmethod
    def _build_dgl(cfg: GNNLayerConfig) -> "torch.nn.Module":
        """
        对应上游 d38b832 后 dgl 路径（gnn_model.py 中 dgl 分支保留）。
        """
        try:
            import dgl.nn as dgl_nn
        except ImportError as exc:
            raise ImportError(
                "[Walpurgis:ConvLayerFactory] dgl 不可用: " + str(exc)
            ) from exc

        model_type = cfg.model_type.lower()
        _dbg("ConvLayerFactory._build_dgl", f"model_type={model_type}")

        if model_type == "sage":
            return dgl_nn.SAGEConv(cfg.in_channels, cfg.out_channels, aggregator_type="mean")
        elif model_type == "gat":
            concat = not cfg.mean_output
            return dgl_nn.GATConv(
                cfg.in_channels,
                cfg.out_channels,
                num_heads=cfg.num_heads,
            )
        else:
            raise ValueError(
                f"[Walpurgis:ConvLayerFactory] DGL backend 未知 model_type='{model_type}'。"
                f"可用: 'sage', 'gat'"
            )


# ---------------------------------------------------------------------------
# 迁移提示模块：cugraph_pyg.nn 旧 import 路径的替代
# ---------------------------------------------------------------------------

def warn_ops_removed(old_symbol: str, new_module: str, new_symbol: str) -> None:
    """
    当检测到代码仍尝试使用已删除的 cugraph-ops 层时打印迁移提示。

    上游 d38b832 之前：
        from cugraph_pyg.nn import SAGEConv  # 依赖 pylibcugraphops
    d38b832 之后：
        from torch_geometric.nn import SAGEConv  # 纯 PyG，无 cugraph-ops

    用法：
        warn_ops_removed("cugraph_pyg.nn.SAGEConv", "torch_geometric.nn", "SAGEConv")
    """
    import warnings

    msg = (
        f"[Walpurgis] {old_symbol} 已在 d38b832 移除（删除 cugraph-ops 依赖）。\n"
        f"请改用: from {new_module} import {new_symbol}\n"
        f"参考 PR: https://github.com/rapidsai/cugraph-gnn/pull/99"
    )
    _dbg("warn_ops_removed", msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


# 常见迁移映射：旧符号 → (新 module, 新符号)
_OPS_MIGRATION_MAP = {
    "cugraph_pyg.nn.SAGEConv": ("torch_geometric.nn", "SAGEConv"),
    "cugraph_pyg.nn.GATConv": ("torch_geometric.nn", "GATConv"),
    "cugraph_pyg.nn.GATv2Conv": ("torch_geometric.nn", "GATv2Conv"),
    "cugraph_pyg.nn.HeteroGATConv": ("torch_geometric.nn", "HGTConv"),
    "cugraph_pyg.nn.RGCNConv": ("torch_geometric.nn", "RGCNConv"),
    "cugraph_pyg.nn.TransformerConv": ("torch_geometric.nn", "TransformerConv"),
    "cugraph_dgl.nn.GATConv": ("dgl.nn", "GATConv"),
    "cugraph_dgl.nn.SAGEConv": ("dgl.nn", "SAGEConv"),
    "cugraph_dgl.nn.RelGraphConv": ("dgl.nn", "RelGraphConv"),
    "cugraph_dgl.nn.TransformerConv": ("dgl.nn", "DotGatConv"),
}


def get_migration_hint(old_path: str) -> str:
    """
    给出旧 cugraph-ops 符号的迁移建议。

    Parameters
    ----------
    old_path : str
        例如 "cugraph_pyg.nn.SAGEConv"

    Returns
    -------
    str
        迁移提示字符串
    """
    if old_path in _OPS_MIGRATION_MAP:
        new_mod, new_sym = _OPS_MIGRATION_MAP[old_path]
        return (
            f"'{old_path}' 已移除（d38b832）。\n"
            f"迁移至: from {new_mod} import {new_sym}"
        )
    return (
        f"'{old_path}' 可能已在 d38b832 移除。"
        f"请查阅: https://github.com/rapidsai/cugraph-gnn/pull/99"
    )


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.environ["WALPURGIS_DEBUG"] = "1"
    print("=== 自测 cugraph_ops_removal.py (migrate d38b832) ===\n")

    # --- 测试 1: ConvBackend 枚举 ---
    b = ConvBackend.from_str("pyg")
    assert b == ConvBackend.PYG
    b.assert_available()  # 不应抛异常
    print("[OK] 测试1: ConvBackend.PYG available")

    # --- 测试 2: cugraph backend 触发 OpsRemovedError ---
    cug = ConvBackend.CUGRAPH_REMOVED
    try:
        cug.assert_available()
        assert False, "应抛 OpsRemovedError"
    except OpsRemovedError as e:
        assert "d38b832" in str(e)
        print("[OK] 测试2: CUGRAPH_REMOVED 抛 OpsRemovedError")

    # --- 测试 3: get_migration_hint ---
    hint = get_migration_hint("cugraph_pyg.nn.SAGEConv")
    assert "torch_geometric.nn" in hint
    print(f"[OK] 测试3: get_migration_hint\n  {hint}")

    # --- 测试 4: 未知 backend ---
    try:
        ConvBackend.from_str("tensorflow")
        assert False
    except ValueError as e:
        assert "未知" in str(e)
        print("[OK] 测试4: 未知 backend ValueError")

    # --- 测试 5: GNNLayerConfig cugraph 分支触发异常 ---
    try:
        GNNLayerConfig(
            in_channels=64,
            out_channels=32,
            model_type="sage",
            backend=ConvBackend.CUGRAPH_REMOVED,
        )
        assert False
    except OpsRemovedError:
        print("[OK] 测试5: GNNLayerConfig cugraph 触发 OpsRemovedError")

    print("\n=== 全部自测通过 ===")
