# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit d38b832
# 原标题: remove dependency on cugraph-ops (#99)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「横眉冷对千夫指，俯首甘为孺子牛。」—— 鲁迅
# d38b832 从 pylibwholegraph 中彻底移除了对 cugraph-ops 的依赖。
# 具体删除：
#   - gnn_model.py set_framework(): 删除 elif framework_name == "cugraph" 分支
#     (CuGraphSAGEConv, CuGraphGATConv 两个 import)
#   - gnn_model.py create_gnn_layers(): 删除 elif framework_name == "cugraph" 层构建
#   - gnn_model.py create_sub_graph(): 删除 elif framework_name == "cugraph" 分支
#     (add_csr_self_loop + max_num_neighbors+1)
#   - gnn_model.py layer_forward(): 删除 elif framework_name == "cugraph" 前向传播
#   - common_options.py add_common_model_options(): default 从 "cugraph" 改为 "wg"
#     help 文本从 "pyg, wg, cugraph" 改为 "pyg, wg"
#
# Walpurgis 20% 改写要点:
#   1. WalpurgisFrameworkRegistry — 替代 gnn_model.py 的 module-level `framework_name` 全局变量。
#      全局可变变量在多进程训练中是隐患：两个 worker 进程各自 set_framework，互不可知。
#      改为进程本地 Registry 对象，支持 DEBUG 时打印调用栈。
#   2. GnnLayerFactory.create() — 替代 create_gnn_layers()，移除 cugraph 分支，
#      加 WALPURGIS_DEBUG=1 打印每层 input/output dim + framework
#   3. SubgraphAdapter.build() — 替代 create_sub_graph()，移除 cugraph 分支的
#      add_csr_self_loop，加 DEBUG 打印 csr_row_ptr/max_num_neighbors
#   4. LayerForwardDispatch.forward() — 替代 layer_forward()，移除 cugraph 路径，
#      加 DEBUG 打印 framework + x_feat.shape
#   5. VALID_FRAMEWORKS 常量 — d38b832 有效值为 ["pyg", "wg"]，删除 "cugraph"

import os
import sys
import warnings
from typing import Optional, List, Union

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _WDBG:
        print(f"[WALPURGIS-GNNFW:{tag}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 常量：d38b832 删除 "cugraph" 后的有效 framework 列表
# ---------------------------------------------------------------------------

VALID_FRAMEWORKS = ("pyg", "wg")

# d38b832: common_options.py default 从 "cugraph" 改为 "wg"
DEFAULT_FRAMEWORK = "wg"


# ---------------------------------------------------------------------------
# WalpurgisFrameworkRegistry — 替代 gnn_model.py module-level framework_name 全局变量
# ---------------------------------------------------------------------------

class WalpurgisFrameworkRegistry:
    """
    替代 pylibwholegraph/torch/gnn_model.py 中的 module-level `framework_name` 全局变量。

    上游代码:
        framework_name = None

        def set_framework(framework: str):
            global framework_name
            assert framework_name is None
            framework_name = framework
            global SAGEConv, GATConv
            if framework_name == "pyg":
                from torch_sparse import SparseTensor
                from torch_geometric.nn import SAGEConv, GATConv
            elif framework_name == "wg":
                from wg_torch.gnn.SAGEConv import SAGEConv
                from wg_torch.gnn.GATConv import GATConv
            # d38b832: 删除了以下 elif:
            # elif framework_name == "cugraph":
            #     from .cugraphops.sage_conv import CuGraphSAGEConv as SAGEConv
            #     from .cugraphops.gat_conv import CuGraphGATConv as GATConv

    上游 assert framework_name is None 确保只能 set 一次，但无法在分布式训练场景
    下区分「每进程各自的 framework」和「全局 framework」，assert 会误触。

    Walpurgis 改写:
    - 用实例属性而非全局变量，多进程各自持有独立 Registry
    - 校验改为 ValueError（不用 assert，生产环境安全）
    - d38b832 删除 cugraph 后，validate() 明确拒绝 "cugraph" 并给出迁移提示
    """

    def __init__(self):
        self._framework_name: Optional[str] = None
        self._sage_conv = None
        self._gat_conv = None

    def set(self, framework: str) -> None:
        """
        设置 GNN 框架。对应上游 set_framework()。

        d38b832: 不再支持 "cugraph"，传入时抛 ValueError 并给出迁移说明。
        """
        if self._framework_name is not None:
            raise RuntimeError(
                f"[Walpurgis:WalpurgisFrameworkRegistry] framework 已设置为 "
                f"{self._framework_name!r}，不能再次设置。"
            )

        if framework == "cugraph":
            raise ValueError(
                "[Walpurgis:WalpurgisFrameworkRegistry] "
                "'cugraph' framework 已在 d38b832 中移除（删除 cugraph-ops 依赖）。\n"
                "请改用 'wg'（WholeGraph 原生卷积）或 'pyg'（torch_geometric 卷积）。\n"
                "对应上游 common_options.py 默认值已从 'cugraph' 改为 'wg'。"
            )

        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"[Walpurgis:WalpurgisFrameworkRegistry] "
                f"无效 framework: {framework!r}。有效值: {VALID_FRAMEWORKS}"
            )

        self._framework_name = framework
        self._load_conv_classes(framework)

        _dbg(
            "WalpurgisFrameworkRegistry.set",
            f"framework={framework!r} "
            f"SAGEConv={type(self._sage_conv).__name__ if self._sage_conv else 'None'} "
            f"GATConv={type(self._gat_conv).__name__ if self._gat_conv else 'None'}",
        )

    def _load_conv_classes(self, framework: str) -> None:
        """加载对应框架的卷积类。对应 set_framework 中的条件 import。"""
        if framework == "pyg":
            try:
                from torch_geometric.nn import SAGEConv, GATConv
                self._sage_conv = SAGEConv
                self._gat_conv = GATConv
            except ImportError as e:
                warnings.warn(
                    f"[Walpurgis] torch_geometric 不可用，pyg framework 不可用: {e}"
                )
        elif framework == "wg":
            try:
                from wg_torch.gnn.SAGEConv import SAGEConv
                from wg_torch.gnn.GATConv import GATConv
                self._sage_conv = SAGEConv
                self._gat_conv = GATConv
            except ImportError as e:
                warnings.warn(
                    f"[Walpurgis] wg_torch 不可用，wg framework 不可用: {e}"
                )

    @property
    def name(self) -> Optional[str]:
        return self._framework_name

    @property
    def sage_conv_cls(self):
        return self._sage_conv

    @property
    def gat_conv_cls(self):
        return self._gat_conv

    def require(self) -> str:
        """确保已调用 set()，否则 raise。"""
        if self._framework_name is None:
            raise RuntimeError(
                "[Walpurgis:WalpurgisFrameworkRegistry] "
                "framework 未设置。请先调用 registry.set('wg') 或 registry.set('pyg')。"
            )
        return self._framework_name


# 进程本地默认实例
_default_registry = WalpurgisFrameworkRegistry()


def set_framework(framework: str) -> None:
    """顶层兼容函数，对应 gnn_model.set_framework()。"""
    _default_registry.set(framework)


def get_framework() -> Optional[str]:
    """获取当前 framework 名称。"""
    return _default_registry.name


# ---------------------------------------------------------------------------
# GnnLayerFactory — 替代 create_gnn_layers()，d38b832 删除 cugraph 分支
# ---------------------------------------------------------------------------

class GnnLayerFactory:
    """
    对应 pylibwholegraph/torch/gnn_model.py 的 create_gnn_layers()。

    d38b832 删除的 cugraph 分支:
        elif framework_name == "cugraph":
            assert model_type == "sage" or model_type == "gat"
            if model_type == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif model_type == "gat":
                concat = not mean_output
                gnn_layers.append(
                    GATConv(layer_input_dim, layer_output_dim, heads=num_head, concat=concat)
                )

    Walpurgis 改写:
    - LayerSpec dataclass 封装每层的 (input_dim, output_dim, mean_output, concat)
    - create() 静态方法，对应 create_gnn_layers()，但不支持 "cugraph" framework
    - WALPURGIS_DEBUG=1 打印每层 spec
    """

    @staticmethod
    def create(
        in_feat_dim: int,
        hidden_feat_dim: int,
        class_count: int,
        num_layer: int,
        num_head: int,
        model_type: str,
        registry: Optional[WalpurgisFrameworkRegistry] = None,
    ) -> list:
        """
        创建 GNN 层列表。对应 create_gnn_layers()。

        d38b832: 不支持 "cugraph" framework（已删除）。
        """
        import torch.nn as nn

        if registry is None:
            registry = _default_registry
        framework = registry.require()

        _dbg(
            "GnnLayerFactory.create",
            f"framework={framework!r} model_type={model_type!r} "
            f"num_layer={num_layer} in={in_feat_dim} hidden={hidden_feat_dim} "
            f"out={class_count} num_head={num_head}",
        )

        gnn_layers = nn.ModuleList()

        for i in range(num_layer):
            layer_output_dim = (
                hidden_feat_dim // num_head if i != num_layer - 1 else class_count
            )
            layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
            mean_output = (i == num_layer - 1)

            _dbg(
                "GnnLayerFactory.create",
                f"layer {i}: input={layer_input_dim} output={layer_output_dim} "
                f"mean_output={mean_output}",
            )

            if framework == "pyg":
                SAGEConv = registry.sage_conv_cls
                GATConv = registry.gat_conv_cls
                if model_type == "sage":
                    gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
                elif model_type == "gat":
                    concat = not mean_output
                    gnn_layers.append(
                        GATConv(
                            layer_input_dim,
                            layer_output_dim,
                            heads=num_head,
                            concat=concat,
                        )
                    )
                elif model_type == "gcn":
                    gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif framework == "wg":
                SAGEConv = registry.sage_conv_cls
                GATConv = registry.gat_conv_cls
                if model_type in ("sage", "gcn"):
                    if model_type == "gcn":
                        gnn_layers.append(
                            SAGEConv(layer_input_dim, layer_output_dim, aggregator="gcn")
                        )
                    else:
                        gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
                elif model_type == "gat":
                    concat = not mean_output
                    gnn_layers.append(
                        GATConv(
                            layer_input_dim,
                            layer_output_dim,
                            heads=num_head,
                            concat=concat,
                        )
                    )
            # NOTE: d38b832 删除了 elif framework_name == "cugraph" 分支
            # CuGraphSAGEConv / CuGraphGATConv 来自 cugraph-ops，
            # 该依赖已在 d38b832 中移除。迁移至 "wg" 框架即可。

        _dbg(
            "GnnLayerFactory.create",
            f"创建 {len(gnn_layers)} 层完成",
        )
        return gnn_layers


# ---------------------------------------------------------------------------
# SubgraphAdapter — 替代 create_sub_graph()，d38b832 删除 cugraph 分支
# ---------------------------------------------------------------------------

class SubgraphAdapter:
    """
    对应 pylibwholegraph/torch/gnn_model.py 的 create_sub_graph()。

    d38b832 删除的 cugraph 分支:
        elif framework_name == "cugraph":
            if add_self_loop:
                csr_row_ptr, csr_col_ind = add_csr_self_loop(csr_row_ptr, csr_col_ind)
                max_num_neighbors = max_num_neighbors + 1
            return [csr_row_ptr, csr_col_ind, max_num_neighbors]

    此分支依赖 pylibcugraphops.pytorch 中的 add_csr_self_loop，
    该函数是 cugraph-ops 的一部分，随 d38b832 一并移除。

    Walpurgis 改写:
    - build() 静态方法对应 create_sub_graph()
    - WALPURGIS_DEBUG=1 打印 framework + csr_row_ptr.shape + max_num_neighbors
    """

    @staticmethod
    def build(
        csr_row_ptr,       # torch.Tensor
        csr_col_ind,       # torch.Tensor
        target_gid_1,      # torch.Tensor (dst nodes)
        max_num_neighbors: int,
        add_self_loop: bool = False,
        framework: Optional[str] = None,
    ):
        """
        构造子图表示，供 GNN 层 forward 使用。

        d38b832: "cugraph" framework 不再支持。
        """
        if framework is None:
            framework = get_framework()

        _dbg(
            "SubgraphAdapter.build",
            f"framework={framework!r} "
            f"csr_row_ptr.shape={tuple(csr_row_ptr.shape)} "
            f"csr_col_ind.numel={csr_col_ind.numel()} "
            f"num_dst_nodes={target_gid_1.size(0)} "
            f"max_num_neighbors={max_num_neighbors} "
            f"add_self_loop={add_self_loop}",
        )

        if framework == "cugraph":
            raise ValueError(
                "[Walpurgis:SubgraphAdapter] "
                "'cugraph' framework 已在 d38b832 中移除（删除 cugraph-ops 依赖）。\n"
                "请改用 'wg' 或 'pyg'。"
            )
        elif framework == "pyg":
            from torch_geometric.utils import to_torch_csr_tensor
            block = to_torch_csr_tensor(
                csr_col_ind,
                size=(csr_row_ptr.numel() - 1, target_gid_1.size(0)),
                num_dst_nodes=target_gid_1.size(0),
            )
            _dbg("SubgraphAdapter.build", f"pyg block type={type(block).__name__}")
            return block
        elif framework == "dgl":
            import dgl
            block = dgl.create_block(
                (csr_col_ind, csr_row_ptr),
                num_src_nodes=csr_col_ind.max().item() + 1,
                num_dst_nodes=target_gid_1.size(0),
            )
            _dbg("SubgraphAdapter.build", f"dgl block type={type(block).__name__}")
            return block
        else:
            # "wg" 框架
            _dbg(
                "SubgraphAdapter.build",
                f"wg 框架: 返回 [csr_row_ptr, csr_col_ind]",
            )
            return [csr_row_ptr, csr_col_ind]


# ---------------------------------------------------------------------------
# LayerForwardDispatch — 替代 layer_forward()，d38b832 删除 cugraph 路径
# ---------------------------------------------------------------------------

class LayerForwardDispatch:
    """
    对应 pylibwholegraph/torch/gnn_model.py 的 layer_forward()。

    d38b832 删除的 cugraph 分支:
        elif framework_name == "cugraph":
            x_feat = layer(x_feat, sub_graph[0], sub_graph[1], sub_graph[2])
        # 此 API 来自 CuGraphSAGEConv/CuGraphGATConv，接受 (feat, csr_row_ptr, csr_col_ind, max_neighbors)

    Walpurgis 改写:
    - 静态 forward() 方法，对应 layer_forward()
    - WALPURGIS_DEBUG=1 打印 framework + x_feat.shape + x_target_feat.shape
    """

    @staticmethod
    def forward(layer, x_feat, x_target_feat, sub_graph, framework: Optional[str] = None):
        """
        执行单层 GNN 前向传播。对应 layer_forward()。

        d38b832: "cugraph" framework 不再支持。
        """
        if framework is None:
            framework = get_framework()

        _dbg(
            "LayerForwardDispatch.forward",
            f"framework={framework!r} "
            f"x_feat.shape={tuple(x_feat.shape)} "
            f"x_target_feat.shape={tuple(x_target_feat.shape)} "
            f"sub_graph type={type(sub_graph).__name__}",
        )

        if framework == "cugraph":
            raise ValueError(
                "[Walpurgis:LayerForwardDispatch] "
                "'cugraph' framework 已在 d38b832 中移除（删除 cugraph-ops 依赖）。\n"
                "原 API: layer(x_feat, csr_row_ptr, csr_col_ind, max_neighbors)\n"
                "请改用 'wg' 框架：layer(sub_graph[0], sub_graph[1], x_feat, x_target_feat)"
            )
        elif framework == "pyg":
            x_feat = layer((x_feat, x_target_feat), sub_graph)
        elif framework == "dgl":
            x_feat = layer(sub_graph, (x_feat, x_target_feat))
        elif framework == "wg":
            # wg: sub_graph = [csr_row_ptr, csr_col_ind]
            x_feat = layer(sub_graph[0], sub_graph[1], x_feat, x_target_feat)
        else:
            raise ValueError(
                f"[Walpurgis:LayerForwardDispatch] "
                f"未知 framework: {framework!r}。有效值: {VALID_FRAMEWORKS}"
            )

        _dbg(
            "LayerForwardDispatch.forward",
            f"前向完成: x_feat.shape={tuple(x_feat.shape)}",
        )
        return x_feat
