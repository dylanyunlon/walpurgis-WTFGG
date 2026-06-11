# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit f4ca484
# 原标题: resolve merge conflicts — 引入 cugraph_dgl/typing.py
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「要论中国人，必须不被搽在表面的自欺欺人的脂粉所诧异，
#   却看看他的筋骨和脊梁。」 —— 鲁迅《中国人失掉自信力了吗》
#
# 上游 typing.py 把 TensorType 和 DGLSamplerOutput 并排放，
# 没有解释两者的职责边界。Walpurgis 按用途分组并加注释。

from typing import List, Union, Tuple

from walpurgis.utils.imports import import_optional

# SparseGraph 来自 walpurgis.tensor（已在先前 commit 中迁移）
from walpurgis.tensor.sparse_graph import SparseGraph

import pandas
import numpy
import cupy
import cudf

torch = import_optional("torch")
dgl = import_optional("dgl")


# ---------------------------------------------------------------------------
# 输入张量类型别名
# ---------------------------------------------------------------------------

TensorType = Union[
    "torch.Tensor",
    "cupy.ndarray",
    "numpy.ndarray",
    "cudf.Series",
    "pandas.Series",
    List[int],
]
"""
walpurgis 图操作接受的张量类型。
覆盖 GPU (cupy/cudf/torch-cuda) 和 CPU (numpy/pandas/torch-cpu/list) 两条路径。
"""


# ---------------------------------------------------------------------------
# DGL 采样器输出类型别名
# ---------------------------------------------------------------------------

DGLSamplerOutput = Tuple[
    "torch.Tensor",                    # input_nodes: 重编号后的输入节点
    "torch.Tensor",                    # output_nodes: 重编号后的输出节点
    List[Union["dgl.Block", SparseGraph]],  # blocks: 每跳的消息传递图
]
"""
DGL 采样器标准输出格式：(input_nodes, output_nodes, blocks)。
blocks 可以是 dgl.Block（标准 DGL 格式）或
walpurgis.tensor.SparseGraph（cuGraph 稀疏格式）。
"""
