"""
node_classification.py — e90d1e6 迁移: 修复 create_node_classification_datasets

migrate e90d1e6: Fix of create_node_classification_datasets (#128)

上游变化 (e90d1e6, cugraph-gnn):

1. pylibwholegraph/torch/data_loader.py — create_node_claffication_datasets 重命名 + 解耦 pickle IO:
   旧:
       def create_node_claffication_datasets(pickle_data_filename: str):
           with open(pickle_data_filename, "rb") as f:
               data_and_label = pickle.load(f)
           train_data = {"idx": data_and_label["train_idx"], ...}
           ...
   新:
       def create_node_classification_datasets(data_and_label: dict):
           train_data = {"idx": data_and_label["train_idx"], ...}
           ...

   变化要点:
   a) 拼写修复: claffication → classification（typo 修复，已存在数版）
   b) 接口解耦: 函数不再直接打开 pickle 文件，接受已解析的 dict。
      pickle 读取移到调用侧 (node_classfication.py 的 main_func())。
   c) __init__.py 导出名同步更新:
      from .data_loader import create_node_claffication_datasets → create_node_classification_datasets

2. pylibwholegraph/examples/node_classfication.py — 调用侧适配:
   旧:
       train_ds, valid_ds, test_ds = wgth.create_node_claffication_datasets(args.pickle_data_path)
   新:
       data_and_label = dict()
       with open(args.pickle_data_path, "rb") as f:
           data_and_label = pickle.load(f)
       train_ds, valid_ds, test_ds = wgth.create_node_classification_datasets(data_and_label)

设计语义:
    - 解耦 IO 与数据集构建: 函数职责更单一，便于单元测试（无需 mock 文件系统）。
    - 拼写修复是 API breaking change（但旧名从未公开宣传，影响范围小）。
    - 旧名 create_node_claffication_datasets 通过 compat 别名保留（FutureWarning）。

Walpurgis 改写 20%（鲁迅拿法）:
- NodeClassificationData 数据类封装 data_and_label dict 的三路拆分结果
  上游函数直接返回三个 Dataset 对象，不携带任何 shape 信息，debug 困难。
  NodeClassificationData 携带 train_size / valid_size / test_size，
  在 WALPURGIS_DEBUG=1 时打印各 split 大小。
- DatasetSplitValidator 封装 dict 字段校验
  上游直接 data_and_label["train_idx"]，缺失 key 时抛 KeyError，无友好报错。
  DatasetSplitValidator.validate() 检查所有必要字段，提前报错。
- PickleLoader 封装 pickle 文件读取（对应调用侧解耦）
  上游 main_func 内联 open + pickle.load，PickleLoader 给这段逻辑命名，
  加 WALPURGIS_DEBUG 断点打印文件路径 + 加载后 key 列表。
- 全链路 WALPURGIS_DEBUG=1 断点 print

作者: dylanyunlon <dogechat@163.com>
"""

import os
import sys
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

_WDBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv):
    if _WDBG:
        parts = [f"[WDBG:{tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DatasetSplitValidator — 字段校验
# ─────────────────────────────────────────────────────────────────────────────

class DatasetSplitValidator:
    """
    校验 data_and_label dict 包含所有必要字段。

    上游 create_node_classification_datasets 直接索引 dict，
    缺失 key 时抛 KeyError（无友好报错，不清楚哪个 key 缺失）。
    DatasetSplitValidator.validate() 提前检查，统一报错格式。
    """

    REQUIRED_KEYS = [
        "train_idx", "train_label",
        "valid_idx", "valid_label",
        "test_idx",  "test_label",
    ]

    @classmethod
    def validate(cls, data_and_label: dict) -> None:
        missing = [k for k in cls.REQUIRED_KEYS if k not in data_and_label]
        if missing:
            raise KeyError(
                f"data_and_label 缺少必要字段: {missing}. "
                f"实际 keys: {list(data_and_label.keys())}"
            )
        _dbg(
            "DatasetSplitValidator",
            "OK",
            keys=list(data_and_label.keys()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# NodeClassificationDataset — 上游 Dataset 类（保持原始接口）
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class NodeClassificationDataset(_TorchDataset):
        """
        节点分类数据集 wrapper，适配 torch.utils.data.DataLoader。

        直接对应上游 pylibwholegraph/torch/data_loader.py 中的同名类。
        dataset dict 包含 'idx' 和 'label' 两个字段。
        """

        def __init__(self, dataset: Dict[str, Any]):
            self.dataset = dataset
            _dbg(
                "NodeClassificationDataset",
                "init",
                idx_len=len(dataset.get("idx", [])),
            )

        def __getitem__(self, index):
            return {k: v[index] for k, v in self.dataset.items()}

        def __len__(self):
            return len(self.dataset["idx"])

except ImportError:
    # 无 torch 环境的 stub
    class NodeClassificationDataset:  # type: ignore
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return {k: v[index] for k, v in self.dataset.items()}

        def __len__(self):
            return len(self.dataset["idx"])


# ─────────────────────────────────────────────────────────────────────────────
# NodeClassificationData — 三路 split 数据容器
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeClassificationData:
    """
    封装 create_node_classification_datasets 的三路返回值。

    上游返回裸三元组 (train_ds, valid_ds, test_ds)，字段无名称。
    NodeClassificationData 给三个 split 命名，并在 WALPURGIS_DEBUG 时打印大小。
    """
    train_ds: NodeClassificationDataset
    valid_ds: NodeClassificationDataset
    test_ds: NodeClassificationDataset

    @property
    def train_size(self) -> int:
        return len(self.train_ds)

    @property
    def valid_size(self) -> int:
        return len(self.valid_ds)

    @property
    def test_size(self) -> int:
        return len(self.test_ds)

    def debug_print(self):
        _dbg(
            "NodeClassificationData",
            "splits",
            train_size=self.train_size,
            valid_size=self.valid_size,
            test_size=self.test_size,
        )

    def unpack(self):
        """兼容上游三元组解包模式: train_ds, valid_ds, test_ds = data.unpack()"""
        return self.train_ds, self.valid_ds, self.test_ds


# ─────────────────────────────────────────────────────────────────────────────
# PickleLoader — 封装 pickle 文件读取（对应 e90d1e6 调用侧解耦）
# ─────────────────────────────────────────────────────────────────────────────

class PickleLoader:
    """
    封装 pickle 文件的读取和校验。

    对应 e90d1e6 将 pickle IO 从 create_node_classification_datasets 移到调用侧。
    PickleLoader.load() 给这段 IO 逻辑命名，加 WALPURGIS_DEBUG 断点打印 key 列表。
    """

    @staticmethod
    def load(pickle_path: str) -> dict:
        """
        加载 pickle 文件，返回 data_and_label dict。

        对应 e90d1e6 后的调用侧:
            data_and_label = dict()
            with open(args.pickle_data_path, "rb") as f:
                data_and_label = pickle.load(f)
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"pickle 文件不存在: {pickle_path}")

        _dbg("PickleLoader", f"loading {pickle_path!r}")

        with open(pickle_path, "rb") as f:
            data_and_label = pickle.load(f)

        if not isinstance(data_and_label, dict):
            raise TypeError(
                f"pickle 文件内容应为 dict，实际类型: {type(data_and_label).__name__}"
            )

        _dbg(
            "PickleLoader",
            "loaded",
            path=pickle_path,
            keys=list(data_and_label.keys()),
        )
        return data_and_label


# ─────────────────────────────────────────────────────────────────────────────
# create_node_classification_datasets — 核心函数 (e90d1e6 修复版)
# ─────────────────────────────────────────────────────────────────────────────

def create_node_classification_datasets(
    data_and_label: dict,
) -> NodeClassificationData:
    """
    从 data_and_label dict 构建三路节点分类数据集。

    e90d1e6 修复要点:
    1. 拼写修复: claffication → classification
    2. 接口解耦: 参数从 pickle_data_filename (str) 改为 data_and_label (dict)
       pickle IO 移到调用侧，函数职责更单一，便于单元测试。

    Parameters
    ----------
    data_and_label : dict
        必须包含:
          - train_idx, train_label
          - valid_idx, valid_label
          - test_idx,  test_label

    Returns
    -------
    NodeClassificationData
        包含 train_ds / valid_ds / test_ds 三个 NodeClassificationDataset。
        调用 .unpack() 可以获得上游兼容的三元组。
    """
    DatasetSplitValidator.validate(data_and_label)

    _dbg("create_node_classification_datasets", "building splits")

    train_ds = NodeClassificationDataset({
        "idx":   data_and_label["train_idx"],
        "label": data_and_label["train_label"],
    })
    valid_ds = NodeClassificationDataset({
        "idx":   data_and_label["valid_idx"],
        "label": data_and_label["valid_label"],
    })
    test_ds = NodeClassificationDataset({
        "idx":   data_and_label["test_idx"],
        "label": data_and_label["test_label"],
    })

    result = NodeClassificationData(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds)
    result.debug_print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 兼容别名 — 旧拼写错误名保留（FutureWarning）
# ─────────────────────────────────────────────────────────────────────────────

def create_node_claffication_datasets(data_and_label):
    """
    兼容别名: create_node_claffication_datasets（原拼写错误版本）。

    e90d1e6 将函数重命名为 create_node_classification_datasets。
    旧名通过此 wrapper 保留，发出 FutureWarning 催迁移。
    """
    warnings.warn(
        "create_node_claffication_datasets is a typo — "
        "use create_node_classification_datasets instead.",
        FutureWarning,
        stacklevel=2,
    )
    _dbg("compat", "create_node_claffication_datasets → create_node_classification_datasets")
    return create_node_classification_datasets(data_and_label)


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数: 从 pickle 文件直接加载数据集（组合 PickleLoader + create_node_classification_datasets）
# ─────────────────────────────────────────────────────────────────────────────

def load_node_classification_datasets_from_pickle(
    pickle_path: str,
) -> NodeClassificationData:
    """
    从 pickle 文件直接加载并构建三路节点分类数据集。

    对应 e90d1e6 之前的旧接口使用模式:
        train_ds, valid_ds, test_ds = wgth.create_node_claffication_datasets(args.pickle_data_path)
    等价于 e90d1e6 之后的新模式:
        data_and_label = pickle.load(open(pickle_path, "rb"))
        train_ds, valid_ds, test_ds = wgth.create_node_classification_datasets(data_and_label)
    """
    data_and_label = PickleLoader.load(pickle_path)
    return create_node_classification_datasets(data_and_label)
