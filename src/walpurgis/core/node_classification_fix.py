# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit e90d1e6
# 原标题: Fix of create_node_classification_datasets (#128)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「辱骂和恐吓决不是战斗。」—— 鲁迅《答中山大学教授的信》
#
# e90d1e6 修复了两个同时存在的问题：
#
# 问题一：函数名拼写错误
#   旧名：create_node_claffication_datasets  （"claffication" 非词）
#   新名：create_node_classification_datasets（标准拼写）
#   受影响文件:
#     - pylibwholegraph/torch/data_loader.py  （定义处）
#     - pylibwholegraph/torch/__init__.py     （导出处）
#     - pylibwholegraph/examples/node_classfication.py（调用处）
#
# 问题二：API 耦合磁盘 I/O
#   旧 API：create_node_claffication_datasets(pickle_data_filename: str)
#     → 内部 open + pickle.load，函数不可测试（需要文件系统）
#   新 API：create_node_classification_datasets(data_and_label: dict)
#     → 调用方负责加载 pickle，函数只处理数据逻辑，可单元测试
#
# Walpurgis 20% 改写要点：
#   1. NodeClassificationDataBundle 数据类 — 将 data_and_label dict 的
#      预期 schema（"train"/"val"/"test" 各含 "feat"/"label"）提取为命名类型，
#      validate() 在 schema 不完整时给出字段级错误而非 KeyError
#   2. load_node_classification_pickle() 工厂函数 — 将调用方散落的
#      open+pickle.load 逻辑封装为带文件不存在/格式错误两层诊断的函数
#   3. create_node_classification_datasets() 新版实现 — 接收
#      NodeClassificationDataBundle 而非裸 dict，含 DEBUG 摘要打印
#   4. TypoAlias: create_node_claffication_datasets — 旧拼写的向后兼容别名，
#      发出 DeprecationWarning，指向新函数，方便搜索旧代码

from __future__ import annotations

import os as _os
import sys as _sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 数据类：节点分类数据集 Bundle
# ---------------------------------------------------------------------------

@dataclass
class NodeClassificationDataBundle:
    """
    封装节点分类任务中 data_and_label dict 的预期结构。

    上游 create_node_classification_datasets 接收裸 dict，
    若调用方传入 schema 不完整的 dict，抛出的是 KeyError，
    无法定位是哪个 split 缺少哪个字段。

    Walpurgis 迁移：提取为命名类型，validate() 给出字段级错误报告。

    Expected schema (from pylibwholegraph node classification example):
        {
            "train": { "feat": Tensor/ndarray, "label": Tensor/ndarray },
            "val":   { "feat": ..., "label": ... },
            "test":  { "feat": ..., "label": ... },
        }
    """

    train: Dict[str, Any]
    val: Dict[str, Any]
    test: Dict[str, Any]

    REQUIRED_SPLITS: Tuple[str, ...] = field(
        default=("train", "val", "test"), init=False, repr=False
    )
    REQUIRED_FIELDS: Tuple[str, ...] = field(
        default=("feat", "label"), init=False, repr=False
    )

    def validate(self) -> None:
        """
        校验所有 split 均包含 feat 和 label 字段。

        上游无此校验；Walpurgis 补充，防止 "slient wrong data" 问题。
        """
        errors: List[str] = []

        for split_name in ("train", "val", "test"):
            split_data = getattr(self, split_name)
            for req_field in ("feat", "label"):
                if req_field not in split_data:
                    errors.append(
                        f"split='{split_name}' 缺少字段 '{req_field}'"
                    )
                    _dbg(
                        "NodeClassificationDataBundle.validate",
                        f"ERROR: split={split_name} missing field={req_field}",
                    )

        if errors:
            raise ValueError(
                "[Walpurgis:NodeClassificationDataBundle] data_and_label schema 不完整:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n期望格式: {'train': {'feat': ..., 'label': ...}, 'val': ..., 'test': ...}"
            )

        _dbg(
            "NodeClassificationDataBundle.validate",
            f"OK  "
            f"train_feat_shape={_shape_str(self.train.get('feat'))}  "
            f"val_feat_shape={_shape_str(self.val.get('feat'))}  "
            f"test_feat_shape={_shape_str(self.test.get('feat'))}",
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeClassificationDataBundle":
        """
        从原始 data_and_label dict 构造，对应 e90d1e6 前后的调用方传参方式。

        上游调用方在 e90d1e6 之后：
            data_and_label = pickle.load(f)
            create_node_classification_datasets(data_and_label)

        Walpurgis 对应：
            bundle = NodeClassificationDataBundle.from_dict(data_and_label)
            bundle.validate()
            create_node_classification_datasets(bundle)
        """
        missing = [s for s in ("train", "val", "test") if s not in d]
        if missing:
            raise ValueError(
                f"[Walpurgis:NodeClassificationDataBundle.from_dict] "
                f"data_and_label 缺少 split: {missing}。"
                f"已有 keys: {list(d.keys())}"
            )
        return cls(train=d["train"], val=d["val"], test=d["test"])


def _shape_str(x: Any) -> str:
    """辅助：返回 tensor/ndarray 的 shape 字符串，None 时返回 'None'。"""
    if x is None:
        return "None"
    try:
        return str(tuple(x.shape))
    except Exception:
        return repr(type(x))


# ---------------------------------------------------------------------------
# pickle 加载工厂：对应 e90d1e6 前的 create_node_claffication_datasets 内部逻辑
# ---------------------------------------------------------------------------

def load_node_classification_pickle(
    pickle_data_path: str,
) -> "NodeClassificationDataBundle":
    """
    从 pickle 文件加载节点分类数据集，返回 NodeClassificationDataBundle。

    e90d1e6 之前，open+pickle.load 逻辑在 create_node_claffication_datasets 内部；
    e90d1e6 之后，调用方自己 open+pickle.load，然后把 dict 传给函数。

    Walpurgis 迁移：将调用方的 open+pickle.load 封装为带诊断的工厂函数，
    区分「文件不存在」和「pickle 格式错误」两种错误模式。

    Parameters
    ----------
    pickle_data_path : str
        pickle 文件路径（对应上游 args.pickle_data_path）

    Returns
    -------
    NodeClassificationDataBundle
    """
    import pickle

    _dbg("load_node_classification_pickle", f"path={pickle_data_path!r}")

    if not _os.path.exists(pickle_data_path):
        raise FileNotFoundError(
            f"[Walpurgis:load_node_classification_pickle] "
            f"找不到数据文件: {pickle_data_path!r}\n"
            f"请确认 --pickle_data_path 参数正确，或先运行数据预处理脚本。"
        )

    try:
        with open(pickle_data_path, "rb") as f:
            data_and_label = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"[Walpurgis:load_node_classification_pickle] "
            f"pickle 加载失败: {pickle_data_path!r}\n"
            f"原始错误: {exc}"
        ) from exc

    _dbg(
        "load_node_classification_pickle",
        f"loaded keys={list(data_and_label.keys())}",
    )

    return NodeClassificationDataBundle.from_dict(data_and_label)


# ---------------------------------------------------------------------------
# 核心函数：create_node_classification_datasets（e90d1e6 新 API）
# ---------------------------------------------------------------------------

def create_node_classification_datasets(
    data_and_label: "NodeClassificationDataBundle | Dict[str, Any]",
) -> Tuple[Any, Any, Any]:
    """
    从节点分类数据集 bundle 创建 train/val/test 数据集对象。

    对应上游 e90d1e6 修复后的
    pylibwholegraph/torch/data_loader.py::create_node_classification_datasets。

    API 变更（e90d1e6）：
      旧：create_node_claffication_datasets(pickle_data_filename: str)
          → 内部 open + pickle.load，耦合磁盘 I/O
      新：create_node_classification_datasets(data_and_label: dict)
          → 调用方负责 I/O，函数只做数据处理，可单元测试

    Walpurgis 改写：
      - 接收 NodeClassificationDataBundle 或裸 dict（向后兼容）
      - validate() 在 schema 不完整时给出字段级错误
      - DEBUG 打印各 split 的 feat/label shape

    Parameters
    ----------
    data_and_label :
        NodeClassificationDataBundle 或包含 "train"/"val"/"test" 键的 dict

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
        与上游返回值类型相同（依赖 WholeGraph TensorDataset）
    """
    # 统一为 NodeClassificationDataBundle
    if isinstance(data_and_label, dict):
        bundle = NodeClassificationDataBundle.from_dict(data_and_label)
    else:
        bundle = data_and_label

    bundle.validate()

    _dbg(
        "create_node_classification_datasets",
        f"train_label_shape={_shape_str(bundle.train.get('label'))}  "
        f"val_label_shape={_shape_str(bundle.val.get('label'))}  "
        f"test_label_shape={_shape_str(bundle.test.get('label'))}",
    )

    # 上游实现依赖 WholeGraph TensorDataset，此处给出兼容的 tuple 返回
    # 以便在无 WholeGraph 环境中运行单元测试
    try:
        from pylibwholegraph.torch.data_loader import (
            create_node_classification_datasets as _upstream_impl,
        )
        _dbg(
            "create_node_classification_datasets",
            "使用上游 pylibwholegraph 实现",
        )
        return _upstream_impl(data_and_label if isinstance(data_and_label, dict)
                              else {"train": bundle.train,
                                    "val": bundle.val,
                                    "test": bundle.test})
    except ImportError:
        _dbg(
            "create_node_classification_datasets",
            "pylibwholegraph 不可用，返回裸 tuple（测试环境）",
        )
        # 测试环境 fallback：直接返回 split dict
        return bundle.train, bundle.val, bundle.test


# ---------------------------------------------------------------------------
# 向后兼容别名：旧错误拼写
# ---------------------------------------------------------------------------

def create_node_claffication_datasets(
    pickle_data_filename: str,
) -> Tuple[Any, Any, Any]:
    """
    旧版函数名（拼写错误：claffication）的向后兼容别名。

    e90d1e6 将函数重命名并重构 API。此别名保留旧接口签名（接收文件路径），
    内部调用新版函数，同时发出 DeprecationWarning。

    Parameters
    ----------
    pickle_data_filename : str
        pickle 文件路径（旧 API）

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    warnings.warn(
        "[Walpurgis] create_node_claffication_datasets（拼写错误）已在 e90d1e6 重命名。\n"
        "请改用: create_node_classification_datasets(data_and_label: dict)\n"
        "新 API 接收 dict 而非文件路径，调用方自行 open+pickle.load。",
        DeprecationWarning,
        stacklevel=2,
    )
    bundle = load_node_classification_pickle(pickle_data_filename)
    return create_node_classification_datasets(bundle)


# ---------------------------------------------------------------------------
# 自测 __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile
    import pickle

    os.environ["WALPURGIS_DEBUG"] = "1"
    print("=== 自测 node_classification_fix.py (migrate e90d1e6) ===\n")

    # 构造测试数据
    try:
        import numpy as np
        _has_np = True
    except ImportError:
        _has_np = False

    def _make_split(n: int) -> dict:
        if _has_np:
            return {
                "feat": np.random.randn(n, 16).astype("float32"),
                "label": np.random.randint(0, 10, size=(n,)),
            }
        return {"feat": list(range(n)), "label": list(range(n))}

    good_data = {
        "train": _make_split(100),
        "val": _make_split(20),
        "test": _make_split(20),
    }

    # --- 测试 1: NodeClassificationDataBundle.from_dict ---
    bundle = NodeClassificationDataBundle.from_dict(good_data)
    assert bundle.train is good_data["train"]
    print("[OK] 测试1: NodeClassificationDataBundle.from_dict")

    # --- 测试 2: validate 成功 ---
    bundle.validate()
    print("[OK] 测试2: validate() 通过")

    # --- 测试 3: validate 缺字段 ---
    bad_data = {"train": {"feat": []}, "val": {"feat": [], "label": []}, "test": {"feat": [], "label": []}}
    bad_bundle = NodeClassificationDataBundle.from_dict(bad_data)
    try:
        bad_bundle.validate()
        assert False, "应该 ValueError"
    except ValueError as e:
        assert "train" in str(e) and "label" in str(e)
        print("[OK] 测试3: validate() 缺字段报错")

    # --- 测试 4: from_dict 缺 split ---
    try:
        NodeClassificationDataBundle.from_dict({"train": {}, "val": {}})
        assert False
    except ValueError as e:
        assert "test" in str(e)
        print("[OK] 测试4: from_dict 缺 split ValueError")

    # --- 测试 5: load_node_classification_pickle 文件不存在 ---
    try:
        load_node_classification_pickle("/nonexistent_xyz.pkl")
        assert False
    except FileNotFoundError as e:
        assert "nonexistent" in str(e)
        print("[OK] 测试5: load_node_classification_pickle 文件不存在报错")

    # --- 测试 6: create_node_classification_datasets（测试环境 fallback）---
    train_ds, val_ds, test_ds = create_node_classification_datasets(good_data)
    assert train_ds is good_data["train"]
    assert val_ds is good_data["val"]
    assert test_ds is good_data["test"]
    print("[OK] 测试6: create_node_classification_datasets 返回三元组")

    # --- 测试 7: load + create 端到端（pickle 文件）---
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(good_data, f)
        tmppath = f.name

    try:
        loaded_bundle = load_node_classification_pickle(tmppath)
        loaded_bundle.validate()
        r = create_node_classification_datasets(loaded_bundle)
        assert len(r) == 3
        print("[OK] 测试7: load + create 端到端（pickle 文件）")
    finally:
        os.unlink(tmppath)

    print("\n=== 全部自测通过 ===")
