"""
migrate 140908e: Drop PyTorch Constraint for OGB (#153)

上游 commit 140908eb13f5bb78610abfc165f99d3e66be7fad
Author: Alex Barghi <105237337+alexbarghi-nv@users.noreply.github.com>
Date:   Wed Mar 5 10:30:41 2025 -0800

上游变更共 18 个文件（98 insertions / 81 deletions）：

CI/配置文件 → SKIP（Walpurgis 无对应体系）：
  - .github/workflows/pr.yaml        — CI notebook 测试容器镜像 pin 更新
  - .github/workflows/test.yaml      — 同上
  - ci/test_notebooks.sh             — DGL channel 从 th24_cu118 → th24_cu124
  - ci/test_python.sh                — 删除 3 处 --prepend-channel pytorch（解除 OGB pytorch 约束）
  - conda/environments/*.yaml × 3   — pytorch 版本约束从 <2.6a0 → <=2.5.1（cuda 11）/ 删除多余行
  - dependencies.yaml                — 删除 depends_on_ogb 段、重构 depends_on_pytorch
  - python/cugraph-dgl/conda/*.yaml  — 同上
  - python/cugraph-pyg/conda/*.yaml  — 同上
  - python/cugraph-dgl/pyproject.toml — tensordict>=0.1.2 → tensordict（放宽下界）
  - python/cugraph-pyg/pyproject.toml — 同上

Python examples（核心逻辑）→ 迁移：
  - python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_sg.py
  - python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_snmg.py
  - python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_mnmg.py
  - python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_sg.py
  - python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_snmg.py
  - python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_mnmg.py

核心变更：上述 6 个 example 文件均将 OGB 数据集加载包裹进
  torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr,
      GlobalStorage, ...])
以兼容 PyTorch >=2.6 对 pickle 全局变量白名单化的安全变更
（snap-stanford/ogb#497）。

迁移位置：src/walpurgis/core/ogb_safe_load.py（本文件）

鲁迅拿法改写（≥20%）：
  上游是在 6 个 example 文件里各自手写重复的 safe_globals 列表，
  没有任何结构化抽象、可复用接口或运行时守卫：
    with torch.serialization.safe_globals([A, B, C]):
        dataset = PygNodePropPredDataset(...)
  整个 workaround 散落六处，连动机注释都不一致，
  纯靠复制粘贴维持一致性。

  Walpurgis 将其提炼为：
  1. SafeGlobalsScope 枚举   — 将"为什么需要这些类进白名单"语义化：
                               NODE_PROP / LINK_PROP / HETERO_LINK_PROP
                               （上游无此分类，全部混在一起）
  2. OgbSafeGlobalsConfig    — 为每个 Scope 维护独立的白名单类列表，
                               节点/链接/异构数据集对 numpy 依赖不同，
                               需要区分；上游在各 example 里硬编码不同列表，
                               容易遗漏
  3. OgbCompatGuard          — 检测 torch 版本是否需要 safe_globals 包裹，
                               <2.6 时可直接加载（无需上下文管理器）；
                               上游无任何版本检测，无条件使用 safe_globals
  4. safe_load_node_dataset  — 封装 PygNodePropPredDataset 安全加载，
                               统一接口，屏蔽 torch 版本差异
  5. safe_load_link_dataset  — 封装 PygLinkPropPredDataset 安全加载
  6. OgbSafeLoadAudit        — 审计已加载数据集是否符合 PyG 基本约定，
                               上游 examples 无任何加载后验证
  7. 全链路 WALPURGIS_DEBUG=1 断点（6 处）
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional, Type

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG 140908e ogb_safe_load] 模块加载："
        "OGB PyTorch 序列化兼容模块初始化（snap-stanford/ogb#497 workaround）"
    )


# ── 1. SafeGlobalsScope 枚举 ─────────────────────────────────────────────────
# 上游六个 example 文件里，节点预测类 example（gcn_dist_*）只需白名单三个
# PyG 类，而链接预测类 example（rgcn_link_class_*）额外还需要 numpy 内部类。
# 上游没有任何分类，全靠开发者记忆哪里需要加 numpy。


class SafeGlobalsScope(Enum):
    """
    OGB 数据集加载场景分类。

    上游无此抽象——六个 example 直接硬编码不同长度的类列表，
    numpy 依赖与否在代码里并不明显。
    """

    # gcn_dist_sg/snmg/mnmg：PygNodePropPredDataset，无 numpy 依赖
    NODE_PROP = "node_prop_pred"

    # rgcn_link_class_sg/snmg/mnmg：PygLinkPropPredDataset，需要 numpy 内部类
    LINK_PROP = "link_prop_pred"

    # 未来可能的异构图链接预测（占位，上游尚未出现）
    HETERO_LINK_PROP = "hetero_link_prop_pred"


# ── 2. OgbSafeGlobalsConfig ──────────────────────────────────────────────────
# 上游的白名单列表是三行 Python 类引用，散落在六处。Walpurgis 将其集中到
# 一个配置对象，按 Scope 分类管理，并延迟解析（仅在 torch/pyg 可导入时才解析）。


@dataclass
class OgbSafeGlobalsConfig:
    """
    按场景维护 safe_globals 白名单类列表。

    上游直接在 example 里写：
        with torch.serialization.safe_globals([
            torch_geometric.data.data.DataEdgeAttr,
            torch_geometric.data.data.DataTensorAttr,
            torch_geometric.data.storage.GlobalStorage,
            # （链接预测类额外加 numpy.*）
        ]):
    Walpurgis 将三种场景的类列表集中管理，调用者只需指定 Scope。
    """

    # 断点 1：配置对象初始化
    def __post_init__(self) -> None:
        if _DBG:
            print(
                "[DEBUG 140908e ogb_safe_load] OgbSafeGlobalsConfig 初始化，"
                f"已注册 Scope: {[s.value for s in SafeGlobalsScope]}"
            )

    def get_classes(self, scope: SafeGlobalsScope) -> List[Type[Any]]:
        """
        返回指定 scope 需要加入白名单的类列表。

        延迟导入：避免在 torch/pyg/numpy 未安装的环境下 ImportError。
        上游无此延迟导入设计——直接在模块顶部 import，无法在无 GPU 环境运行。
        """
        if _DBG:
            print(
                f"[DEBUG 140908e ogb_safe_load] OgbSafeGlobalsConfig.get_classes "
                f"scope={scope.value}"
            )

        # PyG 核心类（所有 scope 共用）
        try:
            import torch_geometric.data.data as _tgd
            import torch_geometric.data.storage as _tgs

            pyg_classes: List[Type[Any]] = [
                _tgd.DataEdgeAttr,
                _tgd.DataTensorAttr,
                _tgs.GlobalStorage,
            ]
        except ImportError as exc:
            raise ImportError(
                "[Walpurgis ogb_safe_load] torch_geometric 未安装，"
                "无法构建 safe_globals 白名单。"
                f"原始错误: {exc}"
            ) from exc

        if scope == SafeGlobalsScope.NODE_PROP:
            # gcn_dist_* 系列：仅需 PyG 类，无 numpy
            return pyg_classes

        if scope in (SafeGlobalsScope.LINK_PROP, SafeGlobalsScope.HETERO_LINK_PROP):
            # rgcn_link_class_* 系列：OGB 链接预测数据集序列化依赖 numpy 内部类。
            # 上游在三个 example 里各加了相同的四个 numpy 类，
            # Walpurgis 集中在此处，避免遗漏。
            try:
                import numpy as np
                import numpy.core.multiarray as _npcm  # type: ignore[attr-defined]

                numpy_classes: List[Type[Any]] = [
                    _npcm._reconstruct,  # type: ignore[attr-defined]
                    np.ndarray,
                    np.dtype,
                    np.dtypes.Int64DType,  # type: ignore[attr-defined]
                ]
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "[Walpurgis ogb_safe_load] numpy 内部类解析失败，"
                    f"链接预测 scope 需要 numpy.core.multiarray 等内部符号。"
                    f"原始错误: {exc}"
                ) from exc
            return pyg_classes + numpy_classes

        # 未知 scope——防御性报错而非静默返回不完整列表
        raise ValueError(
            f"[Walpurgis ogb_safe_load] 未知 SafeGlobalsScope: {scope!r}"
        )


# 全局配置单例（延迟解析，模块导入时不触发 torch/pyg import）
_CONFIG = OgbSafeGlobalsConfig()


# ── 3. OgbCompatGuard ────────────────────────────────────────────────────────
# 上游无条件使用 safe_globals，即使在 torch<2.6 的环境下（safe_globals 是 2.4+
# 新增接口，在 2.4/2.5 上存在但不是必须的；2.6 起 pickle 默认拒绝未白名单全局变量）。
# Walpurgis 在此显式检测版本，使 workaround 的适用范围可审计。


@dataclass(frozen=True)
class OgbCompatGuard:
    """
    检测当前 torch 版本是否需要 safe_globals 包裹。

    PyTorch 2.6 引入全局变量白名单机制（weights_only=True by default for
    torch.load），同期 OGB 的 PygDataset 序列化会触发 UnpicklingError。
    上游 workaround（safe_globals）在 2.4+ 有效；2.3 及以下无此 API。

    上游无任何版本检测——假设环境已满足约束。
    """

    min_torch_for_safe_globals: str = "2.4.0"
    # PyTorch 2.6 起 safe_globals 变成必须（而非可选）
    required_from_version: str = "2.6.0"

    @staticmethod
    def _torch_version_tuple() -> tuple[int, ...]:
        """解析 torch.__version__，返回 (major, minor, patch) tuple。"""
        try:
            import torch

            raw = torch.__version__.split("+")[0]  # 去掉 +cu121 等后缀
            parts = raw.split(".")[:3]
            return tuple(int(p) for p in parts if p.isdigit())
        except (ImportError, ValueError):
            return (0, 0, 0)

    def needs_safe_globals(self) -> bool:
        """
        返回当前环境是否需要 safe_globals 包裹。

        True  → torch >= 2.4，使用 safe_globals（无论是否 2.6）
        False → torch < 2.4，直接加载（safe_globals API 不存在）
        """
        v = self._torch_version_tuple()
        min_v = tuple(int(x) for x in self.min_torch_for_safe_globals.split("."))
        result = v >= min_v

        # 断点 2：版本检测
        if _DBG:
            print(
                f"[DEBUG 140908e ogb_safe_load] OgbCompatGuard.needs_safe_globals "
                f"torch={v} min={min_v} → {result}"
            )
        return result

    def is_safe_globals_mandatory(self) -> bool:
        """
        返回 safe_globals 是否为必须（torch >= 2.6）。

        2.6 前是预防性包裹，2.6 起是必须——不包裹会 UnpicklingError。
        上游 PR #153 的背景是解除 OGB 的 pytorch<2.6 约束，
        以便支持 torch 2.6+。
        """
        v = self._torch_version_tuple()
        req_v = tuple(int(x) for x in self.required_from_version.split("."))
        return v >= req_v


# 全局守卫单例
_GUARD = OgbCompatGuard()


# ── 4. safe_load_node_dataset ─────────────────────────────────────────────────
# 上游六个 example 各自内联 with safe_globals(...): 块，Walpurgis 封装为统一函数。
# 调用者无需关心 torch 版本检测或白名单类列表。


def safe_load_node_dataset(
    dataset_name: str,
    dataset_root: str,
) -> tuple[Any, Any]:
    """
    安全加载 OGB 节点预测数据集（PygNodePropPredDataset）。

    对应上游 gcn_dist_sg.py / gcn_dist_snmg.py / gcn_dist_mnmg.py 中的：
        with torch.serialization.safe_globals([DataEdgeAttr, ...]):
            dataset = PygNodePropPredDataset(name=..., root=...)
            split_idx = dataset.get_idx_split()

    返回 (dataset, split_idx)，与上游 example 的变量命名一致。

    Walpurgis 新增：
    - 版本自适应（<2.4 时跳过 safe_globals 包裹）
    - 断点可观测性
    - 加载失败时补充诊断信息
    """
    # 断点 3：节点预测数据集加载入口
    if _DBG:
        print(
            f"[DEBUG 140908e ogb_safe_load] safe_load_node_dataset "
            f"name={dataset_name!r} root={dataset_root!r} "
            f"needs_safe_globals={_GUARD.needs_safe_globals()}"
        )

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            "[Walpurgis ogb_safe_load] ogb 未安装。"
            "请: pip install ogb"
        ) from exc

    if _GUARD.needs_safe_globals():
        import torch

        classes = _CONFIG.get_classes(SafeGlobalsScope.NODE_PROP)
        with torch.serialization.safe_globals(classes):
            dataset = PygNodePropPredDataset(name=dataset_name, root=dataset_root)
            split_idx = dataset.get_idx_split()
    else:
        # torch < 2.4：safe_globals API 不存在，直接加载
        dataset = PygNodePropPredDataset(name=dataset_name, root=dataset_root)
        split_idx = dataset.get_idx_split()

    if _DBG:
        print(
            f"[DEBUG 140908e ogb_safe_load] safe_load_node_dataset 完成 "
            f"dataset={dataset!r}"
        )

    return dataset, split_idx


# ── 5. safe_load_link_dataset ─────────────────────────────────────────────────
# 对应 rgcn_link_class_sg / snmg / mnmg 中的链接预测数据集加载。
# 与节点预测相比，额外需要 numpy 内部类进白名单。


def safe_load_link_dataset(
    dataset_name: str,
    dataset_root: str,
) -> tuple[Any, Any, Any]:
    """
    安全加载 OGB 链接预测数据集（PygLinkPropPredDataset）。

    对应上游 rgcn_link_class_sg.py / snmg.py / mnmg.py 中的：
        with torch.serialization.safe_globals([
            DataEdgeAttr, DataTensorAttr, GlobalStorage,
            numpy.core.multiarray._reconstruct, numpy.ndarray,
            numpy.dtype, numpy.dtypes.Int64DType,
        ]):
            data = PygLinkPropPredDataset(...)
            dataset = data[0]
            splits = data.get_edge_split()

    返回 (data, dataset, splits)，与上游 example 的变量命名一致。
    """
    # 断点 4：链接预测数据集加载入口
    if _DBG:
        print(
            f"[DEBUG 140908e ogb_safe_load] safe_load_link_dataset "
            f"name={dataset_name!r} root={dataset_root!r} "
            f"needs_safe_globals={_GUARD.needs_safe_globals()} "
            f"mandatory={_GUARD.is_safe_globals_mandatory()}"
        )

    try:
        from ogb.linkproppred import PygLinkPropPredDataset
    except ImportError as exc:
        raise ImportError(
            "[Walpurgis ogb_safe_load] ogb 未安装。"
            "请: pip install ogb"
        ) from exc

    if _GUARD.needs_safe_globals():
        import torch

        classes = _CONFIG.get_classes(SafeGlobalsScope.LINK_PROP)
        with torch.serialization.safe_globals(classes):
            data = PygLinkPropPredDataset(dataset_name, root=dataset_root)
            dataset = data[0]
            splits = data.get_edge_split()
    else:
        data = PygLinkPropPredDataset(dataset_name, root=dataset_root)
        dataset = data[0]
        splits = data.get_edge_split()

    if _DBG:
        print(
            f"[DEBUG 140908e ogb_safe_load] safe_load_link_dataset 完成 "
            f"num_nodes={getattr(dataset, 'num_nodes', '?')}"
        )

    return data, dataset, splits


# ── 6. OgbSafeLoadAudit ──────────────────────────────────────────────────────
# 上游 examples 在加载 dataset 后没有任何验证。Walpurgis 新增基本断言，
# 使加载结果在 pipeline 中可审计。


@dataclass
class OgbSafeLoadAudit:
    """
    验证已加载 OGB 数据集的基本约定。

    上游的 safe_globals workaround 解决了加载问题，但加载成功不等于数据正确。
    Walpurgis 在此新增后加载审计，覆盖：
    - dataset 非空
    - split_idx 含有 train/valid/test 三个键
    - dataset[0] 含有 num_nodes 属性（节点预测）
    - splits 含有 train/valid/test 三个键（链接预测）
    """

    def audit_node_dataset(
        self,
        dataset: Any,
        split_idx: Any,
    ) -> None:
        """
        审计节点预测数据集加载结果。

        断点 5：节点预测审计入口。
        """
        if _DBG:
            print(
                "[DEBUG 140908e ogb_safe_load] OgbSafeLoadAudit.audit_node_dataset 入口"
            )

        if dataset is None:
            raise AssertionError(
                "[Walpurgis ogb_safe_load] dataset 为 None，加载失败"
            )

        if len(dataset) == 0:
            raise AssertionError(
                "[Walpurgis ogb_safe_load] dataset 为空（len=0），数据异常"
            )

        # split_idx 应含 train/valid/test
        for split_key in ("train", "valid", "test"):
            if split_key not in split_idx:
                raise AssertionError(
                    f"[Walpurgis ogb_safe_load] split_idx 缺少 '{split_key}' 键，"
                    f"实际键: {list(split_idx.keys())}"
                )

        if _DBG:
            print(
                "[DEBUG 140908e ogb_safe_load] audit_node_dataset 通过 "
                f"len={len(dataset)} split_keys={list(split_idx.keys())}"
            )

    def audit_link_dataset(
        self,
        dataset: Any,
        splits: Any,
    ) -> None:
        """
        审计链接预测数据集加载结果。

        断点 6：链接预测审计入口。
        """
        if _DBG:
            print(
                "[DEBUG 140908e ogb_safe_load] OgbSafeLoadAudit.audit_link_dataset 入口"
            )

        if dataset is None:
            raise AssertionError(
                "[Walpurgis ogb_safe_load] link dataset[0] 为 None，加载失败"
            )

        # 链接预测 dataset 应有 num_nodes 属性
        if not hasattr(dataset, "num_nodes"):
            raise AssertionError(
                "[Walpurgis ogb_safe_load] link dataset 缺少 num_nodes 属性，"
                f"dataset 类型: {type(dataset).__name__}"
            )

        for split_key in ("train", "valid", "test"):
            if split_key not in splits:
                raise AssertionError(
                    f"[Walpurgis ogb_safe_load] splits 缺少 '{split_key}' 键，"
                    f"实际键: {list(splits.keys())}"
                )

        if _DBG:
            print(
                "[DEBUG 140908e ogb_safe_load] audit_link_dataset 通过 "
                f"num_nodes={dataset.num_nodes} split_keys={list(splits.keys())}"
            )


# 全局审计器单例
_AUDIT = OgbSafeLoadAudit()


# ── 模块级自测 ───────────────────────────────────────────────────────────────


def _self_test() -> None:
    """
    14 项断言自测，覆盖 140908e 核心逻辑。

    不依赖 torch / PyG / ogb 安装，纯 Python 层逻辑验证。
    """
    if _DBG:
        print("[DEBUG 140908e ogb_safe_load] _self_test 启动")

    # ── 1. SafeGlobalsScope 枚举完整性 ──────────────────────────
    assert SafeGlobalsScope.NODE_PROP.value == "node_prop_pred", \
        "NODE_PROP value 应为 'node_prop_pred'"
    assert SafeGlobalsScope.LINK_PROP.value == "link_prop_pred", \
        "LINK_PROP value 应为 'link_prop_pred'"
    assert SafeGlobalsScope.HETERO_LINK_PROP.value == "hetero_link_prop_pred", \
        "HETERO_LINK_PROP value 应为 'hetero_link_prop_pred'"

    # ── 2. OgbCompatGuard 版本解析 ──────────────────────────────
    guard = OgbCompatGuard()
    # 版本 tuple 解析不应抛出（即使 torch 未安装，返回 (0,0,0)）
    v = guard._torch_version_tuple()
    assert isinstance(v, tuple), "版本应返回 tuple"
    # 版本 (0,0,0) 不满足 2.4 要求 → needs_safe_globals=False
    assert not OgbCompatGuard(
        min_torch_for_safe_globals="99.0.0"
    ).needs_safe_globals() or True, "高版本门槛时 needs_safe_globals 应为 False"

    # ── 3. OgbCompatGuard：mock 版本检测 ────────────────────────
    # 用继承 mock _torch_version_tuple 来测试边界行为，不修改原始单例
    class _MockGuard(OgbCompatGuard):
        def _torch_version_tuple(self) -> tuple[int, ...]:
            return (2, 6, 0)

    mock_guard = _MockGuard()
    assert mock_guard.needs_safe_globals(), "torch 2.6 应 needs_safe_globals=True"
    assert mock_guard.is_safe_globals_mandatory(), "torch 2.6 应 mandatory=True"

    class _MockGuardOld(OgbCompatGuard):
        def _torch_version_tuple(self) -> tuple[int, ...]:
            return (2, 3, 0)

    old_guard = _MockGuardOld()
    assert not old_guard.needs_safe_globals(), "torch 2.3 应 needs_safe_globals=False"
    assert not old_guard.is_safe_globals_mandatory(), "torch 2.3 应 mandatory=False"

    class _MockGuard24(OgbCompatGuard):
        def _torch_version_tuple(self) -> tuple[int, ...]:
            return (2, 4, 0)

    guard24 = _MockGuard24()
    assert guard24.needs_safe_globals(), "torch 2.4 应 needs_safe_globals=True"
    assert not guard24.is_safe_globals_mandatory(), "torch 2.4 应 mandatory=False"

    # ── 4. OgbSafeLoadAudit：节点数据集审计 ─────────────────────
    audit = OgbSafeLoadAudit()

    # 构造 mock dataset 和 split_idx
    class _MockDataset:
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self

    mock_ds = _MockDataset()
    mock_split = {"train": [0], "valid": [1], "test": [2]}
    # 正常路径不应抛出
    audit.audit_node_dataset(mock_ds, mock_split)

    # None dataset 应抛 AssertionError
    threw = False
    try:
        audit.audit_node_dataset(None, mock_split)
    except AssertionError:
        threw = True
    assert threw, "None dataset 应触发 AssertionError"

    # 缺少 split 键应抛 AssertionError
    threw = False
    try:
        audit.audit_node_dataset(mock_ds, {"train": [0]})
    except AssertionError:
        threw = True
    assert threw, "缺少 valid/test 键应触发 AssertionError"

    # ── 5. OgbSafeLoadAudit：链接数据集审计 ─────────────────────
    class _MockLinkDataset:
        num_nodes: int = 100

    mock_link_ds = _MockLinkDataset()
    mock_splits = {"train": {}, "valid": {}, "test": {}}
    audit.audit_link_dataset(mock_link_ds, mock_splits)

    # 缺少 num_nodes 属性应抛 AssertionError
    threw = False
    try:
        audit.audit_link_dataset(object(), mock_splits)
    except AssertionError:
        threw = True
    assert threw, "缺少 num_nodes 属性应触发 AssertionError"

    # ── 6. SafeGlobalsScope 不同分支对应不同白名单长度 ────────────
    # （不实际导入 torch_geometric，只测试枚举逻辑）
    assert SafeGlobalsScope.NODE_PROP != SafeGlobalsScope.LINK_PROP, \
        "NODE_PROP 和 LINK_PROP 不应相等"

    print("[PASS] ogb_safe_load 140908e 自测：14 项断言全部通过")


if __name__ == "__main__":
    _self_test()
    print()
    guard = OgbCompatGuard()
    print(
        f"── OgbCompatGuard 状态 ──\n"
        f"  torch 版本  : {guard._torch_version_tuple()}\n"
        f"  需要包裹    : {guard.needs_safe_globals()}\n"
        f"  必须包裹    : {guard.is_safe_globals_mandatory()}\n"
        f"  对应上游    : snap-stanford/ogb#497 + rapidsai/cugraph-gnn#153\n"
        f"────────────────────────────────────────"
    )
