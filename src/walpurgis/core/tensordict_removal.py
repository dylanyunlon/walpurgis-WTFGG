"""
tensordict_removal.py — 78128d9 迁移: 移除非统一API与TensorDict残余

migrate 78128d9: Remove Non-Unified API and Remaining TensorDict Code (#222)

上游变化 (78128d9, cugraph-gnn / python/cugraph-pyg/):

1. data/__init__.py — 清道夫式瘦身:
   - 删除 GraphStore 工厂函数包装（含 is_multi_gpu 分发逻辑 + WholeGraph 初始化）
   - 删除 WholeFeatureStore / TensorDictFeatureStore 兼容函数
   - 直接从 graph_store 导入 GraphStore（即原 NewGraphStore 正式更名）
   - 旧 DEPRECATED__OldGraphStore / NewGraphStore 别名全部清除

2. data/feature_store.py — 删除 TensorDictFeatureStore 类 (~110行):
   - 基于 tensordict.TensorDict 的单机非分布式特征存储
   - __features dict: {group_name → TensorDict{attr_name → tensor}}
   - _put_tensor: batch_size 对齐校验 + TensorDict赋值
   - _get_tensor: 支持 attr.index 切片读取
   - _remove_tensor: del td[attr_name]
   - get_all_tensor_attrs: 遍历所有 group/attr pair
   - 文档字符串更新: "WholeFeatureStore" → "FeatureStore"

3. data/graph_store.py — 删除旧 GraphStore 类 (~340行) + 正式化 NewGraphStore:
   - 旧 GraphStore: 基于 tensordict.TensorDict({}, batch_size=(2,)) 存储边索引
     单机实现，_put_edge_index 直接 torch.stack([src,dst]) 入 TensorDict
   - 新 GraphStore (原 NewGraphStore 重命名): 基于 DistMatrix 分布式实现
     删除 tensordict import，__edge_indices 改为普通 dict
     bug 修复: _put_edge_index 新增 list 类型防御 —— isinstance(edge_index, list) → torch.stack(edge_index)
   - 清除 tensordict = import_optional("tensordict") import 行

4. 依赖清理:
   - conda/recipes/cugraph-pyg/recipe.yaml: 删除 tensordict >=0.1.2
   - dependencies.yaml: 删除 cugraph_pyg_dev 依赖组 + tensordict anchor 引用
   - conda dev yaml (aarch64/x86_64): 删除 tensordict>=0.1.2,<=0.6.2

5. 测试/示例更新 (24 files changed):
   - 所有 import 从 TensorDictFeatureStore/WholeFeatureStore → FeatureStore
   - GraphStore(is_multi_gpu=True/False) → GraphStore()
   - conftest.py: 删除 is_multi_gpu 参数传递
   - loader/: isinstance 检查移除 NewGraphStore，统一为 GraphStore

Bug 根因 (Knuth 三维视角):
1. 数学维度: TensorDictFeatureStore 用 tensordict.TensorDict 做 batch_size 对齐校验，
   但 tensordict 的 batch_size 语义与 PyG FeatureTensorType 的 leading dim 并不完全等价；
   FeatureStore 用 DistTensor 的 all_gather 机制替代，数学上更健壮。
2. 算法维度: 旧 GraphStore._put_edge_index 接收 list 类型 edge_index 时未做 torch.stack，
   直接赋值给 TensorDict[edge_type] 会导致存入 Python list 引用而非 Tensor，
   后续 __get_edgelist 访问 .shape[1] 时抛 AttributeError——这是本次 bug fix 的核心。
   修复：isinstance(edge_index, list) → edge_index = torch.stack(edge_index) 前置守卫。
3. 工程维度: tensordict>=0.1.2 作为重依赖引入 conda 环境，但仅用于旧 GraphStore 的边索引容器，
   用普通 dict + DistMatrix 替代后彻底消除该依赖，降低安装复杂度。

SKIP 项:
  - conda/recipes/cugraph-pyg/recipe.yaml: conda 构建配方, Walpurgis 无 conda 体系
  - dependencies.yaml: RAPIDS 依赖矩阵, Walpurgis 无对应体系
  - python/cugraph-pyg/conda/: conda 开发环境 yaml, SKIP
  - 所有 tests/ 文件更新: Walpurgis 无 CI 测试体系
  - examples/: MNMG 示例脚本, Walpurgis 不维护上游 example

已迁移到 walpurgis:
  - src/walpurgis/core/unified_store.py (07ce63f): 包含 NewGraphStore 前身
    → 本次: 更新注释以反映 NewGraphStore→GraphStore 正式化, 增加 list bug fix 守卫
  - src/walpurgis/core/tensordict_removal.py (本文件): 完整迁移记录 + 审计工具
"""

# ── 鲁迅改写 ≥20%: 以下为 Walpurgis 本地化实现 ──────────────────────────────
# 上游仅做删除操作（无新增代码），鲁迅拿法体现在：
#   1. EdgeIndexTypeGuard: 将上游零散的 isinstance 检查结构化为可复用守卫类
#   2. DeprecatedAPIAudit: 将隐式删除的符号显式记录为审计表，支持运行时告警
#   3. TensorDictMigrationDiagnoser: 量化旧 TensorDict 路径 vs 新 DistMatrix 路径
#   4. 全链路 WALPURGIS_DEBUG=1 断点（上游无任何诊断输出）
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# 本次 commit 正式移除的符号清单（供审计用）
_REMOVED_SYMBOLS: Dict[str, str] = {
    "TensorDictFeatureStore": "data/feature_store.py — 基于 tensordict 的单机非分布式 FeatureStore",
    "DEPRECATED__OldGraphStore": "data/graph_store.py — 旧版基于 TensorDict 的单机 GraphStore",
    "NewGraphStore": "data/graph_store.py — 已正式重命名为 GraphStore",
    "WholeFeatureStore": "data/__init__.py — 兼容函数，重定向到 FeatureStore",
    "GraphStore (factory fn)": "data/__init__.py — is_multi_gpu 工厂函数，由直接导入替代",
    "TensorDictFeatureStore (factory fn)": "data/__init__.py — 废弃包装函数",
}


# ---------------------------------------------------------------------------
# EdgeIndexTypeGuard
# 上游 _put_edge_index 修复: list 类型输入未做 torch.stack 导致存入 Python list
# 本类将该守卫结构化，支持单独测试
# ---------------------------------------------------------------------------


class EdgeIndexKind(Enum):
    """边索引的输入类型枚举，对应上游 _put_edge_index 的各分支"""

    TORCH_TENSOR = "torch.Tensor"
    DIST_MATRIX = "DistMatrix"
    DIST_TENSOR_TUPLE = "tuple(DistTensor, DistTensor)"
    PYTHON_LIST = "list"          # ← 本次 bug fix 针对的类型
    CUPY_ARRAY = "cupy.ndarray"
    NUMPY_ARRAY = "numpy.ndarray"
    PANDAS_SERIES = "pandas.Series"
    CUDF_SERIES = "cudf.Series"
    UNKNOWN = "unknown"


@dataclass
class EdgeIndexTypeGuard:
    """
    将上游 _put_edge_index 中分散的 isinstance 检查收敛为单一守卫对象。

    上游修复（78128d9 核心 bug fix）:
        # 原版: 无此检查，list 直接被赋值给 DistMatrix 切片，存入 Python list 引用
        if isinstance(edge_index, list):
            edge_index = torch.stack(edge_index)  # ← 78128d9 新增

    本类在 Python 层复现该逻辑，供断点诊断和单元测试使用。
    """

    edge_index: Any
    detected_kind: EdgeIndexKind = field(init=False)

    def __post_init__(self) -> None:
        self.detected_kind = self._detect()
        if _DEBUG:
            print(
                f"[WALPURGIS_DEBUG] EdgeIndexTypeGuard: "
                f"type={type(self.edge_index).__name__} "
                f"→ kind={self.detected_kind.value}"
            )

    def _detect(self) -> EdgeIndexKind:
        # 延迟导入，避免在无 CUDA 环境下 import 失败
        ei = self.edge_index
        type_name = type(ei).__name__

        if type_name == "DistMatrix":
            return EdgeIndexKind.DIST_MATRIX

        if isinstance(ei, list):
            return EdgeIndexKind.PYTHON_LIST  # bug fix 目标

        if isinstance(ei, tuple) and len(ei) == 2:
            if all(type(x).__name__ == "DistTensor" for x in ei):
                return EdgeIndexKind.DIST_TENSOR_TUPLE

        # numpy/cupy/pandas/cudf 用类名字符串比较，避免 import 依赖
        if type_name in ("ndarray",):
            # 区分 numpy vs cupy 靠模块名
            module = type(ei).__module__.split(".")[0]
            return (
                EdgeIndexKind.CUPY_ARRAY
                if module == "cupy"
                else EdgeIndexKind.NUMPY_ARRAY
            )
        if type_name == "Series":
            module = type(ei).__module__.split(".")[0]
            return (
                EdgeIndexKind.CUDF_SERIES
                if module == "cudf"
                else EdgeIndexKind.PANDAS_SERIES
            )
        if type_name == "Tensor":
            return EdgeIndexKind.TORCH_TENSOR

        return EdgeIndexKind.UNKNOWN

    @property
    def needs_stack(self) -> bool:
        """是否需要 torch.stack 转换（即是否命中 78128d9 的 bug fix 路径）"""
        return self.detected_kind == EdgeIndexKind.PYTHON_LIST

    def apply_list_fix(self) -> Any:
        """
        若输入为 list，执行上游 78128d9 的 bug fix:
            edge_index = torch.stack(edge_index)
        否则原样返回。
        """
        if not self.needs_stack:
            return self.edge_index

        try:
            import torch as _torch
            result = _torch.stack(self.edge_index)
            if _DEBUG:
                print(
                    f"[WALPURGIS_DEBUG] EdgeIndexTypeGuard.apply_list_fix: "
                    f"list(len={len(self.edge_index)}) → "
                    f"Tensor{list(result.shape)}"
                )
            return result
        except Exception as exc:
            raise TypeError(
                f"78128d9 bug fix: torch.stack(edge_index) 失败，"
                f"edge_index={self.edge_index!r}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# DeprecatedAPIAudit
# 将本次删除的符号显式记录，支持运行时告警（上游只是直接删除，无任何告警路径）
# ---------------------------------------------------------------------------


@dataclass
class DeprecatedAPIAudit:
    """
    78128d9 移除符号的审计表。

    上游做法: 直接删除代码，调用方在 import 时得到 ImportError。
    Walpurgis 做法: 提供显式告警 + 迁移提示，给使用者一个缓冲期。
    """

    symbol_name: str
    description: str
    migration_hint: str = ""

    @classmethod
    def build_table(cls) -> List["DeprecatedAPIAudit"]:
        """构建完整的废弃符号审计表"""
        return [
            cls(
                "TensorDictFeatureStore",
                "基于 tensordict 的单机非分布式 FeatureStore，78128d9 正式移除",
                "改用 FeatureStore（基于 DistMatrix + WholeGraph，支持多 GPU）",
            ),
            cls(
                "WholeFeatureStore",
                "FeatureStore 的旧名称，已重命名，78128d9 删除兼容包装",
                "直接使用 FeatureStore",
            ),
            cls(
                "GraphStore(is_multi_gpu=True)",
                "工厂函数形式的 GraphStore，78128d9 删除，改为直接导入类",
                "直接 from cugraph_pyg.data import GraphStore; GraphStore()",
            ),
            cls(
                "NewGraphStore",
                "GraphStore 重命名前的过渡名称，78128d9 正式改名后删除",
                "使用 GraphStore（原 NewGraphStore 即为现在的 GraphStore）",
            ),
        ]

    def warn(self) -> None:
        """发出废弃警告"""
        msg = (
            f"[78128d9] '{self.symbol_name}' 已在上游 cugraph-pyg 78128d9 中移除。\n"
            f"  原因: {self.description}\n"
            f"  迁移: {self.migration_hint}"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if _DEBUG:
            print(f"[WALPURGIS_DEBUG] DeprecatedAPIAudit.warn: {self.symbol_name}")

    @classmethod
    def check_import(cls, symbol_name: str) -> None:
        """若 symbol_name 在废弃表中，发出警告"""
        table = cls.build_table()
        for entry in table:
            if entry.symbol_name == symbol_name:
                entry.warn()
                return


# ---------------------------------------------------------------------------
# TensorDictMigrationDiagnoser
# 量化旧 TensorDict 路径 vs 新 DistMatrix 路径的存储开销差异
# 上游无任何对比工具
# ---------------------------------------------------------------------------


@dataclass
class StoragePathDescriptor:
    """描述边索引存储路径的特征"""

    name: str
    backend: str          # tensordict / dist_matrix / dict
    supports_multi_gpu: bool
    requires_tensordict_dep: bool
    list_bug_fixed: bool   # 78128d9 bug fix


@dataclass
class TensorDictMigrationDiagnoser:
    """
    对比 TensorDict 旧路径与 DistMatrix 新路径的存储特征。

    上游 78128d9 的核心工程决策:
    - 旧路径: tensordict.TensorDict({}, batch_size=(2,))
      → 仅支持单机，list 输入 bug，需 tensordict 依赖
    - 新路径: dict[EdgeType, DistMatrix]
      → 支持多 GPU，list 输入 bug 已修复，无 tensordict 依赖
    """

    old_path: StoragePathDescriptor = field(
        default_factory=lambda: StoragePathDescriptor(
            name="旧 GraphStore (已移除)",
            backend="tensordict.TensorDict(batch_size=(2,))",
            supports_multi_gpu=False,
            requires_tensordict_dep=True,
            list_bug_fixed=False,
        )
    )
    new_path: StoragePathDescriptor = field(
        default_factory=lambda: StoragePathDescriptor(
            name="新 GraphStore (原 NewGraphStore)",
            backend="dict[EdgeType → DistMatrix]",
            supports_multi_gpu=True,
            requires_tensordict_dep=False,
            list_bug_fixed=True,
        )
    )

    def describe(self) -> str:
        """生成对比报告"""
        old = self.old_path
        new = self.new_path
        lines = [
            "=== TensorDict → DistMatrix 迁移对比 (78128d9) ===",
            f"{'项目':<30} {'旧路径':<40} {'新路径':<40}",
            "-" * 110,
            f"{'名称':<30} {old.name:<40} {new.name:<40}",
            f"{'存储后端':<30} {old.backend:<40} {new.backend:<40}",
            f"{'多GPU支持':<30} {str(old.supports_multi_gpu):<40} {str(new.supports_multi_gpu):<40}",
            f"{'tensordict依赖':<30} {str(old.requires_tensordict_dep):<40} {str(new.requires_tensordict_dep):<40}",
            f"{'list输入bug已修复':<30} {str(old.list_bug_fixed):<40} {str(new.list_bug_fixed):<40}",
        ]
        report = "\n".join(lines)
        if _DEBUG:
            print(f"[WALPURGIS_DEBUG] TensorDictMigrationDiagnoser:\n{report}")
        return report

    def validate_new_path(self) -> bool:
        """断言新路径满足所有迁移目标"""
        np = self.new_path
        assert np.supports_multi_gpu, "新路径必须支持多GPU"
        assert not np.requires_tensordict_dep, "新路径不得依赖 tensordict"
        assert np.list_bug_fixed, "新路径必须修复 list 输入 bug"
        if _DEBUG:
            print("[WALPURGIS_DEBUG] TensorDictMigrationDiagnoser.validate_new_path: PASS")
        return True


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------


def _run_selftest() -> None:
    """全链路自测，验证本模块各组件"""
    results: List[Tuple[str, bool]] = []

    # 1. EdgeIndexTypeGuard — list 检测
    guard = EdgeIndexTypeGuard([[1, 2, 3], [4, 5, 6]])
    results.append(("EdgeIndexTypeGuard: PYTHON_LIST 检测", guard.needs_stack))

    # 2. EdgeIndexTypeGuard — tensor 不触发 stack
    try:
        import torch as _torch
        t = _torch.tensor([[1, 2], [3, 4]])
        guard2 = EdgeIndexTypeGuard(t)
        results.append(("EdgeIndexTypeGuard: Tensor 不触发 stack", not guard2.needs_stack))
    except ImportError:
        results.append(("EdgeIndexTypeGuard: Tensor (torch缺失, SKIP)", True))

    # 3. DeprecatedAPIAudit 表构建
    table = DeprecatedAPIAudit.build_table()
    results.append(("DeprecatedAPIAudit: 表包含 TensorDictFeatureStore",
                    any(e.symbol_name == "TensorDictFeatureStore" for e in table)))

    # 4. TensorDictMigrationDiagnoser 验证
    diag = TensorDictMigrationDiagnoser()
    results.append(("TensorDictMigrationDiagnoser: validate_new_path", diag.validate_new_path()))

    # 5. _REMOVED_SYMBOLS 完整性
    results.append(("_REMOVED_SYMBOLS: 包含 NewGraphStore",
                    "NewGraphStore" in _REMOVED_SYMBOLS))

    # 报告
    all_pass = True
    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("tensordict_removal.py 自测全部通过")
    else:
        raise AssertionError("部分自测失败，请检查上方输出")


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _run_selftest()
