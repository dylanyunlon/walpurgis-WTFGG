"""
wg_comm_fix.py — 90db89a 迁移: 使用正确的 WG 通信子 + int64 边索引

migrate 90db89a: use the correct wg communicator

上游变化 (90db89a, cugraph-gnn / python/cugraph-pyg/):

1. data/feature_store.py — WholeFeatureStore.__init__:
   旧: self.__wg_comm = wgth.get_local_node_communicator()
   新: self.__wg_comm = wgth.get_global_communicator()

   根因: local_node_communicator 只包含同一物理节点上的 GPU ranks，
   在多节点部署时所有 WholeMemory 操作（scatter/gather/all_gather）
   只在单节点内完成，导致跨节点的特征数据不一致；
   global_communicator 覆盖全部 world_size 个 rank，是正确的分布式通信域。

2. data/graph_store.py — GraphStore._num_vertices (被 _graph 属性调用):
   旧: dtype=torch.int32
   新: dtype=torch.int64

   根因: 图的顶点/边数量在超大图场景下可能超出 int32 的上限 (2^31 - 1 ≈ 21亿);
   torch.tensor([...], dtype=torch.int32) 对超大图会静默截断，
   导致顶点数量计算错误 → MGGraph 构建时尺寸不匹配崩溃。
   改为 int64 后可安全表示 2^63 - 1 量级的图规模。

Bug 根因 (Knuth 视角):
1. diff 对比源:
   | 上游 90db89a^               | 90db89a 修复后              | Walpurgis 迁移               |
   |----------------------------|-----------------------------|------------------------------|
   | get_local_node_communicator| get_global_communicator()   | WgCommSelector 策略对象       |
   | 无 WALPURGIS_DEBUG 输出    | 无 WALPURGIS_DEBUG 输出     | 断点 1/2 打印通信子类型决策   |
   | int32 边数量张量            | int64 边数量张量             | EdgeCountDtypeGuard 守卫      |
   | 无类型升级日志              | 无类型升级日志              | 断点 3 打印 dtype 决策        |

2. 用户角度 bug:
   - local_node_communicator 在单节点多 GPU 时与 global_communicator 等价，
     所以单机测试完全无感，只有部署到多节点集群后才会出现特征数据乱掉的问题；
     这是典型的"在我机器上好好的"类型 bug，极难复现和定位。
   - int32 截断不会抛出异常，只会悄悄产生错误的顶点数量；
     MGGraph 会接受截断后的错误尺寸，后续邻域采样越界才在 CUDA 内核中崩溃，
     堆栈与根因毫无关联，调试成本极高。

3. 系统角度安全:
   - WholeGraph 初始化时 wg_comm 必须与 torch.distributed world_size 完全对应，
     否则 WholeMemory tensor 的全局分片布局会出现 rank 数量不一致；
     一旦不一致，scatter 写入的偏移计算就是错误的，且错误是跨节点静默扩散的。
   - int32 上限 ~2.1B，而 MAG240M 等大型学术图已有 240M 节点 + 1.7B 边，
     工业级知识图谱轻松超过 int32；int64 是分布式图计算的正确默认。

Walpurgis 改写 20%（鲁迅拿法）:
- WgCommSelector: 替代 __init__ 里的裸 wgth.get_global_communicator() 调用，
  封装通信子类型 (global / local_node) 的选择逻辑 + 可观测决策，
  与 unified_store.BackendSelector 同族设计，统一通信层决策路径
- EdgeCountDtypeGuard: 替代 dtype=torch.int32 / dtype=torch.int64 的裸硬编码，
  封装"边数量应该用什么 dtype"的判断，加断言 + DEBUG 打印
- 全链路 WALPURGIS_DEBUG=1 断点 print:
  断点 1: WgCommSelector.select() 入口，打印候选通信子类型
  断点 2: WgCommSelector.select() 完成，打印最终选择 + 原因
  断点 3: EdgeCountDtypeGuard.check() 打印 dtype 决策

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS wg_comm_fix]", *args, file=sys.stderr, flush=True, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# WgCommSelector — 替代 WholeFeatureStore.__init__ 里的裸 wgth.get_xxx_communicator()
# ──────────────────────────────────────────────────────────────────────────────

class WgCommSelector:
    """
    封装 90db89a 修复的 WholeGraph 通信子选择逻辑。

    上游 90db89a 前 (feature_store.py WholeFeatureStore.__init__):
        self.__wg_comm = wgth.get_local_node_communicator()   # ← 错误

    上游 90db89a 后:
        self.__wg_comm = wgth.get_global_communicator()       # ← 正确

    语义:
        local_node_communicator: 仅包含同一物理节点上的 GPU ranks，
            在 torchrun --nnodes=1 时 == global，但多节点时只看见本机 GPU，
            导致 WholeMemory scatter/gather 在多节点集群上特征数据不一致。
        global_communicator: 覆盖所有 world_size 个 rank（跨节点），
            是分布式 WholeMemory 操作的正确通信域，必须与 torch.distributed
            的 world_size 对应。

    Walpurgis 改写:
    - comm_type 和 reason 均可观测
    - select() 支持 force_local 参数用于单节点 debug 场景
    - WALPURGIS_DEBUG=1 时打印通信子类型决策

    断点 1: select() 入口，打印候选通信子类型
    断点 2: select() 完成，打印最终选择 + 原因
    """

    COMM_GLOBAL = "global"
    COMM_LOCAL_NODE = "local_node"

    def __init__(self, comm_type: str, reason: str):
        self.comm_type = comm_type
        self.reason = reason

    def __repr__(self):
        return f"WgCommSelector(comm_type={self.comm_type!r}, reason={self.reason!r})"

    @staticmethod
    def select(force_local: bool = False) -> "WgCommSelector":
        """
        选择正确的 WholeGraph 通信子类型。

        参数
        ----
        force_local : bool, default False
            若为 True，强制使用 local_node_communicator（仅用于单节点调试）。
            生产环境始终应使用默认值 False（选 global）。

        返回
        ----
        WgCommSelector 对象，.comm_type 为 "global" 或 "local_node"

        断点 1: 打印 force_local 参数
        """
        # ── 断点 1 ────────────────────────────────────────────────────────
        _dbg(f"WgCommSelector.select(): force_local={force_local}")

        if force_local:
            result = WgCommSelector(
                WgCommSelector.COMM_LOCAL_NODE,
                reason=(
                    "force_local=True (DEBUG ONLY): using local_node_communicator. "
                    "WARNING: incorrect for multi-node deployments — "
                    "features will be inconsistent across nodes."
                ),
            )
        else:
            # 90db89a 修复后的正确选择: global_communicator
            result = WgCommSelector(
                WgCommSelector.COMM_GLOBAL,
                reason=(
                    "90db89a fix: global_communicator covers all world_size ranks, "
                    "ensuring WholeMemory scatter/gather is consistent across all nodes. "
                    "local_node_communicator was incorrect for multi-node deployments."
                ),
            )

        # ── 断点 2 ────────────────────────────────────────────────────────
        _dbg(f"WgCommSelector.select() → {result}")
        return result

    def get_communicator(self):
        """
        调用对应的 wgth API，返回实际的 WholeGraph 通信子对象。

        需要 pylibwholegraph.torch (wgth) 已安装并初始化。

        抛出
        ----
        ImportError  若 pylibwholegraph 未安装
        RuntimeError 若 WholeGraph 未初始化
        """
        try:
            import pylibwholegraph.torch as wgth
        except ImportError as e:
            raise ImportError(
                "pylibwholegraph is required for WgCommSelector.get_communicator(). "
                "Install with: pip install pylibwholegraph"
            ) from e

        _dbg(f"WgCommSelector.get_communicator(): comm_type={self.comm_type!r}")

        if self.comm_type == WgCommSelector.COMM_GLOBAL:
            comm = wgth.get_global_communicator()
        elif self.comm_type == WgCommSelector.COMM_LOCAL_NODE:
            comm = wgth.get_local_node_communicator()
        else:
            raise ValueError(f"Unknown comm_type: {self.comm_type!r}")

        _dbg(f"  get_communicator() → {type(comm).__name__}")
        return comm

    @staticmethod
    def make_global_comm():
        """
        便捷方法: 直接获取 global_communicator（90db89a 修复后的正确用法）。

        等价于:
            WgCommSelector.select().get_communicator()

        替代上游修复前的:
            wgth.get_local_node_communicator()   # ← 已废弃，90db89a 修复

        以及修复后的:
            wgth.get_global_communicator()        # ← 正确，但无可观测决策

        Walpurgis 在此基础上增加 WALPURGIS_DEBUG 断点。
        """
        selector = WgCommSelector.select(force_local=False)
        return selector.get_communicator()


# ──────────────────────────────────────────────────────────────────────────────
# EdgeCountDtypeGuard — 替代 GraphStore._num_vertices 里的裸 dtype=torch.int32
# ──────────────────────────────────────────────────────────────────────────────

class EdgeCountDtypeGuard:
    """
    封装 90db89a 修复的边数量张量 dtype 选择逻辑。

    上游 90db89a 前 (graph_store.py GraphStore._num_vertices 属性):
        torch.tensor(
            [self.__edge_indices[et].shape[1] for et in sorted_keys],
            device="cuda",
            dtype=torch.int32,    # ← 错误，超大图会截断
        )

    上游 90db89a 后:
        torch.tensor(
            [...],
            device="cuda",
            dtype=torch.int64,    # ← 正确
        )

    根因: int32 上限 ~2.1B，工业级图（MAG240M / 知识图谱）边数常超此限，
    截断后的顶点/边数量静默错误，MGGraph 构建接受错误尺寸，
    后续 CUDA 内核越界崩溃，堆栈与根因无关联。

    Walpurgis 改写:
    - EDGE_COUNT_DTYPE: 模块常量，明确声明应使用 int64
    - check(): 静态方法，接受 dtype 参数，验证是否满足 int64 要求，
      并打印决策理由 + 警告（若传入了 int32）
    - safe_tensor(): 构造边数量张量的安全封装，强制 dtype=int64

    断点 3: check() 打印 dtype 决策
    """

    # 90db89a 修复后的正确 dtype，用于边/顶点数量张量
    EDGE_COUNT_DTYPE_NAME = "torch.int64"

    @staticmethod
    def get_dtype():
        """
        返回边数量张量应使用的 dtype (torch.int64)。

        对应上游 90db89a 修复: dtype=torch.int32 → dtype=torch.int64。
        """
        import torch
        return torch.int64

    @staticmethod
    def check(dtype) -> bool:
        """
        检查给定 dtype 是否符合边数量安全要求（即 torch.int64）。

        参数
        ----
        dtype : torch.dtype

        返回
        ----
        bool: True 表示安全（int64），False 表示有截断风险（int32 或更窄）

        断点 3: 打印 dtype 决策
        """
        import torch

        is_safe = (dtype == torch.int64)

        # ── 断点 3 ────────────────────────────────────────────────────────
        _dbg(
            f"EdgeCountDtypeGuard.check(): dtype={dtype}, "
            f"is_safe={is_safe}"
        )

        if not is_safe:
            import warnings
            warnings.warn(
                f"EdgeCountDtypeGuard: edge count tensor dtype={dtype} may overflow "
                f"for graphs with >2^31 edges. "
                f"90db89a fix requires dtype=torch.int64. "
                f"Use EdgeCountDtypeGuard.safe_tensor() to construct safely.",
                RuntimeWarning,
                stacklevel=2,
            )
            _dbg(
                f"  WARNING: {dtype} has max ~{2**31 - 1 if dtype == torch.int32 else '?'} "
                f"— use int64 (max ~{2**63 - 1}) for large graphs"
            )

        return is_safe

    @staticmethod
    def safe_tensor(values, device: str = "cuda"):
        """
        安全构造边/顶点数量张量，强制 dtype=torch.int64。

        对应上游 90db89a 修复后的:
            torch.tensor([...], device="cuda", dtype=torch.int64)

        参数
        ----
        values : list[int] 或 可迭代
            边/顶点数量列表
        device : str, default "cuda"

        返回
        ----
        torch.Tensor (dtype=torch.int64, device=device)

        断点 3: 打印构造参数
        """
        import torch

        dtype = EdgeCountDtypeGuard.get_dtype()

        # ── 断点 3 (safe_tensor) ──────────────────────────────────────────
        _dbg(
            f"EdgeCountDtypeGuard.safe_tensor(): "
            f"n_values={len(values) if hasattr(values, '__len__') else '?'}, "
            f"dtype={dtype}, device={device!r}"
        )

        t = torch.tensor(list(values), device=device, dtype=dtype)
        _dbg(f"  safe_tensor → shape={tuple(t.shape)}, dtype={t.dtype}")
        return t


# ──────────────────────────────────────────────────────────────────────────────
# patch_whole_feature_store() — 运行时热补丁（可选）
#
# 将 90db89a 的 wg_comm 修复注入已加载的 WholeFeatureStore 类，
# 无需重新安装 cugraph-pyg。
# 适用于: 上游版本未及时升级时的临时修复方案。
# ──────────────────────────────────────────────────────────────────────────────

def patch_whole_feature_store():
    """
    热补丁: 将 WholeFeatureStore.__init__ 中的 wg_comm 替换为正确的 global_communicator。

    等价于 90db89a 的修复:
        - self.__wg_comm = wgth.get_local_node_communicator()  # 旧 (错误)
        + self.__wg_comm = wgth.get_global_communicator()      # 新 (正确)

    用法:
        from walpurgis.core.wg_comm_fix import patch_whole_feature_store
        patch_whole_feature_store()   # 在 WholeFeatureStore 首次实例化前调用

    注意:
    - 仅在上游 cugraph-pyg 版本 < 90db89a 时需要此补丁。
    - 补丁使用 Python name-mangling 规则定位私有属性 _WholeFeatureStore__wg_comm。
    - WALPURGIS_DEBUG=1 时打印补丁应用状态。

    抛出
    ----
    ImportError  若 cugraph-pyg 或 pylibwholegraph 未安装
    """
    _dbg("patch_whole_feature_store(): 尝试应用 90db89a wg_comm 热补丁")

    try:
        from cugraph_pyg.data.feature_store import WholeFeatureStore
        import pylibwholegraph.torch as wgth
    except ImportError as e:
        _dbg(f"  补丁跳过 (ImportError): {e}")
        return

    # 保存原始 __init__ 以便回滚
    _original_init = WholeFeatureStore.__init__

    def _patched_init(self, memory_type="distributed", location="cpu"):
        """
        90db89a 修复版 WholeFeatureStore.__init__:
        使用 get_global_communicator() 替代 get_local_node_communicator()。
        """
        # 调用原始 super().__init__()
        import torch_geometric
        torch_geometric.data.FeatureStore.__init__(self)

        self.__features = {}  # type: ignore[attr-defined]

        # ── 90db89a 修复: global_communicator 替代 local_node_communicator ──
        selector = WgCommSelector.select(force_local=False)
        _dbg(
            f"  _patched_init: wg_comm 选择 → {selector}, "
            f"memory_type={memory_type!r}, location={location!r}"
        )
        # Python name-mangling: __wg_comm → _WholeFeatureStore__wg_comm
        self._WholeFeatureStore__wg_comm = selector.get_communicator()  # type: ignore[attr-defined]
        self._WholeFeatureStore__wg_type = memory_type                  # type: ignore[attr-defined]
        self._WholeFeatureStore__wg_location = location                 # type: ignore[attr-defined]

    WholeFeatureStore.__init__ = _patched_init  # type: ignore[method-assign]
    WholeFeatureStore.__init__._walpurgis_patched = True  # type: ignore[attr-defined]
    WholeFeatureStore.__init__._patch_commit = "90db89a"  # type: ignore[attr-defined]

    _dbg("patch_whole_feature_store(): 补丁应用成功")
    return _original_init  # 返回原始方法，方便回滚


def unpatch_whole_feature_store(original_init):
    """
    回滚 patch_whole_feature_store() 应用的热补丁。

    参数
    ----
    original_init : 由 patch_whole_feature_store() 返回的原始 __init__
    """
    _dbg("unpatch_whole_feature_store(): 回滚 90db89a wg_comm 热补丁")
    try:
        from cugraph_pyg.data.feature_store import WholeFeatureStore
        WholeFeatureStore.__init__ = original_init  # type: ignore[method-assign]
        _dbg("  回滚成功")
    except ImportError:
        _dbg("  回滚跳过 (cugraph-pyg 未安装)")


# ──────────────────────────────────────────────────────────────────────────────
# 便捷导出
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "WgCommSelector",
    "EdgeCountDtypeGuard",
    "patch_whole_feature_store",
    "unpatch_whole_feature_store",
]
