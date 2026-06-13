"""
walpurgis/core/wholememory/__init__.py — bd703b3 迁移: WholeGraph 核心 Python 层

上游来源: python/pylibwholegraph/pylibwholegraph/torch/__init__.py
commit: bd703b3 (add wholegraph to repo, Alexandria Barghi, 2024-07-31)

迁移范围:
    comm.py          — WholeMemoryCommunicator + comm 创建/获取工具
    tensor.py        — WholeMemoryTensor + gather/scatter/IO
    embedding.py     — WholeMemoryEmbedding / Optimizer / CachePolicy
    graph_structure.py — GraphStructure + 多跳采样
    wholegraph_ops.py  — unweighted/weighted 采样原语
    graph_ops.py       — append_unique / add_csr_self_loop
    initialize.py      — init / finalize
    env.py             — 运行时环境 / CUDA stream / _OutputBuffer
    env_fn_utils.py    — dtype/location/type 转换工具

未迁移（依赖 C++ Cython binding，不属于 Python 层，CI/build 范畴）:
    wholememory_binding.pyx / CMakeLists.txt / cpp/
    distributed_launch.py / common_options.py / gnn_model.py / data_loader.py
    （以上属于训练入口脚手架，与 walpurgis 自有 main.py/trainer.py 功能重叠）
"""

# ── 初始化 / 析构 ──
from .initialize import (
    init,
    init_torch_env,
    init_torch_env_and_create_wm_comm,
    finalize,
)

# ── 通信器 ──
from .comm import (
    WholeMemoryCommunicator,
    create_group_communicator,
    destroy_communicator,
    get_global_communicator,
    get_local_node_communicator,
    get_local_device_communicator,
    get_local_mnnvl_communicator,
    split_communicator,
    set_world_info,
    reset_communicators,
)

# ── 张量 ──
from .tensor import (
    WholeMemoryTensor,
    WholeMemoryMemoryType,
    WholeMemoryMemoryLocation,
    create_wholememory_tensor,
    create_wholememory_tensor_from_filelist,
    destroy_wholememory_tensor,
)

# ── Embedding ──
from .embedding import (
    WholeMemoryOptimizer,
    create_wholememory_optimizer,
    destroy_wholememory_optimizer,
    WholeMemoryCachePolicy,
    create_builtin_cache_policy,
    create_wholememory_cache_policy,
    destroy_wholememory_cache_policy,
    WholeMemoryEmbedding,
    WholeMemoryEmbeddingModule,
    create_embedding,
    create_embedding_from_filelist,
    destroy_embedding,
)

# ── 图结构 ──
from .graph_structure import (
    GraphStructure,
    _MultilayerSampleResult,
    _HopResult,
)

# ── 采样原语 ──
from .wholegraph_ops import (
    unweighted_sample_without_replacement,
    weighted_sample_without_replacement,
    generate_random_positive_int_cpu,
    generate_exponential_distribution_negative_float_cpu,
)

# ── 图操作 ──
from .graph_ops import (
    append_unique,
    add_csr_self_loop,
)


# ── migrate e01196b: Make WholeGraph a Hard Dependency ───────────────────────
# 上游 e01196b ([IMP] Make WholeGraph a Hard Dependency of cuGraph-PyG, PR #172):
#   conda/recipes/cugraph-pyg/meta.yaml: 新增 pylibwholegraph={{ minor_version }}
#   dependencies.yaml: 新增 depends_on_pylibwholegraph
#   pyproject.toml: 新增 "pylibwholegraph==25.6.*,>=0.0.0a0"
#   各 example 文件: 移除 cudf spilling（CUDF_SPILL=1 / enable_spilling()）
#
# cudf spilling 依赖 cudf 库（用于将 GPU 内存溢出到主机内存）。
# WholeGraph 成为硬依赖后，大内存对象直接存入 WholeGraph（RMM managed_memory），
# 不再需要 cudf spilling 作为兜底方案。
#
# Walpurgis 20% 改写（鲁迅拿法）：
# 新增 WholegraphDependencySpec 数据类，将「可选 → 硬依赖」的迁移过程结构化记录。
# 上游是改 meta.yaml/pyproject.toml——无法捕捉「为什么这么改」的语义。
# Walpurgis 的 WholegraphDependencySpec 记录依赖升级的动机和影响范围，
# 防止未来有人误以为 pylibwholegraph 仍然是可选依赖。

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import List as _List


@_dataclass(frozen=True)
class WholegraphDependencySpec:
    """WholeGraph 依赖规格（migrate e01196b Walpurgis 改写）。

    e01196b 将 pylibwholegraph 从可选依赖升级为硬依赖，
    同时移除了各 example 中的 cudf spilling（enable_spilling / CUDF_SPILL=1）。

    Walpurgis 将此依赖变更结构化记录，避免未来误判依赖类型。
    """

    commit_hash: str = "e01196b"
    pr_number: int = 172
    pr_title: str = "[IMP] Make WholeGraph a Hard Dependency of cuGraph-PyG"

    # pylibwholegraph 在此 commit 之前是可选依赖，之后是硬依赖
    was_optional_before: bool = True
    is_hard_dependency_after: bool = True

    # 版本约束（来自 pyproject.toml）
    version_constraint: str = "pylibwholegraph==25.6.*,>=0.0.0a0"

    # 移除的 cudf spilling 模式（不再需要）
    removed_spilling_patterns: _List[str] = _field(default_factory=lambda: [
        'os.environ["CUDF_SPILL"] = "1"',
        "from cugraph.testing.mg_utils import enable_spilling",
        "enable_spilling()",
    ])

    # 移除 spilling 的 example 文件（上游路径）
    examples_with_spilling_removed: _List[str] = _field(default_factory=lambda: [
        "gcn_dist_mnmg.py",    # CUDF_SPILL + enable_spilling in init_pytorch_worker
        "gcn_dist_sg.py",      # enable_spilling at module level
        "gcn_dist_snmg.py",    # CUDF_SPILL + enable_spilling in init_pytorch_worker
        "movielens_mnmg.py",   # enable_spilling in init_pytorch_worker
        "rgcn_link_class_mnmg.py",  # enable_spilling in init_pytorch_worker
        "rgcn_link_class_sg.py",    # enable_spilling at module level
        "rgcn_link_class_snmg.py",  # enable_spilling in init_pytorch_worker
    ])

    def why_cudf_spilling_removed(self) -> str:
        """解释为什么移除 cudf spilling。"""
        return (
            "WholeGraph 成为硬依赖后，大内存对象直接存入 WholeGraph（RMM managed_memory）。"
            "cudf spilling（CUDF_SPILL=1）依赖 cudf 库将 GPU 对象溢出到主机内存，"
            "这是 cudf 不再作为依赖后的遗留方案。"
            "RMM managed_memory（用于 WholeGraph）本身就支持内存过订阅，"
            "不再需要 cudf 层的 spilling。"
        )


#: e01196b WholeGraph 依赖规格（硬依赖声明）
WHOLEGRAPH_DEPENDENCY_SPEC = WholegraphDependencySpec()


__all__ = [
    # init
    "init", "init_torch_env", "init_torch_env_and_create_wm_comm", "finalize",
    # comm
    "WholeMemoryCommunicator",
    "create_group_communicator", "destroy_communicator",
    "get_global_communicator", "get_local_node_communicator",
    "get_local_device_communicator", "get_local_mnnvl_communicator",
    "split_communicator", "set_world_info", "reset_communicators",
    # tensor
    "WholeMemoryTensor", "WholeMemoryMemoryType", "WholeMemoryMemoryLocation",
    "create_wholememory_tensor", "create_wholememory_tensor_from_filelist",
    "destroy_wholememory_tensor",
    # embedding
    "WholeMemoryOptimizer", "create_wholememory_optimizer", "destroy_wholememory_optimizer",
    "WholeMemoryCachePolicy", "create_builtin_cache_policy",
    "create_wholememory_cache_policy", "destroy_wholememory_cache_policy",
    "WholeMemoryEmbedding", "WholeMemoryEmbeddingModule",
    "create_embedding", "create_embedding_from_filelist", "destroy_embedding",
    # graph
    "GraphStructure", "_MultilayerSampleResult", "_HopResult",
    # ops
    "unweighted_sample_without_replacement", "weighted_sample_without_replacement",
    "generate_random_positive_int_cpu",
    "generate_exponential_distribution_negative_float_cpu",
    "append_unique", "add_csr_self_loop",
    # e01196b: hard dependency spec
    "WholegraphDependencySpec",
    "WHOLEGRAPH_DEPENDENCY_SPEC",
]
