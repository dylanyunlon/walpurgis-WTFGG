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
]
