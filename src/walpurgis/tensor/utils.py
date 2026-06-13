# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 539d0ad: tensor/utils.py
# migrate ac3c900 (partial): _sanitize_file_list — Python层文件路径规范化
# 鲁迅拿法20%改写笔记:
#   上游 utils.py 老老实实把 create_wg_dist_tensor / copy_host_global_tensor_to_local
#   排成一列, 没有任何调试入口——出了问题只能靠 CUDA 报错猜。
#   我们改写的要点:
#     1. 每个公共函数在 WALPURGIS_DEBUG=1 时打印关键参数与执行结果;
#     2. copy_host_global_tensor_to_local 加同步前后的断点确认 barrier 实际发生;
#     3. has_nvlink_network 捕获 LOCAL_WORLD_SIZE 缺失的现实情况 (上游会 KeyError);
#     4. is_empty / empty 加类型断言——上游对非 Tensor 输入默默返回错误结果;
#     5. ac3c900: _sanitize_file_list() — C层 PyUnicode_AsUTF8 → PyUnicode_AsUTF8String+strdup
#        修复的 Python 层配套防御: 具现化 file_list + UTF-8 编码探针 + 类型规范化。

import os
import sys
import time
from typing import Union, List

from walpurgis.utils.imports import import_optional

torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印: 只在 WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(
            f"[WALPURGIS-TENSOR:utils][{time.strftime('%H:%M:%S')}][{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ─── copy_host_global_tensor_to_local ────────────────────────────────────────


def copy_host_global_tensor_to_local(wm_tensor, host_tensor, wm_comm):
    """将宿主全局张量的本地分片拷贝进 WholeGraph 管理张量。

    断点调试要点 (WALPURGIS_DEBUG=1):
      - 打印 local_start, local_tensor.shape 确认分区索引无误
      - barrier 前后各打一行, 确认集合通信实际执行
      - 若 host_tensor 在 CUDA 上会触发 copy_ 隐式同步警告
    """
    local_tensor, local_start = wm_tensor.get_local_tensor(host_view=False)

    _dbg(
        "copy_host→local",
        f"local_start={local_start} "
        f"local_shape={list(local_tensor.shape)} "
        f"host_shape={list(host_tensor.shape)} "
        f"dtype={host_tensor.dtype}",
    )

    # 鲁迅: 原文无越界守护——若 host_tensor 行数 < local_start + local行数 则静默越界
    end_idx = local_start + local_tensor.shape[0]
    if end_idx > host_tensor.shape[0]:
        raise IndexError(
            f"[copy_host_global_tensor_to_local] 越界: "
            f"local_start={local_start} + local_rows={local_tensor.shape[0]} "
            f"= {end_idx} > host_rows={host_tensor.shape[0]}"
        )

    local_tensor.copy_(host_tensor[local_start:end_idx])

    _dbg("copy_host→local", "copy_ 完成, 即将 barrier()")
    wm_comm.barrier()
    _dbg("copy_host→local", "barrier() 返回 ✓")


# ─── create_wg_dist_tensor ───────────────────────────────────────────────────


def create_wg_dist_tensor(
    shape: list,
    dtype: "torch.dtype",
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: str = "nccl",
    **kwargs,
):
    """创建 WholeGraph 管理的分布式张量。

    鲁迅改写:
      - "chunked" backend 上游根本没有处理, 直接落入 else 报 Unsupported;
        此处提前检测并给出可读提示。
      - 调试时打印 global_comm rank/size, 帮助排查"我的分区到底多大"。
    """
    _dbg(
        "create_wg_dist_tensor",
        f"shape={shape} dtype={dtype} location={location} "
        f"backend={backend} kwargs_keys={list(kwargs.keys())}",
    )

    global_comm = wgth.get_global_communicator()

    _dbg(
        "create_wg_dist_tensor",
        f"global_comm 获取成功 | "
        f"world_size={global_comm.world_size} rank={global_comm.rank}",
    )

    # backend → wholememory_type 映射
    # migrate df5bdc4: 新增 "hierarchy" backend，对应上游 embedding_memory_type="hierarchy"
    # hierarchy 模式: 两级内存层次 (GPU HBM + CPU DRAM)，只支持 nccl 通信，不支持 cache/nvshmem。
    if backend == "nccl":
        wm_type = "distributed"
    elif backend == "vmm":
        wm_type = "continuous"
    elif backend == "hierarchy":
        # migrate df5bdc4: hierarchy embedding type — 强制 nccl 通信后端
        # 上游检查: NVSHMEM 下不支持 hierarchy，cache_policy 下不支持 hierarchy。
        # 此处在 Walpurgis 层做提前检查，给出更清晰的错误信息（上游用 raise AssertionError 裸报错）。
        wm_type = "hierarchy"
        if "cache_policy" in kwargs and kwargs["cache_policy"] is not None:
            raise ValueError(
                "[hierarchy backend] cache_policy 与 hierarchy 内存类型不兼容；"
                "请去掉 cache_policy 或换用 'nccl'/'vmm' backend。"
            )
        _dbg(
            "create_wg_dist_tensor",
            "hierarchy backend: wm_type=hierarchy, 强制 nccl 通信 (不支持 nvshmem/cache)",
        )
    elif backend == "nvshmem":
        raise NotImplementedError(
            "NVSHMEM backend 尚未在 Walpurgis 中实现。"
            "请换用 'nccl' 或 'vmm'。"
        )
    elif backend == "chunked":
        raise NotImplementedError(
            "'chunked' backend 在 WholeGraph 公开 API 中尚未稳定, "
            "Walpurgis 暂不支持。参见上游 issue #253。"
        )
    else:
        raise ValueError(
            f"不支持的 backend: '{backend}'。"
            f"可选: 'nccl' | 'vmm' | 'hierarchy'。"
        )

    if "cache_policy" in kwargs:
        if len(shape) != 2:
            raise ValueError(
                "带 cache_policy 的 embedding 张量必须是 2D, "
                f"但 shape={shape}。"
            )
        cache_policy = kwargs.pop("cache_policy")
        wm_embedding = wgth.create_embedding(
            global_comm,
            wm_type,
            location,
            dtype,
            shape,
            cache_policy=cache_policy,
            embedding_entry_partition=partition_book,
            **kwargs,
        )
        _dbg("create_wg_dist_tensor", "create_embedding 完毕 (含 cache_policy)")
    else:
        if len(shape) not in [1, 2]:
            raise ValueError(
                f"张量 shape 必须是 1D 或 2D, 但 shape={shape}。"
            )
        wm_embedding = wgth.create_wholememory_tensor(
            global_comm,
            wm_type,
            location,
            shape,
            dtype,
            strides=None,
            tensor_entry_partition=partition_book,
        )
        _dbg("create_wg_dist_tensor", "create_wholememory_tensor 完毕")

    return wm_embedding


# ─── create_wg_dist_tensor_from_files ────────────────────────────────────────


def _sanitize_file_list(file_list) -> List[str]:
    """将文件路径列表统一规范化为 Python str 列表。

    ac3c900 迁移: 上游 wholememory_binding.pyx 修复了 PyUnicode_AsUTF8 的悬空指针问题
    (borrowed C string 在 GC 后失效), 改用 PyUnicode_AsUTF8String + strdup 获取稳定副本。
    这个修复在 C 层已由 pylibwholegraph >= 26.04 覆盖, 但调用方应在 Python 层也确保:
      1. file_list 是已具现化的 list (非 generator/iterator), 对象在调用期间不被 GC;
      2. 每个路径元素是 str 而非 bytes / pathlib.Path, 避免 C 层隐式转换失败;
      3. 路径可被 UTF-8 编码 (encode 探针验证), 与 strdup(PyBytes_AsString()) 一致。

    鲁迅: 内存安全不只是 C 层的责任——Python 调用方若传入惰性迭代器,
    C 扩展拿到的字符串地址指向的对象随时可能被回收。
    具现化 list 是最廉价的防御。
    """
    # 具现化: 防止惰性迭代器在 C 层遍历期间被 GC
    materialized: List[str] = list(file_list)

    sanitized: List[str] = []
    for i, fpath in enumerate(materialized):
        if isinstance(fpath, bytes):
            # bytes → str: 假设 UTF-8 编码
            fpath = fpath.decode("utf-8")
            _dbg("_sanitize_file_list", f"[{i}] bytes→str: {fpath!r}")
        elif hasattr(fpath, "__fspath__"):
            # pathlib.Path 或其他 os.fspath 兼容对象
            fpath = os.fspath(fpath)
            _dbg("_sanitize_file_list", f"[{i}] fspath→str: {fpath!r}")
        elif not isinstance(fpath, str):
            raise TypeError(
                f"file_list[{i}] 必须是 str/bytes/PathLike, "
                f"实际类型: {type(fpath).__name__!r}"
            )
        # UTF-8 编码探针: 确保字符串可被 C 层 PyUnicode_AsUTF8String 处理
        try:
            fpath.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"file_list[{i}]={fpath!r} 包含无法 UTF-8 编码的字符: {exc}"
            ) from exc
        sanitized.append(fpath)

    _dbg(
        "_sanitize_file_list",
        f"规范化完毕: {len(sanitized)} 条路径, "
        (repr(sanitized[0]) if sanitized else '(空)'),
    )
    return sanitized


def create_wg_dist_tensor_from_files(
    file_list: List[str],
    shape: list,
    dtype: "torch.dtype",
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: str = "nccl",
    **kwargs,
):
    """从文件列表创建 WholeGraph 管理的分布式张量。

    鲁迅改写:
      - 上游用 assert 做校验, assert 在 -O 模式下被跳过 → 改用 raise;
      - 打印每个文件是否实际存在 (WALPURGIS_DEBUG=1), 避免 CUDA 报错让人找不到原因;
      - ac3c900 迁移: _sanitize_file_list() 具现化并规范化路径列表,
        与 C 层 PyUnicode_AsUTF8String + strdup 修复配套。
    """
    # ac3c900: 规范化路径列表 (防 C 层借用指针悬空)
    file_list = _sanitize_file_list(file_list)

    _dbg(
        "create_wg_dist_tensor_from_files",
        f"file_count={len(file_list)} shape={shape} dtype={dtype} "
        f"location={location} backend={backend}",
    )

    # 断点: 逐文件检查存在性
    if _DEBUG:
        for fpath in file_list:
            exists = os.path.exists(fpath)
            _dbg(
                "file_check",
                f"{'✓' if exists else '✗ 不存在'} {fpath}",
            )

    global_comm = wgth.get_global_communicator()

    # migrate df5bdc4: 新增 hierarchy backend（从文件加载路径）
    if backend == "nccl":
        wm_type = "distributed"
    elif backend == "vmm":
        wm_type = "continuous"
    elif backend == "hierarchy":
        wm_type = "hierarchy"
        if "cache_policy" in kwargs and kwargs["cache_policy"] is not None:
            raise ValueError(
                "[hierarchy backend] cache_policy 与 hierarchy 不兼容，"
                "请去掉 cache_policy 或换用 'nccl'/'vmm'。"
            )
        _dbg("create_wg_dist_tensor_from_files", "hierarchy backend: wm_type=hierarchy")
    elif backend == "nvshmem":
        raise NotImplementedError(
            "NVSHMEM backend 尚未在 Walpurgis 中实现。"
        )
    else:
        raise ValueError(f"不支持的 backend: '{backend}'。可选: 'nccl' | 'vmm' | 'hierarchy'。")

    if "cache_policy" in kwargs:
        if len(shape) != 2:
            raise ValueError(
                f"带 cache_policy 时 shape 必须是 2D, 但 shape={shape}。"
            )
        cache_policy = kwargs.pop("cache_policy")
        wm_embedding = wgth.create_embedding_from_filelist(
            global_comm,
            wm_type,
            location,
            file_list,
            dtype,
            shape[1],
            cache_policy=cache_policy,
            embedding_entry_partition=partition_book,
            **kwargs,
        )
        _dbg("create_wg_dist_tensor_from_files", "create_embedding_from_filelist 完毕")
    else:
        if len(shape) not in [1, 2]:
            raise ValueError(
                f"张量 shape 必须是 1D 或 2D, 但 shape={shape}。"
            )
        last_dim_size = 0 if len(shape) == 1 else shape[1]
        wm_embedding = wgth.create_wholememory_tensor_from_filelist(
            global_comm,
            wm_type,
            location,
            file_list,
            dtype,
            last_dim_size,
            tensor_entry_partition=partition_book,
        )
        _dbg(
            "create_wg_dist_tensor_from_files",
            f"create_wholememory_tensor_from_filelist 完毕 "
            f"last_dim_size={last_dim_size}",
        )

    return wm_embedding


# ─── has_nvlink_network ───────────────────────────────────────────────────────


def has_nvlink_network() -> bool:
    r"""检测当前硬件是否支持跨节点 NVLink 网络。

    鲁迅改写:
      - 上游直接 int(os.environ["LOCAL_WORLD_SIZE"]) → KeyError 如果环境变量未设;
        此处改为 get + 兜底, 并在 WALPURGIS_DEBUG 时给出提示。
    """
    global_comm = wgth.comm.get_global_communicator("nccl")
    local_world_size_str = os.environ.get("LOCAL_WORLD_SIZE", "")

    if not local_world_size_str:
        _dbg(
            "has_nvlink_network",
            "LOCAL_WORLD_SIZE 未设置, 假定单节点 (local_size = world_size)",
        )
        # 单节点情况: 只看 p2p 支持
        return global_comm.support_type_location("continuous", "cuda")

    local_size = int(local_world_size_str)
    world_size = torch.distributed.get_world_size()

    _dbg(
        "has_nvlink_network",
        f"local_size={local_size} world_size={world_size}",
    )

    if local_size == world_size:
        result = global_comm.support_type_location("continuous", "cuda")
        _dbg("has_nvlink_network", f"单节点 p2p 结果: {result}")
        return result

    is_cuda_ok = global_comm.support_type_location("continuous", "cuda")
    is_cpu_ok = global_comm.support_type_location("continuous", "cpu")
    result = is_cuda_ok and is_cpu_ok

    _dbg(
        "has_nvlink_network",
        f"多节点: cuda_ok={is_cuda_ok} cpu_ok={is_cpu_ok} → {result}",
    )
    return result


# ─── is_empty / empty ────────────────────────────────────────────────────────


def is_empty(a) -> bool:
    """判断张量是否为空 (numel == 0)。

    鲁迅改写: 上游对非 Tensor 参数静默调用 .numel() 触发 AttributeError;
    此处加类型检查给出可读错误。
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError(
            f"is_empty 期望 torch.Tensor, 实际收到 {type(a).__name__}。"
        )
    result = a.numel() == 0
    _dbg("is_empty", f"shape={list(a.shape)} → {result}")
    return result


def empty(dim: int = 1) -> "torch.Tensor":
    """返回指定维度的空 int32 张量。

    鲁迅改写: 上游只支持 dim=1/2, 其余 raise ValueError——
    但错误信息没说"支持哪些值", 此处改为枚举式提示。
    """
    _dbg("empty", f"dim={dim}")
    if dim == 1:
        return torch.tensor([], dtype=torch.int32)
    elif dim == 2:
        return torch.tensor([], dtype=torch.int32).view(0, 1)
    else:
        raise ValueError(
            f"empty() 仅支持 dim=1 或 dim=2, 传入了 dim={dim}。"
        )
