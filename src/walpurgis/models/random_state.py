"""
random_state.py — 9b89e8a 迁移: Set random state using PyTorch generator

migrate 9b89e8a: [FEA] Set random state using PyTorch generator

上游变化 (9b89e8a):
  1. loader/utils.py (新增文件):
     - generate_seed() 函数:
       rank=0 在 CUDA 上 torch.randint(0, 2**63 - world_size, (1,), dtype=int64)
       rank!=0 seed = torch.tensor([0], dtype=int64, device="cuda")
       torch.distributed.broadcast(seed, src=0)
       return seed.item() + rank
     → 每个 rank 得到唯一、确定性、可复现的 random seed
     → seed 空间 [0, 2**63 - world_size + rank], rank 间互不重叠
  2. loader/node_loader.py:
     - 新增: from .utils import generate_seed
     - sample_from_nodes(...) 调用增加 random_state=generate_seed()
  3. loader/link_loader.py:
     - 新增: from .utils import generate_seed
     - sample_from_edges(...) 调用增加 random_state=generate_seed()

核心设计:
  - rank 0 生成全局唯一随机种子, broadcast 到所有 rank
  - 每个 rank 加上自己的 rank 偏移量 → 各 rank seed 唯一且空间不重叠
  - 使用 CUDA tensor 确保 broadcast 走 GPU 直接通信 (性能)
  - int64 范围 [0, 2**63-1], 减去 world_size 防止 rank 偏移溢出

Walpurgis 改写20%(鲁迅拿法):
  - RandomStateConfig: 封装 generate_seed() 的全部参数和状态,
    替代 Python 的裸函数调用 + 隐式 distributed 状态依赖
  - WalpurgisRandGen: 替代 Python 内联的 torch.randint + broadcast 两步,
    封装为带重试/降级逻辑的生成器对象
  - generate_seed_safe(): 替代上游 generate_seed(), 加 non-distributed 降级
    (Python 在单机非 distributed 环境直接 crash)
  - SamplerRandomState: 对应 node_loader/link_loader 的 random_state= 注入点,
    封装 (loader_type, seed, rank, world_size) 四元组, 便于 audit/replay
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路 print
    - generate_seed 入口: 打印 rank/world_size
    - broadcast 前后: 打印 seed 值对比
    - 最终 return seed: 打印 final seed + rank 偏移
    - SamplerRandomState 构造: 打印 loader_type + seed

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
from enum import Enum
from typing import Optional, Callable, Any

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_rand(tag: str, msg: str) -> None:
    """断点调试: random state 专用 print"""
    if _DBG:
        print(f"[DEBUG 9b89e8a {tag}] {msg}", file=sys.stderr, flush=True)


# ─── LoaderType: 对应 node_loader / link_loader 两条调用路径 ──────────────────
# Python (9b89e8a):
#   node_loader.py: sample_from_nodes(input_data, random_state=generate_seed())
#   link_loader.py: sample_from_edges(input_data, ..., random_state=generate_seed())
# 改写: 枚举明示 loader 类型, 便于 SamplerRandomState audit
class LoaderType(Enum):
    NODE  = "node_loader"   # 对应 node_loader.py sample_from_nodes
    LINK  = "link_loader"   # 对应 link_loader.py sample_from_edges


# ─── RandomStateConfig: 封装 generate_seed 的运行时参数 ───────────────────────
# Python (9b89e8a utils.py):
#   def generate_seed():
#       world_size = torch.distributed.get_world_size()
#       rank = torch.distributed.get_rank()
#       ...
# 这两个值是隐式从 torch.distributed 全局状态读取的。
# 改写: 封装为配置对象, 参数显式传入, 支持单元测试时 mock
class RandomStateConfig:
    """
    封装 generate_seed() 运行时所需的 distributed 配置。

    Python (9b89e8a) 从全局 torch.distributed 读取 rank/world_size。
    改写: 显式配置对象, 参数可注入 (便于测试 + 调试)。

    改写比 Python 更结构化:
      - is_distributed 属性区分单机/分布式路径
      - seed_upper_bound 明示种子空间上界 (Python 内联 2**63 - world_size)
      - dump() 方法一键打印完整配置
    """

    # 上游 (9b89e8a utils.py:21): torch.randint(0, 2**63 - world_size, ...)
    # int64 安全上界, 减去 world_size 防止 rank 偏移溢出 2**63 - 1
    INT64_MAX: int = 2 ** 63 - 1

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        device: str = "cuda",           # 9b89e8a: device="cuda"
        dtype: str = "int64",           # 9b89e8a: dtype=torch.int64
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.dtype = dtype

        # 9b89e8a: torch.randint(0, 2**63 - world_size, ...)
        # 改写: 显式暴露上界 (Python 是内联字面量)
        self.seed_upper_bound: int = max(1, self.INT64_MAX - world_size)

        # 改写: 显式标注是否在分布式环境
        self.is_distributed: bool = (world_size > 1)

        _dbg_rand(
            "RandomStateConfig.__init__",
            f"rank={rank} world_size={world_size} device={device!r} "
            f"is_distributed={self.is_distributed} "
            f"seed_upper_bound={self.seed_upper_bound}"
        )

    def dump(self) -> None:
        """断点调试: 打印完整配置"""
        print(
            f"[DEBUG 9b89e8a RandomStateConfig] "
            f"rank={self.rank} world_size={self.world_size} "
            f"device={self.device!r} dtype={self.dtype} "
            f"is_distributed={self.is_distributed} "
            f"seed_upper_bound={self.seed_upper_bound}",
            file=sys.stderr
        )


# ─── WalpurgisRandGen: 封装 torch.randint + broadcast 两步 ───────────────────
# Python (9b89e8a utils.py:17-27):
#   if rank == 0:
#       seed = torch.randint(0, 2**63 - world_size, (1,), dtype=torch.int64, device="cuda")
#   else:
#       seed = torch.tensor([0], dtype=torch.int64, device="cuda")
#   torch.distributed.broadcast(seed, src=0)
#   seed = seed.item() + rank
#   return seed
#
# 改写: 封装为 WalpurgisRandGen, 带重试/降级 + 调试输出
# Python 在非 distributed 环境 get_world_size() 报错; 改写加 non-distributed 降级
class WalpurgisRandGen:
    """
    Random seed generator mirroring 9b89e8a generate_seed().

    Python: 裸函数, 隐式依赖 torch.distributed 全局状态。
    改写: 类封装, 接受 RandomStateConfig, 支持 non-distributed 降级。

    主要方法:
      generate(config) → int  对应 generate_seed() 完整逻辑
      _rank0_generate() → tensor  对应 rank==0 的 randint 路径
      _nonrank_placeholder() → tensor  对应 rank!=0 的 zeros 路径
    """

    def __init__(self, torch_module: Any = None):
        """
        Args:
            torch_module: 注入 torch (默认 None → 运行时 import)
                          便于单元测试时注入 mock
        """
        self._torch = torch_module

    def _get_torch(self) -> Any:
        """延迟 import torch (与上游保持一致: torch = import_optional("torch"))"""
        if self._torch is None:
            try:
                import torch as _t
                self._torch = _t
            except ImportError:
                raise RuntimeError(
                    "[9b89e8a WalpurgisRandGen] torch not available. "
                    "Install torch to use generate_seed."
                )
        return self._torch

    def _rank0_generate(self, config: RandomStateConfig):
        """
        对应 9b89e8a utils.py:21-24 rank==0 路径:
            seed = torch.randint(
                0, 2**63 - world_size, (1,), dtype=torch.int64, device="cuda"
            )

        断点调试: 打印生成的原始 seed 值
        """
        torch = self._get_torch()
        seed_tensor = torch.randint(
            0,
            config.seed_upper_bound,
            (1,),
            dtype=torch.int64,
            device=config.device,
        )
        _dbg_rand(
            "WalpurgisRandGen._rank0_generate",
            f"rank=0 raw_seed={seed_tensor.item()} "
            f"upper_bound={config.seed_upper_bound} device={config.device!r}"
        )
        return seed_tensor

    def _nonrank_placeholder(self, config: RandomStateConfig):
        """
        对应 9b89e8a utils.py:25-26 rank!=0 路径:
            seed = torch.tensor([0], dtype=torch.int64, device="cuda")
        (将被 broadcast 覆盖为 rank 0 生成的值)

        断点调试: 打印 rank + placeholder 值
        """
        torch = self._get_torch()
        placeholder = torch.tensor([0], dtype=torch.int64, device=config.device)
        _dbg_rand(
            "WalpurgisRandGen._nonrank_placeholder",
            f"rank={config.rank} placeholder=0 (waiting for broadcast)"
        )
        return placeholder

    def generate(self, config: RandomStateConfig) -> int:
        """
        对应 9b89e8a generate_seed() 完整逻辑:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if rank == 0:
                seed = torch.randint(0, 2**63 - world_size, (1,), ...)
            else:
                seed = torch.tensor([0], ...)
            torch.distributed.broadcast(seed, src=0)
            seed = seed.item() + rank
            return seed

        改写: 接受 RandomStateConfig, 加 non-distributed 降级路径
        断点调试: 打印 broadcast 前/后 seed 值 + 最终 rank 偏移
        """
        torch = self._get_torch()

        _dbg_rand(
            "WalpurgisRandGen.generate",
            f"rank={config.rank} world_size={config.world_size} "
            f"is_distributed={config.is_distributed}"
        )

        # 9b89e8a: if rank == 0 → randint; else → zeros
        if config.rank == 0:
            seed_tensor = self._rank0_generate(config)
        else:
            seed_tensor = self._nonrank_placeholder(config)

        # 9b89e8a: torch.distributed.broadcast(seed, src=0)
        # 改写: non-distributed 环境跳过 broadcast
        if config.is_distributed:
            try:
                pre_broadcast_val = seed_tensor.item()
                torch.distributed.broadcast(seed_tensor, src=0)
                post_broadcast_val = seed_tensor.item()
                _dbg_rand(
                    "WalpurgisRandGen.generate",
                    f"broadcast: rank={config.rank} "
                    f"pre={pre_broadcast_val} post={post_broadcast_val}"
                )
            except Exception as e:
                # 降级: distributed 不可用时 warn + 继续 (Python 直接 crash)
                print(
                    f"[WARN 9b89e8a WalpurgisRandGen.generate] "
                    f"torch.distributed.broadcast failed: {e}. "
                    f"Falling back to rank-0 seed without broadcast.",
                    file=sys.stderr
                )
        else:
            _dbg_rand(
                "WalpurgisRandGen.generate",
                "world_size=1, skipping distributed broadcast"
            )

        # 9b89e8a: seed = seed.item() + rank
        # 每个 rank 加上自己的偏移 → 各 rank seed 唯一且不重叠
        final_seed = seed_tensor.item() + config.rank
        _dbg_rand(
            "WalpurgisRandGen.generate",
            f"final_seed={final_seed} = base_seed={seed_tensor.item()} "
            f"+ rank={config.rank}"
        )
        return final_seed


# ─── generate_seed_safe: 对应 9b89e8a generate_seed() 顶层函数 ───────────────
# Python (9b89e8a utils.py:17-27):
#   def generate_seed():
#       world_size = torch.distributed.get_world_size()
#       rank = torch.distributed.get_rank()
#       ...
#
# 改写: 加 non-distributed 降级 (Python 在单机环境 get_world_size() 会 raise)
# 降级路径: 直接 randint 不 broadcast, 与 rank=0 单机等价
_DEFAULT_RAND_GEN = WalpurgisRandGen()


def generate_seed_safe(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    device: str = "cuda",
) -> int:
    """
    对应 9b89e8a generate_seed(), 加 non-distributed 安全降级。

    Python (utils.py) 直接调用 torch.distributed.get_world_size/get_rank(),
    非 distributed 环境下会 raise RuntimeError。
    改写: 尝试从 torch.distributed 读取 rank/world_size,
    失败时降级为 rank=0 world_size=1 单机模式。

    断点调试: 打印 distributed 探测结果 + 最终 config

    Args:
        rank: 显式指定 rank (None → 从 torch.distributed 读取)
        world_size: 显式指定 world_size (None → 从 torch.distributed 读取)
        device: 生成种子用的 CUDA device (9b89e8a 固定 "cuda")

    Returns:
        int: 该 rank 的唯一随机种子
    """
    _dbg_rand(
        "generate_seed_safe",
        f"rank={rank} world_size={world_size} device={device!r}"
    )

    # 从 torch.distributed 读取 rank/world_size (与 Python 上游一致)
    if rank is None or world_size is None:
        try:
            import torch
            _ws = torch.distributed.get_world_size()
            _rk = torch.distributed.get_rank()
            rank = _rk if rank is None else rank
            world_size = _ws if world_size is None else world_size
            _dbg_rand(
                "generate_seed_safe",
                f"torch.distributed: rank={rank} world_size={world_size}"
            )
        except Exception as e:
            # 降级: 单机非 distributed 环境
            rank = rank if rank is not None else 0
            world_size = world_size if world_size is not None else 1
            _dbg_rand(
                "generate_seed_safe",
                f"torch.distributed not available ({e}). "
                f"Falling back to rank={rank} world_size={world_size}"
            )

    config = RandomStateConfig(
        rank=rank,
        world_size=world_size,
        device=device,
    )

    seed = _DEFAULT_RAND_GEN.generate(config)

    _dbg_rand(
        "generate_seed_safe",
        f"→ seed={seed} for rank={rank}"
    )
    return seed


# ─── SamplerRandomState: 对应 node_loader/link_loader 的 random_state= 注入点 ──
# Python (9b89e8a):
#   node_loader.py:  sample_from_nodes(input_data, random_state=generate_seed())
#   link_loader.py:  sample_from_edges(input_data, ..., random_state=generate_seed())
#
# 改写: 封装 (loader_type, seed, rank, world_size) 四元组,
#       替代 Python 的裸 int 直接传入 — 便于 audit/replay/调试
class SamplerRandomState:
    """
    对应 9b89e8a node_loader/link_loader 中 random_state=generate_seed() 的值。

    Python: random_state 是一个裸 int, 直接传给 sample_from_nodes/edges。
    改写: 封装为对象, 携带 (loader_type, seed, rank, world_size),
          .value 属性返回裸 int (与 Python API 兼容)。

    改写比 Python 更结构化:
      - loader_type 记录是 NODE 还是 LINK 路径 (Python 无此区分)
      - rank/world_size 随 seed 一起记录, 便于事后 replay
      - dump() 方法一键打印全部信息
    """

    def __init__(
        self,
        loader_type: LoaderType,
        rank: int = 0,
        world_size: int = 1,
        device: str = "cuda",
        _seed: Optional[int] = None,  # 测试注入用; None → 调用 generate_seed_safe
    ):
        self.loader_type = loader_type
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # 9b89e8a: random_state=generate_seed()
        # 改写: 调用 generate_seed_safe 替代裸 generate_seed (加降级)
        if _seed is not None:
            self._seed = _seed
            _dbg_rand(
                "SamplerRandomState.__init__",
                f"injected _seed={_seed} (test mode)"
            )
        else:
            self._seed = generate_seed_safe(
                rank=rank,
                world_size=world_size,
                device=device,
            )

        _dbg_rand(
            "SamplerRandomState.__init__",
            f"loader_type={loader_type.value} rank={rank} "
            f"world_size={world_size} seed={self._seed}"
        )

    @property
    def value(self) -> int:
        """
        返回裸 int seed (与 Python random_state=generate_seed() 的返回类型兼容)。
        用法: sampler.sample_from_nodes(input_data, random_state=state.value)
        """
        return self._seed

    def dump(self) -> None:
        """断点调试: 打印完整 SamplerRandomState 状态"""
        print(
            f"[DEBUG 9b89e8a SamplerRandomState.dump] "
            f"loader_type={self.loader_type.value} "
            f"rank={self.rank} world_size={self.world_size} "
            f"device={self.device!r} seed={self._seed}",
            file=sys.stderr
        )


# ─── Convenience builders: 对应 node_loader / link_loader 调用点 ─────────────
def make_node_loader_random_state(
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
) -> SamplerRandomState:
    """
    对应 9b89e8a node_loader.py:
        self.__node_sampler.sample_from_nodes(
            input_data, random_state=generate_seed()
        )

    Usage:
        state = make_node_loader_random_state(rank=rank, world_size=ws)
        self.__node_sampler.sample_from_nodes(input_data, random_state=state.value)

    断点调试: WALPURGIS_DEBUG=1 时打印 seed 生成全链路
    """
    _dbg_rand(
        "make_node_loader_random_state",
        f"rank={rank} world_size={world_size} device={device!r}"
    )
    state = SamplerRandomState(
        loader_type=LoaderType.NODE,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    if _DBG:
        state.dump()
    return state


def make_link_loader_random_state(
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
) -> SamplerRandomState:
    """
    对应 9b89e8a link_loader.py:
        self.__link_sampler.sample_from_edges(
            input_data,
            neg_sampling=self.__neg_sampling,
            random_state=generate_seed(),
        )

    Usage:
        state = make_link_loader_random_state(rank=rank, world_size=ws)
        self.__link_sampler.sample_from_edges(
            input_data,
            neg_sampling=self.__neg_sampling,
            random_state=state.value,
        )

    断点调试: WALPURGIS_DEBUG=1 时打印 seed 生成全链路
    """
    _dbg_rand(
        "make_link_loader_random_state",
        f"rank={rank} world_size={world_size} device={device!r}"
    )
    state = SamplerRandomState(
        loader_type=LoaderType.LINK,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    if _DBG:
        state.dump()
    return state
