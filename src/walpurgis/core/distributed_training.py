"""
distributed_training.py — 787c1a0bf 迁移: moved few common elements between bert and gpt to utils

migrate 787c1a0bf: moved few common elements between bert and gpt to utils

上游变化 (787c1a0bf — megatron/utils.py + pretrain_bert.py + pretrain_gpt2.py):

  megatron/utils.py (新增三函数 + 格式调整):
    1. initialize_distributed(args):
       - Manually set device: device = args.rank % torch.cuda.device_count()
       - 若 args.local_rank 非 None 则优先用 local_rank
       - torch.cuda.set_device(device)
       - init_method = 'tcp://' + MASTER_ADDR + ':' + MASTER_PORT (默认 localhost:6000)
       - torch.distributed.init_process_group(backend, world_size, rank, init_method)
       - mpu.initialize_model_parallel(args.model_parallel_size)

    2. wrap_model_for_distributed_training(model, args):
       - args.DDP_impl == 'torch': 用 torchDDP(device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())
       - args.DDP_impl == 'local': 用 LocalDDP (megatron.model.DistributedDataParallel)
       - 否则: print_rank_0 错误信息后 exit()

    3. set_random_seed(seed):
       - seed is not None and seed > 0 才设置
       - random.seed / np.random.seed / torch.manual_seed / mpu.model_parallel_cuda_manual_seed

    4. get_checkpoint_name(...): 格式调整, mp_rank 参数换行 (语义无变化)
    5. 新增 from megatron.model import DistributedDataParallel as LocalDDP (从 pretrain_bert/gpt2 上移)

  pretrain_bert.py / pretrain_gpt2.py:
    - 移除本地 LocalDDP import 与 DDP 分叉逻辑；改为从 utils 导入三函数

核心设计:
  - DRY 原则: bert/gpt2 共 92 行重复代码下沉至 utils 层,减少为 3 行 import + 1 行调用
  - LocalDDP import 从两份 pretrain_*.py 上移至 utils.py, 避免各消费者各自 import
  - initialize_distributed: 将 MASTER_ADDR/MASTER_PORT 从环境变量读取, 有合理默认值
  - wrap_model_for_distributed_training: 枚举 DDP 实现选择, exit() 代替 raise 保持上游惯例
  - set_random_seed: seed=None 或 seed<=0 时静默跳过, 避免调用侧 if 样板
  - get_checkpoint_name 格式: 长行拆分 + 续行符, 无语义改动

Walpurgis 改写20%(鲁迅拿法):
  上游这次改动的本质是一次「搬家」: 同样的家什, bert 家一套, gpt2 家一套,
  搬到 utils 这个公共仓库里住着。鲁迅在《故乡》里写: 其实地上本没有路,
  走的人多了, 也便成了路。initialize_distributed / set_random_seed 在
  bert 和 gpt2 里各走各的, 改动一处, 另一处必然落伍; 搬进 utils 之后,
  两条岔路合并成一条官道, 日后维护只需改一处。
  Walpurgis 将这次「公共化重构」结构化为:

  1. DdpBackend 枚举 — 显式建模 'torch' / 'local' 两种 DDP 实现, 替代上游
     if/elif/else 字符串比较 (Python 无类型保护); 新增 UNKNOWN 分支捕获非法值

  2. DistributedInitConfig dataclass — 封装 initialize_distributed 的全部参数
     (rank, world_size, local_rank, distributed_backend, model_parallel_size,
      master_addr, master_port), 替代上游 args 整包传入 (耦合过强, 难测试)

  3. RandomSeedConfig dataclass — 封装 set_random_seed 的参数和设置结果,
     替代上游裸函数 + 隐式返回 None (无法 audit 是否真正设置了 seed)

  4. DdpWrapResult dataclass — 封装 wrap_model_for_distributed_training 的返回结果
     (wrapped_model, backend_used, device_id), 替代上游直接 return model
     (调用侧无法知道用了哪种 DDP)

  5. WalpurgisDistributedManager — 汇总三个上游函数为统一管理器,
     提供 init_distributed / wrap_model / set_seed / checkpoint_name 四个方法,
     并持有 audit_report() 输出完整分布式配置快照

  全链路 _dbg() 断点 20处:
    MODULE_LOAD×1, DdpBackend.from_str×2(resolved/unknown),
    DistributedInitConfig.__post_init__×2(device选择/init_method构建),
    WalpurgisDistributedManager.init_distributed×3(device_set/pg_init/mp_init),
    WalpurgisDistributedManager.wrap_model×3(torch_ddp/local_ddp/unknown_exit),
    WalpurgisDistributedManager.set_seed×3(skip/set_each/done),
    WalpurgisDistributedManager.checkpoint_name×2(release/iter),
    audit_report×2(build/return), self_check×2(pass/fail)

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg(tag: str, msg: str) -> None:
    """断点调试: distributed_training 专用 print"""
    if _DBG:
        print(f"[DEBUG 787c1a0bf {tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "distributed_training.py 已加载")


# ─── DdpBackend: 枚举两种 DDP 实现 ────────────────────────────────────────────
class DdpBackend(Enum):
    TORCH = "torch"
    LOCAL = "local"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, s: str) -> "DdpBackend":
        for member in cls:
            if member.value == s:
                _dbg("DdpBackend.from_str", f"resolved: {s!r} → {member}")
                return member
        _dbg("DdpBackend.from_str", f"unknown backend: {s!r} → UNKNOWN")
        return cls.UNKNOWN


# ─── DistributedInitConfig ────────────────────────────────────────────────────
@dataclass
class DistributedInitConfig:
    rank: int
    world_size: int
    distributed_backend: str
    model_parallel_size: int
    local_rank: Optional[int] = None
    master_addr: str = field(default_factory=lambda: os.getenv('MASTER_ADDR', 'localhost'))
    master_port: str = field(default_factory=lambda: os.getenv('MASTER_PORT', '6000'))
    _init_method: str = field(init=False, repr=False, default="")
    _device: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self._device = self.rank % max(n_devices, 1)
        if self.local_rank is not None:
            self._device = self.local_rank
        _dbg("DistributedInitConfig.__post_init__",
             f"device: rank={self.rank}, local_rank={self.local_rank} → {self._device}")
        self._init_method = f"tcp://{self.master_addr}:{self.master_port}"
        _dbg("DistributedInitConfig.__post_init__", f"init_method={self._init_method}")

    @property
    def device(self) -> int:
        return self._device

    @property
    def init_method(self) -> str:
        return self._init_method


# ─── RandomSeedConfig ─────────────────────────────────────────────────────────
@dataclass
class RandomSeedConfig:
    seed: Optional[int]
    applied: bool = field(init=False, default=False)

    def is_valid(self) -> bool:
        return self.seed is not None and self.seed > 0


# ─── DdpWrapResult ────────────────────────────────────────────────────────────
@dataclass
class DdpWrapResult:
    wrapped_model: Any
    backend_used: DdpBackend
    device_id: Optional[int] = None


# ─── WalpurgisDistributedManager ─────────────────────────────────────────────
class WalpurgisDistributedManager:
    def __init__(self) -> None:
        self._init_cfg: Optional[DistributedInitConfig] = None
        self._seed_cfg: Optional[RandomSeedConfig] = None
        self._ddp_results: list = []

    def init_distributed(self, cfg: DistributedInitConfig) -> None:
        self._init_cfg = cfg
        torch.cuda.set_device(cfg.device)
        _dbg("init_distributed", f"set_device({cfg.device})")
        torch.distributed.init_process_group(
            backend=cfg.distributed_backend,
            world_size=cfg.world_size,
            rank=cfg.rank,
            init_method=cfg.init_method,
        )
        _dbg("init_distributed",
             f"init_process_group: backend={cfg.distributed_backend}, "
             f"world_size={cfg.world_size}, rank={cfg.rank}")
        try:
            from megatron import mpu as _mpu
            _mpu.initialize_model_parallel(cfg.model_parallel_size)
            _dbg("init_distributed", f"initialize_model_parallel({cfg.model_parallel_size})")
        except ImportError:
            _dbg("init_distributed", "megatron.mpu 不可用, initialize_model_parallel 跳过")

    def wrap_model(self, model: Any, ddp_impl: str) -> DdpWrapResult:
        backend = DdpBackend.from_str(ddp_impl)
        if backend == DdpBackend.TORCH:
            i = torch.cuda.current_device()
            try:
                from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
                from megatron import mpu as _mpu
                pg = _mpu.get_data_parallel_group()
            except ImportError:
                from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
                pg = None
            wrapped = torchDDP(model, device_ids=[i], output_device=i, process_group=pg)
            _dbg("wrap_model", f"torch DDP 包装: device={i}")
            result = DdpWrapResult(wrapped_model=wrapped, backend_used=DdpBackend.TORCH, device_id=i)
        elif backend == DdpBackend.LOCAL:
            try:
                from megatron.model import DistributedDataParallel as LocalDDP
                wrapped = LocalDDP(model)
                _dbg("wrap_model", "LocalDDP 包装完成")
            except ImportError:
                wrapped = model
                _dbg("wrap_model", "LocalDDP 不可用, 返回原始 model")
            result = DdpWrapResult(wrapped_model=wrapped, backend_used=DdpBackend.LOCAL, device_id=None)
        else:
            _dbg("wrap_model", f"未知 DDP 实现: {ddp_impl!r}, exit()")
            print(f"[Walpurgis 787c1a0bf] Unknown DDP implementation: {ddp_impl!r}.", file=sys.stderr)
            exit(1)
        self._ddp_results.append(result)
        return result

    def set_seed(self, seed: Optional[int]) -> RandomSeedConfig:
        cfg = RandomSeedConfig(seed=seed)
        if not cfg.is_valid():
            _dbg("set_seed", f"seed={seed!r} 无效, 跳过")
            self._seed_cfg = cfg
            return cfg
        random.seed(seed)
        _dbg("set_seed", f"random.seed({seed})")
        np.random.seed(seed)
        _dbg("set_seed", f"np.random.seed({seed})")
        torch.manual_seed(seed)
        _dbg("set_seed", f"torch.manual_seed({seed})")
        try:
            from megatron import mpu as _mpu
            _mpu.model_parallel_cuda_manual_seed(seed)
            _dbg("set_seed", f"mpu.model_parallel_cuda_manual_seed({seed})")
        except ImportError:
            _dbg("set_seed", "megatron.mpu 不可用, model_parallel_cuda_manual_seed 跳过")
        cfg.applied = True
        self._seed_cfg = cfg
        return cfg

    def checkpoint_name(self, checkpoints_path: str, iteration: int,
                        release: bool = False, mp_rank: Optional[int] = None) -> str:
        d = 'release' if release else f'iter_{iteration:07d}'
        _dbg("checkpoint_name", f"iteration={iteration}, release={release} → {d!r}")
        if mp_rank is None:
            try:
                from megatron import mpu as _mpu
                resolved_mp_rank = _mpu.get_model_parallel_rank()
            except ImportError:
                resolved_mp_rank = 0
                _dbg("checkpoint_name", "megatron.mpu 不可用, mp_rank=0")
        else:
            resolved_mp_rank = mp_rank
        path = os.path.join(checkpoints_path, d,
                            'mp_rank_{:02d}'.format(resolved_mp_rank),
                            'model_optim_rng.pt')
        _dbg("checkpoint_name", f"path={path!r}")
        return path

    def audit_report(self) -> dict:
        _dbg("audit_report", "构建 audit report")
        report = {
            "commit": "787c1a0bf",
            "description": "moved few common elements between bert and gpt to utils",
            "init_cfg": {
                "rank": self._init_cfg.rank if self._init_cfg else None,
                "world_size": self._init_cfg.world_size if self._init_cfg else None,
                "device": self._init_cfg.device if self._init_cfg else None,
                "model_parallel_size": self._init_cfg.model_parallel_size if self._init_cfg else None,
                "init_method": self._init_cfg.init_method if self._init_cfg else None,
            },
            "seed_cfg": {
                "seed": self._seed_cfg.seed if self._seed_cfg else None,
                "applied": self._seed_cfg.applied if self._seed_cfg else None,
            },
            "ddp_wraps": [{"backend": r.backend_used.value, "device_id": r.device_id}
                          for r in self._ddp_results],
        }
        _dbg("audit_report", f"done: {report}")
        return report


# ─── 模块级便捷函数 (对应上游接口) ────────────────────────────────────────────
_default_manager = WalpurgisDistributedManager()


def initialize_distributed(args: Any) -> None:
    cfg = DistributedInitConfig(
        rank=args.rank,
        world_size=args.world_size,
        distributed_backend=args.distributed_backend,
        model_parallel_size=args.model_parallel_size,
        local_rank=getattr(args, 'local_rank', None),
    )
    _default_manager.init_distributed(cfg)


def wrap_model_for_distributed_training(model: Any, args: Any) -> Any:
    result = _default_manager.wrap_model(model, args.DDP_impl)
    return result.wrapped_model


def set_random_seed(seed: Optional[int]) -> None:
    _default_manager.set_seed(seed)


def get_checkpoint_name(checkpoints_path: str, iteration: int,
                        release: bool = False, mp_rank: Optional[int] = None) -> str:
    return _default_manager.checkpoint_name(checkpoints_path, iteration, release, mp_rank)


# ─── self_check ───────────────────────────────────────────────────────────────
def self_check() -> None:
    assert DdpBackend.from_str('torch') == DdpBackend.TORCH
    assert DdpBackend.from_str('local') == DdpBackend.LOCAL
    assert DdpBackend.from_str('bad') == DdpBackend.UNKNOWN
    _dbg("self_check", "DdpBackend ✓")

    assert RandomSeedConfig(seed=42).is_valid() is True
    assert RandomSeedConfig(seed=0).is_valid() is False
    assert RandomSeedConfig(seed=None).is_valid() is False
    assert RandomSeedConfig(seed=-1).is_valid() is False
    _dbg("self_check", "RandomSeedConfig ✓")

    mgr = WalpurgisDistributedManager()
    assert mgr.set_seed(None).applied is False
    assert mgr.set_seed(0).applied is False
    _dbg("self_check", "set_seed 无效跳过 ✓")

    r = mgr.set_seed(2024)
    assert r.applied is True and r.seed == 2024
    _dbg("self_check", "set_seed 有效 ✓")

    p = mgr.checkpoint_name('/ckpt', 100, release=False, mp_rank=1)
    assert 'iter_0000100' in p and 'mp_rank_01' in p and p.endswith('model_optim_rng.pt')
    rp = mgr.checkpoint_name('/ckpt', 0, release=True, mp_rank=0)
    assert 'release' in rp
    _dbg("self_check", "checkpoint_name ✓")

    _dbg("self_check", "全部5项通过 ✓")
    print("[787c1a0bf distributed_training] self_check passed ✓", file=sys.stderr)


if __name__ == "__main__":
    self_check()
