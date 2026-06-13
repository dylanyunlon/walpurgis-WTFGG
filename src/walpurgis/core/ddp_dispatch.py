"""
ddp_dispatch.py
迁移自 Megatron-LM commit 72c5f666b
原 commit: "Use DDP command line argument instead of source flag in pretrain_bert.py"
上游 diff 摘要 (1 file changed, 23 insertions(+), 17 deletions(-)):
  - 移除顶层 USE_TORCH_DDP = False 硬编码标志
  - 移除条件 import (if USE_TORCH_DDP: from torch.nn.parallel...)，改为无条件
    from model import DistributedDataParallel as LocalDDP
  - get_model(): 分支从 USE_TORCH_DDP → args.DDP_impl == 'torch' / 'local' / else-exit
  - get_optimizer(): while isinstance(model, (DDP, FP16_Module)) → (args.DDP_type, FP16_Module)
  - backward_step(): 新增 timers 参数; allreduce 块加 timers('allreduce').start/stop
  - train(): timers_to_log 按 DDP_impl 分叉
  - evaluate(): isinstance(model, DDP) → isinstance(model, args.DDP_type)

鲁迅拿法改写 (~20%):
  上游的 USE_TORCH_DDP = False 是一道锁死的铁门——改它要动源码，
  要重新打包，要绕道而行，如同《狂人日记》里的礼教：
  名为定制，实为囚笼。一个布尔值写死在文件顶端，
  像族规刻在祠堂牌匾，不得质疑，不得修改，只能服从。
  args.DDP_impl 把这道铁门换成了旋转门：'torch'/'local'/错误退出，
  三条路，各有来历，命令行一字即可切换，不必触碰源码。
  Walpurgis 将此「策略化」结构化为三个可程序化组件：
  1. DDPImplPolicy 枚举 — 强类型化 'torch'/'local' 两种实现，
     消除散落字符串比较，使分支意图在类型层面可见
  2. DDPDispatcher — 封装 get_model 阶段的 DDP 包裹逻辑，
     将 args.DDP_impl/args.DDP_type 赋值与模型包裹分离，
     与 args 解耦，可独立测试
  3. AllreduceGuard — 封装 backward_step 中的 allreduce 调用，
     带 timers 起止，断点可观测，与 backward_step 主体分离

全链路 WALPURGIS_DEBUG=1 断点 print 覆盖:
  MODULE_LOAD, ENUM_INIT×2, DISPATCHER_INIT, WRAP_TORCH, WRAP_LOCAL,
  WRAP_UNKNOWN, ALLREDUCE_START, ALLREDUCE_STOP, TIMERS_LOG_BRANCH,
  ISINSTANCE_GUARD, SELF_CHECK×5
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, TYPE_CHECKING

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"

def _dbg(tag: str, msg: str) -> None:
    """断点调试: WALPURGIS_DEBUG=1 时输出到 stderr"""
    if _DEBUG:
        print(f"[DDP-DBG:{tag}] {msg}", file=sys.stderr, flush=True)

_dbg("MODULE_LOAD", f"ddp_dispatch.py 加载, WALPURGIS_DEBUG={_DEBUG}")


# ── 1. DDPImplPolicy 枚举: 强类型化 --DDP-impl 参数 ──────────────────────────
class DDPImplPolicy(Enum):
    """
    上游 args.DDP_impl 是裸字符串，散落在 get_model/backward_step/train/evaluate
    四处做 == 'torch' / == 'local' 比较。字符串比较无类型检查，typo 静默失败。
    DDPImplPolicy 将其强类型化，使非法值在构造时即刻报错而非运行时悄悄走 else-exit。

    上游设计意图:
      'torch' → torch.nn.parallel.distributed.DistributedDataParallel
                带 process_group=mpu.get_data_parallel_group()，overlapping comm
      'local' → model.DistributedDataParallel (LocalDDP)
                手动 allreduce_params(), reduce_after=False, fp32_allreduce 可选
    """
    TORCH = "torch"   # 上游 torch DDP, 与 process_group 绑定
    LOCAL = "local"   # 上游 LocalDDP, 手动 allreduce 控制

    @classmethod
    def from_str(cls, value: str) -> "DDPImplPolicy":
        """从命令行参数字符串构造; 非法值抛 ValueError 而非静默走 exit()"""
        _dbg("ENUM_INIT", f"DDPImplPolicy.from_str({value!r})")
        mapping = {e.value: e for e in cls}
        if value not in mapping:
            known = list(mapping.keys())
            raise ValueError(
                f"未知 DDP 实现: {value!r}。已知: {known}。"
                f"请通过 --DDP-impl 传入 'torch' 或 'local'。"
            )
        result = mapping[value]
        _dbg("ENUM_INIT", f"解析结果: {result}")
        return result

    @property
    def needs_manual_allreduce(self) -> bool:
        """Local DDP 需要手动调用 allreduce_params; torch DDP 自动 overlap"""
        return self == DDPImplPolicy.LOCAL

    @property
    def needs_process_group(self) -> bool:
        """torch DDP 需要 process_group=mpu.get_data_parallel_group()"""
        return self == DDPImplPolicy.TORCH


# ── 2. DDPDispatcher: 封装 get_model 阶段的包裹逻辑 ─────────────────────────
@dataclass
class DDPDispatcher:
    """
    上游 get_model() 在函数内直接做 args.DDP_impl == 'torch' 分支,
    并将 args.DDP_type 作为副作用写入 args 对象。
    这种「把类型信息混入 args namespace」的模式难以测试: 需要 mock 整个 args。

    DDPDispatcher 将包裹逻辑与 args 副作用分离:
      - wrap() 返回 (wrapped_model, ddp_type_class)
      - 调用方自行决定是否写入 args.DDP_type
      - local_rank/process_group 显式传入, 而非从 torch.cuda 隐式读取

    上游对应代码段 (get_model, 72c5f666b):
      if args.DDP_impl == 'torch':
          i = torch.cuda.current_device()
          args.DDP_type = torch.nn.parallel.distributed.DistributedDataParallel
          model = args.DDP_type(model, device_ids=[i], output_device=i,
                                process_group=mpu.get_data_parallel_group())
      elif args.DDP_impl == 'local':
          args.DDP_type = LocalDDP
          model = args.DDP_type(model)
      else:
          print_rank_0('Unknown DDP implementation specified: {}. Exiting.'.format(args.DDP_impl))
          exit()
    """
    policy: DDPImplPolicy

    def __post_init__(self) -> None:
        _dbg("DISPATCHER_INIT", f"DDPDispatcher 初始化, policy={self.policy}")

    def wrap(
        self,
        model: Any,
        local_rank: Optional[int] = None,
        process_group: Optional[Any] = None,
        local_ddp_cls: Optional[Any] = None,
    ):
        """
        包裹 model 并返回 (wrapped_model, ddp_type_class)。

        Args:
            model: 原始模型 (已 fp16/fp32 处理)
            local_rank: GPU device id (仅 TORCH 路径使用)
            process_group: mpu.get_data_parallel_group() (仅 TORCH 路径使用)
            local_ddp_cls: LocalDDP 类 (仅 LOCAL 路径使用,
                           避免在 policy 模块中 import model 模块产生循环)

        Returns:
            (wrapped_model, ddp_type_class)

        Raises:
            ValueError: policy 既非 TORCH 也非 LOCAL (正常不触达, from_str 已拦截)
        """
        if self.policy == DDPImplPolicy.TORCH:
            _dbg("WRAP_TORCH",
                 f"torch DDP 包裹: local_rank={local_rank} "
                 f"process_group={process_group}")
            import torch
            TorchDDP = torch.nn.parallel.distributed.DistributedDataParallel
            device_id = (local_rank if local_rank is not None
                         else torch.cuda.current_device())
            wrapped = TorchDDP(
                model,
                device_ids=[device_id],
                output_device=device_id,
                process_group=process_group,
            )
            _dbg("WRAP_TORCH",
                 f"包裹完成: device_id={device_id} type={type(wrapped).__name__}")
            return wrapped, TorchDDP

        elif self.policy == DDPImplPolicy.LOCAL:
            _dbg("WRAP_LOCAL",
                 f"LocalDDP 包裹: local_ddp_cls={local_ddp_cls}")
            if local_ddp_cls is None:
                # 调用方未传入 LocalDDP 类时的保护——不做任何 model 子包 import
                raise ValueError(
                    "DDPImplPolicy.LOCAL 路径需要传入 local_ddp_cls 参数。"
                    "请从 model 模块导入 DistributedDataParallel 后传入。"
                )
            wrapped = local_ddp_cls(model)
            _dbg("WRAP_LOCAL",
                 f"包裹完成: type={type(wrapped).__name__}")
            return wrapped, local_ddp_cls

        else:
            # 正常路径不触达: DDPImplPolicy.from_str 已过滤非法值
            # 保留此分支以便直接构造 DDPDispatcher 时的防御
            _dbg("WRAP_UNKNOWN",
                 f"未知 policy: {self.policy}, 视同 'local' 降级")
            raise ValueError(
                f"DDPDispatcher.wrap: 未知 policy {self.policy}。"
                f"上游行为: print_rank_0(...); exit()。"
                f"Walpurgis 路径: 显式 ValueError, 由调用方决定是否 sys.exit()。"
            )


# ── 3. AllreduceGuard: 封装 backward_step 中的 allreduce + timers ────────────
@dataclass
class AllreduceGuard:
    """
    上游 backward_step (72c5f666b) 在 LOCAL DDP 路径下新增了 timers:
      if args.DDP_impl == 'local':
          timers('allreduce').start()
          model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
          timers('allreduce').stop()

    AllreduceGuard 将 timers 的 start/stop 与 allreduce 调用封装在一处,
    使断点完整可观测, 并将 fp32_allreduce 与 reduce_after 显式化
    (上游直接 args.fp32_allreduce 散落在 backward_step 函数体中)。

    鲁迅笔记: timers('allreduce').start/stop 这对括号,
    像极了《孔乙己》里的账本记录——一笔进, 一笔出,
    中间的 allreduce 才是货真价实的通信开销。
    没有 timers 之前, allreduce 是无名之辈, 无处追责;
    有了 timers, 才知道瓶颈在哪里, 才能对症下药。
    """
    policy: DDPImplPolicy
    reduce_after: bool = False       # 上游固定 False
    fp32_allreduce: bool = False     # 来自 args.fp32_allreduce

    def __post_init__(self) -> None:
        _dbg("ALLREDUCE_GUARD_INIT",
             f"policy={self.policy} reduce_after={self.reduce_after} "
             f"fp32_allreduce={self.fp32_allreduce}")

    def run(self, model: Any, timers: Any) -> None:
        """
        执行 allreduce (仅 LOCAL 路径有效)。

        Args:
            model: DDP 包裹后的模型 (需有 allreduce_params 方法)
            timers: Megatron Timers 对象 (支持 timers('name').start/stop)
        """
        if self.policy != DDPImplPolicy.LOCAL:
            # torch DDP 自动 overlap allreduce, 无需手动调用
            _dbg("ALLREDUCE_START",
                 "torch DDP 路径跳过手动 allreduce (自动 overlap)")
            return

        _dbg("ALLREDUCE_START",
             f"LOCAL DDP allreduce 开始: "
             f"reduce_after={self.reduce_after} "
             f"fp32_allreduce={self.fp32_allreduce}")
        timers("allreduce").start()
        model.allreduce_params(
            reduce_after=self.reduce_after,
            fp32_allreduce=self.fp32_allreduce,
        )
        timers("allreduce").stop()
        _dbg("ALLREDUCE_STOP", "LOCAL DDP allreduce 完成")


# ── 4. 辅助: timers_to_log 分叉逻辑 ─────────────────────────────────────────
def get_timers_to_log(policy: DDPImplPolicy) -> List[str]:
    """
    上游 train() 在 72c5f666b 中将 timers_to_log 按 DDP_impl 分叉:
      torch 路径: ['forward', 'backward', 'optimizer', 'batch generator', 'data loader']
      local 路径: ['forward', 'backward', 'allreduce', 'optimizer',
                   'batch generator', 'data loader']
    local 路径多了 'allreduce', 对应 AllreduceGuard.run() 中的 timers('allreduce') 计时。

    将此逻辑提取为纯函数, 使 train() 主体无需内联条件判断。
    """
    _dbg("TIMERS_LOG_BRANCH", f"policy={policy}")
    if policy == DDPImplPolicy.TORCH:
        result = ["forward", "backward", "optimizer",
                  "batch generator", "data loader"]
    else:
        # LOCAL 路径: allreduce 独立计时
        result = ["forward", "backward", "allreduce", "optimizer",
                  "batch generator", "data loader"]
    _dbg("TIMERS_LOG_BRANCH", f"timers_to_log={result}")
    return result


# ── 5. 辅助: isinstance 守卫 (evaluate 阶段) ─────────────────────────────────
def is_ddp_wrapped(model: Any, ddp_type: Any) -> bool:
    """
    上游 evaluate() 中:
      if isinstance(model, DDP):  →  if isinstance(model, args.DDP_type):
    args.DDP_type 是在 get_model 阶段写入的 (TorchDDP 或 LocalDDP 类本身)。

    Walpurgis: 接受显式 ddp_type 参数, 避免从 args namespace 隐式取值,
    同时保留断点可观测性。
    """
    result = isinstance(model, ddp_type)
    _dbg("ISINSTANCE_GUARD",
         f"isinstance({type(model).__name__}, {ddp_type.__name__ if hasattr(ddp_type, '__name__') else ddp_type}) = {result}")
    return result


# ── 6. self_check ─────────────────────────────────────────────────────────────
def _self_check() -> None:
    """模块加载时自检; WALPURGIS_DEBUG=1 时输出结果"""

    # 断言 1: DDPImplPolicy 枚举成员
    assert DDPImplPolicy.TORCH.value == "torch", "TORCH value 错误"
    assert DDPImplPolicy.LOCAL.value == "local", "LOCAL value 错误"
    _dbg("SELF_CHECK", "断言 1: DDPImplPolicy 枚举值 OK")

    # 断言 2: from_str 正常路径
    p = DDPImplPolicy.from_str("torch")
    assert p == DDPImplPolicy.TORCH, "from_str('torch') 解析失败"
    p = DDPImplPolicy.from_str("local")
    assert p == DDPImplPolicy.LOCAL, "from_str('local') 解析失败"
    _dbg("SELF_CHECK", "断言 2: from_str 正常路径 OK")

    # 断言 3: from_str 非法值
    try:
        DDPImplPolicy.from_str("unknown_impl")
        assert False, "应抛 ValueError"
    except ValueError:
        pass
    _dbg("SELF_CHECK", "断言 3: from_str 非法值 ValueError OK")

    # 断言 4: needs_manual_allreduce / needs_process_group
    assert DDPImplPolicy.LOCAL.needs_manual_allreduce is True
    assert DDPImplPolicy.TORCH.needs_manual_allreduce is False
    assert DDPImplPolicy.TORCH.needs_process_group is True
    assert DDPImplPolicy.LOCAL.needs_process_group is False
    _dbg("SELF_CHECK", "断言 4: policy 属性 OK")

    # 断言 5: get_timers_to_log 分叉
    torch_timers = get_timers_to_log(DDPImplPolicy.TORCH)
    local_timers = get_timers_to_log(DDPImplPolicy.LOCAL)
    assert "allreduce" not in torch_timers, "torch 路径不应含 allreduce"
    assert "allreduce" in local_timers, "local 路径必须含 allreduce"
    _dbg("SELF_CHECK", "断言 5: get_timers_to_log 分叉 OK")

    _dbg("SELF_CHECK", "全部 5 项断言通过 ✓")


_self_check()
