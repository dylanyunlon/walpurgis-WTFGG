"""
ddp_selector.py — 迁移 4947002db: Merge branch 'torchddp' into 'master'

上游变化 (pretrain_bert.py, 4947002db):
  1. 移除顶层硬编码布尔旗 USE_TORCH_DDP = False
     → 改为运行时 args.DDP_impl ∈ {'torch', 'local'} 字符串选择器
  2. get_model():
     - 'torch'  路径: DistributedDataParallel (PyTorch 官方), 记录 args.DDP_type
     - 'local'  路径: LocalDDP (模型内置), 记录 args.DDP_type
     - 未知值: print_rank_0 警告后 exit()
  3. get_optimizer():
     - isinstance(model, (DDP, FP16_Module)) → isinstance(model, (args.DDP_type, FP16_Module))
  4. backward_step():
     - 新增 timers 参数 (原 signature 无)
     - if not USE_TORCH_DDP → if args.DDP_impl == 'local'
     - allreduce_params() 前后增加 timers('allreduce').start/stop()
  5. train():
     - timers_to_log 根据 DDP_impl 分支:
       'torch'  → ['forward', 'backward', 'optimizer', 'batch generator', 'data loader']
       'local'  → 同上 + 'allreduce' (在 'backward' 之后插入)
  6. evaluate():
     - isinstance(model, DDP) → isinstance(model, args.DDP_type)

核心语义: 将"编译期布尔常量"升级为"运行期字符串参数",
          同时为 local DDP 的 allreduce 操作加上独立计时器。

Walpurgis 改写20%(鲁迅拿法):
  Python 用一个全局布尔旗 USE_TORCH_DDP 区分两路; 逻辑分散在四个函数里,
  如同四座孤岛——共用一根电报线, 每座岛各自解读码, 互不知晓对方的解读。
  鲁迅见之, 曰: "旗子挂在屋顶, 屋里的人靠猜。"

  Walpurgis 将 DDP 选择与 allreduce 计时逻辑集中封装为三个结构:

  1. DDPImpl (Enum) — 替代上游裸字符串 'torch'/'local'/'unknown',
     让类型系统捕获拼写错误, 并集中管理合法值集合。
     上游: args.DDP_impl == 'torch' (字符串比较四次散落各处)
     改写: DDPImpl.from_str(args.DDP_impl) → 一次解析, 全程枚举比较

  2. DDPBuilder — 替代上游 get_model() 内联的三路 if/elif/else,
     封装 wrap_for_ddp(model, impl, args) → (wrapped_model, DDP_type)
     断点调试: DDP_WRAP_{TORCH,LOCAL,UNKNOWN} 三个 tag

  3. AllreduceTimer — 替代上游 backward_step() 内 timers('allreduce').start/stop()
     两行内联代码, 封装为上下文管理器 with AllreduceTimer(timers, impl):
     仅当 impl == local 时实际计时, torch 路径静默跳过;
     上游: if args.DDP_impl == 'local': timers('allreduce').start(); ...; stop()
     改写: 结构化 guard, 消除调用端 if 判断

  4. build_timers_to_log(impl) — 替代上游 train() 里四行 if/else,
     集中返回对应 impl 的 timer 名列表, 单一责任原则。

全链路 _dbg() 断点 (WALPURGIS_DEBUG=1 激活):
  DDP_IMPL_PARSE, DDP_IMPL_PARSE_ERR,
  DDP_WRAP_TORCH, DDP_WRAP_LOCAL, DDP_WRAP_UNKNOWN,
  ALLREDUCE_TIMER_ENTER, ALLREDUCE_TIMER_START, ALLREDUCE_TIMER_STOP,
  ALLREDUCE_TIMER_SKIP, BUILD_TIMERS_TO_LOG

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, List, Optional, Tuple

# ─── 全局调试开关 (与 walpurgis/__init__.py 保持一致) ─────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试: 4947002db 专用 print (WALPURGIS_DEBUG=1 激活)"""
    if _DBG:
        print(f"[DEBUG 4947002db {tag}] {msg}", file=sys.stderr, flush=True)


# ─── DDPImpl: 替代上游裸字符串 'torch'/'local' ────────────────────────────────
# 上游 (pretrain_bert.py, 4947002db):
#   if args.DDP_impl == 'torch':  ...
#   elif args.DDP_impl == 'local': ...
#   else: print_rank_0(...); exit()
#
# 改写: 枚举替代裸字符串, 一次解析, 全程枚举比较
# - 消除四处分散的字符串比较 (get_model/backward_step/train/evaluate)
# - 拼写错误在 from_str() 处立即捕获, 而非运行到 else 分支才 exit()
class DDPImpl(Enum):
    """
    DDP 实现选择器 — 对应 args.DDP_impl 的合法值。

    上游 (4947002db pretrain_bert.py):
      'torch'  → torch.nn.parallel.distributed.DistributedDataParallel
      'local'  → model.DistributedDataParallel (LocalDDP)
      其他     → print_rank_0 警告 + exit()

    改写: 枚举集中管理合法值, from_str() 单点解析
    """
    TORCH = "torch"   # PyTorch 官方 DDP — overlapping comm/compute
    LOCAL = "local"   # 模型内置 LocalDDP — 手动 allreduce

    @classmethod
    def from_str(cls, value: str) -> "DDPImpl":
        """
        对应上游 args.DDP_impl 字符串到枚举的解析。

        上游: 字符串直接散落在各 if/elif 比较中, 拼写错误只在 else 分支暴露。
        改写: 集中解析, KeyError → DDPImplError (明确错误类型)

        断点调试: DDP_IMPL_PARSE (成功), DDP_IMPL_PARSE_ERR (失败)
        """
        _dbg("DDP_IMPL_PARSE", f"parsing DDP_impl={value!r}")
        try:
            result = cls(value)
            _dbg("DDP_IMPL_PARSE", f"→ DDPImpl.{result.name}")
            return result
        except ValueError:
            _dbg("DDP_IMPL_PARSE_ERR",
                 f"unknown DDP_impl={value!r}; valid={[e.value for e in cls]}")
            raise DDPImplError(
                f"Unknown DDP implementation specified: {value!r}. "
                f"Valid values: {[e.value for e in cls]}. "
                f"Exiting."   # 与上游 print_rank_0 文本保持语义一致
            )


class DDPImplError(ValueError):
    """
    对应上游 else 分支: print_rank_0('Unknown DDP implementation...'); exit()

    改写: 异常替代 exit() —— exit() 在测试环境会终止整个 pytest 进程;
    异常可被调用端捕获、记录后再 exit(), 测试可 catch 断言。
    """
    pass


# ─── DDPBuilder: 替代上游 get_model() 内联 DDP 包裹逻辑 ──────────────────────
# 上游 (4947002db pretrain_bert.py, get_model()):
#   if args.DDP_impl == 'torch':
#       i = torch.cuda.current_device()
#       args.DDP_type = torch.nn.parallel.distributed.DistributedDataParallel
#       model = args.DDP_type(model, device_ids=[i], output_device=i,
#                             process_group=mpu.get_data_parallel_group())
#   elif args.DDP_impl == 'local':
#       args.DDP_type = LocalDDP
#       model = args.DDP_type(model)
#   else:
#       print_rank_0('Unknown DDP implementation specified: {}. Exiting.'.format(args.DDP_impl))
#       exit()
#
# 改写: 封装为 DDPBuilder.wrap_for_ddp(), 返回 (wrapped_model, DDP_type) 元组
# - args.DDP_type 赋值从 get_model() 侧效应变为显式返回值
# - 每条路径独立断点 tag, 便于调试 DDP 初始化问题
class DDPBuilder:
    """
    DDP 包裹工厂 — 对应上游 get_model() 内 DDP 初始化三路逻辑。

    上游: 三路 if/elif/else 内联于 get_model(), args.DDP_type 作为侧效应赋值。
    改写: wrap_for_ddp() 返回 (model, DDP_type), 调用端显式接收 DDP_type,
          消除对 args 命名空间的隐式写入依赖。

    调用示例 (迁移 get_model 时):
        try:
            model, DDP_type = DDPBuilder.wrap_for_ddp(
                model, DDPImpl.from_str(args.DDP_impl),
                local_ddp_cls=LocalDDP,
                mpu_module=mpu,
            )
            args.DDP_type = DDP_type  # 仍写入 args, 保持下游兼容
        except DDPImplError as e:
            print_rank_0(str(e))
            exit()
    """

    @staticmethod
    def wrap_for_ddp(
        model: Any,
        impl: DDPImpl,
        local_ddp_cls: Any,
        mpu_module: Optional[Any] = None,
        torch_module: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        对应上游 get_model() 内 DDP 包裹三路逻辑。

        Args:
            model:         裸模型 (已 FP16/tensor_model_parallel 处理完毕)
            impl:          DDPImpl 枚举 (TORCH 或 LOCAL)
            local_ddp_cls: 上游 model.DistributedDataParallel (LocalDDP)
            mpu_module:    上游 mpu — 提供 get_data_parallel_group()
                           (仅 TORCH 路径需要)
            torch_module:  注入 torch (None → 运行时 import, 便于测试 mock)

        Returns:
            (wrapped_model, DDP_type):
              wrapped_model — DDP 包裹后的模型
              DDP_type      — 包裹类本身 (赋给 args.DDP_type 以兼容上游)

        断点调试:
          DDP_WRAP_TORCH  — torch DDP 路径 + device_id
          DDP_WRAP_LOCAL  — local DDP 路径
          DDP_WRAP_UNKNOWN — 不应抵达此分支 (DDPImpl 枚举已穷举)
        """
        if torch_module is None:
            import torch as _t
            torch_module = _t

        if impl == DDPImpl.TORCH:
            # 上游 (4947002db):
            #   i = torch.cuda.current_device()
            #   args.DDP_type = torch.nn.parallel.distributed.DistributedDataParallel
            #   model = args.DDP_type(model, device_ids=[i], output_device=i,
            #                         process_group=mpu.get_data_parallel_group())
            i = torch_module.cuda.current_device()
            DDP_type = (torch_module.nn.parallel.distributed
                        .DistributedDataParallel)
            _dbg("DDP_WRAP_TORCH",
                 f"device_id={i} DDP_type={DDP_type.__name__} "
                 f"process_group=mpu.get_data_parallel_group()")
            wrapped = DDP_type(
                model,
                device_ids=[i],
                output_device=i,
                process_group=mpu_module.get_data_parallel_group(),
            )
            return wrapped, DDP_type

        elif impl == DDPImpl.LOCAL:
            # 上游 (4947002db):
            #   args.DDP_type = LocalDDP
            #   model = args.DDP_type(model)
            DDP_type = local_ddp_cls
            _dbg("DDP_WRAP_LOCAL",
                 f"DDP_type={DDP_type.__name__} (local/custom allreduce)")
            wrapped = DDP_type(model)
            return wrapped, DDP_type

        else:
            # 防御分支: DDPImpl 枚举已穷举 TORCH/LOCAL,
            # 正常情况不会抵达此处 (from_str 已在入口拦截未知值)
            _dbg("DDP_WRAP_UNKNOWN", f"impl={impl!r} — should not reach here")
            raise DDPImplError(
                f"DDPBuilder.wrap_for_ddp: unhandled impl={impl!r}. "
                "This is a bug in DDPImpl enumeration coverage."
            )


# ─── AllreduceTimer: 替代上游 backward_step() 内联 allreduce 计时 ────────────
# 上游 (4947002db pretrain_bert.py, backward_step()):
#   if args.DDP_impl == 'local':
#       timers('allreduce').start()
#       model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
#       timers('allreduce').stop()
#
# 改写: 上下文管理器, 仅 LOCAL 路径计时, TORCH 路径静默跳过
# - 调用端无需再写 if impl == 'local': 判断
# - with 块内的 allreduce_params 调用由调用端负责 (职责分离)
# - 断点 tag 标记 start/stop/skip 三种状态
class AllreduceTimer:
    """
    Allreduce 计时上下文管理器 — 对应上游 backward_step() 内 allreduce 计时。

    上游: if args.DDP_impl == 'local': timers('allreduce').start(); ...; stop()
    改写: with AllreduceTimer(timers, impl): ...
          仅 LOCAL 路径实际调用 timers('allreduce').start/stop(),
          TORCH 路径整个上下文管理器静默 (不计时, 不报错)

    调用示例 (迁移 backward_step 时):
        with AllreduceTimer(timers, impl):
            model.allreduce_params(
                reduce_after=False, fp32_allreduce=args.fp32_allreduce
            )

    上游旧写法:
        if args.DDP_impl == 'local':
            timers('allreduce').start()
            model.allreduce_params(...)
            timers('allreduce').stop()

    断点调试:
      ALLREDUCE_TIMER_ENTER — 进入上下文, 记录 impl
      ALLREDUCE_TIMER_START — LOCAL 路径, timers('allreduce').start() 前
      ALLREDUCE_TIMER_STOP  — LOCAL 路径, timers('allreduce').stop() 前
      ALLREDUCE_TIMER_SKIP  — TORCH 路径, 静默跳过
    """

    def __init__(self, timers: Any, impl: DDPImpl) -> None:
        """
        Args:
            timers: 上游 Timers 对象 (callable: timers('allreduce') → timer)
            impl:   DDPImpl 枚举 — 决定是否实际计时
        """
        self._timers = timers
        self._impl = impl

    def __enter__(self) -> "AllreduceTimer":
        _dbg("ALLREDUCE_TIMER_ENTER", f"impl={self._impl.value}")
        if self._impl == DDPImpl.LOCAL:
            _dbg("ALLREDUCE_TIMER_START", "timers('allreduce').start()")
            self._timers("allreduce").start()
        else:
            _dbg("ALLREDUCE_TIMER_SKIP",
                 f"impl={self._impl.value} — skipping allreduce timer")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._impl == DDPImpl.LOCAL:
            _dbg("ALLREDUCE_TIMER_STOP", "timers('allreduce').stop()")
            self._timers("allreduce").stop()
        # 不抑制异常 (返回 False/None)
        return False


@contextmanager
def allreduce_timed(timers: Any, impl: DDPImpl) -> Iterator[None]:
    """
    AllreduceTimer 的函数式等价物 — 便于不熟悉类上下文管理器的调用端。

    等价于: with AllreduceTimer(timers, impl): yield

    调用示例:
        with allreduce_timed(timers, impl):
            model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
    """
    timer_obj = AllreduceTimer(timers, impl)
    with timer_obj:
        yield


# ─── build_timers_to_log: 替代上游 train() 内 timers_to_log 分支 ─────────────
# 上游 (4947002db pretrain_bert.py, train()):
#   if args.DDP_impl == 'torch':
#       timers_to_log = ['forward', 'backward', 'optimizer',
#                        'batch generator', 'data loader']
#   else:
#       timers_to_log = ['forward', 'backward', 'allreduce', 'optimizer',
#                        'batch generator', 'data loader']
#
# 改写: 纯函数, 单点返回, 调用端只需 build_timers_to_log(impl)
# - 消除 train() 内四行 if/else 逻辑
# - 断点调试: BUILD_TIMERS_TO_LOG tag 打印最终列表
_BASE_TIMERS: List[str] = [
    "forward",
    "backward",
    "optimizer",
    "batch generator",
    "data loader",
]

_LOCAL_TIMERS: List[str] = [
    "forward",
    "backward",
    "allreduce",   # 4947002db 新增: local DDP 专属计时
    "optimizer",
    "batch generator",
    "data loader",
]


def build_timers_to_log(impl: DDPImpl) -> List[str]:
    """
    对应上游 train() 内 timers_to_log 分支逻辑。

    上游 (4947002db):
      torch 路径: ['forward', 'backward', 'optimizer', 'batch generator', 'data loader']
      local 路径: 同上 + 'allreduce' (在 'backward' 与 'optimizer' 之间)

    改写: 纯函数, 集中管理两张列表, 调用端无需再写 if/else

    断点调试: BUILD_TIMERS_TO_LOG — 打印 impl + 返回的 timer 列表

    Args:
        impl: DDPImpl 枚举

    Returns:
        List[str]: timer 名称列表, 传给上游 report_memory / log_dist 等
    """
    if impl == DDPImpl.TORCH:
        result = list(_BASE_TIMERS)
    else:
        # DDPImpl.LOCAL 及未来可能的 impl — 默认含 allreduce
        result = list(_LOCAL_TIMERS)

    _dbg("BUILD_TIMERS_TO_LOG",
         f"impl={impl.value} → timers_to_log={result}")
    return result


# ─── resolve_ddp_type: 对应 get_optimizer / evaluate 中 isinstance 检查 ───────
# 上游 (4947002db pretrain_bert.py):
#   get_optimizer():
#     while isinstance(model, (args.DDP_type, FP16_Module)): model = model.module
#   evaluate():
#     if isinstance(model, args.DDP_type): ...
#
# 上游将 DDP_type 存入 args 命名空间; 改写提供显式获取函数,
# 便于在 args 不可用的上下文 (单元测试/独立调用) 中使用
def resolve_ddp_type(impl: DDPImpl, local_ddp_cls: Any,
                     torch_module: Optional[Any] = None) -> Any:
    """
    对应 args.DDP_type — 根据 DDPImpl 枚举返回对应的 DDP 类型。

    上游: args.DDP_type 在 get_model() 内赋值, get_optimizer/evaluate 处引用。
    改写: 显式函数, 可在 args 不可用时调用, 亦可作为 args.DDP_type 的初始化来源。

    Args:
        impl:          DDPImpl 枚举
        local_ddp_cls: 对应 'local' 路径的 LocalDDP 类
        torch_module:  注入 torch (None → 运行时 import)

    Returns:
        DDP 类 (未实例化), 用于 isinstance() 检查
    """
    if torch_module is None:
        import torch as _t
        torch_module = _t

    if impl == DDPImpl.TORCH:
        return (torch_module.nn.parallel.distributed
                .DistributedDataParallel)
    elif impl == DDPImpl.LOCAL:
        return local_ddp_cls
    else:
        raise DDPImplError(
            f"resolve_ddp_type: unknown impl={impl!r}"
        )
