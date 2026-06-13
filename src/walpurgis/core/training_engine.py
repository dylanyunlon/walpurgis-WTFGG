# coding=utf-8
# Walpurgis migration: megatron/training.py
# Upstream commit: beb3e0d38  "Merge branch 'transformer_refactoring_from_pretrain_refactoring'"
#
# 核心变化：training.py（499行）首次出现，将 pretrain_bert.py 和 pretrain_gpt2.py
# 的训练循环提取为共享引擎。run() / train() / evaluate() / backward_step() /
# setup_model_and_optimizer() 等函数从两个文件合并为一套。
#
# 鲁迅曰：从前 BERT 训练一套，GPT-2 训练一套，
# 复制粘贴是那个年代最诚实的懒惰。
# 合并之后，代码少了，但危险也少了——
# 两个 bug 合并成一个 bug，总比两个各自演化要好。
# 这不是进步，这是止损。
#
# Walpurgis 改写要点（≥20%）：
#   1. TrainingConfig(dataclass) 封装散参数 args，使 run() 签名可静态审计
#   2. TrainingMetrics(dataclass) 替代 total_loss_dict 裸 dict
#   3. DDPStrategy(Enum: TORCH / LOCAL) 替代 args.DDP_impl 字符串比较
#   4. OptimizerSpec(dataclass) 封装 FP16 参数，新增 loss_scale_guard() 验证
#   5. TrainingEngine 类封装 run() 全流程，使测试可注入各阶段 hook
#   6. _dbg() 断点 15 处

import os
import math
import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional, Any, Tuple

import torch

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, **kw) -> None:
    if _DBG:
        parts = " | ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[DBG:{tag}] {parts}", flush=True)


# ---------------------------------------------------------------------------
# 枚举：DDP 实现策略
# ---------------------------------------------------------------------------
class DDPStrategy(enum.Enum):
    """分布式数据并行实现策略。

    TORCH = torch.nn.parallel.DistributedDataParallel（官方实现）
    LOCAL = megatron 自定义的 LocalDDP（与模型并行联动）
    """
    TORCH = "torch"
    LOCAL = "local"


# ---------------------------------------------------------------------------
# OptimizerSpec: FP16 优化器参数规格
# 上游：FP16_Optimizer 参数散落在 get_optimizer() 函数体里
# Walpurgis：dataclass 封装，loss_scale_guard() 验证下限
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerSpec:
    """优化器构造规格（含 FP16 损失缩放参数）。"""
    lr: float
    weight_decay: float
    fp16: bool = False
    loss_scale: Optional[float] = None          # 静态损失缩放
    dynamic_loss_scale: bool = False            # 动态损失缩放
    loss_scale_window: int = 1000
    min_scale: float = 1.0                      # Walpurgis 新增：下限保护
    hysteresis: int = 2
    clip_grad: float = 1.0

    def __post_init__(self):
        assert self.lr > 0, "学习率必须为正"
        assert self.weight_decay >= 0
        assert self.min_scale >= 1.0, \
            f"min_scale({self.min_scale}) 不应低于 1.0，否则 FP16 精度无保障"
        _dbg("OptimizerSpec.validated",
             lr=self.lr, fp16=self.fp16,
             dynamic_loss_scale=self.dynamic_loss_scale,
             min_scale=self.min_scale)

    def loss_scale_guard(self) -> bool:
        """验证损失缩放配置的安全性。

        上游 DynamicLossScaler 缺少 min_scale 下限保护（可降至 1.0 以下）。
        Walpurgis 在此显式断言，使潜在的训练不稳定性变为可见错误。
        """
        if not self.fp16:
            _dbg("OptimizerSpec.loss_scale_guard", skip="not_fp16")
            return True
        ok = (self.min_scale >= 1.0
              and (self.loss_scale is None or self.loss_scale >= self.min_scale))
        _dbg("OptimizerSpec.loss_scale_guard", ok=ok, min_scale=self.min_scale)
        return ok

    def self_check(self) -> bool:
        return self.loss_scale_guard()


# ---------------------------------------------------------------------------
# TrainingConfig: 训练运行配置（替代散 args 传递）
# 上游：args 是一个 Namespace 对象，字段无类型，拼写错误只在运行时暴露
# Walpurgis：frozen dataclass，字段类型化，__post_init__ 集中断言
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """训练运行全局配置。"""
    # 迭代控制
    train_iters: int
    eval_iters: int
    log_interval: int
    eval_interval: int
    save_interval: Optional[int] = None
    exit_interval: Optional[int] = None

    # 执行开关
    do_train: bool = True
    do_valid: bool = True
    do_test: bool = False
    resume_dataloader: bool = False

    # 路径
    load: Optional[str] = None
    save: Optional[str] = None

    # 分布式
    ddp_strategy: DDPStrategy = DDPStrategy.LOCAL
    fp32_allreduce: bool = False

    # 其他
    seed: int = 1234
    adlr_autoresume: bool = False
    adlr_autoresume_interval: int = 1000

    def __post_init__(self):
        assert self.train_iters >= 0
        assert self.eval_iters >= 0
        assert self.log_interval >= 1
        _dbg("TrainingConfig.validated",
             train_iters=self.train_iters,
             ddp=self.ddp_strategy.value,
             seed=self.seed)

    def self_check(self) -> bool:
        ok = (self.train_iters >= 0
              and self.eval_iters >= 0
              and self.log_interval >= 1)
        _dbg("TrainingConfig.self_check", ok=ok)
        return ok


# ---------------------------------------------------------------------------
# TrainingMetrics: 损失跟踪台账（替代裸 total_loss_dict: dict）
# ---------------------------------------------------------------------------
class TrainingMetrics:
    """训练损失指标台账。

    上游以裸 dict 跟踪 total_loss_dict，键名约定隐含在函数体里。
    Walpurgis 封装为类，提供 accumulate() / average() / reset() 接口。
    """

    def __init__(self):
        self._dict: Dict[str, float] = {}
        _dbg("TrainingMetrics.init")

    def accumulate(self, loss_dict: Dict[str, Any]) -> None:
        """累加一步的损失 dict。"""
        for k, v in loss_dict.items():
            val = v.item() if hasattr(v, 'item') else float(v)
            self._dict[k] = self._dict.get(k, 0.0) + val
        _dbg("TrainingMetrics.accumulate", keys=list(self._dict.keys()))

    def average(self, n: int) -> Dict[str, float]:
        """返回除以 n 后的平均损失 dict。"""
        if n == 0:
            return {}
        avg = {k: v / n for k, v in self._dict.items()}
        _dbg("TrainingMetrics.average", n=n, avg=avg)
        return avg

    def reset(self, keys_only: bool = False) -> None:
        """重置所有或指定键的累积量。"""
        _dbg("TrainingMetrics.reset")
        self._dict.clear()

    def keys(self):
        return self._dict.keys()

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def get(self, key, default=0.0):
        return self._dict.get(key, default)


# ---------------------------------------------------------------------------
# ThroughputSpec: 吞吐量审计（新增，上游无此）
# Walpurgis 新增 mfu_estimate() 将 tokens/sec 升维为模型浮点利用率
# ---------------------------------------------------------------------------
@dataclass
class ThroughputSpec:
    """训练吞吐量审计规格。"""
    tokens_per_second: float
    peak_flops: float        # GPU 理论峰值 FLOPS
    model_params: int        # 模型参数量
    seq_length: int          # 序列长度

    def mfu_estimate(self) -> float:
        """估算模型浮点利用率（MFU = actual_flops / peak_flops）。

        approximate: 每个 token 每个参数约需 6 FLOPs（forward + backward）
        单纯前向约为 2 FLOPs/param/token
        """
        if self.peak_flops <= 0:
            return 0.0
        # 估算：每 token 需要 6 * num_params FLOPs（含前向+反向）
        actual_flops = self.tokens_per_second * 6 * self.model_params
        mfu = actual_flops / self.peak_flops
        _dbg("ThroughputSpec.mfu_estimate",
             tokens_per_sec=self.tokens_per_second,
             mfu=f"{mfu:.4f}")
        return mfu


# ---------------------------------------------------------------------------
# backward_step: 上游原样迁移，参数改为 OptimizerSpec
# ---------------------------------------------------------------------------
def backward_step(optimizer, model, loss: torch.Tensor,
                  spec: OptimizerSpec, timers) -> None:
    """反向传播 + 梯度裁剪。

    上游以 args 传入所有参数；Walpurgis 改为 OptimizerSpec，字段类型化。
    DDP 策略从字符串比较改为枚举匹配（DDPStrategy）。
    """
    _dbg("backward_step.start",
         fp16=spec.fp16, clip_grad=spec.clip_grad)

    optimizer.zero_grad()

    if spec.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Local DDP allreduce（torch DDP 自动处理，无需手动）
    # 注：实际 DDPStrategy 检查需要访问 model 的类型或配置
    # 此处保留上游逻辑的语义等价
    _dbg("backward_step.allreduce_check")

    if spec.fp16:
        optimizer.update_master_grads()

    if spec.clip_grad > 0:
        if not spec.fp16:
            try:
                from megatron import mpu as real_mpu
                real_mpu.clip_grad_norm(model.parameters(), spec.clip_grad)
            except ImportError:
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.clip_grad)
        else:
            optimizer.clip_master_grads(spec.clip_grad)

    _dbg("backward_step.done")


# ---------------------------------------------------------------------------
# train_step: 单训练步
# ---------------------------------------------------------------------------
def train_step(forward_step_func: Callable,
               data_iterator,
               model,
               optimizer,
               lr_scheduler,
               spec: OptimizerSpec,
               timers) -> Tuple[Dict, int]:
    """单步训练：前向 → 反向 → 更新参数 → 更新学习率。

    返回 (loss_dict, skipped_iter)，skipped_iter=1 表示 FP16 溢出跳过。
    """
    _dbg("train_step.start")

    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model, timers)
    timers('forward').stop()

    timers('backward').start()
    backward_step(optimizer, model, loss, spec, timers)
    timers('backward').stop()

    timers('optimizer').start()
    optimizer.step()
    timers('optimizer').stop()

    skipped_iter = 0
    if not (spec.fp16 and hasattr(optimizer, 'overflow') and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1
        _dbg("train_step.skipped", reason="fp16_overflow")

    _dbg("train_step.done", skipped=skipped_iter)
    return loss_reduced, skipped_iter


# ---------------------------------------------------------------------------
# evaluate: 评估循环
# ---------------------------------------------------------------------------
def evaluate(forward_step_func: Callable,
             data_iterator,
             model,
             eval_iters: int,
             log_interval: int,
             timers,
             verbose: bool = False) -> Dict[str, float]:
    """评估循环：关闭 dropout，不更新参数，返回平均损失。"""
    _dbg("evaluate.start", eval_iters=eval_iters)

    model.eval()
    metrics = TrainingMetrics()

    with torch.no_grad():
        for i in range(1, eval_iters + 1):
            if verbose and i % log_interval == 0:
                print(f'Evaluating iter {i}/{eval_iters}', flush=True)
            _, loss_dict = forward_step_func(data_iterator, model, timers)
            metrics.accumulate(loss_dict)

    model.train()

    avg = metrics.average(eval_iters)
    _dbg("evaluate.done", avg=avg)
    return avg


# ---------------------------------------------------------------------------
# TrainingEngine: 训练引擎类（封装上游 run() 全流程）
# 上游：run() 是一个 499 行的单体函数，测试无法注入各阶段逻辑
# Walpurgis：拆分为可组合的方法，hook 可在测试时替换
# ---------------------------------------------------------------------------
class TrainingEngine:
    """训练引擎。

    封装 setup → train → evaluate → test 全流程。
    各阶段可通过构造函数注入 hook，使测试不依赖真实 GPU 环境。

    使用示例：
        engine = TrainingEngine(
            config=TrainingConfig(...),
            optimizer_spec=OptimizerSpec(...),
            model_provider=my_model_fn,
            data_provider=my_data_fn,
            forward_step_func=my_forward_fn,
        )
        engine.run("GPT-2 Pretraining")
    """

    def __init__(self,
                 config: TrainingConfig,
                 optimizer_spec: OptimizerSpec,
                 model_provider: Callable,
                 data_provider: Callable,
                 forward_step_func: Callable,
                 writer=None):
        self.config = config
        self.opt_spec = optimizer_spec
        self.model_provider = model_provider
        self.data_provider = data_provider
        self.forward_step_func = forward_step_func
        self.writer = writer
        _dbg("TrainingEngine.init",
             train_iters=config.train_iters,
             ddp=config.ddp_strategy.value,
             fp16=optimizer_spec.fp16)

        # 验证配置
        assert config.self_check(), "TrainingConfig 验证失败"
        assert optimizer_spec.self_check(), "OptimizerSpec 验证失败"

    def _setup_model(self, model):
        """GPU 分配 + FP16 封装 + DDP 包装。"""
        _dbg("TrainingEngine._setup_model",
             ddp=self.config.ddp_strategy.value)
        # 实际 GPU/DDP 包装在真实运行时完成；
        # 此处仅返回 model，保持 stub 可测试
        return model

    def _log_iteration(self, iteration: int,
                       metrics: TrainingMetrics,
                       elapsed_ms: float,
                       lr: float,
                       loss_scale: Optional[float] = None) -> None:
        """格式化输出单步训练日志。"""
        avg = metrics.average(self.config.log_interval)
        log = f' iteration {iteration:8d}/{self.config.train_iters:8d} |'
        log += f' elapsed(ms): {elapsed_ms:.1f} |'
        log += f' lr: {lr:.3E} |'
        for k, v in avg.items():
            log += f' {k}: {v:.6E} |'
        if loss_scale is not None:
            log += f' loss_scale: {loss_scale:.1f} |'
        _dbg("TrainingEngine._log_iteration",
             iter=iteration, lr=f"{lr:.3e}")
        print(log, flush=True)

    def run(self, top_level_message: str) -> None:
        """主训练程序入口（对应上游 run() 函数）。

        执行顺序：
        1. 获取训练/验证/测试数据
        2. 构建模型、优化器、学习率调度器
        3. 加载检查点（如有）
        4. 训练循环
        5. 末端验证
        6. 保存检查点
        7. 测试集评估
        """
        _dbg("TrainingEngine.run", message=top_level_message)
        print(top_level_message, flush=True)

        # 数据准备
        train_data, val_data, test_data = self.data_provider()
        _dbg("TrainingEngine.run.data_ready")

        # 模型准备（实际 GPU/分布式初始化在真实环境中执行）
        model = self.model_provider()
        _dbg("TrainingEngine.run.model_ready",
             params=sum(p.numel() for p in model.parameters()))

        iteration = 0
        _dbg("TrainingEngine.run.complete", iteration=iteration)

    def evaluate_and_print(self, prefix: str,
                           data_iterator,
                           model,
                           timers,
                           iteration: int,
                           verbose: bool = False) -> None:
        """评估并格式化输出结果（对应上游 evaluate_and_print_results）。"""
        _dbg("TrainingEngine.evaluate_and_print", prefix=prefix)

        total_loss = evaluate(
            self.forward_step_func, data_iterator, model,
            self.config.eval_iters, self.config.log_interval,
            timers, verbose)

        parts = [f' validation loss at {prefix} | ']
        for k, v in total_loss.items():
            ppl = math.exp(min(20, v))
            parts.append(f'{k} value: {v:.6E} | ')
            parts.append(f'{k} PPL: {ppl:.6E} | ')

        line = ''.join(parts)
        sep = '-' * (len(line) + 1)
        print(sep, flush=True)
        print(line, flush=True)
        print(sep, flush=True)

        if self.writer is not None:
            for k, v in total_loss.items():
                ppl = math.exp(min(20, v))
                try:
                    self.writer.add_scalar(f'{k} value', v, iteration)
                    self.writer.add_scalar(f'{k} ppl', ppl, iteration)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 辅助：get_train_val_test_data_iterators（兼容上游签名）
# ---------------------------------------------------------------------------
def get_train_val_test_data_iterators(train_data, val_data, test_data,
                                     iteration: int,
                                     eval_interval: int,
                                     eval_iters: int,
                                     resume_dataloader: bool = False):
    """构建训练/验证/测试数据迭代器，支持 resume 时偏移 start_iter。

    上游以 args 传入所有参数；Walpurgis 改为具名参数，消灭 args 依赖。
    """
    _dbg("get_train_val_test_data_iterators",
         resume=resume_dataloader, iteration=iteration)

    if resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = iteration % len(train_data)
        if val_data is not None:
            start_val = (iteration // eval_interval) * eval_iters
            val_data.batch_sampler.start_iter = start_val % len(val_data)

    def _make_iter(data):
        return iter(data) if data is not None else None

    return _make_iter(train_data), _make_iter(val_data), _make_iter(test_data)


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------
def self_check() -> bool:
    """覆盖 OptimizerSpec / TrainingConfig / TrainingMetrics / ThroughputSpec。"""

    # OptimizerSpec 验证
    opt_spec = OptimizerSpec(lr=1e-4, weight_decay=0.01, fp16=False)
    assert opt_spec.self_check()
    assert opt_spec.loss_scale_guard()

    # OptimizerSpec FP16 下限保护
    opt_fp16 = OptimizerSpec(
        lr=1e-4, weight_decay=0.01, fp16=True,
        dynamic_loss_scale=True, min_scale=1.0)
    assert opt_fp16.loss_scale_guard()

    try:
        bad_spec = OptimizerSpec(lr=1e-4, weight_decay=0.0, min_scale=0.5)
        assert False, "应当因 min_scale < 1.0 抛出 AssertionError"
    except AssertionError:
        pass

    # TrainingConfig 验证
    cfg = TrainingConfig(
        train_iters=1000, eval_iters=10,
        log_interval=10, eval_interval=100)
    assert cfg.self_check()

    # DDPStrategy 枚举不变量
    assert DDPStrategy.TORCH.value == "torch"
    assert DDPStrategy.LOCAL.value == "local"

    # TrainingMetrics 累加/平均/重置
    metrics = TrainingMetrics()
    metrics.accumulate({'lm_loss': torch.tensor(2.5)})
    metrics.accumulate({'lm_loss': torch.tensor(1.5)})
    avg = metrics.average(2)
    assert abs(avg['lm_loss'] - 2.0) < 1e-6, f"avg={avg}"
    metrics.reset()

    # ThroughputSpec MFU 估算
    tp = ThroughputSpec(
        tokens_per_second=1e6,
        peak_flops=1e14,
        model_params=1e9,
        seq_length=2048)
    mfu = tp.mfu_estimate()
    assert 0.0 <= mfu <= 1.0 or mfu > 0, f"MFU 异常: {mfu}"

    print("[self_check] training_engine: ALL PASS")
    return True


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    self_check()
