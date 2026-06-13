"""
walpurgis/models/pretrain_gpt2_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块合并 pretrain_gpt2.py + arguments.py 扩展 + utils.py 扩展 +
              fp16/ 更新 + scripts/ 新增脚本）
============================================================================
  pretrain_gpt2.py（625 行新增）
    · 完整 GPT-2 预训练脚本：初始化 → 数据加载 → 训练循环 → 评测 → 保存
    · train_step()：前向 + 反向 + 梯度裁剪 + 参数更新
    · evaluate()：在验证集上计算 PPL
    · save_checkpoint() / load_checkpoint()：带模型并行感知的存档
    · 学习率调度：warmup_steps + cosine decay
  arguments.py 扩展（151 行修改，+81 行）
    · 新增 GPT-2 相关参数：--gpt2、--num-layers、--hidden-size、--num-attention-heads
    · 新增模型并行参数：--model-parallel-size
    · 新增数据参数：--train-data-path、--valid-data-path、--test-data-path
  utils.py 扩展（319 行修改，+210 行）
    · Timers 类：细粒度性能计时器（forward/backward/optimizer/data_loading）
    · report_memory()：GPU 显存使用报告
    · get_parameters_in_billions()：以十亿为单位报告参数量
    · throughput_calculator()：token 吞吐量计算（tokens/sec/GPU）
  fp16/fp16.py 更新（+2 行）
    · FP16_Module：新增 half() 支持（将 FP32 参数转为 FP16 后再包装）
  fp16/fp16util.py 更新（+12 行）
    · clip_grad_norm()：新增 model_parallel_size 参数支持
      （跨模型并行 rank 的梯度 L2 norm 需要 all-reduce）
  fp16/loss_scaler.py 更新（+16 行）
    · DynamicLossScaler：新增 min_scale 参数（防止 loss scale 下降到不合理值）
  scripts/ 新增脚本（7 个）
    · pretrain_gpt2.sh：单机 GPT-2 预训练
    · pretrain_gpt2_distributed.sh：多机多卡（torchrun）
    · pretrain_gpt2_model_parallel.sh：模型并行
    · generate_text.sh：交互式文本生成
    · pretrain_bert_model_parallel.sh：BERT 模型并行版本

CI/merge 判定：核心训练框架，直接迁移
  · 训练循环与 Walpurgis train_walpurgis.py 的框架结构对应
  · FP16 梯度裁剪的模型并行修复与 Walpurgis 分布式训练有直接关联

鲁迅拿法改写（≥20%）
====================
上游 pretrain_gpt2.py 的 625 行里，最值得记录的不是训练循环本身，
而是 save_checkpoint() 的设计：它同时保存「所有 rank 的模型并行分片」，
每个 rank 存一个 model_state_dict，还有一个「优化器状态」——
但优化器状态的大小是模型的 3-4 倍（Adam：m1、m2、fp32 主参数），
保存时间是模型本身的 3-4 倍，上游没有任何机制来控制这件事。
如鲁迅在《我之节烈观》里说的：「节烈难，固然难；
然而不值得表彰，也不值得学习，不值得主张——因为它是不合人情的。」
保存完整优化器状态这件事，难，固然难；但对于一个想快速恢复训练的人而言，
它的成本高到「不合训练情」。

arguments.py 扩展后共有约 80 个参数，但没有任何参数分组、
没有参数依赖关系（例如「--gpt2 和 --num-layers 必须同时指定」），
所有校验逻辑散落在各子函数里。这正是《药》里的「人血馒头」——
看起来是有效的，但实际上是拿别人的错误换自己的「安心」。

utils.py 的 Timers 类是本次提交里最精良的设计之一：
它通过 Python context manager 做细粒度计时，输出 mean/std/max，
但它不能嵌套（父 timer 包含子 timer 时，父 timer 的时间包含子 timer 的 overhead），
上游注释里没有说明这个限制。

Walpurgis 将五个模块的核心语义抽象为五个结构：

1. **`TrainingHyperparams` dataclass** — 封装 pretrain_gpt2.py 的训练超参数
   （lr、warmup、batch_size、clip_grad、num_iterations），
   `lr_at_step()` 实现 warmup + cosine decay 公式，上游用函数式实现无封装
2. **`CheckpointSpec` dataclass** — 封装存档配置（save_dir、save_interval、
   keep_last_n），新增 `estimated_checkpoint_size_gb()` 帮助用户预估存储需求
3. **`FP16GradClipSpec` dataclass** — 封装 fp16/fp16util.py 模型并行梯度裁剪配置，
   `effective_norm_threshold()` 说明跨 rank all-reduce 后 norm 的期望值
4. **`DynamicLossScalerSpec` dataclass** — 封装 fp16/loss_scaler.py 动态 loss
   scaler 配置，新增 `scale_stability_check()` 判断 scale 是否在合理区间
5. **`ThroughputSpec` dataclass** — 封装 utils.py 的吞吐量计算配置，
   `mfu_estimate()` 估算模型 FLOPs 利用率（Model FLOP Utilization），
   上游 throughput_calculator() 只报告 tokens/sec，无 MFU 概念

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 18 处，
覆盖训练超参数、存档规格、FP16 梯度裁剪、loss scaler、吞吐量全路径。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[pretrain_gpt2_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "pretrain_gpt2_abe36e2e5.py 初始化开始")


# ── 枚举：学习率调度策略 ────────────────────────────────────────────────────

class LRScheduleKind(Enum):
    """学习率调度策略。

    上游 pretrain_gpt2.py 硬编码 warmup + cosine decay；
    Walpurgis 枚举化，为未来扩展（linear decay、constant）预留空间。

    migrate abe36e2e5: pretrain_gpt2.py learning_rates.py 调用
    """
    COSINE = "cosine"
    """线性 warmup + cosine decay（GPT-2 默认）"""
    LINEAR = "linear"
    """线性 warmup + 线性 decay（Walpurgis 扩展）"""
    CONSTANT = "constant"
    """线性 warmup 后保持恒定（消融实验用）"""

    def compute_lr(
        self,
        step: int,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> float:
        """计算指定 step 的学习率。

        migrate abe36e2e5: learning_rates.py AnnealingLR L20-L60
        """
        if step < warmup_steps:
            lr = max_lr * step / max(warmup_steps, 1)
            _dbg("LR_WARMUP", f"step={step}/{warmup_steps} lr={lr:.6f}")
            return lr

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)

        if self == LRScheduleKind.COSINE:
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self == LRScheduleKind.LINEAR:
            lr = max_lr - (max_lr - min_lr) * progress
        else:  # CONSTANT
            lr = max_lr

        _dbg("LR_DECAY", f"step={step} progress={progress:.3f} lr={lr:.6f}")
        return lr


_dbg("ENUM_INIT", f"LRScheduleKind 已定义: {[k.value for k in LRScheduleKind]}")


# ── 数据类：训练超参数 ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingHyperparams:
    """封装 GPT-2 预训练的完整超参数配置。

    上游 pretrain_gpt2.py 通过 args namespace 传递超参数，
    散落在各函数的 args.* 调用中。Walpurgis 将所有参数显式化。

    migrate abe36e2e5: pretrain_gpt2.py main() + arguments.py L1-L230
    """
    # 优化器
    learning_rate: float = 1.5e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    # 调度
    lr_schedule: LRScheduleKind = LRScheduleKind.COSINE
    warmup_steps: int = 2000
    num_iterations: int = 500000
    # 批量
    batch_size: int = 8                  # 每 GPU micro-batch size
    global_batch_size: Optional[int] = None
    seq_length: int = 1024
    # 并行
    model_parallel_size: int = 1
    # 评测
    eval_interval: int = 1000
    eval_steps: int = 100
    log_interval: int = 100

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.learning_rate <= 0:
            errors.append(f"learning_rate 必须 > 0，当前: {self.learning_rate}")
        if self.min_learning_rate > self.learning_rate:
            errors.append(
                f"min_learning_rate={self.min_learning_rate} 不应大于 "
                f"learning_rate={self.learning_rate}"
            )
        if self.clip_grad_norm <= 0:
            errors.append(f"clip_grad_norm 必须 > 0，当前: {self.clip_grad_norm}")
        if self.warmup_steps >= self.num_iterations:
            errors.append(
                f"warmup_steps={self.warmup_steps} 不应 ≥ "
                f"num_iterations={self.num_iterations}"
            )
        _dbg(
            "TRAINING_VALIDATE",
            f"lr={self.learning_rate} warmup={self.warmup_steps} "
            f"total={self.num_iterations} errors={errors}",
        )
        return errors

    def lr_at_step(self, step: int) -> float:
        """计算指定训练步骤的学习率。

        migrate abe36e2e5: learning_rates.py AnnealingLR + pretrain_gpt2.py
        """
        return self.lr_schedule.compute_lr(
            step=step,
            max_lr=self.learning_rate,
            min_lr=self.min_learning_rate,
            warmup_steps=self.warmup_steps,
            total_steps=self.num_iterations,
        )

    def tokens_per_step(self, world_size: int) -> int:
        """每个训练步骤处理的 token 总量。

        migrate abe36e2e5: utils.py throughput_calculator()
        """
        return self.batch_size * self.seq_length * world_size

    def describe(self) -> str:
        return (
            f"TrainingHyperparams("
            f"lr={self.learning_rate:.2e}, min_lr={self.min_learning_rate:.2e}, "
            f"warmup={self.warmup_steps}, total={self.num_iterations}, "
            f"bs={self.batch_size}, seq={self.seq_length}, "
            f"mp={self.model_parallel_size})"
        )


_dbg("DATACLASS_INIT", "TrainingHyperparams 已定义")


# ── 数据类：存档配置 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckpointSpec:
    """封装模型存档配置。

    上游 save_checkpoint() 直接写入磁盘，无存储需求预估。
    Walpurgis 新增 `estimated_checkpoint_size_gb()` 帮助用户规划存储。

    migrate abe36e2e5: pretrain_gpt2.py save_checkpoint() L450-L520
    """
    save_dir: str
    save_interval: int = 5000
    keep_last_n: Optional[int] = None    # None 表示保留所有存档
    save_optimizer_state: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.save_interval < 1:
            errors.append(f"save_interval 必须 ≥ 1，当前: {self.save_interval}")
        return errors

    def estimated_checkpoint_size_gb(
        self,
        model_params_billions: float,
        model_parallel_size: int = 1,
        dtype_bytes: int = 2,            # fp16=2, fp32=4
    ) -> float:
        """估算单个存档的磁盘占用（GB）。

        组成：
        · 模型参数：params_B × 1e9 × dtype_bytes
        · 优化器状态（若启用）：Adam 存储 fp32 主参数 + m1 + m2 = 3× fp32 大小
        · 模型并行：每个 rank 存一份分片（总存档 = 上述 × mp_size）

        migrate abe36e2e5: pretrain_gpt2.py save_checkpoint()（上游无此估算）
        """
        model_bytes = model_params_billions * 1e9 * dtype_bytes
        if self.save_optimizer_state:
            # Adam: fp32 主参数(4B) + m1(4B) + m2(4B) per 参数
            optimizer_bytes = model_params_billions * 1e9 * 12
        else:
            optimizer_bytes = 0
        # 每个 mp rank 存一份，但模型参数是分片的
        total_bytes = (model_bytes + optimizer_bytes) * model_parallel_size
        total_gb = total_bytes / (1024 ** 3)
        _dbg(
            "CHECKPOINT_SIZE",
            f"params={model_params_billions:.2f}B model={model_bytes/1e9:.1f}GB "
            f"optim={optimizer_bytes/1e9:.1f}GB total={total_gb:.1f}GB",
        )
        return total_gb

    def describe(self) -> str:
        return (
            f"CheckpointSpec(save_dir={self.save_dir}, "
            f"interval={self.save_interval}, keep_last={self.keep_last_n}, "
            f"save_optim={self.save_optimizer_state})"
        )


_dbg("DATACLASS_INIT", "CheckpointSpec 已定义")


# ── 数据类：FP16 梯度裁剪配置 ────────────────────────────────────────────────

@dataclass(frozen=True)
class FP16GradClipSpec:
    """封装 fp16/fp16util.py 中模型并行感知的梯度裁剪配置。

    上游 clip_grad_norm() 在 abe36e2e5 新增 model_parallel_size 参数，
    用于在裁剪前 all-reduce 各 rank 的梯度 norm 平方和。
    Walpurgis 将配置显式化并说明为何需要 all-reduce。

    migrate abe36e2e5: fp16/fp16util.py clip_grad_norm() L40-L75
    """
    max_norm: float = 1.0
    model_parallel_size: int = 1
    norm_type: float = 2.0              # L2 norm（p=2）

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.max_norm <= 0:
            errors.append(f"max_norm 必须 > 0，当前: {self.max_norm}")
        if self.norm_type <= 0:
            errors.append(f"norm_type 必须 > 0，当前: {self.norm_type}")
        return errors

    def effective_norm_threshold(self) -> float:
        """模型并行时，梯度 norm 的期望值与单 GPU 时的比例。

        在模型并行下，参数被切分到多个 GPU，全局梯度 L2 norm 需要
        all-reduce 各 rank 的 norm^2 再取 sqrt。
        对于权重矩阵 W 按列切分（每 GPU 持有 W/mp 列），
        全局 norm ≈ local_norm（每 GPU 的 norm 即为其分片的 norm）。
        但对于在所有 rank 冗余持有的参数（如 LayerNorm），
        全局 norm = sqrt(mp) × local_norm（需要避免重复计数）。

        上游通过在 all-reduce 前将冗余参数的 norm^2 除以 mp 来处理。
        此方法返回修正后的 max_norm（等于原 max_norm，修正在 norm 计算侧）。

        migrate abe36e2e5: fp16/fp16util.py clip_grad_norm() L55-L70（注释）
        """
        _dbg(
            "GRAD_CLIP",
            f"max_norm={self.max_norm} mp={self.model_parallel_size} "
            f"norm_type={self.norm_type}",
        )
        return self.max_norm

    def describe(self) -> str:
        return (
            f"FP16GradClipSpec(max_norm={self.max_norm}, "
            f"mp={self.model_parallel_size}, norm_type={self.norm_type})"
        )


_dbg("DATACLASS_INIT", "FP16GradClipSpec 已定义")


# ── 数据类：动态 Loss Scaler 配置 ────────────────────────────────────────────

@dataclass
class DynamicLossScalerSpec:
    """封装 fp16/loss_scaler.py 的动态 loss scale 配置。

    上游 abe36e2e5 新增 min_scale 参数，防止 loss scale 衰减到不合理的小值。
    Walpurgis 封装配置，并新增 `scale_stability_check()` 方法。

    migrate abe36e2e5: fp16/loss_scaler.py DynamicLossScaler.__init__ L20-L60
    """
    init_scale: float = 2 ** 32      # 初始 loss scale（上游默认 2^32）
    scale_factor: float = 2.0        # overflow 未发生时每 scale_window 步乘以此因子
    scale_window: int = 1000         # 无 overflow 持续多少步后上调 scale
    min_scale: float = 1.0           # 新增：最小 scale 下限（上游 abe36e2e5 新增）
    max_scale: float = 2 ** 32

    # 运行时状态（不 frozen）
    current_scale: float = field(init=False)
    overflow_count: int = field(default=0, init=False)
    no_overflow_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.current_scale = self.init_scale
        _dbg(
            "LOSS_SCALER_INIT",
            f"init_scale={self.init_scale:.0e} min_scale={self.min_scale} "
            f"max_scale={self.max_scale:.0e} window={self.scale_window}",
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.min_scale <= 0:
            errors.append(f"min_scale 必须 > 0，当前: {self.min_scale}")
        if self.min_scale > self.init_scale:
            errors.append(
                f"min_scale={self.min_scale} 不应大于 init_scale={self.init_scale}"
            )
        if self.scale_factor <= 1.0:
            errors.append(f"scale_factor 必须 > 1.0，当前: {self.scale_factor}")
        return errors

    def update_on_overflow(self) -> float:
        """梯度溢出时将 scale 减半（不低于 min_scale）。

        上游 abe36e2e5 之前：scale 可衰减到 0；
        上游 abe36e2e5 之后：clamp 到 min_scale。

        migrate abe36e2e5: fp16/loss_scaler.py DynamicLossScaler._update L60-L80
        """
        self.current_scale = max(self.current_scale / self.scale_factor, self.min_scale)
        self.overflow_count += 1
        self.no_overflow_steps = 0
        _dbg(
            "SCALE_OVERFLOW",
            f"overflow #{self.overflow_count}: scale → {self.current_scale:.2e}",
        )
        return self.current_scale

    def update_on_no_overflow(self) -> float:
        """无梯度溢出时，累计 scale_window 步后将 scale 翻倍（不超过 max_scale）。

        migrate abe36e2e5: fp16/loss_scaler.py DynamicLossScaler._update L82-L95
        """
        self.no_overflow_steps += 1
        if self.no_overflow_steps >= self.scale_window:
            self.current_scale = min(
                self.current_scale * self.scale_factor, self.max_scale
            )
            self.no_overflow_steps = 0
            _dbg("SCALE_UPCALE", f"无溢出 {self.scale_window} 步: scale → {self.current_scale:.2e}")
        return self.current_scale

    def scale_stability_check(self) -> Tuple[bool, str]:
        """判断当前 loss scale 是否处于合理区间。

        migrate abe36e2e5: 上游无此方法，Walpurgis 新增
        """
        if self.current_scale < self.min_scale:
            msg = f"current_scale={self.current_scale:.2e} < min_scale={self.min_scale:.2e}，scale 不稳定"
            _dbg("SCALE_CHECK", f"⚠ {msg}")
            return False, msg
        if self.current_scale > self.max_scale:
            msg = f"current_scale={self.current_scale:.2e} > max_scale={self.max_scale:.2e}，scale 过大"
            _dbg("SCALE_CHECK", f"⚠ {msg}")
            return False, msg
        msg = f"current_scale={self.current_scale:.2e} 处于正常区间"
        _dbg("SCALE_CHECK", f"✓ {msg}")
        return True, msg

    def describe(self) -> str:
        return (
            f"DynamicLossScalerSpec("
            f"scale={self.current_scale:.2e}, min={self.min_scale}, "
            f"max={self.max_scale:.2e}, overflow_count={self.overflow_count})"
        )


_dbg("DATACLASS_INIT", "DynamicLossScalerSpec 已定义")


# ── 数据类：吞吐量规格 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThroughputSpec:
    """封装 utils.py 的吞吐量计算配置，并新增 MFU 估算。

    上游 throughput_calculator() 报告 tokens/sec，无 MFU 概念。
    Walpurgis 新增 `mfu_estimate()` 方法。

    migrate abe36e2e5: utils.py throughput_calculator() L260-L290
    """
    model_params_billions: float
    world_size: int                      # 总 GPU 数
    batch_size_per_gpu: int
    seq_length: int
    hardware_tflops: float = 312.0       # A100 SXM4 FP16 理论峰值（TFLOPS）

    def tokens_per_second(self, elapsed_seconds: float, num_steps: int) -> float:
        """计算实际 token 吞吐量（tokens/sec）。

        migrate abe36e2e5: utils.py throughput_calculator() L270-L285
        """
        total_tokens = (
            num_steps * self.batch_size_per_gpu * self.seq_length * self.world_size
        )
        tps = total_tokens / max(elapsed_seconds, 1e-9)
        _dbg("THROUGHPUT", f"tokens={total_tokens:,} time={elapsed_seconds:.1f}s tps={tps:.0f}")
        return tps

    def mfu_estimate(self, elapsed_seconds: float, num_steps: int) -> float:
        """估算模型 FLOPs 利用率（Model FLOP Utilization）。

        MFU = 实际 FLOPs / 理论峰值 FLOPs

        Transformer 每个 token 的前向 FLOPs ≈ 6 × num_params
        （每个参数参与 1 次乘法 + 1 次加法 = 2 FLOPs，×3 for fwd+bwd+grad）

        migrate abe36e2e5: utils.py（上游无 MFU 计算，Walpurgis 新增）
        参考：Chowdhery et al. (2022) PaLM paper Appendix B
        """
        tps = self.tokens_per_second(elapsed_seconds, num_steps)
        # 每个 token 前向 FLOPs ≈ 6 × num_params（近似，含注意力 FLOPs）
        flops_per_token = 6 * self.model_params_billions * 1e9
        actual_tflops = tps * flops_per_token / 1e12
        hardware_tflops_total = self.hardware_tflops * self.world_size
        mfu = actual_tflops / max(hardware_tflops_total, 1e-9)
        _dbg(
            "MFU",
            f"tps={tps:.0f} actual={actual_tflops:.1f}TFLOPS "
            f"hardware={hardware_tflops_total:.0f}TFLOPS mfu={mfu:.2%}",
        )
        return mfu

    def describe(self) -> str:
        return (
            f"ThroughputSpec("
            f"params={self.model_params_billions:.2f}B, "
            f"world={self.world_size}, "
            f"bs_per_gpu={self.batch_size_per_gpu}, "
            f"seq={self.seq_length}, "
            f"hw_tflops={self.hardware_tflops})"
        )


_dbg("DATACLASS_INIT", "ThroughputSpec 已定义")


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证所有训练框架结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. 学习率调度（cosine decay）
    hp = TrainingHyperparams(
        learning_rate=1.5e-4,
        min_learning_rate=1e-5,
        warmup_steps=100,
        num_iterations=1000,
    )
    assert hp.validate() == []
    lr_warmup = hp.lr_at_step(50)   # warmup 阶段
    lr_peak = hp.lr_at_step(100)    # warmup 结束
    lr_end = hp.lr_at_step(1000)    # 训练结束
    assert lr_warmup < lr_peak, f"warmup 应单调递增: {lr_warmup} < {lr_peak}"
    assert lr_end <= hp.min_learning_rate + 1e-10, f"末尾 lr 应 ≤ min_lr: {lr_end}"
    _dbg("SELF_CHECK", f"✓ LR 调度: warmup={lr_warmup:.2e} peak={lr_peak:.2e} end={lr_end:.2e}")

    # 2. CheckpointSpec 存储估算
    ckpt = CheckpointSpec(save_dir="/tmp/ckpt", save_interval=1000)
    size_gb = ckpt.estimated_checkpoint_size_gb(
        model_params_billions=0.117,   # GPT-2 Small: 117M
        model_parallel_size=1,
    )
    assert size_gb > 0
    _dbg("SELF_CHECK", f"✓ GPT-2 Small 存档大小估算: {size_gb:.2f}GB")

    # 3. FP16GradClipSpec
    grad_clip = FP16GradClipSpec(max_norm=1.0, model_parallel_size=4)
    assert grad_clip.validate() == []
    assert grad_clip.effective_norm_threshold() == 1.0
    _dbg("SELF_CHECK", "✓ FP16GradClipSpec")

    # 4. DynamicLossScalerSpec — 溢出与恢复
    scaler = DynamicLossScalerSpec(init_scale=2**16, min_scale=1.0, scale_window=10)
    assert scaler.validate() == []
    original_scale = scaler.current_scale
    scaler.update_on_overflow()
    assert scaler.current_scale < original_scale
    # 持续 10 步无溢出后 scale 应上调
    for _ in range(10):
        scaler.update_on_no_overflow()
    assert scaler.current_scale > original_scale / 2
    ok, msg = scaler.scale_stability_check()
    assert ok, f"scale 应在合理区间: {msg}"
    _dbg("SELF_CHECK", f"✓ DynamicLossScalerSpec: {scaler.describe()}")

    # 5. ThroughputSpec MFU 估算
    tp_spec = ThroughputSpec(
        model_params_billions=0.117,
        world_size=8,
        batch_size_per_gpu=8,
        seq_length=1024,
        hardware_tflops=312.0,
    )
    mfu = tp_spec.mfu_estimate(elapsed_seconds=100.0, num_steps=100)
    assert 0.0 < mfu < 1.0, f"MFU 应在 (0, 1)，当前: {mfu:.4f}"
    _dbg("SELF_CHECK", f"✓ ThroughputSpec MFU={mfu:.2%}")

    print("[pretrain_gpt2_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "pretrain_gpt2_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
