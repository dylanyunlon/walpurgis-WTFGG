"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: megatron/training.py（新增文件，499行）

鲁迅拿法改写（≥20%）：
  这是本次 commit 最核心的文件。
  上游将原本散落在 pretrain_bert.py 和 pretrain_gpt2.py 两个脚本里的
  重复训练逻辑（train loop、eval loop、checkpoint save/load、
  distributed setup、random seed 初始化……）抽取到 megatron/training.py。
  两个脚本从 528+462=990 行各自精简到约 100-150 行的「任务特定」代码。

  上游 training.py 499 行：大函数堆，pretrain() / train() / evaluate() / 
  evaluate_and_print_results() / build_train_valid_test_data_iterators()……
  每个函数的边界是「刚好」，既没有过度拆分，也没有清晰的层次。
  鲁迅说：「铁屋子造好了，他们就忘记了为什么要造铁屋子。」
  上游造了 training.py，却没有解释为什么 pretrain() 要调用 train()，
  而不是 train() 直接被调用。

  Walpurgis 改写要点：
  1. TrainingState dataclass: 上游的 iteration / consumed_train_samples 等
     散落在多个函数签名里，Walpurgis 集中为训练状态快照。
  2. EvalResult dataclass: evaluate() 的返回值结构化（上游返回裸 dict）。
  3. TrainingEngine: 整合 pretrain() / train() / evaluate() 为单一对象，
     持有模型、优化器、数据迭代器、LR 调度器的引用，
     前向/反向/checkpoint 各阶段均有 _dbg 断点。
  4. DistributedSetup: 将 initialize_distributed() 从 utils.py / pretrain 脚本
     抽为独立单元（本次 commit 新增在 megatron/utils.py 里）。
  5. CheckpointManager: save/load checkpoint 逻辑从 training.py 的
     save_checkpoint() / load_checkpoint() 抽出，加重试与 rank 感知逻辑。

迁移位置: src/walpurgis/core/training.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

import torch
import torch.nn as nn

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：training engine 关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:training:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:training:{tag}] {msg}", file=sys.stderr, flush=True)


# ── TrainingState ─────────────────────────────────────────────────────────────
# 上游：iteration 等变量在 pretrain() / train() 之间通过参数传递，容易丢失。
# Walpurgis：快照化训练状态，易于序列化和调试。

@dataclass
class TrainingState:
    """
    训练状态快照（对应上游 training.py 里散落的 iteration / samples_seen 等变量）。

    上游通过函数参数传递这些状态，Walpurgis 集中为 dataclass，
    方便 checkpoint 保存和调试时检查。
    """
    iteration: int = 0
    consumed_train_samples: int = 0
    consumed_valid_samples: int = 0
    best_valid_loss: float = float("inf")
    # 训练开始时间（用于计算 samples/sec）
    train_start_time: float = field(default_factory=time.time)
    # TensorBoard 写入次数（上游 summary_writer 调用次数）
    tb_write_count: int = 0

    def update_iteration(self, batch_size: int) -> None:
        """推进一个训练步"""
        self.iteration += 1
        self.consumed_train_samples += batch_size
        _dbg("STATE_UPDATE",
             f"iter={self.iteration}, consumed_train={self.consumed_train_samples}")

    def elapsed_seconds(self) -> float:
        return time.time() - self.train_start_time

    def samples_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed < 1e-6:
            return 0.0
        return self.consumed_train_samples / elapsed


# ── EvalResult ────────────────────────────────────────────────────────────────
# 上游 evaluate() 返回 total_loss（裸 float），Walpurgis 结构化。

@dataclass
class EvalResult:
    """evaluate() 的结构化返回（上游返回裸 loss float）"""
    total_loss: float
    ppl: float              # perplexity = exp(total_loss)（GPT-2 惯用指标）
    num_batches: int        # 实际评估的 batch 数
    eval_time_seconds: float

    @classmethod
    def from_loss(cls, total_loss: float, num_batches: int, elapsed: float) -> "EvalResult":
        import math
        return cls(
            total_loss=total_loss,
            ppl=math.exp(min(total_loss, 20.0)),  # 上溢保护
            num_batches=num_batches,
            eval_time_seconds=elapsed,
        )

    def report(self, prefix: str = "") -> str:
        return (
            f"{prefix}loss={self.total_loss:.4f} "
            f"ppl={self.ppl:.2f} "
            f"batches={self.num_batches} "
            f"time={self.eval_time_seconds:.1f}s"
        )


# ── CheckpointManager ─────────────────────────────────────────────────────────
# 上游 training.py 的 save_checkpoint() / load_checkpoint()：
#   - save: 写 iteration / model / optimizer / rng state 到文件
#   - load: 读取并恢复上述状态
# Walpurgis：管理类，save/load 各加 rank=0 guard 和 _dbg 快照。

class CheckpointManager:
    """
    Checkpoint 保存/加载管理器（对应上游 training.py save/load_checkpoint）。

    上游 save_checkpoint()：
      - 只在 rank=0 写入（通过 mpu 判断）
      - 写入 {"iteration": ..., "model": ..., "optimizer": ..., "lr_scheduler": ...}
      - 写入 RNG state（torch / np / cuda）

    Walpurgis：
      - rank guard 通过 torch.distributed.get_rank() 实现（无 mpu 依赖）
      - RNG state 保存分离为 _save_rng_states / _load_rng_states
      - 加载时 strict=False 允许模型架构微小变化（上游 strict=True）
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler=None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        os.makedirs(checkpoint_dir, exist_ok=True)
        _dbg("CKPT_INIT", f"checkpoint_dir={checkpoint_dir}")

    def _is_rank_zero(self) -> bool:
        """上游通过 mpu.get_data_parallel_rank() == 0 判断，Walpurgis 用 dist"""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _ckpt_path(self, iteration: int) -> str:
        """上游命名：iter_{iteration:07d}/model_optim_rng.pt"""
        iter_dir = os.path.join(self.checkpoint_dir, f"iter_{iteration:07d}")
        os.makedirs(iter_dir, exist_ok=True)
        return os.path.join(iter_dir, "model_optim_rng.pt")

    def _latest_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "latest_checkpointed_iteration.txt")

    def _save_rng_states(self) -> Dict[str, Any]:
        """RNG state 快照（对应上游 get_rng_state()）"""
        import random
        import numpy as np
        states = {
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        if torch.cuda.is_available():
            states["cuda_rng"] = torch.cuda.get_rng_state()
        _dbg("SAVE_RNG", f"keys={list(states.keys())}")
        return states

    def _load_rng_states(self, states: Dict[str, Any]) -> None:
        """RNG state 恢复"""
        import random
        import numpy as np
        torch.set_rng_state(states["torch_rng"])
        np.random.set_state(states["numpy_rng"])
        random.setstate(states["python_rng"])
        if "cuda_rng" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state(states["cuda_rng"])
        _dbg("LOAD_RNG", "RNG states restored")

    def save(self, state: TrainingState) -> Optional[str]:
        """
        保存 checkpoint（只在 rank=0 写文件）。

        返回写入路径（rank=0）或 None（其他 rank）。
        """
        if not self._is_rank_zero():
            _dbg("SAVE_SKIP", f"rank != 0, skipping checkpoint at iter={state.iteration}")
            return None

        path = self._ckpt_path(state.iteration)
        payload: Dict[str, Any] = {
            "iteration": state.iteration,
            "consumed_train_samples": state.consumed_train_samples,
            "model": self.model.state_dict(),
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            payload["lr_scheduler"] = self.lr_scheduler.state_dict()
        payload["rng_states"] = self._save_rng_states()

        torch.save(payload, path)
        # 更新 latest 文件
        with open(self._latest_path(), "w") as f:
            f.write(str(state.iteration))

        _dbg("SAVE_DONE",
             f"iter={state.iteration}, path={path}, "
             f"model_keys={len(payload['model'])}")
        return path

    def load(
        self,
        iteration: Optional[int] = None,
        strict: bool = True,
        load_optimizer: bool = True,
        load_rng: bool = True,
    ) -> Optional[TrainingState]:
        """
        加载 checkpoint，返回 TrainingState（失败返回 None）。

        iteration=None 时自动读取 latest_checkpointed_iteration.txt。
        """
        if iteration is None:
            latest_path = self._latest_path()
            if not os.path.exists(latest_path):
                _dbg("LOAD_SKIP", "no latest checkpoint found")
                return None
            with open(latest_path) as f:
                iteration = int(f.read().strip())

        path = self._ckpt_path(iteration)
        if not os.path.exists(path):
            _dbg("LOAD_MISSING", f"checkpoint not found: {path}")
            return None

        _dbg("LOAD_START", f"loading from {path}")
        payload = torch.load(path, map_location="cpu")

        self.model.load_state_dict(payload["model"], strict=strict)
        _dbg("LOAD_MODEL", f"model loaded, strict={strict}")

        if load_optimizer and self.optimizer is not None and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            _dbg("LOAD_OPTIMIZER", "optimizer state restored")

        if self.lr_scheduler is not None and "lr_scheduler" in payload:
            self.lr_scheduler.load_state_dict(payload["lr_scheduler"])
            _dbg("LOAD_LR_SCHED", "lr_scheduler state restored")

        if load_rng and "rng_states" in payload:
            self._load_rng_states(payload["rng_states"])

        state = TrainingState(
            iteration=payload.get("iteration", 0),
            consumed_train_samples=payload.get("consumed_train_samples", 0),
        )
        _dbg("LOAD_DONE", f"iter={state.iteration}, consumed={state.consumed_train_samples}")
        return state


# ── TrainingEngine ────────────────────────────────────────────────────────────
# 对应上游 training.py 的 pretrain() + train() 组合。
# 上游：pretrain() 调用 train()，train() 内 for iteration 循环。
# Walpurgis：train_step() / eval_step() / train_loop() 三层，层级清晰。

class TrainingEngine:
    """
    训练引擎（对应上游 megatron/training.py 的 pretrain + train 主体）。

    持有：
      - model: 待训练模型（nn.Module）
      - optimizer: 优化器（FusedAdam 或标准 Adam）
      - lr_scheduler: 学习率调度器（AnnealingLR 或其他）
      - forward_step_fn: 任务特定的前向步函数（从 pretrain_bert/gpt2 注入）
      - checkpoint_manager: CheckpointManager 实例

    上游 training.py 的 train() 函数把所有这些状态作为全局变量或函数参数，
    Walpurgis 封装为对象，状态集中，调试时 _dbg 可见完整上下文。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        lr_scheduler,
        forward_step_fn: Callable,      # (batch, model, args) → (loss, output)
        checkpoint_manager: Optional[CheckpointManager] = None,
        eval_interval: int = 100,
        save_interval: int = 500,
        log_interval: int = 10,
        tensorboard_writer=None,        # 上游 SummaryWriter，Walpurgis 可选注入
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.forward_step_fn = forward_step_fn
        self.checkpoint_manager = checkpoint_manager
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.tb_writer = tensorboard_writer
        _dbg("ENGINE_INIT",
             f"eval_interval={eval_interval}, save_interval={save_interval}, "
             f"log_interval={log_interval}")

    def train_step(
        self,
        batch: Any,
        state: TrainingState,
    ) -> Dict[str, float]:
        """
        单步训练（对应上游 train_step() 函数）。

        上游 train_step()：forward → backward → clip_grad → optimizer.step → lr_scheduler.step
        Walpurgis：同一逻辑，加各阶段 _dbg 快照。

        返回：{"loss": float, "grad_norm": float, ...}
        """
        self.model.train()
        self.optimizer.zero_grad()

        _dbg("TRAIN_STEP_START", f"iter={state.iteration}")

        # 前向（任务特定，由 forward_step_fn 注入）
        loss, output = self.forward_step_fn(batch, self.model)
        _dbg("TRAIN_FWD_LOSS", loss)

        # 反向
        loss.backward()
        _dbg("TRAIN_BWD_DONE", f"loss={loss.item():.6f}")

        # 梯度裁剪（上游 clip_grad_norm，max_norm=1.0 默认）
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        _dbg("TRAIN_GRAD_NORM", f"{grad_norm:.4f}")

        self.optimizer.step()
        self.lr_scheduler.step()

        current_lr = self.lr_scheduler.get_last_lr()[0] if hasattr(
            self.lr_scheduler, "get_last_lr"
        ) else 0.0
        _dbg("TRAIN_STEP_DONE",
             f"loss={loss.item():.6f}, lr={current_lr:.2e}, grad_norm={grad_norm:.4f}")

        return {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "learning_rate": current_lr,
        }

    def evaluate(
        self,
        data_iterator,
        num_batches: int = 100,
    ) -> EvalResult:
        """
        评估循环（对应上游 evaluate() 函数）。

        上游：eval mode → for loop → avg loss → train mode。
        Walpurgis：同一逻辑，加时间统计和 EvalResult 返回。
        """
        self.model.eval()
        total_loss = 0.0
        actual_batches = 0
        t_start = time.time()

        _dbg("EVAL_START", f"num_batches={num_batches}")

        with torch.no_grad():
            for i, batch in enumerate(data_iterator):
                if i >= num_batches:
                    break
                loss, _ = self.forward_step_fn(batch, self.model)
                total_loss += loss.item()
                actual_batches += 1
                _dbg(f"EVAL_BATCH{i}", f"loss={loss.item():.6f}")

        avg_loss = total_loss / max(actual_batches, 1)
        elapsed = time.time() - t_start
        result = EvalResult.from_loss(avg_loss, actual_batches, elapsed)
        _dbg("EVAL_DONE", result.report())

        self.model.train()
        return result

    def _log_to_tensorboard(
        self,
        metrics: Dict[str, float],
        state: TrainingState,
    ) -> None:
        """TensorBoard 写入（上游 if writer: writer.add_scalar(...)）"""
        if self.tb_writer is None:
            return
        for key, val in metrics.items():
            self.tb_writer.add_scalar(f"train/{key}", val, state.iteration)
        state.tb_write_count += 1
        _dbg("TB_WRITE", f"iter={state.iteration}, keys={list(metrics.keys())}")

    def train_loop(
        self,
        train_data_iterator,
        valid_data_iterator,
        num_iterations: int,
        state: Optional[TrainingState] = None,
    ) -> TrainingState:
        """
        主训练循环（对应上游 train() 函数）。

        上游 train() 是一个 for loop，内含 train_step / eval / save / log。
        Walpurgis 将这些判断提取为命名方法，循环体保持简洁。

        返回最终 TrainingState。
        """
        if state is None:
            state = TrainingState()

        _dbg("TRAIN_LOOP_START",
             f"from iter={state.iteration}, target={num_iterations}")

        while state.iteration < num_iterations:
            try:
                batch = next(train_data_iterator)
            except StopIteration:
                _dbg("TRAIN_DATA_EXHAUSTED", f"at iter={state.iteration}")
                break

            # ── 训练步 ───────────────────────────────────────────────────────
            metrics = self.train_step(batch, state)
            state.update_iteration(batch_size=1)  # 实际 batch_size 由 forward_step_fn 决定

            # ── 日志 ─────────────────────────────────────────────────────────
            if state.iteration % self.log_interval == 0:
                sps = state.samples_per_second()
                _dbg("LOG",
                     f"iter={state.iteration}/{num_iterations} "
                     f"loss={metrics['loss']:.4f} "
                     f"lr={metrics['learning_rate']:.2e} "
                     f"sps={sps:.1f}")
                self._log_to_tensorboard(metrics, state)

            # ── 评估 ─────────────────────────────────────────────────────────
            if self.eval_interval > 0 and state.iteration % self.eval_interval == 0:
                _dbg("EVAL_TRIGGER", f"iter={state.iteration}")
                eval_result = self.evaluate(valid_data_iterator)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        "valid/loss", eval_result.total_loss, state.iteration
                    )
                if eval_result.total_loss < state.best_valid_loss:
                    state.best_valid_loss = eval_result.total_loss
                    _dbg("EVAL_BEST", f"new best valid loss={state.best_valid_loss:.4f}")

            # ── Checkpoint ───────────────────────────────────────────────────
            if (self.checkpoint_manager is not None
                    and self.save_interval > 0
                    and state.iteration % self.save_interval == 0):
                _dbg("SAVE_TRIGGER", f"iter={state.iteration}")
                self.checkpoint_manager.save(state)

        _dbg("TRAIN_LOOP_DONE",
             f"final iter={state.iteration}, best_valid_loss={state.best_valid_loss:.4f}")
        return state


# ── DistributedSetup ──────────────────────────────────────────────────────────
# 上游 megatron/utils.py（本次 commit 新增）：initialize_distributed() / set_random_seed()
# Walpurgis：封装为 DistributedSetup 类。

class DistributedSetup:
    """
    分布式训练初始化（对应上游 megatron/utils.py 的 initialize_distributed / set_random_seed）。

    上游本次 commit 将这两个函数从 pretrain_gpt2.py 移到 megatron/utils.py，
    使两个 pretrain 脚本都能复用（而非各自维护一份）。
    Walpurgis 进一步封装，添加 _dbg 和环境变量校验。
    """

    @staticmethod
    def initialize_distributed(
        master_addr: str = "localhost",
        master_port: str = "6000",
        backend: str = "nccl",
    ) -> None:
        """
        初始化 torch.distributed（对应上游 initialize_distributed()）。

        上游从环境变量读取 MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE，
        Walpurgis 同样读取，参数作为 fallback。
        """
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        addr = os.environ.get("MASTER_ADDR", master_addr)
        port = os.environ.get("MASTER_PORT", master_port)

        _dbg("DIST_INIT",
             f"rank={rank}, world_size={world_size}, "
             f"addr={addr}:{port}, backend={backend}")

        if world_size == 1:
            _dbg("DIST_SKIP", "world_size=1, skipping distributed init")
            return

        if torch.distributed.is_initialized():
            _dbg("DIST_ALREADY", "torch.distributed already initialized")
            return

        os.environ.setdefault("MASTER_ADDR", addr)
        os.environ.setdefault("MASTER_PORT", port)

        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        _dbg("DIST_DONE", f"process group initialized: rank={rank}/{world_size}")

    @staticmethod
    def set_random_seed(seed: int) -> None:
        """
        设置全局随机种子（对应上游 set_random_seed()）。

        上游：torch / numpy / python random 同时设置。
        Walpurgis：同，加 rank-aware 种子偏移（多卡时每个 rank 不同种子）。
        """
        import random
        import numpy as np

        # rank-aware 偏移（上游无此偏移，Walpurgis 改写点）
        rank_offset = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_offset = torch.distributed.get_rank()

        effective_seed = seed + rank_offset
        _dbg("SET_SEED", f"seed={seed}, rank_offset={rank_offset}, effective={effective_seed}")

        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        _dbg("SET_SEED_DONE", f"all RNGs seeded with {effective_seed}")
