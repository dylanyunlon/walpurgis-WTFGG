"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: pretrain_gpt2.py + pretrain_bert.py（合并迁移）

鲁迅拿法改写（≥20%）：
  上游 pretrain_gpt2.py 本次做了三件事：
  其一，get_batch() 里新增 eod_mask 的构造——几行张量操作，
  解释为零，后人看了不知道这个 mask 是干什么用的；
  其二，model_provider() 里新增 DDP_impl 分支（local vs torch），
  代码 if/else 各半，但两个分支的行为差异没有任何文字说明；
  其三，train_step() 里接入了 TensorBoard writer，
  一行 if writer: writer.add_scalar(...)，悄悄塞进去，
  如同在铁屋子里装了一盏电灯，却不告诉人开关在哪儿。
  pretrain_bert.py 同期新增了类似的 TensorBoard 接入和 autoresume 逻辑。

  鲁迅说："改革者的工作，往往是把别人已经做了的事情，
  再做一遍，而且要做得更清楚。"
  Walpurgis 改写：
  1. BatchBuilder: get_batch 逻辑抽为独立类，含 eod_mask 构造说明
  2. ModelProvider: DDP_impl 分支显式建模，差异文档化
  3. TrainStepResult: 训练步结果的结构化数据类（上游是裸 dict/元组）
  4. TensorBoardHook: TensorBoard 接入抽为可替换 hook（解耦 writer 位置）
  5. AutoResumeGuard: ADLR autoresume 逻辑抽为 context manager

迁移位置: src/walpurgis/core/pretrain_gpt2.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：预训练循环关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = f"shape={list(t.shape)} dtype={t.dtype} mean={t.float().mean().item():.4f}"
        print(f"[_dbg:pretrain_gpt2:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:pretrain_gpt2:{tag}] {msg}", file=sys.stderr, flush=True)


# ── TrainStepResult（Walpurgis特有：结构化训练步结果） ────────────────────────
@dataclass
class TrainStepResult:
    """
    单步训练的结构化输出。
    上游是裸 tuple (loss, skipped_iter, ...)，Walpurgis 显式建模。
    """
    loss: float
    grad_norm: float
    skipped: bool = False
    iteration: int = 0
    elapsed_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ── BatchBuilder（对应上游 get_batch + eod_mask 构造） ───────────────────────
class BatchBuilder:
    """
    从数据迭代器取一批数据，构造 token/label/eod_mask 张量。

    上游 get_batch() 中 eod_mask 构造逻辑（a1d04b793 新增）：
      eod_mask = (tokens == eod_token_id).float()
      用于 --eod-mask-loss：将 EOD 位置的 loss 权重设为 0。

    Walpurgis: 将构造逻辑显式化，加断点。
    """

    def __init__(
        self,
        eod_token_id: int = 50256,
        eod_mask_loss: bool = False,
    ) -> None:
        self.eod_token_id = eod_token_id
        self.eod_mask_loss = eod_mask_loss
        _dbg("BATCH_BUILDER_INIT", f"eod_id={eod_token_id} eod_mask_loss={eod_mask_loss}")

    def build(
        self,
        tokens: torch.Tensor,   # [B, T+1]
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        从 tokens [B, T+1] 构造 input_ids, labels, (可选) eod_mask。
        tokens 的最后一个位置是下一个 token 的真值。
        """
        tokens = tokens.to(device)
        input_ids = tokens[:, :-1].contiguous()   # [B, T]
        labels = tokens[:, 1:].contiguous()        # [B, T]

        _dbg("BATCH_INPUT_IDS", input_ids)
        _dbg("BATCH_LABELS", labels)

        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if self.eod_mask_loss:
            # EOD 位置 → 该 label 步的 loss 应被掩掉
            # eod_mask: 1=正常token, 0=EOD（loss 不计）
            eod_mask = (labels != self.eod_token_id).float()  # [B, T]
            result["eod_mask"] = eod_mask
            eod_frac = (1.0 - eod_mask.mean()).item()
            _dbg("EOD_MASK_BUILT", f"eod_fraction={eod_frac:.4f}")

        return result


# ── ModelProvider（对应上游 model_provider + DDP_impl 分支） ─────────────────
class ModelProvider:
    """
    构造并包装模型，根据 --DDP-impl 选择并行实现。

    上游 DDP_impl 分支（a1d04b793 新增）差异：
      'local':  Megatron 手工梯度规约（需要 mpu 层，适合 tensor parallel）
      'torch':  torch.nn.parallel.DistributedDataParallel（标准 PyTorch DDP）

    Walpurgis: 不实现 Megatron tensor parallel（无 mpu 依赖），
    local 路径等价为 single-GPU 训练（无规约），
    torch 路径使用标准 DDP 包装。
    """

    @staticmethod
    def wrap(
        model: nn.Module,
        ddp_impl: str = "local",
        device_ids: Optional[list] = None,
    ) -> nn.Module:
        """
        根据 ddp_impl 包装模型。
        Walpurgis: 在非分布式环境下两种模式均退化为 no-op（直接返回模型）。
        """
        _dbg("MODEL_PROVIDER_WRAP", f"ddp_impl={ddp_impl}")

        try:
            import torch.distributed as dist
            is_dist = dist.is_initialized()
        except Exception:
            is_dist = False

        if not is_dist:
            _dbg("MODEL_PROVIDER_WRAP", "non-distributed → skip DDP wrapping")
            return model.to(device_ids[0] if device_ids else "cpu")

        if ddp_impl == "torch":
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=device_ids
            )
            _dbg("MODEL_PROVIDER_WRAP", "wrapped with torch.nn.parallel.DDP")
            return wrapped
        elif ddp_impl == "local":
            # Megatron local DDP: 单 GPU 时为 no-op，多 GPU 时需 mpu（此处 stub）
            _dbg("MODEL_PROVIDER_WRAP", "local DDP (stub: no-op in walpurgis single-GPU mode)")
            return model
        else:
            raise ValueError(f"未知 DDP-impl: {ddp_impl!r}，可选: ['local', 'torch']")


# ── TensorBoardHook（对应上游 pretrain 中 writer.add_scalar 散点） ─────────────
class TensorBoardHook:
    """
    TensorBoard 日志 hook，对应 --tensorboard-dir（a1d04b793）。

    上游的做法是在 train_step / train 里直接调用 writer.add_scalar，
    代码与日志强耦合。Walpurgis 将日志逻辑抽为 hook，可被替换或禁用。

    用法:
        hook = TensorBoardHook(log_dir="/tmp/tb_logs")
        hook.on_step(result)    # 每步调用
        hook.on_epoch(epoch, metrics)  # 每 epoch 调用
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._log_dir = log_dir
        self._writer = None
        _dbg("TB_HOOK_INIT", f"log_dir={log_dir}")

    def _get_writer(self):
        if self._writer is None and self._log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self._log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=self._log_dir)
                _dbg("TB_HOOK_WRITER_CREATED", self._log_dir)
            except ImportError:
                _dbg("TB_HOOK_WRITER_SKIP", "tensorboard not installed")
        return self._writer

    def on_step(self, result: TrainStepResult) -> None:
        """每步训练后回调：记录 loss/grad_norm。"""
        w = self._get_writer()
        if w is None:
            return
        w.add_scalar("train/loss", result.loss, result.iteration)
        w.add_scalar("train/grad_norm", result.grad_norm, result.iteration)
        _dbg("TB_STEP", f"iter={result.iteration} loss={result.loss:.6f}")

    def on_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """每 epoch 后回调：记录指标字典。"""
        w = self._get_writer()
        if w is None:
            return
        for k, v in metrics.items():
            w.add_scalar(f"epoch/{k}", v, epoch)
        _dbg("TB_EPOCH", f"epoch={epoch} metrics={metrics}")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            _dbg("TB_HOOK_CLOSED", self._log_dir)


# ── AutoResumeGuard（对应 --adlr-autoresume，a1d04b793 新增） ─────────────────
class AutoResumeGuard:
    """
    ADLR autoresume context manager。

    上游在 pretrain 主循环里有条件地轮询 autoresume signal file，
    Walpurgis 将其抽为 context manager，逻辑集中，环境隔离。

    使用方式:
        with AutoResumeGuard(enabled=args.adlr_autoresume,
                             interval=args.adlr_autoresume_interval,
                             iteration=iteration):
            ... train step ...

    非 ADLR 环境下（autoresume_signal_file 不存在）完全透明。
    """
    _SIGNAL_FILE = os.environ.get(
        "ADLR_AUTORESUME_SIGNAL_FILE",
        "/run/adlr_autoresume_signal"
    )

    def __init__(
        self,
        enabled: bool,
        interval: int,
        iteration: int,
        on_resume: Optional[Callable] = None,
    ) -> None:
        self.enabled = enabled
        self.interval = interval
        self.iteration = iteration
        self.on_resume = on_resume

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self.enabled:
            return
        if self.iteration % self.interval != 0:
            return
        if os.path.exists(self._SIGNAL_FILE):
            _dbg(
                "AUTORESUME_SIGNAL",
                f"signal file found at iter={self.iteration}: {self._SIGNAL_FILE}",
            )
            print(
                f"[AutoResume] signal detected at iteration {self.iteration}, "
                f"triggering resume callback.",
                file=sys.stderr,
            )
            if self.on_resume is not None:
                self.on_resume(self.iteration)
        else:
            _dbg("AUTORESUME_CHECK", f"iter={self.iteration} no signal")
