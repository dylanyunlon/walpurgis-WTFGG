"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: utils.py

鲁迅拿法改写（≥20%）：
  上游本次在 utils.py 新增了两件事：
  其一，get_checkpoint_name() 加了 iteration 参数，checkpoint 命名方式从"一刀切"
  变成"按步数分文件夹"，但变更前后的命名格式兼容性只字未提；
  其二，print_rank_0() / report_memory() / get_parameters_in_millions()
  这几个工具函数被加进来，散落在函数堆里，像铁屋里的几盏油灯，
  各照各的，不成系统。
  鲁迅说过："零散的材料，不比成系统的著述。"
  Walpurgis 将本次新增的工具函数按"用途"重新归类，
  并补全了 checkpoint 命名迁移文档（新旧格式对照表）、
  rank 感知日志工具（RankLogger）、内存诊断工具（MemoryReporter）。

迁移位置: src/walpurgis/utils/megatron_utils.py
"""

import os
import sys
import socket
import torch
from typing import Optional, Any

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：megatron utils 关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    print(f"[_dbg:megatron_utils:{tag}] {msg}", file=sys.stderr, flush=True)


# ── Checkpoint 命名（含迁移注释） ─────────────────────────────────────────────
#
# 上游 a1d04b793 之前: checkpoints/<name>/model_optim_rng.pt
# 上游 a1d04b793 之后: checkpoints/<name>/iter_<iter>/model_optim_rng.pt
#
# Walpurgis: 兼容两种格式（读取时自动探测，写入时统一用新格式）
#
_CKPT_FILENAME = "model_optim_rng.pt"
_CKPT_LATEST = "latest_checkpointed_iteration.txt"


def get_checkpoint_name(
    checkpoints_path: str,
    iteration: Optional[int] = None,
    release: bool = False,
) -> str:
    """
    返回 checkpoint 文件路径。

    格式（对应 a1d04b793 新格式）:
      checkpoints_path / iter_{iteration:07d} / model_optim_rng.pt
    兼容旧格式（无 iter 子目录）自动探测，供 load_checkpoint 使用。

    Walpurgis: 与上游相比，此函数补全了新旧格式说明，
    并在 iteration=None 时读取 latest_checkpointed_iteration.txt。
    """
    if iteration is None and not release:
        latest_file = os.path.join(checkpoints_path, _CKPT_LATEST)
        if os.path.exists(latest_file):
            with open(latest_file) as f:
                iteration = int(f.read().strip())
            _dbg("CKPT_LATEST", f"loaded latest iteration={iteration}")
        else:
            _dbg("CKPT_LATEST", f"no {_CKPT_LATEST} found, returning base path")
            return os.path.join(checkpoints_path, _CKPT_FILENAME)

    if release:
        d = os.path.join(checkpoints_path, "release")
    else:
        d = os.path.join(checkpoints_path, f"iter_{iteration:07d}")

    path = os.path.join(d, _CKPT_FILENAME)
    _dbg("CKPT_NAME", path)
    return path


def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """返回记录最新 iteration 编号的 tracker 文件路径。"""
    return os.path.join(checkpoints_path, _CKPT_LATEST)


def save_checkpoint_tracker(checkpoints_path: str, iteration: int) -> None:
    """写入最新 iteration 编号到 tracker 文件。"""
    tracker = get_checkpoint_tracker_filename(checkpoints_path)
    os.makedirs(checkpoints_path, exist_ok=True)
    with open(tracker, "w") as f:
        f.write(str(iteration))
    _dbg("CKPT_TRACKER_SAVED", f"iteration={iteration} -> {tracker}")


# ── RankLogger（Walpurgis特有：对应上游 print_rank_0 + 分布式感知） ───────────
class RankLogger:
    """
    Rank 感知的日志工具。

    上游 print_rank_0() 是一行 if dist.get_rank() == 0: print(...)，
    Walpurgis 将其包装为类，支持：
    - rank=0 或 all_ranks 模式
    - 可选前缀和时间戳
    - _dbg 集成
    """

    def __init__(
        self,
        prefix: str = "",
        all_ranks: bool = False,
    ) -> None:
        self.prefix = prefix
        self.all_ranks = all_ranks
        self._rank = self._get_rank()
        _dbg("RANK_LOGGER_INIT", f"rank={self._rank} all_ranks={all_ranks}")

    @staticmethod
    def _get_rank() -> int:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def log(self, msg: str, file=sys.stdout) -> None:
        """打印消息；all_ranks=False 时只有 rank 0 打印。"""
        rank = self._get_rank()
        if self.all_ranks or rank == 0:
            prefix = f"[{self.prefix}]" if self.prefix else ""
            host = socket.gethostname()
            full = f"{prefix}[rank{rank}@{host}] {msg}" if self.all_ranks else f"{prefix} {msg}"
            print(full, file=file, flush=True)


# 向后兼容上游接口
def print_rank_0(message: str) -> None:
    """上游接口兼容函数：只在 rank 0 打印。"""
    _logger = RankLogger()
    _logger.log(message)
    _dbg("PRINT_RANK0", message[:80] if len(message) > 80 else message)


# ── MemoryReporter（对应上游 report_memory()） ────────────────────────────────
class MemoryReporter:
    """
    GPU 内存使用报告工具。

    上游 report_memory(name) 是一行 torch.cuda.memory_allocated()，
    Walpurgis 扩展为结构化报告：allocated/reserved/peak，含 MB 换算。
    """

    @staticmethod
    def report(tag: str = "", device: Optional[torch.device] = None) -> dict:
        """
        打印当前 GPU 内存状态，返回统计字典。
        tag: 标注当前是什么阶段（如 "after_forward"）
        """
        if not torch.cuda.is_available():
            _dbg("MEM_REPORT", "CUDA not available, skip")
            return {}

        dev = device or torch.device("cuda")
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        msg = (
            f"[MemReport:{tag}] "
            f"allocated={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB"
        )
        print(msg, file=sys.stderr, flush=True)
        _dbg("MEM_REPORT", msg)

        return {"allocated_mb": alloc, "reserved_mb": reserved, "peak_mb": peak}

    @staticmethod
    def reset_peak() -> None:
        """重置 peak memory 统计（在每个 epoch/eval 前调用）。"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _dbg("MEM_PEAK_RESET", "done")


def report_memory(name: str) -> None:
    """上游接口兼容函数。"""
    MemoryReporter.report(tag=name)


# ── 参数量统计（对应上游 get_parameters_in_millions()） ─────────────────────
def get_parameters_in_millions(model: torch.nn.Module) -> float:
    """
    返回模型参数量（单位: 百万）。
    Walpurgis: 区分 trainable / frozen，分别报告。
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    trainable_m = trainable / 1e6
    frozen_m = frozen / 1e6
    total_m = total / 1e6

    _dbg(
        "PARAM_COUNT",
        f"total={total_m:.2f}M trainable={trainable_m:.2f}M frozen={frozen_m:.2f}M",
    )
    print(
        f"[ParamCount] total={total_m:.2f}M "
        f"(trainable={trainable_m:.2f}M frozen={frozen_m:.2f}M)",
        file=sys.stderr,
    )
    return total_m


# ── TensorBoard writer（对应 --tensorboard-dir 新增参数） ─────────────────────
class TensorBoardWriter:
    """
    TensorBoard 写入器包装，对应 --tensorboard-dir（a1d04b793 新增）。

    Walpurgis: lazy 初始化（首次 log 时才创建 SummaryWriter），
    不依赖 tensorboard 时静默降级为 no-op。
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._log_dir = log_dir
        self._writer = None
        _dbg("TB_WRITER_INIT", f"log_dir={log_dir}")

    def _get_writer(self):
        if self._writer is None and self._log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self._log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=self._log_dir)
                _dbg("TB_WRITER_CREATED", self._log_dir)
            except ImportError:
                _dbg("TB_WRITER_SKIP", "tensorboard not installed, logging disabled")
        return self._writer

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        w = self._get_writer()
        if w is not None:
            w.add_scalar(tag, value, step)
            _dbg("TB_SCALAR", f"{tag}={value:.6f} step={step}")

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        w = self._get_writer()
        if w is not None:
            w.add_scalars(main_tag, tag_scalar_dict, step)
            _dbg("TB_SCALARS", f"{main_tag} keys={list(tag_scalar_dict.keys())} step={step}")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            _dbg("TB_WRITER_CLOSED", self._log_dir)
            self._writer = None
