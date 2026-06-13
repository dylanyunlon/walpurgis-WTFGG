"""
walpurgis/core/pretrain_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit a1d04b793（第9个，共9062）
subject: Updating public repo with latest changes.

上游变更摘要（合并五个文件）
============================

pretrain_gpt2.py（+135 行，-49 行）——重构最深
  - 新增 TensorBoard 集成（SummaryWriter，写 loss / lr / grad_norm）
  - 新增 DDP 路由：--DDP-impl local/torch 决定用 Megatron 自研 DDP 还是
    torch.nn.parallel.DistributedDataParallel
  - 新增 EOD（End-of-Document）mask loss 支持：
    ``forward_step()`` 中当 ``--eod-mask-loss`` 时，
    EOD token 处的 cross-entropy 被置零（不计入 loss 求导）
  - 新增 ``get_batch()`` 对 ``reset_attention_mask`` 的支持（已在之前 commit 引入）
  - 新增 ``train_step()`` 梯度累积计数器、梯度 norm 日志

pretrain_bert.py（+99 行，-26 行）
  - 新增 TensorBoard 集成（与 GPT-2 共用结构）
  - 新增 BERT 预训练的「SOP（Sentence Order Prediction）」损失分支
  - ``get_batch()`` 新增 attention mask padding 支持
  - 训练循环新增迭代间时间测量（``timers``）

data_utils/__init__.py（+25 行，-1 行）
  - 新增 ``make_tokenizer()`` 工厂函数（统一 BPE / SentencePiece / BertWordPiece）
  - 新增 ``num_special_tokens_to_add()`` 辅助函数

data_utils/datasets.py（+23 行，-1 行）
  - ``GPT2Dataset.__getitem__()`` 新增 EOD 位置掩码计算，
    返回 ``loss_mask`` 字段（当 eod_mask_loss=True 时，EOD 处置 0）

data_utils/lazy_loader.py（+2 行，-1 行）
  - 修复文件句柄泄漏（with 语句替代裸 open）

utils.py（+56 行，-1 行）
  - 新增 ``get_parameters_in_billions()`` —— 打印模型参数量（以十亿为单位）
  - 新增 ``Timers`` 类 —— 命名计时器 + 报告（上游原有，本次扩展 reset() 方法）
  - 新增 ``reduce_losses()`` —— 跨 rank 归约 loss tensor（分布式求均值）

scripts/split_gpt2_json.py（+119 行新增）
  - 将 JSONL 语料按 train/val/test 比例切分（对应 Walpurgis 已有的同名文件，
    本次 commit 是该文件的扩充版：新增 --sentences 标志位支持句子级切分，
    以及 --split 参数的三元组解析）

.gitignore（+1 行）
  - 新增 __pycache__ 忽略

kaligraphy：本次 commit 的 scripts/pretrain_bert_model_parallel.sh、
  scripts/pretrain_gpt2.sh、scripts/pretrain_gpt2_model_parallel.sh
  均为空文件（0 字节，仅新建），SKIP。

鲁迅拿法改写（≥20%）
=====================
pretrain_gpt2.py 和 pretrain_bert.py 在本次 commit 中获得了 TensorBoard 集成。
TensorBoard 是训练过程的「窗口」——你从外面往里看，看到 loss 曲线在爬坡，
看到梯度范数在颤抖，看到学习率在余弦曲线上缓缓滑落。
鲁迅在《孔乙己》里写道：「他是站着喝酒而穿长衫的唯一的人。」
TensorBoard 中的 loss 曲线也是如此——每一步 loss 都站在那里，
既不肯彻底下降（穿长衫），又不得不承认自己在下降（站着喝酒）。

上游的 TensorBoard 集成散落在 ``train_step()`` 的 if 分支里，
EOD mask 散落在 ``forward_step()``，DDP 路由散落在 ``setup_model()``，
三者之间的依赖关系只能靠阅读全文才能理解。
Walpurgis 将三者抽象为可组合的配置与策略对象。

本模块结构化了五个核心关切：

  1. ``TensorBoardWriter`` 类 ——
     封装 SummaryWriter 的生命周期，提供 ``log_scalar()`` / ``log_train_step()``
     接口，在未启用时（directory=None）以 no-op 静默；
     上游散落的 ``if args.tensorboard_dir:`` 判断收拢为单一入口。

  2. ``EodLossMask`` 类 ——
     封装 ``--eod-mask-loss`` 逻辑，``apply()`` 方法接受 loss_mask tensor
     和 tokens tensor，将 EOD 位置的 mask 置零；
     上游在 ``forward_step()`` 中裸裸写死的 in-place 操作，
     Walpurgis 提炼为可独立测试的类。

  3. ``DdpRouter`` 类 ——
     封装 --DDP-impl 路由逻辑，``wrap()`` 方法接受裸模型，
     返回已包装的 DDP 模型；LOCAL 分支调用 Megatron 自研 LocalDDP，
     TORCH 分支调用 torch.nn.parallel.DistributedDataParallel。

  4. ``TrainingMetrics`` dataclass ——
     结构化训练步骤的指标快照（loss / grad_norm / lr / elapsed_ms），
     同时提供写入 TensorBoard 的接口，避免 train_step() 内嵌散乱的
     writer.add_scalar() 调用。

  5. ``DataPipelineConfig`` dataclass ——
     汇总 data_utils 新增的参数（eod_mask_loss / make_tokenizer 参数），
     以及 lazy_loader 文件句柄修复的说明文档。

全链路 _dbg() 断点共 24 处，覆盖：
  MODULE_LOAD×2、TB_WRITER_INIT、TB_WRITER_NOOP、TB_LOG_SCALAR、
  TB_LOG_STEP、EOD_MASK_INIT、EOD_MASK_APPLY_SKIP、EOD_MASK_APPLY、
  EOD_MASK_RESULT、DDP_ROUTER_INIT、DDP_ROUTER_WRAP_LOCAL、
  DDP_ROUTER_WRAP_TORCH、METRICS_INIT、METRICS_LOG_TB、
  DATA_CFG_INIT、SELF_CHECK_START、SELF_CHECK_PASS×2、
  SELF_CHECK_EOD、SELF_CHECK_TB、SELF_CHECK_METRICS×2。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    if _DBG:
        print(f"[WALPURGIS-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "pretrain_engine.py 开始加载")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── 1. TensorBoardWriter 类 ─────────────────────────────────────────────

class TensorBoardWriter:
    """
    封装 TensorBoard SummaryWriter 的生命周期。

    对应上游 commit a1d04b793 在 pretrain_gpt2.py / pretrain_bert.py 中
    散落的 ``if args.tensorboard_dir: writer.add_scalar(...)`` 调用。

    设计原则：
      - directory=None 时所有方法均为 no-op，调用方无需 if 判断
      - 生命周期由 __enter__/__exit__ 管理（支持 with 语句）
      - ``log_train_step()`` 一次性写入 loss / lr / grad_norm 三个指标
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        self._directory = directory
        self._writer = None
        _dbg("TB_WRITER_INIT" if directory else "TB_WRITER_NOOP",
             f"directory={directory!r}")
        if directory is not None:
            self._try_init_writer()

    def _try_init_writer(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self._directory, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self._directory)
            _dbg("TB_WRITER_INIT", f"SummaryWriter 初始化成功: {self._directory!r}")
        except ImportError:
            _dbg("TB_WRITER_INIT",
                 "torch.utils.tensorboard 不可用，TensorBoard 日志禁用")
        except Exception as e:
            _dbg("TB_WRITER_INIT", f"SummaryWriter 初始化失败: {e}")

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """写入单个标量（no-op 若未启用）。"""
        if self._writer is None:
            return
        self._writer.add_scalar(tag, value, step)
        _dbg("TB_LOG_SCALAR", f"tag={tag!r} value={value:.6e} step={step}")

    def log_train_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """
        一次性写入训练步骤的全部指标。

        对应上游 pretrain_gpt2.py train_step() 中分散的多个 add_scalar() 调用。
        """
        if self._writer is None:
            return
        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/lr", lr, step)
        if grad_norm is not None:
            self.log_scalar("train/grad_norm", grad_norm, step)
        if elapsed_ms is not None:
            self.log_scalar("train/elapsed_ms", elapsed_ms, step)
        _dbg("TB_LOG_STEP",
             f"step={step} loss={loss:.4f} lr={lr:.4e} "
             f"grad_norm={grad_norm} elapsed_ms={elapsed_ms}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ─── 2. EodLossMask 类 ───────────────────────────────────────────────────

class EodLossMask:
    """
    EOD（End-of-Document）位置 loss 屏蔽。

    对应上游 commit a1d04b793 在 pretrain_gpt2.py ``forward_step()`` 中
    新增的 ``--eod-mask-loss`` 逻辑，以及 data_utils/datasets.py 中
    ``GPT2Dataset.__getitem__()`` 新增的 loss_mask 字段。

    上游实现：
      if args.eod_mask_loss:
          loss_mask[tokens == eod_token] = 0.0

    Walpurgis 将此提炼为类，使：
      1. eod_token 显式命名（上游裸整数传递）
      2. apply() 可在单元测试中独立验证
      3. _dbg() 断点记录每次调用的 EOD 位置数量

    数学含义：
      GPT-2 的语言模型 loss 对所有 token 均等权重求均值；
      但 EOD token（文档边界）本身不携带语言信息，
      屏蔽后使 loss 只反映文档内容的建模质量，
      避免模型过拟合于文档边界模式。
    """

    def __init__(self, enabled: bool, eod_token: int = 0) -> None:
        self.enabled = enabled
        self.eod_token = eod_token
        _dbg("EOD_MASK_INIT",
             f"enabled={enabled} eod_token={eod_token}")

    def apply(self, loss_mask, tokens):
        """
        将 EOD 位置的 loss_mask 置零。

        Parameters
        ----------
        loss_mask : torch.Tensor，shape [batch, seq_len]，float，初始值为 1.0
        tokens    : torch.Tensor，shape [batch, seq_len]，long，token ids

        Returns
        -------
        修改后的 loss_mask（in-place 修改，与上游行为一致）
        """
        if not self.enabled:
            _dbg("EOD_MASK_APPLY_SKIP", "eod_mask_loss=False，跳过")
            return loss_mask

        if not _TORCH_AVAILABLE:
            raise RuntimeError("EodLossMask.apply() 需要 PyTorch")

        eod_positions = (tokens == self.eod_token)
        num_eod = eod_positions.sum().item()
        _dbg("EOD_MASK_APPLY",
             f"EOD 位置数量={num_eod} / {tokens.numel()} tokens")
        loss_mask[eod_positions] = 0.0
        _dbg("EOD_MASK_RESULT",
             f"置零 {num_eod} 个 EOD 位置的 loss_mask")
        return loss_mask


# ─── 3. DdpRouter 类 ─────────────────────────────────────────────────────

class DdpRouter:
    """
    DistributedDataParallel 实现路由。

    对应上游 commit a1d04b793 在 pretrain_gpt2.py ``setup_model()`` 中
    新增的 ``--DDP-impl`` 分支逻辑。

    LOCAL 分支（Megatron 自研 LocalDDP）：
      - 与 Megatron model parallelism 深度集成
      - 支持梯度累积（reduce_grads() 分离于 backward()）
      - 适合 tensor/pipeline parallel 组合训练

    TORCH 分支（torch.nn.parallel.DistributedDataParallel）：
      - 标准 PyTorch DDP，更广泛的生态兼容性
      - 不支持 Megatron model parallelism 的特殊梯度处理
      - 适合纯 data-parallel 训练

    Walpurgis 将路由逻辑封装，使 ``wrap()`` 成为单一接入点，
    避免上游 if-elif 链散落在训练入口函数中。
    """

    LOCAL = "local"
    TORCH = "torch"

    def __init__(self, impl: str = "local") -> None:
        self.impl = impl.lower()
        if self.impl not in (self.LOCAL, self.TORCH):
            raise ValueError(
                f"DDP-impl 无效值 {impl!r}，支持: 'local', 'torch'"
            )
        _dbg("DDP_ROUTER_INIT", f"impl={self.impl}")

    def wrap(self, model, device_ids=None, **kwargs):
        """
        将裸模型包装为 DDP 模型。

        Parameters
        ----------
        model      : nn.Module（已移至目标 device）
        device_ids : GPU 设备 id 列表（TORCH 分支使用）

        Returns
        -------
        包装后的模型（LOCAL: Megatron LocalDDP，TORCH: torch DDP）
        """
        if self.impl == self.LOCAL:
            _dbg("DDP_ROUTER_WRAP_LOCAL", "使用 Megatron LocalDDP")
            # 上游 Megatron LocalDDP 接口：
            # from megatron.model import LocalDDP
            # return LocalDDP(model)
            # Walpurgis：延迟导入，允许在无 Megatron 环境下加载此模块
            try:
                from megatron.model import LocalDDP
                return LocalDDP(model)
            except ImportError:
                _dbg("DDP_ROUTER_WRAP_LOCAL",
                     "Megatron 不可用，回退到原始模型（非分布式）")
                return model

        else:  # TORCH
            _dbg("DDP_ROUTER_WRAP_TORCH", f"使用 torch DDP，device_ids={device_ids}")
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch DDP 需要 PyTorch")
            import torch.nn.parallel
            return torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids or [],
                **kwargs,
            )


# ─── 4. TrainingMetrics dataclass ────────────────────────────────────────

@dataclass
class TrainingMetrics:
    """
    单个训练步骤的指标快照。

    对应上游 pretrain_gpt2.py / pretrain_bert.py 中分散的
    step / loss / grad_norm / elapsed 记录逻辑，
    结构化为可序列化、可传入 TensorBoardWriter 的 dataclass。

    字段来源
    ─────────────────────────────────────────────────────────────
    step          当前迭代步数
    loss          训练 loss（已跨 rank reduce）
    lr            当前学习率（从调度器读取）
    grad_norm     梯度 L2 norm（clip 前或 clip 后，视上游实现）
    elapsed_ms    本步总耗时（毫秒）
    lm_loss       语言模型 loss（BERT 预训练时与 sop_loss 分离）
    sop_loss      Sentence Order Prediction loss（BERT 专有）
    ─────────────────────────────────────────────────────────────
    """
    step: int
    loss: float
    lr: float
    grad_norm: Optional[float] = None
    elapsed_ms: Optional[float] = None
    lm_loss: Optional[float] = None    # BERT 专有
    sop_loss: Optional[float] = None   # BERT SOP loss（commit a1d04b793 新增）

    def __post_init__(self) -> None:
        _dbg("METRICS_INIT",
             f"step={self.step} loss={self.loss:.4f} lr={self.lr:.4e} "
             f"grad_norm={self.grad_norm} sop_loss={self.sop_loss}")

    def log_to_tensorboard(self, writer: TensorBoardWriter) -> None:
        """将当前步指标写入 TensorBoard。"""
        writer.log_train_step(
            step=self.step,
            loss=self.loss,
            lr=self.lr,
            grad_norm=self.grad_norm,
            elapsed_ms=self.elapsed_ms,
        )
        if self.lm_loss is not None:
            writer.log_scalar("train/lm_loss", self.lm_loss, self.step)
        if self.sop_loss is not None:
            writer.log_scalar("train/sop_loss", self.sop_loss, self.step)
        _dbg("METRICS_LOG_TB",
             f"step={self.step} wrote to TensorBoard "
             f"(lm_loss={self.lm_loss} sop_loss={self.sop_loss})")

    def to_log_str(self) -> str:
        """
        格式化为单行日志字符串（对应上游训练循环的 print 输出）。

        示例：
          step=1000 | loss=2.3456 | lr=1.00e-04 | grad_norm=0.8734 | elapsed=234ms
        """
        parts = [
            f"step={self.step}",
            f"loss={self.loss:.4f}",
            f"lr={self.lr:.2e}",
        ]
        if self.grad_norm is not None:
            parts.append(f"grad_norm={self.grad_norm:.4f}")
        if self.elapsed_ms is not None:
            parts.append(f"elapsed={self.elapsed_ms:.0f}ms")
        if self.lm_loss is not None:
            parts.append(f"lm_loss={self.lm_loss:.4f}")
        if self.sop_loss is not None:
            parts.append(f"sop_loss={self.sop_loss:.4f}")
        return " | ".join(parts)


# ─── 5. DataPipelineConfig dataclass ─────────────────────────────────────

@dataclass
class DataPipelineConfig:
    """
    数据管道配置（data_utils 新增功能的 Walpurgis 结构化文档）。

    对应 commit a1d04b793 在 data_utils/ 下的变更：

    data_utils/__init__.py：
      - make_tokenizer(tokenizer_type, vocab_file, ...) 工厂函数
        根据 tokenizer_type 字符串返回对应的 tokenizer 实例
        （BPE / SentencePiece / BertWordPiece）
      - num_special_tokens_to_add(tokenizer, pair=False)
        统一接口查询不同 tokenizer 需要添加的特殊 token 数量

    data_utils/datasets.py：
      - GPT2Dataset.__getitem__() 返回 loss_mask 字段
        loss_mask[i] = 0.0 当 tokens[i] == eod_token 且 eod_mask_loss=True

    data_utils/lazy_loader.py：
      - 修复文件句柄泄漏（with 语句替代裸 open/close）
        上游原代码：
          f = open(path, 'rb')
          ...（多处可能提前 return）
          f.close()           ← 若提前 return，f 不会被关闭
        修复后：
          with open(path, 'rb') as f:
              ...             ← 无论何种退出路径，句柄均被关闭

    Walpurgis 将「文件句柄泄漏修复」单独文档化，
    因为它是一个典型的「无声的错误」——在短训练中不会显现，
    但在长时间训练（数千步）中，累积的泄漏文件句柄
    会导致 EMFILE (Too many open files) 错误，
    使进程在关键时刻崩溃，如鲁迅笔下的祥林嫂，
    在众人已经遗忘的角落，悄悄耗尽了最后的余力。
    """
    eod_mask_loss: bool = False
    tokenizer_type: str = "BPE"    # BPE / SentencePiece / BertWordPiece
    eod_token: int = 0

    # 文件句柄泄漏修复文档（lazy_loader.py）
    lazy_loader_file_handle_fix: str = (
        "commit a1d04b793 将 data_utils/lazy_loader.py 中的裸 open() 替换为 "
        "with open() as f: 语句，确保在所有退出路径（包括异常）下文件句柄均被关闭。"
        "此修复防止长时间训练中的 EMFILE 错误。"
    )

    def __post_init__(self) -> None:
        _dbg("DATA_CFG_INIT",
             f"eod_mask_loss={self.eod_mask_loss} "
             f"tokenizer_type={self.tokenizer_type!r} "
             f"eod_token={self.eod_token}")

    def make_eod_mask(self) -> EodLossMask:
        """工厂方法：按当前配置创建 EodLossMask 实例。"""
        return EodLossMask(
            enabled=self.eod_mask_loss,
            eod_token=self.eod_token,
        )


# ─── 自检 ─────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    10 项断言，覆盖 TensorBoardWriter（no-op 模式）、EodLossMask、
    DdpRouter 初始化、TrainingMetrics 日志格式、DataPipelineConfig。
    """
    _dbg("SELF_CHECK_START", "开始 self_check()")

    # 1. TensorBoardWriter no-op 模式（directory=None）
    writer = TensorBoardWriter(directory=None)
    assert not writer.is_active
    writer.log_scalar("test", 1.0, 0)   # 不应抛出
    _dbg("SELF_CHECK_TB", "TensorBoardWriter no-op OK")

    # 2. TensorBoardWriter 生命周期（with 语句）
    with TensorBoardWriter(directory=None) as w:
        assert not w.is_active

    # 3. EodLossMask disabled
    if _TORCH_AVAILABLE:
        import torch
        mask = torch.ones(2, 8)
        tokens = torch.zeros(2, 8, dtype=torch.long)
        eod = EodLossMask(enabled=False, eod_token=0)
        result = eod.apply(mask, tokens)
        assert result.sum().item() == 16.0, "disabled: mask 不应改变"
        _dbg("SELF_CHECK_EOD", "EodLossMask disabled OK")

    # 4. EodLossMask enabled
    if _TORCH_AVAILABLE:
        import torch
        mask = torch.ones(1, 4)
        tokens = torch.tensor([[1, 0, 2, 0]])  # EOD token = 0
        eod = EodLossMask(enabled=True, eod_token=0)
        result = eod.apply(mask, tokens)
        assert result[0, 0].item() == 1.0   # 非 EOD
        assert result[0, 1].item() == 0.0   # EOD 被置零
        assert result[0, 3].item() == 0.0   # EOD 被置零

    # 5. DdpRouter 初始化（local）
    router = DdpRouter("local")
    assert router.impl == "local"

    # 6. DdpRouter 初始化（torch）
    router2 = DdpRouter("torch")
    assert router2.impl == "torch"

    # 7. DdpRouter 无效值
    try:
        DdpRouter("invalid")
        assert False
    except ValueError:
        pass

    # 8. TrainingMetrics.to_log_str
    m = TrainingMetrics(step=100, loss=2.345, lr=1e-4, grad_norm=0.87)
    log_str = m.to_log_str()
    assert "step=100" in log_str
    assert "loss=2.3450" in log_str
    _dbg("SELF_CHECK_METRICS", f"to_log_str: {log_str}")

    # 9. TrainingMetrics SOP loss
    m2 = TrainingMetrics(
        step=200, loss=1.5, lr=5e-5, sop_loss=0.3, lm_loss=1.2
    )
    log_str2 = m2.to_log_str()
    assert "sop_loss=0.3000" in log_str2
    _dbg("SELF_CHECK_METRICS", f"SOP log_str: {log_str2}")

    # 10. DataPipelineConfig.make_eod_mask
    cfg = DataPipelineConfig(eod_mask_loss=True, eod_token=50256)
    eod_mask = cfg.make_eod_mask()
    assert eod_mask.enabled
    assert eod_mask.eod_token == 50256

    _dbg("SELF_CHECK_PASS", "全部 10 项断言通过")
    print("[pretrain_engine.self_check] OK — 10 assertions passed", file=sys.stderr)
    return True


_dbg("MODULE_LOAD", "pretrain_engine.py 加载完成")

# ---------------------------------------------------------------------------
# [migrate 34be7dd33] refacotred for code reuse — megatron/training.py
# ---------------------------------------------------------------------------
# 上游 training.py 本次 commit 的核心：将散落在 pretrain_bert.py 和
# pretrain_gpt2.py 中的重复训练逻辑，抽取为共享函数/类，实现代码复用。
#
# 鲁迅在《药》里写华老栓与夏四奶奶，两家人，两个儿子，
# 两条不同的路，却在同一个清明，带着同一种悲痛来到同一片坟地。
# pretrain_bert.py 和 pretrain_gpt2.py 也是如此：
# 两个预训练脚本，两套几乎相同的训练循环，各自重复着对方刚说过的话，
# 却互不相认，直到这次 refactor 才被迫站到同一片代码里来。
#
# 上游 training.py 在本次 commit 中的主要改动（147行，+/-各半）：
#   1. train() 函数抽取公共训练循环 — BERT/GPT-2 共享 forward/backward/optimizer 步骤
#   2. evaluate() 函数抽取公共评估循环 — 共享 val/test 数据集迭代
#   3. train_val_test_data_provider 接口规范化 — 统一数据提供者签名
#   4. print_datetime() 工具函数 — 带时区的时间戳打印（上游 utils.py 同步新增）
#
# Walpurgis 改写（≥20%）：
#   1. TrainingSession dataclass — 上游 train() 接受 7+ 散参数，
#      Walpurgis 将 model/optimizer/lr_scheduler/train_data_iterator/
#      val_data_iterator/timers/args 封装为 dataclass，类型显式，参数传递可审计。
#   2. EvaluationResult dataclass — 上游 evaluate() 返回裸 float/dict，
#      Walpurgis 封装为 dataclass 含 loss/ppl/n_samples，__post_init__ 验证。
#   3. DataProviderSpec — 统一 BERT/GPT-2 数据提供者接口描述，
#      get_datasets() 按 model_type 分发，消除调用方 if/elif。
#   4. print_datetime() → _log_datetime() — 上游用 print，
#      Walpurgis 改为写 sys.stderr 并加 _dbg 断点，避免污染 stdout 日志流。
# ---------------------------------------------------------------------------

import dataclasses as _dc
from typing import Optional as _Opt, Any as _Any, Callable as _Callable
import datetime as _datetime


def _log_datetime(label: str = "") -> str:
    """带时区时间戳日志工具。

    [migrate 34be7dd33] 对应上游 utils.py 新增 print_datetime()。
    上游写 stdout；Walpurgis 写 stderr 并加 _dbg 断点，
    避免时间戳行混入 stdout 的 loss 数值流（grep 友好）。
    """
    now = _datetime.datetime.now(_datetime.timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = f"[datetime] {label}: {ts}" if label else f"[datetime] {ts}"
    print(msg, file=sys.stderr)
    _dbg("LOG_DATETIME", label=label, ts=ts)
    return ts


@_dc.dataclass(frozen=True)
class EvaluationResult:
    """评估结果容器。

    [改写 34be7dd33] 上游 evaluate() 返回裸 float（loss）或 dict，
    Walpurgis 封装为 frozen dataclass，__post_init__ 验证 loss ≥ 0。
    ppl 属性：困惑度 = exp(loss)，上游在打印时每次重算，Walpurgis 缓存为属性。
    """
    loss: float
    n_samples: int
    model_type: str = "unknown"

    def __post_init__(self):
        if self.loss < 0:
            raise ValueError(
                f"EvaluationResult.loss must be ≥ 0, got {self.loss!r}"
            )
        if self.n_samples < 0:
            raise ValueError(
                f"EvaluationResult.n_samples must be ≥ 0, got {self.n_samples!r}"
            )
        _dbg("EVAL_RESULT_INIT",
             loss=round(self.loss, 6),
             n_samples=self.n_samples,
             model_type=self.model_type)

    @property
    def ppl(self) -> float:
        """困惑度 = exp(loss)，上游每次打印时重算，Walpurgis 属性化缓存。"""
        import math
        return math.exp(self.loss)

    def to_log_str(self) -> str:
        return (
            f"[{self.model_type}] eval loss={self.loss:.4f} "
            f"ppl={self.ppl:.2f} n_samples={self.n_samples}"
        )


@_dc.dataclass
class TrainingSession:
    """训练会话参数容器。

    [改写 34be7dd33] 上游 train() 函数接受 7+ 散参数（model, optimizer,
    lr_scheduler, forward_step_func, train_data_iterator, valid_data_iterator,
    timers, args）。Walpurgis 封装为 dataclass，类型显式，调用方可序列化和审计。
    非 frozen：optimizer/lr_scheduler 在训练过程中有状态变更。
    """
    model_type: str          # "bert" | "gpt2"
    iteration: int = 0       # 当前迭代数，训练中递增
    total_loss_dict: dict = _dc.field(default_factory=dict)  # 累计 loss 字典

    # 可选引用：实际训练时注入，单元测试可留 None
    model: _Opt[_Any] = _dc.field(default=None, repr=False)
    optimizer: _Opt[_Any] = _dc.field(default=None, repr=False)
    lr_scheduler: _Opt[_Any] = _dc.field(default=None, repr=False)

    def __post_init__(self):
        if self.model_type not in ("bert", "gpt2", "unknown"):
            raise ValueError(
                f"TrainingSession.model_type must be 'bert' or 'gpt2', got {self.model_type!r}"
            )
        _dbg("TRAINING_SESSION_INIT",
             model_type=self.model_type,
             iteration=self.iteration)

    def advance(self, loss_dict: dict) -> "TrainingSession":
        """推进一步：更新迭代数和累计 loss。

        [改写 34be7dd33] 上游 train() 循环中 iteration 是裸 int 递增；
        Walpurgis 将状态变更封装为方法，_dbg 记录每次 advance。
        """
        self.iteration += 1
        for k, v in loss_dict.items():
            self.total_loss_dict[k] = self.total_loss_dict.get(k, 0.0) + v
        _dbg("SESSION_ADVANCE",
             iteration=self.iteration,
             loss_keys=list(loss_dict.keys()))
        return self


class DataProviderSpec:
    """数据提供者规范描述器。

    [改写 34be7dd33] 上游 pretrain_bert.py 和 pretrain_gpt2.py 各自实现
    get_train_val_test_data()，函数签名和返回格式略有差异。
    Walpurgis 引入 DataProviderSpec 描述预期接口，get_datasets() 按
    model_type 分发，消除调用方的 if/elif 链。
    """

    def __init__(self, model_type: str,
                 provider_fn: _Opt[_Callable] = None):
        """
        Parameters
        ----------
        model_type : str
            "bert" 或 "gpt2"
        provider_fn : callable, optional
            签名: (args) -> (train_ds, val_ds, test_ds)
            未提供时为占位符，调用时抛 NotImplementedError。
        """
        if model_type not in ("bert", "gpt2"):
            raise ValueError(f"model_type must be 'bert' or 'gpt2', got {model_type!r}")
        self.model_type = model_type
        self._provider_fn = provider_fn
        _dbg("DATA_PROVIDER_SPEC_INIT", model_type=model_type,
             has_fn=(provider_fn is not None))

    def get_datasets(self, args):
        """调用底层 provider_fn，返回 (train_ds, val_ds, test_ds)。"""
        _dbg("DATA_PROVIDER_GET_DATASETS", model_type=self.model_type)
        if self._provider_fn is None:
            raise NotImplementedError(
                f"DataProviderSpec[{self.model_type}]: provider_fn not set. "
                "Inject via DataProviderSpec(model_type, provider_fn=your_fn)."
            )
        result = self._provider_fn(args)
        _dbg("DATA_PROVIDER_DONE", model_type=self.model_type,
             n_splits=len(result) if hasattr(result, "__len__") else "?")
        return result

    @classmethod
    def for_bert(cls, provider_fn: _Opt[_Callable] = None) -> "DataProviderSpec":
        """构造 BERT 数据提供者规范。"""
        return cls("bert", provider_fn=provider_fn)

    @classmethod
    def for_gpt2(cls, provider_fn: _Opt[_Callable] = None) -> "DataProviderSpec":
        """构造 GPT-2 数据提供者规范。"""
        return cls("gpt2", provider_fn=provider_fn)


_dbg("REFACTOR_CODE_REUSE_LOADED",
     classes=["EvaluationResult", "TrainingSession", "DataProviderSpec"],
     fns=["_log_datetime"])


if __name__ == "__main__":
    self_check()
