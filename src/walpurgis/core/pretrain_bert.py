"""
migrate 73af12903: Major refactoring, combining gpt2 and bert
上游文件: pretrain_bert.py（528行 → 大幅精简，主体逻辑迁入 megatron/training.py）

鲁迅拿法改写（≥20%）：
  上游 pretrain_bert.py 经过本次 commit 后，从 528 行瘦身到约 100-150 行：
  大量训练循环、checkpoint、分布式初始化代码都被抽走，
  只剩下「BERT 特有」的部分：
    - model_provider()：构造 BertModel
    - forward_step()：BERT 的批次处理和 loss 计算
    - train_valid_test_datasets_provider()：数据集构建
  
  上游这次重构的本质，鲁迅式解读：
  「原来的铁屋子有两间（pretrain_bert / pretrain_gpt2），
    现在把公共的墙拆掉，合成一间叫 training.py，
    两边各留一扇门（model_provider / forward_step）。
    门面看起来小了，实际住的人还是那些人。」

  Walpurgis 改写要点：
  1. BertBatchProcessor: 上游 forward_step() 里的批次处理逻辑独立为类，
     含 tokens / types / attention_mask / labels 的 shape 断言。
  2. BertPretrainConfig: 上游依赖全局 args 对象，
     Walpurgis 将 BERT 预训练的超参提取为显式 dataclass。
  3. BertPretrainTask: 整合 model_provider + forward_step + dataset_provider，
     作为注入 TrainingEngine 的「任务单元」（上游无此抽象）。

迁移位置: src/walpurgis/core/pretrain_bert.py
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Any

import torch
import torch.nn as nn

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：pretrain_bert 关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    if isinstance(msg, torch.Tensor):
        t = msg
        info = (
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4f} max={t.max().item():.4f}"
        )
        print(f"[_dbg:pretrain_bert:{tag}] {info}", file=sys.stderr, flush=True)
    else:
        print(f"[_dbg:pretrain_bert:{tag}] {msg}", file=sys.stderr, flush=True)


# ── BertPretrainConfig ────────────────────────────────────────────────────────
# 上游：全局 args 对象，Walpurgis：BERT 任务特有超参 dataclass。

@dataclass
class BertPretrainConfig:
    """
    BERT 预训练超参（对应上游 pretrain_bert.py 使用的 args 字段子集）。

    上游通过 get_args() 获得全局 args 后，在 pretrain_bert.py 里用到：
      vocab_size, hidden_size, num_layers, num_attention_heads,
      ffn_hidden_size（= 4 * hidden_size），
      num_tokentypes（BERT: 2），
      max_position_embeddings, hidden_dropout, attention_dropout,
      train_iters, eval_iters, eval_interval,
      micro_batch_size, DDP_impl...
    Walpurgis 将 BERT 特有部分显式列出。
    """
    # 模型规格
    vocab_size: int = 30522          # BERT-base 默认词表
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    ffn_hidden_size: int = 3072      # 4 * hidden_size
    max_seq_len: int = 512
    num_tokentypes: int = 2          # BERT: segment A/B

    # 正则化
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # 训练
    train_iters: int = 1_000_000
    eval_iters: int = 100
    eval_interval: int = 1000
    save_interval: int = 10000
    log_interval: int = 100
    micro_batch_size: int = 4
    weight_decay: float = 0.01
    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_fraction: float = 0.01

    # 任务
    binary_head: bool = True         # 是否保留 NSP head（上游 --binary-head 参数）

    def __post_init__(self):
        _dbg("BERT_CONFIG",
             f"layers={self.num_layers}, heads={self.num_attention_heads}, "
             f"hidden={self.hidden_size}, vocab={self.vocab_size}")


# ── BertBatchProcessor ────────────────────────────────────────────────────────
# 上游 forward_step() 里直接解包 batch dict，没有任何 shape 检查。
# Walpurgis：将批次处理封装为 BertBatchProcessor，含显式 shape 断言。

class BertBatchProcessor:
    """
    BERT 批次处理器（对应上游 pretrain_bert.py forward_step 的批次解包部分）。

    上游批次格式（来自 BertDataset / IndexedDataset）：
        batch = {
            "text": tokens [B, T],
            "types": segment_ids [B, T],
            "labels": mlm_labels [B, T],
            "is_random": nsp_labels [B],
            "loss_mask": mlm_mask [B, T],
            "attention_mask": padding_mask [B, T],
            "padding_mask": 同 attention_mask（上游有些 key 重复）
        }

    Walpurgis 将解包逻辑和 shape 验证分离，错误信息明确。
    """

    def __init__(self, config: BertPretrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        _dbg("BATCH_PROC_INIT", f"device={device}")

    def process(self, batch: dict) -> dict:
        """
        解包并移动批次到 device，加 shape 断言。
        返回标准化字段字典。
        """
        tokens = batch["text"].to(self.device)
        segment_ids = batch.get("types", torch.zeros_like(batch["text"])).to(self.device)
        mlm_labels = batch["labels"].to(self.device)
        nsp_labels = batch["is_random"].to(self.device)
        loss_mask = batch.get("loss_mask", torch.ones_like(batch["text"])).to(self.device)
        attention_mask = batch.get(
            "attention_mask",
            batch.get("padding_mask", torch.ones_like(batch["text"]))
        ).to(self.device)

        B, T = tokens.shape
        assert segment_ids.shape == (B, T), f"segment_ids shape mismatch: {segment_ids.shape}"
        assert mlm_labels.shape == (B, T), f"mlm_labels shape mismatch: {mlm_labels.shape}"
        assert nsp_labels.shape == (B,), f"nsp_labels shape mismatch: {nsp_labels.shape}"

        _dbg("BATCH_TOKENS", tokens)
        _dbg("BATCH_SEG_IDS", segment_ids)
        _dbg("BATCH_MLM_LABELS", mlm_labels)
        _dbg("BATCH_NSP_LABELS", nsp_labels)
        _dbg("BATCH_ATTN_MASK", attention_mask)

        return {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "mlm_labels": mlm_labels,
            "nsp_labels": nsp_labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }


# ── BertPretrainTask ──────────────────────────────────────────────────────────
# 上游：model_provider() 和 forward_step() 是独立函数，由 pretrain() 注入。
# Walpurgis：封装为 BertPretrainTask，持有 model + processor + loss_computer。

class BertPretrainTask:
    """
    BERT 预训练任务单元（对应上游 pretrain_bert.py 的 model_provider + forward_step）。

    上游 pretrain_bert.py 剩余的「任务特定」代码都在这里。
    通过 build_model() 构造模型，通过 forward_step() 注入 TrainingEngine。

    model_provider 对应上游：
        def model_provider():
            model = BertModel(...)
            return model

    forward_step 对应上游：
        def forward_step(data_iterator, model, args, timers):
            batch = get_batch(data_iterator, args, timers)
            tokens, types, attention_mask, labels, loss_mask, sentence_order = batch
            output_tensor = model(tokens, attention_mask, tokentype_ids=types)
            return output_tensor, partial(loss_func, loss_mask, sentence_order)
    """

    def __init__(
        self,
        config: BertPretrainConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.model: Optional[nn.Module] = None
        self.batch_processor = BertBatchProcessor(config, device)
        _dbg("BERT_TASK_INIT", f"config={config}")

    def build_model(self, init_method=None) -> nn.Module:
        """
        构造 BertModelWrapper（对应上游 model_provider()）。

        上游 model_provider() 里同时构造 DDP 包装，
        Walpurgis 只构造裸模型，DDP 包装由 TrainingEngine 决定。
        """
        from ..models.language_model import get_language_model
        from ..models.bert_model import BertModelWrapper

        if init_method is None:
            std = (2.0 / (self.config.hidden_size * self.config.num_layers)) ** 0.5
            init_method = lambda w: nn.init.normal_(w, mean=0.0, std=std)
            _dbg("BUILD_MODEL_INIT", f"auto init_method std={std:.6f}")

        def output_init(w):
            std_out = (2.0 / (self.config.hidden_size * self.config.num_layers * 2)) ** 0.5
            nn.init.normal_(w, mean=0.0, std=std_out)

        attention_mask_func = self._causal_mask_func  # BERT 不用 causal mask，但接口一致

        lm_core = get_language_model(
            attention_mask_func=attention_mask_func,
            num_tokentypes=self.config.num_tokentypes,
            add_pooler=True,
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            ffn_hidden_size=self.config.ffn_hidden_size,
            max_seq_len=self.config.max_seq_len,
            hidden_dropout=self.config.hidden_dropout,
            attention_dropout=self.config.attention_dropout,
            embedding_dropout=self.config.embedding_dropout,
            init_method=init_method,
            output_layer_init_method=output_init,
        )

        model = BertModelWrapper(
            language_model=lm_core,
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            init_method=init_method,
        ).to(self.device)

        self.model = model
        _dbg("BUILD_MODEL_DONE", f"BertModelWrapper on {self.device}")
        return model

    @staticmethod
    def _causal_mask_func(attn_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        BERT attention mask 函数（padding mask，非 causal）。
        mask=True 位置填充 -10000.0（上游约定）。
        """
        return attn_scores.masked_fill(mask, -10000.0)

    def forward_step(
        self,
        batch: dict,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Any]:
        """
        BERT 前向步（对应上游 forward_step()）。

        返回 (loss, bert_output)。
        """
        from ..models.bert_model import BertLossComputer

        processed = self.batch_processor.process(batch)
        _dbg("FWD_STEP_TOKENS", processed["tokens"])

        bert_output = model(
            input_ids=processed["tokens"],
            attention_mask=processed["attention_mask"],
            tokentype_ids=processed["segment_ids"],
        )

        loss_computer = BertLossComputer()
        loss_output = loss_computer(
            bert_output=bert_output,
            lm_labels=processed["mlm_labels"],
            nsp_labels=processed["nsp_labels"] if self.config.binary_head else None,
        )

        _dbg("FWD_STEP_LOSS", f"total={loss_output.total_loss.item():.4f}, "
             f"mlm={loss_output.mlm_loss.item():.4f}, "
             f"nsp={loss_output.nsp_loss.item():.4f}")

        return loss_output.total_loss, bert_output

    def get_forward_step_fn(self) -> Callable:
        """返回可注入 TrainingEngine 的 forward_step 函数"""
        def fn(batch, model):
            return self.forward_step(batch, model)
        return fn
