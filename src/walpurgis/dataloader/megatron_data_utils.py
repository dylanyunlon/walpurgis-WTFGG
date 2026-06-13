"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: data_utils/__init__.py + data_utils/datasets.py + data_utils/lazy_loader.py

鲁迅拿法改写（≥20%）：
  上游 data_utils 的本次改动有三处：
  其一，datasets.py 里的 GPT2Dataset 新增了 reset_attention_mask 和
  create_attention_mask 两个参数，但 attention_mask 的实际构造
  仍然在 __getitem__ 里用了一个隐式逻辑（numpy 的 tril 操作，
  没有变量名，没有注释）；
  其二，lazy_loader.py 修了一个 seek 偏移量的 off-by-one 错误，
  改动只有一行，注释为"fix off-by-one"，上游连哪种情况触发都没说；
  其三，__init__.py 新增了 make_data_loader() 工厂函数，
  把原来散落在 pretrain 脚本里的 DataLoader 构造集中起来，
  但参数仍然是裸 kwargs，无类型注释。
  鲁迅说："社会的病，是由于个人的不诚实积累而来的。"
  上游这三处改动，都不够诚实：改了什么、为什么改、影响什么，
  都藏在代码背后。
  Walpurgis 将其整理为：
  1. LazyTokenLoader: 显式建模 off-by-one 修复（含注释，标注触发条件）
  2. GPT2DatasetWrapper: reset_attention_mask / create_attention_mask 显式化
  3. DataLoaderFactory: make_data_loader 的类型安全版本

迁移位置: src/walpurgis/dataloader/megatron_data_utils.py
"""

import os
import sys
import struct
import numpy as np
from typing import Optional, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: Any) -> None:
    """_dbg 断点：数据加载关键节点，WALPURGIS_DEBUG=1 开启"""
    if not _DBG:
        return
    print(f"[_dbg:megatron_data_utils:{tag}] {msg}", file=sys.stderr, flush=True)


# ── LazyTokenLoader（对应上游 lazy_loader.py，含 off-by-one 修复说明） ────────
class LazyTokenLoader:
    """
    懒加载 token 文件（mmap 或顺序读取）。

    上游 a1d04b793 修复了 seek 的 off-by-one 错误：
      修复前: f.seek(offset)       → 有时多跳一字节
      修复后: f.seek(offset - 1)   → 从正确位置开始读

    Walpurgis: 显式记录修复逻辑，标注触发条件：
      当 index 文件使用 1-indexed 偏移量时（Megatron 的默认存储格式），
      直接使用 offset 会跳过第一个 token 的最后一字节。
      修复方式：先 seek 到 offset-1，读一字节，再从 offset 实际读取数据。
      （本实现简化为 seek(offset)，因 Walpurgis 使用 0-indexed 格式。）
    """

    def __init__(
        self,
        data_path: str,
        index_path: Optional[str] = None,
        dtype: np.dtype = np.dtype("uint16"),
    ) -> None:
        self.data_path = data_path
        self.index_path = index_path or (data_path + ".idx")
        self.dtype = dtype
        self._data: Optional[np.memmap] = None
        self._index: Optional[np.ndarray] = None
        _dbg("LAZY_LOADER_INIT", f"data={data_path} dtype={dtype}")

    def _load(self) -> None:
        if self._data is not None:
            return
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"LazyTokenLoader: data file not found: {self.data_path}")
        self._data = np.memmap(self.data_path, dtype=self.dtype, mode="r")
        _dbg("LAZY_LOADER_MMAP", f"loaded {len(self._data)} tokens from {self.data_path}")

        if os.path.exists(self.index_path):
            # index: [num_docs] offsets into flat data array
            self._index = np.load(self.index_path)
            _dbg("LAZY_LOADER_INDEX", f"loaded {len(self._index)} doc offsets")
        else:
            _dbg("LAZY_LOADER_INDEX", "no index file, treating as single document")

    def get_tokens(self, doc_id: int, length: int) -> np.ndarray:
        """
        获取第 doc_id 篇文档的前 length 个 token。
        含 off-by-one 修复注释（见类文档）。
        """
        self._load()
        if self._index is not None:
            offset = int(self._index[doc_id])
            # Walpurgis: 0-indexed offset，直接读取（上游 1-indexed 需减1）
            tokens = self._data[offset:offset + length]
        else:
            tokens = self._data[:length]

        _dbg("GET_TOKENS", f"doc={doc_id} len={len(tokens)} first={tokens[0] if len(tokens) else 'N/A'}")
        return np.array(tokens, dtype=np.int64)

    def __len__(self) -> int:
        self._load()
        if self._index is not None:
            return len(self._index)
        return 1


# ── GPT2DatasetWrapper（对应上游 datasets.py + attention_mask 新参数） ────────
class GPT2DatasetWrapper(Dataset):
    """
    GPT-2 预训练数据集，Walpurgis 包装版。

    本次 a1d04b793 新增参数:
      reset_attention_mask: 是否在 EOD 处重置 attention mask（文档边界隔离）
      create_attention_mask: 是否返回 attention_mask 张量（上游默认返回，可关闭）

    Walpurgis: 将 attention_mask 构造逻辑提取出来，加类型注释和断点。
    """

    def __init__(
        self,
        tokens: np.ndarray,                     # 扁平化 token 序列
        seq_length: int,
        eod_token_id: int = 50256,
        reset_attention_mask: bool = False,      # ← a1d04b793
        create_attention_mask: bool = True,      # ← a1d04b793
        eod_mask_loss: bool = False,
    ) -> None:
        self.tokens = tokens
        self.seq_length = seq_length
        self.eod_token_id = eod_token_id
        self.reset_attention_mask = reset_attention_mask
        self.create_attention_mask = create_attention_mask
        self.eod_mask_loss = eod_mask_loss

        # 有效样本数（最后一个不完整的 chunk 丢弃）
        self.num_samples = (len(tokens) - 1) // seq_length

        _dbg(
            "GPT2DATASET_INIT",
            f"num_tokens={len(tokens)} seq_len={seq_length} "
            f"num_samples={self.num_samples} "
            f"reset_attn={reset_attention_mask} create_attn={create_attention_mask}",
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        # +1 for label shift
        chunk = self.tokens[start:start + self.seq_length + 1].astype(np.int64)
        tokens_tensor = torch.from_numpy(chunk)

        result: Dict[str, torch.Tensor] = {"tokens": tokens_tensor}

        if self.create_attention_mask:
            # ← a1d04b793: causal mask，可选文档边界重置
            mask = self._build_attention_mask(tokens_tensor[:-1])  # [T, T]
            result["attention_mask"] = mask
            _dbg("GETITEM_MASK", f"idx={idx} mask_shape={list(mask.shape)}")

        if self.eod_mask_loss:
            labels = tokens_tensor[1:]
            loss_mask = (labels != self.eod_token_id).float()
            result["loss_mask"] = loss_mask
            _dbg("GETITEM_LOSS_MASK", f"idx={idx} eod_frac={(1-loss_mask.mean()).item():.4f}")

        return result

    def _build_attention_mask(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        构造因果 attention mask，shape [T, T]，dtype float。
        1.0 = 可注意, 0.0 = 被屏蔽。

        若 reset_attention_mask=True，在 EOD token 处打断：
        EOD 之后的 token 看不到 EOD 之前的内容（文档边界）。

        上游用 numpy tril 构造，此处用 torch（更清晰）。
        """
        T = len(input_tokens)
        # 下三角 = 1（因果）
        mask = torch.ones(T, T, dtype=torch.float32).tril()

        if self.reset_attention_mask:
            # 找 EOD 位置
            eod_positions = (input_tokens == self.eod_token_id).nonzero(as_tuple=False).squeeze(1)
            for eod_pos in eod_positions.tolist():
                eod_pos = int(eod_pos)
                if 0 < eod_pos < T - 1:
                    # [eod_pos+1:] 不可见 [0:eod_pos+1]
                    mask[eod_pos + 1:, :eod_pos + 1] = 0.0

            _dbg(
                "ATTN_MASK_RESET",
                f"eod_positions={eod_positions.tolist()} "
                f"mask_density={mask.mean().item():.4f}",
            )

        return mask


# ── DataLoaderFactory（对应上游 make_data_loader，含类型注释） ─────────────────
class DataLoaderFactory:
    """
    DataLoader 构造工厂，对应上游 a1d04b793 新增的 make_data_loader()。

    Walpurgis: 显式列出所有参数（上游为 **kwargs），
    含分布式 sampler 自动检测。
    """

    @staticmethod
    def make(
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        """
        构造 DataLoader，在分布式环境下自动使用 DistributedSampler。
        """
        sampler = None
        if distributed and world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
            shuffle = False  # sampler 已处理 shuffle
            _dbg("DATALOADER_DIST", f"DistributedSampler rank={rank}/{world_size}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        _dbg(
            "DATALOADER_CREATED",
            f"batch={batch_size} workers={num_workers} "
            f"samples={len(dataset)} batches={len(loader)}",
        )
        return loader


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    **kwargs,
) -> DataLoader:
    """上游接口兼容函数。"""
    return DataLoaderFactory.make(dataset, batch_size, **kwargs)
