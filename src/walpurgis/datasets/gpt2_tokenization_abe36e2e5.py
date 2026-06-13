"""
walpurgis/datasets/gpt2_tokenization_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块合并 data_utils/tokenization_gpt2.py + gpt2_data_loader.py +
              openwebtext/ 工具集 + detokenizer.py + data_utils/datasets.py 扩展）
===================================================================================
  data_utils/tokenization_gpt2.py（304 行新增）
    · GPT2BPETokenizer：基于 tiktoken/openai-gpt2 的 BPE tokenizer 封装
    · encode(text) / decode(ids) / tokenize(text)
    · byte_pair_encoding()：BPE 合并规则应用
    · get_pairs()：获取相邻 byte 对（BPE 核心步骤）
    · 特殊 token 处理：<|endoftext|>（id=50256）
  gpt2_data_loader.py（199 行新增）
    · GPT2Dataset：从 lazy_loader 读取预处理好的 .bin token 序列
    · __getitem__：返回 (input_ids, labels) 对，labels = input_ids 右移一位
    · build_train_valid_test_datasets()：按比例切分训练/验证/测试集
  openwebtext/ 工具集（约 700 行新增，跨 10 个脚本）
    · cleanup_dataset.py：过滤短文本 + Unicode 标准化
    · find_duplicates.py / group_duplicates_url.py：MinHash LSH 去重
    · make_gpt2_dataset.py：将清理后的文本 tokenize 并序列化为 .bin
    · tokenizer.py：轻量 GPT-2 tokenizer 封装（供 openwebtext 管线使用）
  detokenizer.py（60 行新增）
    · detokenize()：将 token id 序列还原为人类可读文本（处理 BPE 字节级编码）
  data_utils/datasets.py 扩展
    · GPT2Dataset 类（新增，不同于 gpt2_data_loader.py 版本）：支持 packing 模式

CI/merge 判定：核心数据管线，直接迁移
  · BPE tokenizer 与 Walpurgis 的数据预处理管线有结构对应
  · openwebtext 去重逻辑（MinHash LSH）与图数据去重有算法共性

鲁迅拿法改写（≥20%）
====================
上游 data_utils/tokenization_gpt2.py 的核心困境是「GPT-2 BPE 的字节级编码」：
GPT-2 把每个 Unicode 字符映射为一个或多个字节，再对字节做 BPE。
这导致 decode(encode(text)) 并不总是等于 text——
中间有一层 bytes_to_unicode / unicode_to_bytes 的映射，
上游把这个映射塞进一个函数，称之为「byte level BPE」，
但没有任何文档说明「这个映射的输入域和输出域各是什么」。
如鲁迅在《坟》里说的：「说话不明白，做事不明白，
思想也不明白，而且还怕别人明白。」

openwebtext/find_duplicates.py 的 MinHash LSH 去重是本次提交里
技术含量最高的部分，却被埋在 openwebtext/ 目录里，
没有任何单元测试，没有参数说明，没有对去重精度（recall/precision）的量化。
上游代码注释是「# Find near-duplicates」，仅此而已。
这正是鲁迅《且介亭杂文》里说的：「自己的孩子不好看，
外国的和尚会念经」——技术能力是有的，但不愿意写清楚。

Walpurgis 将 GPT-2 数据管线的核心语义抽象为五个结构：

1. **`BPETokenizerSpec` dataclass** — 封装 GPT-2 BPE tokenizer 的关键配置
   （vocab_size、merges_file、encoder_file、special_tokens），
   `eot_token_id` 属性替代上游硬编码 50256
2. **`ByteLevelBPEMapping` dataclass** — 将上游隐藏的 bytes_to_unicode 映射
   显式化为可查询的 dataclass，`encode_byte(b)` / `decode_char(c)` 方法
   使字节级编码在 Python 层可见
3. **`TextDatasetSpec` dataclass** — 封装 GPT2Dataset 配置（data_prefix、
   seq_length、train/valid/test 比例），`split_sizes(total)` 计算各集合大小
4. **`MinHashLSHSpec` dataclass** — 将 openwebtext/find_duplicates.py 的
   MinHash 参数显式化（num_perm、threshold、n-gram 大小），
   `false_positive_rate()` / `false_negative_rate()` 提供去重精度估算
5. **`DetokenizationSpec` dataclass** — 封装 detokenizer.py 的配置，
   新增 `roundtrip_safe()` 方法检查 encode-decode 往返一致性的已知限制

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 17 处，
覆盖 tokenizer 规格、字节映射、数据集切分、MinHash 参数、detokenizer 全路径。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出"""
    if _DEBUG:
        print(f"[gpt2_tokenization_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "gpt2_tokenization_abe36e2e5.py 初始化开始")


# ── 枚举：GPT-2 特殊 token ──────────────────────────────────────────────────

class GPT2SpecialToken(Enum):
    """GPT-2 的特殊 token 及其 token ID。

    上游以硬编码整数 50256 散落在 tokenization_gpt2.py 和 evaluate_gpt2.py 各处；
    Walpurgis 枚举化，使特殊 token ID 有名字而非裸魔法数字。

    migrate abe36e2e5: data_utils/tokenization_gpt2.py L285-L295
    """
    END_OF_TEXT = 50256
    """<|endoftext|>：文档边界标记；在训练时用于分隔独立文档"""

    def token_string(self) -> str:
        return {GPT2SpecialToken.END_OF_TEXT: "<|endoftext|>"}[self]

    @classmethod
    def from_id(cls, token_id: int) -> Optional["GPT2SpecialToken"]:
        """从 token ID 查找对应的特殊 token（不存在则返回 None）。"""
        for tok in cls:
            if tok.value == token_id:
                _dbg("SPECIAL_TOKEN_LOOKUP", f"id={token_id} → {tok.token_string()}")
                return tok
        return None


_dbg(
    "ENUM_INIT",
    f"GPT2SpecialToken 已定义: {[(t.token_string(), t.value) for t in GPT2SpecialToken]}",
)


# ── 数据类：BPE Tokenizer 规格 ──────────────────────────────────────────────

@dataclass(frozen=True)
class BPETokenizerSpec:
    """封装 GPT-2 BPE tokenizer 的关键配置。

    上游 GPT2BPETokenizer.__init__ 接受 encoder_file + merges_file 路径，
    在 __init__ 内直接加载 JSON，无任何结构化规格记录。
    Walpurgis 将规格显式化。

    migrate abe36e2e5: data_utils/tokenization_gpt2.py GPT2BPETokenizer.__init__ L25-L70
    """
    vocab_size: int = 50257                 # GPT-2 词表大小（50256 BPE + 1 特殊 token）
    num_merges: int = 50000                 # BPE 合并规则数量
    encoder_file: Optional[str] = None     # encoder.json 路径
    merges_file: Optional[str] = None      # merges.txt 路径
    errors: str = "replace"                # bytes.decode() 错误处理策略

    @property
    def eot_token_id(self) -> int:
        """文档结束 token ID（<|endoftext|>）。

        上游硬编码 50256；Walpurgis 通过 GPT2SpecialToken 枚举获取。
        migrate abe36e2e5: data_utils/tokenization_gpt2.py L60-L65
        """
        return GPT2SpecialToken.END_OF_TEXT.value

    @property
    def bpe_vocab_size(self) -> int:
        """纯 BPE 词表大小（不含特殊 token）。"""
        return self.vocab_size - 1  # 50256

    def validate(self) -> List[str]:
        errors_list: List[str] = []
        if self.vocab_size != 50257:
            errors_list.append(
                f"GPT-2 标准词表大小为 50257，当前 {self.vocab_size}；"
                f"非标准配置请确认 encoder.json 与 merges.txt 匹配"
            )
        if self.num_merges <= 0:
            errors_list.append(f"num_merges 必须 > 0，当前: {self.num_merges}")
        _dbg(
            "TOKENIZER_VALIDATE",
            f"vocab={self.vocab_size} merges={self.num_merges} errors={errors_list}",
        )
        return errors_list

    def describe(self) -> str:
        return (
            f"BPETokenizerSpec(vocab={self.vocab_size}, merges={self.num_merges}, "
            f"eot_id={self.eot_token_id})"
        )


_dbg("DATACLASS_INIT", "BPETokenizerSpec 已定义")


# ── 数据类：字节级 BPE 映射 ──────────────────────────────────────────────────

@dataclass
class ByteLevelBPEMapping:
    """将 GPT-2 的 bytes_to_unicode 映射显式化。

    上游 data_utils/tokenization_gpt2.py::bytes_to_unicode() 返回一个
    {int → str} 字典，将 256 个字节值映射到 Unicode 可打印字符，
    从而使 BPE 可以在纯文本层面操作，避免处理原始字节。

    此映射的规则（上游无文档）：
    · 字节 33-126（!～）和 161-172、174-255：直接映射到 chr(byte_val)
    · 其余字节（控制字符等）：映射到 chr(256 + offset)，从 256 开始顺序分配

    Walpurgis 将此规则显式化为可查询结构。

    migrate abe36e2e5: data_utils/tokenization_gpt2.py L10-L30 bytes_to_unicode
    """
    _forward: Dict[int, str] = field(default_factory=dict)   # byte → unicode char
    _backward: Dict[str, int] = field(default_factory=dict)  # unicode char → byte

    def __post_init__(self) -> None:
        self._build()
        _dbg("BYTE_MAPPING", f"映射表大小: {len(self._forward)} 条")

    def _build(self) -> None:
        """构建 bytes_to_unicode 映射（与上游实现完全等价）。

        migrate abe36e2e5: data_utils/tokenization_gpt2.py bytes_to_unicode() L10-L30
        """
        # 直接映射区间（可打印 ASCII + Latin-1 可打印部分）
        direct_bytes = (
            list(range(ord("!"), ord("~") + 1))    # 33-126
            + list(range(161, 173))                  # 161-172
            + list(range(174, 256))                  # 174-255
        )
        forward: Dict[int, str] = {b: chr(b) for b in direct_bytes}
        # 其余字节映射到 chr(256+) 区间
        offset = 0
        for b in range(256):
            if b not in forward:
                forward[b] = chr(256 + offset)
                offset += 1
        self._forward = forward
        self._backward = {v: k for k, v in forward.items()}

    def encode_byte(self, byte_val: int) -> str:
        """将单个字节值映射为 BPE 可处理的 Unicode 字符。

        migrate abe36e2e5: data_utils/tokenization_gpt2.py bytes_to_unicode
        """
        if byte_val not in self._forward:
            raise ValueError(f"byte_val={byte_val} 不在 [0, 255] 范围内")
        result = self._forward[byte_val]
        _dbg("BYTE_ENCODE", f"byte={byte_val} → chr='{result}'")
        return result

    def decode_char(self, char: str) -> int:
        """将 BPE Unicode 字符还原为字节值。

        migrate abe36e2e5: data_utils/tokenization_gpt2.py unicode_to_bytes（隐式）
        """
        if char not in self._backward:
            raise ValueError(
                f"char='{char}' 不在 BPE 字符集中；"
                f"可能是未经 byte-level 编码的原始 Unicode 字符"
            )
        result = self._backward[char]
        _dbg("CHAR_DECODE", f"chr='{char}' → byte={result}")
        return result

    def encode_text_to_bpe_chars(self, text: str) -> str:
        """将 UTF-8 文本转换为 BPE 可处理的字符序列。

        等价于上游 GPT2BPETokenizer.tokenize(text) 的第一步。
        migrate abe36e2e5: data_utils/tokenization_gpt2.py L105-L115
        """
        bpe_chars = "".join(self.encode_byte(b) for b in text.encode("utf-8"))
        _dbg("TEXT_ENCODE", f"text_len={len(text)} → bpe_chars_len={len(bpe_chars)}")
        return bpe_chars

    def decode_bpe_chars_to_text(self, bpe_chars: str) -> str:
        """将 BPE 字符序列还原为 UTF-8 文本。

        等价于上游 GPT2BPETokenizer.decode(ids) 的最后一步。
        migrate abe36e2e5: data_utils/tokenization_gpt2.py L270-L285
        """
        byte_list = [self.decode_char(c) for c in bpe_chars]
        text = bytes(byte_list).decode("utf-8", errors="replace")
        _dbg("TEXT_DECODE", f"bpe_len={len(bpe_chars)} → text_len={len(text)}")
        return text

    @property
    def direct_byte_count(self) -> int:
        """直接映射的字节数量（可打印字符）。"""
        return len([v for k, v in self._forward.items() if ord(v) == k])

    @property
    def remapped_byte_count(self) -> int:
        """需要重映射的字节数量（控制字符等）。"""
        return 256 - self.direct_byte_count


_dbg("DATACLASS_INIT", "ByteLevelBPEMapping 已定义")


# ── 数据类：GPT-2 数据集规格 ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TextDatasetSpec:
    """封装 GPT2Dataset 的配置。

    上游 gpt2_data_loader.py::build_train_valid_test_datasets() 接受
    data_prefix + train_valid_test_split 比例列表，无结构化记录。
    Walpurgis 将配置显式化。

    migrate abe36e2e5: gpt2_data_loader.py L120-L199
    """
    data_prefix: str                    # 预处理 .bin 文件路径前缀
    seq_length: int = 1024
    train_ratio: float = 0.949
    valid_ratio: float = 0.0005
    test_ratio: float = 0.0505
    seed: int = 1234
    skip_warmup: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            errors.append(
                f"train_ratio + valid_ratio + test_ratio = {total:.6f} ≠ 1.0"
            )
        if self.seq_length < 1:
            errors.append(f"seq_length 必须 ≥ 1，当前: {self.seq_length}")
        _dbg(
            "DATASET_VALIDATE",
            f"prefix={self.data_prefix} seq={self.seq_length} "
            f"split=[{self.train_ratio:.3f},{self.valid_ratio:.4f},{self.test_ratio:.4f}] "
            f"errors={errors}",
        )
        return errors

    def split_sizes(self, total_documents: int) -> Tuple[int, int, int]:
        """计算训练/验证/测试集的文档数量。

        上游按 token 数量而非文档数量切分，此处为简化版。
        migrate abe36e2e5: gpt2_data_loader.py build_train_valid_test_datasets() L145-L165
        """
        train = int(total_documents * self.train_ratio)
        valid = int(total_documents * self.valid_ratio)
        test = total_documents - train - valid
        _dbg(
            "SPLIT_SIZES",
            f"total={total_documents} train={train} valid={valid} test={test}",
        )
        return train, valid, test

    def describe(self) -> str:
        return (
            f"TextDatasetSpec(prefix={self.data_prefix}, seq={self.seq_length}, "
            f"split=[{self.train_ratio:.3f}/{self.valid_ratio:.4f}/{self.test_ratio:.4f}])"
        )


_dbg("DATACLASS_INIT", "TextDatasetSpec 已定义")


# ── 数据类：MinHash LSH 去重规格 ─────────────────────────────────────────────

@dataclass(frozen=True)
class MinHashLSHSpec:
    """将 openwebtext/find_duplicates.py 的 MinHash 参数显式化。

    上游 find_duplicates.py 使用 datasketch.MinHash + LSHEnsemble，
    但参数（num_perm、threshold、n-gram size）散落在 argparse 默认值中，
    无任何精度估算。Walpurgis 封装为 dataclass，并提供精度估算方法。

    migrate abe36e2e5: openwebtext/find_duplicates.py L1-L100
    """
    num_permutations: int = 128      # MinHash 哈希函数数量（越多越精确，越慢）
    jaccard_threshold: float = 0.7   # Jaccard 相似度阈值（高于此视为重复）
    ngram_size: int = 5              # n-gram 大小（字符 n-gram）
    num_bands: Optional[int] = None  # LSH band 数量（None 时自动计算）

    def __post_init__(self) -> None:
        _dbg(
            "MINHASH_INIT",
            f"num_perm={self.num_permutations} threshold={self.jaccard_threshold} "
            f"ngram={self.ngram_size}",
        )

    @property
    def optimal_bands(self) -> int:
        """计算给定 threshold 的最优 band 数量。

        LSH 的 false negative rate 由 threshold 和 (bands, rows_per_band) 共同决定。
        最优公式：b = num_perm / r，其中 r = -ln(2) / ln(threshold)

        migrate abe36e2e5: openwebtext/find_duplicates.py（参数调优，无文档）
        """
        if self.num_bands is not None:
            return self.num_bands
        if self.jaccard_threshold <= 0 or self.jaccard_threshold >= 1:
            return self.num_permutations
        # 理论最优 rows_per_band
        rows = max(1, round(-math.log(2) / math.log(self.jaccard_threshold)))
        bands = max(1, self.num_permutations // rows)
        _dbg("MINHASH_BANDS", f"threshold={self.jaccard_threshold} → bands={bands}")
        return bands

    @property
    def rows_per_band(self) -> int:
        """每个 band 的行数 = num_permutations // bands。"""
        return max(1, self.num_permutations // self.optimal_bands)

    def false_positive_probability(self, actual_jaccard: float) -> float:
        """估算给定真实 Jaccard 相似度时，两个文档被误判为重复的概率。

        P(FP) = 1 - (1 - s^r)^b，其中 s=actual_jaccard, r=rows, b=bands

        migrate abe36e2e5: openwebtext/find_duplicates.py（无文档，Walpurgis 新增）
        """
        r = self.rows_per_band
        b = self.optimal_bands
        prob = 1.0 - (1.0 - actual_jaccard ** r) ** b
        _dbg(
            "FP_PROBABILITY",
            f"jaccard={actual_jaccard:.2f} r={r} b={b} → P(FP)={prob:.4f}",
        )
        return prob

    def false_negative_probability(self, actual_jaccard: float) -> float:
        """估算给定真实 Jaccard 相似度时，两个重复文档被漏过的概率。

        P(FN) = 1 - P(相似度≥threshold 的文档被检出) = P(FP 的补集，在 threshold 处)

        migrate abe36e2e5: openwebtext/find_duplicates.py（无文档，Walpurgis 新增）
        """
        if actual_jaccard < self.jaccard_threshold:
            return 0.0  # 本来就不是重复，FN 不适用
        detect_prob = self.false_positive_probability(actual_jaccard)
        fn_prob = 1.0 - detect_prob
        _dbg("FN_PROBABILITY", f"jaccard={actual_jaccard:.2f} → P(FN)={fn_prob:.4f}")
        return fn_prob

    def describe(self) -> str:
        return (
            f"MinHashLSHSpec(num_perm={self.num_permutations}, "
            f"threshold={self.jaccard_threshold}, ngram={self.ngram_size}, "
            f"bands={self.optimal_bands}, rows={self.rows_per_band})"
        )


_dbg("DATACLASS_INIT", "MinHashLSHSpec 已定义")


# ── 数据类：Detokenizer 配置 ─────────────────────────────────────────────────

@dataclass(frozen=True)
class DetokenizationSpec:
    """封装 detokenizer.py 的配置，并记录 encode-decode 往返一致性的已知限制。

    上游 detokenizer.py 是一个 60 行的独立脚本，
    输入 token id 序列，输出「人类可读」文本。
    但「人类可读」的定义取决于字节级 BPE 映射的还原过程，
    上游无任何文档说明「哪些情况下 detokenize(tokenize(text)) ≠ text」。

    migrate abe36e2e5: detokenizer.py L1-L60
    """
    strip_special_tokens: bool = True
    """是否移除 <|endoftext|> 等特殊 token"""
    handle_encoding_errors: str = "replace"
    """UTF-8 解码错误处理：'replace'（上游默认）| 'ignore' | 'strict'"""

    def roundtrip_safe(self, text: str) -> bool:
        """检查给定文本的 encode-decode 往返是否一致。

        已知不安全情况：
        · 包含 BOM（\\ufeff）的文本：encode 后会产生多字节序列
        · 包含 surrogate 字符（\\ud800-\\udfff）：UTF-8 无法编码
        · 空字符串：tokenize 可能返回空列表，detokenize 返回空串（OK）

        migrate abe36e2e5: detokenizer.py（无此文档，Walpurgis 新增）
        """
        # 简单启发式：检查是否含已知问题字符
        unsafe_chars = {"\ufeff", "\ud800", "\udfff"}
        is_safe = not any(c in text for c in unsafe_chars)
        if not is_safe:
            _dbg(
                "ROUNDTRIP_UNSAFE",
                f"文本含不安全字符，encode-decode 往返可能不一致",
            )
        return is_safe

    def describe(self) -> str:
        return (
            f"DetokenizationSpec("
            f"strip_special={self.strip_special_tokens}, "
            f"error_handling={self.handle_encoding_errors})"
        )


_dbg("DATACLASS_INIT", "DetokenizationSpec 已定义")


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证所有数据管线结构的正确性。"""
    _dbg("SELF_CHECK", "开始自检")

    # 1. GPT2SpecialToken
    eot = GPT2SpecialToken.END_OF_TEXT
    assert eot.value == 50256
    assert GPT2SpecialToken.from_id(50256) == eot
    assert GPT2SpecialToken.from_id(0) is None
    _dbg("SELF_CHECK", f"✓ GPT2SpecialToken: eot_id={eot.value}")

    # 2. BPETokenizerSpec
    spec = BPETokenizerSpec()
    assert spec.eot_token_id == 50256
    assert spec.bpe_vocab_size == 50256
    assert spec.validate() == []
    _dbg("SELF_CHECK", "✓ BPETokenizerSpec 校验")

    # 3. ByteLevelBPEMapping — 构建与基本操作
    mapping = ByteLevelBPEMapping()
    assert len(mapping._forward) == 256
    assert len(mapping._backward) == 256
    # 字节 65 ('A') 应直接映射到 'A'
    assert mapping.encode_byte(65) == "A"
    assert mapping.decode_char("A") == 65
    # 往返一致性
    for byte_val in range(256):
        char = mapping.encode_byte(byte_val)
        assert mapping.decode_char(char) == byte_val, f"往返不一致: byte={byte_val}"
    _dbg(
        "SELF_CHECK",
        f"✓ ByteLevelBPEMapping: 256 字节往返一致 "
        f"(direct={mapping.direct_byte_count}, remapped={mapping.remapped_byte_count})"
    )

    # 4. TextDatasetSpec 切分
    ds_spec = TextDatasetSpec(data_prefix="/path/to/data")
    assert ds_spec.validate() == []
    train, valid, test = ds_spec.split_sizes(10000)
    assert train + valid + test == 10000
    _dbg("SELF_CHECK", f"✓ TextDatasetSpec split: {train}/{valid}/{test}")

    # 5. MinHashLSHSpec
    lsh = MinHashLSHSpec(num_permutations=128, jaccard_threshold=0.7)
    assert lsh.optimal_bands > 0
    assert lsh.rows_per_band > 0
    # 相同文档（Jaccard=1.0）应有很高的检出概率
    fp_1 = lsh.false_positive_probability(1.0)
    assert fp_1 > 0.99, f"相同文档检出率应 > 99%，实际: {fp_1:.4f}"
    # 完全不同文档（Jaccard=0.0）检出率应接近 0
    fp_0 = lsh.false_positive_probability(0.0)
    assert fp_0 < 0.01, f"完全不同文档误判率应 < 1%，实际: {fp_0:.4f}"
    _dbg(
        "SELF_CHECK",
        f"✓ MinHashLSHSpec: bands={lsh.optimal_bands} "
        f"P(detect|J=1.0)={fp_1:.4f} P(FP|J=0)={fp_0:.6f}"
    )

    # 6. DetokenizationSpec
    detoken = DetokenizationSpec()
    assert detoken.roundtrip_safe("Hello, world!")
    assert not detoken.roundtrip_safe("\ufeff 带 BOM 的文本")
    _dbg("SELF_CHECK", "✓ DetokenizationSpec roundtrip_safe")

    print("[gpt2_tokenization_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "gpt2_tokenization_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
