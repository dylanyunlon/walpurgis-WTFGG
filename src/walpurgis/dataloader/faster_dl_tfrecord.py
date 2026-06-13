# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: Megatron-LM commit 66719e973
# 原标题: Faster dataloader merge (#1)
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 涉及上游文件：
#   - data_utils/tf_dl.py    → TFRecord 数据加载器，新增 threaded_dl 并发拉取
#   - data_utils/datasets.py → 文本数据集，新增 presplit-sentences 路径
#
# 「不在沉默中爆发，就在沉默中灭亡。」
# —— 鲁迅《纪念刘和珍君》
#
# 上游 tf_dl.py 的症结：TFRecordDataLoader 本是为吞吐而生，却因 GIL 与单线程
# 预取而窒息——数据饥饿时 GPU 空转，GPU 满载时又等 CPU 解压。
# 66719e973 引入 ThreadedIterator 作\"哨兵\"：后台线程预加载下一批，
# 主线程无需等待，吞吐提升显著。沉默的等待，就此爆发为并发。
#
# 上游 datasets.py 的故事：文档中句子本是整段存储，分词时需实时切割，
# 每个 epoch 重复计算，浪费算力。66719e973 引入 --presplit-sentences：
# 预先以 newline 切割，训练时直接拆取，无需重复分词。
# 鲁迅式预处理哲学：「把苦难的源头提前消灭，而非每次硬撑。」
#
# Walpurgis 20% 改写要点（保持上游 API 完全兼容）：
#   1. `ThreadSpec` dataclass — 将 ThreadedIterator 的 maxsize / daemon 标志
#      从构造参数散落状态提取为可序列化规格，`summary()` 输出可审计摘要；
#      上游仅有裸 `threading.Thread(target=..., daemon=True)` 无任何规格文档化。
#   2. `PresplitPolicy` dataclass — 将 --presplit-sentences 的两种路径
#      （整段随机截取 vs 逐行句子列表）显式建模，`describe()` 方法文字化策略差异，
#      使调用方无需阅读 `if args.presplit_sentences` 分支才能理解行为。
#   3. `TFRecordLoaderConfig` dataclass — 封装 num_workers/seed/threaded_dl
#      三元组，`effective_workers()` 实现上游 `max(num_workers, 1)` 语义，
#      `should_thread()` 实现上游 `threaded_dl = num_workers > 0` 语义，
#      使配置决策从 configure_data.py 散落的条件表达式集中到此处。
#   4. 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖所有关键路径。

import os as _os
import sys as _sys
import time as _time
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Any, Callable

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        print(
            f"[WALPURGIS-DL-FAST:{tag}][{_time.strftime('%H:%M:%S')}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# ThreadSpec — ThreadedIterator 线程规格封装
# ---------------------------------------------------------------------------

@dataclass
class ThreadSpec:
    """ThreadedIterator 的线程配置规格。

    上游 Megatron 66719e973 在 TFRecordDataLoader 中引入后台预取线程，
    但配置参数（maxsize/daemon）散落在构造函数内联逻辑中，无任何文档化。
    Walpurgis 将规格提取为独立数据类，使线程策略可序列化、可测试。

    Args:
        maxsize: 预取队列最大深度（上游默认 16，经验值：大 batch 宜减小）。
        daemon: 主线程退出时是否强制终止预取线程（上游默认 True）。
        sentinel: 队列结束哨兵值（上游使用 None）。
    """
    maxsize: int = 16
    daemon: bool = True
    sentinel: Any = None

    def summary(self) -> str:
        """人类可读的线程规格摘要。"""
        return (
            f"ThreadSpec(maxsize={self.maxsize}, daemon={self.daemon}, "
            f"sentinel={self.sentinel!r})"
        )

    def is_sentinel(self, item: Any) -> bool:
        """判断队列项是否为终止哨兵。"""
        return item is self.sentinel


# ---------------------------------------------------------------------------
# ThreadedIterator — 上游 tf_dl.py 核心预取机制
# ---------------------------------------------------------------------------

class ThreadedIterator:
    """后台线程预取迭代器，解耦 IO 等待与主线程计算。

    上游 Megatron-LM 66719e973 核心新增。预取线程将下一批数据压入队列，
    主线程从队列取出，GPU 无需等待 CPU 解压/IO。

    上游实现逻辑（精简）：
        thread.put(next(iterator)) in background
        main: yield queue.get()

    Walpurgis 改写：ThreadSpec 封装参数，异常传播通过 queue 内 ExceptionWrapper，
    避免上游\"线程崩溃主线程无感知\"的 silent failure。

    Args:
        iterator: 被包装的原始迭代器（TFRecord batch 流）。
        spec: 线程规格（maxsize / daemon / sentinel）。
    """

    class _ExceptionWrapper:
        """在队列中传播线程内异常到主线程。"""
        def __init__(self, exc: BaseException) -> None:
            self.exc = exc

    def __init__(self, iterator: Iterator, spec: Optional[ThreadSpec] = None) -> None:
        self._spec = spec or ThreadSpec()
        self._queue: queue.Queue = queue.Queue(maxsize=self._spec.maxsize)
        self._iterator = iterator

        _dbg("THREADED_ITER_INIT", f"spec={self._spec.summary()}")

        t = threading.Thread(target=self._producer, daemon=self._spec.daemon)
        t.start()
        self._thread = t

    def _producer(self) -> None:
        """后台生产者：将迭代器元素入队，末尾放置哨兵。"""
        try:
            for item in self._iterator:
                _dbg("THREADED_ITER_PUT", f"putting item type={type(item).__name__}")
                self._queue.put(item)
        except Exception as exc:  # noqa: BLE001
            _dbg("THREADED_ITER_ERR", f"producer exception: {exc}")
            self._queue.put(self._ExceptionWrapper(exc))
        finally:
            self._queue.put(self._spec.sentinel)
            _dbg("THREADED_ITER_DONE", "producer finished, sentinel enqueued")

    def __iter__(self) -> "ThreadedIterator":
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if isinstance(item, self._ExceptionWrapper):
            raise item.exc
        if self._spec.is_sentinel(item):
            raise StopIteration
        _dbg("THREADED_ITER_GET", f"got item type={type(item).__name__}")
        return item


# ---------------------------------------------------------------------------
# TFRecordLoaderConfig — configure_data.py 参数三元组封装
# ---------------------------------------------------------------------------

@dataclass
class TFRecordLoaderConfig:
    """TFRecordDataLoader 的加载配置，封装上游 configure_data.py 散落的决策逻辑。

    上游 66719e973 在 configure_data.py 中做了两处关键修复：
        1. ``'num_workers': max(args.num_workers, 1)``
           — 原版允许 0 worker，导致 TFRecord reader 起不来；修复后至少 1 个 worker。
        2. ``'threaded_dl': args.num_workers > 0``
           — 仅当用户显式指定 num_workers > 0 时开启 ThreadedIterator 预取。

    Walpurgis 将这两个决策集中到 `effective_workers()` 和 `should_thread()`，
    使配置意图在一处可读、可测试，而不是散落在 configure_data 的条件表达式中。

    Args:
        num_workers: 用户原始 ``--num-workers`` 值（可为 0）。
        seed: 随机种子（用于 TFRecord reader 内部 shuffle）。
        rank: 当前进程 rank（用于 seed 偏移：``seed + rank + 1``）。
    """
    num_workers: int
    seed: int
    rank: int = 0

    def effective_workers(self) -> int:
        """实际 worker 数，上游用 ``max(num_workers, 1)`` 兜底。"""
        eff = max(self.num_workers, 1)
        _dbg(
            "TF_LOADER_CONFIG",
            f"num_workers={self.num_workers} effective={eff}"
        )
        return eff

    def should_thread(self) -> bool:
        """是否启用 ThreadedIterator，上游条件：``num_workers > 0``。"""
        use = self.num_workers > 0
        _dbg(
            "TF_LOADER_CONFIG",
            f"num_workers={self.num_workers} threaded_dl={use}"
        )
        return use

    def rank_seed(self) -> int:
        """rank 偏移后的种子，上游：``seed + rank + 1``。"""
        rs = self.seed + self.rank + 1
        _dbg("TF_LOADER_CONFIG", f"seed={self.seed} rank={self.rank} rank_seed={rs}")
        return rs


# ---------------------------------------------------------------------------
# PresplitPolicy — presplit-sentences 两种数据集路径封装
# ---------------------------------------------------------------------------

@dataclass
class PresplitPolicy:
    """presplit-sentences 数据集路径策略。

    上游 data_utils/datasets.py 66719e973 新增 ``--presplit-sentences`` 支持：
    - **presplit=False（原版）**：文档以整段文字存储，训练时随机截取 seq_len 窗口；
    - **presplit=True（新增）**：文档以 newline 分隔的句子列表存储，
      训练时逐句迭代，拼接至 seq_len，避免 epoch 内重复分词。

    Walpurgis 将此二态显式建模，`describe()` 文字化策略，
    使调用方无需阅读 `if presplit_sentences:` 分支才能理解行为。

    Args:
        presplit: 是否使用预分割句子格式（对应 ``--presplit-sentences`` flag）。
        newline_sep: 句子分隔符（上游固定为 ``\\n``）。
    """
    presplit: bool
    newline_sep: str = "\n"

    def describe(self) -> str:
        """人类可读的策略描述。"""
        if self.presplit:
            return (
                f"PRESPLIT: 文档已预先以 {self.newline_sep!r} 分割为句子列表，"
                "训练时逐句拼接，无需实时分词。推荐配合 scripts/presplit_sentences_json.py 使用。"
            )
        return (
            "FULL_DOC: 文档整段存储，训练时随机截取 seq_len 窗口，"
            "每 epoch 重复分词（原版行为，速度较慢）。"
        )

    def split_document(self, doc_text: str) -> List[str]:
        """将文档按策略拆分为句子列表或整段列表。

        Args:
            doc_text: 原始文档文本。

        Returns:
            presplit=True 时返回按 newline_sep 分割的句子列表（过滤空行）；
            presplit=False 时返回包含整段文本的单元素列表。
        """
        _dbg(
            "PRESPLIT_SPLIT",
            f"presplit={self.presplit} doc_len={len(doc_text)} "
            f"sep={self.newline_sep!r}"
        )
        if self.presplit:
            sentences = [s.strip() for s in doc_text.split(self.newline_sep) if s.strip()]
            _dbg("PRESPLIT_SPLIT", f"split into {len(sentences)} sentences")
            return sentences
        return [doc_text]

    def sentences_from_json_doc(self, doc: dict, text_key: str = "text") -> List[str]:
        """从 JSON 文档对象中提取句子列表。

        上游 datasets.py：
            presplit=True  → doc[text_key].split(\"\\n\")
            presplit=False → [doc[text_key]] （整段）

        Args:
            doc: 一行 JSON 解析结果（dict）。
            text_key: 文本字段名（对应 ``--text-key`` 参数）。

        Returns:
            句子字符串列表。
        """
        raw = doc.get(text_key, "")
        result = self.split_document(raw)
        _dbg(
            "PRESPLIT_JSON",
            f"text_key={text_key!r} raw_len={len(raw)} "
            f"sentences={len(result)} policy={'presplit' if self.presplit else 'full_doc'}"
        )
        return result


# ---------------------------------------------------------------------------
# WalpurgisDatasetConfig — 数据集综合配置（arguments.py 新增参数的结构化表示）
# ---------------------------------------------------------------------------

@dataclass
class WalpurgisDatasetConfig:
    """Megatron 66719e973 新增的数据集配置参数，结构化封装。

    对应上游 arguments.py 新增的两个参数：
    - ``--shuffle``：确定性随机采样开关（与 seed+epoch 绑定）；
    - ``--presplit-sentences``：预分割句子格式开关。

    并整合 TFRecordLoaderConfig 与 PresplitPolicy，
    形成完整的\"faster dataloader\"配置视图。

    Args:
        shuffle: 是否对训练数据执行确定性随机打乱。
        presplit_sentences: 是否使用预分割句子格式。
        num_workers: DataLoader 工作进程数。
        seed: 随机基准种子。
        train_iters: 训练总迭代数（用于 RandomSampler 预算计算）。
        batch_size: 每批大小（用于 RandomSampler 预算计算）。
        rank: 当前进程 rank。
    """
    shuffle: bool = False
    presplit_sentences: bool = False
    num_workers: int = 2
    seed: int = 1234
    train_iters: int = 0
    batch_size: int = 1
    rank: int = 0

    def to_tfrecord_config(self) -> TFRecordLoaderConfig:
        """生成 TFRecord 加载配置。"""
        cfg = TFRecordLoaderConfig(
            num_workers=self.num_workers,
            seed=self.seed,
            rank=self.rank,
        )
        _dbg(
            "DATASET_CONFIG_TF",
            f"effective_workers={cfg.effective_workers()} "
            f"threaded={cfg.should_thread()} rank_seed={cfg.rank_seed()}"
        )
        return cfg

    def to_presplit_policy(self) -> PresplitPolicy:
        """生成 presplit-sentences 策略实例。"""
        policy = PresplitPolicy(presplit=self.presplit_sentences)
        _dbg("DATASET_CONFIG_PRESPLIT", policy.describe())
        return policy

    def sampler_num_samples(self) -> int:
        """RandomSampler 的总采样数 = batch_size × train_iters。"""
        n = self.batch_size * self.train_iters
        _dbg("DATASET_CONFIG_SAMPLER", f"batch_size={self.batch_size} "
             f"train_iters={self.train_iters} num_samples={n}")
        return n

    def audit_report(self) -> str:
        """输出完整配置审计报告（用于日志 / DEBUG）。"""
        tf_cfg = self.to_tfrecord_config()
        presplit = self.to_presplit_policy()
        lines = [
            "=== WalpurgisDatasetConfig Audit Report ===",
            f"  shuffle:            {self.shuffle}",
            f"  presplit_sentences: {self.presplit_sentences}",
            f"  num_workers:        {self.num_workers} → effective={tf_cfg.effective_workers()}",
            f"  threaded_dl:        {tf_cfg.should_thread()}",
            f"  seed:               {self.seed} → rank_seed={tf_cfg.rank_seed()}",
            f"  sampler_num_samples:{self.sampler_num_samples()} "
            f"(batch={self.batch_size} × iters={self.train_iters})",
            f"  presplit_policy:    {presplit.describe()}",
            "==========================================",
        ]
        report = "\n".join(lines)
        _dbg("DATASET_CONFIG_AUDIT", "\n" + report)
        return report


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """模块加载时的五项断言（WALPURGIS_DEBUG=1 时执行）。"""
    if not _DEBUG:
        return

    # 1. ThreadSpec 默认值与 summary
    spec = ThreadSpec()
    assert spec.maxsize == 16 and spec.daemon is True
    assert spec.is_sentinel(None)
    assert not spec.is_sentinel(42)
    _dbg("SELF_CHECK", f"ThreadSpec OK: {spec.summary()}")

    # 2. TFRecordLoaderConfig 两个决策
    cfg0 = TFRecordLoaderConfig(num_workers=0, seed=42, rank=2)
    assert cfg0.effective_workers() == 1
    assert not cfg0.should_thread()
    assert cfg0.rank_seed() == 45  # 42 + 2 + 1
    cfg2 = TFRecordLoaderConfig(num_workers=4, seed=100, rank=0)
    assert cfg2.effective_workers() == 4
    assert cfg2.should_thread()
    _dbg("SELF_CHECK", "TFRecordLoaderConfig OK")

    # 3. PresplitPolicy split_document
    p_pre = PresplitPolicy(presplit=True)
    p_full = PresplitPolicy(presplit=False)
    doc = "Hello world.\nThis is a sentence.\n\nAnother one."
    pre_result = p_pre.split_document(doc)
    full_result = p_full.split_document(doc)
    assert len(pre_result) == 3, f"expected 3 sentences, got {len(pre_result)}: {pre_result}"
    assert len(full_result) == 1 and full_result[0] == doc
    _dbg("SELF_CHECK", f"PresplitPolicy OK: presplit={pre_result}")

    # 4. WalpurgisDatasetConfig.sampler_num_samples
    dc = WalpurgisDatasetConfig(batch_size=32, train_iters=100)
    assert dc.sampler_num_samples() == 3200
    _dbg("SELF_CHECK", "WalpurgisDatasetConfig.sampler_num_samples OK")

    # 5. audit_report 包含关键字段
    dc2 = WalpurgisDatasetConfig(
        shuffle=True, presplit_sentences=True,
        num_workers=2, seed=1234, train_iters=1000, batch_size=8, rank=0
    )
    report = dc2.audit_report()
    assert "shuffle:            True" in report
    assert "threaded_dl:        True" in report
    assert "PRESPLIT" in report
    _dbg("SELF_CHECK", "audit_report OK")

    _dbg("SELF_CHECK", "✓ 全部 5 项断言通过")


_dbg("MODULE_LOAD", "faster_dl_tfrecord_66719e9 载入开始")
_self_check()
_dbg("MODULE_LOAD", "faster_dl_tfrecord_66719e9 载入完成")
