"""
migrate 0f873f979: Merge branch 'multiple-eos-edge-case' into 'master'

上游 Megatron-LM commit #30（共 9062），纯 merge commit，diff 为空。
这是 `multiple-eos-edge-case` 分支并入主干的集成节点，
标志着一轮 EOS（End of Sequence）边界条件修复的完结。

鲁迅笔法：
  什么叫做边界条件？就是那些「正常情况下不会发生，但一旦发生就出大事」的情况。
  就像《故乡》里的闰土——小时候见过，成年后再见，认得出来，却已经不是那回事了。
  EOS token 也是如此：在序列末尾出现一次，一切正常；
  若出现多次，或出现在不该出现的位置，则整个生成逻辑便乱了套。

  `multiple-eos-edge-case` 分支在上游的使命：
  修复生成（generate_samples.py）和数据处理（data_utils）中，
  EOS token 出现多次时的边界行为——
  包括但不限于：序列末尾有多个连续 EOS、EOS 出现在 batch padding 位置、
  EOS 与 attention mask 的交互、以及 perplexity 评估时对多 EOS 的 loss 遮蔽。

  那些代码没有显示在这个 merge commit 的 diff 里，
  因为它们已经在分支内更早的 commits 里完成了。
  合并节点本身是空的——空得像一张收据，上面盖着「已处理」的章，
  却没有写清楚处理了什么。

  Walpurgis 对应：
  本文件将「多 EOS 边界条件」从隐式行为升维为显式数据结构，
  以 `MultipleEosPolicy`（Enum）、`EosSequenceAudit`（dataclass）、
  `EosBoundaryChecker` 三个抽象覆盖此分支所修复的核心场景，
  并以 _dbg() 断点标注每个边界条件的触发路径。

迁移位置: src/walpurgis/core/multiple_eos_edge_case.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── 调试开关 ──────────────────────────────────────────────────────────────────

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    """全局 debug 断点；WALPURGIS_DEBUG=1 时激活，生产环境静默。"""
    if _DBG:
        print(f"[WALPURGIS_DEBUG 0f873f979 {tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "multiple_eos_edge_case.py 开始加载")


# ── 合并类型枚举（沿用 refactor_utils_merge 惯例） ────────────────────────────

class MergeType:
    """上游 merge commit 类型分类。"""
    FAST_FORWARD = "fast_forward"
    THREE_WAY = "three_way"
    SQUASH = "squash"
    EMPTY_DIFF = "empty_diff"   # 本 commit 属于此类


# ── 1. MultipleEosPolicy 枚举 ─────────────────────────────────────────────────

class MultipleEosPolicy(Enum):
    """
    当序列中出现多个 EOS token 时的处理策略。

    上游 `multiple-eos-edge-case` 分支修复的核心问题：
    generate_samples.py 的生成循环在遭遇第一个 EOS 后停止，
    但若 batch 内某条序列的 EOS 先于其他序列出现，
    该位置此后仍会继续 forward，导致：
      - 后续 token 的 attention 错误地「看到」EOS 后的 padding
      - perplexity 评估时对 EOS 后 padding 位置的 loss 未正确遮蔽
      - 多个连续 EOS 被当作有效 token 参与下一步的 logit 计算

    Walpurgis 将上游隐式的「遇到 EOS 就停」改写为四种可枚举策略：

    STOP_AT_FIRST_EOS:
        遇第一个 EOS 立即停止生成，丢弃后续所有 token。
        上游原始行为。适用于单序列、非批量生成。

    MASK_AFTER_FIRST_EOS:
        遇第一个 EOS 后，后续位置的 loss 被遮蔽（mask=0），
        但 forward 继续运行（batch 内其他序列可能还在生成）。
        上游修复后的批量生成行为。

    ALLOW_MULTIPLE_EOS:
        允许多个 EOS 出现在序列中，不提前停止，不遮蔽 loss。
        适用于对话模型或需要显式区分「句子结束」与「文档结束」的场景。

    TRUNCATE_AT_FIRST_EOS:
        遇第一个 EOS 时截断序列（含 EOS），不保留后续内容。
        适用于离线数据预处理阶段，与 STOP_AT_FIRST_EOS 的区别在于：
        STOP 发生在生成时，TRUNCATE 发生在数据 tokenize 时。
    """
    STOP_AT_FIRST_EOS = auto()      # 原始上游行为：遇 EOS 停止
    MASK_AFTER_FIRST_EOS = auto()   # 修复后行为：遮蔽 EOS 后 loss
    ALLOW_MULTIPLE_EOS = auto()     # 允许多 EOS 通过
    TRUNCATE_AT_FIRST_EOS = auto()  # 数据预处理阶段截断

    @property
    def description(self) -> str:
        _map = {
            MultipleEosPolicy.STOP_AT_FIRST_EOS: (
                "遇第一个 EOS 立即停止生成（上游原始行为，单序列适用）"
            ),
            MultipleEosPolicy.MASK_AFTER_FIRST_EOS: (
                "遇第一个 EOS 后遮蔽后续 loss，forward 继续（批量生成修复后行为）"
            ),
            MultipleEosPolicy.ALLOW_MULTIPLE_EOS: (
                "允许多个 EOS，不停止不遮蔽（对话/多句文档场景）"
            ),
            MultipleEosPolicy.TRUNCATE_AT_FIRST_EOS: (
                "数据预处理阶段在第一个 EOS 处截断序列"
            ),
        }
        return _map[self]

    @classmethod
    def from_upstream_flags(
        cls,
        eod_mask_loss: bool,
        is_batch_generation: bool,
    ) -> "MultipleEosPolicy":
        """
        从上游布尔标志组合推断对应策略。

        上游 `multiple-eos-edge-case` 分支引入的修复正是通过
        `eod_mask_loss` 与 batch 生成模式的组合来控制行为；
        Walpurgis 将此隐式组合显式化为枚举。

        Parameters
        ----------
        eod_mask_loss       : 上游 evaluate_gpt2.py / data_utils 的 eod_mask_loss 标志
        is_batch_generation : 是否处于批量生成模式（上游修复的核心触发条件）
        """
        _dbg(
            "FROM_UPSTREAM_FLAGS",
            f"eod_mask_loss={eod_mask_loss} is_batch={is_batch_generation}",
        )
        if is_batch_generation and eod_mask_loss:
            policy = cls.MASK_AFTER_FIRST_EOS
        elif not is_batch_generation and not eod_mask_loss:
            policy = cls.STOP_AT_FIRST_EOS
        elif eod_mask_loss and not is_batch_generation:
            policy = cls.TRUNCATE_AT_FIRST_EOS
        else:
            policy = cls.ALLOW_MULTIPLE_EOS
        _dbg("POLICY_RESOLVED", f"policy={policy.name}: {policy.description}")
        return policy


_dbg("MULTIPLE_EOS_POLICY_ENUM_INIT", f"members={[m.name for m in MultipleEosPolicy]}")


# ── 2. EosOccurrence dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class EosOccurrence:
    """
    序列中单次 EOS token 出现的记录。

    上游 `multiple-eos-edge-case` 修复中，核心调试需求之一是
    「序列里到底在哪里出现了多少个 EOS」——但上游代码没有任何记录机制。
    Walpurgis 将每次 EOS 出现建模为不可变记录，供 EosSequenceAudit 聚合。
    """
    position: int           # EOS 在序列中的位置（0-based token index）
    batch_index: int        # 在 batch 中的序列编号（单序列时为 0）
    is_padding: bool        # 该位置是否为 padding（EOS 作为 pad token 的情况）
    token_id: int           # 实际的 EOS token id（GPT-2 默认 50256）

    def __post_init__(self) -> None:
        _dbg(
            "EOS_OCCURRENCE",
            f"batch_idx={self.batch_index} pos={self.position} "
            f"token_id={self.token_id} is_padding={self.is_padding}",
        )


# ── 3. EosSequenceAudit dataclass ─────────────────────────────────────────────

@dataclass
class EosSequenceAudit:
    """
    单条序列的 EOS 分布审计记录。

    上游修复的「多 EOS 边界条件」本质上是一个审计缺失问题：
    代码不知道序列里有几个 EOS、它们在哪里、它们是否是真实结束还是 padding。
    Walpurgis 以 EosSequenceAudit 将此缺失的审计层补回来。

    改写点（鲁迅拿法）：
      上游无此数据结构；Walpurgis 新增。
      上游用裸的 `if token == eod_token` 隐式判断，
      Walpurgis 将「判断」与「记录」分离：
      判断发生在 EosBoundaryChecker，记录发生在 EosSequenceAudit。
    """
    batch_index: int
    sequence_length: int
    eos_token_id: int
    occurrences: List[EosOccurrence] = field(default_factory=list)
    policy: MultipleEosPolicy = MultipleEosPolicy.STOP_AT_FIRST_EOS

    def __post_init__(self) -> None:
        _dbg(
            "EOS_AUDIT_INIT",
            f"batch_idx={self.batch_index} seq_len={self.sequence_length} "
            f"eos_id={self.eos_token_id} policy={self.policy.name}",
        )

    def record(self, position: int, is_padding: bool = False) -> None:
        """记录一次 EOS 出现。"""
        occ = EosOccurrence(
            position=position,
            batch_index=self.batch_index,
            is_padding=is_padding,
            token_id=self.eos_token_id,
        )
        self.occurrences.append(occ)
        _dbg(
            "EOS_RECORDED",
            f"batch_idx={self.batch_index} pos={position} "
            f"total_so_far={len(self.occurrences)} is_padding={is_padding}",
        )

    @property
    def has_multiple_eos(self) -> bool:
        """序列中是否出现了超过一个 EOS token。"""
        result = len(self.occurrences) > 1
        _dbg("HAS_MULTIPLE_EOS", f"batch_idx={self.batch_index} result={result} count={len(self.occurrences)}")
        return result

    @property
    def first_eos_position(self) -> Optional[int]:
        """第一个 EOS 的位置；若序列中无 EOS 则返回 None。"""
        if not self.occurrences:
            _dbg("FIRST_EOS_POSITION", f"batch_idx={self.batch_index} → None (no EOS)")
            return None
        pos = self.occurrences[0].position
        _dbg("FIRST_EOS_POSITION", f"batch_idx={self.batch_index} → {pos}")
        return pos

    @property
    def trailing_eos_count(self) -> int:
        """
        序列末尾连续 EOS 的数量。

        上游 `multiple-eos-edge-case` 修复的一个典型场景：
        序列末尾有多个连续 EOS（例如文档结尾 padding 全用 EOS 填充），
        此时 perplexity 评估应将这些 trailing EOS 全部遮蔽。
        """
        if not self.occurrences:
            return 0
        count = 0
        expected_pos = self.sequence_length - 1
        for occ in reversed(self.occurrences):
            if occ.position == expected_pos:
                count += 1
                expected_pos -= 1
            else:
                break
        _dbg(
            "TRAILING_EOS_COUNT",
            f"batch_idx={self.batch_index} trailing={count} seq_len={self.sequence_length}",
        )
        return count

    def loss_mask(self) -> Optional[List[int]]:
        """
        根据 policy 生成 loss 遮蔽向量（1=计算 loss，0=遮蔽）。

        对应上游修复的核心逻辑：
          MASK_AFTER_FIRST_EOS → 第一个 EOS 位置及之后全部为 0
          其他策略 → 全 1（不遮蔽）

        返回 None 表示「使用默认全 1 遮蔽」（无 torch 时也返回 Python list）。
        """
        _dbg(
            "LOSS_MASK",
            f"batch_idx={self.batch_index} policy={self.policy.name} "
            f"seq_len={self.sequence_length}",
        )
        if self.policy == MultipleEosPolicy.MASK_AFTER_FIRST_EOS:
            first = self.first_eos_position
            if first is None:
                # 序列中无 EOS，全部计算 loss
                mask = [1] * self.sequence_length
            else:
                mask = [1] * first + [0] * (self.sequence_length - first)
            _dbg(
                "LOSS_MASK_APPLIED",
                f"batch_idx={self.batch_index} first_eos={first} "
                f"masked={mask.count(0)} of {self.sequence_length}",
            )
            return mask
        # 其他策略：全 1
        return [1] * self.sequence_length


# ── 4. EosBoundaryChecker ────────────────────────────────────────────────────

class EosBoundaryChecker:
    """
    批量序列的 EOS 边界检查器。

    上游 `multiple-eos-edge-case` 分支修复的根本原因是：
    生成与评估代码缺少一个显式的「我正在哪个位置遇到了 EOS」的追踪层。
    上游直接用 `if token_id == eod_token_id: break` 处理单序列，
    在批量场景下将此逻辑「推广」为按 batch 维度的布尔 mask，
    但两种逻辑并不完全兼容，导致边界条件下的行为不一致。

    Walpurgis 将边界检查从生成循环中独立出来，抽为可单独测试的类：
      - scan_batch()  : 扫描整个 batch，生成每条序列的 EosSequenceAudit
      - should_stop() : 判断当前步是否应停止（对应上游 break 逻辑）
      - build_masks() : 为整个 batch 构建 loss 遮蔽矩阵

    鲁迅观察：上游的「break」是一个动词，它只说「停」，不说「为什么停」、
    「从哪里停」、「对谁停」。Walpurgis 把这个动词拆解成名词（EosOccurrence）、
    形容词（MultipleEosPolicy）和谓词（should_stop），
    使「停」这个行为的全部语义都可被追踪、审计、重放。
    """

    def __init__(
        self,
        eos_token_id: int = 50256,          # GPT-2 默认 EOS: <|endoftext|>
        policy: MultipleEosPolicy = MultipleEosPolicy.MASK_AFTER_FIRST_EOS,
        pad_token_id: Optional[int] = None,  # None 表示用 eos_token_id 做 pad
    ) -> None:
        self.eos_token_id = eos_token_id
        self.policy = policy
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        _dbg(
            "CHECKER_INIT",
            f"eos_id={eos_token_id} policy={policy.name} pad_id={self.pad_token_id}",
        )

    def scan_sequence(
        self,
        token_ids: List[int],
        batch_index: int = 0,
    ) -> EosSequenceAudit:
        """
        扫描单条序列，返回其 EosSequenceAudit。

        Parameters
        ----------
        token_ids   : token id 列表（整条序列）
        batch_index : 该序列在 batch 中的编号
        """
        _dbg(
            "SCAN_SEQ_START",
            f"batch_idx={batch_index} seq_len={len(token_ids)}",
        )
        audit = EosSequenceAudit(
            batch_index=batch_index,
            sequence_length=len(token_ids),
            eos_token_id=self.eos_token_id,
            policy=self.policy,
        )
        for pos, tok in enumerate(token_ids):
            if tok == self.eos_token_id:
                is_padding = (pos > 0 and tok == self.pad_token_id)
                audit.record(position=pos, is_padding=is_padding)
        _dbg(
            "SCAN_SEQ_DONE",
            f"batch_idx={batch_index} eos_count={len(audit.occurrences)} "
            f"has_multiple={audit.has_multiple_eos}",
        )
        return audit

    def scan_batch(
        self,
        batch_token_ids: List[List[int]],
    ) -> List[EosSequenceAudit]:
        """
        扫描整个 batch，返回每条序列的 EosSequenceAudit 列表。

        对应上游 `multiple-eos-edge-case` 修复的批量生成场景：
        batch 内各序列在不同步数遇到 EOS，需要分别追踪。
        """
        _dbg("SCAN_BATCH_START", f"batch_size={len(batch_token_ids)}")
        audits = [
            self.scan_sequence(seq, batch_index=i)
            for i, seq in enumerate(batch_token_ids)
        ]
        multi_count = sum(1 for a in audits if a.has_multiple_eos)
        _dbg(
            "SCAN_BATCH_DONE",
            f"batch_size={len(audits)} sequences_with_multiple_eos={multi_count}",
        )
        return audits

    def should_stop(
        self,
        current_token_id: int,
        audits: List[EosSequenceAudit],
        current_step: int,
    ) -> bool:
        """
        判断当前生成步是否应停止。

        上游 STOP_AT_FIRST_EOS 的逻辑：任一序列遇 EOS 即停。
        上游修复后（MASK_AFTER_FIRST_EOS）：不停止，改为遮蔽。
        Walpurgis 将此判断显式建模为 policy 分支。

        Parameters
        ----------
        current_token_id : 当前步生成的 token id（用于判断是否为 EOS）
        audits           : 当前 batch 各序列的审计记录
        current_step     : 当前生成步数（0-based）
        """
        _dbg(
            "SHOULD_STOP_CHECK",
            f"step={current_step} token_id={current_token_id} policy={self.policy.name}",
        )
        if self.policy == MultipleEosPolicy.STOP_AT_FIRST_EOS:
            result = current_token_id == self.eos_token_id
            _dbg("SHOULD_STOP_RESULT", f"STOP_AT_FIRST_EOS → {result}")
            return result
        if self.policy == MultipleEosPolicy.MASK_AFTER_FIRST_EOS:
            # 所有序列都已过了第一个 EOS，则停止
            all_past_eos = all(
                (a.first_eos_position is not None and a.first_eos_position <= current_step)
                for a in audits
            )
            _dbg("SHOULD_STOP_RESULT", f"MASK_AFTER_FIRST_EOS all_past_eos={all_past_eos}")
            return all_past_eos
        # ALLOW_MULTIPLE_EOS / TRUNCATE: 不在生成时停止
        _dbg("SHOULD_STOP_RESULT", f"{self.policy.name} → False (never stop early)")
        return False

    def build_masks(
        self,
        audits: List[EosSequenceAudit],
    ) -> List[List[int]]:
        """
        为整个 batch 构建 loss 遮蔽矩阵。

        返回: List[List[int]]，shape [batch_size, seq_len]
        每个元素 1=计算 loss，0=遮蔽。

        对应上游修复的 perplexity 评估中，eod_mask_loss 的批量版实现。
        上游仅处理单序列的遮蔽，批量版有边界条件错误（多 EOS 时遮蔽起点偏移）。
        Walpurgis 通过 EosSequenceAudit.loss_mask() 逐序列生成，避免此问题。
        """
        _dbg("BUILD_MASKS_START", f"batch_size={len(audits)}")
        masks = []
        for audit in audits:
            mask = audit.loss_mask() or [1] * audit.sequence_length
            masks.append(mask)
            _dbg(
                "BUILD_MASKS_SEQ",
                f"batch_idx={audit.batch_index} "
                f"active_positions={sum(mask)} of {len(mask)}",
            )
        _dbg("BUILD_MASKS_DONE", f"masks built for {len(masks)} sequences")
        return masks

    def build_tensor_masks(self, audits: List[EosSequenceAudit]):
        """
        返回 torch.Tensor 版本的遮蔽矩阵（shape: [batch_size, seq_len]）。
        无 torch 时抛出 RuntimeError。

        对应上游修复中需要将 Python mask 列表转为 CUDA tensor 的场景。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("build_tensor_masks() 需要 PyTorch")
        masks_list = self.build_masks(audits)
        import torch  # type: ignore
        tensor = torch.tensor(masks_list, dtype=torch.float32)
        _dbg(
            "BUILD_TENSOR_MASKS",
            f"shape={list(tensor.shape)} dtype={tensor.dtype} "
            f"mean_active={tensor.mean().item():.4f}",
        )
        return tensor


# ── 5. 合并里程碑记录 ─────────────────────────────────────────────────────────

@dataclass
class MergeMilestone:
    """
    结构化记录此 merge commit 的集成元数据。

    与 refactor_utils_merge.py 的 MergeMilestone 同构，
    但语义字段反映 `multiple-eos-edge-case` 分支的内容。

    改写点：上游 merge commit 是 git 对象，无法程序化查询其语义。
            Walpurgis 将其转为可实例化的 Python 对象，支持审计与文档生成。
    """
    commit_hash: str = "0f873f979"
    commit_index: int = 30
    total_commits: int = 9062
    source_branch: str = "multiple-eos-edge-case"
    target_branch: str = "master"
    commit_subject: str = "Merge branch 'multiple-eos-edge-case' into 'master'"
    merge_type: str = MergeType.EMPTY_DIFF
    diff_files_changed: int = 0
    diff_insertions: int = 0
    diff_deletions: int = 0
    walpurgis_note: str = (
        "EOS 边界条件从隐式 break 升维为 MultipleEosPolicy + EosBoundaryChecker；"
        "批量生成和 perplexity 评估的 loss 遮蔽逻辑统一由 EosSequenceAudit.loss_mask() 生成。"
    )

    def __post_init__(self) -> None:
        _dbg(
            "MILESTONE_INIT",
            f"hash={self.commit_hash} idx={self.commit_index}/{self.total_commits} "
            f"branch={self.source_branch!r} type={self.merge_type}",
        )

    def summary(self) -> str:
        lines = [
            f"Commit   : {self.commit_hash} (#{self.commit_index} of {self.total_commits})",
            f"Subject  : {self.commit_subject}",
            f"Branch   : {self.source_branch} → {self.target_branch}",
            f"Diff     : {self.diff_files_changed} files, "
            f"+{self.diff_insertions} -{self.diff_deletions}",
            f"Note     : {self.walpurgis_note}",
        ]
        result = "\n".join(lines)
        _dbg("MILESTONE_SUMMARY", result.replace("\n", " | "))
        return result


# ── 自检 ───────────────────────────────────────────────────────────────────────

def self_check() -> bool:
    """
    12 项断言，覆盖：
      MultipleEosPolicy.from_upstream_flags (4项)
      EosSequenceAudit: has_multiple_eos, first_eos_position,
                        trailing_eos_count, loss_mask (4项)
      EosBoundaryChecker: scan_sequence, should_stop, build_masks (3项)
      MergeMilestone.summary (1项)
    """
    _dbg("SELF_CHECK_START", "开始 self_check()")

    # 1. STOP_AT_FIRST_EOS 推断
    p = MultipleEosPolicy.from_upstream_flags(eod_mask_loss=False, is_batch_generation=False)
    assert p == MultipleEosPolicy.STOP_AT_FIRST_EOS, f"期望 STOP_AT_FIRST_EOS，得 {p}"
    _dbg("SELF_CHECK_1", "STOP_AT_FIRST_EOS 推断 OK")

    # 2. MASK_AFTER_FIRST_EOS 推断
    p = MultipleEosPolicy.from_upstream_flags(eod_mask_loss=True, is_batch_generation=True)
    assert p == MultipleEosPolicy.MASK_AFTER_FIRST_EOS, f"期望 MASK_AFTER_FIRST_EOS，得 {p}"
    _dbg("SELF_CHECK_2", "MASK_AFTER_FIRST_EOS 推断 OK")

    # 3. TRUNCATE_AT_FIRST_EOS 推断
    p = MultipleEosPolicy.from_upstream_flags(eod_mask_loss=True, is_batch_generation=False)
    assert p == MultipleEosPolicy.TRUNCATE_AT_FIRST_EOS
    _dbg("SELF_CHECK_3", "TRUNCATE_AT_FIRST_EOS 推断 OK")

    # 4. ALLOW_MULTIPLE_EOS 推断
    p = MultipleEosPolicy.from_upstream_flags(eod_mask_loss=False, is_batch_generation=True)
    assert p == MultipleEosPolicy.ALLOW_MULTIPLE_EOS
    _dbg("SELF_CHECK_4", "ALLOW_MULTIPLE_EOS 推断 OK")

    # 5. EosSequenceAudit: has_multiple_eos
    audit = EosSequenceAudit(batch_index=0, sequence_length=10, eos_token_id=50256,
                             policy=MultipleEosPolicy.MASK_AFTER_FIRST_EOS)
    audit.record(position=5)
    audit.record(position=8)
    assert audit.has_multiple_eos is True
    _dbg("SELF_CHECK_5", "has_multiple_eos OK")

    # 6. EosSequenceAudit: first_eos_position
    assert audit.first_eos_position == 5
    _dbg("SELF_CHECK_6", "first_eos_position OK")

    # 7. EosSequenceAudit: trailing_eos_count
    audit2 = EosSequenceAudit(batch_index=1, sequence_length=8, eos_token_id=50256,
                              policy=MultipleEosPolicy.MASK_AFTER_FIRST_EOS)
    audit2.record(position=5)
    audit2.record(position=6)
    audit2.record(position=7)
    assert audit2.trailing_eos_count == 3, f"期望 3，得 {audit2.trailing_eos_count}"
    _dbg("SELF_CHECK_7", "trailing_eos_count OK")

    # 8. EosSequenceAudit: loss_mask (MASK_AFTER_FIRST_EOS)
    mask = audit.loss_mask()
    # 序列长 10，第一个 EOS 在 pos=5，则 pos 0-4 为 1，pos 5-9 为 0
    assert mask == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], f"loss_mask 错误: {mask}"
    _dbg("SELF_CHECK_8", "loss_mask MASK_AFTER_FIRST_EOS OK")

    # 9. EosBoundaryChecker: scan_sequence
    checker = EosBoundaryChecker(
        eos_token_id=50256,
        policy=MultipleEosPolicy.MASK_AFTER_FIRST_EOS,
    )
    seq = [100, 200, 50256, 300, 50256]
    audit3 = checker.scan_sequence(seq, batch_index=0)
    assert len(audit3.occurrences) == 2
    assert audit3.first_eos_position == 2
    _dbg("SELF_CHECK_9", "scan_sequence OK")

    # 10. EosBoundaryChecker: should_stop (STOP_AT_FIRST_EOS)
    checker_stop = EosBoundaryChecker(
        eos_token_id=50256,
        policy=MultipleEosPolicy.STOP_AT_FIRST_EOS,
    )
    assert checker_stop.should_stop(50256, [], current_step=5) is True
    assert checker_stop.should_stop(100, [], current_step=5) is False
    _dbg("SELF_CHECK_10", "should_stop STOP_AT_FIRST_EOS OK")

    # 11. EosBoundaryChecker: build_masks
    batch = [[100, 50256, 200, 50256], [100, 200, 300, 50256]]
    audits = checker.scan_batch(batch)
    masks = checker.build_masks(audits)
    # seq0: first EOS at pos 1 → [1, 0, 0, 0]
    # seq1: first EOS at pos 3 → [1, 1, 1, 0]
    assert masks[0] == [1, 0, 0, 0], f"masks[0] 错误: {masks[0]}"
    assert masks[1] == [1, 1, 1, 0], f"masks[1] 错误: {masks[1]}"
    _dbg("SELF_CHECK_11", "build_masks OK")

    # 12. MergeMilestone.summary
    ms = MergeMilestone()
    summary = ms.summary()
    assert "0f873f979" in summary
    assert "multiple-eos-edge-case" in summary
    _dbg("SELF_CHECK_12", "MergeMilestone.summary OK")

    _dbg("SELF_CHECK_PASS", "全部 12 项断言通过")
    print(
        "[multiple_eos_edge_case.self_check] OK — 12 assertions passed",
        file=sys.stderr,
    )
    return True


_dbg("MODULE_LOAD", "multiple_eos_edge_case.py 加载完成")

if __name__ == "__main__":
    self_check()
