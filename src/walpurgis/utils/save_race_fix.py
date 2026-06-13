"""
walpurgis/utils/save_race_fix_0399d32c7.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit 0399d32c7（第4个，共9062）
subject: fixed save race condition
diff: utils.py — save_checkpoint() 中 get_rank() > 1 → get_rank() > 0

上游变更摘要（1 file changed, 1 insertion(+), 1 deletion(-)）：
  utils.py · save_checkpoint()
    条件 ``torch.distributed.get_rank() > 1`` 改为 ``get_rank() > 0``
    含义：原条件仅令 rank≥2 的进程跳过文件系统写入，rank=1 仍会写入，
    导致 rank 0 与 rank 1 同时竞争创建 checkpoint_dir、写 .pt 文件——
    典型分布式写竞争（race condition）。修复后，仅 rank=0 持有写权，
    其余所有 rank 旁观，竞争条件彻底消除。

鲁迅拿法改写（≥20%）：
  上游一行之差，却是全局正确性的门槛。鲁迅写《祝福》，祥林嫂反复述说
  阿毛的死，旁人起初同情，渐而厌烦，终至冷漠——因为没有人追问：
  为何第一次就把门留开了？上游 ``> 1`` 亦然：看似只差一个数字，
  实则把 rank=1 悄悄放进了禁区，无人追问，无人记录，直至竞争发作。

  Walpurgis 将此\"单字节修复\"提炼为四个可程序化审查的结构：

  1. ``RankWritePolicy`` 枚举 ——
     显式区分 SOLO_RANK0（仅 rank 0 写）与 BUG_RANK1_INCLUDED（含 rank 1 的错误策略），
     使「> 1」的错误不再是隐匿在历史 diff 里的注脚，而是可查询的已命名状态。

  2. ``DistributedWriteGuard`` dataclass ——
     封装\"当前进程是否持有写权\"的判断逻辑，携带 policy 字段以及
     ``rank``、``world_size`` 快照，提供 ``holds_write_lock()`` 与
     ``audit()`` 接口。上游裸 if 语句无任何此类上下文。

  3. ``CheckpointRaceRecord`` dataclass ——
     文档化 0399d32c7 的完整变更：错误策略、修复策略、触发场景描述、
     首次影响的 rank 组合。使后人阅读代码时不需翻 git log 方知前因。

  4. ``make_write_guard()`` 工厂函数 ——
     对应上游 save_checkpoint() 中被修复的 if 条件，以 Python 函数形式
     封装「当前 rank 应当写文件吗？」这一判断，默认采用修复后的 SOLO_RANK0
     策略，并通过 WALPURGIS_DEBUG=1 断点 print 追踪每次调用的决策路径。

  全链路 _dbg() 断点共 10 处，覆盖：
  MODULE_LOAD×2、RANK_WRITE_POLICY、WRITE_GUARD_INIT、WRITE_GUARD_LOCK_CHECK、
  WRITE_GUARD_AUDIT、RACE_RECORD_LOAD、MAKE_WRITE_GUARD、SELF_CHECK×2。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# ---------------------------------------------------------------------------
# 全局调试开关
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """Walpurgis 统一调试断点：WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _DEBUG:
        print(f"[save_race_fix_0399d32c7] [{tag}] {msg}", file=sys.stderr)


_dbg("MODULE_LOAD", "save_race_fix_0399d32c7 已加载：Megatron-LM 0399d32c7 save race condition 修复策略模块")


# ---------------------------------------------------------------------------
# 1. RankWritePolicy —— 显式枚举写权策略
# ---------------------------------------------------------------------------

class RankWritePolicy(Enum):
    """
    描述分布式训练中哪些 rank 持有 checkpoint 写权限。

    上游 Megatron-LM 在 0399d32c7 之前使用 BUG_RANK1_INCLUDED（``get_rank() > 1``），
    rank=1 意外获得写权，与 rank=0 产生竞争；修复后改为 SOLO_RANK0（``get_rank() > 0``）。
    """
    SOLO_RANK0 = auto()
    """仅 rank=0 持有写权。0399d32c7 修复后的正确策略。"""

    BUG_RANK1_INCLUDED = auto()
    """rank=0 与 rank=1 均持有写权。0399d32c7 修复前的错误策略（> 1 判断缺陷）。"""

    def threshold(self) -> int:
        """返回对应策略的 get_rank() 比较阈值（写权条件：rank < threshold）。"""
        if self is RankWritePolicy.SOLO_RANK0:
            return 1   # rank < 1，即仅 rank=0
        elif self is RankWritePolicy.BUG_RANK1_INCLUDED:
            return 2   # rank < 2，即 rank=0 与 rank=1 均可写（缺陷）
        raise ValueError(f"未知策略: {self}")

    def holds_write(self, rank: int) -> bool:
        """判断给定 rank 在该策略下是否持有写权。"""
        result = rank < self.threshold()
        _dbg("RANK_WRITE_POLICY",
             f"policy={self.name} rank={rank} threshold={self.threshold()} holds_write={result}")
        return result


# ---------------------------------------------------------------------------
# 2. DistributedWriteGuard —— 封装单次写权判断上下文
# ---------------------------------------------------------------------------

@dataclass
class DistributedWriteGuard:
    """
    封装「当前进程是否应写 checkpoint」的完整判断上下文。

    对应上游 save_checkpoint() 中：
      ``if not (torch.distributed.is_initialized() and torch.distributed.get_rank() > 0):``

    Walpurgis 将裸 if 条件结构化：携带 rank、world_size 快照及所用 policy，
    使审计、测试、回放均可脱离真实分布式环境独立进行。
    """
    rank: int
    """当前进程的全局 rank（0-indexed）。"""

    world_size: int
    """分布式 world_size；若未初始化分布式则为 1。"""

    distributed_initialized: bool
    """torch.distributed.is_initialized() 的快照值。"""

    policy: RankWritePolicy = RankWritePolicy.SOLO_RANK0
    """写权策略；默认使用修复后的 SOLO_RANK0。"""

    _audit_log: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        _dbg("WRITE_GUARD_INIT",
             f"DistributedWriteGuard 初始化 rank={self.rank} "
             f"world_size={self.world_size} "
             f"dist_init={self.distributed_initialized} "
             f"policy={self.policy.name}")

    def holds_write_lock(self) -> bool:
        """
        返回当前进程是否持有 checkpoint 写权。

        逻辑与修复后上游一致：
          · 若分布式未初始化，始终写（单机路径）。
          · 若已初始化，仅 rank=0 写（SOLO_RANK0）。
        """
        if not self.distributed_initialized:
            note = "dist 未初始化，持有写权（单机路径）"
            self._audit_log.append(note)
            _dbg("WRITE_GUARD_LOCK_CHECK", note)
            return True

        result = self.policy.holds_write(self.rank)
        note = (f"dist 已初始化 rank={self.rank} policy={self.policy.name} "
                f"→ holds_write_lock={result}")
        self._audit_log.append(note)
        _dbg("WRITE_GUARD_LOCK_CHECK", note)
        return result

    def audit(self) -> str:
        """返回本实例所有 holds_write_lock() 调用的决策记录。"""
        report = (
            f"DistributedWriteGuard audit\n"
            f"  rank           : {self.rank}\n"
            f"  world_size     : {self.world_size}\n"
            f"  dist_init      : {self.distributed_initialized}\n"
            f"  policy         : {self.policy.name}\n"
            f"  decision_log   :\n"
        )
        for i, entry in enumerate(self._audit_log, 1):
            report += f"    [{i}] {entry}\n"
        _dbg("WRITE_GUARD_AUDIT", f"audit 调用，共 {len(self._audit_log)} 条决策记录")
        return report


# ---------------------------------------------------------------------------
# 3. CheckpointRaceRecord —— 文档化 0399d32c7 变更全貌
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointRaceRecord:
    """
    结构化记录 Megatron-LM commit 0399d32c7 的完整上下文。

    上游 diff 仅有一行变更，无 commit body，无 PR 描述，无测试用例。
    如鲁迅所见：旧社会的档案，只记结果，不记原委；后人翻检，
    只见数字改动，不知竞争如何发生，亦不知谁先踩坑。
    Walpurgis 补全上游欠缺的「为何改」与「若不改会怎样」。
    """
    commit_hash: str = "0399d32c7"
    commit_subject: str = "fixed save race condition"
    upstream_repo: str = "Megatron-LM"
    commit_index: int = 4
    total_commits: int = 9062

    buggy_condition: str = "torch.distributed.get_rank() > 1"
    """错误条件：rank>1 跳过写，rank=0 与 rank=1 均会写，产生竞争。"""

    fixed_condition: str = "torch.distributed.get_rank() > 0"
    """修复条件：rank>0 跳过写，仅 rank=0 写，竞争消除。"""

    race_scenario: str = (
        "当 world_size≥2 且 checkpoint_dir 不存在时，rank=0 与 rank=1 "
        "同时调用 os.makedirs()；若二者在同一文件系统节点上，"
        "makedirs 会因目录已存在而抛出 FileExistsError（Python<3.4.1 无 exist_ok），"
        "或发生 .pt 文件互相覆盖的数据竞争，导致 checkpoint 损坏。"
    )

    affected_ranks: str = "rank=1（在 world_size≥2 的任何训练任务中）"
    fix_policy: RankWritePolicy = RankWritePolicy.SOLO_RANK0
    bug_policy: RankWritePolicy = RankWritePolicy.BUG_RANK1_INCLUDED

    def summary(self) -> str:
        return (
            f"[{self.upstream_repo} #{self.commit_index}/{self.total_commits}] "
            f"{self.commit_hash}: {self.commit_subject}\n"
            f"  buggy : {self.buggy_condition!r}\n"
            f"  fixed : {self.fixed_condition!r}\n"
            f"  race  : {self.race_scenario}\n"
            f"  fix_policy : {self.fix_policy.name}\n"
            f"  bug_policy : {self.bug_policy.name}"
        )


_RACE_RECORD = CheckpointRaceRecord()
_dbg("RACE_RECORD_LOAD", f"CheckpointRaceRecord 已构建: {_RACE_RECORD.commit_hash} {_RACE_RECORD.commit_subject}")


# ---------------------------------------------------------------------------
# 4. make_write_guard() —— 工厂函数，对应上游修复后的 if 条件
# ---------------------------------------------------------------------------

def make_write_guard(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    policy: RankWritePolicy = RankWritePolicy.SOLO_RANK0,
) -> DistributedWriteGuard:
    """
    工厂函数：构造 DistributedWriteGuard，自动探测分布式环境。

    如果 torch.distributed 未导入或未初始化，则视为单机场景（rank=0，world_size=1）。
    rank / world_size 参数可在测试中显式传入以脱离真实分布式环境。

    对应上游修复后的逻辑：
      ``if not (torch.distributed.is_initialized() and torch.distributed.get_rank() > 0):``

    Parameters
    ----------
    rank : int, optional
        显式指定 rank（测试用）；None 时从 torch.distributed 自动获取。
    world_size : int, optional
        显式指定 world_size（测试用）；None 时从 torch.distributed 自动获取。
    policy : RankWritePolicy
        写权策略，默认 SOLO_RANK0（修复后行为）。
    """
    _dbg("MAKE_WRITE_GUARD", f"make_write_guard 调用 policy={policy.name} "
         f"explicit_rank={rank} explicit_world_size={world_size}")

    try:
        import torch.distributed as dist  # type: ignore
        dist_init = dist.is_initialized()
    except ImportError:
        dist_init = False

    if rank is None:
        try:
            import torch.distributed as dist  # type: ignore
            rank = dist.get_rank() if dist_init else 0
        except Exception:
            rank = 0

    if world_size is None:
        try:
            import torch.distributed as dist  # type: ignore
            world_size = dist.get_world_size() if dist_init else 1
        except Exception:
            world_size = 1

    guard = DistributedWriteGuard(
        rank=rank,
        world_size=world_size,
        distributed_initialized=dist_init,
        policy=policy,
    )
    _dbg("MAKE_WRITE_GUARD",
         f"guard 构建完成 rank={guard.rank} world_size={guard.world_size} "
         f"dist_init={guard.distributed_initialized}")
    return guard


# ---------------------------------------------------------------------------
# 自检（模块加载时执行）
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """验证修复策略的正确性：SOLO_RANK0 中仅 rank=0 持有写权。"""
    _dbg("SELF_CHECK", "开始自检...")

    # 测试 SOLO_RANK0 策略
    solo = RankWritePolicy.SOLO_RANK0
    assert solo.holds_write(0) is True,  "rank=0 必须持有写权"
    assert solo.holds_write(1) is False, "rank=1 不得持有写权（修复后）"
    assert solo.holds_write(2) is False, "rank=2 不得持有写权"

    # 测试 BUG_RANK1_INCLUDED 策略（记录缺陷行为）
    bug = RankWritePolicy.BUG_RANK1_INCLUDED
    assert bug.holds_write(0) is True,  "rank=0 在缺陷策略下持有写权"
    assert bug.holds_write(1) is True,  "rank=1 在缺陷策略下错误地持有写权"
    assert bug.holds_write(2) is False, "rank=2 在缺陷策略下不持有写权"

    # 测试 make_write_guard（单机路径）
    guard = make_write_guard(rank=0, world_size=1, policy=RankWritePolicy.SOLO_RANK0)
    # 单机下 dist_init=False，必然持有写权
    assert guard.holds_write_lock() is True, "单机路径 rank=0 必须持有写权"

    # 测试 make_write_guard（模拟分布式 rank=1 修复后）
    guard_r1 = DistributedWriteGuard(
        rank=1, world_size=4, distributed_initialized=True,
        policy=RankWritePolicy.SOLO_RANK0,
    )
    assert guard_r1.holds_write_lock() is False, "分布式 rank=1 修复后不得持有写权"

    _dbg("SELF_CHECK", "全部 5 项断言通过 ✓")


_self_check()

_dbg("MODULE_LOAD",
     "save_race_fix_0399d32c7 初始化完成。"
     "使用 make_write_guard() 获取写权判断实例；"
     "CheckpointRaceRecord.summary() 查阅变更上下文。")
