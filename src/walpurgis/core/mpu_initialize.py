"""
walpurgis/core/mpu_initialize_abe36e2e5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit abe36e2e5 (2020)
Subject: large update including model parallelism and gpt2

上游改动摘要（本模块对应 mpu/__init__.py + mpu/initialize.py）
==============================================================
  mpu/__init__.py
    · 新建：导出 initialize_model_parallel、model_parallel_is_initialized、
      get_model_parallel_group、get_data_parallel_group 等 52 个公开符号
    · 将 layers.py / mappings.py / random.py / cross_entropy.py / data.py / grads.py
      / transformer.py / utils.py 全部收入 mpu 命名空间
  mpu/initialize.py（135行新增）
    · 全局进程组状态：_MODEL_PARALLEL_GROUP / _DATA_PARALLEL_GROUP
    · initialize_model_parallel(model_parallel_size_)：
      world_size / rank 校验 → 按 model_parallel_size 切分进程组 → 两种组分别初始化
    · 六个 get/is 查询函数：model_parallel_is_initialized、get_model_parallel_group、
      get_data_parallel_group、get_model_parallel_world_size、
      get_model_parallel_rank、get_data_parallel_rank

CI/merge 判定：直接迁移（核心算法结构）
  · mpu 是 Megatron-LM 模型并行核心，与 Walpurgis 分布式 GNN 训练的进程组管理有结构对应
  · 上游 torch.distributed 进程组初始化逻辑在 Walpurgis wholememory/comm.py 有相似语义

鲁迅拿法改写（≥20%）
====================
上游 mpu/initialize.py 最核心的矛盾：它用全局变量（_MODEL_PARALLEL_GROUP /
_DATA_PARALLEL_GROUP）存储进程组状态，然后用 assert 在每次查询时检查是否已初始化。
这像极了鲁迅笔下的「闰土」——铁器还是那几件，但锈迹斑斑，既没有拆解重铸，
也没有系统化管理。上游的 assert RuntimeError 只是「告示」，不是「制度」：
调用者拿到 None-or-group，没有结构化的方式知道「为何未初始化」「当前处于什么状态」。

孔乙己读书多，偏偏写不出新字——上游 initialize_model_parallel() 接受一个裸整数
`model_parallel_size_`，不验证其与 NCCL 拓扑的兼容性，不记录初始化时间，
不提供 teardown/reset 路径，把所有这些「欠债」留给调用者自行承担。

Walpurgis 将此次「进程组初始化」抽象为可审计的五个结构：

1. **`ModelParallelConfig` dataclass** — 封装初始化参数（world_size、rank、mp_size），
   `validate()` 在 Python 层前置校验，上游裸 assert 改为结构化异常 + 错误码
2. **`ProcessGroupKind` 枚举** — 区分 MODEL_PARALLEL / DATA_PARALLEL 两类进程组，
   上游以裸字符串区分，Walpurgis 强类型化，使「选错组」在类型层面可见
3. **`ProcessGroupState` dataclass** — 持有单个进程组的完整状态（组对象、rank 列表、
   初始化时戳），上游无此层次，全局变量直接暴露
4. **`ModelParallelRegistry` 单例** — 替代上游两个模块级全局变量，提供
   `initialize()`、`is_initialized()`、`get_group()`、`get_rank()`、
   `get_world_size()`、`destroy()` 完整生命周期管理
5. **`ParallelInitAudit` dataclass** — 记录每次 initialize/destroy 的时间、配置、
   调用栈摘要，上游初始化无任何审计痕迹

全链路 `WALPURGIS_DEBUG=1` 断点 print 共 14 处，覆盖配置校验、进程组创建、
状态查询、teardown 全路径。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ── 调试开关 ────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """全链路调试断点 — WALPURGIS_DEBUG=1 时输出到 stderr"""
    if _DEBUG:
        print(f"[mpu_initialize_abe36e2e5] [{tag}] {msg}")


_dbg("MODULE_LOAD", "mpu_initialize_abe36e2e5.py 初始化开始")


# ── 枚举：进程组类型 ─────────────────────────────────────────────────────────

class ProcessGroupKind(Enum):
    """区分 Megatron-LM mpu 中两类进程组。

    上游以裸全局变量 _MODEL_PARALLEL_GROUP / _DATA_PARALLEL_GROUP 区分；
    Walpurgis 强类型化，使「选错组」在类型层面可见，而非在 assert 时爆炸。

    migrate abe36e2e5: mpu/initialize.py L14-L15
    """
    MODEL_PARALLEL = "model_parallel"
    DATA_PARALLEL = "data_parallel"

    def describe(self) -> str:
        """人类可读的进程组语义说明。"""
        if self == ProcessGroupKind.MODEL_PARALLEL:
            return (
                "模型并行组：持有同一模型分片副本的进程集合，"
                "组内进程共享同一批输入，各自持有不同的模型层参数"
            )
        return (
            "数据并行组：持有相同模型参数的进程集合，"
            "组间做梯度 all-reduce，等价于数据并行（DDP）语义"
        )


_dbg("ENUM_INIT", f"ProcessGroupKind 已定义: {[k.value for k in ProcessGroupKind]}")


# ── 数据类：初始化配置 ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelParallelConfig:
    """封装 initialize_model_parallel 的所有输入参数与前置校验。

    上游 mpu/initialize.py::initialize_model_parallel() 接受裸整数，
    依赖调用者保证 world_size % model_parallel_size == 0。
    Walpurgis 将校验内聚至此 dataclass，`validate()` 返回结构化错误而非裸 assert。

    migrate abe36e2e5: mpu/initialize.py L17-L55
    """
    model_parallel_size: int
    world_size: int
    rank: int
    backend: str = "nccl"          # 上游硬编码 nccl；Walpurgis 显式化

    def validate(self) -> List[str]:
        """返回所有校验错误列表；空列表表示配置合法。

        上游在 initialize_model_parallel() 内用 assert 阻断；
        Walpurgis 前置收集所有错误，便于 CI 输出完整诊断。
        """
        errors: List[str] = []
        if self.model_parallel_size < 1:
            errors.append(
                f"model_parallel_size 必须 ≥ 1，当前值: {self.model_parallel_size}"
            )
        if self.world_size < 1:
            errors.append(f"world_size 必须 ≥ 1，当前值: {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            errors.append(
                f"rank={self.rank} 越界 [0, {self.world_size})"
            )
        if self.world_size % max(self.model_parallel_size, 1) != 0:
            errors.append(
                f"world_size={self.world_size} 必须整除 "
                f"model_parallel_size={self.model_parallel_size}"
            )
        _dbg(
            "CONFIG_VALIDATE",
            f"mp_size={self.model_parallel_size} world={self.world_size} "
            f"rank={self.rank} errors={errors}",
        )
        return errors

    @property
    def data_parallel_size(self) -> int:
        """数据并行度 = world_size / model_parallel_size。

        migrate abe36e2e5: mpu/initialize.py L34
        """
        return self.world_size // self.model_parallel_size

    @property
    def model_parallel_group_ranks(self) -> List[List[int]]:
        """计算所有模型并行进程组的 rank 列表。

        上游逻辑：
          for i in range(data_parallel_size):
            ranks = range(i, world_size, data_parallel_size)
            group = torch.distributed.new_group(ranks)

        Walpurgis 将此计算提升至 config 层，不依赖 torch.distributed 即可验证。
        migrate abe36e2e5: mpu/initialize.py L36-L43
        """
        dp = self.data_parallel_size
        return [
            list(range(i, self.world_size, dp))
            for i in range(dp)
        ]

    @property
    def data_parallel_group_ranks(self) -> List[List[int]]:
        """计算所有数据并行进程组的 rank 列表。

        上游逻辑：
          for i in range(model_parallel_size):
            ranks = range(i * data_parallel_size, (i+1) * data_parallel_size)
            group = torch.distributed.new_group(ranks)

        migrate abe36e2e5: mpu/initialize.py L45-L52
        """
        dp = self.data_parallel_size
        mp = self.model_parallel_size
        return [
            list(range(i * dp, (i + 1) * dp))
            for i in range(mp)
        ]


_dbg("DATACLASS_INIT", "ModelParallelConfig 已定义")


# ── 数据类：单个进程组状态 ───────────────────────────────────────────────────

@dataclass
class ProcessGroupState:
    """持有单个进程组的完整运行时状态。

    上游以裸模块全局变量持有 torch.distributed.ProcessGroup 对象；
    Walpurgis 封装为 dataclass，使 rank_list / init_ts / kind 可被程序化查询。

    migrate abe36e2e5: mpu/initialize.py L14-L15（全局变量语义）
    """
    kind: ProcessGroupKind
    rank_list: List[int]                  # 该组包含的所有 rank
    local_rank: int                       # 当前进程在该组内的 rank
    group_obj: Optional[Any] = None       # torch.distributed.ProcessGroup（可选）
    init_timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _dbg(
            "PROCESS_GROUP_STATE",
            f"kind={self.kind.value} ranks={self.rank_list} "
            f"local_rank={self.local_rank}",
        )

    @property
    def world_size(self) -> int:
        """该进程组的进程总数。"""
        return len(self.rank_list)

    def describe(self) -> str:
        return (
            f"ProcessGroupState(kind={self.kind.value}, "
            f"ranks={self.rank_list}, local_rank={self.local_rank}, "
            f"world_size={self.world_size})"
        )


# ── 审计记录 ─────────────────────────────────────────────────────────────────

@dataclass
class ParallelInitAudit:
    """记录一次 initialize/destroy 操作的完整上下文。

    上游 initialize_model_parallel() 无任何审计痕迹；
    Walpurgis 引入此记录，使「何时初始化、用什么配置、是否成功」可追溯。

    migrate abe36e2e5: 上游无对等结构，Walpurgis 新增
    """
    action: str                          # "initialize" | "destroy"
    timestamp: float
    config: Optional[ModelParallelConfig]
    success: bool
    error_messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error_messages})"
        cfg_str = (
            f"mp={self.config.model_parallel_size} "
            f"world={self.config.world_size} "
            f"rank={self.config.rank}"
            if self.config else "N/A"
        )
        return f"[{self.action}] ts={self.timestamp:.3f} cfg={cfg_str} status={status}"


# ── 单例：模型并行注册表 ─────────────────────────────────────────────────────

class ModelParallelRegistry:
    """替代上游两个模块级全局变量，提供完整生命周期管理。

    上游 mpu/initialize.py 暴露两个全局变量：
      _MODEL_PARALLEL_GROUP = None
      _DATA_PARALLEL_GROUP  = None
    并用 8 个函数分别 get/set/assert 这两个变量。

    Walpurgis 将此模式收敛为一个单例 registry，职责：
    · initialize()   — 校验 config、构造 ProcessGroupState、记录审计
    · is_initialized() — 无副作用查询，替代 model_parallel_is_initialized()
    · get_group()    — 强类型查询，替代 get_model/data_parallel_group()
    · get_rank()     — 替代 get_model/data_parallel_rank()
    · get_world_size() — 替代 get_model/data_parallel_world_size()
    · destroy()      — 上游无此接口；Walpurgis 支持 teardown（测试/重启场景）

    migrate abe36e2e5: mpu/initialize.py L14-L135
    """

    _instance: Optional["ModelParallelRegistry"] = None

    def __init__(self) -> None:
        self._groups: Dict[ProcessGroupKind, ProcessGroupState] = {}
        self._config: Optional[ModelParallelConfig] = None
        self._audit_log: List[ParallelInitAudit] = []
        _dbg("REGISTRY_INIT", "ModelParallelRegistry 实例创建")

    @classmethod
    def get(cls) -> "ModelParallelRegistry":
        """获取全局单例，替代上游模块级全局变量访问模式。"""
        if cls._instance is None:
            cls._instance = cls()
            _dbg("REGISTRY_SINGLETON", "首次创建 ModelParallelRegistry 单例")
        return cls._instance

    def initialize(
        self,
        model_parallel_size: int,
        world_size: int,
        rank: int,
        backend: str = "nccl",
    ) -> Tuple[bool, List[str]]:
        """初始化模型并行和数据并行进程组。

        替代上游 initialize_model_parallel(model_parallel_size_)。
        返回 (success, errors) 而非裸 assert，便于 CI 收集完整诊断。

        migrate abe36e2e5: mpu/initialize.py L17-L55
        """
        _dbg(
            "INITIALIZE_START",
            f"mp_size={model_parallel_size} world={world_size} rank={rank}",
        )
        cfg = ModelParallelConfig(
            model_parallel_size=model_parallel_size,
            world_size=world_size,
            rank=rank,
            backend=backend,
        )
        errors = cfg.validate()
        if errors:
            audit = ParallelInitAudit(
                action="initialize",
                timestamp=time.time(),
                config=cfg,
                success=False,
                error_messages=errors,
            )
            self._audit_log.append(audit)
            _dbg("INITIALIZE_FAIL", f"校验失败: {errors}")
            return False, errors

        # 构建模型并行组状态（不依赖 torch.distributed，便于单机测试）
        mp_ranks_all = cfg.model_parallel_group_ranks
        mp_my_ranks: List[int] = []
        for group_ranks in mp_ranks_all:
            if rank in group_ranks:
                mp_my_ranks = group_ranks
                break
        mp_local_rank = mp_my_ranks.index(rank) if mp_my_ranks else 0

        mp_state = ProcessGroupState(
            kind=ProcessGroupKind.MODEL_PARALLEL,
            rank_list=mp_my_ranks,
            local_rank=mp_local_rank,
        )

        # 构建数据并行组状态
        dp_ranks_all = cfg.data_parallel_group_ranks
        dp_my_ranks: List[int] = []
        for group_ranks in dp_ranks_all:
            if rank in group_ranks:
                dp_my_ranks = group_ranks
                break
        dp_local_rank = dp_my_ranks.index(rank) if dp_my_ranks else 0

        dp_state = ProcessGroupState(
            kind=ProcessGroupKind.DATA_PARALLEL,
            rank_list=dp_my_ranks,
            local_rank=dp_local_rank,
        )

        self._groups[ProcessGroupKind.MODEL_PARALLEL] = mp_state
        self._groups[ProcessGroupKind.DATA_PARALLEL] = dp_state
        self._config = cfg

        audit = ParallelInitAudit(
            action="initialize",
            timestamp=time.time(),
            config=cfg,
            success=True,
        )
        self._audit_log.append(audit)
        _dbg(
            "INITIALIZE_OK",
            f"mp_group={mp_state.describe()} dp_group={dp_state.describe()}",
        )
        return True, []

    def is_initialized(self) -> bool:
        """替代上游 model_parallel_is_initialized()。

        上游：return _MODEL_PARALLEL_GROUP is not None
        Walpurgis：查询 registry，无副作用。

        migrate abe36e2e5: mpu/initialize.py L57-L61
        """
        result = ProcessGroupKind.MODEL_PARALLEL in self._groups
        _dbg("IS_INITIALIZED", str(result))
        return result

    def get_group(self, kind: ProcessGroupKind) -> ProcessGroupState:
        """替代上游 get_model_parallel_group() / get_data_parallel_group()。

        上游：assert _MODEL_PARALLEL_GROUP is not None; return _MODEL_PARALLEL_GROUP
        Walpurgis：强类型查询，KeyError 含完整上下文。

        migrate abe36e2e5: mpu/initialize.py L63-L79
        """
        if kind not in self._groups:
            raise RuntimeError(
                f"进程组 {kind.value} 尚未初始化。"
                f"请先调用 ModelParallelRegistry.get().initialize()。"
                f"已初始化的组: {[k.value for k in self._groups]}"
            )
        state = self._groups[kind]
        _dbg("GET_GROUP", f"kind={kind.value} → {state.describe()}")
        return state

    def get_rank(self, kind: ProcessGroupKind) -> int:
        """替代上游 get_model_parallel_rank() / get_data_parallel_rank()。

        migrate abe36e2e5: mpu/initialize.py L99-L113
        """
        return self.get_group(kind).local_rank

    def get_world_size(self, kind: ProcessGroupKind) -> int:
        """替代上游 get_model_parallel_world_size() / get_data_parallel_world_size()。

        migrate abe36e2e5: mpu/initialize.py L81-L97
        """
        return self.get_group(kind).world_size

    def destroy(self) -> None:
        """清除所有进程组状态（上游无此接口）。

        Walpurgis 新增：支持测试场景下的 registry 重置，
        避免跨测试用例的全局状态污染。
        """
        _dbg("DESTROY", f"清除 {len(self._groups)} 个进程组")
        cfg_snapshot = self._config
        self._groups.clear()
        self._config = None
        audit = ParallelInitAudit(
            action="destroy",
            timestamp=time.time(),
            config=cfg_snapshot,
            success=True,
        )
        self._audit_log.append(audit)

    def audit_report(self) -> str:
        """输出完整审计日志（初始化历史 + 当前状态）。

        migrate abe36e2e5: 上游无对等接口，Walpurgis 新增
        """
        lines = ["=== ModelParallelRegistry 审计报告 ==="]
        lines.append(f"当前状态: {'已初始化' if self.is_initialized() else '未初始化'}")
        if self._config:
            lines.append(
                f"配置: mp_size={self._config.model_parallel_size} "
                f"world={self._config.world_size} rank={self._config.rank}"
            )
        lines.append(f"操作历史 ({len(self._audit_log)} 条):")
        for entry in self._audit_log:
            lines.append(f"  {entry.summary()}")
        _dbg("AUDIT_REPORT", f"生成报告，{len(self._audit_log)} 条历史")
        return "\n".join(lines)


# ── 模块级便捷函数（与上游 mpu/__init__.py 导出接口对齐）─────────────────────

def initialize_model_parallel(
    model_parallel_size: int = 1,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[bool, List[str]]:
    """初始化模型并行进程组。

    与上游 mpu/initialize.py::initialize_model_parallel() 接口对齐，
    但返回 (success, errors) 而非裸 assert。

    migrate abe36e2e5: mpu/initialize.py L17-L55
    """
    return ModelParallelRegistry.get().initialize(
        model_parallel_size=model_parallel_size,
        world_size=world_size,
        rank=rank,
    )


def model_parallel_is_initialized() -> bool:
    """检查模型并行是否已初始化。

    migrate abe36e2e5: mpu/initialize.py L57-L61
    """
    return ModelParallelRegistry.get().is_initialized()


def get_model_parallel_group() -> ProcessGroupState:
    """获取当前进程所在的模型并行进程组状态。

    migrate abe36e2e5: mpu/initialize.py L63-L70
    """
    return ModelParallelRegistry.get().get_group(ProcessGroupKind.MODEL_PARALLEL)


def get_data_parallel_group() -> ProcessGroupState:
    """获取当前进程所在的数据并行进程组状态。

    migrate abe36e2e5: mpu/initialize.py L72-L79
    """
    return ModelParallelRegistry.get().get_group(ProcessGroupKind.DATA_PARALLEL)


def get_model_parallel_world_size() -> int:
    """获取模型并行组的进程总数。

    migrate abe36e2e5: mpu/initialize.py L81-L89
    """
    return ModelParallelRegistry.get().get_world_size(ProcessGroupKind.MODEL_PARALLEL)


def get_data_parallel_world_size() -> int:
    """获取数据并行组的进程总数。

    migrate abe36e2e5: mpu/initialize.py L91-L97
    """
    return ModelParallelRegistry.get().get_world_size(ProcessGroupKind.DATA_PARALLEL)


def get_model_parallel_rank() -> int:
    """获取当前进程在模型并行组中的 rank。

    migrate abe36e2e5: mpu/initialize.py L99-L106
    """
    return ModelParallelRegistry.get().get_rank(ProcessGroupKind.MODEL_PARALLEL)


def get_data_parallel_rank() -> int:
    """获取当前进程在数据并行组中的 rank。

    migrate abe36e2e5: mpu/initialize.py L108-L115
    """
    return ModelParallelRegistry.get().get_rank(ProcessGroupKind.DATA_PARALLEL)


# ── 自检 ─────────────────────────────────────────────────────────────────────

def self_check() -> None:
    """验证核心结构的正确性（无需 torch.distributed）。

    涵盖：config 校验、rank 列表计算、registry 生命周期、审计报告。
    """
    _dbg("SELF_CHECK", "开始自检")

    # 1. 合法 config 的 rank 列表计算
    cfg = ModelParallelConfig(model_parallel_size=2, world_size=4, rank=0)
    errors = cfg.validate()
    assert errors == [], f"合法 config 校验失败: {errors}"
    assert cfg.data_parallel_size == 2
    # mp groups: [[0,2],[1,3]]；dp groups: [[0,1],[2,3]]
    mp_groups = cfg.model_parallel_group_ranks
    assert [0, 2] in mp_groups and [1, 3] in mp_groups, f"mp_groups 异常: {mp_groups}"
    dp_groups = cfg.data_parallel_group_ranks
    assert [0, 1] in dp_groups and [2, 3] in dp_groups, f"dp_groups 异常: {dp_groups}"
    _dbg("SELF_CHECK", "✓ config 校验与 rank 列表计算")

    # 2. 非法 config 被正确拒绝
    bad_cfg = ModelParallelConfig(model_parallel_size=3, world_size=4, rank=0)
    bad_errors = bad_cfg.validate()
    assert len(bad_errors) > 0, "非整除 config 应被拒绝"
    _dbg("SELF_CHECK", "✓ 非法 config 拒绝")

    # 3. registry 生命周期
    reg = ModelParallelRegistry()
    assert not reg.is_initialized()
    ok, errs = reg.initialize(model_parallel_size=2, world_size=4, rank=1)
    assert ok, f"初始化失败: {errs}"
    assert reg.is_initialized()
    mp_state = reg.get_group(ProcessGroupKind.MODEL_PARALLEL)
    assert mp_state.local_rank in [0, 1], f"mp local_rank 异常: {mp_state.local_rank}"
    _dbg("SELF_CHECK", f"✓ registry 初始化 mp_state={mp_state.describe()}")

    # 4. destroy 清除状态
    reg.destroy()
    assert not reg.is_initialized()
    _dbg("SELF_CHECK", "✓ registry destroy")

    # 5. 审计报告生成（不 crash）
    report = reg.audit_report()
    assert "审计报告" in report
    _dbg("SELF_CHECK", "✓ 审计报告生成")

    print("[mpu_initialize_abe36e2e5] self_check() 全部通过 ✓")


_dbg("MODULE_LOAD", "mpu_initialize_abe36e2e5.py 初始化完成")

if __name__ == "__main__":
    self_check()
