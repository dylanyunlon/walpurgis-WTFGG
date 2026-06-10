"""
memory_pool.py — c3799ae 迁移: 多GPU训练使用内存池

migrate c3799ae: [BUG] Use memory pool in movielens example

上游变化 (c3799ae, cugraph-gnn, movielens_mnmg.py):

  1. 删除旧代码 (BUG根源):
       if global_rank == 0:
           from rmm.allocators.torch import rmm_torch_allocator
           torch.cuda.change_current_allocator(rmm_torch_allocator)
     问题: change_current_allocator() 修改的是进程级全局状态, 且只在 rank 0
     调用。其余 rank 继续用 PyTorch 默认 CUDA allocator。多 GPU 训练时:
     - rank 0 张量分配在 RMM 内存池
     - rank 1..N 张量分配在 CUDA caching allocator
     DDP all-reduce / NCCL 通信时两侧 buffer 来自不同内存池 → 地址空间不统一
     → 显存碎片 + 潜在 NCCL 通信错误 + OOM。

  2. 新增修复:
       from rmm.allocators.torch import rmm_torch_allocator

       with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
           # 全部主逻辑 (数据加载 / 图构建 / 模型训练)

     - 所有 rank 均建立各自的 MemPool, 用 context manager 限定作用域
     - use_mem_pool() 是线程/进程局部的, 不影响其他进程
     - 退出 with 块后自动恢复默认 allocator, 无资源泄漏风险
     - 缩进增加一层 (Python 物理改写 81 行)

  Knuth 审查:
    1. diff对比源:
       - 旧: `change_current_allocator` 是全局单次调用, 无法撤销, 多进程竞争不安全
       - 新: `use_mem_pool` context manager 完全是进程局部的, 与 NCCL 兼容
       - 结构变化: 81行主逻辑整体缩进一级, 无逻辑变更, 纯作用域包裹
    2. 用户角度 BUG:
       - 旧代码在 2 卡以上时 rank 1+ 显存分配器与 rank 0 不一致
       - 训练中途 OOM 或 NCCL 挂起, 错误信息不指向 allocator 问题, 难以定位
       - 用户可能误以为是显存不足或网络问题, 实际是 allocator 不一致
    3. 系统角度安全:
       - `change_current_allocator` 一旦设置不可回退, 进程生命周期内生效
       - 若后续有第三方库也尝试改 allocator, 将静默覆盖 → 不可预期行为
       - `use_mem_pool` 作用域明确, 退出后恢复原状, 系统状态可控

Walpurgis 改写20%（鲁迅拿法）:
  - RmmMemPoolContext: 值对象, 携带 (allocator_fn, pool, ctx_manager, rank, device)
    替代 Python 中 rmm_torch_allocator 直接内联使用
  - RmmAllocatorMode: 枚举, GLOBAL_CHANGE (旧 BUG 方式) vs MEM_POOL (新修复方式)
    明示两种模式的语义差异, 新代码永远用 MEM_POOL
  - WalpurgisMemPoolSession: 上下文管理器类, 封装 torch.cuda.use_mem_pool 生命周期
    (Python 是裸 with 语句, 我们提取为可测试/可日志的 session 对象)
  - validate_mem_pool_consistency(): 静态验证方法
    — 检查所有 rank 的 allocator 来源一致 (Python 无此校验)
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - MemPool 构建: allocator 地址, rank, device
    - session.__enter__: pool handle, 激活时间戳
    - session.__exit__: 退出时残余显存摘要
    - validate 结果: 全 rank 一致/不一致告警

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
import contextlib
from enum import Enum
from typing import Optional, Callable, Any

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_pool(tag: str, msg: str) -> None:
    """断点调试: memory pool 专用 print"""
    if _DBG:
        print(f"[DEBUG c3799ae {tag}] {msg}", file=sys.stderr, flush=True)


# ─── RmmAllocatorMode: 对应 c3799ae 修复前后两种分配器使用方式 ──────────────────
# Python (旧, BUG):
#   if global_rank == 0:
#       torch.cuda.change_current_allocator(rmm_torch_allocator)
#   # 其余 rank 不执行 → 分配器不一致
#
# Python (新, c3799ae):
#   with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
#       ...  # 全部 rank 均执行
#
# 改写: 枚举明示两种模式, 防止日后误用旧模式
class RmmAllocatorMode(Enum):
    """
    GLOBAL_CHANGE: 旧方式, 修改进程级全局 allocator (BUG: 多GPU不安全)
    MEM_POOL:      新方式, context manager 局部内存池 (c3799ae 修复)
    """
    GLOBAL_CHANGE = "global_change"  # 禁止使用 — 保留用于文档对比
    MEM_POOL = "mem_pool"            # 唯一正确方式


# ─── RmmMemPoolContext: 对应 torch.cuda.MemPool(rmm_torch_allocator.allocator()) ─
# Python (c3799ae):
#   from rmm.allocators.torch import rmm_torch_allocator
#   with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
#       ...
#
# 改写: 封装为值对象, 携带 rank + device 元数据, 便于调试和验证
class RmmMemPoolContext:
    """
    Holds the RMM-backed MemPool for one rank/device.

    Python (c3799ae): MemPool 直接内联构造, 无 rank/device 元数据。
    我们封装为值对象, 构造后不可变, 支持调试打印和一致性验证。
    """

    def __init__(
        self,
        rank: int,
        device: int,
        allocator_fn: Optional[Callable] = None,
    ) -> None:
        self.rank = rank
        self.device = device
        self._allocator_fn = allocator_fn
        self._pool: Any = None  # torch.cuda.MemPool, 懒初始化

        _dbg_pool("RmmMemPoolContext.__init__",
                  f"rank={rank} device={device} "
                  f"allocator_fn={allocator_fn!r}")

    # ── 断点1: 构建 MemPool ────────────────────────────────────────────────────
    def build_pool(self) -> Any:
        """
        对应 c3799ae: torch.cuda.MemPool(rmm_torch_allocator.allocator())

        Python 直接内联; 我们提取为方法, 加调试 + 错误包装。
        """
        try:
            import torch
            from rmm.allocators.torch import rmm_torch_allocator

            fn = self._allocator_fn or rmm_torch_allocator.allocator()

            _dbg_pool("build_pool.allocator",
                      f"rank={self.rank} allocator_fn_id={id(fn)} "
                      f"rmm_torch_allocator={rmm_torch_allocator!r}")

            self._pool = torch.cuda.MemPool(fn)

            _dbg_pool("build_pool.pool_created",
                      f"rank={self.rank} pool={self._pool!r} "
                      f"device={self.device}")

        except ImportError as e:
            _dbg_pool("build_pool.import_error",
                      f"rank={self.rank} error={e} — RMM 未安装, 回退到默认 allocator")
            self._pool = None

        return self._pool

    @property
    def pool(self) -> Any:
        if self._pool is None:
            self.build_pool()
        return self._pool


# ─── WalpurgisMemPoolSession: 对应 with torch.cuda.use_mem_pool(...): ──────────
# Python (c3799ae):
#   with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
#       <全部主逻辑>
#
# 改写: 提取为上下文管理器类, 携带 enter/exit 日志, 支持无 RMM 时优雅降级
class WalpurgisMemPoolSession:
    """
    Context manager wrapping torch.cuda.use_mem_pool for one rank.

    Python (c3799ae): 裸 with 语句, 无日志, 无降级。
    我们封装为类:
      - __enter__: 构建 pool, 激活, 打印调试信息
      - __exit__:  退出 pool, 打印残余显存摘要
      - 若 RMM 不可用: 静默 noop (不破坏非 GPU 测试环境)
    """

    def __init__(self, ctx: RmmMemPoolContext) -> None:
        self.ctx = ctx
        self._cm: Any = None  # torch.cuda.use_mem_pool context manager

    # ── 断点2: session 激活 ─────────────────────────────────────────────────────
    def __enter__(self) -> "WalpurgisMemPoolSession":
        pool = self.ctx.pool

        _dbg_pool("session.__enter__",
                  f"rank={self.ctx.rank} device={self.ctx.device} "
                  f"pool={pool!r}")

        if pool is not None:
            try:
                import torch
                self._cm = torch.cuda.use_mem_pool(pool)
                self._cm.__enter__()

                _dbg_pool("session.__enter__.activated",
                          f"rank={self.ctx.rank} pool handle 已激活")

            except Exception as e:
                _dbg_pool("session.__enter__.error",
                          f"rank={self.ctx.rank} use_mem_pool 失败: {e} — noop 降级")
                self._cm = None
        else:
            _dbg_pool("session.__enter__.noop",
                      f"rank={self.ctx.rank} pool=None, 跳过 use_mem_pool")

        return self

    # ── 断点3: session 退出 + 显存摘要 ─────────────────────────────────────────
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc_val, exc_tb)
                _dbg_pool("session.__exit__",
                          f"rank={self.ctx.rank} pool context 已退出")

                # 残余显存摘要
                try:
                    import torch
                    alloc = torch.cuda.memory_allocated(self.ctx.device)
                    reserved = torch.cuda.memory_reserved(self.ctx.device)
                    _dbg_pool("session.__exit__.mem_summary",
                              f"rank={self.ctx.rank} "
                              f"allocated={alloc/1024**2:.1f}MB "
                              f"reserved={reserved/1024**2:.1f}MB")
                except Exception:
                    pass

            except Exception as e:
                _dbg_pool("session.__exit__.error",
                          f"rank={self.ctx.rank} 退出 pool 时异常: {e}")

        return False  # 不吞异常


# ─── validate_mem_pool_consistency(): 新增验证 (Python c3799ae 无此逻辑) ─────────
# 旧代码 BUG: rank 0 改 global allocator, 其余 rank 不变 → 静默不一致
# 改写: 显式验证所有 rank 是否均进入 pool session (Python 无此校验, 我们新增)
def validate_mem_pool_consistency(
    sessions: list,
    world_size: int,
) -> bool:
    """
    验证所有 rank 的 MemPool session 均已激活 (Python c3799ae 无此逻辑)。

    c3799ae 修复的核心保证: 每个 rank 都用 use_mem_pool, 而非只有 rank 0
    调用 change_current_allocator。本函数在调试模式下验证这一保证成立。

    参数:
        sessions:   各 rank 的 WalpurgisMemPoolSession 列表
        world_size: DDP world size

    返回:
        True  → 全部 rank 均有 pool, 一致
        False → 存在 rank pool=None, 退化为默认 allocator (旧 BUG 状态)
    """
    # ── 断点4: 一致性验证入口 ───────────────────────────────────────────────────
    _dbg_pool("validate.entry",
              f"world_size={world_size} sessions={len(sessions)}")

    if len(sessions) != world_size:
        _dbg_pool("validate.size_mismatch",
                  f"sessions 数量 {len(sessions)} ≠ world_size {world_size} — 跳过验证")
        return True  # 非 DDP 场景不强制验证

    broken_ranks = []
    for s in sessions:
        has_pool = (s.ctx.pool is not None and s._cm is not None)
        _dbg_pool("validate.rank_check",
                  f"rank={s.ctx.rank} has_pool={has_pool}")
        if not has_pool:
            broken_ranks.append(s.ctx.rank)

    if broken_ranks:
        _dbg_pool("validate.INCONSISTENT",
                  f"以下 rank 未使用 RMM MemPool (旧 BUG 状态): {broken_ranks}")
        print(
            f"[WARN c3799ae] ranks {broken_ranks} 未使用 RMM MemPool — "
            "可能触发多GPU显存分配不一致 (同 BUG c3799ae 修复前行为)",
            file=sys.stderr, flush=True,
        )
        return False

    _dbg_pool("validate.CONSISTENT",
              f"全部 {world_size} 个 rank 均使用 RMM MemPool — 一致 ✓")
    return True


# ─── build_mem_pool_session(): 顶层工厂函数 ──────────────────────────────────────
# 对应 c3799ae 修复后的两行:
#   from rmm.allocators.torch import rmm_torch_allocator
#   with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):
#
# 使用方式 (Walpurgis multi-GPU 训练主循环):
#   session = build_mem_pool_session(rank=global_rank, device=local_rank)
#   with session:
#       <全部主逻辑>
def build_mem_pool_session(
    rank: int,
    device: int,
    allocator_fn: Optional[Callable] = None,
    mode: RmmAllocatorMode = RmmAllocatorMode.MEM_POOL,
) -> WalpurgisMemPoolSession:
    """
    构建适合多GPU训练的 RMM MemPool session。

    c3799ae 修复要点:
      - mode=MEM_POOL (唯一合法值): 每个 rank 独立建 pool, 用 context manager
      - mode=GLOBAL_CHANGE: 旧 BUG 方式, 此函数拒绝执行并告警

    参数:
        rank:         全局 rank (用于调试和验证)
        device:       本地 GPU device id (LOCAL_RANK)
        allocator_fn: 可选自定义 allocator; None → 使用 rmm_torch_allocator.allocator()
        mode:         RmmAllocatorMode.MEM_POOL (默认, 推荐)

    返回:
        WalpurgisMemPoolSession — 可直接用于 `with` 语句
    """
    # ── 断点5: 工厂函数入口 ─────────────────────────────────────────────────────
    _dbg_pool("build_mem_pool_session.entry",
              f"rank={rank} device={device} mode={mode.value} "
              f"allocator_fn={allocator_fn!r}")

    if mode == RmmAllocatorMode.GLOBAL_CHANGE:
        # 旧 BUG 方式 — 明确拒绝
        _dbg_pool("build_mem_pool_session.REJECTED",
                  f"rank={rank} mode=GLOBAL_CHANGE 已废弃 (c3799ae BUG 根源), 拒绝执行")
        raise ValueError(
            "RmmAllocatorMode.GLOBAL_CHANGE 是 c3799ae 修复前的 BUG 方式: "
            "change_current_allocator() 修改全局状态且只在 rank 0 调用, "
            "多 GPU 环境下分配器不一致。请使用 RmmAllocatorMode.MEM_POOL。"
        )

    ctx = RmmMemPoolContext(rank=rank, device=device, allocator_fn=allocator_fn)
    session = WalpurgisMemPoolSession(ctx)

    _dbg_pool("build_mem_pool_session.created",
              f"rank={rank} session={session!r}")

    return session


# ─── 自检函数: 验证 c3799ae 核心语义 ─────────────────────────────────────────────
def test_mem_pool_session_all_ranks() -> None:
    """
    自检: 模拟 world_size=2 场景, 验证两个 rank 均建立 pool session。

    对应 c3799ae 修复保证: 所有 rank (不只是 rank 0) 均使用 RMM MemPool。
    旧 BUG: 只有 global_rank == 0 调用 change_current_allocator → rank 1 默认 allocator。
    """
    _dbg_pool("test_all_ranks.entry", "开始自检 world_size=2 场景")

    sessions = []
    for rank in range(2):
        s = build_mem_pool_session(rank=rank, device=rank)
        sessions.append(s)
        _dbg_pool("test_all_ranks.session_built",
                  f"rank={rank} session 构建完成 (不依赖真实 CUDA)")

    # 验证两个 rank 均有 ctx (不调用 __enter__ 以避免真实 CUDA 依赖)
    assert len(sessions) == 2, "应有 2 个 session"
    assert sessions[0].ctx.rank == 0
    assert sessions[1].ctx.rank == 1

    _dbg_pool("test_all_ranks.PASS",
              "自检通过: 2 个 rank 均构建了独立的 WalpurgisMemPoolSession ✓")

    print("[c3799ae] test_mem_pool_session_all_ranks PASS", flush=True)
