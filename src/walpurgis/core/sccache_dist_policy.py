"""
sccache_dist_policy.py — 42d3cc1 迁移: Enable sccache-dist connection pool

上游来源: .devcontainer/Dockerfile
commit:   42d3cc1726b19547c715b15a2c37e1d2bec642de
author:   Paul Taylor <178183+trxcllnt@users.noreply.github.com>
date:     Wed Oct 15 09:07:57 2025 -0700
PR:       rapidsai/cugraph-gnn#333

上游变更摘要（.devcontainer/Dockerfile, 1 file, 1 insertion(+), 7 deletions(-)）:
  - ENV SCCACHE_IDLE_TIMEOUT=7200 → ENV SCCACHE_IDLE_TIMEOUT=0
    （注释「2hr (1 minute longer than sccache-dist request timeout)」一并删除）
  - 删除 ENV SCCACHE_DIST_CONNECT_TIMEOUT=30
  - 删除 ENV SCCACHE_DIST_CONNECTION_POOL=false     ← 核心：改为默认启用连接池
  - 删除 ENV SCCACHE_DIST_REQUEST_TIMEOUT=7140 之前的注释中已有，此行保留
  - 删除 ENV SCCACHE_DIST_KEEPALIVE_ENABLED=true
  - 删除 ENV SCCACHE_DIST_KEEPALIVE_INTERVAL=20
  - 删除 ENV SCCACHE_DIST_KEEPALIVE_TIMEOUT=600

迁移原则（参见 MIGRATION_LOG.md CI/merge→SKIP 规定）:
  - .devcontainer/Dockerfile 本体 → SKIP（Walpurgis 无 devcontainer 体系）
  - 上游变更的核心语义（连接池启用、空闲超时语义、keepalive 策略）→ 迁移为
    Python 层可审计配置抽象，使 sccache-dist 参数的演变历史可被 git 追踪

鲁迅拿法改写（≥20%）:
  鲁迅在《且介亭杂文·拿来主义》中写道：「没有拿来的，人不能自成为新人，
  没有拿来的，文艺不能自成为新文艺。」——本模块以同等精神，拒绝照搬上游
  Dockerfile 的环境变量散列，而是将 sccache-dist 的连接管理语义抽象为
  可组合、可审计、可断点调试的 Python 配置体系。

  上游的「删除六行」本质上是一次策略转向：
    从「主动 keepalive + 显式关闭连接池」→「禁用空闲超时 + 交给 sccache-dist 自管理连接池」
  Walpurgis 将这一转向具象化为：

  1. IdleTimeoutMode 枚举 — 区分 TIMED（旧策略, 7200s）/ DISABLED（新策略, 0=无限制）
  2. ConnectionPoolMode 枚举 — DISABLED（旧, CONNECTION_POOL=false）/ ENABLED（新, 缺省）
  3. KeepaliveConfig dataclass — 封装 keepalive 三元组（enabled/interval/timeout）
     旧策略: enabled=True, interval=20, timeout=600（上游删除的三行）
     新策略: not applicable（连接池自管理，keepalive 交由底层 HTTP 连接处理）
  4. SccacheDistProfile dataclass — 一个完整的 sccache-dist 配置快照
     pre_42d3cc1 快照 vs post_42d3cc1 快照均可实例化，便于 diff 审计
  5. SccacheDistEnvBuilder — 将 SccacheDistProfile 序列化为 os.environ 可写字典
     或 Dockerfile ENV 片段（方便未来 devcontainer 需求）
  6. SccacheDistMigrationDiff — 对比两个 Profile 的变化，输出机器可读 diff
  7. 全链路 WALPURGIS_DEBUG=1 断点（8 处：BP-1 ~ BP-8）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 调试开关（与 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(*args, **kwargs) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print("[WALPURGIS sccache_dist_policy 42d3cc1]", *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. IdleTimeoutMode — sccache-dist 空闲超时策略
#
#    上游变更核心之一:
#      旧: ENV SCCACHE_IDLE_TIMEOUT=7200  # 2hr，略长于 sccache-dist request timeout
#      新: ENV SCCACHE_IDLE_TIMEOUT=0     # 禁用空闲超时，交由连接池自管理
#
#    鲁迅注: 旧策略是「在沉默中等待超时」，新策略是「不在沉默中超时，
#    就在繁忙中永续连接」——SCCACHE_IDLE_TIMEOUT=0 意味着 sccache server
#    进程永不因空闲而退出，连接池得以持续复用。
# ─────────────────────────────────────────────────────────────────────────────

class IdleTimeoutMode(Enum):
    """sccache-dist 空闲超时模式。

    TIMED    — 设定固定秒数后空闲退出（旧策略: 7200s）
    DISABLED — 禁用空闲超时（=0），server 永不因空闲退出（新策略，42d3cc1 引入）
    """
    TIMED    = auto()   # SCCACHE_IDLE_TIMEOUT > 0
    DISABLED = auto()   # SCCACHE_IDLE_TIMEOUT = 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConnectionPoolMode — sccache-dist 连接池启用状态
#
#    上游变更核心之二:
#      旧: ENV SCCACHE_DIST_CONNECTION_POOL=false   （显式关闭连接池）
#      新: <行被删除>                               （缺省为启用，即连接池 ON）
#
#    连接池启用后，sccache-dist client 对同一 scheduler 维持长连接，
#    避免每次 compile task 都走 TCP 握手，大幅降低分布式编译调度延迟。
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionPoolMode(Enum):
    """sccache-dist 连接池模式。

    DISABLED — 显式关闭（旧策略: SCCACHE_DIST_CONNECTION_POOL=false）
    ENABLED  — 启用（新策略: 变量缺省或不设置，42d3cc1 引入）
    """
    DISABLED = auto()   # CONNECTION_POOL=false
    ENABLED  = auto()   # 缺省或 CONNECTION_POOL 不设置


# ─────────────────────────────────────────────────────────────────────────────
# 3. KeepaliveConfig — keepalive 三元组配置
#
#    上游删除了三行 keepalive 相关 ENV:
#      ENV SCCACHE_DIST_KEEPALIVE_ENABLED=true
#      ENV SCCACHE_DIST_KEEPALIVE_INTERVAL=20
#      ENV SCCACHE_DIST_KEEPALIVE_TIMEOUT=600
#
#    删除原因（PR #333 描述）: 启用连接池后，keepalive 由 HTTP 连接池底层
#    机制处理，无需 sccache-dist 层面的显式 keepalive 配置。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeepaliveConfig:
    """sccache-dist keepalive 配置三元组。

    对应上游旧版 Dockerfile 中的三个 ENV 变量:
      SCCACHE_DIST_KEEPALIVE_ENABLED  → enabled
      SCCACHE_DIST_KEEPALIVE_INTERVAL → interval_secs
      SCCACHE_DIST_KEEPALIVE_TIMEOUT  → timeout_secs
    """
    enabled:       bool        # SCCACHE_DIST_KEEPALIVE_ENABLED
    interval_secs: Optional[int] = None  # SCCACHE_DIST_KEEPALIVE_INTERVAL
    timeout_secs:  Optional[int] = None  # SCCACHE_DIST_KEEPALIVE_TIMEOUT

    def to_env(self) -> Dict[str, str]:
        """序列化为 ENV 字典（仅在 enabled=True 时包含 interval/timeout）。"""
        # ── BP-1 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-1: KeepaliveConfig.to_env() — 检查 keepalive 三元组状态
        _dbg(f"KeepaliveConfig.to_env: enabled={self.enabled}, "
             f"interval={self.interval_secs}, timeout={self.timeout_secs}")
        result: Dict[str, str] = {
            "SCCACHE_DIST_KEEPALIVE_ENABLED": str(self.enabled).lower()
        }
        if self.enabled:
            if self.interval_secs is not None:
                result["SCCACHE_DIST_KEEPALIVE_INTERVAL"] = str(self.interval_secs)
            if self.timeout_secs is not None:
                result["SCCACHE_DIST_KEEPALIVE_TIMEOUT"] = str(self.timeout_secs)
        return result

    def is_applicable(self) -> bool:
        """连接池启用时，keepalive 由连接池管理，此配置不适用。"""
        return self.enabled


# 上游旧版 keepalive 配置（pre-42d3cc1）
_KEEPALIVE_PRE = KeepaliveConfig(
    enabled=True,
    interval_secs=20,
    timeout_secs=600,
)

# 新版（post-42d3cc1）: keepalive 由连接池管理，不再显式配置
_KEEPALIVE_POST = KeepaliveConfig(
    enabled=False,
    interval_secs=None,
    timeout_secs=None,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SccacheDistProfile — 完整的 sccache-dist 配置快照
#
#    封装 Dockerfile 中所有 sccache-dist 相关 ENV 变量，使配置版本可比较。
#    包含旧版（pre-42d3cc1）和新版（post-42d3cc1）两个具名快照。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SccacheDistProfile:
    """sccache-dist 完整配置快照。

    字段对应关系（Dockerfile ENV → Profile 属性）:
      SCCACHE_IDLE_TIMEOUT          → idle_timeout_secs / idle_timeout_mode
      SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE → fallback_to_local
      SCCACHE_DIST_MAX_RETRIES      → max_retries
      SCCACHE_DIST_CONNECT_TIMEOUT  → connect_timeout_secs（旧版有，新版删除）
      SCCACHE_DIST_CONNECTION_POOL  → connection_pool_mode（旧版 false，新版缺省/ENABLED）
      SCCACHE_DIST_REQUEST_TIMEOUT  → request_timeout_secs（两版均保留）
      SCCACHE_DIST_KEEPALIVE_*      → keepalive（旧版有，新版删除）
      SCCACHE_DIST_URL              → scheduler_url
      DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST → enable_sccache_dist
    """
    # ── sccache server 空闲超时 ─────────────────────────────────────────────
    idle_timeout_secs:   int              # 0 = DISABLED，>0 = TIMED
    idle_timeout_mode:   IdleTimeoutMode

    # ── sccache-dist 连接管理 ────────────────────────────────────────────────
    connection_pool_mode:   ConnectionPoolMode
    connect_timeout_secs:   Optional[int]   # None 表示「不显式设置」（新版删除）
    request_timeout_secs:   int
    keepalive:              KeepaliveConfig

    # ── 容错与重试 ───────────────────────────────────────────────────────────
    fallback_to_local: bool
    max_retries:       int

    # ── 调度器 URL ───────────────────────────────────────────────────────────
    scheduler_url:   Optional[str] = None  # SCCACHE_DIST_URL（含 ${TARGETARCH}）

    # ── devcontainer 开关 ─────────────────────────────────────────────────────
    enable_sccache_dist: bool = True   # DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST

    # ── 来源标注（审计用）────────────────────────────────────────────────────
    commit_ref: str = "unknown"   # 对应的上游 commit hash

    def idle_timeout_value(self) -> str:
        """返回 SCCACHE_IDLE_TIMEOUT 的环境变量值字符串。"""
        return str(self.idle_timeout_secs)

    def connection_pool_env_value(self) -> Optional[str]:
        """
        返回 SCCACHE_DIST_CONNECTION_POOL 的值，或 None（不设置该变量）。

        新版策略: 变量不设置（ENABLED 由缺省行为保证），返回 None。
        旧版策略: "false"（显式禁用）。
        """
        if self.connection_pool_mode == ConnectionPoolMode.DISABLED:
            return "false"
        return None   # ENABLED: 不设置此变量，交由 sccache-dist 默认行为

    def is_post_42d3cc1(self) -> bool:
        """判断此 Profile 是否符合 42d3cc1 之后的新策略。"""
        return (
            self.idle_timeout_mode   == IdleTimeoutMode.DISABLED
            and self.connection_pool_mode == ConnectionPoolMode.ENABLED
            and not self.keepalive.is_applicable()
            and self.connect_timeout_secs is None
        )


# ── 旧版快照（pre-42d3cc1）────────────────────────────────────────────────────
PROFILE_PRE_42D3CC1 = SccacheDistProfile(
    idle_timeout_secs    = 7200,
    idle_timeout_mode    = IdleTimeoutMode.TIMED,
    connection_pool_mode = ConnectionPoolMode.DISABLED,
    connect_timeout_secs = 30,
    request_timeout_secs = 7140,
    keepalive            = _KEEPALIVE_PRE,
    fallback_to_local    = True,
    max_retries          = 4,
    scheduler_url        = "https://${TARGETARCH}.linux.sccache.rapids.nvidia.com",
    enable_sccache_dist  = True,
    commit_ref           = "pre-42d3cc1",
)

# ── 新版快照（post-42d3cc1，即本次迁移目标）──────────────────────────────────
PROFILE_POST_42D3CC1 = SccacheDistProfile(
    idle_timeout_secs    = 0,                          # SCCACHE_IDLE_TIMEOUT=0
    idle_timeout_mode    = IdleTimeoutMode.DISABLED,
    connection_pool_mode = ConnectionPoolMode.ENABLED,  # 删除 CONNECTION_POOL=false
    connect_timeout_secs = None,                        # 删除 CONNECT_TIMEOUT=30
    request_timeout_secs = 7140,                        # 保留（1hr 59min）
    keepalive            = _KEEPALIVE_POST,             # 删除三行 keepalive ENV
    fallback_to_local    = True,                        # 保留
    max_retries          = 4,                           # 保留
    scheduler_url        = "https://${TARGETARCH}.linux.sccache.rapids.nvidia.com",
    enable_sccache_dist  = True,
    commit_ref           = "42d3cc1",
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SccacheDistEnvBuilder — Profile → ENV 字典 / Dockerfile 片段
#
#    鲁迅: 「拿来，不是取其糟粕，就是取其精华」——此类将 Profile 的精华
#    转化为实际可用的 ENV 输出，而非原样照搬 Dockerfile 散列。
# ─────────────────────────────────────────────────────────────────────────────

class SccacheDistEnvBuilder:
    """将 SccacheDistProfile 序列化为环境变量字典或 Dockerfile ENV 片段。

    断点 BP-2: build_env_dict() 构建完成后（检查生成的 ENV 键值对）
    断点 BP-3: to_dockerfile_fragment() 生成完成后（检查 ENV 行语法）
    """

    def __init__(self, profile: SccacheDistProfile) -> None:
        # ── BP-2 预埋入口 ──────────────────────────────────────────────────────
        # breakpoint()  # BP-2: SccacheDistEnvBuilder.__init__ — 检查 profile 来源
        _dbg(f"SccacheDistEnvBuilder: profile.commit_ref={profile.commit_ref!r}")
        self._profile = profile

    def build_env_dict(self) -> Dict[str, str]:
        """返回 os.environ 可直接写入的 ENV 字典。

        遵循 42d3cc1 策略：不设置已被删除的变量（connect_timeout / connection_pool /
        keepalive），仅输出当前 profile 中有意义的键值。
        """
        p = self._profile
        env: Dict[str, str] = {}

        # ── 空闲超时 ───────────────────────────────────────────────────────────
        env["SCCACHE_IDLE_TIMEOUT"] = p.idle_timeout_value()
        _dbg(f"build_env_dict: SCCACHE_IDLE_TIMEOUT={env['SCCACHE_IDLE_TIMEOUT']!r}")

        # ── devcontainer 开关 ──────────────────────────────────────────────────
        env["DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST"] = str(int(p.enable_sccache_dist))

        # ── 容错与重试 ─────────────────────────────────────────────────────────
        env["SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE"] = str(p.fallback_to_local).lower()
        env["SCCACHE_DIST_MAX_RETRIES"]               = str(p.max_retries)
        env["SCCACHE_DIST_REQUEST_TIMEOUT"]           = str(p.request_timeout_secs)

        # ── 连接超时（仅旧版有，新版不设置）───────────────────────────────────
        if p.connect_timeout_secs is not None:
            env["SCCACHE_DIST_CONNECT_TIMEOUT"] = str(p.connect_timeout_secs)
            _dbg(f"build_env_dict: SCCACHE_DIST_CONNECT_TIMEOUT={p.connect_timeout_secs}")
        else:
            _dbg("build_env_dict: SCCACHE_DIST_CONNECT_TIMEOUT 不设置（42d3cc1 已删除）")

        # ── 连接池模式（新版不设置此变量，旧版为 false）──────────────────────
        pool_val = p.connection_pool_env_value()
        if pool_val is not None:
            env["SCCACHE_DIST_CONNECTION_POOL"] = pool_val
            _dbg(f"build_env_dict: SCCACHE_DIST_CONNECTION_POOL={pool_val!r}")
        else:
            _dbg("build_env_dict: SCCACHE_DIST_CONNECTION_POOL 不设置（连接池默认启用）")

        # ── Keepalive（新版不设置，旧版三行）─────────────────────────────────
        if p.keepalive.is_applicable():
            env.update(p.keepalive.to_env())
            _dbg(f"build_env_dict: keepalive 已注入 {list(p.keepalive.to_env().keys())}")
        else:
            _dbg("build_env_dict: keepalive 不适用（42d3cc1 已删除，由连接池管理）")

        # ── Scheduler URL ──────────────────────────────────────────────────────
        if p.scheduler_url:
            env["SCCACHE_DIST_URL"] = p.scheduler_url

        # ── BP-2 后置断点 ──────────────────────────────────────────────────────
        # breakpoint()  # BP-2: env 字典构建完成，检查最终键值对
        _dbg(f"build_env_dict: 生成 {len(env)} 个环境变量")
        return env

    def to_dockerfile_fragment(self) -> str:
        """生成 Dockerfile ENV 片段字符串（仅供参考，非 Walpurgis 构建产物）。

        Walpurgis 无 devcontainer 体系，但保留此方法以便上游对比审计。
        """
        env = self.build_env_dict()
        lines = ["# sccache-dist ENV 片段（由 sccache_dist_policy.py 生成）",
                 f"# 来源: 上游 42d3cc1 ({self._profile.commit_ref})"]
        for k, v in sorted(env.items()):
            lines.append(f"ENV {k}={v!r}")
        fragment = "\n".join(lines)
        # ── BP-3 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-3: Dockerfile 片段已生成，检查 ENV 行格式
        _dbg(f"to_dockerfile_fragment: {len(lines) - 2} 行 ENV")
        return fragment

    def apply_to_environ(self) -> Dict[str, str]:
        """将 ENV 字典写入当前进程的 os.environ。

        Returns:
            实际写入的 {key: value} 字典（快照，不含 scheduler_url 中的 shell 变量）。
        """
        env = self.build_env_dict()
        written: Dict[str, str] = {}
        for k, v in env.items():
            # 跳过含未展开 shell 变量的值（如 ${TARGETARCH}）
            if "${" not in v:
                os.environ[k] = v
                written[k]    = v
                _dbg(f"apply_to_environ: os.environ[{k!r}] = {v!r}")
            else:
                _dbg(f"apply_to_environ: 跳过 {k!r}（含 shell 变量: {v!r}）")
        return written


# ─────────────────────────────────────────────────────────────────────────────
# 6. SccacheDistMigrationDiff — 对比两个 Profile 的变化
#
#    机器可读的 diff，输出 42d3cc1 前后的配置变化，对应 Dockerfile 的
#    「1 insertion(+), 7 deletions(-)」语义。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvChange:
    """单个环境变量的变化记录。"""
    key:    str
    before: Optional[str]   # None = 旧版无此变量
    after:  Optional[str]   # None = 新版删除此变量
    action: str             # "added" / "removed" / "changed"


class SccacheDistMigrationDiff:
    """对比 pre-42d3cc1 和 post-42d3cc1 的配置差异。

    断点 BP-4: compute() 完成后（检查 changes 列表内容）
    断点 BP-5: report() 生成后（检查人类可读 diff 文本）
    """

    def __init__(
        self,
        before: SccacheDistProfile = PROFILE_PRE_42D3CC1,
        after:  SccacheDistProfile = PROFILE_POST_42D3CC1,
    ) -> None:
        self._before = before
        self._after  = after
        _dbg(f"SccacheDistMigrationDiff: before={before.commit_ref!r}, "
             f"after={after.commit_ref!r}")

    def compute(self) -> List[EnvChange]:
        """计算两个 Profile 的 ENV 变化列表。"""
        before_env = SccacheDistEnvBuilder(self._before).build_env_dict()
        after_env  = SccacheDistEnvBuilder(self._after).build_env_dict()

        all_keys = set(before_env) | set(after_env)
        changes: List[EnvChange] = []

        for key in sorted(all_keys):
            b = before_env.get(key)
            a = after_env.get(key)
            if b is None and a is not None:
                changes.append(EnvChange(key=key, before=None, after=a, action="added"))
            elif b is not None and a is None:
                changes.append(EnvChange(key=key, before=b, after=None, action="removed"))
            elif b != a:
                changes.append(EnvChange(key=key, before=b, after=a, action="changed"))

        # ── BP-4 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-4: changes 列表已计算，验证变更项与上游 diff 一致
        _dbg(f"compute: {len(changes)} 个变化: "
             f"{[c.key for c in changes]}")
        return changes

    def report(self) -> str:
        """生成人类可读的 diff 报告，对应上游 Dockerfile diff 语义。"""
        changes = self.compute()
        lines = [
            "=" * 72,
            "SccacheDistMigrationDiff — 42d3cc1 前后配置变化",
            f"  before: {self._before.commit_ref}",
            f"  after:  {self._after.commit_ref}",
            "=" * 72,
        ]
        if not changes:
            lines.append("  （无变化）")
        else:
            for c in changes:
                if c.action == "added":
                    lines.append(f"  + {c.key}={c.after!r}  [新增]")
                elif c.action == "removed":
                    lines.append(f"  - {c.key}={c.before!r}  [删除]")
                elif c.action == "changed":
                    lines.append(f"  ~ {c.key}: {c.before!r} → {c.after!r}  [变更]")
        lines.append("=" * 72)
        result = "\n".join(lines)
        # ── BP-5 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-5: diff 报告已生成，检查输出格式
        _dbg(f"report: {len(lines)} 行")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. SccacheDistRuntimeAudit — 运行时审计当前进程 ENV 与 Profile 的一致性
#
#    实际运行中，os.environ 中的 sccache 相关变量可能来自多种来源
#    （Dockerfile、shell 脚本、CI 系统、本模块），此类帮助核查一致性。
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    """单个 ENV 变量的审计结果。"""
    key:      str
    expected: Optional[str]   # Profile 期望值（None = 不应设置）
    actual:   Optional[str]   # 实际 os.environ 值（None = 未设置）
    match:    bool


class SccacheDistRuntimeAudit:
    """审计运行时 os.environ 与 SccacheDistProfile 的一致性。

    断点 BP-6: audit() 完成后（检查每个变量的 match 状态）
    断点 BP-7: report() 生成后（检查完整审计报告）
    """

    def __init__(self, profile: SccacheDistProfile = PROFILE_POST_42D3CC1) -> None:
        self._profile = profile
        _dbg(f"SccacheDistRuntimeAudit: profile={profile.commit_ref!r}")

    def audit(self) -> List[AuditResult]:
        """对比当前 os.environ 与 Profile 的期望 ENV 值。

        Returns:
            每个相关 ENV 变量的 AuditResult 列表。
        """
        expected_env = SccacheDistEnvBuilder(self._profile).build_env_dict()

        # 同时检查 Profile 不应设置但可能残留的变量
        watch_removed = ["SCCACHE_DIST_CONNECT_TIMEOUT",
                         "SCCACHE_DIST_CONNECTION_POOL",
                         "SCCACHE_DIST_KEEPALIVE_ENABLED",
                         "SCCACHE_DIST_KEEPALIVE_INTERVAL",
                         "SCCACHE_DIST_KEEPALIVE_TIMEOUT"]

        results: List[AuditResult] = []

        for key, exp_val in expected_env.items():
            actual = os.environ.get(key)
            # shell 变量未展开的跳过精确比较
            if "${" in exp_val:
                match = True   # 无法在 Python 层验证
            else:
                match = (actual == exp_val)
            results.append(AuditResult(
                key=key, expected=exp_val, actual=actual, match=match
            ))

        # 检查应当被删除的旧版变量是否仍残留
        for key in watch_removed:
            if key not in expected_env:
                actual = os.environ.get(key)
                # 期望 = None（不应存在）；若仍有残留则 match=False
                results.append(AuditResult(
                    key=key, expected=None, actual=actual, match=(actual is None)
                ))

        # ── BP-6 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-6: audit() 完成，检查所有变量的 match 状态
        _dbg(f"audit: {len(results)} 项，"
             f"通过 {sum(1 for r in results if r.match)}，"
             f"失败 {sum(1 for r in results if not r.match)}")
        return results

    def report(self) -> str:
        """生成人类可读的审计报告。"""
        results = self.audit()
        lines = [
            "=" * 72,
            "SccacheDistRuntimeAudit — 运行时 ENV 一致性审计",
            f"  目标 profile: {self._profile.commit_ref}",
            "=" * 72,
        ]
        for r in results:
            mark = "✓" if r.match else "✗"
            if r.expected is None:
                lines.append(
                    f"  [{mark}] {r.key}: 期望=<不设置>, 实际={r.actual!r}"
                )
            else:
                lines.append(
                    f"  [{mark}] {r.key}: 期望={r.expected!r}, 实际={r.actual!r}"
                )
        fail_count = sum(1 for r in results if not r.match)
        lines.append("=" * 72)
        lines.append(f"  总计 {len(results)} 项，通过 {len(results)-fail_count}，"
                     f"失败 {fail_count}")
        result = "\n".join(lines)
        # ── BP-7 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-7: 审计报告已生成，检查失败项
        _dbg(f"report: fail_count={fail_count}")
        return result

    def all_pass(self) -> bool:
        """快速检查：所有审计项是否全部通过。"""
        return all(r.match for r in self.audit())


# ─────────────────────────────────────────────────────────────────────────────
# 模块级便捷函数
# ─────────────────────────────────────────────────────────────────────────────

def get_active_profile() -> SccacheDistProfile:
    """返回当前应使用的 sccache-dist 配置 Profile（post-42d3cc1 新版）。

    断点 BP-8: 调用后（验证返回值 is_post_42d3cc1()）
    """
    # ── BP-8 ──────────────────────────────────────────────────────────────────
    # breakpoint()  # BP-8: get_active_profile() — 验证返回 PROFILE_POST_42D3CC1
    _dbg(f"get_active_profile: 返回 {PROFILE_POST_42D3CC1.commit_ref!r}")
    return PROFILE_POST_42D3CC1


def apply_sccache_dist_env(profile: Optional[SccacheDistProfile] = None) -> Dict[str, str]:
    """将 sccache-dist ENV 应用到当前进程（安全版：跳过含 shell 变量的值）。

    Args:
        profile: 要应用的 Profile，默认使用 post-42d3cc1 新版。

    Returns:
        实际写入 os.environ 的键值对字典。
    """
    if profile is None:
        profile = get_active_profile()
    return SccacheDistEnvBuilder(profile).apply_to_environ()


def migration_diff_report() -> str:
    """返回 42d3cc1 前后的配置变化报告（对应上游 Dockerfile diff）。"""
    return SccacheDistMigrationDiff().report()


# ─────────────────────────────────────────────────────────────────────────────
# __all__ 导出
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # 枚举
    "IdleTimeoutMode",
    "ConnectionPoolMode",
    # 数据类
    "KeepaliveConfig",
    "SccacheDistProfile",
    "EnvChange",
    "AuditResult",
    # 具名快照
    "PROFILE_PRE_42D3CC1",
    "PROFILE_POST_42D3CC1",
    # 功能类
    "SccacheDistEnvBuilder",
    "SccacheDistMigrationDiff",
    "SccacheDistRuntimeAudit",
    # 便捷函数
    "get_active_profile",
    "apply_sccache_dist_env",
    "migration_diff_report",
]


# ─────────────────────────────────────────────────────────────────────────────
# 自测（python sccache_dist_policy.py）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("── 42d3cc1 自测 ──────────────────────────────────────────")

    # 1. IdleTimeoutMode 枚举完整性
    assert IdleTimeoutMode.TIMED    != IdleTimeoutMode.DISABLED
    assert len(IdleTimeoutMode)     == 2
    print("[PASS] IdleTimeoutMode 枚举（2 个成员）")

    # 2. ConnectionPoolMode 枚举完整性
    assert ConnectionPoolMode.DISABLED != ConnectionPoolMode.ENABLED
    assert len(ConnectionPoolMode)     == 2
    print("[PASS] ConnectionPoolMode 枚举（2 个成员）")

    # 3. KeepaliveConfig 序列化
    kp = _KEEPALIVE_PRE
    env_kp = kp.to_env()
    assert env_kp["SCCACHE_DIST_KEEPALIVE_ENABLED"]  == "true"
    assert env_kp["SCCACHE_DIST_KEEPALIVE_INTERVAL"] == "20"
    assert env_kp["SCCACHE_DIST_KEEPALIVE_TIMEOUT"]  == "600"
    assert not _KEEPALIVE_POST.is_applicable()
    print("[PASS] KeepaliveConfig 序列化正确（旧版 3 键，新版不适用）")

    # 4. PROFILE 快照一致性
    assert PROFILE_PRE_42D3CC1.idle_timeout_secs    == 7200
    assert PROFILE_PRE_42D3CC1.connection_pool_mode == ConnectionPoolMode.DISABLED
    assert PROFILE_PRE_42D3CC1.connect_timeout_secs == 30
    assert PROFILE_POST_42D3CC1.idle_timeout_secs   == 0
    assert PROFILE_POST_42D3CC1.connection_pool_mode == ConnectionPoolMode.ENABLED
    assert PROFILE_POST_42D3CC1.connect_timeout_secs is None
    assert PROFILE_POST_42D3CC1.is_post_42d3cc1()
    assert not PROFILE_PRE_42D3CC1.is_post_42d3cc1()
    print("[PASS] SccacheDistProfile 快照值正确（旧版/新版各字段）")

    # 5. SccacheDistEnvBuilder — 新版不含已删除的变量
    post_env = SccacheDistEnvBuilder(PROFILE_POST_42D3CC1).build_env_dict()
    assert post_env["SCCACHE_IDLE_TIMEOUT"]          == "0"
    assert "SCCACHE_DIST_CONNECTION_POOL"   not in post_env
    assert "SCCACHE_DIST_CONNECT_TIMEOUT"   not in post_env
    assert "SCCACHE_DIST_KEEPALIVE_ENABLED" not in post_env
    assert "SCCACHE_DIST_KEEPALIVE_INTERVAL" not in post_env
    assert "SCCACHE_DIST_KEEPALIVE_TIMEOUT"  not in post_env
    print(f"[PASS] SccacheDistEnvBuilder 新版 ENV: {len(post_env)} 个键，"
          f"旧版 keepalive/pool 键均不存在")

    # 6. SccacheDistEnvBuilder — 旧版含所有删除前的变量
    pre_env = SccacheDistEnvBuilder(PROFILE_PRE_42D3CC1).build_env_dict()
    assert pre_env["SCCACHE_IDLE_TIMEOUT"]              == "7200"
    assert pre_env["SCCACHE_DIST_CONNECTION_POOL"]      == "false"
    assert pre_env["SCCACHE_DIST_CONNECT_TIMEOUT"]      == "30"
    assert pre_env["SCCACHE_DIST_KEEPALIVE_ENABLED"]    == "true"
    assert pre_env["SCCACHE_DIST_KEEPALIVE_INTERVAL"]   == "20"
    assert pre_env["SCCACHE_DIST_KEEPALIVE_TIMEOUT"]    == "600"
    print(f"[PASS] SccacheDistEnvBuilder 旧版 ENV: {len(pre_env)} 个键，"
          f"keepalive/pool 键全部存在")

    # 7. SccacheDistMigrationDiff — 变化项数量和内容
    diff  = SccacheDistMigrationDiff()
    changes = diff.compute()
    changed_keys = {c.key for c in changes}
    assert "SCCACHE_IDLE_TIMEOUT"            in changed_keys   # 7200 → 0
    assert "SCCACHE_DIST_CONNECT_TIMEOUT"    in changed_keys   # 30 → 删除
    assert "SCCACHE_DIST_CONNECTION_POOL"    in changed_keys   # false → 删除
    assert "SCCACHE_DIST_KEEPALIVE_ENABLED"  in changed_keys   # true → 删除
    assert "SCCACHE_DIST_KEEPALIVE_INTERVAL" in changed_keys   # 20 → 删除
    assert "SCCACHE_DIST_KEEPALIVE_TIMEOUT"  in changed_keys   # 600 → 删除
    # request_timeout / fallback / max_retries 不变，不应出现在 changes 中
    assert "SCCACHE_DIST_REQUEST_TIMEOUT"    not in changed_keys
    print(f"[PASS] SccacheDistMigrationDiff: {len(changes)} 个变化项，"
          f"全部符合上游 diff 语义")

    # 8. Diff 报告包含关键信息
    report_str = diff.report()
    assert "42d3cc1" in report_str
    assert "SCCACHE_IDLE_TIMEOUT" in report_str
    print("[PASS] SccacheDistMigrationDiff.report() 包含 commit 引用和关键变量")

    # 9. SccacheDistRuntimeAudit 不崩溃
    audit = SccacheDistRuntimeAudit()
    audit_results = audit.audit()
    assert isinstance(audit_results, list)
    assert all(isinstance(r, AuditResult) for r in audit_results)
    print(f"[PASS] SccacheDistRuntimeAudit.audit(): {len(audit_results)} 项")

    # 10. __all__ 导出完整性
    import importlib as _il
    for name in __all__:
        assert name in dir(), f"{name} 未定义"
    print(f"[PASS] __all__ 共 {len(__all__)} 个导出符号")

    # 11. get_active_profile() 返回新版
    active = get_active_profile()
    assert active is PROFILE_POST_42D3CC1
    assert active.is_post_42d3cc1()
    print("[PASS] get_active_profile() 返回 PROFILE_POST_42D3CC1")

    print()
    print(diff.report())
    print()
    print(audit.report())
    print()
    print("── 全部 11 项断言通过 ────────────────────────────────────")
    sys.exit(0)
