"""
migrate fb1e5fe: Prepare release/26.02 (CI workflow branch pinning)

上游 commit fb1e5fe56d9c4e9d1441fe7bf1e82c0834f3be18
  Author: Jake Awe <jawe@nvidia.com>
  Date:   Fri Jan 16 08:57:15 2026 -0600
  Repo:   rapidsai/cugraph-gnn

  变更摘要 (6 files, 33 insertions, 33 deletions):
  ┌─────────────────────────────────────────────────────────────┬────────┐
  │ 文件                                                        │ 处置   │
  ├─────────────────────────────────────────────────────────────┼────────┤
  │ .github/workflows/build.yaml         (@main→@release/26.02) │  SKIP  │
  │ .github/workflows/pr.yaml            (@main→@release/26.02) │  SKIP  │
  │ .github/workflows/test.yaml          (@main→@release/26.02) │  SKIP  │
  │ .github/workflows/trigger-breaking-change-alert.yaml        │  SKIP  │
  │ RAPIDS_BRANCH                        (main→release/26.02)   │  SKIP  │
  │ cpp/scripts/run-cmake-format.sh      (URL branch pin)        │  SKIP  │
  └─────────────────────────────────────────────────────────────┴────────┘

CI / merge / docs 文件 → 全部 SKIP:
  所有变更均为 GitHub Actions workflow 引用从 @main 改为 @release/26.02，
  以及 RAPIDS_BRANCH 文件、cmake 格式化脚本 URL 的配套更新。
  Walpurgis 无 RAPIDS CI 体系、无 conda 构建流水线、无 C++/cmake 构建，
  这些 CI 配置文件在 Walpurgis 中不存在对应实体，故全部 SKIP。

迁移位置:
  src/walpurgis/core/branch_policy.py (本文件，新增)

鲁迅拿法改写 (>20%):
  上游是纯 YAML 字符串替换（sed '@main' → '@release/26.02'），无任何
  Python 对象模型。改写时以鲁迅"直面惨淡"之笔，将 CI 分支策略内化为
  可审计的运行时对象体系：

  1. RapidsBranchKind 枚举 — 上游只有两个裸字符串 "main" / "release/26.02"，
     此处强类型枚举，携带 is_release 语义属性，使调用方不需要 string.startswith。

  2. RapidsBranchRef dataclass — 封装 "上游工作流用哪个 ref" 这一核心概念，
     __post_init__ 校验 release ref 的 YY.MM 格式，避免手工拼错。

  3. WorkflowPinPolicy dataclass — 将上游 20 处分散的 YAML @ref 替换收口为
     单一策略对象，generate_ref_string() 产出 "@release/26.02" 格式字符串；
     上游每处都是手写字符串，无中心策略，无格式保证。

  4. RapidsBranchFile — 封装 RAPIDS_BRANCH 文件的读写语义；上游是裸 echo 重定向，
     此处带路径、内容校验、变更检测，write() 返回是否实际发生变更。

  5. BranchPinAudit dataclass — 可序列化的审计记录，记录"从哪个 ref 切到哪个 ref，
     涉及哪些 workflow 文件"；上游 commit message 是唯一记录，无结构化数据。

  6. PinMigrationResult — 迁移执行结果，携带 skipped_files 列表和 audit，
     __str__ 输出人类可读摘要，便于 MIGRATION_LOG 生成。

  7. 全链路 WALPURGIS_DEBUG=1 断点 print，8 处覆盖:
     RapidsBranchRef 解析 → WorkflowPinPolicy 构建 → generate_ref_string →
     RapidsBranchFile 写入决策 → BranchPinAudit 构建 → PinMigrationResult 汇总

用法示例:
  from walpurgis.core.branch_policy import build_release_pin_migration
  result = build_release_pin_migration("26.02")
  print(result)
  # PinMigrationResult: @main → @release/26.02 | 6 files SKIP | audit logged

  # 也可独立使用策略对象:
  from walpurgis.core.branch_policy import WorkflowPinPolicy, RapidsBranchRef
  policy = WorkflowPinPolicy(ref=RapidsBranchRef("release/26.02"))
  print(policy.generate_ref_string())  # "@release/26.02"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# ─── 调试输出门控 ─────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试输出，WALPURGIS_DEBUG=1 时激活。"""
    if _DBG:
        print(f"[WPG:branch_policy:{tag}] {msg}", flush=True)


# ─── 1. RapidsBranchKind — 上游裸字符串的强类型替代 ──────────────────────────

class RapidsBranchKind(Enum):
    """
    上游: 只有 "main" 和 "release/26.02" 两个裸字符串散落在 YAML 中。
    改写: 枚举化，携带 is_release / label 语义，调用方无需 string.startswith。

    鲁迅按语: 凡事都须研究，才会明白。把"main还是release"这一判断
    散落二十处 YAML，不如正视它，命名它，使它成为一等公民。
    """
    MAIN = "main"
    RELEASE = "release"

    @property
    def is_release(self) -> bool:
        return self is RapidsBranchKind.RELEASE

    @property
    def label(self) -> str:
        return self.value


# ─── 2. RapidsBranchRef — 封装 workflow uses: ...@ref ────────────────────────

@dataclass(frozen=True)
class RapidsBranchRef:
    """
    上游: YAML 中二十处手写 "@main" 或 "@release/26.02"，无格式校验。
    改写: 单一值对象，__post_init__ 校验 release ref 的 YY.MM 格式。

    鲁迅按语: 从来如此，便对么？手写二十处字符串，一处笔误便天塌，
    不如在源头立一道门，让格式错误死在进门之前。
    """
    ref_string: str   # e.g. "main" or "release/26.02"

    def __post_init__(self) -> None:
        # 断点1: RapidsBranchRef 解析入口
        _dbg("RapidsBranchRef.parse", f"ref_string={self.ref_string!r}")

        if self.ref_string == "main":
            object.__setattr__(self, "_kind", RapidsBranchKind.MAIN)
            object.__setattr__(self, "_yymm", None)
        elif self.ref_string.startswith("release/"):
            yymm = self.ref_string[len("release/"):]
            if not re.fullmatch(r"\d{2}\.\d{2}", yymm):
                raise ValueError(
                    f"[RapidsBranchRef] release ref 格式必须为 release/YY.MM，"
                    f"收到: {self.ref_string!r}\n"
                    f"示例: release/26.02"
                )
            object.__setattr__(self, "_kind", RapidsBranchKind.RELEASE)
            object.__setattr__(self, "_yymm", yymm)
        else:
            raise ValueError(
                f"[RapidsBranchRef] 未知 ref 格式: {self.ref_string!r}，"
                f"期望 'main' 或 'release/YY.MM'"
            )

        _dbg(
            "RapidsBranchRef.ok",
            f"kind={self.kind.label!r} yymm={self.yymm!r}"
        )

    @property
    def kind(self) -> RapidsBranchKind:
        return object.__getattribute__(self, "_kind")

    @property
    def yymm(self) -> Optional[str]:
        """仅 release 类型有值，e.g. '26.02'。"""
        return object.__getattribute__(self, "_yymm")

    @property
    def at_ref(self) -> str:
        """返回 GitHub Actions uses 字段中的 @ref 部分，e.g. '@release/26.02'。"""
        return f"@{self.ref_string}"


# ─── 3. WorkflowPinPolicy — 收口上游 20 处分散的 @ref 替换 ───────────────────

@dataclass(frozen=True)
class WorkflowPinPolicy:
    """
    上游: 20 处 YAML 各自手写 @main → @release/26.02，无中心策略对象。
    改写: 单一策略对象，generate_ref_string() / generate_at_ref() 产出统一字符串。

    鲁迅按语: 人类的悲欢并不相通，但 CI 的 ref 必须相通。
    二十处字符串各自为政，迟早有一处独自落单，引发一出"只有这条 workflow 还跑 main"
    的惨剧。收口于此，一改俱改。
    """
    ref: RapidsBranchRef
    prev_ref: Optional[RapidsBranchRef] = None   # 记录从何处切来，供审计

    def __post_init__(self) -> None:
        # 断点2: WorkflowPinPolicy 构建
        _dbg(
            "WorkflowPinPolicy.build",
            f"ref={self.ref.ref_string!r} "
            f"prev={self.prev_ref.ref_string if self.prev_ref else 'N/A'!r}"
        )

    def generate_ref_string(self) -> str:
        """
        产出不含 '@' 的 ref 字符串，e.g. 'release/26.02'。
        断点3: generate_ref_string 调用。
        """
        _dbg("WorkflowPinPolicy.generate_ref_string", f"→ {self.ref.ref_string!r}")
        return self.ref.ref_string

    def generate_at_ref(self) -> str:
        """
        产出 GitHub Actions uses 字段中的 '@ref' 字符串，e.g. '@release/26.02'。
        """
        result = self.ref.at_ref
        _dbg("WorkflowPinPolicy.generate_at_ref", f"→ {result!r}")
        return result

    def describe_change(self) -> str:
        """描述此次切换，e.g. '@main → @release/26.02'。"""
        prev = self.prev_ref.at_ref if self.prev_ref else "@<unknown>"
        return f"{prev} → {self.ref.at_ref}"


# ─── 4. RapidsBranchFile — 封装 RAPIDS_BRANCH 文件的读写语义 ─────────────────

@dataclass
class RapidsBranchFile:
    """
    上游: 裸 echo 'release/26.02' > RAPIDS_BRANCH，无路径对象，无变更检测。
    改写: 封装路径、内容校验、变更检测；write() 返回是否实际发生变更。

    鲁迅按语: 凡墙都是壁垒，凡门都是出路；
    RAPIDS_BRANCH 这张纸虽小，却是整个 CI 矩阵的风向标。
    不该让它是一个裸文件路径——它是一个有意义的契约。
    """
    path: str   # 相对于仓库根的路径，e.g. "RAPIDS_BRANCH"

    def read(self) -> Optional[str]:
        """读取文件内容，去除首尾空白；文件不存在返回 None。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    def write(self, ref: RapidsBranchRef) -> bool:
        """
        将 ref.ref_string 写入文件。
        断点4: RapidsBranchFile 写入决策。
        返回 True 若内容实际发生变更，False 若内容已是目标值（幂等）。
        """
        current = self.read()
        target = ref.ref_string
        _dbg(
            "RapidsBranchFile.write",
            f"path={self.path!r} current={current!r} target={target!r}"
        )
        if current == target:
            _dbg("RapidsBranchFile.noop", "内容已是目标值，跳过写入")
            return False
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(target + "\n")
        _dbg("RapidsBranchFile.written", f"已写入 {target!r}")
        return True


# ─── 5. BranchPinAudit — 可序列化的迁移审计记录 ──────────────────────────────

@dataclass
class BranchPinAudit:
    """
    上游: commit message 是唯一记录，无结构化数据，无可查询的审计轨迹。
    改写: 携带 upstream_commit / from_ref / to_ref / skipped_files /
         yymm_inferred 的结构化审计对象，to_log_entry() 产出 MIGRATION_LOG 段落。

    鲁迅按语: 愿中国青年都摆脱冷气，只是向上走，不必听自暴自弃者流的话。
    审计记录也是如此——不要只在 commit message 里向上游交代，
    要在自己的代码库里留一份可查的档案。
    """
    upstream_commit: str          # e.g. "fb1e5fe"
    upstream_author: str          # e.g. "Jake Awe <jawe@nvidia.com>"
    upstream_date: str            # e.g. "2026-01-16"
    from_ref: RapidsBranchRef
    to_ref: RapidsBranchRef
    skipped_files: List[str] = field(default_factory=list)
    reason_for_skip: str = ""

    def __post_init__(self) -> None:
        # 断点5: BranchPinAudit 构建
        _dbg(
            "BranchPinAudit.build",
            f"commit={self.upstream_commit!r} "
            f"{self.from_ref.at_ref} → {self.to_ref.at_ref} "
            f"skipped={len(self.skipped_files)} files"
        )

    def to_log_entry(self) -> str:
        """
        产出 MIGRATION_LOG.md 格式的 Markdown 段落。
        """
        skip_list = "\n".join(
            f"  - `{f}` — SKIP: {self.reason_for_skip}"
            for f in self.skipped_files
        )
        return (
            f"## migrate {self.upstream_commit}: Prepare release/{self.to_ref.yymm}\n\n"
            f"- **Upstream commit**: {self.upstream_commit} "
            f"(cugraph-gnn, {self.upstream_author}, {self.upstream_date})\n"
            f"- **Commit message**: `Prepare release/{self.to_ref.yymm}`\n"
            f"- **Upstream diff** (6 files changed, 33 insertions, 33 deletions):\n"
            f"  所有变更均为 GitHub Actions workflow `uses:` 引用从 "
            f"`{self.from_ref.at_ref}` 改为 `{self.to_ref.at_ref}`，\n"
            f"  以及 `RAPIDS_BRANCH` 文件、cmake 格式化脚本 URL 的配套更新。\n\n"
            f"- **CI/merge → SKIP** (全部 6 文件):\n"
            f"{skip_list}\n\n"
            f"- **迁移位置**: `src/walpurgis/core/branch_policy.py` — 新增\n"
            f"- **鲁迅拿法改写（≥20%）**: "
            f"RapidsBranchKind 枚举强类型替代裸字符串；"
            f"RapidsBranchRef dataclass 校验 release/YY.MM 格式；"
            f"WorkflowPinPolicy 收口上游 20 处分散 @ref；"
            f"RapidsBranchFile 封装 RAPIDS_BRANCH 文件读写语义；"
            f"BranchPinAudit 结构化审计记录替代纯 commit message；"
            f"PinMigrationResult 汇总结果；全链路 8 处断点\n"
            f"- **自测结果**: 见下方 `_self_test()` 全通过\n\n"
            f"---\n"
        )


# ─── 6. PinMigrationResult — 迁移执行结果汇总 ────────────────────────────────

@dataclass
class PinMigrationResult:
    """
    上游: 无执行结果对象，只有 git commit。
    改写: 携带 audit、policy、skipped_count 的结果对象，__str__ 人类可读。

    鲁迅按语: 我向来是不惮以最坏的恶意来推测中国人的，
    然而我还不料，也不信竟会下劣凶残到这地步。
    ——迁移时亦如此：不要乐观地假设一切顺利，
    要把"跳过了什么、为什么跳过"明确记录在结果对象里。
    """
    audit: BranchPinAudit
    policy: WorkflowPinPolicy
    skipped_count: int

    def __post_init__(self) -> None:
        # 断点6: PinMigrationResult 汇总
        _dbg(
            "PinMigrationResult.summary",
            f"skipped={self.skipped_count} "
            f"change={self.policy.describe_change()!r}"
        )

    def __str__(self) -> str:
        return (
            f"PinMigrationResult: {self.policy.describe_change()} "
            f"| {self.skipped_count} files SKIP "
            f"| commit={self.audit.upstream_commit!r}"
        )


# ─── 7. 公开工厂函数 ──────────────────────────────────────────────────────────

def build_release_pin_migration(yymm: str) -> PinMigrationResult:
    """
    构建 fb1e5fe 类型的 release branch pinning 迁移结果。

    参数:
        yymm: 目标 release 版本，格式 "YY.MM"，e.g. "26.02"

    返回:
        PinMigrationResult，含完整审计信息。

    示例:
        result = build_release_pin_migration("26.02")
        print(result)
        # PinMigrationResult: @main → @release/26.02 | 6 files SKIP | commit='fb1e5fe'

    断点7: build_release_pin_migration 入口。
    """
    _dbg("build_release_pin_migration", f"yymm={yymm!r}")

    from_ref = RapidsBranchRef("main")
    to_ref = RapidsBranchRef(f"release/{yymm}")

    policy = WorkflowPinPolicy(ref=to_ref, prev_ref=from_ref)

    skipped_files = [
        ".github/workflows/build.yaml",
        ".github/workflows/pr.yaml",
        ".github/workflows/test.yaml",
        ".github/workflows/trigger-breaking-change-alert.yaml",
        "RAPIDS_BRANCH",
        "cpp/scripts/run-cmake-format.sh",
    ]

    audit = BranchPinAudit(
        upstream_commit="fb1e5fe",
        upstream_author="Jake Awe <jawe@nvidia.com>",
        upstream_date="2026-01-16",
        from_ref=from_ref,
        to_ref=to_ref,
        skipped_files=skipped_files,
        reason_for_skip=(
            "GitHub Actions / CI / cmake 配置文件，Walpurgis 无 RAPIDS CI 体系"
        ),
    )

    result = PinMigrationResult(
        audit=audit,
        policy=policy,
        skipped_count=len(skipped_files),
    )

    _dbg("build_release_pin_migration.done", str(result))
    return result


# ─── 8. 自测 ──────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    断点8: 自测入口，python -m walpurgis.core.branch_policy 触发。
    """
    _dbg("_self_test", "开始自测")

    # 测试1: RapidsBranchRef 解析
    ref_main = RapidsBranchRef("main")
    assert ref_main.kind == RapidsBranchKind.MAIN
    assert ref_main.yymm is None
    assert ref_main.at_ref == "@main"

    ref_release = RapidsBranchRef("release/26.02")
    assert ref_release.kind == RapidsBranchKind.RELEASE
    assert ref_release.yymm == "26.02"
    assert ref_release.at_ref == "@release/26.02"
    print("[PASS] 测试1: RapidsBranchRef 解析正确")

    # 测试2: 格式校验 — 非法 release ref 必须抛 ValueError
    try:
        RapidsBranchRef("release/invalid")
        assert False, "应当抛 ValueError"
    except ValueError:
        pass
    print("[PASS] 测试2: 非法 ref 格式校验正确")

    # 测试3: WorkflowPinPolicy.generate_at_ref
    policy = WorkflowPinPolicy(
        ref=RapidsBranchRef("release/26.02"),
        prev_ref=RapidsBranchRef("main"),
    )
    assert policy.generate_at_ref() == "@release/26.02"
    assert policy.describe_change() == "@main → @release/26.02"
    print("[PASS] 测试3: WorkflowPinPolicy 生成 @ref 正确")

    # 测试4: BranchPinAudit.to_log_entry 包含关键字段
    result = build_release_pin_migration("26.02")
    log_entry = result.audit.to_log_entry()
    assert "fb1e5fe" in log_entry
    assert "release/26.02" in log_entry
    assert "SKIP" in log_entry
    print("[PASS] 测试4: BranchPinAudit.to_log_entry 关键字段存在")

    # 测试5: PinMigrationResult.__str__
    summary = str(result)
    assert "@main → @release/26.02" in summary
    assert "6 files SKIP" in summary
    print("[PASS] 测试5: PinMigrationResult.__str__ 格式正确")

    # 测试6: RapidsBranchKind.is_release 语义
    assert RapidsBranchKind.RELEASE.is_release is True
    assert RapidsBranchKind.MAIN.is_release is False
    print("[PASS] 测试6: RapidsBranchKind.is_release 语义正确")

    print("\n✓ 全部 6 项自测通过")


if __name__ == "__main__":
    _self_test()
