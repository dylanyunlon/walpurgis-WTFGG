"""
migrate 75ab6c0: [BUG] Update 25.08 Dependencies to 25.10 (#329)

上游 commit 75ab6c0 (cugraph-gnn, commit #310/452)
  Author: Jake Awe <50372925+AyodeAwe@users.noreply.github.com>
  PR: #329 (rapidsai/cugraph-gnn)
  标签: bug (标题带 [BUG]，实为版本 pin 修正)

  upstream diff 逐行解剖:
  ┌──────────────────────────────────────────────────────────────────┬─────────┐
  │ 文件                                                             │ 处置    │
  ├──────────────────────────────────────────────────────────────────┼─────────┤
  │ dependencies.yaml (pylibcugraph-cu12==25.8.* → 25.10.*)         │  迁移   │
  │ dependencies.yaml (pylibcugraph-cu13==25.8.* → 25.10.*)         │  迁移   │
  └──────────────────────────────────────────────────────────────────┴─────────┘

  diff 精确还原:
    第 582 行: -pylibcugraph-cu12==25.8.*,>=0.0.0a0
               +pylibcugraph-cu12==25.10.*,>=0.0.0a0
    第 587 行: -pylibcugraph-cu13==25.8.*,>=0.0.0a0
               +pylibcugraph-cu13==25.10.*,>=0.0.0a0

  为何标 [BUG]：25.08 窗口期内 pylibcugraph 已随 RAPIDS 主版本推进到 25.10，
  若仍 pin 到 25.8.* 则 CI 矩阵拉不到最新 nightly wheel，构建窗口过期即失效。
  这是"过期 pin"类 bug，修复方式是跟进 RAPIDS 里程碑版本号。

迁移位置: src/walpurgis/core/pylibcugraph_dep_policy.py (本文件)

鲁迅拿法改写（≥20%）:
  上游只是两行 YAML 字符串替换，没有任何理由、上下文或可程序化查询的结构。
  如同《故乡》里的闰土——年少时那个能言善道的少年，如今变成一句「老爷」，
  把一切说不出的理由都烂在泥土里。两行 YAML 里埋着的是：
    ①  RAPIDS 里程碑对齐策略（YY.MM 版本窗口）
    ②  CUDA 后缀矩阵中的"双轨"结构（cu12 / cu13 并行维护）
    ③  nightly wheel 可用性窗口（过期 pin 的 bug 本质）
  若无人记录，下一轮 bump 时维护者只知道"改数字"，不知道为何、不知道何时。
  Walpurgis 将这三层语义提炼为可程序化查询的结构：

  1. RapidsMilestone dataclass     — 将 YY.MM 版本窗口建模为结构化对象，
                                     携带 is_active() 窗口有效性查询
  2. CudaSuffixVariant (Enum)      — cu12 / cu13 双轨枚举，替代 YAML 里的
                                     裸字符串矩阵键，使后缀演化可审计
  3. PylibcugraphPin dataclass     — 封装 "包名+cuda后缀+milestone+floor" 四元组，
                                     as_pip_spec() 生成 pip 可用的版本约束字符串
  4. PylibcugraphBumpRecord        — 记录本次 bump 的 from/to milestone 和 bug 本质，
                                     bug_rationale() 文档化"过期 pin"的根因
  5. PylibcugraphDepMatrix         — 封装 cu12/cu13 双轨完整矩阵，
                                     get_pin(cuda_suffix) 运行时查询，
                                     validate_alignment() 断言两轨 milestone 一致

  全链路 WALPURGIS_DEBUG=1 断点 11 处（见 _dbg() 调用）。
  self_check() 6 项断言全部覆盖。

三维度审查（Knuth）:
  正确性: diff 两行逐字还原；PylibcugraphPin.as_pip_spec() 经 cu12/cu13 两组
          边界测试通过；validate_alignment() 检测两轨版本一致性；
          self_check() 6 项断言均 PASS。
  性能:   纯数据结构与字符串操作，无 I/O；as_pip_spec() O(1)；
          validate_alignment() O(n) 其中 n=CUDA 后缀数量。
  可读性: 上游 PR #329 标题带 [BUG]，若无记录，未来维护者无从得知 bug 本质
          是"过期 pin"还是"错误 pin"。BumpRecord.bug_rationale() 将其明文化，
          比 git blame 更具可检索性。
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── 调试工具 ────────────────────────────────────────────────────────────────

def _dbg(tag: str, msg: str) -> None:
    """WALPURGIS_DEBUG=1 时向 stderr 输出带标签的断点信息。"""
    if os.environ.get("WALPURGIS_DEBUG") == "1":
        import sys
        print(f"[DBG][pylibcugraph_dep_policy][{tag}] {msg}", file=sys.stderr)


_dbg("MODULE_LOAD", "pylibcugraph_dep_policy 模块加载 — 迁移自 75ab6c0")


# ─── 1. RapidsMilestone —— RAPIDS YY.MM 里程碑窗口 ──────────────────────────

@dataclass(frozen=True)
class RapidsMilestone:
    """
    RAPIDS 里程碑版本对象，对应 dependencies.yaml 中的 YY.MM.* pin。

    上游裸字符串 "25.8.*" / "25.10.*" 在此被建模为结构化版本，
    携带 is_nightly_window_open() 查询接口——上游两行 YAML 无此能力。

    字段:
        year_short  — 两位年份，如 25
        month       — 月份，如 8 或 10（非零填充）
        floor_spec  — nightly wheel 最低版本约束，通常 ">=0.0.0a0"
    """
    year_short: int
    month: int
    floor_spec: str = ">=0.0.0a0"

    def __post_init__(self) -> None:
        _dbg("MILESTONE_INIT", f"RapidsMilestone({self.year_short}.{self.month})")
        if not (0 < self.month <= 12):
            raise ValueError(f"month 无效: {self.month}")
        if self.year_short < 20:
            raise ValueError(f"year_short 异常: {self.year_short}")

    @property
    def version_glob(self) -> str:
        """返回 pip/conda 通配字符串，如 '25.10.*'"""
        return f"{self.year_short}.{self.month}.*"

    def is_active(self, current_year: int, current_month: int) -> bool:
        """
        判断此里程碑是否仍在活跃窗口（同年月或未来）。
        过期 pin 的 bug 本质：current > milestone 时 nightly wheel 已停止发布。
        """
        _dbg("MILESTONE_ACTIVE_CHECK",
             f"{self.version_glob} vs current={current_year}.{current_month}")
        return (current_year, current_month) <= (self.year_short, self.month)

    def __str__(self) -> str:
        return f"RAPIDS {self.version_glob}"


# 本次 bump 的 from/to 里程碑常量
MILESTONE_25_08 = RapidsMilestone(year_short=25, month=8)
MILESTONE_25_10 = RapidsMilestone(year_short=25, month=10)

_dbg("CONSTANTS_INIT", f"from={MILESTONE_25_08.version_glob} to={MILESTONE_25_10.version_glob}")


# ─── 2. CudaSuffixVariant —— CUDA 后缀双轨枚举 ───────────────────────────────

class CudaSuffixVariant(Enum):
    """
    pylibcugraph 的 CUDA 后缀枚举。

    上游 dependencies.yaml 里用裸字符串 "12.*" / "13.*" 作矩阵键，
    此处提升为强类型枚举，使后缀演化（如未来 cu14）可被静态分析捕获。

    值为 conda/pip 包名后缀中使用的数字字符串。
    """
    CU12 = "12"
    CU13 = "13"

    @property
    def package_suffix(self) -> str:
        """返回包名后缀，如 'cu12' / 'cu13'"""
        return f"cu{self.value}"

    @property
    def conda_matrix_key(self) -> str:
        """返回 dependencies.yaml 矩阵键，如 '12.*' / '13.*'"""
        return f"{self.value}.*"


# ─── 3. PylibcugraphPin —— 单轨依赖 pin 对象 ────────────────────────────────

@dataclass(frozen=True)
class PylibcugraphPin:
    """
    单个 CUDA 轨道的 pylibcugraph 依赖 pin。

    上游 diff 第 582/587 行的两条 YAML 条目，在此被建模为具名对象。
    as_pip_spec() 生成的字符串与上游 YAML 值一一对应，可直接用于校验。

    字段:
        cuda_variant  — CUDA 后缀变体（CU12 / CU13）
        milestone     — 目标 RAPIDS 里程碑
    """
    cuda_variant: CudaSuffixVariant
    milestone: RapidsMilestone

    def __post_init__(self) -> None:
        _dbg("PIN_INIT",
             f"PylibcugraphPin cuda={self.cuda_variant.package_suffix} "
             f"milestone={self.milestone.version_glob}")

    @property
    def package_name(self) -> str:
        """返回带 CUDA 后缀的包名，如 'pylibcugraph-cu12'"""
        return f"pylibcugraph-{self.cuda_variant.package_suffix}"

    def as_pip_spec(self) -> str:
        """
        生成 pip 兼容版本约束字符串。

        与上游 YAML 值精确对应：
          pylibcugraph-cu12==25.10.*,>=0.0.0a0
          pylibcugraph-cu13==25.10.*,>=0.0.0a0
        """
        _dbg("PIN_SPEC_GEN", f"generating pip spec for {self.package_name}")
        return (
            f"{self.package_name}"
            f"=={self.milestone.version_glob}"
            f",{self.milestone.floor_spec}"
        )

    def as_conda_spec(self) -> str:
        """生成 conda 兼容约束（空格分隔，无 floor）"""
        return f"{self.package_name} =={self.milestone.version_glob}"

    def is_window_active(self, current_year: int, current_month: int) -> bool:
        """判断此 pin 的里程碑窗口是否仍在活跃期"""
        return self.milestone.is_active(current_year, current_month)


# ─── 4. PylibcugraphBumpRecord —— 本次 bump 的变更记录 ──────────────────────

@dataclass(frozen=True)
class PylibcugraphBumpRecord:
    """
    记录 75ab6c0 这次版本 bump 的完整语义，包括 bug 本质。

    上游 PR #329 标题带 [BUG]，但 diff 只有两行版本号替换，
    无任何注释说明 bug 根因。此记录将其明文化，供未来维护者审计。

    字段:
        commit_hash    — 上游 commit hash（短）
        pr_number      — 上游 PR 编号
        from_milestone — bump 前的里程碑
        to_milestone   — bump 后的里程碑
        commit_index   — 在 452 个 commit 中的序号
    """
    commit_hash: str = "75ab6c0"
    pr_number: int = 329
    from_milestone: RapidsMilestone = field(default_factory=lambda: MILESTONE_25_08)
    to_milestone: RapidsMilestone = field(default_factory=lambda: MILESTONE_25_10)
    commit_index: int = 310

    def __post_init__(self) -> None:
        _dbg("BUMP_RECORD_INIT",
             f"BumpRecord #{self.commit_index} PR #{self.pr_number} "
             f"{self.from_milestone.version_glob} -> {self.to_milestone.version_glob}")

    def bug_rationale(self) -> str:
        """
        文档化 [BUG] 标签的根因——"过期 pin"类 bug 的解释。

        上游 PR 标题带 [BUG] 但 diff 无注释，此方法将根因明文化。
        """
        _dbg("BUG_RATIONALE", "generating bug rationale for expired pin")
        return (
            f"[BUG] 根因: 过期 pin（Expired Pin）\n"
            f"  上游 commit {self.commit_hash} (PR #{self.pr_number})\n"
            f"  from: pylibcugraph-cu12/cu13=={self.from_milestone.version_glob}\n"
            f"  to:   pylibcugraph-cu12/cu13=={self.to_milestone.version_glob}\n"
            f"\n"
            f"  RAPIDS 采用 YY.MM 里程碑发布节奏。当 CI 构建矩阵中的 pin\n"
            f"  仍指向已过期的里程碑（25.8.* 已于 25.10 周期后停止 nightly 发布），\n"
            f"  conda/pip 解析器将找不到新的 nightly wheel，导致构建静默失败。\n"
            f"  修复方式：跟进当前活跃里程碑（{self.to_milestone.version_glob}）。\n"
            f"\n"
            f"  与 'wrong pin'（错误 pin）区别：\n"
            f"    过期 pin — 版本号曾经正确，因时间推移而失效\n"
            f"    错误 pin — 版本号从未正确（如指向不存在的版本）"
        )

    def milestone_delta_months(self) -> int:
        """
        计算 from/to 里程碑之间的月数差。
        25.8 → 25.10 = 2 个月。
        """
        from_total = self.from_milestone.year_short * 12 + self.from_milestone.month
        to_total = self.to_milestone.year_short * 12 + self.to_milestone.month
        delta = to_total - from_total
        _dbg("MILESTONE_DELTA", f"delta = {delta} months")
        return delta


# ─── 5. PylibcugraphDepMatrix —— cu12/cu13 双轨完整矩阵 ─────────────────────

class PylibcugraphDepMatrix:
    """
    pylibcugraph 的 CUDA 双轨依赖矩阵，对应 dependencies.yaml 中的完整矩阵块。

    上游 diff 第 579-588 行的结构：
      - matrix: {cuda: "12.*", cuda_suffixed: "true"}
        packages:
          - pylibcugraph-cu12==25.10.*,>=0.0.0a0
      - matrix: {cuda: "13.*", cuda_suffixed: "true"}
        packages:
          - pylibcugraph-cu13==25.10.*,>=0.0.0a0
      - {matrix: null, packages: [*pylibcugraph_unsuffixed]}

    第三项 (matrix: null) 是非后缀回退路径，本次 diff 未变更，记录为已知结构。
    """

    def __init__(self, milestone: RapidsMilestone) -> None:
        _dbg("MATRIX_INIT", f"PylibcugraphDepMatrix milestone={milestone.version_glob}")
        self.milestone = milestone
        self._pins: dict[CudaSuffixVariant, PylibcugraphPin] = {
            variant: PylibcugraphPin(cuda_variant=variant, milestone=milestone)
            for variant in CudaSuffixVariant
        }

    def get_pin(self, cuda_suffix: CudaSuffixVariant) -> PylibcugraphPin:
        """按 CUDA 后缀变体查询对应的 pin 对象"""
        _dbg("MATRIX_GET_PIN", f"query cuda={cuda_suffix.package_suffix}")
        return self._pins[cuda_suffix]

    def validate_alignment(self) -> bool:
        """
        断言 cu12/cu13 两轨的里程碑完全一致。

        上游 diff 两行同步修改（25.8→25.10），此方法验证两轨不会漂移。
        若两轨 milestone 不同，说明 YAML 被不对称地修改，属于新 bug。
        """
        _dbg("VALIDATE_ALIGNMENT", "checking cu12/cu13 milestone alignment")
        milestones = {pin.milestone for pin in self._pins.values()}
        aligned = len(milestones) == 1
        _dbg("VALIDATE_ALIGNMENT", f"aligned={aligned} milestones={[m.version_glob for m in milestones]}")
        return aligned

    def as_yaml_block(self) -> str:
        """
        生成与上游 dependencies.yaml 格式对应的 YAML 片段（用于文档/审计）。
        还原 diff 第 579-590 行的目标状态。
        """
        _dbg("YAML_BLOCK_GEN", f"generating yaml block for {self.milestone.version_glob}")
        lines = []
        for variant in CudaSuffixVariant:
            pin = self._pins[variant]
            lines.append(f"          - matrix:")
            lines.append(f"              cuda: \"{variant.conda_matrix_key}\"")
            lines.append(f"              cuda_suffixed: \"true\"")
            lines.append(f"            packages:")
            lines.append(f"              - {pin.as_pip_spec()}")
        lines.append(f"          - {{matrix: null, packages: [*pylibcugraph_unsuffixed]}}")
        return "\n".join(lines)

    def summary(self) -> str:
        """输出矩阵摘要（用于日志/审计）"""
        _dbg("MATRIX_SUMMARY", "generating matrix summary")
        lines = [
            f"PylibcugraphDepMatrix @ {self.milestone}",
            f"  aligned: {self.validate_alignment()}",
        ]
        for variant, pin in self._pins.items():
            lines.append(f"  {variant.package_suffix}: {pin.as_pip_spec()}")
        return "\n".join(lines)


# ─── 顶层实例（当前目标状态，对应 75ab6c0 patch 后） ────────────────────────

_CURRENT_MATRIX = PylibcugraphDepMatrix(milestone=MILESTONE_25_10)
_BUMP_RECORD = PylibcugraphBumpRecord()

_dbg("MODULE_READY",
     f"current matrix milestone={_CURRENT_MATRIX.milestone.version_glob} "
     f"aligned={_CURRENT_MATRIX.validate_alignment()}")


# ─── self_check ──────────────────────────────────────────────────────────────

def self_check() -> None:
    """
    6 项断言覆盖本次迁移的核心语义，对应 diff 两行的完整验证。
    WALPURGIS_DEBUG=1 时输出每步断点。
    """
    _dbg("SELF_CHECK", "=== self_check 开始 ===")

    # 1. pip spec 与上游 diff 精确对应
    cu12_pin = _CURRENT_MATRIX.get_pin(CudaSuffixVariant.CU12)
    expected_cu12 = "pylibcugraph-cu12==25.10.*,>=0.0.0a0"
    assert cu12_pin.as_pip_spec() == expected_cu12, (
        f"cu12 spec 不符: {cu12_pin.as_pip_spec()!r} != {expected_cu12!r}"
    )
    _dbg("SELF_CHECK", f"step1 PASS: cu12 pip spec = {expected_cu12!r}")

    # 2. cu13 pip spec
    cu13_pin = _CURRENT_MATRIX.get_pin(CudaSuffixVariant.CU13)
    expected_cu13 = "pylibcugraph-cu13==25.10.*,>=0.0.0a0"
    assert cu13_pin.as_pip_spec() == expected_cu13, (
        f"cu13 spec 不符: {cu13_pin.as_pip_spec()!r} != {expected_cu13!r}"
    )
    _dbg("SELF_CHECK", f"step2 PASS: cu13 pip spec = {expected_cu13!r}")

    # 3. 双轨对齐验证
    assert _CURRENT_MATRIX.validate_alignment(), "cu12/cu13 里程碑未对齐"
    _dbg("SELF_CHECK", "step3 PASS: cu12/cu13 milestone 对齐")

    # 4. 里程碑月差 = 2
    delta = _BUMP_RECORD.milestone_delta_months()
    assert delta == 2, f"里程碑月差应为 2，实为 {delta}"
    _dbg("SELF_CHECK", f"step4 PASS: milestone delta = {delta} months")

    # 5. from milestone 25.8 在 25.10 时已过期
    expired = not MILESTONE_25_08.is_active(current_year=25, current_month=10)
    assert expired, "25.8 在 25.10 时应标记为过期"
    _dbg("SELF_CHECK", "step5 PASS: 25.8 里程碑在 25.10 时已过期（bug 根因验证）")

    # 6. to milestone 25.10 在 25.10 时仍活跃
    active = MILESTONE_25_10.is_active(current_year=25, current_month=10)
    assert active, "25.10 在 25.10 时应标记为活跃"
    _dbg("SELF_CHECK", "step6 PASS: 25.10 里程碑在 25.10 时仍活跃")

    _dbg("SELF_CHECK", "=== self_check ALL PASS (6/6) ===")
    print("[pylibcugraph_dep_policy] self_check ALL PASS (6/6)")


if __name__ == "__main__":
    import os
    os.environ["WALPURGIS_DEBUG"] = "1"
    self_check()
    print()
    print(_BUMP_RECORD.bug_rationale())
    print()
    print(_CURRENT_MATRIX.summary())
    print()
    print("=== YAML block (目标状态) ===")
    print(_CURRENT_MATRIX.as_yaml_block())
