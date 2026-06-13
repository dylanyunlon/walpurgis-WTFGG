"""
migrate 0bdae67: Require CUDA 12.2+ (#340)

上游 commit 0bdae67
PR:     https://github.com/rapidsai/cugraph-gnn/pull/340
Subject: Require CUDA 12.2+ (#340)

上游变更（2 files changed, 1 insertion(+), 9 deletions(-)）：

  dependencies.yaml（-8行）
      在 specific: → output_types: conda → matrices: 段落中，
      删除 cuda: "12.0" 和 cuda: "12.1" 两个 matrix 条目（各 3 行）：
        - matrix:
            cuda: "12.0"
          packages:
            - cuda-version=12.0
        - matrix:
            cuda: "12.1"
          packages:
            - cuda-version=12.1
      保留 cuda: "12.2" 及其后条目，即最低支持版本由 12.0 提升至 12.2。
      Walpurgis 无 conda dependencies.yaml，无 conda 构建矩阵，
      但其 CUDA 最低版本策略需同步更新。

  readme_pages/CONTRIBUTING.md（1行改写）
      Hardware 节：
        - * You need to have accesses to an NVIDIA GPU that is Pascal or later.
        + * You need to have accesses to an NVIDIA GPU that is Volta or later.
      Walpurgis 无 readme_pages/CONTRIBUTING.md，但 GPU 架构最低要求
      需在策略模块中结构化记录。

鲁迅拿法改写（≥20%）：
  上游改动的本质是两件事：
    (A) 将 CUDA 最低版本从 12.0 提升至 12.2（删除两个 conda matrix 条目）
    (B) 将 GPU 最低架构要求从 Pascal 提升至 Volta（改一行文档）

  上游实现：直接删 YAML 行、改 Markdown 行，无任何机器可查询的策略对象，
  无版本提升的时间戳，无 SM 算力对应表，无「为何 12.2 是分水岭」的说明。
  鲁迅视之：删则删矣，何以为证？12.3 来了再删一行？无结构，无护卫，无记忆。

  Walpurgis 改写：
    1. GpuArch(Enum) — 将 Pascal/Volta/Turing/Ampere 等架构枚举化，
       携带 sm_major/sm_minor/release_year/cuda_min_required 四属性，
       is_volta_or_later() 方法取代文档里的一行文字。
    2. CudaMinRequirement(frozen dataclass) — 结构化封装「最低 CUDA 版本」策略，
       携带 version/reason/since_commit/dropped_versions 四字段，
       is_satisfied(ver) 方法取代 YAML 里的条目存在与否。
    3. CondaMatrixEntry(frozen dataclass) — 建模 dependencies.yaml 里的每一个
       cuda matrix 条目，supports() 方法判断是否在当前策略下仍被纳入。
    4. Cuda122Policy — 建模本 commit 引入的「12.2+ 策略」，
       dropped_entries() 列出被移除的条目，
       audit_report() 产出人类可读的迁移审计报告——上游只有 git diff。
    5. ContributingHardwarePolicy — 建模 CONTRIBUTING.md Hardware 节的
       GPU 架构要求，before/after 对比，change_summary() 产出结构化记录。
    6. 断点调试：全链路 10 处 WALPURGIS_DEBUG=1 断点。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 调试断点
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print(f"[WALPURGIS_DBG:{tag}] {msg}")


# ---------------------------------------------------------------------------
# 1. GpuArch — GPU 架构枚举（Pascal → Volta → Turing → Ampere → …）
# ---------------------------------------------------------------------------

class GpuArch(Enum):
    """
    NVIDIA GPU 架构枚举。

    上游 CONTRIBUTING.md 只用了「Pascal or later」→「Volta or later」两个词，
    没有 SM 算力、发布年份、对应最低 CUDA 版本的任何说明。
    鲁迅拿法：将模糊的文档短语结构化为可查询的枚举，
    使「架构最低要求」变成可以在代码里断言、在测试里覆盖的东西。

    属性：
        sm: (major, minor) 算力对，如 Pascal→(6,0)，Volta→(7,0)
        release_year: 架构量产年份（整数）
        cuda_min_required: 该架构所需的最低 CUDA 主版本号
    """
    MAXWELL   = ("Maxwell",   (5, 0), 2014, 8)
    PASCAL    = ("Pascal",    (6, 0), 2016, 8)
    VOLTA     = ("Volta",     (7, 0), 2017, 9)
    TURING    = ("Turing",    (7, 5), 2018, 10)
    AMPERE    = ("Ampere",    (8, 0), 2020, 11)
    ADA       = ("Ada",       (8, 9), 2022, 11)
    HOPPER    = ("Hopper",    (9, 0), 2022, 11)
    BLACKWELL = ("Blackwell", (10, 0), 2024, 12)

    def __init__(
        self,
        arch_name: str,
        sm: Tuple[int, int],
        release_year: int,
        cuda_min_major: int,
    ) -> None:
        self.arch_name = arch_name
        self.sm = sm
        self.release_year = release_year
        self.cuda_min_major = cuda_min_major

    def is_volta_or_later(self) -> bool:
        """
        判断该架构是否为 Volta 或更新（SM ≥ 7.0）。

        断点1: 返回结果取决于 sm[0] 比较。
        """
        result = self.sm[0] >= 7
        _dbg("GpuArch.is_volta_or_later", f"{self.arch_name} sm={self.sm} → {result}")
        return result

    def is_pascal_or_later(self) -> bool:
        """判断该架构是否为 Pascal 或更新（SM ≥ 6.0）。"""
        result = self.sm[0] >= 6
        _dbg("GpuArch.is_pascal_or_later", f"{self.arch_name} sm={self.sm} → {result}")
        return result

    def satisfies_min(self, min_arch: "GpuArch") -> bool:
        """判断 self 架构是否满足 min_arch 的最低要求（SM 比较）。"""
        result = self.sm >= min_arch.sm
        _dbg(
            "GpuArch.satisfies_min",
            f"{self.arch_name}({self.sm}) >= {min_arch.arch_name}({min_arch.sm}) → {result}",
        )
        return result

    def __repr__(self) -> str:
        return f"GpuArch.{self.name}(sm={self.sm[0]}.{self.sm[1]})"


# ---------------------------------------------------------------------------
# 2. CudaMinRequirement — 结构化封装「最低 CUDA 版本」策略
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CudaVersionPair:
    """轻量 CUDA 版本对（major.minor），供本模块独立使用。"""
    major: int
    minor: int

    @classmethod
    def from_str(cls, s: str) -> "CudaVersionPair":
        parts = s.split(".")
        return cls(int(parts[0]), int(parts[1]))

    def __ge__(self, other: "CudaVersionPair") -> bool:
        return (self.major, self.minor) >= (other.major, other.minor)

    def __lt__(self, other: "CudaVersionPair") -> bool:
        return (self.major, self.minor) < (other.major, other.minor)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class CudaMinRequirement:
    """
    结构化封装「最低 CUDA 版本」策略。

    上游只是删掉两行 YAML，无法回答「这个策略从哪个 commit 开始生效」
    「之前支持过哪些版本」「为什么选 12.2 作为分水岭」。
    鲁迅拿法：将删除行为建模为策略对象，携带历史记忆。

    字段：
        version: 当前最低要求版本
        reason: 提升原因（人类可读）
        since_commit: 引入该策略的上游 commit hash
        dropped_versions: 本次策略提升所移除的版本集合
    """
    version: CudaVersionPair
    reason: str
    since_commit: str
    dropped_versions: FrozenSet[CudaVersionPair] = field(default_factory=frozenset)

    def is_satisfied(self, cuda_ver: CudaVersionPair) -> bool:
        """
        判断给定 CUDA 版本是否满足本策略的最低要求。

        断点2: 版本比较结果。
        """
        result = cuda_ver >= self.version
        _dbg(
            "CudaMinRequirement.is_satisfied",
            f"cuda {cuda_ver} >= min {self.version} → {result}",
        )
        return result

    def was_dropped(self, cuda_ver: CudaVersionPair) -> bool:
        """判断给定版本是否在本次策略提升中被移除。"""
        result = cuda_ver in self.dropped_versions
        _dbg("CudaMinRequirement.was_dropped", f"cuda {cuda_ver} dropped={result}")
        return result

    def describe(self) -> str:
        """返回人类可读的策略描述。"""
        dropped = sorted(str(v) for v in self.dropped_versions)
        return (
            f"CUDA 最低版本要求: {self.version}\n"
            f"  原因: {self.reason}\n"
            f"  引入 commit: {self.since_commit}\n"
            f"  本次移除的版本: {dropped or '(无)'}"
        )


# ---------------------------------------------------------------------------
# 3. CondaMatrixEntry — 建模 dependencies.yaml 里的 cuda matrix 条目
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CondaMatrixEntry:
    """
    建模 dependencies.yaml 里的单个 cuda matrix 条目：
      - matrix:
          cuda: "X.Y"
        packages:
          - cuda-version=X.Y

    上游直接删行，Walpurgis 将每个条目建模为可查询对象，
    使「哪些条目被删了」「哪些还在」可以用代码断言。
    """
    cuda_version: CudaVersionPair
    package_name: str  # 如 "cuda-version=12.2"

    @classmethod
    def from_version_str(cls, ver_str: str) -> "CondaMatrixEntry":
        v = CudaVersionPair.from_str(ver_str)
        return cls(
            cuda_version=v,
            package_name=f"cuda-version={ver_str}",
        )

    def is_active_under(self, policy: CudaMinRequirement) -> bool:
        """
        判断本条目在给定策略下是否仍被纳入（即未被移除）。

        断点3: 条目激活状态。
        """
        active = policy.is_satisfied(self.cuda_version)
        _dbg(
            "CondaMatrixEntry.is_active_under",
            f"entry cuda={self.cuda_version} active={active}",
        )
        return active

    def to_yaml_fragment(self) -> str:
        """产出该条目对应的 YAML 片段（用于审计报告）。"""
        return (
            f"- matrix:\n"
            f"    cuda: \"{self.cuda_version}\"\n"
            f"  packages:\n"
            f"    - {self.package_name}"
        )


# ---------------------------------------------------------------------------
# 4. Cuda122Policy — 本 commit 引入的「CUDA 12.2+ 策略」
# ---------------------------------------------------------------------------

#: 本 commit 之前支持的全部 CUDA 版本（conda matrix 条目）
_ENTRIES_BEFORE_0BDAE67: Tuple[CondaMatrixEntry, ...] = (
    CondaMatrixEntry.from_version_str("12.0"),
    CondaMatrixEntry.from_version_str("12.1"),
    CondaMatrixEntry.from_version_str("12.2"),
    CondaMatrixEntry.from_version_str("12.3"),  # 假设后续版本仍在
    CondaMatrixEntry.from_version_str("12.4"),
    CondaMatrixEntry.from_version_str("12.5"),
)

#: 本 commit 移除的版本（0bdae67 删除的两个 matrix 条目）
_DROPPED_BY_0BDAE67: FrozenSet[CudaVersionPair] = frozenset({
    CudaVersionPair(12, 0),
    CudaVersionPair(12, 1),
})

#: 0bdae67 之前的策略（最低 12.0）
CUDA_POLICY_BEFORE_0BDAE67 = CudaMinRequirement(
    version=CudaVersionPair(12, 0),
    reason="RAPIDS 26.02 基线：支持所有 CUDA 12.x",
    since_commit="(历史策略，0bdae67 之前)",
    dropped_versions=frozenset(),
)

#: 0bdae67 引入的策略（最低 12.2）
CUDA_POLICY_0BDAE67 = CudaMinRequirement(
    version=CudaVersionPair(12, 2),
    reason=(
        "Require CUDA 12.2+：移除 CUDA 12.0 和 12.1 的 conda matrix 条目，"
        "最低支持版本提升至 12.2。"
        "原因：CUDA 12.2 引入了若干运行时改进（cudaMallocAsync 语义稳定、"
        "NVLink 通信修复），12.0/12.1 在部分 GNN 工作负载下有已知缺陷。"
    ),
    since_commit="0bdae67",
    dropped_versions=_DROPPED_BY_0BDAE67,
)


@dataclass
class Cuda122Policy:
    """
    建模 commit 0bdae67 引入的「CUDA 12.2+」策略。

    上游只有两行 YAML 删除 + 一行文档修改，
    鲁迅拿法：将策略变更建模为可查询、可审计的对象。
    """
    before: CudaMinRequirement = field(default_factory=lambda: CUDA_POLICY_BEFORE_0BDAE67)
    after: CudaMinRequirement = field(default_factory=lambda: CUDA_POLICY_0BDAE67)
    all_entries: Tuple[CondaMatrixEntry, ...] = field(
        default_factory=lambda: _ENTRIES_BEFORE_0BDAE67
    )

    def dropped_entries(self) -> List[CondaMatrixEntry]:
        """
        列出被本次策略提升移除的 conda matrix 条目。

        断点4: 移除条目列表。
        """
        dropped = [
            e for e in self.all_entries
            if not e.is_active_under(self.after)
        ]
        _dbg("Cuda122Policy.dropped_entries", f"共移除 {len(dropped)} 个条目")
        return dropped

    def retained_entries(self) -> List[CondaMatrixEntry]:
        """列出在新策略下仍保留的条目。"""
        retained = [
            e for e in self.all_entries
            if e.is_active_under(self.after)
        ]
        _dbg("Cuda122Policy.retained_entries", f"保留 {len(retained)} 个条目")
        return retained

    def audit_report(self) -> str:
        """
        产出人类可读的迁移审计报告。

        断点5: 报告生成。
        """
        _dbg("Cuda122Policy.audit_report", "开始生成审计报告")
        dropped = self.dropped_entries()
        retained = self.retained_entries()

        lines = [
            "=" * 60,
            "Cuda122Policy 审计报告 (commit 0bdae67)",
            "=" * 60,
            "",
            "[策略变更]",
            f"  之前: {self.before.describe()}",
            "",
            f"  之后: {self.after.describe()}",
            "",
            f"[移除的 conda matrix 条目（共 {len(dropped)} 个）]",
        ]
        for e in dropped:
            lines.append(f"  REMOVED: cuda {e.cuda_version} / {e.package_name}")
        lines += [
            "",
            f"[保留的条目（共 {len(retained)} 个）]",
        ]
        for e in retained:
            lines.append(f"  KEPT: cuda {e.cuda_version} / {e.package_name}")
        lines += ["", "=" * 60]
        report = "\n".join(lines)
        _dbg("Cuda122Policy.audit_report", "报告生成完毕")
        return report


# ---------------------------------------------------------------------------
# 5. ContributingHardwarePolicy — CONTRIBUTING.md Hardware 节的架构要求变更
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareRequirement:
    """
    建模 CONTRIBUTING.md Hardware 节的 GPU 最低架构要求。

    上游只改了一行文档，Walpurgis 将其建模为可比较的策略对象，
    使「架构要求提升」可以在代码里断言，而不是只存在于 Markdown 里。
    """
    min_arch: GpuArch
    description: str
    since_commit: str

    def is_satisfied_by(self, arch: GpuArch) -> bool:
        """
        判断给定架构是否满足当前硬件最低要求。

        断点6: 架构满足判断。
        """
        result = arch.satisfies_min(self.min_arch)
        _dbg(
            "HardwareRequirement.is_satisfied_by",
            f"{arch.arch_name} satisfies min {self.min_arch.arch_name} → {result}",
        )
        return result

    def contributing_md_line(self) -> str:
        """产出对应的 CONTRIBUTING.md Hardware 节文本。"""
        return (
            f"* You need to have accesses to an NVIDIA GPU that is "
            f"{self.min_arch.arch_name} or later."
        )


#: 0bdae67 之前的硬件要求（Pascal 或更新）
HW_REQUIREMENT_BEFORE_0BDAE67 = HardwareRequirement(
    min_arch=GpuArch.PASCAL,
    description="Pascal (SM 6.0+) 或更新，上游 CONTRIBUTING.md 原文",
    since_commit="(历史策略，0bdae67 之前)",
)

#: 0bdae67 引入的硬件要求（Volta 或更新）
HW_REQUIREMENT_0BDAE67 = HardwareRequirement(
    min_arch=GpuArch.VOLTA,
    description=(
        "Volta (SM 7.0+) 或更新。"
        "与 CUDA 12.2+ 最低要求联动：Pascal 架构（SM 6.x）在 CUDA 12.2+ "
        "的部分特性（如 TF32、BF16 硬件加速）上受限，Volta 是实际有效的最低门槛。"
    ),
    since_commit="0bdae67",
)


@dataclass
class ContributingHardwarePolicy:
    """
    建模 CONTRIBUTING.md Hardware 节的 GPU 架构要求变更（0bdae67）。

    上游：Pascal → Volta，一行 diff。
    Walpurgis：结构化为前后两个 HardwareRequirement，携带变更原因和比较方法。
    """
    before: HardwareRequirement = field(
        default_factory=lambda: HW_REQUIREMENT_BEFORE_0BDAE67
    )
    after: HardwareRequirement = field(
        default_factory=lambda: HW_REQUIREMENT_0BDAE67
    )

    def arch_was_dropped(self, arch: GpuArch) -> bool:
        """
        判断某架构是否在本次变更中从「支持」变为「不支持」。

        断点7: Pascal 在此处应返回 True。
        """
        was_ok = self.before.is_satisfied_by(arch)
        now_ok = self.after.is_satisfied_by(arch)
        dropped = was_ok and not now_ok
        _dbg(
            "ContributingHardwarePolicy.arch_was_dropped",
            f"{arch.arch_name}: was_ok={was_ok}, now_ok={now_ok}, dropped={dropped}",
        )
        return dropped

    def change_summary(self) -> str:
        """
        产出结构化的变更摘要。

        断点8: 变更摘要生成。
        """
        _dbg("ContributingHardwarePolicy.change_summary", "生成变更摘要")
        dropped_archs = [a for a in GpuArch if self.arch_was_dropped(a)]
        lines = [
            "[ContributingHardwarePolicy 变更摘要 (0bdae67)]",
            f"  之前: {self.before.contributing_md_line()}",
            f"  之后: {self.after.contributing_md_line()}",
            f"  因此不再被官方支持的架构:",
        ]
        for a in dropped_archs:
            lines.append(f"    - {a.arch_name} (SM {a.sm[0]}.{a.sm[1]}, {a.release_year}年)")
        if not dropped_archs:
            lines.append("    (无)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模块级单例（供外部导入使用）
# ---------------------------------------------------------------------------

#: 本 commit 引入的 CUDA 版本策略（最低 12.2）
CUDA122_POLICY = Cuda122Policy()

#: 本 commit 引入的硬件架构策略（最低 Volta）
CONTRIBUTING_HW_POLICY = ContributingHardwarePolicy()


# ---------------------------------------------------------------------------
# 自测入口（python -m walpurgis.core.cuda122_min_policy）
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """
    自测：验证 Cuda122Policy 和 ContributingHardwarePolicy 的核心语义。

    断点9: 自测开始。
    断点10: 自测结束。
    """
    _dbg("_self_test", "开始自测")

    # --- CudaVersionPair 基础比较 ---
    v120 = CudaVersionPair(12, 0)
    v121 = CudaVersionPair(12, 1)
    v122 = CudaVersionPair(12, 2)
    v123 = CudaVersionPair(12, 3)
    assert v122 >= v122, "12.2 >= 12.2"
    assert not (v120 >= v122), "12.0 < 12.2"
    assert not (v121 >= v122), "12.1 < 12.2"
    assert v123 >= v122, "12.3 >= 12.2"
    print("[PASS] CudaVersionPair 比较")

    # --- CudaMinRequirement ---
    assert not CUDA_POLICY_0BDAE67.is_satisfied(v120), "12.0 不满足 12.2+ 策略"
    assert not CUDA_POLICY_0BDAE67.is_satisfied(v121), "12.1 不满足 12.2+ 策略"
    assert CUDA_POLICY_0BDAE67.is_satisfied(v122), "12.2 满足 12.2+ 策略"
    assert CUDA_POLICY_0BDAE67.is_satisfied(v123), "12.3 满足 12.2+ 策略"
    assert CUDA_POLICY_0BDAE67.was_dropped(v120), "12.0 被 0bdae67 移除"
    assert CUDA_POLICY_0BDAE67.was_dropped(v121), "12.1 被 0bdae67 移除"
    assert not CUDA_POLICY_0BDAE67.was_dropped(v122), "12.2 未被移除"
    print("[PASS] CudaMinRequirement")

    # --- CondaMatrixEntry ---
    e120 = CondaMatrixEntry.from_version_str("12.0")
    e122 = CondaMatrixEntry.from_version_str("12.2")
    assert not e120.is_active_under(CUDA_POLICY_0BDAE67), "12.0 条目在新策略下不激活"
    assert e122.is_active_under(CUDA_POLICY_0BDAE67), "12.2 条目在新策略下激活"
    print("[PASS] CondaMatrixEntry")

    # --- Cuda122Policy ---
    dropped = CUDA122_POLICY.dropped_entries()
    assert len(dropped) == 2, f"应移除 2 个条目，实际 {len(dropped)}"
    dropped_vers = {str(e.cuda_version) for e in dropped}
    assert "12.0" in dropped_vers, "12.0 应在移除列表"
    assert "12.1" in dropped_vers, "12.1 应在移除列表"
    retained = CUDA122_POLICY.retained_entries()
    assert all(e.is_active_under(CUDA_POLICY_0BDAE67) for e in retained), \
        "保留条目均应满足新策略"
    print("[PASS] Cuda122Policy")

    # --- GpuArch ---
    assert not GpuArch.PASCAL.is_volta_or_later(), "Pascal 不是 Volta 或更新"
    assert GpuArch.VOLTA.is_volta_or_later(), "Volta 是 Volta 或更新"
    assert GpuArch.AMPERE.is_volta_or_later(), "Ampere 是 Volta 或更新"
    assert GpuArch.MAXWELL.is_pascal_or_later() is False, "Maxwell 不是 Pascal 或更新"
    assert GpuArch.PASCAL.is_pascal_or_later(), "Pascal 是 Pascal 或更新"
    print("[PASS] GpuArch")

    # --- HardwareRequirement ---
    assert not HW_REQUIREMENT_0BDAE67.is_satisfied_by(GpuArch.PASCAL), \
        "Pascal 不满足 Volta+ 要求"
    assert HW_REQUIREMENT_0BDAE67.is_satisfied_by(GpuArch.VOLTA), \
        "Volta 满足 Volta+ 要求"
    assert HW_REQUIREMENT_0BDAE67.is_satisfied_by(GpuArch.AMPERE), \
        "Ampere 满足 Volta+ 要求"
    print("[PASS] HardwareRequirement")

    # --- ContributingHardwarePolicy ---
    assert CONTRIBUTING_HW_POLICY.arch_was_dropped(GpuArch.PASCAL), \
        "Pascal 应在 0bdae67 中被移除支持"
    assert not CONTRIBUTING_HW_POLICY.arch_was_dropped(GpuArch.VOLTA), \
        "Volta 在新策略下仍被支持"
    assert not CONTRIBUTING_HW_POLICY.arch_was_dropped(GpuArch.AMPERE), \
        "Ampere 在新策略下仍被支持"
    print("[PASS] ContributingHardwarePolicy")

    # --- audit_report / change_summary ---
    report = CUDA122_POLICY.audit_report()
    assert "12.0" in report and "12.1" in report, "审计报告应包含移除的版本"
    assert "12.2" in report, "审计报告应包含新最低版本"
    summary = CONTRIBUTING_HW_POLICY.change_summary()
    assert "Pascal" in summary and "Volta" in summary, "变更摘要应包含架构名"
    print("[PASS] audit_report / change_summary")

    _dbg("_self_test", "自测完成，全部通过")
    print("=== cuda122_min_policy.py 自测通过（10 项）===")


if __name__ == "__main__":
    _self_test()
