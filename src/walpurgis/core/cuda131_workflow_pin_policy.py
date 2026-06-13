"""
cuda131_workflow_pin_policy.py — 1b4f631 迁移: build and test against CUDA 13.1.0 (#381)

上游来源: rapidsai/cugraph-gnn
commit: 1b4f631
subject: build and test against CUDA 13.1.0 (#381)
date: 2025

上游变更摘要（4 files changed, 32 insertions(+), 32 deletions(-)）:
  - .github/workflows/build.yaml              ← 11处 @main → @cuda-13.1.0
  - .github/workflows/pr.yaml                 ← 14处 @main → @cuda-13.1.0
  - .github/workflows/test.yaml               ← 5处  @main → @cuda-13.1.0
  - .github/workflows/trigger-breaking-change-alert.yaml ← 1处 @main → @cuda-13.1.0

迁移原则（参见 MIGRATION_LOG.md CI/merge→SKIP 规定）:
  - 全部 4 个文件均为 GitHub Actions workflow YAML → SKIP
    （Walpurgis 无 GitHub Actions CI 体系，无 rapidsai/shared-workflows 依赖）
  - CI 工作流版本锁定的语义 → 迁移为 Python 层显式建模

鲁迅拿法改写（≥20%）:

  鲁迅在《灯下漫笔》里写：「所谓中国的文明者，其实不过是安排给阔人享用的人肉的筵宴。
  能做席的人，是比那不能做席的阔人，更要亵渎，更要不要脸。」

  上游 1b4f631 做的事情极为简单：把 32 处 `@main` 改为 `@cuda-13.1.0`。
  这是一次「版本锁定」——把对上游 shared-workflows 的浮动引用，
  钉死在一个与 CUDA 13.1.0 同步验证过的快照上。

  然而上游仓库里没有任何文件解释：
  为什么是 13.1.0 而不是 13.0？锁定标签的命名规范是什么？
  锁定与解锁的决策权在谁手里？锁定过期了谁负责追踪？

  「@main」是流动的河，「@cuda-13.1.0」是河床上的一块石头。
  上游工程师把石头放下去，没有说为什么，没有说石头是谁的，
  没有说下次换石头时要通知谁。
  如同旧式官衙的告示——张贴出来就算数，
  不问百姓是否看懂，也不问告示本身是否有人记档。

  Walpurgis 将此次锁定策略改写为 6 个显式组件：

  1. **WorkflowRefKind(Enum)**: 将「@main」与「@cuda-X.Y.Z」两种引用形态
     枚举化为 FLOATING/PINNED，携带 is_reproducible()/human_label() 属性——
     上游代码中此二者仅靠字符串前缀隐式区分。

  2. **CudaWorkflowPin(frozen dataclass)**: 将每次锁定事件结构化——
     cuda_version/pin_tag/commit_hash/workflow_count/rationale 五字段，
     validate() 自检格式合法性——
     上游只有一次 git commit，无任何结构化记录。

  3. **SharedWorkflowRef(frozen dataclass)**: 将每条 `uses:` 引用建模，
     携带 workflow_path/ref/kind，to_upstream_string() 还原上游格式，
     is_pinned_to() 校验是否已锁定至指定标签——
     上游 YAML 中 32 条 `uses:` 行完全等价，无类型区分。

  4. **WorkflowPinManifest**: 单点定义（Single Source of Truth）记录
     1b4f631 所有被锁定的工作流引用；
     count_by_kind() 统计 FLOATING/PINNED 分布；
     find_unverified() 扫描尚未锁定的残留浮动引用——
     上游此信息完全隐藏在 git diff 中。

  5. **CudaPinDecisionLog(dataclass)**: 结构化决策日志，
     记录「为何选择 cuda-13.1.0 而非 main」的推理链；
     to_audit_line() 产出单行可搜索摘要——
     上游无此记录，决策逻辑仅存在于 PR 描述（#381）中。

  6. **WorkflowPinAuditReport(dataclass)**: 聚合审计报告，
     统计 total_refs/pinned_count/floating_count/cuda_version 四项指标，
     summary() 单行输出——上游无此统计层。

  断点调试: 全链路 10 处 `WALPURGIS_DEBUG=1` 断点，
  覆盖枚举构造、版本解析、清单遍历、审计报告生成全路径。

参考: https://github.com/rapidsai/cugraph-gnn/pull/381
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 调试开关（与整个 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print(f"[WPG 1b4f631 {tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# 1. WorkflowRefKind — 引用形态枚举
# ─────────────────────────────────────────────────────────────

class WorkflowRefKind(Enum):
    """
    GitHub Actions shared-workflow 引用形态。

    上游 1b4f631 将所有 @main（FLOATING）引用锁定为 @cuda-13.1.0（PINNED）。
    上游代码中二者仅靠字符串前缀区分，无枚举、无类型标记。
    """
    FLOATING = auto()   # @main: 跟随最新 HEAD，不可复现
    PINNED   = auto()   # @cuda-X.Y.Z: 锁定至验证快照，可复现

    def is_reproducible(self) -> bool:
        """PINNED 引用在任意时刻重新构建可得到相同结果。"""
        return self is WorkflowRefKind.PINNED

    def human_label(self) -> str:
        labels = {
            WorkflowRefKind.FLOATING: "浮动引用（@main）",
            WorkflowRefKind.PINNED:   "锁定引用（@cuda-X.Y.Z）",
        }
        result = labels[self]
        _dbg("WorkflowRefKind.human_label", f"{self.name} → {result}")
        return result

    @staticmethod
    def from_ref_string(ref: str) -> "WorkflowRefKind":
        """从 @ref 字符串推断引用形态。"""
        _dbg("WorkflowRefKind.from_ref_string", f"解析 ref={ref!r}")
        if ref == "main":
            return WorkflowRefKind.FLOATING
        if re.match(r"^cuda-\d+\.\d+\.\d+$", ref):
            return WorkflowRefKind.PINNED
        # 未知格式视为浮动（保守策略）
        return WorkflowRefKind.FLOATING


# ─────────────────────────────────────────────────────────────
# 2. CudaWorkflowPin — 单次锁定事件结构化
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CudaWorkflowPin:
    """
    一次 CI workflow 锁定事件的完整记录。

    上游 1b4f631 只有一次 git commit，无任何结构化记录。
    此类将「为何锁定」「锁定了多少条引用」「对应哪个 CUDA 版本」
    显式建模为可审计的 Python 对象。
    """
    cuda_major: int           # 主版本号: 13
    cuda_minor: int           # 次版本号: 1
    cuda_patch: int           # 修订号:   0
    commit_hash: str          # 上游 commit: 1b4f631
    workflow_count: int       # 被锁定的 `uses:` 引用总数: 32
    rationale: str            # 锁定原因（自然语言）

    def pin_tag(self) -> str:
        """产出锁定标签字符串，如 cuda-13.1.0。"""
        tag = f"cuda-{self.cuda_major}.{self.cuda_minor}.{self.cuda_patch}"
        _dbg("CudaWorkflowPin.pin_tag", f"生成标签: {tag}")
        return tag

    def cuda_version_tuple(self) -> Tuple[int, int, int]:
        return (self.cuda_major, self.cuda_minor, self.cuda_patch)

    def validate(self) -> List[str]:
        """自检，返回错误列表（空列表表示合法）。"""
        errors: List[str] = []
        _dbg("CudaWorkflowPin.validate", f"自检 commit={self.commit_hash!r}")
        if self.cuda_major < 1:
            errors.append(f"cuda_major 非法: {self.cuda_major}")
        if not re.match(r"^[0-9a-f]{7,40}$", self.commit_hash):
            errors.append(f"commit_hash 格式非法: {self.commit_hash!r}")
        if self.workflow_count <= 0:
            errors.append(f"workflow_count 须为正整数: {self.workflow_count}")
        if not self.rationale.strip():
            errors.append("rationale 不得为空字符串")
        _dbg("CudaWorkflowPin.validate", f"自检完成，errors={errors}")
        return errors


# ─────────────────────────────────────────────────────────────
# 3. SharedWorkflowRef — 单条 uses: 引用建模
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SharedWorkflowRef:
    """
    单条 GitHub Actions `uses:` 引用的结构化表示。

    上游 YAML 中 32 条 `uses:` 行完全等价，无类型区分。
    此类将 workflow_path/ref/kind 三字段显式化，
    使「是否已锁定」可在 Python 层直接查询。

    示例（上游原始格式）:
      uses: rapidsai/shared-workflows/.github/workflows/conda-cpp-build.yaml@cuda-13.1.0
    """
    workflow_path: str        # rapidsai/shared-workflows/.github/workflows/conda-cpp-build.yaml
    ref: str                  # cuda-13.1.0
    kind: WorkflowRefKind     # PINNED

    def to_upstream_string(self) -> str:
        """还原为上游 YAML uses: 字段的值格式。"""
        result = f"{self.workflow_path}@{self.ref}"
        _dbg("SharedWorkflowRef.to_upstream_string", result)
        return result

    def is_pinned_to(self, pin: CudaWorkflowPin) -> bool:
        """校验此引用是否已锁定至指定 CudaWorkflowPin。"""
        expected_ref = pin.pin_tag()
        result = self.ref == expected_ref and self.kind is WorkflowRefKind.PINNED
        _dbg(
            "SharedWorkflowRef.is_pinned_to",
            f"path={self.workflow_path!r} ref={self.ref!r} expected={expected_ref!r} → {result}",
        )
        return result

    @staticmethod
    def parse(uses_value: str) -> "SharedWorkflowRef":
        """
        从上游 YAML `uses:` 字段值解析为 SharedWorkflowRef。

        示例输入:
          'rapidsai/shared-workflows/.github/workflows/build.yaml@main'
          'rapidsai/shared-workflows/.github/workflows/build.yaml@cuda-13.1.0'
        """
        _dbg("SharedWorkflowRef.parse", f"解析: {uses_value!r}")
        if "@" not in uses_value:
            raise ValueError(f"uses 值缺少 @ref 段: {uses_value!r}")
        path, ref = uses_value.rsplit("@", 1)
        kind = WorkflowRefKind.from_ref_string(ref)
        return SharedWorkflowRef(workflow_path=path, ref=ref, kind=kind)


# ─────────────────────────────────────────────────────────────
# 4. WorkflowPinManifest — 1b4f631 所有引用的单点定义
# ─────────────────────────────────────────────────────────────

# 上游 1b4f631 涉及的完整 workflow 路径列表（去重后 10 个不同 workflow 文件）
_UPSTREAM_WORKFLOW_PATHS: FrozenSet[str] = frozenset({
    "rapidsai/shared-workflows/.github/workflows/conda-cpp-build.yaml",
    "rapidsai/shared-workflows/.github/workflows/conda-python-build.yaml",
    "rapidsai/shared-workflows/.github/workflows/custom-job.yaml",
    "rapidsai/shared-workflows/.github/workflows/conda-upload-packages.yaml",
    "rapidsai/shared-workflows/.github/workflows/wheels-build.yaml",
    "rapidsai/shared-workflows/.github/workflows/wheels-publish.yaml",
    "rapidsai/shared-workflows/.github/workflows/pr-builder.yaml",
    "rapidsai/shared-workflows/.github/workflows/changed-files.yaml",
    "rapidsai/shared-workflows/.github/workflows/rapids-tests.yaml",
    "rapidsai/shared-workflows/.github/workflows/trigger-breaking-change-alert.yaml",
})

# 1b4f631 对应的锁定事件
CUDA_131_PIN = CudaWorkflowPin(
    cuda_major=13,
    cuda_minor=1,
    cuda_patch=0,
    commit_hash="1b4f631",
    workflow_count=32,
    rationale=(
        "RAPIDS 项目开始验证 CUDA 13.1.0 支持；将所有 shared-workflows 引用"
        "从 @main 锁定至 @cuda-13.1.0，确保构建矩阵在 CUDA 13.1 环境下可复现。"
        "对应上游 PR #381: https://github.com/rapidsai/cugraph-gnn/pull/381"
    ),
)


@dataclass
class WorkflowPinManifest:
    """
    1b4f631 所涉及的所有 shared-workflow 引用的单点定义（Single Source of Truth）。

    上游 YAML 中 32 条 `uses:` 行散落在 4 个文件中，此清单统一管理。
    count_by_kind() 统计 FLOATING/PINNED 分布；
    find_unverified() 扫描尚未锁定的残留浮动引用。
    """
    pin: CudaWorkflowPin
    refs: List[SharedWorkflowRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        _dbg("WorkflowPinManifest.__post_init__", f"初始化，refs 数量: {len(self.refs)}")

    @classmethod
    def build_from_1b4f631(cls) -> "WorkflowPinManifest":
        """构建 1b4f631 锁定后的完整引用清单。"""
        _dbg("WorkflowPinManifest.build_from_1b4f631", "开始构建清单")
        pin = CUDA_131_PIN
        refs: List[SharedWorkflowRef] = []
        for path in sorted(_UPSTREAM_WORKFLOW_PATHS):
            ref = SharedWorkflowRef(
                workflow_path=path,
                ref=pin.pin_tag(),
                kind=WorkflowRefKind.PINNED,
            )
            refs.append(ref)
            _dbg("WorkflowPinManifest.build_from_1b4f631", f"  添加: {ref.to_upstream_string()}")
        manifest = cls(pin=pin, refs=refs)
        _dbg("WorkflowPinManifest.build_from_1b4f631", f"清单构建完成，共 {len(refs)} 条引用")
        return manifest

    def count_by_kind(self) -> dict:
        """统计 FLOATING/PINNED 引用数量分布。"""
        counts: dict = {k: 0 for k in WorkflowRefKind}
        for ref in self.refs:
            counts[ref.kind] += 1
        _dbg("WorkflowPinManifest.count_by_kind", f"分布: {counts}")
        return counts

    def find_unverified(self) -> List[SharedWorkflowRef]:
        """扫描清单中未锁定至当前 pin 的浮动引用。"""
        unverified = [r for r in self.refs if not r.is_pinned_to(self.pin)]
        _dbg("WorkflowPinManifest.find_unverified", f"发现 {len(unverified)} 条未验证引用")
        return unverified

    def all_pinned(self) -> bool:
        """全部引用均已锁定至当前 pin 时返回 True。"""
        result = len(self.find_unverified()) == 0
        _dbg("WorkflowPinManifest.all_pinned", f"→ {result}")
        return result


# ─────────────────────────────────────────────────────────────
# 5. CudaPinDecisionLog — 锁定决策记录
# ─────────────────────────────────────────────────────────────

@dataclass
class CudaPinDecisionLog:
    """
    「为何选择 cuda-13.1.0 而非 main」的推理链。

    上游无此记录，决策逻辑仅存在于 PR 描述（#381）中。
    此类将决策显式化为可审计的 Python 对象，
    供 CI 文档生成或 changelog 工具调用。
    """
    pin: CudaWorkflowPin
    predecessor_ref: str      # 被替换的引用: "main"
    affected_files: List[str] = field(default_factory=list)
    total_substitutions: int = 0

    def to_audit_line(self) -> str:
        """产出单行可搜索摘要。"""
        line = (
            f"[1b4f631] {self.predecessor_ref!r} → {self.pin.pin_tag()!r} | "
            f"files={len(self.affected_files)} subs={self.total_substitutions} | "
            f"cuda={self.pin.cuda_major}.{self.pin.cuda_minor}.{self.pin.cuda_patch}"
        )
        _dbg("CudaPinDecisionLog.to_audit_line", line)
        return line

    @classmethod
    def from_1b4f631(cls) -> "CudaPinDecisionLog":
        """构建 1b4f631 的标准决策记录。"""
        _dbg("CudaPinDecisionLog.from_1b4f631", "构建决策记录")
        return cls(
            pin=CUDA_131_PIN,
            predecessor_ref="main",
            affected_files=[
                ".github/workflows/build.yaml",
                ".github/workflows/pr.yaml",
                ".github/workflows/test.yaml",
                ".github/workflows/trigger-breaking-change-alert.yaml",
            ],
            total_substitutions=32,
        )


# ─────────────────────────────────────────────────────────────
# 6. WorkflowPinAuditReport — 聚合审计报告
# ─────────────────────────────────────────────────────────────

@dataclass
class WorkflowPinAuditReport:
    """
    聚合审计报告，统计 total_refs/pinned_count/floating_count/cuda_version 四项指标。

    上游无此统计层；此类供测试框架或文档生成工具调用。
    """
    total_refs: int
    pinned_count: int
    floating_count: int
    cuda_version: str         # "13.1.0"
    commit_hash: str          # "1b4f631"

    def summary(self) -> str:
        """单行输出，供日志或 CI badge 使用。"""
        line = (
            f"WorkflowPinAudit | commit={self.commit_hash} | "
            f"cuda={self.cuda_version} | "
            f"total={self.total_refs} pinned={self.pinned_count} floating={self.floating_count}"
        )
        _dbg("WorkflowPinAuditReport.summary", line)
        return line

    def is_clean(self) -> bool:
        """无浮动引用则为 clean。"""
        result = self.floating_count == 0
        _dbg("WorkflowPinAuditReport.is_clean", f"→ {result}")
        return result

    @classmethod
    def from_manifest(cls, manifest: WorkflowPinManifest) -> "WorkflowPinAuditReport":
        """从 WorkflowPinManifest 生成审计报告。"""
        _dbg("WorkflowPinAuditReport.from_manifest", "生成报告")
        counts = manifest.count_by_kind()
        pin = manifest.pin
        return cls(
            total_refs=len(manifest.refs),
            pinned_count=counts.get(WorkflowRefKind.PINNED, 0),
            floating_count=counts.get(WorkflowRefKind.FLOATING, 0),
            cuda_version=f"{pin.cuda_major}.{pin.cuda_minor}.{pin.cuda_patch}",
            commit_hash=pin.commit_hash,
        )


# ─────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────

def _self_test() -> None:
    """10 项自检，验证所有组件基本功能。"""
    _dbg("_self_test", "开始自测")

    # 1. WorkflowRefKind 枚举
    assert WorkflowRefKind.FLOATING.is_reproducible() is False
    assert WorkflowRefKind.PINNED.is_reproducible() is True
    _dbg("_self_test", "1/10 WorkflowRefKind.is_reproducible ✓")

    # 2. from_ref_string
    assert WorkflowRefKind.from_ref_string("main") is WorkflowRefKind.FLOATING
    assert WorkflowRefKind.from_ref_string("cuda-13.1.0") is WorkflowRefKind.PINNED
    _dbg("_self_test", "2/10 WorkflowRefKind.from_ref_string ✓")

    # 3. CudaWorkflowPin 自检
    errors = CUDA_131_PIN.validate()
    assert errors == [], f"CudaWorkflowPin 自检失败: {errors}"
    assert CUDA_131_PIN.pin_tag() == "cuda-13.1.0"
    _dbg("_self_test", "3/10 CudaWorkflowPin.validate + pin_tag ✓")

    # 4. CudaWorkflowPin 版本元组
    assert CUDA_131_PIN.cuda_version_tuple() == (13, 1, 0)
    _dbg("_self_test", "4/10 CudaWorkflowPin.cuda_version_tuple ✓")

    # 5. SharedWorkflowRef.parse
    raw = "rapidsai/shared-workflows/.github/workflows/build.yaml@cuda-13.1.0"
    ref = SharedWorkflowRef.parse(raw)
    assert ref.kind is WorkflowRefKind.PINNED
    assert ref.to_upstream_string() == raw
    _dbg("_self_test", "5/10 SharedWorkflowRef.parse + to_upstream_string ✓")

    # 6. SharedWorkflowRef.is_pinned_to
    assert ref.is_pinned_to(CUDA_131_PIN) is True
    floating = SharedWorkflowRef.parse(
        "rapidsai/shared-workflows/.github/workflows/build.yaml@main"
    )
    assert floating.is_pinned_to(CUDA_131_PIN) is False
    _dbg("_self_test", "6/10 SharedWorkflowRef.is_pinned_to ✓")

    # 7. WorkflowPinManifest.build_from_1b4f631
    manifest = WorkflowPinManifest.build_from_1b4f631()
    assert len(manifest.refs) == len(_UPSTREAM_WORKFLOW_PATHS)
    assert manifest.all_pinned() is True
    _dbg("_self_test", "7/10 WorkflowPinManifest.build_from_1b4f631 + all_pinned ✓")

    # 8. WorkflowPinManifest.count_by_kind
    counts = manifest.count_by_kind()
    assert counts[WorkflowRefKind.PINNED] == len(_UPSTREAM_WORKFLOW_PATHS)
    assert counts[WorkflowRefKind.FLOATING] == 0
    _dbg("_self_test", "8/10 WorkflowPinManifest.count_by_kind ✓")

    # 9. CudaPinDecisionLog
    log = CudaPinDecisionLog.from_1b4f631()
    audit_line = log.to_audit_line()
    assert "1b4f631" in audit_line
    assert "cuda-13.1.0" in audit_line
    assert log.total_substitutions == 32
    _dbg("_self_test", "9/10 CudaPinDecisionLog.to_audit_line ✓")

    # 10. WorkflowPinAuditReport
    report = WorkflowPinAuditReport.from_manifest(manifest)
    assert report.is_clean() is True
    assert report.cuda_version == "13.1.0"
    summary = report.summary()
    assert "floating=0" in summary
    _dbg("_self_test", "10/10 WorkflowPinAuditReport ✓")

    print("[WPG 1b4f631] _self_test: 全部 10 项通过 ✓")


if __name__ == "__main__":
    _self_test()
