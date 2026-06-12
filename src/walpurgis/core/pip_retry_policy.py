"""
pip_retry_policy.py — 迁移自 rapidsai/cugraph-gnn@9661018
upstream: "Use `rapids-pip-retry` in CI jobs that might need retries (#133)"

上游变更本质：4个CI脚本，`python -m pip` → `rapids-pip-retry`，
散落在 build_wheel.sh / test_wheel_cugraph-dgl.sh /
test_wheel_cugraph-pyg.sh / test_wheel_pylibwholegraph.sh。
无任何 Python 层策略抽象。

Walpurgis 将其核心语义提炼为：
  1. PipInvokeMode 枚举 — 区分直接调用 / 带重试包装 两种模式
  2. RetryPolicy dataclass — 重试次数、退避策略、触发条件（hash 不匹配等）
  3. PipCommand dataclass — 封装单次 pip 调用的完整参数集
  4. CIRetrySpec dataclass — 描述哪些 CI 阶段需要重试包装
  5. PipRetryOrchestrator — 汇总以上，simulate() 模拟执行（含全链路断点）
  6. HashMismatchSignature — 识别需要 retry 的错误特征
  7. WALPURGIS_DEBUG=1 断点（7 处）

鲁迅风格注释：横眉冷对 hash 不匹配，俯首甘为重试包。

CI/merge → SKIP：
  ci/build_wheel.sh                — RAPIDS wheel 构建脚本，Walpurgis 无 wheel 构建体系
  ci/test_wheel_cugraph-dgl.sh     — RAPIDS DGL 测试脚本，同上
  ci/test_wheel_cugraph-pyg.sh     — RAPIDS PyG 测试脚本，同上
  ci/test_wheel_pylibwholegraph.sh — RAPIDS wholegraph 测试脚本，同上
  （四个文件全部 SKIP，但核心策略语义迁入本模块）
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 调试断点工具：WALPURGIS_DEBUG=1 时激活 pdb.set_trace()
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _bp(tag: str) -> None:
    """鲁迅式断点：欲穷千里目，更上一层楼——先停在这里看清楚。"""
    if _DEBUG:
        logger.debug("[WALPURGIS_BREAKPOINT] %s", tag)
        import pdb; pdb.set_trace()  # noqa: E702  # BP-1..BP-7 入口


# ---------------------------------------------------------------------------
# 1. PipInvokeMode — 区分两种 pip 调用方式
# ---------------------------------------------------------------------------
class PipInvokeMode(Enum):
    """pip 调用模式。

    上游变更前：DIRECT（python -m pip）
    上游变更后：RETRY_WRAPPED（rapids-pip-retry）
    Walpurgis 将二者显式枚举，便于策略切换与审计。
    """
    DIRECT = auto()        # python -m pip <subcmd>  （上游变更前）
    RETRY_WRAPPED = auto() # rapids-pip-retry <subcmd>（上游变更后）


# ---------------------------------------------------------------------------
# 2. HashMismatchSignature — 识别需要 retry 的错误特征
# ---------------------------------------------------------------------------
# 上游 PR 描述中摘录的触发场景：
#   "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE"
# 这是网络抖动导致下载截断后 hash 对不上的典型特征。
HASH_MISMATCH_PATTERNS: Tuple[str, ...] = (
    r"THESE PACKAGES DO NOT MATCH THE HASHES",
    r"Expected sha256 \w+ Got \w+",
    r"hash mismatch",
    r"HashMismatch",
)


@dataclass(frozen=True)
class HashMismatchSignature:
    """封装 hash 不匹配的错误特征识别逻辑。

    鲁迅：世上本没有 hash 不匹配，网络抖动多了，也便有了。
    """
    patterns: Tuple[str, ...] = HASH_MISMATCH_PATTERNS

    def matches(self, stderr_output: str) -> bool:
        """BP-2: 判断是否命中 hash 不匹配特征。"""
        _bp("HashMismatchSignature.matches")  # BP-2
        return any(re.search(p, stderr_output) for p in self.patterns)


# ---------------------------------------------------------------------------
# 3. RetryPolicy — 重试策略
# ---------------------------------------------------------------------------
@dataclass
class RetryPolicy:
    """pip 重试策略。

    max_attempts: 最大重试次数（含首次，upstream rapids-pip-retry 默认 3）
    backoff_seconds: 每次重试前等待秒数（指数退避基数）
    retriable_signatures: 触发重试的错误特征列表
    """
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    retriable_signatures: List[HashMismatchSignature] = field(
        default_factory=lambda: [HashMismatchSignature()]
    )

    def should_retry(self, attempt: int, stderr_output: str) -> bool:
        """BP-3: 判断当前失败是否应重试。"""
        _bp("RetryPolicy.should_retry")  # BP-3
        if attempt >= self.max_attempts:
            logger.warning(
                "已达最大重试次数 %d，放弃重试。鲁迅曰：不在沉默中爆发，就在沉默中灭亡。",
                self.max_attempts,
            )
            return False
        return any(sig.matches(stderr_output) for sig in self.retriable_signatures)

    def backoff_for(self, attempt: int) -> float:
        """指数退避：第 attempt 次重试等待 backoff_seconds * 2^(attempt-1) 秒。"""
        return self.backoff_seconds * (2 ** (attempt - 1))


# ---------------------------------------------------------------------------
# 4. PipCommand — 封装单次 pip 调用参数
# ---------------------------------------------------------------------------
@dataclass
class PipCommand:
    """单次 pip 调用的完整参数集。

    subcommand: 'wheel' / 'install'
    mode: PipInvokeMode
    args: 额外参数列表（--extra-index-url、--find-links 等）
    target_packages: 目标包列表
    """
    subcommand: str                        # 'wheel' 或 'install'
    mode: PipInvokeMode = PipInvokeMode.RETRY_WRAPPED
    args: List[str] = field(default_factory=list)
    target_packages: List[str] = field(default_factory=list)
    verbose: bool = True

    def to_cli_tokens(self) -> List[str]:
        """BP-4: 生成 CLI token 列表（仅用于模拟/审计，不实际执行）。"""
        _bp("PipCommand.to_cli_tokens")  # BP-4
        if self.mode is PipInvokeMode.RETRY_WRAPPED:
            prefix = ["rapids-pip-retry"]
        else:
            prefix = ["python", "-m", "pip"]

        tokens = prefix + [self.subcommand]
        if self.verbose:
            tokens += ["-v"]
        tokens += self.args
        tokens += self.target_packages
        return tokens

    def describe(self) -> str:
        return " ".join(self.to_cli_tokens())


# ---------------------------------------------------------------------------
# 5. CIRetrySpec — 描述哪些 CI 阶段需要重试包装
# ---------------------------------------------------------------------------
@dataclass
class CIRetrySpec:
    """描述一个 CI 阶段是否需要 rapids-pip-retry 包装。

    上游 9661018 涉及四个脚本，每个对应一个 CIRetrySpec：
      build_wheel           : subcommand=wheel,   需要重试（wheel 构建时下载大包）
      test_wheel_cugraph_dgl: subcommand=install, 需要重试（PyTorch/DGL 包大）
      test_wheel_cugraph_pyg: subcommand=install, 需要重试（PyTorch/PyG 包大）
      test_wheel_pylibwholegraph: subcommand=install, 需要重试（wholegraph 包大）

    Walpurgis SKIP 说明：以上四个脚本均为 RAPIDS CI 专属，
    Walpurgis 无 wheel 构建体系，故均 SKIP。
    但本 dataclass 保留语义，供未来 Walpurgis 自有 CI 扩展参考。
    """
    stage_name: str
    script_path: str                          # 上游脚本相对路径
    command: PipCommand
    walpurgis_skip: bool = True               # Walpurgis 是否跳过此阶段
    skip_reason: str = "Walpurgis 无 RAPIDS wheel 构建体系"

    def validate(self) -> List[str]:
        """BP-5: 校验 spec 自洽性，返回问题列表。"""
        _bp("CIRetrySpec.validate")  # BP-5
        issues: List[str] = []
        if self.command.mode is PipInvokeMode.DIRECT and not self.walpurgis_skip:
            issues.append(
                f"{self.stage_name}: mode=DIRECT 但 walpurgis_skip=False，"
                "未启用重试包装的 CI 阶段可能因网络抖动失败。"
            )
        return issues


# ---------------------------------------------------------------------------
# 6. PipRetryOrchestrator — 汇总四个 CI 阶段的重试配置
# ---------------------------------------------------------------------------

# 上游 9661018 四个阶段的标准配置
_DEFAULT_STAGES: List[CIRetrySpec] = [
    CIRetrySpec(
        stage_name="build_wheel",
        script_path="ci/build_wheel.sh",
        command=PipCommand(
            subcommand="wheel",
            mode=PipInvokeMode.RETRY_WRAPPED,
            args=["-w", "dist", "--no-deps"],
        ),
        walpurgis_skip=True,
        skip_reason="Walpurgis 无 wheel 构建体系；上游对应 ci/build_wheel.sh",
    ),
    CIRetrySpec(
        stage_name="test_wheel_cugraph_dgl",
        script_path="ci/test_wheel_cugraph-dgl.sh",
        command=PipCommand(
            subcommand="install",
            mode=PipInvokeMode.RETRY_WRAPPED,
            args=["--extra-index-url", "${PYTORCH_URL}", "--find-links", "${DGL_URL}"],
        ),
        walpurgis_skip=True,
        skip_reason="Walpurgis 无 DGL wheel 测试；上游对应 ci/test_wheel_cugraph-dgl.sh",
    ),
    CIRetrySpec(
        stage_name="test_wheel_cugraph_pyg",
        script_path="ci/test_wheel_cugraph-pyg.sh",
        command=PipCommand(
            subcommand="install",
            mode=PipInvokeMode.RETRY_WRAPPED,
            args=["--extra-index-url", "${PYTORCH_URL}", "--find-links", "${PYG_URL}"],
        ),
        walpurgis_skip=True,
        skip_reason="Walpurgis 无 PyG wheel 测试；上游对应 ci/test_wheel_cugraph-pyg.sh",
    ),
    CIRetrySpec(
        stage_name="test_wheel_pylibwholegraph",
        script_path="ci/test_wheel_pylibwholegraph.sh",
        command=PipCommand(
            subcommand="install",
            mode=PipInvokeMode.RETRY_WRAPPED,
            args=["--extra-index-url", "${INDEX_URL}"],
            target_packages=[
                "$(echo ./dist/pylibwholegraph*.whl)[test]",
                "'torch>=2.3'",
            ],
        ),
        walpurgis_skip=True,
        skip_reason=(
            "Walpurgis 无 wholegraph wheel 测试；"
            "上游注意：test_wheel_pylibwholegraph.sh 原为 `rapids-retry python -m pip`，"
            "9661018 统一改为 `rapids-pip-retry install`（去掉 `python -m pip` 中间层）"
        ),
    ),
]


@dataclass
class PipRetryOrchestrator:
    """汇总所有 CI 阶段的重试配置，提供模拟执行与审计接口。

    鲁迅：真正的勇士，敢于直面 hash 不匹配，敢于正视网络抖动。
    """
    stages: List[CIRetrySpec] = field(default_factory=lambda: list(_DEFAULT_STAGES))
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def active_stages(self) -> List[CIRetrySpec]:
        """返回 Walpurgis 实际执行的阶段（walpurgis_skip=False）。"""
        return [s for s in self.stages if not s.walpurgis_skip]

    def skipped_stages(self) -> List[CIRetrySpec]:
        """返回 Walpurgis 跳过的阶段（walpurgis_skip=True）。"""
        return [s for s in self.stages if s.walpurgis_skip]

    def validate_all(self) -> dict:
        """BP-6: 校验所有阶段，返回 {stage_name: [issues]} 字典。"""
        _bp("PipRetryOrchestrator.validate_all")  # BP-6
        result = {}
        for stage in self.stages:
            issues = stage.validate()
            if issues:
                result[stage.stage_name] = issues
        return result

    def simulate(self, stage_name: Optional[str] = None) -> List[dict]:
        """BP-1: 模拟执行（仅生成 CLI token，不实际调用 subprocess）。

        若指定 stage_name 则只模拟该阶段，否则模拟全部 active 阶段。
        """
        _bp("PipRetryOrchestrator.simulate")  # BP-1
        targets = (
            [s for s in self.active_stages() if s.stage_name == stage_name]
            if stage_name
            else self.active_stages()
        )
        results = []
        for stage in targets:
            tokens = stage.command.to_cli_tokens()
            results.append(
                {
                    "stage": stage.stage_name,
                    "script": stage.script_path,
                    "cli": " ".join(tokens),
                    "max_attempts": self.retry_policy.max_attempts,
                    "walpurgis_skip": stage.walpurgis_skip,
                }
            )
        return results

    def audit_report(self) -> str:
        """BP-7: 生成审计报告字符串，供 MIGRATION_LOG / CI 日志消费。"""
        _bp("PipRetryOrchestrator.audit_report")  # BP-7
        lines = [
            "=== PipRetryOrchestrator 审计报告 (9661018) ===",
            f"总阶段数: {len(self.stages)}",
            f"Walpurgis 激活: {len(self.active_stages())}",
            f"Walpurgis SKIP: {len(self.skipped_stages())}",
            "",
            "--- SKIP 阶段 ---",
        ]
        for s in self.skipped_stages():
            lines.append(f"  [{s.stage_name}]  {s.skip_reason}")
            lines.append(f"    上游: {s.script_path}")
            lines.append(f"    CLI:  {s.command.describe()}")
        lines += ["", "--- 激活阶段 ---"]
        if not self.active_stages():
            lines.append("  （本次迁移 9661018 全部 SKIP，无激活阶段）")
        for s in self.active_stages():
            lines.append(f"  [{s.stage_name}]  {s.command.describe()}")
        lines += ["", f"RetryPolicy: max_attempts={self.retry_policy.max_attempts}, "
                  f"backoff={self.retry_policy.backoff_seconds}s"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 模块级默认实例（方便直接 import 使用）
# ---------------------------------------------------------------------------
DEFAULT_ORCHESTRATOR = PipRetryOrchestrator()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    orch = PipRetryOrchestrator()
    print(orch.audit_report())
    issues = orch.validate_all()
    if issues:
        print("\n[WARN] 校验问题:", issues, file=sys.stderr)
    else:
        print("\n[OK] 所有阶段校验通过。")
