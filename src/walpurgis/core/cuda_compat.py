"""
cuda_compat.py — d491fae 迁移: 移除 CUDA 11 依赖，统一 CUDA 12+ 兼容策略

上游来源: cugraph-gnn / dependencies.yaml + conda/environments/
commit: d491fae479fdfd811c0cd251e8732e491057cb84
author: Kyle Edwards <kyedwards@nvidia.com>
date: 2025-06-04

上游变更摘要（5 files changed, 3 insertions, 229 deletions）:
  - conda/environments/all_cuda-118_arch-aarch64.yaml  ← 删除（48行）
  - conda/environments/all_cuda-118_arch-x86_64.yaml   ← 删除（48行）
  - dependencies.yaml                                   ← cuda: ["11.8","12.8"] → ["12.8"]
  - python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-aarch64.yaml ← 删除（21行）
  - python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-x86_64.yaml  ← 删除（21行）

迁移原则（参见 MIGRATION_LOG.md CI/merge→SKIP 规定）:
  - conda 环境文件 / CI 脚本 → SKIP（Walpurgis 无 conda 体系）
  - dependencies.yaml cuda 矩阵变更 → 迁移为 Python 层兼容性守卫

鲁迅拿法改写（≥20%）:
1. CudaVersionSpec dataclass: 替代上游散落在 dependencies.yaml 里的版本字符串，
   字段有类型注解、比较方法，可单独单元测试。
   上游做法：`cuda: ["12.8"]` 裸字符串列表。
2. CudaCompatPolicy: 封装"哪些 CUDA 版本受支持"的决策，
   validate_runtime_cuda() 在运行时抛 RuntimeError 而非静默错误。
   上游做法：构建时通过 conda recipe 隐式过滤，无运行时防御。
3. Cuda11RemovalAudit: 文档化 d491fae 删除的所有 CUDA 11 artifact，
   可被 CI 或 pytest 调用来验证项目内无残留 CUDA 11 引用。
   上游做法：无，删除即删除，无可审计记录。
4. WalpurgisCudaEnv: 汇总当前运行环境的 CUDA 信息，
   dump() 方法供 WALPURGIS_DEBUG=1 时一次性打印所有 CUDA 相关状态。
   上游做法：无，各调用方各自读取环境变量。
5. 全链路 WALPURGIS_DEBUG=1 断点 print：
   - 版本解析、策略决策、运行时检测、audit 扫描各阶段均有断点

参考: rapidsai/build-planning#184
      https://github.com/rapidsai/cugraph-gnn/pull/224
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 调试开关（与整个 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print(f"[WPG d491fae {tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# CudaVersionSpec — CUDA 版本的结构化表示
#
# 上游 dependencies.yaml 只有裸字符串 "12.8"，无法编程比较。
# 改写：冻结 dataclass，支持 __lt__/__eq__，可做 sorted() / min() 等操作。
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class CudaVersionSpec:
    """结构化 CUDA 版本，便于比较和断言。

    Examples::

        assert CudaVersionSpec(12, 8) > CudaVersionSpec(11, 8)
        assert CudaVersionSpec.from_str("12.8") == CudaVersionSpec(12, 8)
    """
    major: int
    minor: int
    patch: int = 0

    @classmethod
    def from_str(cls, version_str: str) -> "CudaVersionSpec":
        """从 '12.8' 或 '12.8.0' 格式字符串解析。"""
        _dbg("CudaVersionSpec.from_str", f"parsing '{version_str}'")
        parts = [int(x) for x in version_str.strip().split(".")]
        if len(parts) == 2:
            spec = cls(parts[0], parts[1], 0)
        elif len(parts) == 3:
            spec = cls(parts[0], parts[1], parts[2])
        else:
            raise ValueError(
                f"[Walpurgis:CudaVersionSpec] 无法解析版本字符串 '{version_str}'，"
                f"预期格式: 'X.Y' 或 'X.Y.Z'"
            )
        _dbg("CudaVersionSpec.from_str", f"结果: {spec}")
        return spec

    def to_str(self, include_patch: bool = False) -> str:
        """转为字符串，默认省略 patch。"""
        if include_patch:
            return f"{self.major}.{self.minor}.{self.patch}"
        return f"{self.major}.{self.minor}"

    @property
    def is_cuda11(self) -> bool:
        """判断是否为 CUDA 11.x（已被 d491fae 移除支持）。"""
        return self.major == 11

    @property
    def is_cuda12_plus(self) -> bool:
        """判断是否为 CUDA 12+（d491fae 后的支持范围）。"""
        return self.major >= 12

    def __repr__(self) -> str:
        return f"CudaVersionSpec({self.major}.{self.minor}.{self.patch})"


# ─────────────────────────────────────────────────────────────
# d491fae 前后的 CUDA 版本矩阵变更（可审计的常量）
#
# 上游 dependencies.yaml 迁移:
#   旧: cuda: ["11.8", "12.8"]
#   新: cuda: ["12.8"]
# ─────────────────────────────────────────────────────────────

#: d491fae 之前 cugraph-gnn 支持的 CUDA 版本集合（历史参考）
_CUDA_VERSIONS_BEFORE_D491FAE: FrozenSet[CudaVersionSpec] = frozenset({
    CudaVersionSpec(11, 8),
    CudaVersionSpec(12, 8),
})

#: d491fae 之后 cugraph-gnn 支持的 CUDA 版本集合（当前标准）
_CUDA_VERSIONS_AFTER_D491FAE: FrozenSet[CudaVersionSpec] = frozenset({
    CudaVersionSpec(12, 8),
})

#: Walpurgis 采用的最低 CUDA 版本（与上游 d491fae 后对齐）
WALPURGIS_MIN_CUDA_VERSION = CudaVersionSpec(12, 0)

#: Walpurgis 采用的推荐 CUDA 版本
WALPURGIS_RECOMMENDED_CUDA_VERSION = CudaVersionSpec(12, 8)

# ─────────────────────────────────────────────────────────────
# e16ddf5 迁移: Build and test with CUDA 13.2.0
#
# 上游 commit: e16ddf5a3137024434cfa545eaea6142354ac175
# Author: Bradley Dice <bdice@bradleydice.com>
# Date:   2026-05-12
# PR:     cugraph-gnn#456
#
# 上游变更摘要（11 files changed, 59 insertions, 50 deletions）：
#   .devcontainer/cuda13.1-conda/  → cuda13.2-conda/  (rename+patch)
#   .devcontainer/cuda13.1-pip/    → cuda13.2-pip/    (rename+patch)
#   .github/workflows/build.yaml   → @main → @cuda-13.2.0（8处）
#   .github/workflows/pr.yaml      → @main → @cuda-13.2.0（15处）
#   .github/workflows/test.yaml    → @main → @cuda-13.2.0（4处）
#   .github/workflows/trigger-breaking-change-alert.yaml（1处）
#   conda/environments/all_cuda-131_arch-*.yaml → cuda-132（rename+patch）
#   dependencies.yaml              → cuda: ["12.9","13.1"] → ["12.9","13.2"]
#   python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-13{1,2}*.yaml（rename）
#
# CI/merge/devcontainer 文件 → SKIP（Walpurgis 无 GH Actions / conda 体系）：
#   .devcontainer/**                SKIP: devcontainer 配置，Walpurgis 不使用
#   .github/workflows/**            SKIP: 所有 GH Actions workflow 文件
#   conda/environments/**           SKIP: conda 环境矩阵，Walpurgis 无 conda 体系
#   dependencies.yaml               SKIP: RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
#   python/cugraph-pyg/conda/**     SKIP: cugraph-pyg conda 开发环境
#
# 鲁迅拿法改写（≥20%）：
#   上游仅做文件改名+字符串替换（13.1→13.2），无任何 Python 层变更。
#   Walpurgis 迁移：引入 CudaMinorUpgradeAudit 数据类，将\"一次小版本升级\"
#   结构化为可审计、可回溯、可程序化查询的版本迁移事件，而非散落的
#   文件改名历史。结合 CudaVersionBump dataclass 精确表达升级语义，
#   并通过全链路断点（WALPURGIS_DEBUG=1）暴露判决路径。
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CudaVersionBump:
    """
    封装一次 CUDA 小版本升级事件。

    上游做法：文件改名（cuda13.1-* → cuda13.2-*）+ 字符串全局替换，
    无结构化记录，无 Python 层查询接口。

    改写：不可变值对象，携带完整元数据；
    ``is_minor_bump`` / ``delta_minor`` 属性支持程序化断言；
    ``describe()`` 生成人类可读摘要，与 MIGRATION_LOG.md 保持一致。
    """

    commit: str            # 上游 git commit hash（7位）
    pr_number: int         # 上游 PR 编号
    author: str            # 提交作者
    from_version: CudaVersionSpec   # 升级前版本
    to_version: CudaVersionSpec     # 升级后版本
    files_changed: int     # 上游改动文件数
    insertions: int        # 上游新增行数
    deletions: int         # 上游删除行数

    def __post_init__(self) -> None:
        _dbg(
            "CudaVersionBump.__init__",
            f"commit={self.commit!r}  {self.from_version} → {self.to_version}",
        )
        if self.to_version <= self.from_version:
            raise ValueError(
                f"[CudaVersionBump] to_version 必须大于 from_version，"
                f"收到: {self.from_version} → {self.to_version}"
            )

    @property
    def is_minor_bump(self) -> bool:
        """e16ddf5 是 minor 升级（major 不变，minor +1）。"""
        result = (
            self.from_version.major == self.to_version.major
            and self.to_version.minor == self.from_version.minor + 1
        )
        _dbg("CudaVersionBump.is_minor_bump", str(result))
        return result

    @property
    def delta_minor(self) -> int:
        """minor 版本差值（e16ddf5: 13.1 → 13.2，delta=1）。"""
        return self.to_version.minor - self.from_version.minor

    def describe(self) -> str:
        """人类可读摘要，与 MIGRATION_LOG.md 条目格式对齐。"""
        kind = "minor" if self.is_minor_bump else "major"
        return (
            f"[CudaVersionBump:{self.commit}] "
            f"{kind} upgrade {self.from_version} → {self.to_version} "
            f"(PR#{self.pr_number}, {self.author}; "
            f"{self.files_changed}F +{self.insertions}/-{self.deletions})"
        )


@dataclass(frozen=True)
class CudaMinorUpgradeAudit:
    """
    审计 e16ddf5 引入的 CUDA 13.1 → 13.2 升级的全部上游变更。

    上游做法：git diff 中文件改名 + 字符串替换，改动散落在 11 个文件中；
    无 Python 层记录，无可查询接口。

    改写：结构化枚举所有被升级的制品类型（devcontainer / workflow / conda env
    / dep matrix / pyg conda），提供 ``skipped_artifacts`` / ``affected_types``
    两级查询，以及 ``assert_no_old_version_refs(path)`` 扫描残留。
    """

    #: e16ddf5 升级事件描述对象
    BUMP: CudaVersionBump = field(
        default_factory=lambda: CudaVersionBump(
            commit="e16ddf5",
            pr_number=456,
            author="Bradley Dice",
            from_version=CudaVersionSpec(13, 1),
            to_version=CudaVersionSpec(13, 2),
            files_changed=11,
            insertions=59,
            deletions=50,
        )
    )

    # ── 被跳过的上游制品（全部 SKIP，Walpurgis 无对应体系） ──────────────────

    #: devcontainer 配置（rename: cuda13.1-* → cuda13.2-*）
    SKIPPED_DEVCONTAINERS: Tuple[str, ...] = field(
        default_factory=lambda: (
            ".devcontainer/cuda13.1-conda/devcontainer.json",
            ".devcontainer/cuda13.1-pip/devcontainer.json",
        )
    )

    #: GH Actions workflow 文件（@main → @cuda-13.2.0 替换，共28处）
    SKIPPED_WORKFLOWS: Tuple[str, ...] = field(
        default_factory=lambda: (
            ".github/workflows/build.yaml",
            ".github/workflows/pr.yaml",
            ".github/workflows/test.yaml",
            ".github/workflows/trigger-breaking-change-alert.yaml",
        )
    )

    #: conda 环境矩阵文件（rename: cuda-131 → cuda-132）
    SKIPPED_CONDA_ENVS: Tuple[str, ...] = field(
        default_factory=lambda: (
            "conda/environments/all_cuda-131_arch-aarch64.yaml",
            "conda/environments/all_cuda-131_arch-x86_64.yaml",
        )
    )

    #: RAPIDS 构建依赖文件（cuda: ["12.9","13.1"] → ["12.9","13.2"]）
    SKIPPED_DEPS: Tuple[str, ...] = field(
        default_factory=lambda: ("dependencies.yaml",)
    )

    #: cugraph-pyg conda 开发环境（rename: cuda-131 → cuda-132）
    SKIPPED_PYG_CONDA: Tuple[str, ...] = field(
        default_factory=lambda: (
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-aarch64.yaml",
            "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-x86_64.yaml",
        )
    )

    @property
    def skipped_artifacts(self) -> Tuple[str, ...]:
        """所有被跳过的上游制品路径（11个）。"""
        result = (
            self.SKIPPED_DEVCONTAINERS
            + self.SKIPPED_WORKFLOWS
            + self.SKIPPED_CONDA_ENVS
            + self.SKIPPED_DEPS
            + self.SKIPPED_PYG_CONDA
        )
        _dbg("CudaMinorUpgradeAudit.skipped_artifacts", f"count={len(result)}")
        return result

    @property
    def affected_types(self) -> FrozenSet[str]:
        """e16ddf5 涉及的制品类型集合。"""
        return frozenset({"devcontainer", "workflow", "conda_env", "dep_matrix", "pyg_conda"})

    def dump(self) -> None:
        """打印审计摘要（WALPURGIS_DEBUG=1 或手动调用）。"""
        print(self.BUMP.describe())
        print(f"  SKIP总计: {len(self.skipped_artifacts)} 个制品")
        for art in self.skipped_artifacts:
            print(f"    SKIP: {art}")

    def assert_no_old_version_refs(self, search_path: str) -> list:
        """
        扫描 ``search_path`` 下是否残留 ``cuda13.1`` / ``cuda-131`` 旧版引用。

        断点7: 扫描路径 + 命中数。

        Returns:
            命中文件路径列表（空表示无残留，理想状态）。
        """
        import re as _re
        import pathlib as _pathlib

        patterns = [
            _re.compile(r"cuda[-_.]?13\.1", _re.IGNORECASE),
            _re.compile(r"cuda[-_]131", _re.IGNORECASE),
            _re.compile(r"shared-workflows[^\s\"']*@cuda-13\.1", _re.IGNORECASE),
        ]
        hits: list = []
        base = _pathlib.Path(search_path)
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = list(base.rglob("*"))
        else:
            return hits

        for fpath in files:
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pat in patterns:
                if pat.search(text):
                    hits.append(str(fpath))
                    break

        # 断点7
        _dbg(
            "CudaMinorUpgradeAudit.assert_no_old_version_refs",
            f"search_path={search_path!r}  hits={len(hits)}",
        )
        return hits


#: e16ddf5 升级审计记录（模块级单例）
E16DDF5_CUDA_UPGRADE_AUDIT: CudaMinorUpgradeAudit = CudaMinorUpgradeAudit()

#: e16ddf5 后 cugraph-gnn 支持的 CUDA 版本集合（含 13.2）
_CUDA_VERSIONS_AFTER_E16DDF5: FrozenSet[CudaVersionSpec] = frozenset({
    CudaVersionSpec(12, 9),
    CudaVersionSpec(13, 2),
})


# ─────────────────────────────────────────────────────────────
# CudaCompatPolicy — 运行时 CUDA 兼容性策略
#
# 上游无 Python 层 CUDA 版本校验，由 conda 环境构建期隐式过滤。
# Walpurgis 改写：加 validate_runtime_cuda() 在 import 期或调用期主动检测。
# ─────────────────────────────────────────────────────────────

@dataclass
class CudaCompatPolicy:
    """CUDA 版本兼容性策略，封装 d491fae 的版本矩阵决策。

    Attributes:
        min_version: 最低支持版本（inclusive），默认 12.0。
        removed_versions: 明确不支持的版本集合（含 CUDA 11.x）。
        strict: True 时版本不满足即 raise，False 时仅 warn。
    """
    min_version: CudaVersionSpec = field(
        default_factory=lambda: CudaVersionSpec(12, 0)
    )
    removed_versions: FrozenSet[CudaVersionSpec] = field(
        default_factory=lambda: frozenset({
            CudaVersionSpec(11, 2),
            CudaVersionSpec(11, 4),
            CudaVersionSpec(11, 5),
            CudaVersionSpec(11, 8),
        })
    )
    strict: bool = True

    def is_supported(self, version: CudaVersionSpec) -> bool:
        """检查 version 是否在支持范围内。"""
        _dbg("CudaCompatPolicy.is_supported", f"检查 {version}")
        if version in self.removed_versions:
            _dbg("CudaCompatPolicy.is_supported", f"→ False（在 removed_versions 中）")
            return False
        if version < self.min_version:
            _dbg("CudaCompatPolicy.is_supported",
                 f"→ False（{version} < min {self.min_version}）")
            return False
        _dbg("CudaCompatPolicy.is_supported", f"→ True")
        return True

    def validate_runtime_cuda(self) -> Optional[CudaVersionSpec]:
        """检测当前环境 CUDA 版本，不满足策略时按 strict 模式处理。

        Returns:
            检测到的版本，或 None（CUDA 未安装）。

        Raises:
            RuntimeError: strict=True 且版本不符合策略时。
        """
        _dbg("CudaCompatPolicy.validate_runtime_cuda", "开始运行时 CUDA 检测")
        detected = _detect_runtime_cuda_version()

        if detected is None:
            _dbg("CudaCompatPolicy.validate_runtime_cuda",
                 "未检测到 CUDA（CPU-only 环境）")
            return None

        _dbg("CudaCompatPolicy.validate_runtime_cuda",
             f"检测到 CUDA {detected}")

        if not self.is_supported(detected):
            msg = (
                f"[Walpurgis d491fae] 检测到不受支持的 CUDA 版本 {detected}。\n"
                f"  d491fae（Remove CUDA 11 from dependencies）之后，\n"
                f"  Walpurgis 要求 CUDA >= {self.min_version}。\n"
                f"  已移除的版本: {sorted(self.removed_versions)}\n"
                f"  推荐版本: {WALPURGIS_RECOMMENDED_CUDA_VERSION}\n"
                f"  参考: https://github.com/rapidsai/cugraph-gnn/pull/224"
            )
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return detected


# ─────────────────────────────────────────────────────────────
# Cuda11RemovalAudit — d491fae 删除条目的可审计记录
#
# 上游直接删 5 个文件，无 Python 层记录。
# 改写：枚举所有被删除的 conda artifact，供 CI 验证"项目内无残留 CUDA 11 引用"。
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _RemovedArtifact:
    """d491fae 删除的单个 conda 环境/配置文件记录。"""
    path: str        # 在上游仓库中的路径
    purpose: str     # 原来的用途描述
    arch: str        # 目标架构，如 "x86_64" / "aarch64"
    cuda_ver: str    # 对应的 CUDA 版本（如 "11.8"）


class Cuda11RemovalAudit:
    """d491fae 移除 CUDA 11 的完整审计记录。

    提供 assert_no_cuda11_refs() 方法供测试调用，确认项目内无残留的
    CUDA 11 引用（如 "cu118"、"cuda-11"、"cudatoolkit"）。
    """

    #: d491fae 删除的所有 conda artifact（可机器验证）
    REMOVED_ARTIFACTS: Tuple[_RemovedArtifact, ...] = (
        _RemovedArtifact(
            path="conda/environments/all_cuda-118_arch-aarch64.yaml",
            purpose="CUDA 11.8 aarch64 全量 conda 环境（含 breathe/cmake/nvcc_linux-aarch64=11.8 等 36 项）",
            arch="aarch64",
            cuda_ver="11.8",
        ),
        _RemovedArtifact(
            path="conda/environments/all_cuda-118_arch-x86_64.yaml",
            purpose="CUDA 11.8 x86_64 全量 conda 环境（含 breathe/cmake/nvcc_linux-64=11.8 等 36 项）",
            arch="x86_64",
            cuda_ver="11.8",
        ),
        _RemovedArtifact(
            path="python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-aarch64.yaml",
            purpose="cugraph-pyg 开发环境 CUDA 11.8 aarch64（含 pytorch>=2.3,<=2.5.1）",
            arch="aarch64",
            cuda_ver="11.8",
        ),
        _RemovedArtifact(
            path="python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-x86_64.yaml",
            purpose="cugraph-pyg 开发环境 CUDA 11.8 x86_64（含 pytorch>=2.3,<=2.5.1）",
            arch="x86_64",
            cuda_ver="11.8",
        ),
    )

    #: d491fae 在 dependencies.yaml 中删除的版本矩阵条目
    REMOVED_MATRIX_ENTRIES: Tuple[str, ...] = (
        "cuda-version=11.2",
        "cuda-version=11.4",
        "cuda-version=11.5",
        "cuda-version=11.8",
        "cudatoolkit",          # CUDA 11 conda runtime（CUDA 12 改用 cuda-cudart）
        "cuda-nvtx",            # CUDA 11 专属（CUDA 12 已内置）
        "gcc_linux-64=11.*",    # CUDA 11.8 x86_64 编译器
        "gcc_linux-aarch64=11.*",  # CUDA 11.8 aarch64 编译器
        "nvcc_linux-64=11.8",
        "nvcc_linux-aarch64=11.8",
        "--extra-index-url=https://download.pytorch.org/whl/cu118",
        "pytorch>=2.3,<=2.5.1",  # CUDA 11 的 final PyTorch build（conda-forge）
        "pylibwholegraph-cu11",
        "libraft-cu11",
        "librmm-cu11",
        "libwholegraph-cu11",
        "rmm-cu11",
        "cugraph-cu11",
        "cudf-cu11",
        "dask-cudf-cu11",
        "pylibcugraph-cu11",
        "cupy-cuda11x>=13.2.0",
    )

    def dump(self) -> None:
        """打印所有 removed artifacts 和 matrix entries。"""
        _dbg("Cuda11RemovalAudit.dump",
             f"{len(self.REMOVED_ARTIFACTS)} 个删除的 conda artifact:")
        for a in self.REMOVED_ARTIFACTS:
            _dbg("  artifact",
                 f"[{a.arch}] CUDA {a.cuda_ver}: {a.path}")
        _dbg("Cuda11RemovalAudit.dump",
             f"{len(self.REMOVED_MATRIX_ENTRIES)} 个删除的 matrix entry:")
        for e in self.REMOVED_MATRIX_ENTRIES:
            _dbg("  matrix_entry", e)

    def assert_no_cuda11_refs(self, search_root: str) -> List[str]:
        """扫描 search_root 目录，返回所有疑似残留 CUDA 11 引用的文件路径列表。

        常见 CUDA 11 标记: "cu118", "cu11", "cuda-11", "11.8", "cudatoolkit"。
        空列表表示无残留，非空则需人工确认。

        Args:
            search_root: 待扫描的根目录路径（通常是 walpurgis-WTFGG/src）。

        Returns:
            含有疑似 CUDA 11 引用的文件路径列表。
        """
        _dbg("Cuda11RemovalAudit.assert_no_cuda11_refs",
             f"扫描 '{search_root}'")

        # 这些模式出现在 Python/YAML 源文件中才需要关注
        _CUDA11_PATTERNS = [
            r"\bcu118\b", r"\bcu11\b", r"\bcuda-11\b",
            r"cuda.version.*=.*11\.", r"cudatoolkit\b",
            r"cuda-nvtx\b", r"cupy-cuda11x",
            r"pylibwholegraph-cu11", r"libraft-cu11",
            r"rmm-cu11\b", r"cugraph-cu11\b",
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in _CUDA11_PATTERNS]

        hits: List[str] = []
        _EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".sh"}

        for dirpath, dirnames, filenames in os.walk(search_root):
            # 跳过 __pycache__ 等目录
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "node_modules")]
            for fname in filenames:
                if not any(fname.endswith(ext) for ext in _EXTENSIONS):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    text = open(fpath, encoding="utf-8", errors="ignore").read()
                    for pat in compiled:
                        if pat.search(text):
                            hits.append(fpath)
                            _dbg("  HIT", fpath)
                            break
                except OSError:
                    pass

        _dbg("Cuda11RemovalAudit.assert_no_cuda11_refs",
             f"扫描完成: {len(hits)} 个疑似残留")
        return hits


# ─────────────────────────────────────────────────────────────
# WalpurgisCudaEnv — 运行环境 CUDA 信息汇总
#
# 上游：各调用方零散读取 CUDA_VERSION / CUDA_VISIBLE_DEVICES 等环境变量。
# 改写：统一汇总，dump() 供调试，validate() 供运行时守卫。
# ─────────────────────────────────────────────────────────────

@dataclass
class WalpurgisCudaEnv:
    """当前 Python 进程的 CUDA 运行环境快照。

    可在 train_walpurgis.py / __init__.py 中早期调用，统一输出 CUDA 状态。
    """
    runtime_version: Optional[CudaVersionSpec] = field(init=False)
    visible_devices: str = field(init=False)
    torch_cuda_available: Optional[bool] = field(init=False)
    torch_cuda_version: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        _dbg("WalpurgisCudaEnv.__post_init__", "初始化 CUDA 环境快照")
        self.runtime_version = _detect_runtime_cuda_version()
        self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)")
        # 懒检测 torch（避免不必要 import 延迟）
        self.torch_cuda_available = None
        self.torch_cuda_version = None
        try:
            import torch
            self.torch_cuda_available = torch.cuda.is_available()
            self.torch_cuda_version = torch.version.cuda
            _dbg("WalpurgisCudaEnv.__post_init__",
                 f"torch.cuda.is_available()={self.torch_cuda_available}, "
                 f"torch.version.cuda={self.torch_cuda_version}")
        except ImportError:
            _dbg("WalpurgisCudaEnv.__post_init__", "torch 未安装，跳过")

        _dbg("WalpurgisCudaEnv.__post_init__", repr(self))

    def dump(self) -> None:
        """打印完整 CUDA 环境信息（不受 WALPURGIS_DEBUG 控制，始终输出）。"""
        lines = [
            "=== Walpurgis CUDA 环境 (d491fae: CUDA 11 已移除) ===",
            f"  runtime_version      : {self.runtime_version}",
            f"  visible_devices      : {self.visible_devices}",
            f"  torch_cuda_available : {self.torch_cuda_available}",
            f"  torch_cuda_version   : {self.torch_cuda_version}",
            f"  min_required         : {WALPURGIS_MIN_CUDA_VERSION}",
            f"  recommended          : {WALPURGIS_RECOMMENDED_CUDA_VERSION}",
            "=================================================",
        ]
        for line in lines:
            print(line, flush=True)

    def validate(self, policy: Optional[CudaCompatPolicy] = None) -> None:
        """使用给定策略验证运行时 CUDA 版本。"""
        _policy = policy or CudaCompatPolicy()
        _policy.validate_runtime_cuda()

    def __repr__(self) -> str:
        return (
            f"WalpurgisCudaEnv("
            f"runtime={self.runtime_version}, "
            f"devices={self.visible_devices!r}, "
            f"torch_cuda={self.torch_cuda_available})"
        )


# ─────────────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────────────

def _detect_runtime_cuda_version() -> Optional[CudaVersionSpec]:
    """检测当前运行环境的 CUDA 版本。

    检测优先级:
    1. CUDA_VERSION 环境变量（最可靠，torchrun/conda 会设置）
    2. nvidia-smi 命令输出
    3. nvcc --version（若 PATH 中有）

    Returns:
        CudaVersionSpec，或 None（CPU-only 环境）。
    """
    _dbg("_detect_runtime_cuda_version", "开始检测 CUDA 版本")

    # --- 方法 1: 环境变量 ---
    cuda_env = os.environ.get("CUDA_VERSION", "")
    if cuda_env:
        try:
            spec = CudaVersionSpec.from_str(cuda_env)
            _dbg("_detect_runtime_cuda_version",
                 f"来自 CUDA_VERSION env: {spec}")
            return spec
        except ValueError:
            _dbg("_detect_runtime_cuda_version",
                 f"CUDA_VERSION='{cuda_env}' 无法解析，继续尝试其他方法")

    # --- 方法 2: nvidia-smi ---
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        # driver_version 不是 CUDA 版本，但可确认 GPU 存在；
        # 真正的 CUDA version 在 nvidia-smi 首行
        smi_out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_out)
        if m:
            spec = CudaVersionSpec.from_str(m.group(1))
            _dbg("_detect_runtime_cuda_version",
                 f"来自 nvidia-smi: {spec}")
            return spec
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        _dbg("_detect_runtime_cuda_version", "nvidia-smi 不可用，跳过")

    # --- 方法 3: nvcc ---
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        m = re.search(r"release\s+(\d+\.\d+)", out)
        if m:
            spec = CudaVersionSpec.from_str(m.group(1))
            _dbg("_detect_runtime_cuda_version",
                 f"来自 nvcc: {spec}")
            return spec
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        _dbg("_detect_runtime_cuda_version", "nvcc 不可用，跳过")

    _dbg("_detect_runtime_cuda_version", "未检测到 CUDA（CPU-only 或无工具链）")
    return None


# ─────────────────────────────────────────────────────────────
# 模块级公开接口
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# f83f6ae 迁移: Switch to release channel for PyTorch + CUDA 13
# 上游 commit: f83f6aea7ecc1f2d99fd66969cc07693f087e797
# author: Alex Barghi <alexbarghi-nv>, 2025-11-25, PR #355
# co-author: Gil Forsyth <gforsyth@nvidia.com>
#
# 上游变更摘要（3 files changed, 5 insertions, 9 deletions）:
#   ci/test_wheel_cugraph-pyg.sh  → SKIP（Walpurgis 无 CI wheel 体系）
#   dependencies.yaml             → SKIP（RAPIDS conda 依赖矩阵，Walpurgis 无 conda 体系）
#   python/cugraph-pyg/pyproject.toml → SKIP（包元数据，Walpurgis 用自有 pyproject.toml）
#
# 核心语义（鲁迅拿法改写，≥20%）:
#   上游三处均只改了字符串字面量：
#     "nightly/cu130" → "cu130"（两处）
#     "torch>=2.9.0.dev0" → "torch>=2.9.0"（两处，外加 dev0 允许 nightly 的注释删除）
#   Walpurgis 将此决策结构化为 PyTorchCuda13ChannelPolicy 数据类：
#   1. 显式区分 NIGHTLY / RELEASE 两个渠道枚举（上游靠 URL 字符串区分，无结构）
#   2. min_torch_version 字段类型化为 tuple 而非裸字符串，可做版本比较和断言
#   3. resolve_index_url() 方法统一渠道→URL 转换，替代上游 if/else 硬编码
#   4. validate_torch_version() 做运行时 torch 版本检查（上游无此防御）
#   5. 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖渠道决策、URL 解析、版本检查
#
# 鲁迅语：「时间就是性命，无端的空耗别人的时间，其实是无异于谋财害命的。」
# 上游 TODO 注释的删除，正是如此——nightly 依赖是对下游用户时间的无端空耗，
# release channel 归正了这一浪费，Walpurgis 将其铭刻为可审计的策略对象。
# ─────────────────────────────────────────────────────────────

from enum import Enum as _Enum


class _PyTorchChannel(_Enum):
    """PyTorch 下载渠道枚举。

    上游 f83f6ae 的语义：CUDA 13（cu130）从 NIGHTLY 切换到 RELEASE。
    """
    NIGHTLY = "nightly"
    RELEASE = "release"


@dataclass(frozen=True)
class PyTorchCuda13ChannelPolicy:
    """CUDA 13 PyTorch 渠道策略。

    封装 f83f6ae 将 CUDA 13 PyTorch 渠道从 nightly 切换到 release 的核心决策。

    上游做法（散落三处字符串字面量）::

        # ci/test_wheel_cugraph-pyg.sh（旧）:
        PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"
        # dependencies.yaml（旧）:
        - --extra-index-url=https://download.pytorch.org/whl/nightly/cu130
        - torch>=2.9.0.dev0

        # ci/test_wheel_cugraph-pyg.sh（新，f83f6ae）:
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
        # dependencies.yaml（新，f83f6ae）:
        - --extra-index-url=https://download.pytorch.org/whl/cu130
        - torch>=2.9.0

    Walpurgis 改写：以数据类封装上述决策，渠道选择和版本约束各有字段，
    可单元测试，可在运行时输出调试信息。

    Attributes:
        channel: 当前渠道（f83f6ae 后固定为 RELEASE）。
        cuda_tag: CUDA 轮子标签，如 ``"cu130"``、``"cu126"``。
        min_torch_version: torch 最低版本，三元 int tuple。
          f83f6ae 前：(2, 9, 0) + dev0 flag；f83f6ae 后：(2, 9, 0)。
        allow_dev: 是否接受 dev/nightly 构建（f83f6ae 前为 True，后为 False）。
    """

    channel: _PyTorchChannel = _PyTorchChannel.RELEASE
    cuda_tag: str = "cu130"
    min_torch_version: Tuple[int, ...] = (2, 9, 0)
    allow_dev: bool = False  # f83f6ae 删除了 dev0 允许

    # 断点1: 构造时输出策略摘要
    def __post_init__(self) -> None:
        _dbg(
            "PyTorchCuda13ChannelPolicy.__init__",
            f"channel={self.channel.value!r}  cuda_tag={self.cuda_tag!r}  "
            f"min_torch={self.min_torch_version}  allow_dev={self.allow_dev}",
        )

    @property
    def base_whl_url(self) -> str:
        """基础 PyTorch wheel 根 URL（不含 index-url 前缀）。

        Examples::

            >>> p = PyTorchCuda13ChannelPolicy()
            >>> p.base_whl_url
            'https://download.pytorch.org/whl/cu130'
        """
        if self.channel is _PyTorchChannel.NIGHTLY:
            url = f"https://download.pytorch.org/whl/nightly/{self.cuda_tag}"
        else:
            url = f"https://download.pytorch.org/whl/{self.cuda_tag}"
        # 断点2: 渠道→URL 解析结果
        _dbg(
            "PyTorchCuda13ChannelPolicy.base_whl_url",
            f"channel={self.channel.value!r} → url={url!r}",
        )
        return url

    @property
    def extra_index_url_flag(self) -> str:
        """生成 pip ``--extra-index-url=<url>`` 字符串。

        对应 dependencies.yaml 中的::

            - --extra-index-url=https://download.pytorch.org/whl/cu130
        """
        flag = f"--extra-index-url={self.base_whl_url}"
        _dbg("PyTorchCuda13ChannelPolicy.extra_index_url_flag", flag)
        return flag

    @property
    def torch_requirement(self) -> str:
        """生成 pip/conda torch 版本要求字符串。

        f83f6ae 前：``torch>=2.9.0.dev0``
        f83f6ae 后：``torch>=2.9.0``

        Examples::

            >>> p = PyTorchCuda13ChannelPolicy()
            >>> p.torch_requirement
            'torch>=2.9.0'
        """
        ver_str = ".".join(str(v) for v in self.min_torch_version)
        if self.allow_dev:
            req = f"torch>={ver_str}.dev0"
        else:
            req = f"torch>={ver_str}"
        _dbg("PyTorchCuda13ChannelPolicy.torch_requirement", req)
        return req

    def validate_torch_version(self) -> Optional[Tuple[int, ...]]:
        """运行时检查已安装 torch 是否满足最低版本要求。

        上游无此防御——切换到 release channel 后若用户环境仍装的是 dev0 版本
        会产生细微的行为差异（dev0 中含未稳定 API）。此方法提前暴露此问题。

        Returns:
            (major, minor, patch) tuple，或 None（torch 未安装）。

        断点3 / 断点4: 检测结果输出。
        """
        _dbg("PyTorchCuda13ChannelPolicy.validate_torch_version", "开始检测 torch 版本")
        try:
            import torch  # type: ignore[import-untyped]
            raw = torch.__version__  # 例如 "2.9.0" 或 "2.9.0.dev20250101"
            # 截取数字部分（忽略 .devXXX 后缀）
            numeric = raw.split(".dev")[0].split("+")[0]
            parts = tuple(int(x) for x in numeric.split(".")[:3])
            # 断点3: 已安装版本
            _dbg(
                "PyTorchCuda13ChannelPolicy.validate_torch_version",
                f"已安装 torch={raw!r}  解析为 {parts}",
            )
            is_ok = parts >= self.min_torch_version
            is_dev = "dev" in raw
            if is_dev and not self.allow_dev:
                import warnings as _warnings
                _warnings.warn(
                    f"[f83f6ae] torch={raw!r} 是 nightly/dev 构建，"
                    f"f83f6ae 后 CUDA 13 已切换到 release channel（{self.torch_requirement}）。"
                    "建议升级到 release 版本以获得稳定行为。",
                    FutureWarning,
                    stacklevel=2,
                )
            if not is_ok:
                msg = (
                    f"[f83f6ae] torch={raw!r} < 要求 {self.torch_requirement}。"
                    "请通过 release channel 升级：\n"
                    f"  pip install '{self.torch_requirement}' {self.extra_index_url_flag}"
                )
                # 断点4: 版本不满足
                _dbg("PyTorchCuda13ChannelPolicy.validate_torch_version", f"版本检查失败: {msg}")
                raise RuntimeError(msg)
            _dbg("PyTorchCuda13ChannelPolicy.validate_torch_version", f"版本检查通过: {parts}")
            return parts
        except ImportError:
            _dbg("PyTorchCuda13ChannelPolicy.validate_torch_version", "torch 未安装，跳过检查")
            return None


# ─── CUDA 版本 → PyTorch 渠道策略映射 ───────────────────────
# 对应 ci/test_wheel_cugraph-pyg.sh 的 if/else 分支逻辑：
#   CUDA 12 → cu126（始终使用 release）
#   CUDA 13 → cu130（f83f6ae 前 nightly，f83f6ae 后 release）

#: CUDA 12.x PyTorch 渠道策略（参考值，cu126 始终为 release）
PYTORCH_CUDA12_POLICY: PyTorchCuda13ChannelPolicy = PyTorchCuda13ChannelPolicy(
    channel=_PyTorchChannel.RELEASE,
    cuda_tag="cu126",
    min_torch_version=(2, 6, 0),
    allow_dev=False,
)

#: CUDA 13.x PyTorch 渠道策略（f83f6ae 后：release channel，torch>=2.9.0）
#: 与上游 pyproject.toml test dep 对齐：torch>=2.9.0
PYTORCH_CUDA13_POLICY: PyTorchCuda13ChannelPolicy = PyTorchCuda13ChannelPolicy(
    channel=_PyTorchChannel.RELEASE,   # f83f6ae: nightly → release
    cuda_tag="cu130",
    min_torch_version=(2, 9, 0),       # f83f6ae: >=2.9.0.dev0 → >=2.9.0
    allow_dev=False,                   # f83f6ae: 删除 dev0 注释，明确不允许 nightly
)


def get_pytorch_policy(cuda_version: CudaVersionSpec) -> PyTorchCuda13ChannelPolicy:
    """根据 CUDA 版本返回对应的 PyTorch 渠道策略。

    对应 ci/test_wheel_cugraph-pyg.sh 中的 if/else 分支逻辑（f83f6ae 后）::

        if [[ "${CUDA_MAJOR}" == "12" ]]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
        else
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
        fi

    断点5: 输出 CUDA 版本 → 策略选择路径。

    Args:
        cuda_version: 当前运行时 CUDA 版本。

    Returns:
        对应的 :class:`PyTorchCuda13ChannelPolicy` 单例。
    """
    if cuda_version.major == 12:
        policy = PYTORCH_CUDA12_POLICY
    else:
        # CUDA 13+（f83f6ae 切换为 release）
        policy = PYTORCH_CUDA13_POLICY
    # 断点5
    _dbg(
        "get_pytorch_policy",
        f"cuda={cuda_version} → channel={policy.channel.value!r}  "
        f"index_url={policy.base_whl_url!r}  req={policy.torch_requirement!r}",
    )
    return policy


#: 默认 CUDA 兼容性策略（与 d491fae 后的上游对齐）
DEFAULT_CUDA_POLICY: CudaCompatPolicy = CudaCompatPolicy(
    min_version=WALPURGIS_MIN_CUDA_VERSION,
    strict=False,  # warn 而非 raise，避免阻断 CPU-only 开发
)

#: d491fae 移除审计记录（模块级单例）
CUDA11_REMOVAL_AUDIT: Cuda11RemovalAudit = Cuda11RemovalAudit()


def check_cuda_compat(strict: bool = False) -> Optional[CudaVersionSpec]:
    """一行式 CUDA 兼容性检查，供 train_walpurgis.py 等入口调用。

    Args:
        strict: True 时版本不符合则 raise RuntimeError。

    Returns:
        检测到的 CUDA 版本，或 None。
    """
    _dbg("check_cuda_compat", f"strict={strict}")
    policy = CudaCompatPolicy(
        min_version=WALPURGIS_MIN_CUDA_VERSION,
        strict=strict,
    )
    return policy.validate_runtime_cuda()


# ─────────────────────────────────────────────────────────────
# 自测入口（python -m walpurgis.core.cuda_compat）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== cuda_compat.py 自测 (d491fae migrate) ===")

    # 1. 版本解析
    v = CudaVersionSpec.from_str("12.8")
    assert v.major == 12 and v.minor == 8, f"解析错误: {v}"
    assert v.is_cuda12_plus, "12.8 应为 CUDA12+"
    assert not v.is_cuda11, "12.8 不应为 CUDA11"
    print(f"[PASS] CudaVersionSpec.from_str('12.8') = {v}")

    v11 = CudaVersionSpec.from_str("11.8")
    assert v11.is_cuda11, "11.8 应为 CUDA11"
    assert not v11.is_cuda12_plus, "11.8 不应为 CUDA12+"
    print(f"[PASS] CudaVersionSpec CUDA11 检测: {v11.is_cuda11}")

    # 2. 版本比较（d491fae 前后矩阵对比）
    assert CudaVersionSpec(12, 8) > CudaVersionSpec(11, 8), "12.8 > 11.8"
    assert CudaVersionSpec(12, 0) <= CudaVersionSpec(12, 8), "12.0 <= 12.8"
    print("[PASS] CudaVersionSpec 大小比较正确")

    # 3. CudaCompatPolicy
    policy = CudaCompatPolicy(strict=False)
    assert policy.is_supported(CudaVersionSpec(12, 8)), "12.8 应被支持"
    assert not policy.is_supported(CudaVersionSpec(11, 8)), "11.8 不应被支持（d491fae移除）"
    assert not policy.is_supported(CudaVersionSpec(10, 2)), "10.2 < min 12.0，不支持"
    print("[PASS] CudaCompatPolicy.is_supported() 正确")

    # 4. Audit 完整性
    audit = Cuda11RemovalAudit()
    assert len(audit.REMOVED_ARTIFACTS) == 4, "应有 4 个被删除的 conda artifact"
    assert len(audit.REMOVED_MATRIX_ENTRIES) > 10, "应有多条 CUDA 11 matrix entry"
    audit.dump()
    print(f"[PASS] Cuda11RemovalAudit: {len(audit.REMOVED_ARTIFACTS)} artifacts, "
          f"{len(audit.REMOVED_MATRIX_ENTRIES)} matrix entries")

    # 5. assert_no_cuda11_refs（扫描 /tmp 无残留）
    hits = audit.assert_no_cuda11_refs("/tmp")
    print(f"[INFO] /tmp 中 CUDA 11 残留: {hits}")

    # 6. WalpurgisCudaEnv
    env = WalpurgisCudaEnv()
    env.dump()
    print(f"[PASS] WalpurgisCudaEnv 初始化: {env}")

    # 7. f83f6ae PyTorchCuda13ChannelPolicy 自测
    p13 = PyTorchCuda13ChannelPolicy()
    assert p13.channel is _PyTorchChannel.RELEASE, "f83f6ae 后应为 RELEASE"
    assert p13.cuda_tag == "cu130", "CUDA 13 tag 应为 cu130"
    assert p13.min_torch_version == (2, 9, 0), "f83f6ae 后 min torch = 2.9.0"
    assert not p13.allow_dev, "f83f6ae 后不允许 dev0"
    assert "nightly" not in p13.base_whl_url, f"release URL 不含 nightly: {p13.base_whl_url}"
    assert p13.base_whl_url == "https://download.pytorch.org/whl/cu130", p13.base_whl_url
    assert p13.torch_requirement == "torch>=2.9.0", p13.torch_requirement
    print(f"[PASS] PyTorchCuda13ChannelPolicy release: url={p13.base_whl_url!r}  req={p13.torch_requirement!r}")

    p13_nightly = PyTorchCuda13ChannelPolicy(channel=_PyTorchChannel.NIGHTLY, allow_dev=True)
    assert "nightly/cu130" in p13_nightly.base_whl_url, p13_nightly.base_whl_url
    assert "dev0" in p13_nightly.torch_requirement, p13_nightly.torch_requirement
    print(f"[PASS] PyTorchCuda13ChannelPolicy nightly (pre-f83f6ae): url={p13_nightly.base_whl_url!r}")

    assert get_pytorch_policy(CudaVersionSpec(12, 9)) is PYTORCH_CUDA12_POLICY
    assert get_pytorch_policy(CudaVersionSpec(13, 0)) is PYTORCH_CUDA13_POLICY
    print("[PASS] get_pytorch_policy 路由正确")

    print("=== 所有自测通过 ===")
    sys.exit(0)
