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
# 2d2bc51 迁移: Build and test with CUDA 13.3.0
#
# 上游 commit: 2d2bc51a0d1336ee16343994cd98606116b39c1f
# Author: Bradley Dice <bdice@bradleydice.com>
# Date:   2026-06-11
# PR:     cugraph#5553
#
# 上游变更摘要（9 files changed, 61 insertions, 52 deletions）：
#   .devcontainer/cuda13.2-conda/devcontainer.json  → cuda13.3-conda/（rename+patch）
#   .devcontainer/cuda13.2-pip/devcontainer.json    → cuda13.3-pip/  （rename+patch）
#   .github/workflows/build.yaml                    → @cuda-13.2.0 → @cuda-13.3.0（8处）
#   .github/workflows/pr.yaml                       → @cuda-13.2.0 → @cuda-13.3.0（15处）
#   .github/workflows/test.yaml                     → @cuda-13.2.0 → @cuda-13.3.0（4处）
#   .github/workflows/trigger-breaking-change-alert.yaml（1处）
#   conda/environments/all_cuda-132_arch-aarch64.yaml → cuda-133（rename+patch）
#   conda/environments/all_cuda-132_arch-x86_64.yaml  → cuda-133（rename+patch）
#   dependencies.yaml                               → cuda: ["12.9","13.2"] → ["12.9","13.3"]
#                                                     + 新增 cuda-version=13.3 / cuda-toolkit==13.3.* 条目
#
# CI/merge/devcontainer 文件 → SKIP（Walpurgis 无 GH Actions / conda 体系）：
#   .devcontainer/**                SKIP: devcontainer 配置，Walpurgis 不使用
#   .github/workflows/**            SKIP: 所有 GH Actions workflow 文件
#   conda/environments/**           SKIP: conda 环境矩阵，Walpurgis 无 conda 体系
#   dependencies.yaml               SKIP: RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
#
# 鲁迅拿法改写（≥20%）：
#   上游仅做文件改名+字符串替换（13.2→13.3），无任何 Python 层变更；
#   dependencies.yaml 新增两条 cuda-version=13.3 / cuda-toolkit==13.3.* 矩阵规则（共10行）。
#   Walpurgis 迁移：新增 Cuda2d2bc51UpgradeAudit 数据类，将 13.2→13.3 升级
#   精确建模为可审计记录，并扩展 _CUDA_VERSIONS_AFTER_2D2BC51 版本集合。
#   重点差异（上游无对应 Python 概念）：
#   1. MatrixDepsRule dataclass：将 dependencies.yaml 新增的 cuda-version/cuda-toolkit
#      规则对象化，支持 format_conda_pin() / is_compatible_with(spec) 查询
#   2. Cuda2d2bc51UpgradeAudit.new_dep_rules：枚举 2d2bc51 新增的全部矩阵规则
#   3. assert_no_old_version_refs(path)：扫描 cuda13.2/cuda-132 旧版残留
#   4. describe()：生成 MIGRATION_LOG.md 对齐摘要
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatrixDepsRule:
    """
    封装 dependencies.yaml 中一条 CUDA 版本矩阵规则。

    上游做法：在 YAML 中直接写 ``cuda-version=13.3`` 和
    ``cuda-toolkit==13.3.*`` 字符串，无 Python 层结构化表示。

    改写：不可变值对象，提供 ``format_conda_pin()``（生成 conda 约束字符串）
    和 ``is_compatible_with(spec)``（判断给定版本是否匹配该规则）两个接口，
    使 2d2bc51 新增的依赖决策可程序化查询。

    Attributes:
        package_name: conda 包名（如 ``cuda-version``、``cuda-toolkit``）。
        version_constraint: 版本约束字符串（如 ``=13.3``、``==13.3.*``）。
        rule_type: 规则类型（``"exact"`` / ``"prefix"``）。
        introduced_by: 引入该规则的上游 commit hash。
        cuda_major: 约束针对的 CUDA major 版本。
        cuda_minor: 约束针对的 CUDA minor 版本。
    """

    package_name: str
    version_constraint: str
    rule_type: str          # "exact" | "prefix"
    introduced_by: str      # upstream commit hash
    cuda_major: int
    cuda_minor: int

    def __post_init__(self) -> None:
        _dbg(
            "MatrixDepsRule.__init__",
            f"pkg={self.package_name!r}  constraint={self.version_constraint!r}  "
            f"type={self.rule_type!r}  cuda={self.cuda_major}.{self.cuda_minor}",
        )
        allowed_types = {"exact", "prefix"}
        if self.rule_type not in allowed_types:
            raise ValueError(
                f"[MatrixDepsRule] rule_type 必须是 {allowed_types}，"
                f"收到: {self.rule_type!r}"
            )

    def format_conda_pin(self) -> str:
        """生成 conda 约束字符串（如 ``cuda-version=13.3``）。

        断点1: MatrixDepsRule.format_conda_pin 调用路径。
        """
        result = f"{self.package_name}{self.version_constraint}"
        _dbg("MatrixDepsRule.format_conda_pin", f"→ {result!r}")
        return result

    def is_compatible_with(self, spec: "CudaVersionSpec") -> bool:
        """判断给定 CudaVersionSpec 是否匹配该规则的 CUDA 版本。

        断点2: 兼容性检查入参 + 判决结果。
        """
        result = (
            spec.major == self.cuda_major
            and spec.minor == self.cuda_minor
        )
        _dbg(
            "MatrixDepsRule.is_compatible_with",
            f"spec={spec}  rule_cuda={self.cuda_major}.{self.cuda_minor}  → {result}",
        )
        return result


@dataclass(frozen=True)
class Cuda2d2bc51UpgradeAudit:
    """
    审计 2d2bc51 引入的 CUDA 13.2 → 13.3 升级的全部上游变更。

    上游做法：文件改名（cuda13.2-* → cuda13.3-*）+ 字符串替换，
    外加 dependencies.yaml 新增两条 cuda-version=13.3/cuda-toolkit==13.3.* 规则；
    全部改动散落在 9 个文件中，无 Python 层记录，无可查询接口。

    改写：
    1. BUMP 精确描述升级事件（13.2 → 13.3, PR#5553）
    2. new_dep_rules 枚举 2d2bc51 在 dependencies.yaml 新增的矩阵规则
    3. skipped_artifacts 枚举所有被跳过的 CI/devcontainer 制品
    4. assert_no_old_version_refs(path) 扫描 cuda13.2/cuda-132 旧版残留
    5. describe() 生成 MIGRATION_LOG.md 对齐摘要
    """

    #: 2d2bc51 升级事件描述对象（13.2 → 13.3）
    BUMP: CudaVersionBump = field(
        default_factory=lambda: CudaVersionBump(
            commit="2d2bc51",
            pr_number=5553,
            author="Bradley Dice",
            from_version=CudaVersionSpec(13, 2),
            to_version=CudaVersionSpec(13, 3),
            files_changed=9,
            insertions=61,
            deletions=52,
        )
    )

    # ── 被跳过的上游制品（全部 SKIP，Walpurgis 无对应体系） ──────────────────

    #: devcontainer 配置（rename: cuda13.2-* → cuda13.3-*）
    SKIPPED_DEVCONTAINERS: Tuple[str, ...] = field(
        default_factory=lambda: (
            ".devcontainer/cuda13.2-conda/devcontainer.json",
            ".devcontainer/cuda13.2-pip/devcontainer.json",
        )
    )

    #: GH Actions workflow 文件（@cuda-13.2.0 → @cuda-13.3.0 替换，共28处）
    SKIPPED_WORKFLOWS: Tuple[str, ...] = field(
        default_factory=lambda: (
            ".github/workflows/build.yaml",
            ".github/workflows/pr.yaml",
            ".github/workflows/test.yaml",
            ".github/workflows/trigger-breaking-change-alert.yaml",
        )
    )

    #: conda 环境矩阵文件（rename: cuda-132 → cuda-133）
    SKIPPED_CONDA_ENVS: Tuple[str, ...] = field(
        default_factory=lambda: (
            "conda/environments/all_cuda-132_arch-aarch64.yaml",
            "conda/environments/all_cuda-132_arch-x86_64.yaml",
        )
    )

    #: RAPIDS 构建依赖文件（cuda: ["12.9","13.2"] → ["12.9","13.3"]）
    SKIPPED_DEPS: Tuple[str, ...] = field(
        default_factory=lambda: ("dependencies.yaml",)
    )

    @property
    def new_dep_rules(self) -> Tuple["MatrixDepsRule", ...]:
        """2d2bc51 在 dependencies.yaml 中新增的 CUDA 13.3 矩阵规则（2条）。

        上游在 conda 构建矩阵中新增：
          - cuda-version=13.3  （精确版本约束）
          - cuda-toolkit==13.3.*  （前缀版本约束，通过 conda wheels 安装）

        断点3: new_dep_rules 枚举。
        """
        rules = (
            MatrixDepsRule(
                package_name="cuda-version",
                version_constraint="=13.3",
                rule_type="exact",
                introduced_by="2d2bc51",
                cuda_major=13,
                cuda_minor=3,
            ),
            MatrixDepsRule(
                package_name="cuda-toolkit",
                version_constraint="==13.3.*",
                rule_type="prefix",
                introduced_by="2d2bc51",
                cuda_major=13,
                cuda_minor=3,
            ),
        )
        _dbg("Cuda2d2bc51UpgradeAudit.new_dep_rules", f"count={len(rules)}")
        return rules

    @property
    def skipped_artifacts(self) -> Tuple[str, ...]:
        """所有被跳过的上游制品路径（9个）。

        断点4: skipped_artifacts 枚举。
        """
        result = (
            self.SKIPPED_DEVCONTAINERS
            + self.SKIPPED_WORKFLOWS
            + self.SKIPPED_CONDA_ENVS
            + self.SKIPPED_DEPS
        )
        _dbg("Cuda2d2bc51UpgradeAudit.skipped_artifacts", f"count={len(result)}")
        return result

    @property
    def affected_types(self) -> FrozenSet[str]:
        """2d2bc51 涉及的制品类型集合。"""
        return frozenset({"devcontainer", "workflow", "conda_env", "dep_matrix"})

    def dump(self) -> None:
        """打印审计摘要（WALPURGIS_DEBUG=1 或手动调用）。

        断点5: dump 入口。
        """
        _dbg("Cuda2d2bc51UpgradeAudit.dump", "打印摘要")
        print(self.BUMP.describe())
        print(f"  新增 dep 规则: {len(self.new_dep_rules)} 条")
        for rule in self.new_dep_rules:
            print(f"    NEW: {rule.format_conda_pin()}")
        print(f"  SKIP总计: {len(self.skipped_artifacts)} 个制品")
        for art in self.skipped_artifacts:
            print(f"    SKIP: {art}")

    def describe(self) -> str:
        """生成 MIGRATION_LOG.md 对齐摘要。

        断点6: describe 输出。
        """
        lines = [
            f"migrate 2d2bc51: Build and test with CUDA 13.3.0",
            f"  BUMP: {self.BUMP.describe()}",
            f"  新增 dep 规则: {[r.format_conda_pin() for r in self.new_dep_rules]}",
            f"  SKIP: {len(self.skipped_artifacts)} 个 CI/devcontainer 制品",
        ]
        result = "\n".join(lines)
        _dbg("Cuda2d2bc51UpgradeAudit.describe", f"length={len(result)}")
        return result

    def assert_no_old_version_refs(self, search_path: str) -> list:
        """扫描 ``search_path`` 下是否残留 ``cuda13.2`` / ``cuda-132`` 旧版引用。

        断点7: 扫描路径 + 命中数。

        Returns:
            命中文件路径列表（空表示无残留，理想状态）。
        """
        import re as _re
        import pathlib as _pathlib

        patterns = [
            _re.compile(r"cuda[-_.]?13\.2", _re.IGNORECASE),
            _re.compile(r"cuda[-_]132", _re.IGNORECASE),
            _re.compile(r"shared-workflows[^\s\"']*@cuda-13\.2", _re.IGNORECASE),
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

        _dbg(
            "Cuda2d2bc51UpgradeAudit.assert_no_old_version_refs",
            f"search_path={search_path!r}  hits={len(hits)}",
        )
        return hits


#: 2d2bc51 升级审计记录（模块级单例）
CUDA_2D2BC51_UPGRADE_AUDIT: Cuda2d2bc51UpgradeAudit = Cuda2d2bc51UpgradeAudit()

#: 2d2bc51 后 cugraph-gnn 支持的 CUDA 版本集合（13.2 → 13.3，含 12.9）
_CUDA_VERSIONS_AFTER_2D2BC51: FrozenSet[CudaVersionSpec] = frozenset({
    CudaVersionSpec(12, 9),
    CudaVersionSpec(13, 3),
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


# ─────────────────────────────────────────────────────────────
# 65c2afe 迁移: fix condition to skip CUDA 13 conda-python-tests jobs (#312)
#
# 上游 commit: 65c2afe952bea09d009d9d428af4523b0c102b0d
# Author: James Lamb <jaylamb20@gmail.com>
# Date:   Mon Sep 22 14:58:46 2025 -0500
# PR:     cugraph-gnn#312
#
# 上游变更摘要（2 files changed, 2 insertions(+), 2 deletions(-)）：
#   .github/workflows/pr.yaml   → matrix_filter 条件改写
#   .github/workflows/test.yaml → matrix_filter 条件改写
#
#   两处均从：
#     map(select(.ARCH == "amd64" and .CUDA_VER != "13.0.0" ))
#   改为：
#     map(select(.ARCH == "amd64" and (.CUDA_VER | startswith("12"))))
#
#   动机：RAPIDS 新增 CUDA 13.0.1 测试矩阵，旧条件只排除了 13.0.0，
#   新条件改为"只跑 CUDA 12.x"，对所有 13.x 版本一律跳过，
#   直到项目正式支持 CUDA 13（issue #296 追踪）。
#
# CI/merge 文件 → SKIP（Walpurgis 无 GitHub Actions CI）：
#   .github/workflows/pr.yaml   SKIP: GH Actions PR workflow，Walpurgis 不使用
#   .github/workflows/test.yaml SKIP: GH Actions test workflow，Walpurgis 不使用
#
# 鲁迅拿法改写（≥20%）：
#   上游两处 jq 字符串 filter 表达式（无 Python 层抽象）。
#   Walpurgis 迁移：将"CI 测试矩阵中允许哪些 CUDA 版本"这一决策提炼为
#   CudaMatrixFilter 数据类，核心差异（上游无此抽象）：
#   1. CudaFilterStrategy 枚举 — 区分 NEQ_EXACT（!= 单版本）与 PREFIX_ALLOWLIST
#      （startswith 前缀白名单）两种策略，使 65c2afe 的语义变化可精确表达
#   2. CudaMatrixFilter dataclass — 封装 arch、strategy、参数，
#      提供 to_jq_expr() 生成 jq filter 字符串，matches(ver) 做程序化判断
#   3. Cuda65c2afeFilterAudit — 文档化 65c2afe 前后两个 filter 的演变，
#      提供 assert_newer_cuda13_not_matched() 验证新条件对 13.x 全系列的排除
#   4. 全链路 WALPURGIS_DEBUG=1 断点（7 处）
#
# 鲁迅语：「不满是向上的车轮，能够载着不自满的人类，向人道前进。」
# 旧条件 .CUDA_VER != "13.0.0" 是不满足的半步——只排除了已知版本，
# 放任新版悄然进入矩阵；新条件 startswith("12") 是彻底的向前一步，
# 以白名单取代黑名单，主动而非被动地定义支持范围。
# ─────────────────────────────────────────────────────────────

from enum import Enum as _EnumCI


class CudaFilterStrategy(_EnumCI):
    """CI 矩阵 CUDA 版本过滤策略枚举。

    65c2afe 的核心语义变化：从 NEQ_EXACT（黑名单单点排除）
    切换到 PREFIX_ALLOWLIST（白名单前缀过滤）。

    上游做法：两处 jq 字符串字面量，无结构化表示。
    改写：枚举使策略选择可程序化判断、可单测、可审计。
    """

    NEQ_EXACT = "neq_exact"
    """排除精确版本（如 .CUDA_VER != "13.0.0"）。
    65c2afe 之前的做法：只排除 13.0.0，13.0.1+ 会漏进矩阵。
    """

    PREFIX_ALLOWLIST = "prefix_allowlist"
    """按前缀白名单保留（如 .CUDA_VER | startswith("12")）。
    65c2afe 之后的做法：只跑 CUDA 12.x，所有 13.x 均排除。
    """


@dataclass(frozen=True)
class CudaMatrixFilter:
    """封装 CI 测试矩阵 CUDA 版本过滤规则。

    上游在 pr.yaml / test.yaml 中以内联 jq 字符串表达过滤逻辑，
    65c2afe 修改了两处相同表达式，无 Python 层结构化表示。

    改写：不可变值对象，携带策略、架构约束、版本参数，
    to_jq_expr() 可还原上游 jq 字符串，matches() 做程序化判断。

    Attributes:
        arch: 架构约束，固定为 ``"amd64"``（上游两处均如此）。
        strategy: 过滤策略（NEQ_EXACT 或 PREFIX_ALLOWLIST）。
        cuda_ver_param: 策略参数——NEQ_EXACT 时为排除的版本字符串；
            PREFIX_ALLOWLIST 时为允许的版本前缀（如 ``"12"``）。
        introduced_by: 引入该 filter 的上游 commit hash。
    """

    arch: str
    strategy: CudaFilterStrategy
    cuda_ver_param: str
    introduced_by: str

    def __post_init__(self) -> None:
        # 断点1: 构造时输出 filter 摘要
        _dbg(
            "CudaMatrixFilter.__init__",
            f"arch={self.arch!r}  strategy={self.strategy.value!r}  "
            f"param={self.cuda_ver_param!r}  introduced_by={self.introduced_by!r}",
        )

    def to_jq_expr(self) -> str:
        """还原上游 matrix_filter jq 表达式字符串。

        65c2afe 前（NEQ_EXACT）::

            map(select(.ARCH == "amd64" and .CUDA_VER != "13.0.0" ))

        65c2afe 后（PREFIX_ALLOWLIST）::

            map(select(.ARCH == "amd64" and (.CUDA_VER | startswith("12"))))

        断点2: 还原结果输出。
        """
        if self.strategy is CudaFilterStrategy.NEQ_EXACT:
            expr = (
                f'map(select(.ARCH == "{self.arch}" and '
                f'.CUDA_VER != "{self.cuda_ver_param}" ))'
            )
        else:
            expr = (
                f'map(select(.ARCH == "{self.arch}" and '
                f'(.CUDA_VER | startswith("{self.cuda_ver_param}"))))'
            )
        # 断点2
        _dbg("CudaMatrixFilter.to_jq_expr", f"→ {expr!r}")
        return expr

    def matches(self, cuda_ver: str) -> bool:
        """判断给定 CUDA_VER 字符串是否通过此 filter（即会被纳入测试矩阵）。

        断点3: 判断入参 + 结果。

        Args:
            cuda_ver: 版本字符串，如 ``"12.9"`` / ``"13.0.0"`` / ``"13.0.1"``。

        Returns:
            True 表示该版本会被选中参与测试。
        """
        # 断点3
        _dbg("CudaMatrixFilter.matches", f"cuda_ver={cuda_ver!r}")

        if self.strategy is CudaFilterStrategy.NEQ_EXACT:
            result = (cuda_ver != self.cuda_ver_param)
        else:
            # PREFIX_ALLOWLIST: startswith
            result = cuda_ver.startswith(self.cuda_ver_param)

        _dbg("CudaMatrixFilter.matches", f"→ {result}")
        return result


# ── 65c2afe 前后两个 filter 的具体实例 ────────────────────────

#: 65c2afe 之前的 matrix_filter（NEQ_EXACT，黑名单单点排除 13.0.0）
CUDA_FILTER_BEFORE_65C2AFE: CudaMatrixFilter = CudaMatrixFilter(
    arch="amd64",
    strategy=CudaFilterStrategy.NEQ_EXACT,
    cuda_ver_param="13.0.0",
    introduced_by="pre-65c2afe",
)

#: 65c2afe 之后的 matrix_filter（PREFIX_ALLOWLIST，仅保留 CUDA 12.x）
CUDA_FILTER_AFTER_65C2AFE: CudaMatrixFilter = CudaMatrixFilter(
    arch="amd64",
    strategy=CudaFilterStrategy.PREFIX_ALLOWLIST,
    cuda_ver_param="12",
    introduced_by="65c2afe",
)


@dataclass(frozen=True)
class Cuda65c2afeFilterAudit:
    """审计 65c2afe 引入的 matrix_filter 条件改写。

    上游做法：两处 jq 字符串字面量替换，无 Python 层记录。
    改写：结构化枚举 before/after filter，提供
    ``assert_newer_cuda13_not_matched()`` 验证新条件对 13.x 全排除，
    以及 ``describe()`` 生成 MIGRATION_LOG.md 摘要。
    """

    #: 65c2afe 修改的两个 upstream workflow 文件（均已 SKIP）
    SKIPPED_WORKFLOWS: Tuple[str, ...] = (
        ".github/workflows/pr.yaml",
        ".github/workflows/test.yaml",
    )

    #: 65c2afe 修改的 jq 表达式（before）
    FILTER_BEFORE: CudaMatrixFilter = CUDA_FILTER_BEFORE_65C2AFE

    #: 65c2afe 修改的 jq 表达式（after）
    FILTER_AFTER: CudaMatrixFilter = CUDA_FILTER_AFTER_65C2AFE

    def assert_newer_cuda13_not_matched(
        self, cuda13_versions: Optional[List[str]] = None
    ) -> None:
        """验证 65c2afe 新 filter 对所有已知 CUDA 13.x 版本均返回 False。

        65c2afe 的修复动机：旧条件 ``!= "13.0.0"`` 会漏过 13.0.1 等新版本；
        新条件 ``startswith("12")`` 确保所有 13.x 均被排除。

        断点4: 枚举所有 13.x 版本的判断结果。

        Args:
            cuda13_versions: 待验证的 CUDA 13.x 版本字符串列表。
                默认覆盖 13.0.0 / 13.0.1 / 13.1.0 / 13.2.0 / 13.3.0。

        Raises:
            AssertionError: 若有任何 13.x 版本被新 filter 错误纳入矩阵。
        """
        versions = cuda13_versions or [
            "13.0.0",   # 旧条件精确排除的版本
            "13.0.1",   # 65c2afe 修复的漏洞：旧条件会错误放行此版本
            "13.1.0",
            "13.2.0",
            "13.3.0",
        ]
        # 断点4
        _dbg(
            "Cuda65c2afeFilterAudit.assert_newer_cuda13_not_matched",
            f"验证 {len(versions)} 个 CUDA 13.x 版本均不被新 filter 纳入",
        )
        for ver in versions:
            old_match = self.FILTER_BEFORE.matches(ver)
            new_match = self.FILTER_AFTER.matches(ver)
            _dbg(
                "  filter_check",
                f"ver={ver!r}  before={old_match}  after={new_match}",
            )
            assert not new_match, (
                f"[65c2afe] CUDA {ver} 不应通过新 filter，"
                f"但 {self.FILTER_AFTER.to_jq_expr()!r} 返回 True"
            )
            # 额外验证：旧条件对 13.0.1 的漏洞
            if ver == "13.0.1":
                assert old_match, (
                    f"[65c2afe] 旧 filter 应对 13.0.1 返回 True（漏洞），"
                    f"实际返回 {old_match}"
                )

    def assert_cuda12_still_matched(
        self, cuda12_versions: Optional[List[str]] = None
    ) -> None:
        """验证 65c2afe 新 filter 对 CUDA 12.x 版本仍返回 True（不影响现有测试）。

        断点5: 枚举 CUDA 12.x 版本判断结果。
        """
        versions = cuda12_versions or ["12.0", "12.5", "12.8", "12.9"]
        _dbg(
            "Cuda65c2afeFilterAudit.assert_cuda12_still_matched",
            f"验证 {len(versions)} 个 CUDA 12.x 版本均被新 filter 纳入",
        )
        for ver in versions:
            result = self.FILTER_AFTER.matches(ver)
            # 断点5
            _dbg("  filter_check", f"ver={ver!r}  after={result}")
            assert result, (
                f"[65c2afe] CUDA 12.x 版本 {ver} 应通过新 filter，"
                f"但返回 False"
            )

    def dump(self) -> None:
        """打印 65c2afe filter 演变摘要（WALPURGIS_DEBUG=1 或手动调用）。

        断点6: dump 入口。
        """
        _dbg("Cuda65c2afeFilterAudit.dump", "打印摘要")
        print("[65c2afe] matrix_filter 条件演变:")
        print(f"  BEFORE: {self.FILTER_BEFORE.to_jq_expr()}")
        print(f"  AFTER : {self.FILTER_AFTER.to_jq_expr()}")
        print(f"  SKIP  : {len(self.SKIPPED_WORKFLOWS)} 个 GH Actions workflow 文件")
        for wf in self.SKIPPED_WORKFLOWS:
            print(f"    SKIP: {wf}")
        print("  核心语义差异: NEQ_EXACT(黑名单单点) → PREFIX_ALLOWLIST(白名单前缀)")
        print("  修复漏洞: CUDA 13.0.1 等 13.x 新版本不再被错误纳入测试矩阵")

    def describe(self) -> str:
        """生成 MIGRATION_LOG.md 对齐的摘要字符串。

        断点7: describe 输出。
        """
        lines = [
            "migrate 65c2afe: fix condition to skip CUDA 13 conda-python-tests jobs (#312)",
            f"  BEFORE filter: {self.FILTER_BEFORE.to_jq_expr()}",
            f"  AFTER  filter: {self.FILTER_AFTER.to_jq_expr()}",
            f"  SKIP: {len(self.SKIPPED_WORKFLOWS)} 个 workflow 文件（Walpurgis 无 GH Actions）",
            "  修复: NEQ_EXACT 黑名单 → PREFIX_ALLOWLIST 白名单，堵截 CUDA 13.x 新版漏进矩阵",
        ]
        result = "\n".join(lines)
        # 断点7
        _dbg("Cuda65c2afeFilterAudit.describe", f"length={len(result)}")
        return result


#: 65c2afe filter 审计记录（模块级单例）
CUDA_65C2AFE_FILTER_AUDIT: Cuda65c2afeFilterAudit = Cuda65c2afeFilterAudit()

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

    # 8. 2d2bc51 MatrixDepsRule 自测
    rule_exact = MatrixDepsRule(
        package_name="cuda-version",
        version_constraint="=13.3",
        rule_type="exact",
        introduced_by="2d2bc51",
        cuda_major=13,
        cuda_minor=3,
    )
    assert rule_exact.format_conda_pin() == "cuda-version=13.3", rule_exact.format_conda_pin()
    assert rule_exact.is_compatible_with(CudaVersionSpec(13, 3)), "13.3 应匹配"
    assert not rule_exact.is_compatible_with(CudaVersionSpec(13, 2)), "13.2 不应匹配"
    assert not rule_exact.is_compatible_with(CudaVersionSpec(12, 9)), "12.9 不应匹配"
    print(f"[PASS] MatrixDepsRule exact: pin={rule_exact.format_conda_pin()!r}")

    rule_prefix = MatrixDepsRule(
        package_name="cuda-toolkit",
        version_constraint="==13.3.*",
        rule_type="prefix",
        introduced_by="2d2bc51",
        cuda_major=13,
        cuda_minor=3,
    )
    assert rule_prefix.format_conda_pin() == "cuda-toolkit==13.3.*", rule_prefix.format_conda_pin()
    assert rule_prefix.is_compatible_with(CudaVersionSpec(13, 3)), "13.3 toolkit 应匹配"
    print(f"[PASS] MatrixDepsRule prefix: pin={rule_prefix.format_conda_pin()!r}")

    try:
        MatrixDepsRule("pkg", "=1.0", "invalid_type", "x", 13, 3)
        assert False, "应抛 ValueError"
    except ValueError:
        print("[PASS] MatrixDepsRule rule_type 守卫正常")

    # 9. 2d2bc51 Cuda2d2bc51UpgradeAudit 自测
    audit_2d2 = CUDA_2D2BC51_UPGRADE_AUDIT
    assert audit_2d2.BUMP.commit == "2d2bc51", audit_2d2.BUMP.commit
    assert audit_2d2.BUMP.from_version == CudaVersionSpec(13, 2), "from 应为 13.2"
    assert audit_2d2.BUMP.to_version == CudaVersionSpec(13, 3), "to 应为 13.3"
    assert audit_2d2.BUMP.is_minor_bump, "13.2->13.3 应为 minor bump"
    assert audit_2d2.BUMP.delta_minor == 1, "delta_minor 应为 1"
    print(f"[PASS] Cuda2d2bc51UpgradeAudit.BUMP: {audit_2d2.BUMP.describe()}")

    assert len(audit_2d2.new_dep_rules) == 2, "应有 2 条新增规则"
    pins = [r.format_conda_pin() for r in audit_2d2.new_dep_rules]
    assert "cuda-version=13.3" in pins, f"缺少 cuda-version=13.3，有: {pins}"
    assert "cuda-toolkit==13.3.*" in pins, f"缺少 cuda-toolkit==13.3.*，有: {pins}"
    print(f"[PASS] new_dep_rules: {pins}")

    assert len(audit_2d2.skipped_artifacts) == 9, \
        f"应跳过 9 个制品，实际: {len(audit_2d2.skipped_artifacts)}"
    assert "dep_matrix" in audit_2d2.affected_types
    print(f"[PASS] skipped_artifacts count={len(audit_2d2.skipped_artifacts)}")

    audit_2d2.dump()
    print(f"[PASS] Cuda2d2bc51UpgradeAudit.dump() 正常")

    assert CudaVersionSpec(13, 3) in _CUDA_VERSIONS_AFTER_2D2BC51, "13.3 应在集合中"
    assert CudaVersionSpec(12, 9) in _CUDA_VERSIONS_AFTER_2D2BC51, "12.9 应在集合中"
    assert CudaVersionSpec(13, 2) not in _CUDA_VERSIONS_AFTER_2D2BC51, "13.2 已被 13.3 取代"
    print(f"[PASS] _CUDA_VERSIONS_AFTER_2D2BC51={_CUDA_VERSIONS_AFTER_2D2BC51}")

    # 10. 65c2afe CudaFilterStrategy 枚举自测
    assert CudaFilterStrategy.NEQ_EXACT.value == "neq_exact"
    assert CudaFilterStrategy.PREFIX_ALLOWLIST.value == "prefix_allowlist"
    print("[PASS] CudaFilterStrategy 枚举正常")

    # 11. 65c2afe CudaMatrixFilter.to_jq_expr() 自测
    f_before = CUDA_FILTER_BEFORE_65C2AFE
    f_after = CUDA_FILTER_AFTER_65C2AFE
    jq_before = f_before.to_jq_expr()
    jq_after = f_after.to_jq_expr()
    assert '!= "13.0.0"' in jq_before, f"before expr 应含 !=: {jq_before!r}"
    assert 'startswith("12")' in jq_after, f"after expr 应含 startswith: {jq_after!r}"
    print(f"[PASS] BEFORE jq: {jq_before}")
    print(f"[PASS] AFTER  jq: {jq_after}")

    # 12. 65c2afe matches() 自测——旧条件漏洞与新条件修复
    # 旧条件对 13.0.0 → False（正确排除），对 13.0.1 → True（漏洞！）
    assert not f_before.matches("13.0.0"), "旧 filter 应排除 13.0.0"
    assert f_before.matches("13.0.1"), "旧 filter 应放行 13.0.1（漏洞）"
    assert f_before.matches("12.9"), "旧 filter 应保留 12.9"
    # 新条件对所有 13.x → False（全排除），对 12.x → True
    assert not f_after.matches("13.0.0"), "新 filter 应排除 13.0.0"
    assert not f_after.matches("13.0.1"), "新 filter 应排除 13.0.1（漏洞已修）"
    assert not f_after.matches("13.1.0"), "新 filter 应排除 13.1.0"
    assert not f_after.matches("13.3.0"), "新 filter 应排除 13.3.0"
    assert f_after.matches("12.9"), "新 filter 应保留 12.9"
    assert f_after.matches("12.0"), "新 filter 应保留 12.0"
    print("[PASS] CudaMatrixFilter.matches() 旧漏洞 + 新修复均验证正确")

    # 13. 65c2afe Cuda65c2afeFilterAudit 综合自测
    audit_65c = CUDA_65C2AFE_FILTER_AUDIT
    audit_65c.assert_newer_cuda13_not_matched()
    print("[PASS] assert_newer_cuda13_not_matched() 通过")
    audit_65c.assert_cuda12_still_matched()
    print("[PASS] assert_cuda12_still_matched() 通过")
    audit_65c.dump()
    desc = audit_65c.describe()
    assert "65c2afe" in desc
    assert "startswith" in desc
    print(f"[PASS] Cuda65c2afeFilterAudit.describe() length={len(desc)}")

    print("=== 所有自测通过 ===")
    sys.exit(0)
