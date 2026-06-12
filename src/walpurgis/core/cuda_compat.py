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

    print("=== 所有自测通过 ===")
    sys.exit(0)
