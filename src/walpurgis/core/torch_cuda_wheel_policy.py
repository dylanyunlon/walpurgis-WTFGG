# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: cugraph-gnn commit 330b135
# 原标题: ensure 'torch' CUDA wheels are installed in CI,
#         test that 'torch' is an optional dependency (#425)
# 上游 PR: https://github.com/rapidsai/cugraph-gnn/pull/425
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 「不满是向上的车轮，能够载着不自满的人类，向人道前进。」
# —— 鲁迅《热风·随感录二十五》
#
# 330b135 做了两件事，彼此独立却互相支撑，如同车轮的两侧：
#
# 第一件：在 CI 里强制安装 CUDA 版本的 torch wheel。
#   CPU-only torch 从 PyPI 悄悄混入——这是一类"无声失败"，
#   测试跑完，数字看起来对，但 GPU 从未参与过运算。
#   上游的解法是写 ci/download-torch-wheels.sh：
#     · 根据 CUDA 主次版本号决定是否有可用的 CUDA torch wheel
#     · 先 download 再 install（而非 --extra-index-url），
#       防止 pip 仍从 pypi.org 拉到 CPU-only 版本
#     · 使用 local version tag (+cu130) 锁住 CUDA variant
#
# 第二件：让 torch 真正成为可选依赖。
#   DLFW 构建环境不安装 torch，但要求 pylibwholegraph / cugraph-pyg
#   仍然可 import。原代码在模块顶层裸写 `import torch`，一旦 torch
#   不存在就整包崩溃。330b135 的修复：
#     · 将 import_optional 机制从 cugraph-pyg 复制到 pylibwholegraph
#     · 所有 `import torch` 替换为 `torch = import_optional("torch")`
#     · `torch.Tensor` 等类型注解改为运行时解析（避免 import 时求值）
#     · 新增 ruff flake8-tidy-imports banned-api 规则，
#       禁止代码库内出现裸 `import torch`
#     · CI 增加 pip uninstall torch 后仍能运行的单元测试
#     · ci/validate_wheel.sh 检查 wheel metadata 里无 torch 依赖项
#
# 主要涉及文件（47 个文件，775 行增加，323 行删除）：
#   - ci/download-torch-wheels.sh（新增）：CUDA wheel 下载脚本
#   - ci/uninstall-torch-wheels.sh（新增）：torch 卸载后测试入口
#   - ci/validate_wheel.sh（新增）：wheel metadata 中 torch 依赖检查
#   - python/pylibwholegraph/pylibwholegraph/utils/imports.py（新增）：
#       与 cugraph-pyg 相同的 MissingModule + import_optional
#   - python/pylibwholegraph/pylibwholegraph/torch/*.py：
#       全部 `import torch` → `torch = import_optional("torch")`
#   - python/cugraph-pyg/cugraph_pyg/utils/imports.py：
#       新增 TorchImportGuard（torch 专用 MissingModule 子类）
#   - pyproject.toml（根目录）：新增 ruff banned-api 配置
#
# Walpurgis 20% 改写要点：
#   1. CudaVersionSpec dataclass — 将"CUDA major.minor → 是否有可用 torch wheel"
#      的判断逻辑从 bash 脚本提升为 Python 数据类，
#      支持 is_torch_cuda_wheel_available() 查询
#   2. TorchOptionalImport — 专门针对 torch 的 ImportGuard 子类，
#      在 __getattr__ 中额外提示"检查是否安装了 CUDA variant"
#   3. TorchImportPolicy 枚举 — 三档策略：
#      REQUIRE（测试时必须有）/ OPTIONAL（运行时可缺席）/ BANNED（不允许出现）
#   4. validate_no_torch_in_metadata() 函数 — Python 版本的 validate_wheel.sh 逻辑，
#      检查给定 wheel 的 METADATA 文件中 Requires-Dist 是否包含 torch
#   5. WALPURGIS_DEBUG=1 全链路输出：CUDA 版本检测、wheel 可用性判断、
#      torch import 尝试结果、metadata 校验过程

from __future__ import annotations

import enum
import importlib
import os as _os
import re
import sys as _sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG:{tag}] {msg}", file=_sys.stderr, flush=True)


_dbg("torch_cuda_wheel", "module init — 迁移自 cugraph-gnn 330b135")

# ---------------------------------------------------------------------------
# 数据类：CUDA 版本规格
# ---------------------------------------------------------------------------

@dataclass
class CudaVersionSpec:
    """
    封装 CUDA 版本号，并提供 torch CUDA wheel 可用性查询。

    上游 ci/download-torch-wheels.sh 用 bash 算术比较判断：
      - CUDA 12.x 且 minor >= 9 → 有 CUDA torch wheel
      - CUDA 13.0 → 有 CUDA torch wheel
      - 其余 → 跳过下载

    Walpurgis 将此逻辑提升为 Python 数据类，便于在测试与审计中复用。
    """

    major: int
    minor: int

    @classmethod
    def from_string(cls, version_str: str) -> "CudaVersionSpec":
        """
        解析 "12.9"、"13.0" 等格式的 CUDA 版本字符串。
        """
        m = re.match(r"^(\d+)\.(\d+)", version_str.strip())
        if not m:
            raise ValueError(f"[Walpurgis:CudaVersionSpec] 无法解析版本字符串: {version_str!r}")
        spec = cls(major=int(m.group(1)), minor=int(m.group(2)))
        _dbg("CudaVersionSpec.from_string", f"{version_str!r} → major={spec.major} minor={spec.minor}")
        return spec

    @classmethod
    def from_env(cls) -> Optional["CudaVersionSpec"]:
        """从 RAPIDS_CUDA_VERSION 环境变量读取版本。"""
        val = _os.getenv("RAPIDS_CUDA_VERSION", "")
        if not val:
            _dbg("CudaVersionSpec.from_env", "RAPIDS_CUDA_VERSION 未设置")
            return None
        return cls.from_string(val)

    def is_torch_cuda_wheel_available(self) -> bool:
        """
        判断当前 CUDA 版本是否有可下载的 torch CUDA wheel。

        根据上游 330b135 的条件（bash 脚本逻辑对应）：
          有效范围: CUDA 12.9+ 或 CUDA 13.0
          无效范围: CUDA 12.x (x < 9), CUDA 13.x (x > 0), CUDA 14+
        """
        available = (
            (self.major == 12 and self.minor >= 9)
            or (self.major == 13 and self.minor == 0)
        )
        _dbg(
            "is_torch_cuda_wheel_available",
            f"CUDA {self.major}.{self.minor} → {'✓ 可用' if available else '✗ 不可用'}",
        )
        return available

    def torch_index_url(self) -> str:
        """
        返回对应 CUDA 版本的 torch 下载 index URL。

        上游 330b135 之前用 cu126/cu130 硬编码；
        Walpurgis 根据 CUDA 版本动态生成，使用 local version tag 格式。
        """
        cuda_tag = f"cu{self.major}{self.minor:02d}"
        url = f"https://download.pytorch.org/whl/{cuda_tag}"
        _dbg("torch_index_url", f"→ {url}")
        return url


# ---------------------------------------------------------------------------
# 枚举：torch 导入策略
# ---------------------------------------------------------------------------

class TorchImportPolicy(enum.Enum):
    """
    描述代码模块对 torch 的依赖策略。

    330b135 在三个层面处理 torch：
      - 运行时库代码: OPTIONAL（import_optional 替代裸 import）
      - 测试代码: REQUIRE（pytest.importorskip("torch") 跳过而非失败）
      - ruff banned-api: BANNED（禁止裸 import torch 出现在库代码中）
    """

    REQUIRE = "require"      # 测试时必须存在；缺失则 skip 整个测试
    OPTIONAL = "optional"    # 运行时可缺席；缺失返回 TorchOptionalImport
    BANNED = "banned"        # ruff 规则：库代码中不允许裸 import torch


# ---------------------------------------------------------------------------
# TorchOptionalImport：torch 专用的 MissingModule
# ---------------------------------------------------------------------------

class TorchOptionalImport:
    """
    torch 专用的 ImportGuard。

    上游 pylibwholegraph/utils/imports.py 中的 MissingModule 只输出
    "This feature requires the torch package/module"。
    Walpurgis 版本额外提示用户检查是否安装了 CUDA variant（而非 CPU-only）。
    """

    def __init__(self, mod_name: str = "torch"):
        self._name = mod_name
        _dbg("TorchOptionalImport.__init__", f"创建 TorchOptionalImport({mod_name!r})")

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        cuda_spec = CudaVersionSpec.from_env()
        cuda_hint = ""
        if cuda_spec:
            if cuda_spec.is_torch_cuda_wheel_available():
                cuda_hint = (
                    f"\n  CUDA wheel 提示: pip install torch "
                    f"--index-url {cuda_spec.torch_index_url()}"
                )
            else:
                cuda_hint = (
                    f"\n  当前 CUDA {cuda_spec.major}.{cuda_spec.minor} "
                    f"无对应的 torch CUDA wheel（需要 12.9+ 或 13.0）"
                )
        _dbg(
            "TorchOptionalImport.__getattr__",
            f"属性 {attr!r} 被访问但 torch 未安装",
        )
        raise RuntimeError(
            f"[Walpurgis:TorchOptionalImport] 此功能需要安装 {self._name!r}。\n"
            f"  基本安装: pip install torch\n"
            f"  注意: 确保安装的是 CUDA variant 而非 CPU-only 版本。"
            f"{cuda_hint}\n"
            f"  参考: cugraph-gnn 330b135 (PR #425) — torch 已设为可选依赖"
        )


# ---------------------------------------------------------------------------
# import_optional：torch 感知版本
# ---------------------------------------------------------------------------

def import_optional(
    mod: str,
    default_mod_class: Optional[Type] = None,
) -> Any:
    """
    尝试 import mod，失败时返回替代对象。

    与上游 pylibwholegraph/utils/imports.py 的 import_optional 等价，
    但对 torch（及 torch.* 子模块）自动使用 TorchOptionalImport。

    Parameters
    ----------
    mod:
        模块名（如 "torch"、"torch.nn"）
    default_mod_class:
        None 时自动推断（torch 系列 → TorchOptionalImport，其余 → MissingModule）
    """
    _dbg("import_optional", f"尝试 import {mod!r}")
    try:
        result = importlib.import_module(mod)
        _dbg("import_optional", f"{mod!r} 加载成功")
        return result
    except ModuleNotFoundError:
        _dbg("import_optional", f"{mod!r} 不可用")
        if default_mod_class is not None:
            return default_mod_class(mod)
        if mod == "torch" or mod.startswith("torch."):
            return TorchOptionalImport(mod)
        return _MissingModule(mod)


class _MissingModule:
    """通用 MissingModule（非 torch 模块的 fallback）。"""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        raise RuntimeError(
            f"[Walpurgis:MissingModule] 此功能需要 {self._name!r} 包。"
        )


# ---------------------------------------------------------------------------
# validate_no_torch_in_metadata：wheel metadata 检查
# ---------------------------------------------------------------------------

def validate_no_torch_in_metadata(wheel_path: str) -> List[str]:
    """
    检查 wheel 文件的 METADATA 中 Requires-Dist 是否包含 torch。

    Python 版本的 ci/validate_wheel.sh 逻辑（330b135 新增）。
    上游 bash 脚本用 grep 检查 *.dist-info/METADATA；
    Walpurgis 用 zipfile + re 实现等价逻辑，返回违规行列表。

    Parameters
    ----------
    wheel_path:
        .whl 文件路径

    Returns
    -------
    list of str:
        包含 "torch" 的 Requires-Dist 行（为空表示通过检查）
    """
    import zipfile

    _dbg("validate_no_torch_in_metadata", f"检查 {wheel_path!r}")
    violations: List[str] = []

    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            metadata_files = [
                n for n in zf.namelist()
                if n.endswith("/METADATA") or n == "METADATA"
            ]
            for meta_file in metadata_files:
                content = zf.read(meta_file).decode("utf-8", errors="replace")
                for line in content.splitlines():
                    if line.startswith("Requires-Dist:"):
                        # 匹配 "torch"（整词，不匹配 pytorch-lightning 等）
                        if re.search(r"\btorch\b", line):
                            violations.append(line.strip())
                            _dbg(
                                "validate_no_torch_in_metadata",
                                f"发现违规行: {line.strip()!r}",
                            )
    except (zipfile.BadZipFile, FileNotFoundError) as exc:
        _dbg("validate_no_torch_in_metadata", f"无法读取 wheel: {exc}")
        raise ValueError(
            f"[Walpurgis:validate_no_torch_in_metadata] 无法读取 wheel {wheel_path!r}: {exc}"
        ) from exc

    if not violations:
        _dbg("validate_no_torch_in_metadata", "通过检查: metadata 中无 torch 依赖")
    return violations


# ---------------------------------------------------------------------------
# 模块级信息导出
# ---------------------------------------------------------------------------

#: 上游迁移信息，供审计工具查询
UPSTREAM_COMMIT = "330b135"
UPSTREAM_REPO = "rapidsai/cugraph-gnn"
UPSTREAM_PR = 425
UPSTREAM_TITLE = "ensure 'torch' CUDA wheels are installed in CI, test that 'torch' is an optional dependency"
MIGRATION_TARGET = "walpurgis/core/torch_cuda_wheel_policy.py"

#: 330b135 新增的 CI 脚本（记录用）
NEW_CI_SCRIPTS: List[str] = [
    "ci/download-torch-wheels.sh",
    "ci/uninstall-torch-wheels.sh",
    "ci/validate_wheel.sh",
]

#: ruff banned-api：330b135 在 pyproject.toml 中新增的禁止规则
BANNED_IMPORTS: List[str] = [
    "torch",  # 库代码中禁止裸 import torch，必须使用 import_optional()
]
