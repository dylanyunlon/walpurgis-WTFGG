"""
dgl_package_policy.py — migrate 5f12b2f: Update DGL Packages to PyTorch 2.4 (PR #148)

上游历史:
  - 5f12b2f: Update DGL Packages to PyTorch 2.4 (#148)
      * Author: Alex Barghi (alexbarghi-nv, NVIDIA)
      * Date: 2025-02-19
      * Context: DGL 2.4 packages for PyTorch 2.3 (th23) 已不再提供；
        将所有 conda 渠道引用从 th23_cu* 切换到 th24_cu*，
        解除 CI 阻塞。
      * 受影响文件 (6):
          conda/environments/all_cuda-118_arch-x86_64.yaml  th23_cu118 → th24_cu118
          conda/environments/all_cuda-121_arch-x86_64.yaml  th23_cu121 → th24_cu124
          conda/environments/all_cuda-124_arch-x86_64.yaml  th23_cu121 → th24_cu124
          conda/recipes/cugraph-dgl/meta.yaml               th23.cu*   → th24.cu*
          dependencies.yaml                                 th23_cu121/cu118 → th24_cu124/cu118
          python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml  th23_cu118 → th24_cu118

  - 与 fb8296e (DGL 永久删除) 关系:
      5f12b2f 升级 DGL 包到 PyTorch 2.4 版渠道，属于 DGL 仍在维护期的维护动作；
      fb8296e 在后续将整个 cugraph-dgl 包删除。
      本文件建立 PyTorchLabel 类型体系，完整记录 th23→th24 跃迁语义，
      并通过 DGL_REMOVED_IN_FB8296E 标志与 dgl_deprecation.py 串联。

Walpurgis 迁移语义（全 SKIP，新建此文件记录策略）:
  - 上游 6 个文件均为 conda 构建环境配置，Walpurgis 无 conda 构建矩阵
  - 迁移位置: src/walpurgis/core/dgl_package_policy.py — 新建

20% 改写（鲁迅拿法）:
  1. PyTorchLabel dataclass: 将 th23/th24 渠道标签强类型化，
     携带 pytorch_version/cuda_variants/eol 字段，
     上游只有裸字符串替换，无任何类型化表示
  2. DglChannelSpec dataclass: 封装 conda channel::package 完整规格，
     parse() / as_conda_dep() / as_meta_yaml_pin() 派生格式，
     上游无此抽象
  3. DglPackageMigration dataclass: 封装 FROM→TO 跃迁事实，
     携带 commit_sha/pr_number/author/rationale，
     migration_kind() 区分 LABEL_ONLY/VERSION_RELAXED/VERSION_PINNED，
     上游只有 sed 替换
  4. CudaChannelMatrix: 将 cuda→channel 映射表类型化，
     lookup() / affected_envs() / validate_all_covered() 方法，
     上游散落在多个 yaml 文件中
  5. DglEolProbe: 在 Python import 层探测 DGL 包是否已 EOL（th23 不可用），
     上游通过 CI 失败才发现，无主动探测机制
  6. DglPackageAudit: 扫描文件检查 th23 旧渠道标签残留，
     assert_clean() 供 CI 调用，上游无此回头检查机制
  7. 全链路 WALPURGIS_DEBUG=1 断点（6 处）:
     覆盖 label 解析→跃迁计算→矩阵查找→EOL 探测→残留扫描→自测各阶段

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    """WALPURGIS_DEBUG=1 时打印调试信息。"""
    if _DEBUG:
        print(f"[WALPURGIS dgl_package_policy] {msg}", file=sys.stderr, flush=True)


# ── 断点1: PyTorchLabel —— DGL conda 渠道标签类型化 ──────────────────────────

@dataclass(frozen=True, order=True)
class PyTorchLabel:
    """
    DGL conda 渠道中的 PyTorch 版本标签（如 th23、th24）。

    上游 5f12b2f 将所有 `dglteam/label/th23_cu*` 替换为 `dglteam/label/th24_cu*`；
    上游只有裸字符串，无版本对象。

    Walpurgis 改写:
    - tag: 渠道标签（"th23" / "th24"）
    - pytorch_version: 对应 PyTorch 版本字符串（"2.3" / "2.4"）
    - eol: 是否已停止维护（5f12b2f 原因正是 th23 包不再可用）
    - cuda_variants: 支持的 CUDA 变体集合

    断点1: parse() 调用时。
    """
    tag: str                    # e.g. "th23", "th24"
    pytorch_version: str        # e.g. "2.3", "2.4"
    eol: bool = False           # True → 对应包已不再提供，5f12b2f 中 th23 触发此标志
    cuda_variants: tuple = ()   # e.g. ("cu118", "cu121", "cu124")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"th\d+", self.tag):
            raise ValueError(
                f"PyTorchLabel.tag 必须匹配 th\\d+，实际: {self.tag!r}"
            )
        if not re.fullmatch(r"\d+\.\d+", self.pytorch_version):
            raise ValueError(
                f"PyTorchLabel.pytorch_version 必须为 X.Y 格式，实际: {self.pytorch_version!r}"
            )
        _dbg(
            f"PyTorchLabel.__post_init__: tag={self.tag!r} "
            f"pytorch={self.pytorch_version!r} eol={self.eol} "
            f"cuda_variants={self.cuda_variants}"
        )

    @classmethod
    def parse(cls, label_str: str) -> "PyTorchLabel":
        """
        从渠道标签字符串解析 PyTorchLabel。

        示例:
            "th23" → PyTorchLabel(tag="th23", pytorch_version="2.3", eol=True, ...)
            "th24" → PyTorchLabel(tag="th24", pytorch_version="2.4", eol=False, ...)

        断点1: 此处。
        """
        _dbg(f"PyTorchLabel.parse(label_str={label_str!r})")
        label_str = label_str.strip().lower()
        return _KNOWN_PYTORCH_LABELS.get(label_str) or cls._parse_unknown(label_str)

    @classmethod
    def _parse_unknown(cls, label_str: str) -> "PyTorchLabel":
        m = re.fullmatch(r"th(\d)(\d+)", label_str)
        if not m:
            raise ValueError(f"无法解析 PyTorchLabel: {label_str!r}")
        major, minor = m.group(1), m.group(2)
        return cls(
            tag=label_str,
            pytorch_version=f"{major}.{minor}",
            eol=False,
            cuda_variants=(),
        )

    @property
    def major_minor(self) -> tuple:
        """返回 (major, minor) int 元组，便于比较。"""
        a, b = self.pytorch_version.split(".")
        return (int(a), int(b))

    def is_newer_than(self, other: "PyTorchLabel") -> bool:
        return self.major_minor > other.major_minor

    def conda_label_prefix(self, cuda_variant: str) -> str:
        """生成 conda 渠道前缀，如 'dglteam/label/th24_cu124'。"""
        return f"dglteam/label/{self.tag}_{cuda_variant}"

    def __repr__(self) -> str:
        eol_marker = " [EOL]" if self.eol else ""
        return (
            f"PyTorchLabel(tag={self.tag!r}, pytorch={self.pytorch_version!r}"
            f"{eol_marker}, cuda={self.cuda_variants})"
        )


# 已知标签注册表（对应 5f12b2f 涉及的两个版本）
_TH23 = PyTorchLabel(
    tag="th23",
    pytorch_version="2.3",
    eol=True,   # 5f12b2f 注释: "DGL 2.4 packages for PyTorch 2.3 are no longer available"
    cuda_variants=("cu118", "cu121"),
)
_TH24 = PyTorchLabel(
    tag="th24",
    pytorch_version="2.4",
    eol=False,
    cuda_variants=("cu118", "cu124"),
)
_KNOWN_PYTORCH_LABELS: Dict[str, PyTorchLabel] = {
    "th23": _TH23,
    "th24": _TH24,
}


# ── DglChannelSpec —— conda channel::package 完整规格 ────────────────────────

@dataclass(frozen=True)
class DglChannelSpec:
    """
    封装一条 DGL conda 依赖规格。

    上游 5f12b2f 直接对字符串做替换，如:
        dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*
        →
        dglteam/label/th24_cu118::dgl

    Walpurgis 改写：将其解构为结构化字段，可派生多种格式。
    """
    label: PyTorchLabel         # PyTorch 标签
    cuda_variant: str           # e.g. "cu118", "cu124"
    package_name: str = "dgl"
    version_constraint: Optional[str] = None  # e.g. ">=2.4.0.th23.cu*", None = 无约束

    def __post_init__(self) -> None:
        if not re.fullmatch(r"cu\d{3}", self.cuda_variant):
            raise ValueError(
                f"DglChannelSpec.cuda_variant 必须匹配 cu\\d{{3}}，实际: {self.cuda_variant!r}"
            )

    @classmethod
    def parse(cls, dep_str: str) -> "DglChannelSpec":
        """
        从 conda 依赖字符串解析 DglChannelSpec。

        支持格式:
            "dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*"
            "dglteam/label/th24_cu118::dgl"
        """
        _dbg(f"DglChannelSpec.parse(dep_str={dep_str!r})")
        m = re.match(
            r"dglteam/label/(th\d+)_(cu\d+)::([\w\-]+)(.*)",
            dep_str.strip(),
        )
        if not m:
            raise ValueError(f"无法解析 DglChannelSpec: {dep_str!r}")
        tag, cuda, pkg, constraint = m.group(1), m.group(2), m.group(3), m.group(4)
        label = PyTorchLabel.parse(tag)
        return cls(
            label=label,
            cuda_variant=cuda,
            package_name=pkg,
            version_constraint=constraint.strip() or None,
        )

    def as_conda_dep(self) -> str:
        """生成 conda yaml dependencies 行。"""
        channel = self.label.conda_label_prefix(self.cuda_variant)
        base = f"{channel}::{self.package_name}"
        return base + (self.version_constraint or "")

    def as_meta_yaml_pin(self) -> str:
        """生成 conda recipe meta.yaml run dependencies 行。"""
        # meta.yaml 格式: - dgl >=2.4.0.th24.cu*
        constraint = self.version_constraint or ""
        return f"- {self.package_name} {constraint}".rstrip()

    def __repr__(self) -> str:
        return (
            f"DglChannelSpec("
            f"label={self.label.tag!r}, cuda={self.cuda_variant!r}, "
            f"pkg={self.package_name!r}, constraint={self.version_constraint!r})"
        )


# ── 断点2: DglPackageMigration —— 跃迁事实封装 ───────────────────────────────

class MigrationKind(Enum):
    """DGL 包迁移类型。"""
    LABEL_ONLY = auto()         # 仅渠道标签变更（th23→th24），包名和版本约束结构不变
    VERSION_RELAXED = auto()    # 版本约束从有约束（>=...）放宽为无约束
    VERSION_PINNED = auto()     # 版本约束从无约束变为有约束（反向，罕见）
    CUDA_VARIANT_CHANGED = auto()  # CUDA 变体变更（cu121→cu124）


@dataclass(frozen=True)
class DglPackageMigration:
    """
    封装 5f12b2f 中单条 DGL 包渠道的 FROM→TO 跃迁。

    上游只有裸 sed；Walpurgis 添加:
    - commit_sha / pr_number / author / rationale
    - migration_kind() 分类
    - involved_files() 关联文件列表
    - as_patch_line() 生成可读的 diff 摘要

    断点2: migration_kind() 调用时。
    """
    from_spec: DglChannelSpec
    to_spec: DglChannelSpec
    commit_sha: str = "5f12b2fa2e6cb5942118d001cb09dac84efc6abc"
    pr_number: int = 148
    author: str = "Alex Barghi (alexbarghi-nv)"
    rationale: str = (
        "DGL 2.4 packages for PyTorch 2.3 (th23) are no longer available; "
        "upgrade to th24 channel to unblock CI."
    )
    involved_files: tuple = ()

    def migration_kind(self) -> Set[MigrationKind]:
        """
        判断此跃迁的类型集合（可同时包含多种）。

        断点2: 此处。
        """
        kinds: Set[MigrationKind] = set()

        _dbg(
            f"DglPackageMigration.migration_kind(): "
            f"from={self.from_spec!r} to={self.to_spec!r}"
        )

        if self.from_spec.label.tag != self.to_spec.label.tag:
            kinds.add(MigrationKind.LABEL_ONLY)

        if self.from_spec.cuda_variant != self.to_spec.cuda_variant:
            kinds.add(MigrationKind.CUDA_VARIANT_CHANGED)

        if self.from_spec.version_constraint and not self.to_spec.version_constraint:
            kinds.add(MigrationKind.VERSION_RELAXED)
        elif not self.from_spec.version_constraint and self.to_spec.version_constraint:
            kinds.add(MigrationKind.VERSION_PINNED)

        return kinds

    def is_forward(self) -> bool:
        """返回 True 表示向更新版本跃迁（th24 > th23）。"""
        return self.to_spec.label.is_newer_than(self.from_spec.label)

    def as_patch_line(self) -> str:
        """生成可读的单行 diff 摘要。"""
        return (
            f"- {self.from_spec.as_conda_dep()}\n"
            f"+ {self.to_spec.as_conda_dep()}"
        )

    def __repr__(self) -> str:
        kinds = "/".join(k.name for k in self.migration_kind())
        return (
            f"DglPackageMigration("
            f"from={self.from_spec.label.tag}_{self.from_spec.cuda_variant} "
            f"to={self.to_spec.label.tag}_{self.to_spec.cuda_variant} "
            f"kind={kinds!r})"
        )


# ── 5f12b2f 全部跃迁实例化 ────────────────────────────────────────────────────
# 6 个文件 × 对应渠道变更，完整覆盖上游 diff

_5F12B2F_MIGRATIONS: List[DglPackageMigration] = [
    # conda/environments/all_cuda-118_arch-x86_64.yaml
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu118"),
        involved_files=("conda/environments/all_cuda-118_arch-x86_64.yaml",),
    ),
    # conda/environments/all_cuda-121_arch-x86_64.yaml  (cu121 → cu124)
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu121::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu124"),
        involved_files=("conda/environments/all_cuda-121_arch-x86_64.yaml",),
    ),
    # conda/environments/all_cuda-124_arch-x86_64.yaml  (cu121 → cu124)
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu121::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu124"),
        involved_files=("conda/environments/all_cuda-124_arch-x86_64.yaml",),
    ),
    # conda/recipes/cugraph-dgl/meta.yaml  (版本约束 th23 → th24)
    DglPackageMigration(
        from_spec=DglChannelSpec(
            label=_TH23, cuda_variant="cu118",
            package_name="dgl", version_constraint=">=2.4.0.th23.cu*",
        ),
        to_spec=DglChannelSpec(
            label=_TH24, cuda_variant="cu118",
            package_name="dgl", version_constraint=">=2.4.0.th24.cu*",
        ),
        involved_files=("conda/recipes/cugraph-dgl/meta.yaml",),
    ),
    # dependencies.yaml  cuda 12.* 行
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu121::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu124"),
        involved_files=("dependencies.yaml (cuda 12.* matrix)",),
    ),
    # dependencies.yaml  cuda 11.* 行
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu118"),
        involved_files=("dependencies.yaml (cuda 11.* matrix)",),
    ),
    # dependencies.yaml  fallback 行
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu121::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu124"),
        involved_files=("dependencies.yaml (matrix: null fallback)",),
    ),
    # python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml
    DglPackageMigration(
        from_spec=DglChannelSpec.parse("dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*"),
        to_spec=DglChannelSpec(label=_TH24, cuda_variant="cu118"),
        involved_files=("python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml",),
    ),
]


# ── 断点3: CudaChannelMatrix —— cuda → channel 映射表类型化 ─────────────────

@dataclass
class CudaChannelMatrix:
    """
    将 5f12b2f 后的 cuda→DGL channel 映射表类型化。

    上游 dependencies.yaml matrices 段:
        - matrix: {cuda: "12.*"}  →  dglteam/label/th24_cu124::dgl
        - matrix: {cuda: "11.*"}  →  dglteam/label/th24_cu118::dgl
        - {matrix: null}           →  dglteam/label/th24_cu124::dgl

    Walpurgis 改写: 结构化表示 + lookup / validate 方法。

    断点3: lookup() 调用时。
    """
    entries: Dict[str, DglChannelSpec] = field(default_factory=dict)

    def register(self, cuda_glob: str, spec: DglChannelSpec) -> None:
        """注册一个 cuda glob → DGL channel spec 映射。"""
        self.entries[cuda_glob] = spec
        _dbg(f"CudaChannelMatrix.register: cuda_glob={cuda_glob!r} spec={spec!r}")

    def lookup(self, cuda_glob: str) -> Optional[DglChannelSpec]:
        """
        查找 cuda glob 对应的 DGL channel spec。

        断点3: 此处。
        """
        result = self.entries.get(cuda_glob)
        _dbg(
            f"CudaChannelMatrix.lookup(cuda_glob={cuda_glob!r}) "
            f"→ {result!r}"
        )
        return result

    def affected_envs(self) -> List[str]:
        """返回所有已注册的 cuda glob 列表。"""
        return list(self.entries.keys())

    def validate_all_covered(self, required_globs: List[str]) -> None:
        """校验所有 required_globs 均已注册，否则 ValueError。"""
        missing = [g for g in required_globs if g not in self.entries]
        if missing:
            raise ValueError(
                f"CudaChannelMatrix: 以下 cuda glob 未注册: {missing}"
            )

    def __repr__(self) -> str:
        return f"CudaChannelMatrix(entries={list(self.entries.keys())})"


# 5f12b2f 后的矩阵（对应 dependencies.yaml 迁移后状态）
POST_5F12B2F_MATRIX = CudaChannelMatrix()
POST_5F12B2F_MATRIX.register(
    "12.*", DglChannelSpec(label=_TH24, cuda_variant="cu124")
)
POST_5F12B2F_MATRIX.register(
    "11.*", DglChannelSpec(label=_TH24, cuda_variant="cu118")
)
POST_5F12B2F_MATRIX.register(
    "null", DglChannelSpec(label=_TH24, cuda_variant="cu124")
)


# ── 断点4: DglEolProbe —— EOL 主动探测 ───────────────────────────────────────

class DglEolProbe:
    """
    主动探测 DGL 包是否已 EOL（上游 th23 渠道不再可用）。

    上游通过 CI 失败才发现 th23 包不可用；5f12b2f 的修复正是此原因。
    Walpurgis 改写：在 import 层主动检查已安装的 DGL 版本标签。

    断点4: probe() 调用时。
    """

    def probe(self) -> Optional[str]:
        """
        探测当前环境中 DGL 安装情况。

        返回已安装 DGL 版本字符串，若未安装返回 None。
        EOL 判断：版本字符串包含 "th23" → 视为 EOL 渠道包。

        断点4: 此处。
        """
        _dbg("DglEolProbe.probe(): 开始探测 DGL 安装状态")
        try:
            import importlib.metadata as _meta
            dgl_version = _meta.version("dgl")
            _dbg(f"DglEolProbe.probe(): dgl version={dgl_version!r}")

            if "th23" in dgl_version.lower():
                warnings.warn(
                    f"[Walpurgis:DglEolProbe] 检测到 th23 渠道 DGL 包 (version={dgl_version!r})。"
                    f"DGL 2.4 for PyTorch 2.3 已停止维护。"
                    f"请升级到 th24 渠道（dglteam/label/th24_cu*::dgl），"
                    f"参见上游 5f12b2f (PR #148)。",
                    DeprecationWarning,
                    stacklevel=2,
                )

            return dgl_version

        except Exception as exc:  # importlib.metadata.PackageNotFoundError 或其他
            _dbg(f"DglEolProbe.probe(): DGL 未安装或探测失败 ({type(exc).__name__}: {exc})")
            return None

    def is_eol_channel(self, dgl_version: str) -> bool:
        """判断给定版本字符串是否来自 th23（EOL）渠道。"""
        result = "th23" in dgl_version.lower()
        _dbg(
            f"DglEolProbe.is_eol_channel(version={dgl_version!r}) → {result}"
        )
        return result

    def __repr__(self) -> str:
        return "DglEolProbe()"


#: 模块级单例
DGL_EOL_PROBE = DglEolProbe()


# ── 断点5: DglPackageAudit —— 旧渠道标签残留扫描 ─────────────────────────────

class DglPackageAudit:
    """
    扫描文件中的 th23 旧渠道标签残留。

    上游通过全局 sed 替换后无回头检查；Walpurgis 添加此扫描机制。

    断点5: scan() / assert_clean() 调用时。
    """

    # 匹配 th23 渠道的正则
    _EOL_PATTERN = re.compile(r"dglteam/label/th23_cu\d+", re.IGNORECASE)

    def scan(self, text: str) -> List[str]:
        """
        扫描文本中所有 th23 渠道引用，返回匹配列表。

        断点5: 此处。
        """
        matches = self._EOL_PATTERN.findall(text)
        _dbg(
            f"DglPackageAudit.scan(): 找到 {len(matches)} 个 th23 残留: {matches}"
        )
        return matches

    def scan_file(self, filepath: str) -> List[str]:
        """扫描磁盘文件。"""
        _dbg(f"DglPackageAudit.scan_file(filepath={filepath!r})")
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return self.scan(f.read())
        except OSError as e:
            _dbg(f"DglPackageAudit.scan_file: 无法读取 {filepath!r}: {e}")
            return []

    def assert_clean(self, text: str, label: str = "") -> None:
        """
        断言文本中无 th23 残留，否则 AssertionError。
        供 CI 调用以验证迁移是否彻底。
        """
        matches = self.scan(text)
        if matches:
            loc = f" in {label!r}" if label else ""
            raise AssertionError(
                f"DglPackageAudit.assert_clean(): 发现 {len(matches)} 个 th23 渠道残留{loc}: "
                f"{matches}\n"
                f"请参见上游 5f12b2f (PR #148) 完成渠道升级。"
            )
        _dbg(f"DglPackageAudit.assert_clean({label!r}): CLEAN")

    def __repr__(self) -> str:
        return "DglPackageAudit()"


#: 模块级单例
DGL_PACKAGE_AUDIT = DglPackageAudit()


# ── 断点6: 自测 ───────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """
    模块加载时自测，WALPURGIS_DEBUG=1 时输出详情。

    断点6: 此处。
    """
    _dbg("_run_self_tests(): 开始自测")
    results = []

    def _check(name: str, cond: bool) -> None:
        status = "PASS" if cond else "FAIL"
        results.append((name, status))
        _dbg(f"  [{status}] {name}")

    # 1. PyTorchLabel 解析
    th23 = PyTorchLabel.parse("th23")
    th24 = PyTorchLabel.parse("th24")
    _check("th23 解析正确", th23.pytorch_version == "2.3" and th23.eol is True)
    _check("th24 解析正确", th24.pytorch_version == "2.4" and th24.eol is False)
    _check("th24 > th23", th24.is_newer_than(th23))

    # 2. DglChannelSpec 解析与格式化
    spec_from = DglChannelSpec.parse("dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*")
    spec_to = DglChannelSpec(label=_TH24, cuda_variant="cu118")
    _check("from_spec conda_dep 包含 th23_cu118", "th23_cu118" in spec_from.as_conda_dep())
    _check("to_spec conda_dep 包含 th24_cu118", "th24_cu118" in spec_to.as_conda_dep())

    # 3. 迁移跃迁类型
    m = _5F12B2F_MIGRATIONS[0]
    kinds = m.migration_kind()
    _check("第0条包含 LABEL_ONLY", MigrationKind.LABEL_ONLY in kinds)
    _check("第0条 is_forward=True", m.is_forward())
    # meta.yaml 条目同时有 LABEL_ONLY + VERSION_RELAXED（不含，只改 th23→th24 版本字符串）
    meta_m = _5F12B2F_MIGRATIONS[3]
    meta_kinds = meta_m.migration_kind()
    _check("meta.yaml 条目包含 LABEL_ONLY", MigrationKind.LABEL_ONLY in meta_kinds)

    # 4. CudaChannelMatrix lookup
    ch12 = POST_5F12B2F_MATRIX.lookup("12.*")
    ch11 = POST_5F12B2F_MATRIX.lookup("11.*")
    _check("cuda 12.* → cu124", ch12 is not None and ch12.cuda_variant == "cu124")
    _check("cuda 11.* → cu118", ch11 is not None and ch11.cuda_variant == "cu118")

    # 5. DglPackageAudit 残留扫描
    dirty_text = "dglteam/label/th23_cu118::dgl>=2.4.0.th23.cu*"
    clean_text = "dglteam/label/th24_cu118::dgl"
    _check("dirty_text 检出 th23", len(DGL_PACKAGE_AUDIT.scan(dirty_text)) > 0)
    _check("clean_text 无 th23", len(DGL_PACKAGE_AUDIT.scan(clean_text)) == 0)
    try:
        DGL_PACKAGE_AUDIT.assert_clean(dirty_text, "test_dirty")
        _check("assert_clean(dirty) 应抛 AssertionError", False)
    except AssertionError:
        _check("assert_clean(dirty) 正确抛 AssertionError", True)

    # 6. 迁移条目完整性
    _check("5f12b2f 迁移条目数 = 8", len(_5F12B2F_MIGRATIONS) == 8)
    all_forward = all(mg.is_forward() for mg in _5F12B2F_MIGRATIONS)
    _check("所有迁移均为 is_forward=True", all_forward)

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    _dbg(f"_run_self_tests(): {passed}/{total} 通过")
    if passed < total:
        failed = [name for name, s in results if s == "FAIL"]
        raise RuntimeError(
            f"dgl_package_policy 自测失败 ({total - passed}/{total}): {failed}"
        )


# 模块加载时执行自测（仅 DEBUG 模式下输出详情）
_run_self_tests()

_dbg(
    f"dgl_package_policy module loaded: "
    f"{len(_5F12B2F_MIGRATIONS)} migrations, "
    f"matrix_envs={POST_5F12B2F_MATRIX.affected_envs()}"
)


__all__ = [
    # 类型
    "PyTorchLabel",
    "DglChannelSpec",
    "MigrationKind",
    "DglPackageMigration",
    "CudaChannelMatrix",
    "DglEolProbe",
    "DglPackageAudit",
    # 已知标签常量
    "_TH23",
    "_TH24",
    # 数据
    "_5F12B2F_MIGRATIONS",
    "POST_5F12B2F_MATRIX",
    # 单例
    "DGL_EOL_PROBE",
    "DGL_PACKAGE_AUDIT",
]
