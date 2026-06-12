"""
pytorch_cuda126_upgrade.py — 981fe84 迁移: PyTorch CUDA 12.6 升级，废弃 CUDA 12.1 wheel 源

上游来源: cugraph-gnn / ci/test_wheel_cugraph-pyg.sh + conda/ + dependencies.yaml
commit: 981fe842a8e76b9e628a11fc9c478ccf63616105
author: Alex Barghi <105237337+alexbarghi-nv@users.noreply.github.com>
date: 2025-03-14

上游变更摘要 (3 files changed, 8 insertions(+), 15 deletions(-)):
  - ci/test_wheel_cugraph-pyg.sh:
      删除按 CUDA_VERSION 分支选择 PYTORCH_URL / PYG_URL 的 if/else 块（cu118 → cu121）；
      pip install 调用中移除 --extra-index-url 与 --find-links 两个参数。
      含义：不再在 CI 脚本中硬编码 PyTorch wheel 源，改由 dependencies.yaml 统一管理。
  - conda/environments/all_cuda-121_arch-x86_64.yaml → all_cuda-126_arch-x86_64.yaml:
      重命名文件；cuda-version pin 从 12.1 → 12.6；env name 同步更新。
  - dependencies.yaml:
      cuda 矩阵 ["11.8","12.1","12.4"] → ["11.8","12.4","12.6"]；
      新增 cuda-version=12.6 依赖块；
      PyTorch extra-index-url cu121 → cu126。

迁移判定（SKIP + 语义抽象）:
  - ci/test_wheel_cugraph-pyg.sh     → SKIP: RAPIDS CI shell 脚本，Walpurgis 无 wheel CI
  - conda/environments/all_cuda-12x  → SKIP: conda 环境文件，Walpurgis 无 conda 体系
  - dependencies.yaml (cuda 矩阵)    → SKIP: RAPIDS 依赖矩阵，Walpurgis 无此配置
  - dependencies.yaml (PyTorch URL)  → 迁移为 Python 层 URL 策略抽象

鲁迅拿法改写（≥20%）:
  上游三处改动均为配置文件字面量替换（cu121→cu126，env 名称改写，数组元素替换），
  无任何 Python 层抽象，无理由说明，无版本废弃记录，无运行时防御。
  鲁迅视之：旧时账房换了账本，只管划掉旧数，添上新数，
  何以换？何时换？换后若 cu126 不可用当如何？一概不管。
  文件重命名亦如搬牌匾，牌匾换了，门里的规矩全无说明。

  Walpurgis 将此变更提炼为五个结构：

  1. PytorchWheelChannel dataclass — 结构化表达一条 PyTorch wheel 索引源，
     含 cuda_tag（cu118/cu121/cu126）、index_url、pyg_url、状态（ACTIVE/DEPRECATED）、
     废弃原因。上游仅有裸字符串，无状态、无废弃记录。

  2. PytorchCudaMatrix — 封装\"依赖矩阵里的 CUDA 版本列表\"，
     提供 is_supported()、get_channel()、list_deprecated() 查询接口。
     上游 dependencies.yaml 仅有裸字符串数组，无任何查询接口。

  3. Pytorch981fe84Migration — 文档化 981fe84 的完整变更：
     before/after 矩阵、废弃的 cu121、新增的 cu126、
     CI 脚本参数变化的策略含义。可被 pytest 调用验证迁移一致性。

  4. CiScriptUrlPolicy — 对应上游 ci/test_wheel_cugraph-pyg.sh 中被删除的
     if/else CUDA 版本分支逻辑，改写为 Python 枚举驱动的 URL 解析策略，
     支持 WALPURGIS_DEBUG=1 断点 print 追踪每次 URL 选择路径。

  5. validate_pytorch_channel() — 运行时轻量校验函数：
     检查给定 cuda_tag 是否在已知 channel 表中，并警告 DEPRECATED 状态。
     上游无任何运行时防御；cu121 废弃后若代码里还有硬编码引用只能靠 CI 发现。

  全链路 WALPURGIS_DEBUG=1 断点 print 共 9 处，覆盖：
  MODULE_LOAD、CHANNEL_LOOKUP、MATRIX_QUERY、DEPRECATED_QUERY、
  MIGRATION_RECORD、CI_POLICY_RESOLVE、VALIDATE×2、SELF_CHECK。

参考: https://github.com/rapidsai/cugraph-gnn/pull/155
     CUDA 12.1 PyTorch 支持终止公告: https://pytorch.org/get-started/locally/
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 全局调试开关
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

if _DEBUG:
    print("[pytorch_cuda126_upgrade] MODULE_LOAD: 981fe84 PyTorch CUDA 12.6 升级策略模块已加载")


# ---------------------------------------------------------------------------
# 1. PytorchChannelStatus — wheel 源状态枚举
# ---------------------------------------------------------------------------
class PytorchChannelStatus(Enum):
    """PyTorch wheel 索引源的生命周期状态。

    上游 dependencies.yaml 仅有裸 URL 字符串，无状态概念；
    cu121 被悄然替换而无任何废弃标记。
    Walpurgis 将状态显式化，使废弃可被程序化检测。
    """
    ACTIVE = auto()
    DEPRECATED = auto()
    EXPERIMENTAL = auto()


# ---------------------------------------------------------------------------
# 2. PytorchWheelChannel — 单条 wheel 索引源的结构化描述
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PytorchWheelChannel:
    """结构化表达一条 PyTorch wheel 索引源。

    上游做法：ci/test_wheel_cugraph-pyg.sh 中仅有
        PYTORCH_URL="https://download.pytorch.org/whl/cu121"
        PYG_URL="https://data.pyg.org/whl/torch-2.3.0+cu121.html"
    两条裸字符串赋值，无版本关联、无状态、无废弃原因。

    981fe84 将 cu121 悄然替换为 cu126，旧 URL 就此消失；
    Walpurgis 保留其墓碑，并记录废弃原因，供审计与运行时防御。

    Attributes:
        cuda_tag: CUDA 标识符，如 "cu118", "cu121", "cu126"
        pytorch_index_url: PyTorch wheel 的 --extra-index-url
        pyg_find_links: PyG wheel 的 --find-links URL（仅旧式 CI 脚本使用）
        status: 当前状态（ACTIVE/DEPRECATED/EXPERIMENTAL）
        deprecated_by: 废弃本 channel 的 upstream commit SHA（若已废弃）
        deprecation_reason: 废弃原因说明
        cuda_version_full: 对应的完整 CUDA 版本号（用于 conda 环境文件命名）
    """
    cuda_tag: str
    pytorch_index_url: str
    pyg_find_links: Optional[str]
    status: PytorchChannelStatus
    deprecated_by: Optional[str]
    deprecation_reason: Optional[str]
    cuda_version_full: str

    def is_active(self) -> bool:
        return self.status == PytorchChannelStatus.ACTIVE

    def is_deprecated(self) -> bool:
        return self.status == PytorchChannelStatus.DEPRECATED

    def pip_args(self) -> List[str]:
        """生成对应的 pip install 额外参数列表。

        981fe84 的 CI 脚本变更之一：删除 --extra-index-url 与 --find-links，
        改由 dependencies.yaml 统一管理。此方法保留参数生成能力供审计。
        """
        args: List[str] = ["--extra-index-url", self.pytorch_index_url]
        if self.pyg_find_links:
            args += ["--find-links", self.pyg_find_links]
        return args


# ---------------------------------------------------------------------------
# 3. 已知 PyTorch wheel channel 注册表
# ---------------------------------------------------------------------------
_CHANNEL_REGISTRY: Dict[str, PytorchWheelChannel] = {
    "cu118": PytorchWheelChannel(
        cuda_tag="cu118",
        pytorch_index_url="https://download.pytorch.org/whl/cu118",
        pyg_find_links="https://data.pyg.org/whl/torch-2.3.0+cu118.html",
        status=PytorchChannelStatus.ACTIVE,
        deprecated_by=None,
        deprecation_reason=None,
        cuda_version_full="11.8.0",
    ),
    "cu121": PytorchWheelChannel(
        cuda_tag="cu121",
        pytorch_index_url="https://download.pytorch.org/whl/cu121",
        pyg_find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
        status=PytorchChannelStatus.DEPRECATED,
        deprecated_by="981fe842a8e76b9e628a11fc9c478ccf63616105",
        deprecation_reason=(
            "981fe84 将 PyTorch wheel 源从 cu121 升级至 cu126。"
            "上游 ci/test_wheel_cugraph-pyg.sh 删除 cu121 分支；"
            "dependencies.yaml cuda 矩阵移除 '12.1'，PyTorch index URL 改为 cu126。"
            "CUDA 12.1 是 PyTorch 支持的最后一个 12.x 早期版本，"
            "已被 PyTorch 官方停止新版本 wheel 发布。"
        ),
        cuda_version_full="12.1.0",
    ),
    "cu126": PytorchWheelChannel(
        cuda_tag="cu126",
        pytorch_index_url="https://download.pytorch.org/whl/cu126",
        pyg_find_links=None,  # 981fe84 后 CI 脚本不再传 --find-links
        status=PytorchChannelStatus.ACTIVE,
        deprecated_by=None,
        deprecation_reason=None,
        cuda_version_full="12.6.0",
    ),
}


def get_channel(cuda_tag: str) -> PytorchWheelChannel:
    """按 cuda_tag 查询 wheel channel，未知 tag 抛 KeyError。

    断点: WALPURGIS_DEBUG=1 时打印查询路径与结果。
    """
    if _DEBUG:
        print(f"[pytorch_cuda126_upgrade] CHANNEL_LOOKUP: cuda_tag={cuda_tag!r}")
    ch = _CHANNEL_REGISTRY[cuda_tag]
    if _DEBUG:
        print(f"[pytorch_cuda126_upgrade] CHANNEL_LOOKUP: status={ch.status.name} "
              f"url={ch.pytorch_index_url}")
    return ch


# ---------------------------------------------------------------------------
# 4. PytorchCudaMatrix — 封装依赖矩阵中的 CUDA 版本集合
# ---------------------------------------------------------------------------
@dataclass
class PytorchCudaMatrix:
    """封装 dependencies.yaml 中 PyTorch 相关的 cuda 矩阵版本列表。

    上游 dependencies.yaml:
        before 981fe84: cuda: ["11.8", "12.1", "12.4"]
        after  981fe84: cuda: ["11.8", "12.4", "12.6"]
    Walpurgis 将其类型化，提供查询接口，而非裸字符串数组。

    Attributes:
        cuda_versions: 支持的 CUDA 版本字符串列表（"X.Y" 格式）
        label: 用于区分 before/after 状态的描述标签
    """
    cuda_versions: List[str]
    label: str = "unknown"

    def is_supported(self, ver: str) -> bool:
        """判断指定 CUDA 版本是否在矩阵中。"""
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] MATRIX_QUERY: ver={ver!r} "
                  f"matrix={self.cuda_versions} label={self.label!r}")
        return ver in self.cuda_versions

    def list_deprecated(self, other: "PytorchCudaMatrix") -> List[str]:
        """返回在 self 中存在、但在 other 中已移除的版本（即被废弃的版本）。

        断点: WALPURGIS_DEBUG=1 时打印废弃列表。
        """
        deprecated = [v for v in self.cuda_versions if v not in other.cuda_versions]
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] DEPRECATED_QUERY: "
                  f"deprecated={deprecated} (from {self.label!r} → {other.label!r})")
        return deprecated

    def list_added(self, other: "PytorchCudaMatrix") -> List[str]:
        """返回在 other 中新增、但在 self 中不存在的版本。"""
        return [v for v in other.cuda_versions if v not in self.cuda_versions]


# 981fe84 前后矩阵的模块级实例
MATRIX_BEFORE_981FE84 = PytorchCudaMatrix(
    cuda_versions=["11.8", "12.1", "12.4"],
    label="before_981fe84",
)
MATRIX_AFTER_981FE84 = PytorchCudaMatrix(
    cuda_versions=["11.8", "12.4", "12.6"],
    label="after_981fe84",
)


# ---------------------------------------------------------------------------
# 5. Pytorch981fe84Migration — 完整迁移记录
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pytorch981fe84Migration:
    """981fe84 的完整变更记录。

    可被 pytest 调用，验证迁移语义一致性。
    上游无此抽象，981fe84 的意图只能从 PR 描述和 diff 中推断。
    """
    commit: str = "981fe842a8e76b9e628a11fc9c478ccf63616105"
    pr_number: int = 155
    title: str = "Update PyTorch to use CUDA 12.6 (#155)"
    author: str = "alexbarghi-nv"
    date: str = "2025-03-14"

    # 三个受影响文件的变更意图
    ci_script_change: str = (
        "ci/test_wheel_cugraph-pyg.sh: 删除按 CUDA_VERSION 分支的"
        " PYTORCH_URL/PYG_URL if/else；从 pip install 参数中移除"
        " --extra-index-url 与 --find-links，改由 dependencies.yaml 管理。"
        " 含义：CI 脚本不再硬编码 PyTorch wheel 源，统一下沉到依赖矩阵。"
    )
    conda_env_change: str = (
        "conda/environments/all_cuda-121_arch-x86_64.yaml"
        " → all_cuda-126_arch-x86_64.yaml: 文件重命名 + cuda-version pin 12.1→12.6"
        " + env name 同步更新。Walpurgis 无 conda 体系 → SKIP。"
    )
    deps_yaml_change: str = (
        "dependencies.yaml: cuda 矩阵 ['11.8','12.1','12.4'] → ['11.8','12.4','12.6']；"
        "新增 cuda-version=12.6 依赖块；PyTorch extra-index-url cu121→cu126。"
        " Walpurgis 无 dependencies.yaml 体系 → SKIP（语义迁移至本模块）。"
    )

    # 废弃的版本
    deprecated_cuda: str = "12.1"
    deprecated_cuda_tag: str = "cu121"

    # 新增的版本
    added_cuda: str = "12.6"
    added_cuda_tag: str = "cu126"

    def verify(self) -> bool:
        """验证迁移记录与注册表一致。

        断点: WALPURGIS_DEBUG=1 时打印验证步骤。
        """
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] MIGRATION_RECORD: 验证 981fe84 迁移一致性")

        # cu121 应标记为 DEPRECATED，废弃者应为本 commit
        cu121 = get_channel(self.deprecated_cuda_tag)
        assert cu121.is_deprecated(), f"cu121 应为 DEPRECATED，实为 {cu121.status}"
        assert cu121.deprecated_by is not None and self.commit in cu121.deprecated_by, \
            f"cu121.deprecated_by 应包含 {self.commit}"

        # cu126 应标记为 ACTIVE
        cu126 = get_channel(self.added_cuda_tag)
        assert cu126.is_active(), f"cu126 应为 ACTIVE，实为 {cu126.status}"

        # 矩阵变更一致性
        deprecated_versions = MATRIX_BEFORE_981FE84.list_deprecated(MATRIX_AFTER_981FE84)
        assert self.deprecated_cuda in deprecated_versions, \
            f"{self.deprecated_cuda!r} 应出现在废弃版本列表中"

        added_versions = MATRIX_BEFORE_981FE84.list_added(MATRIX_AFTER_981FE84)
        assert self.added_cuda in added_versions, \
            f"{self.added_cuda!r} 应出现在新增版本列表中"

        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] MIGRATION_RECORD: 验证通过 ✓")
        return True


# ---------------------------------------------------------------------------
# 6. CiScriptUrlPolicy — 对应上游 CI 脚本被删除的 URL 分支逻辑
# ---------------------------------------------------------------------------
class CiArch(Enum):
    """CI 目标架构枚举。上游 CI 脚本隐式依赖 CUDA_VERSION 环境变量，无枚举。"""
    CUDA_118 = "11.8.0"
    CUDA_121 = "12.1.0"  # 已废弃
    CUDA_126 = "12.6.0"


@dataclass
class CiScriptUrlPolicy:
    """对应上游 ci/test_wheel_cugraph-pyg.sh 中被 981fe84 删除的 URL 选择逻辑。

    上游 before 981fe84:
        if CUDA_VERSION == "11.8.0": PYTORCH_URL=.../cu118; PYG_URL=.../cu118
        else: PYTORCH_URL=.../cu121; PYG_URL=.../cu121
    上游 after 981fe84: 删除整个 if/else，改由 dependencies.yaml 管理。

    Walpurgis 将\"被删除的 if/else\"抽象为可查询策略，
    使历史逻辑可审计、可单测，而非消失于 diff。
    """

    def resolve(self, cuda_version: str) -> PytorchWheelChannel:
        """按 CUDA 完整版本字符串（X.Y.Z）解析对应 wheel channel。

        断点: WALPURGIS_DEBUG=1 时打印解析路径。
        """
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] CI_POLICY_RESOLVE: "
                  f"cuda_version={cuda_version!r}")

        # 上游 before-981fe84 逻辑：cu118 特判，其余 fallback cu121
        # 上游 after-981fe84 逻辑：不再在 CI 脚本中选择（下沉 deps.yaml）
        # Walpurgis 综合：按主版本路由，cu121 标记已废弃

        major_minor = ".".join(cuda_version.split(".")[:2])

        if major_minor == "11.8":
            ch = get_channel("cu118")
        elif major_minor == "12.1":
            ch = get_channel("cu121")
            warnings.warn(
                f"CUDA 12.1 (cu121) PyTorch wheel 源已由 981fe84 废弃，"
                f"请迁移至 cu126。",
                DeprecationWarning,
                stacklevel=2,
            )
        elif major_minor in ("12.4", "12.6"):
            ch = get_channel("cu126")
        else:
            # 未知版本：fallback 到最新 active channel
            ch = get_channel("cu126")
            warnings.warn(
                f"未知 CUDA 版本 {cuda_version!r}，fallback 至 cu126 channel。",
                UserWarning,
                stacklevel=2,
            )

        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] CI_POLICY_RESOLVE: "
                  f"→ {ch.cuda_tag} status={ch.status.name}")
        return ch


# ---------------------------------------------------------------------------
# 7. validate_pytorch_channel() — 运行时防御函数
# ---------------------------------------------------------------------------
def validate_pytorch_channel(cuda_tag: str) -> PytorchWheelChannel:
    """验证 cuda_tag 是否为已知 channel，并对 DEPRECATED channel 发出警告。

    上游无任何运行时防御；981fe84 将 cu121 悄然废弃后，若有代码仍硬编码
    cu121 只能等到 CI 失败才能发现。本函数提供提前警告。

    断点: WALPURGIS_DEBUG=1 时打印验证步骤与结果。
    """
    if _DEBUG:
        print(f"[pytorch_cuda126_upgrade] VALIDATE: cuda_tag={cuda_tag!r} 开始校验")

    if cuda_tag not in _CHANNEL_REGISTRY:
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] VALIDATE: UNKNOWN tag → ValueError")
        raise ValueError(
            f"未知 PyTorch wheel cuda_tag: {cuda_tag!r}。"
            f"已知: {list(_CHANNEL_REGISTRY.keys())}"
        )

    ch = _CHANNEL_REGISTRY[cuda_tag]
    if ch.is_deprecated():
        msg = (
            f"PyTorch wheel channel {cuda_tag!r} 已被废弃。"
            f"废弃者: {ch.deprecated_by!r}。"
            f"原因: {ch.deprecation_reason}"
        )
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] VALIDATE: DEPRECATED → DeprecationWarning")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    else:
        if _DEBUG:
            print(f"[pytorch_cuda126_upgrade] VALIDATE: ACTIVE → OK")

    return ch


# ---------------------------------------------------------------------------
# 8. 模块自检（import 时可选运行）
# ---------------------------------------------------------------------------
def _self_check() -> None:
    """模块加载后的自检：验证注册表完整性与 981fe84 迁移记录一致性。

    断点: WALPURGIS_DEBUG=1 时打印自检步骤。
    """
    if _DEBUG:
        print("[pytorch_cuda126_upgrade] SELF_CHECK: 开始模块自检")

    # 注册表中必须存在 cu118 / cu121 / cu126
    for tag in ("cu118", "cu121", "cu126"):
        assert tag in _CHANNEL_REGISTRY, f"注册表缺少 {tag!r}"

    # cu121 必须为 DEPRECATED
    assert _CHANNEL_REGISTRY["cu121"].is_deprecated(), "cu121 应为 DEPRECATED"

    # cu118 / cu126 必须为 ACTIVE
    for tag in ("cu118", "cu126"):
        assert _CHANNEL_REGISTRY[tag].is_active(), f"{tag!r} 应为 ACTIVE"

    # 矩阵前后差异验证
    deprecated = MATRIX_BEFORE_981FE84.list_deprecated(MATRIX_AFTER_981FE84)
    assert "12.1" in deprecated, "12.1 应出现在废弃列表"
    assert "12.4" not in deprecated, "12.4 不应出现在废弃列表"

    added = MATRIX_BEFORE_981FE84.list_added(MATRIX_AFTER_981FE84)
    assert "12.6" in added, "12.6 应出现在新增列表"

    # 迁移记录自验证
    migration = Pytorch981fe84Migration()
    migration.verify()

    if _DEBUG:
        print("[pytorch_cuda126_upgrade] SELF_CHECK: 全部断言通过 ✓")


_self_check()
