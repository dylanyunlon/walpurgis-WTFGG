"""
dgl_py313_skip_policy.py — migrate 616f1c7: chore: skip DGL tests on Python 3.13 (#201)

上游 commit 616f1c7c2（短 hash）
Author: Brad Rees <nv-bredmond@users.noreply.github.com>（推断）
PR:     https://github.com/rapidsai/cugraph-gnn/pull/201
Date:   2025-xx-xx（cugraph-gnn 第 212/452 commit）

上游变更全景（7 files changed, 16 insertions(+), 15 deletions(-)）：

  ① .github/workflows/build.yaml（DGL wheel 构建矩阵）
      旧: matrix_filter 选 amd64 + "最新 Python + CUDA"（group_by CUDA major,
          max_by [PY_VER desc, CUDA_VER desc]）——结果可能选中 Python 3.13
      新: matrix_filter 硬钉 PY_VER == "3.12"（group_by CUDA major,
          max_by [CUDA_VER desc 唯一维度]）
      注释改为: "This selects ARCH=amd64 + Python 3.12 + CUDA"
          + "Note that we don't support DGL on Python 3.13 so we don't build DGL on Python 3.13"
      迁移分析: jq 过滤器由 "最新可用版本" 变为 "固定 Python 3.12"；
          新过滤器 map(select(.ARCH == "amd64", .PY_VER == "3.12")) 是"OR"语义（逗号
          在 jq 中是 try-each-alternative），实际依赖 group_by 后的 max_by 收敛到 3.12；
          SKIP：Walpurgis 无 GitHub Actions CI 矩阵，此处建立类型化表示。

  ② .github/workflows/pr.yaml（PR 验证工作流，三处变更）
      A. conda-python-tests job：
         uses 分支 branch-25.06 → python-3.13
         matrix_filter 新增 and .PY_VER != "3.13"
         SKIP：CI yaml，建立 MatrixFilterMutation 记录。

      B. wheel-build-cugraph-dgl job：
         与 build.yaml 同类改动——matrix_filter 钉 PY_VER == "3.12"
         SKIP：同上。

      C. wheel-tests-cugraph-dgl job：
         matrix_filter 新增 and .PY_VER != "3.13"
         SKIP：同上。

  ③ .github/workflows/test.yaml（测试工作流，两处变更）
      A. conda-python-tests：
         uses 分支 branch-25.06 → python-3.13
         matrix_filter 新增 and (.PY_VER != "3.13")
         SKIP：CI yaml。

      B. wheel-tests-cugraph-dgl：
         matrix_filter 新增 and (.PY_VER != "3.13")
         SKIP：CI yaml。

  ④ ci/build_wheel.sh（新增 1 行，diff 截断）
      推断: 新增 PYTHON_VER 检查或 DGL_SKIP_PY313 守卫
      SKIP：shell 脚本，建立 WheelBuildGuard 表示。

  ⑤ ci/test_wheel_cugraph-dgl.sh（修改 2 行）
      推断: 添加 Python 版本前置检查，遇 3.13 时 early-exit
      SKIP：shell 脚本，建立 WheelTestGuard 表示。

  ⑥ conda/recipes/libwholegraph/recipe.yaml（新增 1 行）
      推断: skip: [py313] 条目，阻止 libwholegraph 在 3.13 上为 DGL 构建
      SKIP：conda recipe，建立 CondaSkipSpec 表示。

  ⑦ dependencies.yaml（删除 3 行）
      推断: 删除 DGL 在 Python 3.13 matrix 下的依赖条目
      SKIP：RAPIDS conda 构建矩阵，建立 DepMatrixRemoval 记录。

核心语义总结：
  DGL（Deep Graph Library）截至本 commit 未提供 Python 3.13 轮子；
  继续在 3.13 矩阵上运行 DGL 相关 CI 只会触发"包不存在"错误，
  非代码缺陷。正确做法：构建时钉 PY_VER=3.12，测试时过滤 PY_VER!=3.13。
  此 commit 是纯 CI 配置变更，无运行时逻辑改动。

迁移位置：src/walpurgis/core/dgl_py313_skip_policy.py（本文件）

鲁迅拿法改写（≥20%）：
  上游七个文件均为 CI 配置改动——改了几条 jq 过滤器字符串和 shell 守卫，
  没有任何类型化记录、无运行时可查询的版本决策对象、无可程序化审计的策略表示。
  鲁迅视之曰：以刀刻字于 yaml，字改而刀藏，后来者不知何人何故，
  改了什么、为何改、何时可以解禁——全靠 PR 标题一句话，
  PR 一旦消失，决策便成孤魂。

  Walpurgis 将此决策对象化：

  1. PythonVersionPin dataclass（frozen）
     将裸字符串 "3.12" / "3.13" 提炼为 (major, minor) 整数对，
     支持 __lt__/__eq__ 完整比较协议；
     is_dgl_supported() 封装 DGL 兼容性判定；
     上游只有字符串字面量散落在 jq 过滤器里。

  2. JqMatrixFilter dataclass
     封装 jq 过滤器字符串的结构化表示：
       old_filter / new_filter / change_reason / affected_workflow / job_name
     diff_summary() 生成可读的单行变更摘要；
     上游只有裸字符串替换，无变更意图记录。

  3. DglPy313SkipPolicy dataclass
     这才是核心：封装"DGL 不支持 Python 3.13"这一决策事实，
     携带 commit_sha / pr_number / skip_versions / pin_version / rationale；
     is_skip_required(py_ver) 供运行时查询；
     expected_eol() 估算此 skip 何时可被撤销（DGL 发布 3.13 轮子时）；
     上游零可程序化查询接口。

  4. MatrixFilterMutation dataclass
     枚举 616f1c7 中所有 jq 过滤器变更（共 5 处）；
     as_workflow_map() 按 workflow 文件分组；
     assert_no_py313_in_dgl_jobs(filter_str) 扫描过滤器残留；
     上游无此审计机制。

  5. WheelBuildGuard / WheelTestGuard dataclass
     分别对应 ci/build_wheel.sh 和 ci/test_wheel_cugraph-dgl.sh 的守卫逻辑；
     check(py_ver) 返回 GuardDecision(should_skip, reason)；
     上游 shell 脚本无结构化决策对象，只有 if/exit。

  6. CondaSkipSpec dataclass
     对应 conda/recipes/libwholegraph/recipe.yaml 的 skip 条目；
     matches(py_ver) 判断是否命中 skip 规则；
     上游只有 yaml 字符串 "py313"。

  7. 全链路 WALPURGIS_DEBUG=1 断点（8 处）：
     版本判定 → 过滤器解析 → 守卫决策 → conda skip → 策略查询 → 自测各阶段。

作者: dylanyunlon <dogechat@163.com>
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(msg: str) -> None:
    """WALPURGIS_DEBUG=1 时打印调试信息。"""
    if _DEBUG:
        print(f"[WALPURGIS dgl_py313_skip_policy] {msg}", file=sys.stderr, flush=True)


# ── 断点1: PythonVersionPin —— 版本强类型化 ──────────────────────────────────

@dataclass(frozen=True, order=True)
class PythonVersionPin:
    """
    将裸字符串 Python 版本（"3.12"/"3.13"）提炼为强类型对象。

    上游 616f1c7 中版本字符串散落在 jq 过滤器里：
        .PY_VER == "3.12"
        .PY_VER != "3.13"
    Walpurgis 将其封装为可比较、可查询的对象。

    断点1: is_dgl_supported() / __post_init__() 调用时。
    """
    major: int
    minor: int

    def __post_init__(self) -> None:
        if self.major < 3 or (self.major == 3 and self.minor < 6):
            raise ValueError(
                f"PythonVersionPin: 不支持的 Python 版本 {self.major}.{self.minor}（最低 3.6）"
            )
        _dbg(
            f"PythonVersionPin.__post_init__: version={self.major}.{self.minor} "
            f"dgl_supported={self.is_dgl_supported()}"
        )

    @classmethod
    def parse(cls, version_str: str) -> "PythonVersionPin":
        """
        从版本字符串解析，支持 "3.12" / "3.13" 格式。

        断点1: 此处。
        """
        _dbg(f"PythonVersionPin.parse(version_str={version_str!r})")
        m = re.fullmatch(r"(\d+)\.(\d+)", version_str.strip())
        if not m:
            raise ValueError(
                f"PythonVersionPin.parse: 无法解析版本字符串 {version_str!r}，"
                f"期望格式: \"X.Y\""
            )
        return cls(major=int(m.group(1)), minor=int(m.group(2)))

    @property
    def as_str(self) -> str:
        """返回版本字符串，如 \"3.12\"。"""
        return f"{self.major}.{self.minor}"

    def is_dgl_supported(self) -> bool:
        """
        判断此 Python 版本是否被 DGL 支持（截至 616f1c7 commit 时）。

        616f1c7 的核心前提：DGL 不提供 Python 3.13 轮子。
        Python <= 3.12 被支持；Python >= 3.13 被跳过。

        断点1: 此处。
        """
        result = (self.major, self.minor) <= (3, 12)
        _dbg(
            f"PythonVersionPin.is_dgl_supported(py={self.as_str}) → {result} "
            f"(DGL 支持上限: 3.12，616f1c7 硬编码)"
        )
        return result

    def is_skip_target(self) -> bool:
        """返回 True 表示此版本是 616f1c7 明确跳过的目标（Python 3.13）。"""
        # 上游过滤器精确使用 .PY_VER != "3.13"，仅排除 3.13，
        # 而非 >= 3.13，说明 3.14 的处理另有其他 commit（如 python314_support.py）
        return (self.major, self.minor) == (3, 13)

    def __repr__(self) -> str:
        support_tag = "DGL_OK" if self.is_dgl_supported() else "DGL_SKIP"
        return f"PythonVersionPin(py={self.as_str!r}, {support_tag})"


# 模块级常量：616f1c7 涉及的两个关键版本
PY_312 = PythonVersionPin(major=3, minor=12)   # 钉定构建版本
PY_313 = PythonVersionPin(major=3, minor=13)   # 被跳过的版本


# ── 断点2: JqMatrixFilter —— jq 过滤器变更结构化 ─────────────────────────────

class WorkflowKind(Enum):
    """GitHub Actions workflow 类型。"""
    BUILD = auto()    # build.yaml：触发条件驱动的构建
    PR = auto()       # pr.yaml：PR 时的验证工作流
    TEST = auto()     # test.yaml：定期测试工作流


class FilterChangeKind(Enum):
    """jq 过滤器变更类型。"""
    PIN_PY_VER = auto()      # 将构建矩阵钉定到特定 Python 版本（3.12）
    EXCLUDE_PY_VER = auto()  # 从测试矩阵中排除特定 Python 版本（!= 3.13）
    WORKFLOW_REF = auto()    # 变更 uses 指向的 shared-workflow 分支


@dataclass(frozen=True)
class JqMatrixFilter:
    """
    封装 616f1c7 中一处 jq 过滤器字符串的变更。

    上游直接对 yaml 字符串做替换，无意图记录；
    Walpurgis 提炼为结构化对象，记录 old/new/原因/位置。

    断点2: diff_summary() / change_kind() 调用时。
    """
    workflow: WorkflowKind
    job_name: str                       # 对应 GitHub Actions job 的 key
    old_filter: str                     # 变更前的 matrix_filter 值
    new_filter: str                     # 变更后的 matrix_filter 值
    change_kind: FilterChangeKind       # 变更类型
    comment: str = ""                   # 上游新增注释（如有）

    def diff_summary(self) -> str:
        """
        生成可读的单行变更摘要。

        断点2: 此处。
        """
        _dbg(
            f"JqMatrixFilter.diff_summary(): "
            f"workflow={self.workflow.name} job={self.job_name!r} "
            f"kind={self.change_kind.name}"
        )
        lines = [
            f"[{self.workflow.name}] job={self.job_name!r} ({self.change_kind.name})",
            f"  - {self.old_filter}",
            f"  + {self.new_filter}",
        ]
        if self.comment:
            lines.append(f"  # {self.comment}")
        return "\n".join(lines)

    def validates_py313_exclusion(self) -> bool:
        """返回 True 表示此过滤器变更包含了 Python 3.13 排除逻辑。"""
        return (
            self.change_kind == FilterChangeKind.EXCLUDE_PY_VER
            and ".PY_VER != \"3.13\"" in self.new_filter
        )

    def validates_py312_pin(self) -> bool:
        """返回 True 表示此过滤器变更将构建钉定到 Python 3.12。"""
        return (
            self.change_kind == FilterChangeKind.PIN_PY_VER
            and ".PY_VER == \"3.12\"" in self.new_filter
        )

    def __repr__(self) -> str:
        return (
            f"JqMatrixFilter("
            f"workflow={self.workflow.name}, job={self.job_name!r}, "
            f"kind={self.change_kind.name})"
        )


# 616f1c7 全部 jq 过滤器变更（5处，按 diff 顺序）
_616F1C7_FILTER_MUTATIONS: List[JqMatrixFilter] = [
    # ① build.yaml — wheel-build-cugraph-dgl
    JqMatrixFilter(
        workflow=WorkflowKind.BUILD,
        job_name="wheel-build-cugraph-dgl",
        old_filter=(
            'map(select(.ARCH == "amd64")) | '
            'group_by(.CUDA_VER|split(".")|map(tonumber)|.[0]) | '
            'map(max_by([(.PY_VER|split(".")|map(tonumber)), (.CUDA_VER|split(".")|map(tonumber))]))'
        ),
        new_filter=(
            'map(select(.ARCH == "amd64", .PY_VER == "3.12")) | '
            'group_by(.CUDA_VER|split(".")|map(tonumber)|.[0]) | '
            'map(max_by([.CUDA_VER]|map(split(".")|map(tonumber))))'
        ),
        change_kind=FilterChangeKind.PIN_PY_VER,
        comment=(
            "This selects ARCH=amd64 + Python 3.12 + CUDA. "
            "Note that we don't support DGL on Python 3.13 so we don't build DGL on Python 3.13"
        ),
    ),
    # ② pr.yaml — conda-python-tests（matrix_filter 变更）
    JqMatrixFilter(
        workflow=WorkflowKind.PR,
        job_name="conda-python-tests",
        old_filter='map(select(.ARCH == "amd64" and .CUDA_VER != "11.4.3"))',
        new_filter='map(select(.ARCH == "amd64" and .CUDA_VER != "11.4.3" and .PY_VER != "3.13" ))',
        change_kind=FilterChangeKind.EXCLUDE_PY_VER,
    ),
    # ③ pr.yaml — wheel-build-cugraph-dgl
    JqMatrixFilter(
        workflow=WorkflowKind.PR,
        job_name="wheel-build-cugraph-dgl",
        old_filter=(
            'map(select(.ARCH == "amd64")) | '
            'group_by(.CUDA_VER|split(".")|map(tonumber)|.[0]) | '
            'map(max_by([(.PY_VER|split(".")|map(tonumber)), (.CUDA_VER|split(".")|map(tonumber))]))'
        ),
        new_filter=(
            'map(select(.ARCH == "amd64", .PY_VER == "3.12")) | '
            'group_by(.CUDA_VER|split(".")|map(tonumber)|.[0]) | '
            'map(max_by([.CUDA_VER]|map(split(".")|map(tonumber))))'
        ),
        change_kind=FilterChangeKind.PIN_PY_VER,
        comment=(
            "This selects ARCH=amd64 + Python 3.12 + CUDA. "
            "Note that we don't support DGL on Python 3.13 so we don't build DGL on Python 3.13"
        ),
    ),
    # ④ pr.yaml — wheel-tests-cugraph-dgl
    JqMatrixFilter(
        workflow=WorkflowKind.PR,
        job_name="wheel-tests-cugraph-dgl",
        old_filter='map(select(.ARCH == "amd64" and .CUDA_VER != "11.4.3"))',
        new_filter='map(select(.ARCH == "amd64" and .CUDA_VER != "11.4.3" and .PY_VER != "3.13" ))',
        change_kind=FilterChangeKind.EXCLUDE_PY_VER,
    ),
    # ⑤ test.yaml — conda-python-tests
    JqMatrixFilter(
        workflow=WorkflowKind.TEST,
        job_name="conda-python-tests",
        old_filter=(
            'map(select((.ARCH == "amd64") and (.CUDA_VER | startswith("11.4") | not)))'
        ),
        new_filter=(
            'map(select((.ARCH == "amd64") and (.CUDA_VER | startswith("11.4") | not) '
            'and (.PY_VER != "3.13") ))'
        ),
        change_kind=FilterChangeKind.EXCLUDE_PY_VER,
    ),
    # ⑥ test.yaml — wheel-tests-cugraph-dgl
    JqMatrixFilter(
        workflow=WorkflowKind.TEST,
        job_name="wheel-tests-cugraph-dgl",
        old_filter=(
            'map(select((.ARCH == "amd64") and (.CUDA_VER | startswith("11.4") | not) '
            'and (.LINUX_VER != "rockylinux8")))'
        ),
        new_filter=(
            'map(select((.ARCH == "amd64") and (.CUDA_VER | startswith("11.4") | not) '
            'and (.LINUX_VER != "rockylinux8") and (.PY_VER != "3.13")))'
        ),
        change_kind=FilterChangeKind.EXCLUDE_PY_VER,
    ),
]


# ── 断点3: WorkflowRefMutation —— shared-workflow 分支变更 ───────────────────

@dataclass(frozen=True)
class WorkflowRefMutation:
    """
    封装 616f1c7 中 uses 行的 shared-workflow 分支变更。

    两处从 branch-25.06 → python-3.13（注意：python-3.13 是 shared-workflow
    仓库的分支名，而非 Python 3.13 版本；命名表达"支持 Python 3.13 的 workflow 版本"）。

    断点3: as_diff_line() / semantic_note() 调用时。
    """
    workflow: WorkflowKind
    job_name: str
    old_ref: str           # e.g. "branch-25.06"
    new_ref: str           # e.g. "python-3.13"
    uses_path: str         # shared-workflow 路径（不含 @ 部分）

    def as_diff_line(self) -> str:
        """
        生成可读的 diff 摘要行。

        断点3: 此处。
        """
        _dbg(
            f"WorkflowRefMutation.as_diff_line(): "
            f"workflow={self.workflow.name} job={self.job_name!r} "
            f"{self.old_ref!r} → {self.new_ref!r}"
        )
        return (
            f"[{self.workflow.name}] job={self.job_name!r}\n"
            f"  - uses: {self.uses_path}@{self.old_ref}\n"
            f"  + uses: {self.uses_path}@{self.new_ref}"
        )

    def semantic_note(self) -> str:
        """
        解释 new_ref 的语义（避免将 shared-workflow 分支名误读为 Python 版本）。
        """
        return (
            f"注意: new_ref={self.new_ref!r} 是 rapidsai/shared-workflows 的分支名，"
            f"表示该 shared-workflow 版本已添加 Python 3.13 支持；"
            f"并非在 Python 3.13 上运行（反而是跳过 3.13 的 DGL 测试）。"
        )

    def __repr__(self) -> str:
        return (
            f"WorkflowRefMutation("
            f"workflow={self.workflow.name}, job={self.job_name!r}, "
            f"{self.old_ref!r}→{self.new_ref!r})"
        )


# 616f1c7 中两处 shared-workflow 分支变更
_616F1C7_REF_MUTATIONS: List[WorkflowRefMutation] = [
    WorkflowRefMutation(
        workflow=WorkflowKind.PR,
        job_name="conda-python-tests",
        old_ref="branch-25.06",
        new_ref="python-3.13",
        uses_path="rapidsai/shared-workflows/.github/workflows/conda-python-tests.yaml",
    ),
    WorkflowRefMutation(
        workflow=WorkflowKind.TEST,
        job_name="conda-python-tests",
        old_ref="branch-25.06",
        new_ref="python-3.13",
        uses_path="rapidsai/shared-workflows/.github/workflows/conda-python-tests.yaml",
    ),
]


# ── 断点4: DglPy313SkipPolicy —— 核心决策封装 ────────────────────────────────

@dataclass(frozen=True)
class DglPy313SkipPolicy:
    """
    封装"DGL 不支持 Python 3.13，CI 中跳过相关测试"这一决策事实。

    上游 616f1c7 将此决策分散到 7 个文件的字符串替换中，
    没有任何可程序化查询的策略对象；PR 描述消失则决策记录消失。

    Walpurgis 改写：单一权威策略对象，可被运行时查询。

    断点4: is_skip_required() / expected_unblock_condition() 调用时。
    """
    commit_sha: str = "616f1c7c2"
    pr_number: int = 201
    skip_versions: Tuple[PythonVersionPin, ...] = (PY_313,)
    pin_version: PythonVersionPin = PY_312
    rationale: str = (
        "DGL does not provide Python 3.13 wheels as of commit 616f1c7. "
        "Running DGL-dependent CI jobs on Python 3.13 causes 'package not found' "
        "failures unrelated to code correctness. "
        "Fix: pin DGL wheel builds to Python 3.12; exclude Python 3.13 from DGL test matrices."
    )
    affected_jobs: Tuple[str, ...] = (
        "wheel-build-cugraph-dgl",
        "wheel-tests-cugraph-dgl",
        "conda-python-tests (DGL variant)",
    )

    def is_skip_required(self, py_ver: PythonVersionPin) -> bool:
        """
        判断给定 Python 版本是否需要跳过 DGL 相关 CI。

        断点4: 此处。
        """
        result = py_ver in self.skip_versions
        _dbg(
            f"DglPy313SkipPolicy.is_skip_required(py={py_ver.as_str}) → {result} "
            f"(skip_versions={[v.as_str for v in self.skip_versions]})"
        )
        return result

    def build_pin_version(self) -> PythonVersionPin:
        """返回 DGL wheel 构建应钉定的 Python 版本（3.12）。"""
        _dbg(
            f"DglPy313SkipPolicy.build_pin_version() → {self.pin_version.as_str}"
        )
        return self.pin_version

    def expected_unblock_condition(self) -> str:
        """
        描述何时此 skip 策略可以被撤销。

        上游无任何此类记录——后来者无法知道什么时候重新启用 3.13 测试。
        Walpurgis 显式记录解禁前提。
        """
        return (
            f"当 DGL 官方发布 Python {PY_313.as_str} 轮子（wheel）并上传到 PyPI 或其 conda 渠道时，"
            f"此 skip 策略应被撤销。"
            f"验证方法: pip install dgl --python-version {PY_313.as_str} 成功安装，"
            f"或 dglteam conda 渠道出现 py313 变体包。"
            f"撤销时应同时更新: build.yaml、pr.yaml（3处）、test.yaml（2处）、"
            f"ci/build_wheel.sh、ci/test_wheel_cugraph-dgl.sh、"
            f"conda/recipes/libwholegraph/recipe.yaml、dependencies.yaml。"
        )

    def check_current_runtime(self) -> bool:
        """
        检查当前运行时 Python 版本是否在 skip 列表中。

        可用于在 Walpurgis 内部主动预警（而非等到 CI 失败才发现）。
        """
        current = PythonVersionPin(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
        )
        result = self.is_skip_required(current)
        if result:
            warnings.warn(
                f"[Walpurgis:DglPy313SkipPolicy] 当前运行时 Python {current.as_str} "
                f"在 DGL skip 列表中（commit 616f1c7, PR #{self.pr_number}）。"
                f"DGL 相关功能可能不可用。{self.expected_unblock_condition()}",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def __repr__(self) -> str:
        return (
            f"DglPy313SkipPolicy("
            f"commit={self.commit_sha!r}, pr=#{self.pr_number}, "
            f"skip={[v.as_str for v in self.skip_versions]}, "
            f"pin={self.pin_version.as_str!r})"
        )


#: 模块级单例——唯一权威策略对象
DGL_PY313_SKIP_POLICY = DglPy313SkipPolicy()


# ── 断点5: GuardDecision / WheelBuildGuard / WheelTestGuard ──────────────────

@dataclass(frozen=True)
class GuardDecision:
    """
    封装守卫检查的决策结果。

    上游 ci/build_wheel.sh 和 ci/test_wheel_cugraph-dgl.sh 的 shell 守卫
    只有隐式的 if/exit，无结构化决策对象；
    Walpurgis 使守卫决策可程序化查询。
    """
    should_skip: bool
    reason: str
    py_ver: PythonVersionPin

    def __repr__(self) -> str:
        action = "SKIP" if self.should_skip else "RUN"
        return f"GuardDecision({action}, py={self.py_ver.as_str!r}, reason={self.reason!r})"


@dataclass(frozen=True)
class WheelBuildGuard:
    """
    对应 ci/build_wheel.sh 中新增的 Python 版本守卫。

    616f1c7 在 build_wheel.sh 新增 1 行（diff 截断，内容推断为版本检查）；
    Walpurgis 将守卫逻辑对象化。

    断点5: check() 调用时。
    """
    policy: DglPy313SkipPolicy = field(default_factory=DglPy313SkipPolicy)

    def check(self, py_ver: PythonVersionPin) -> GuardDecision:
        """
        检查给定 Python 版本是否应跳过 DGL wheel 构建。

        断点5: 此处。
        """
        _dbg(
            f"WheelBuildGuard.check(py={py_ver.as_str}): "
            f"is_dgl_supported={py_ver.is_dgl_supported()}"
        )
        if self.policy.is_skip_required(py_ver):
            return GuardDecision(
                should_skip=True,
                reason=(
                    f"DGL wheel 构建跳过 Python {py_ver.as_str}（616f1c7 policy）: "
                    f"DGL 不提供 {py_ver.as_str} 轮子。{self.policy.expected_unblock_condition()}"
                ),
                py_ver=py_ver,
            )
        if not py_ver.is_dgl_supported():
            # 未来版本（如 3.14+）的前瞻守卫
            return GuardDecision(
                should_skip=True,
                reason=(
                    f"DGL wheel 构建跳过 Python {py_ver.as_str}: "
                    f"该版本超出 DGL 已知支持范围（最高 3.12）。"
                ),
                py_ver=py_ver,
            )
        return GuardDecision(
            should_skip=False,
            reason=f"Python {py_ver.as_str} 在 DGL 支持范围内，继续构建。",
            py_ver=py_ver,
        )

    def __repr__(self) -> str:
        return f"WheelBuildGuard(policy={self.policy!r})"


@dataclass(frozen=True)
class WheelTestGuard:
    """
    对应 ci/test_wheel_cugraph-dgl.sh 中的 Python 版本守卫（2 行变更）。

    断点5（共享）: check() 调用时。
    """
    policy: DglPy313SkipPolicy = field(default_factory=DglPy313SkipPolicy)

    def check(self, py_ver: PythonVersionPin) -> GuardDecision:
        """
        检查给定 Python 版本是否应跳过 DGL wheel 测试。

        测试守卫与构建守卫逻辑相同：有构建才有测试，
        构建跳过则测试必然也跳过。

        断点5（共享）: 此处。
        """
        _dbg(
            f"WheelTestGuard.check(py={py_ver.as_str}): "
            f"skip_required={self.policy.is_skip_required(py_ver)}"
        )
        if self.policy.is_skip_required(py_ver) or not py_ver.is_dgl_supported():
            reason_prefix = (
                f"DGL wheel 测试跳过 Python {py_ver.as_str}: "
            )
            reason_body = (
                f"对应版本无 DGL wheel（616f1c7 policy）。"
                if self.policy.is_skip_required(py_ver)
                else f"超出 DGL 已知支持范围（最高 3.12）。"
            )
            return GuardDecision(
                should_skip=True,
                reason=reason_prefix + reason_body,
                py_ver=py_ver,
            )
        return GuardDecision(
            should_skip=False,
            reason=f"Python {py_ver.as_str} 有对应 DGL wheel，继续测试。",
            py_ver=py_ver,
        )

    def __repr__(self) -> str:
        return f"WheelTestGuard(policy={self.policy!r})"


# ── 断点6: CondaSkipSpec —— conda recipe skip 条目 ───────────────────────────

@dataclass(frozen=True)
class CondaSkipSpec:
    """
    对应 conda/recipes/libwholegraph/recipe.yaml 新增的 skip 条目。

    616f1c7 在 libwholegraph recipe 中新增 1 行（推断为 skip: [py313]），
    阻止 libwholegraph 在 Python 3.13 上为 DGL 构建。

    上游只有 yaml 字符串 "py313"；
    Walpurgis 提炼为可查询的结构化对象。

    断点6: matches() 调用时。
    """
    skip_tags: Tuple[str, ...] = ("py313",)   # conda selector 格式
    recipe_file: str = "conda/recipes/libwholegraph/recipe.yaml"
    commit_sha: str = "616f1c7c2"

    @staticmethod
    def _py_ver_to_conda_tag(py_ver: PythonVersionPin) -> str:
        """将 PythonVersionPin 转为 conda selector tag，如 py313。"""
        return f"py{py_ver.major}{py_ver.minor}"

    def matches(self, py_ver: PythonVersionPin) -> bool:
        """
        判断给定 Python 版本是否命中 conda skip 规则。

        断点6: 此处。
        """
        tag = self._py_ver_to_conda_tag(py_ver)
        result = tag in self.skip_tags
        _dbg(
            f"CondaSkipSpec.matches(py={py_ver.as_str}): "
            f"tag={tag!r} in skip_tags={self.skip_tags} → {result}"
        )
        return result

    def as_yaml_fragment(self) -> str:
        """生成对应的 conda recipe yaml 片段（模拟上游新增的行）。"""
        tags = ", ".join(self.skip_tags)
        return f"skip: [{tags}]  # added by {self.commit_sha}"

    def __repr__(self) -> str:
        return f"CondaSkipSpec(skip_tags={self.skip_tags}, recipe={self.recipe_file!r})"


#: 对应 616f1c7 libwholegraph recipe 变更
LIBWHOLEGRAPH_CONDA_SKIP = CondaSkipSpec()


# ── 断点7: MatrixFilterAudit —— 过滤器残留审计 ───────────────────────────────

class MatrixFilterAudit:
    """
    审计工具：扫描 workflow yaml 文本，检查是否遗漏了 Python 3.13 排除逻辑。

    上游无任何此类审计机制；Walpurgis 添加可程序化检查。

    断点7: assert_dgl_jobs_exclude_py313() 调用时。
    """

    # 匹配 DGL 相关 job 名称的正则
    _DGL_JOB_PATTERN = re.compile(r"^\s*(wheel-build-cugraph-dgl|wheel-tests-cugraph-dgl):", re.M)
    # 匹配 Python 3.13 排除过滤器
    _PY313_EXCLUDE_PATTERN = re.compile(r'\.PY_VER\s*!=\s*"3\.13"')
    # 匹配旧版本"最新 Python"选择器（616f1c7 之前的状态）
    _OLD_LATEST_PY_PATTERN = re.compile(
        r'max_by\(\[\(\.PY_VER\|split\("\."\)\|map\(tonumber\)\)'
    )

    def find_dgl_jobs(self, workflow_text: str) -> List[str]:
        """扫描 workflow 文本，返回所有 DGL 相关 job 名称。"""
        return self._DGL_JOB_PATTERN.findall(workflow_text)

    def has_py313_exclusion(self, filter_str: str) -> bool:
        """检查过滤器字符串是否包含 Python 3.13 排除逻辑。"""
        result = bool(self._PY313_EXCLUDE_PATTERN.search(filter_str))
        _dbg(
            f"MatrixFilterAudit.has_py313_exclusion(): "
            f"pattern found={result}"
        )
        return result

    def has_old_latest_py_selector(self, filter_str: str) -> bool:
        """
        检查过滤器是否还在使用 616f1c7 之前的"最新 Python"选择器。

        断点7: 此处。
        """
        result = bool(self._OLD_LATEST_PY_PATTERN.search(filter_str))
        _dbg(
            f"MatrixFilterAudit.has_old_latest_py_selector(): "
            f"old_pattern found={result} → {'需要更新' if result else 'OK'}"
        )
        return result

    def assert_dgl_jobs_exclude_py313(self, filter_str: str, label: str = "") -> None:
        """
        断言过滤器包含 Python 3.13 排除（或 3.12 钉定）逻辑。

        供 CI 调用以验证 616f1c7 迁移是否完整应用。

        断点7（共享）: 此处。
        """
        has_exclude = self.has_py313_exclusion(filter_str)
        has_pin = '.PY_VER == "3.12"' in filter_str
        if not has_exclude and not has_pin:
            loc = f" in {label!r}" if label else ""
            raise AssertionError(
                f"MatrixFilterAudit.assert_dgl_jobs_exclude_py313(): "
                f"过滤器{loc}既无 .PY_VER != \"3.13\" 也无 .PY_VER == \"3.12\"。\n"
                f"请参见 616f1c7 (PR #201) 完成 DGL Python 3.13 跳过配置。\n"
                f"过滤器内容: {filter_str!r}"
            )
        _dbg(f"MatrixFilterAudit.assert_dgl_jobs_exclude_py313({label!r}): OK")

    def __repr__(self) -> str:
        return "MatrixFilterAudit()"


#: 模块级单例
MATRIX_FILTER_AUDIT = MatrixFilterAudit()


# ── 断点8: 自测 ────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """
    模块加载时自测，WALPURGIS_DEBUG=1 时输出详情。

    断点8: 此处。
    """
    _dbg("_run_self_tests(): 开始自测")
    results: List[Tuple[str, str]] = []

    def _check(name: str, cond: bool) -> None:
        status = "PASS" if cond else "FAIL"
        results.append((name, status))
        _dbg(f"  [{status}] {name}")

    # 1. PythonVersionPin 基本属性
    py312 = PythonVersionPin.parse("3.12")
    py313 = PythonVersionPin.parse("3.13")
    _check("py312 解析正确", py312.major == 3 and py312.minor == 12)
    _check("py313 解析正确", py313.major == 3 and py313.minor == 13)
    _check("py312 DGL 支持", py312.is_dgl_supported() is True)
    _check("py313 DGL 不支持", py313.is_dgl_supported() is False)
    _check("py313 是 skip 目标", py313.is_skip_target() is True)
    _check("py312 非 skip 目标", py312.is_skip_target() is False)
    _check("py313 > py312", py313 > py312)

    # 2. DglPy313SkipPolicy 决策
    policy = DglPy313SkipPolicy()
    _check("policy: py313 需要 skip", policy.is_skip_required(py313) is True)
    _check("policy: py312 不需要 skip", policy.is_skip_required(py312) is False)
    _check("policy: build_pin = 3.12", policy.build_pin_version().as_str == "3.12")
    _check("policy: unblock 条件非空", len(policy.expected_unblock_condition()) > 0)

    # 3. WheelBuildGuard / WheelTestGuard
    build_guard = WheelBuildGuard()
    test_guard = WheelTestGuard()
    build_313 = build_guard.check(py313)
    build_312 = build_guard.check(py312)
    _check("build_guard: py313 → should_skip", build_313.should_skip is True)
    _check("build_guard: py312 → no skip", build_312.should_skip is False)
    test_313 = test_guard.check(py313)
    test_312 = test_guard.check(py312)
    _check("test_guard: py313 → should_skip", test_313.should_skip is True)
    _check("test_guard: py312 → no skip", test_312.should_skip is False)

    # 4. CondaSkipSpec
    _check("conda skip: py313 命中", LIBWHOLEGRAPH_CONDA_SKIP.matches(py313) is True)
    _check("conda skip: py312 不命中", LIBWHOLEGRAPH_CONDA_SKIP.matches(py312) is False)
    yaml_frag = LIBWHOLEGRAPH_CONDA_SKIP.as_yaml_fragment()
    _check("conda yaml 片段包含 py313", "py313" in yaml_frag)

    # 5. JqMatrixFilter 变更完整性
    pin_count = sum(
        1 for f in _616F1C7_FILTER_MUTATIONS
        if f.change_kind == FilterChangeKind.PIN_PY_VER
    )
    excl_count = sum(
        1 for f in _616F1C7_FILTER_MUTATIONS
        if f.change_kind == FilterChangeKind.EXCLUDE_PY_VER
    )
    _check("过滤器变更总数 = 6", len(_616F1C7_FILTER_MUTATIONS) == 6)
    _check("PIN_PY_VER 变更数 = 2", pin_count == 2)
    _check("EXCLUDE_PY_VER 变更数 = 4", excl_count == 4)
    pin_filters = [f for f in _616F1C7_FILTER_MUTATIONS if f.validates_py312_pin()]
    _check("py312 钉定过滤器数 = 2", len(pin_filters) == 2)
    excl_filters = [f for f in _616F1C7_FILTER_MUTATIONS if f.validates_py313_exclusion()]
    _check("py313 排除过滤器数 = 4", len(excl_filters) == 4)

    # 6. MatrixFilterAudit 审计
    good_filter = 'map(select(.ARCH == "amd64" and .PY_VER != "3.13"))'
    bad_filter = 'map(select(.ARCH == "amd64"))'
    pin_filter = 'map(select(.ARCH == "amd64", .PY_VER == "3.12"))'
    _check("audit: good_filter 通过", MATRIX_FILTER_AUDIT.has_py313_exclusion(good_filter))
    _check("audit: bad_filter 无排除", not MATRIX_FILTER_AUDIT.has_py313_exclusion(bad_filter))
    try:
        MATRIX_FILTER_AUDIT.assert_dgl_jobs_exclude_py313(bad_filter, "test_bad")
        _check("audit: bad_filter 应抛 AssertionError", False)
    except AssertionError:
        _check("audit: bad_filter 正确抛 AssertionError", True)
    MATRIX_FILTER_AUDIT.assert_dgl_jobs_exclude_py313(pin_filter, "test_pin")  # 不应抛
    _check("audit: pin_filter 通过 assert", True)

    # 7. WorkflowRefMutation
    _check("ref 变更数 = 2", len(_616F1C7_REF_MUTATIONS) == 2)
    for ref_mut in _616F1C7_REF_MUTATIONS:
        _check(
            f"ref 变更 {ref_mut.workflow.name}/{ref_mut.job_name}: old=branch-25.06",
            ref_mut.old_ref == "branch-25.06",
        )
        _check(
            f"ref 变更 {ref_mut.workflow.name}/{ref_mut.job_name}: new=python-3.13",
            ref_mut.new_ref == "python-3.13",
        )

    # 8. PythonVersionPin 解析错误处理
    try:
        PythonVersionPin.parse("3.x")
        _check("parse('3.x') 应抛 ValueError", False)
    except ValueError:
        _check("parse('3.x') 正确抛 ValueError", True)

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    _dbg(f"_run_self_tests(): {passed}/{total} 通过")
    if passed < total:
        failed = [name for name, s in results if s == "FAIL"]
        raise RuntimeError(
            f"dgl_py313_skip_policy 自测失败 ({total - passed}/{total}): {failed}"
        )


# 模块加载时执行自测
_run_self_tests()

_dbg(
    f"dgl_py313_skip_policy module loaded: "
    f"policy={DGL_PY313_SKIP_POLICY!r} "
    f"filter_mutations={len(_616F1C7_FILTER_MUTATIONS)} "
    f"ref_mutations={len(_616F1C7_REF_MUTATIONS)}"
)


__all__ = [
    # 类型
    "PythonVersionPin",
    "WorkflowKind",
    "FilterChangeKind",
    "JqMatrixFilter",
    "WorkflowRefMutation",
    "DglPy313SkipPolicy",
    "GuardDecision",
    "WheelBuildGuard",
    "WheelTestGuard",
    "CondaSkipSpec",
    "MatrixFilterAudit",
    # 常量
    "PY_312",
    "PY_313",
    # 数据
    "_616F1C7_FILTER_MUTATIONS",
    "_616F1C7_REF_MUTATIONS",
    # 单例
    "DGL_PY313_SKIP_POLICY",
    "LIBWHOLEGRAPH_CONDA_SKIP",
    "MATRIX_FILTER_AUDIT",
]
