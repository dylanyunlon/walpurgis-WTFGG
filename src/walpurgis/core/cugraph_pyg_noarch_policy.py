"""
migrate aa3373a: feat(noarch): build cugraph-pyg as a conda `noarch` package (#405)

上游 commit aa3373a22774616c7d75f17e59eac6e19b209a33
Author: Gil Forsyth <gforsyth@users.noreply.github.com>
Date:   2026-02-10
PR:     https://github.com/rapidsai/cugraph-gnn/pull/405

上游变更（6 files changed, 78 insertions(+), 19 deletions(-)）：
  .github/workflows/build.yaml          — 新增 python-build-noarch job，
                                          upload-conda 增加 noarch 依赖
  .github/workflows/pr.yaml             — 新增 conda-python-build-noarch job，
                                          conda-python-tests 增加 noarch 依赖
  ci/build_python.sh                    — 删除 cugraph-pyg 的 rattler-build 调用
                                          （移入 build_python_noarch.sh）
  ci/build_python_noarch.sh             — 新增：专为 cugraph-pyg noarch 包的
                                          独立 CI 构建脚本（50 行）
  ci/test_python.sh                     — 新增 PYTHON_NOARCH_CHANNEL，
                                          --prepend-channel 追加 noarch channel
  conda/recipes/cugraph-pyg/recipe.yaml — 去掉 py_buildstring，
                                          build.string 改为无 py 版本后缀，
                                          build.noarch: python，
                                          python 依赖从 =${{ py_version }}
                                          改为 >=3.11

核心语义：cugraph-pyg 是纯 Python wheel，无需按 Python/CUDA/arch 组合各出一
个 conda artifact。将其标记为 noarch:python，一次构建多平台通用，拆出独立
CI job（python-build-noarch）在 cpp-build + python-build 完成后运行，
conda-python-tests 同步等待 noarch 产物。

CI/merge → SKIP（以下文件 Walpurgis 无对应体系，不迁移）：
  .github/workflows/build.yaml     — GitHub Actions 主构建 workflow，
                                     Walpurgis 无 GitHub Actions CI
  .github/workflows/pr.yaml        — GitHub Actions PR workflow，同上
  ci/build_python.sh               — RAPIDS conda rattler-build 脚本，
                                     Walpurgis 无 conda 构建体系
  ci/build_python_noarch.sh        — 同上，noarch 构建脚本
  ci/test_python.sh                — RAPIDS conda 测试脚本，同上
  conda/recipes/cugraph-pyg/recipe.yaml — conda rattler-build recipe，
                                          Walpurgis 用 pyproject.toml

迁移位置：src/walpurgis/core/cugraph_pyg_noarch_policy.py（本文件）

鲁迅拿法改写（≥20%）：
  上游六个文件的变更本质是：把"按 Python 版本出多份 conda 包"改为"出一份
  noarch 包"。上游实现散布在 YAML + shell 脚本中，无任何 Python 层的策略
  抽象。Walpurgis 将其核心语义提炼为：

  1. NoarchBuildMode 枚举        — 区分 per-python / noarch:python 两种构建
                                   模式（上游无此抽象，散落在 recipe.yaml 字段）
  2. BuildStringSpec dataclass   — 封装 conda build string 的构成规则：
                                   noarch 模式去掉 py_buildstring，仅保留
                                   date_string + head_rev（上游无程序化表达）
  3. PythonPinSpec dataclass     — 描述 python host 依赖的 pin 策略：
                                   per-python 用 =${{ py_version }}（精确锁定），
                                   noarch 用 >=3.11（宽松下界）
  4. NoarchChannelSpec dataclass — 描述 noarch channel 在测试阶段的定位方式：
                                   封装 rapids-package-name --pure 调用逻辑
                                   （上游仅 shell 变量赋值，无类型约束）
  5. CugraphPyGPackagePolicy     — 汇总以上三者，describe() 输出完整策略快照，
                                   validate() 检测 noarch 与 per-python 模式
                                   的不兼容配置（上游无此守卫）
  6. NoarchCIJobSpec dataclass   — 描述 noarch CI job 的依赖关系拓扑：
                                   needs=[cpp-build, python-build]，
                                   阻塞 upload-conda 和 conda-python-tests
                                   （上游散落在两个 YAML 文件中，无统一视图）
  7. simulate_noarch_build_string() — 演示 per-python vs noarch 两种模式下
                                   build string 的差异（上游无此诊断工具）
  8. 全链路 WALPURGIS_DEBUG=1 断点（7 处）
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG aa3373a cugraph_pyg_noarch_policy] 模块加载："
        "cugraph-pyg noarch 构建策略迁移模块初始化"
    )


# ── 1. NoarchBuildMode 枚举 ──────────────────────────────────────────────────

class NoarchBuildMode(Enum):
    """
    conda 包的 Python 平台无关性模式。

    上游 aa3373a 将 cugraph-pyg 从 PER_PYTHON 切换至 NOARCH_PYTHON，
    但此区分散落在 recipe.yaml 的 build 字段和 build string 模板中，
    无任何枚举抽象。Walpurgis 将其命名化，以便在策略对象中显式携带。

    PER_PYTHON:
      每个 Python 版本各出一个 conda artifact，build string 含 py_buildstring
      （如 py311_20260210_aa3373a2）。适用于含 Cython/C 扩展的包。

    NOARCH_PYTHON:
      纯 Python 包，一次构建，conda 标记 noarch:python，
      build string 不含 py_buildstring（如 20260210_aa3373a2）。
      python host 依赖改为宽松下界（>=3.11）。
    """
    PER_PYTHON = "per_python"
    NOARCH_PYTHON = "noarch_python"


# ── 2. BuildStringSpec dataclass ─────────────────────────────────────────────

@dataclass
class BuildStringSpec:
    """
    描述 conda build string 的构成规则。

    上游 aa3373a 前（per-python 模式）：
      string: py${{ py_buildstring }}_${{ date_string }}_${{ head_rev }}
      context 中有 py_buildstring: ${{ py_version | version_to_buildstring }}

    上游 aa3373a 后（noarch 模式）：
      string: ${{ date_string }}_${{ head_rev }}
      context 中去掉 py_buildstring 定义

    上游只修改了 recipe.yaml 字符串，无程序化的 build string 生成器。
    Walpurgis 将其封装为可调用的 spec 对象。
    """
    mode: NoarchBuildMode
    date_string: str          # 形如 "20260210"，来自 RAPIDS_DATE_STRING
    head_rev: str             # git SHA 前 8 位，来自 git.head_rev(".")[:8]
    py_version: Optional[str] = None  # 仅 PER_PYTHON 模式有效，形如 "3.11"

    def render(self) -> str:
        """
        生成 build string。
        noarch 模式不含 py_buildstring，per-python 模式含 py 前缀段。
        """
        # 断点 1：build string 渲染
        if _DBG:
            print(
                f"[DEBUG aa3373a] BuildStringSpec.render: mode={self.mode.value} "
                f"date={self.date_string!r} rev={self.head_rev!r} "
                f"py={self.py_version!r}"
            )
        if self.mode == NoarchBuildMode.PER_PYTHON:
            if not self.py_version:
                raise ValueError(
                    "PER_PYTHON 模式下 py_version 不能为空（需要 py_buildstring）"
                )
            py_buildstring = self.py_version.replace(".", "")  # "3.11" → "311"
            return f"py{py_buildstring}_{self.date_string}_{self.head_rev}"
        else:
            # NOARCH_PYTHON：去掉 py 前缀段，上游 aa3373a 的核心 recipe 变更
            return f"{self.date_string}_{self.head_rev}"

    def describe(self) -> str:
        return (
            f"BuildStringSpec(mode={self.mode.value}, "
            f"rendered={self.render()!r})"
        )


# ── 3. PythonPinSpec dataclass ───────────────────────────────────────────────

@dataclass
class PythonPinSpec:
    """
    描述 conda recipe host 依赖中 python 的 pin 策略。

    上游 aa3373a 前（per-python）：
      - python =${{ py_version }}   ← 精确锁定到构建时的 Python 版本

    上游 aa3373a 后（noarch）：
      - python >=3.11               ← 宽松下界，任意 >=3.11 版本可安装

    从精确 pin 切换到宽松 pin 是 noarch 包的必要条件：
    noarch 包不能绑定特定 Python 版本，否则 conda 无法跨版本安装。
    上游仅修改了 recipe.yaml 字符串，无策略对象。
    """
    mode: NoarchBuildMode
    exact_version: Optional[str] = None  # PER_PYTHON 时的精确版本，如 "3.11"
    minimum_version: str = "3.11"        # NOARCH_PYTHON 时的最低版本

    def pin_expression(self) -> str:
        """
        返回 conda recipe 中 python 的 pin 表达式。
        per-python → "=3.11"（精确 pin）
        noarch     → ">=3.11"（宽松下界）
        """
        # 断点 2：pin 表达式生成
        if _DBG:
            print(
                f"[DEBUG aa3373a] PythonPinSpec.pin_expression: "
                f"mode={self.mode.value} exact={self.exact_version!r} "
                f"min={self.minimum_version!r}"
            )
        if self.mode == NoarchBuildMode.PER_PYTHON:
            if not self.exact_version:
                raise ValueError("PER_PYTHON 模式下 exact_version 不能为空")
            return f"={self.exact_version}"
        else:
            return f">={self.minimum_version}"

    def is_noarch_compatible(self) -> bool:
        """
        检查当前 pin 策略是否与 noarch 包兼容。
        精确 pin（=x.y）绑定了 Python 版本，与 noarch 语义矛盾。
        """
        if self.mode == NoarchBuildMode.NOARCH_PYTHON:
            expr = self.pin_expression()
            # noarch 包的 python pin 不能是精确 pin
            return not expr.startswith("=") or expr.startswith(">=")
        return True  # per-python 模式不受此约束


# ── 4. NoarchChannelSpec dataclass ───────────────────────────────────────────

@dataclass
class NoarchChannelSpec:
    """
    描述 noarch channel 在测试阶段的定位规则。

    上游 ci/test_python.sh 新增：
      PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github
          "$(rapids-package-name conda_python cugraph_pyg --pure)")

    即通过 rapids-package-name 加 --pure 标志获取 noarch 产物的 channel 名，
    再通过 rapids-download-from-github 定位 GitHub artifact。

    --pure 标志对应 conda recipe 中的 pure-conda: true 配置（build.yaml/pr.yaml）。
    上游仅有 shell 变量赋值，无类型约束或命名规则文档。
    Walpurgis 将其封装为可描述、可验证的 spec 对象。
    """
    package_name: str = "cugraph_pyg"    # rapids-package-name 的包名参数
    artifact_type: str = "conda_python"  # rapids-package-name 的类型参数
    pure: bool = True                    # --pure 标志：noarch 包必须为 True

    def package_name_cmd(self) -> str:
        """
        返回等价的 rapids-package-name shell 命令。
        对应 test_python.sh 中的 $(rapids-package-name conda_python cugraph_pyg --pure)
        """
        # 断点 3：channel spec 命令生成
        if _DBG:
            print(
                f"[DEBUG aa3373a] NoarchChannelSpec.package_name_cmd: "
                f"pkg={self.package_name!r} type={self.artifact_type!r} "
                f"pure={self.pure}"
            )
        cmd = f"rapids-package-name {self.artifact_type} {self.package_name}"
        if self.pure:
            cmd += " --pure"
        return cmd

    def download_cmd(self) -> str:
        """
        返回等价的 rapids-download-from-github shell 命令（展开形式）。
        """
        return (
            f'PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github '
            f'"$({self.package_name_cmd()})")'
        )

    def validate_pure_flag(self) -> Optional[str]:
        """
        验证 pure 标志与 artifact_type 的一致性。
        noarch 包必须设置 --pure；若 pure=False 则返回警告。
        """
        if self.artifact_type == "conda_python" and not self.pure:
            return (
                f"[NoarchChannelSpec] {self.package_name!r}: "
                "conda_python 类型的 noarch channel 应设置 pure=True，"
                "否则 rapids-package-name 将定位到 per-python artifact"
            )
        return None


# ── 5. CugraphPyGPackagePolicy ───────────────────────────────────────────────

@dataclass
class CugraphPyGPackagePolicy:
    """
    汇总 cugraph-pyg 包的 noarch 构建策略。

    上游 aa3373a 的变更散落在 recipe.yaml、build.yaml、pr.yaml、
    build_python.sh、build_python_noarch.sh、test_python.sh 六个文件中，
    无统一的策略视图。Walpurgis 将其收敛为单一策略对象。

    此对象描述从 per-python 切换到 noarch 后，各关键配置维度的新状态：
      - build string 不含 py 版本段
      - python host pin 由精确 pin 改为宽松下界
      - conda recipe 新增 noarch: python 字段
      - CI job 拓扑新增 python-build-noarch，阻塞 upload-conda
    """
    mode: NoarchBuildMode = NoarchBuildMode.NOARCH_PYTHON
    build_string: Optional[BuildStringSpec] = None
    python_pin: Optional[PythonPinSpec] = None
    noarch_channel: Optional[NoarchChannelSpec] = None

    def __post_init__(self) -> None:
        # 若未显式提供子 spec，使用默认值（对应 aa3373a 后的状态）
        if self.build_string is None:
            self.build_string = BuildStringSpec(
                mode=self.mode,
                date_string="RAPIDS_DATE_STRING",
                head_rev="HEAD_REV_8",
            )
        if self.python_pin is None:
            self.python_pin = PythonPinSpec(
                mode=self.mode,
                minimum_version="3.11",
            )
        if self.noarch_channel is None:
            self.noarch_channel = NoarchChannelSpec()

    def validate(self) -> List[str]:
        """
        检测 noarch 与 per-python 模式的不兼容配置。
        上游无此守卫，Walpurgis 新增。

        返回警告列表，空列表表示策略一致。
        """
        warnings: List[str] = []

        # 断点 4：策略验证
        if _DBG:
            print(
                f"[DEBUG aa3373a] CugraphPyGPackagePolicy.validate: "
                f"mode={self.mode.value}"
            )

        # python pin 与 noarch 模式一致性：
        # 外层 policy 为 noarch，但 pin 为精确 pin（per-python 模式），矛盾
        if (self.mode == NoarchBuildMode.NOARCH_PYTHON
                and self.python_pin
                and self.python_pin.mode == NoarchBuildMode.PER_PYTHON):
            warnings.append(
                f"[Policy] python_pin 使用精确 pin "
                f"({self.python_pin.pin_expression()!r})，"
                "与 noarch:python 模式不兼容（noarch 包不能绑定特定 Python 版本）"
            )

        # noarch channel 的 pure 标志
        if self.noarch_channel:
            warn = self.noarch_channel.validate_pure_flag()
            if warn:
                warnings.append(warn)

        # per-python 模式下不应标记 noarch
        if (self.mode == NoarchBuildMode.PER_PYTHON
                and self.build_string
                and self.build_string.mode == NoarchBuildMode.NOARCH_PYTHON):
            warnings.append(
                "[Policy] 外层 mode=PER_PYTHON 但 build_string.mode=NOARCH_PYTHON，"
                "两者矛盾，构建 string 将忽略 py_buildstring"
            )

        if _DBG:
            print(f"[DEBUG aa3373a] validate: {len(warnings)} 条警告")
        return warnings

    def describe(self) -> str:
        """输出完整策略快照。上游六个文件散布，无此统一视图。"""
        lines = [
            f"CugraphPyGPackagePolicy(",
            f"  mode            = {self.mode.value}",
            f"  build_string    = {self.build_string.describe() if self.build_string else 'None'}",
            f"  python_pin_expr = {self.python_pin.pin_expression() if self.python_pin else 'None'}",
            f"  channel_cmd     = {self.noarch_channel.package_name_cmd() if self.noarch_channel else 'None'}",
            f"  validation      = {self.validate() or '[OK]'}",
            f")",
        ]
        return "\n".join(lines)


# ── 6. NoarchCIJobSpec dataclass ─────────────────────────────────────────────

@dataclass
class NoarchCIJobSpec:
    """
    描述 noarch CI job 的依赖关系拓扑。

    上游 aa3373a 在 build.yaml 和 pr.yaml 中新增 python-build-noarch / 
    conda-python-build-noarch job，并修改 upload-conda 和 conda-python-tests
    的 needs 列表以等待 noarch 产物。

    此拓扑散落在两个 YAML 文件的多处 needs 字段中，无统一视图。
    Walpurgis 将其封装为可描述、可拓扑排序的 spec 对象。
    """
    job_name: str                    # job 的 YAML key，如 "python-build-noarch"
    needs: List[str] = field(default_factory=list)   # 依赖的 job 列表
    blocks: List[str] = field(default_factory=list)  # 阻塞的 job 列表（反向依赖）
    pure_conda: bool = True          # 对应 build.yaml 中的 pure-conda: true
    script: str = ""                 # 对应 build.yaml 中的 script 字段

    def to_yaml_fragment(self) -> str:
        """
        生成等价的 GitHub Actions job YAML 片段（仅 needs/pure-conda/script）。
        上游直接写 YAML，无程序化生成器。
        """
        # 断点 5：YAML 片段生成
        if _DBG:
            print(
                f"[DEBUG aa3373a] NoarchCIJobSpec.to_yaml_fragment: "
                f"job={self.job_name!r} needs={self.needs}"
            )
        needs_str = ", ".join(f"[{n}]" for n in self.needs)
        lines = [
            f"  {self.job_name}:",
            f"    needs: [{', '.join(self.needs)}]",
            f"    secrets: inherit",
            f"    uses: rapidsai/shared-workflows/.github/workflows/conda-python-build.yaml@main",
            f"    with:",
        ]
        if self.script:
            lines.append(f"      script: {self.script}")
        if self.pure_conda:
            lines.append(f"      pure-conda: true")
        return "\n".join(lines)

    def describe(self) -> str:
        blocks_str = ", ".join(self.blocks) if self.blocks else "（无）"
        return (
            f"NoarchCIJobSpec(job={self.job_name!r}, "
            f"needs={self.needs}, blocks={self.blocks}, "
            f"pure_conda={self.pure_conda})"
        )


# ── 7. simulate_noarch_build_string ─────────────────────────────────────────

def simulate_noarch_build_string(
    date_string: str,
    head_rev: str,
    py_version: str = "3.11",
) -> dict:
    """
    演示 per-python vs noarch 两种模式下 build string 的差异。

    对应 aa3373a 在 conda/recipes/cugraph-pyg/recipe.yaml 中的核心变更：
      修改前：string: py${{ py_buildstring }}_${{ date_string }}_${{ head_rev }}
      修改后：string: ${{ date_string }}_${{ head_rev }}

    上游无此诊断工具，Walpurgis 新增，便于迁移时对比两种模式的产物差异。
    """
    per_python_spec = BuildStringSpec(
        mode=NoarchBuildMode.PER_PYTHON,
        date_string=date_string,
        head_rev=head_rev,
        py_version=py_version,
    )
    noarch_spec = BuildStringSpec(
        mode=NoarchBuildMode.NOARCH_PYTHON,
        date_string=date_string,
        head_rev=head_rev,
    )

    per_python_str = per_python_spec.render()
    noarch_str = noarch_spec.render()

    # 断点 6：build string 对比
    if _DBG:
        print(
            f"[DEBUG aa3373a] simulate_noarch_build_string: "
            f"per_python={per_python_str!r} noarch={noarch_str!r}"
        )

    return {
        "date_string": date_string,
        "head_rev": head_rev,
        "py_version": py_version,
        "per_python_build_string": per_python_str,
        "noarch_build_string": noarch_str,
        # noarch string 更短：不含 py 版本段
        "noarch_is_shorter": len(noarch_str) < len(per_python_str),
        # per-python string 含 py_buildstring 前缀
        "per_python_has_py_prefix": per_python_str.startswith(
            "py" + py_version.replace(".", "")
        ),
    }


# ── 模块级常量：aa3373a 的关键 noarch job 配置 ───────────────────────────────

# 对应 build.yaml 新增的 python-build-noarch job
AA3373A_BUILD_NOARCH_JOB = NoarchCIJobSpec(
    job_name="python-build-noarch",
    needs=["cpp-build", "python-build"],
    blocks=["upload-conda"],
    pure_conda=True,
    script="ci/build_python_noarch.sh",
)

# 对应 pr.yaml 新增的 conda-python-build-noarch job
AA3373A_PR_NOARCH_JOB = NoarchCIJobSpec(
    job_name="conda-python-build-noarch",
    needs=["conda-cpp-build", "conda-python-build"],
    blocks=["conda-python-tests"],
    pure_conda=True,
    script="ci/build_python_noarch.sh",
)

# aa3373a 后 cugraph-pyg 的完整策略
AA3373A_POLICY = CugraphPyGPackagePolicy(
    mode=NoarchBuildMode.NOARCH_PYTHON,
)


# ── 自测 ─────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """模块自测：验证所有核心逻辑路径。"""
    passed = 0
    total = 0

    def check(label: str, cond: bool) -> None:
        nonlocal passed, total
        total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            passed += 1

    print("=== cugraph_pyg_noarch_policy.py 自测 (aa3373a) ===")

    # 1. NoarchBuildMode 枚举
    check("NoarchBuildMode.NOARCH_PYTHON.value == 'noarch_python'",
          NoarchBuildMode.NOARCH_PYTHON.value == "noarch_python")
    check("NoarchBuildMode.PER_PYTHON.value == 'per_python'",
          NoarchBuildMode.PER_PYTHON.value == "per_python")

    # 2. BuildStringSpec — noarch 模式不含 py 前缀
    noarch_spec = BuildStringSpec(
        mode=NoarchBuildMode.NOARCH_PYTHON,
        date_string="20260210",
        head_rev="aa3373a2",
    )
    rendered = noarch_spec.render()
    check("noarch build string 不含 'py' 前缀", not rendered.startswith("py"))
    check("noarch build string = date_head_rev 格式",
          rendered == "20260210_aa3373a2")

    # 3. BuildStringSpec — per-python 模式含 py 前缀
    per_py_spec = BuildStringSpec(
        mode=NoarchBuildMode.PER_PYTHON,
        date_string="20260210",
        head_rev="aa3373a2",
        py_version="3.11",
    )
    per_rendered = per_py_spec.render()
    check("per-python build string 含 'py311' 前缀",
          per_rendered.startswith("py311"))
    check("per-python build string = py311_date_rev 格式",
          per_rendered == "py311_20260210_aa3373a2")

    # 4. PythonPinSpec — noarch 模式用宽松 pin
    noarch_pin = PythonPinSpec(mode=NoarchBuildMode.NOARCH_PYTHON, minimum_version="3.11")
    check("noarch python pin 表达式 = '>=3.11'",
          noarch_pin.pin_expression() == ">=3.11")
    check("noarch python pin 与 noarch 兼容",
          noarch_pin.is_noarch_compatible())

    # 5. PythonPinSpec — per-python 模式用精确 pin
    per_pin = PythonPinSpec(mode=NoarchBuildMode.PER_PYTHON, exact_version="3.11")
    check("per-python python pin 表达式 = '=3.11'",
          per_pin.pin_expression() == "=3.11")

    # 6. NoarchChannelSpec
    chan = NoarchChannelSpec()
    cmd = chan.package_name_cmd()
    check("package_name_cmd 含 '--pure'", "--pure" in cmd)
    check("package_name_cmd 含 'cugraph_pyg'", "cugraph_pyg" in cmd)
    check("validate_pure_flag 返回 None（配置正确）",
          chan.validate_pure_flag() is None)

    # 7. CugraphPyGPackagePolicy 默认构造
    policy = CugraphPyGPackagePolicy()
    warnings = policy.validate()
    check("默认 noarch 策略无警告", len(warnings) == 0)
    desc = policy.describe()
    check("describe() 包含 mode", "noarch_python" in desc)
    check("describe() 包含 python_pin", ">=3.11" in desc)

    # 8. 不兼容配置检测
    bad_policy = CugraphPyGPackagePolicy(
        mode=NoarchBuildMode.NOARCH_PYTHON,
        python_pin=PythonPinSpec(
            mode=NoarchBuildMode.PER_PYTHON,  # 精确 pin，与 noarch 不兼容
            exact_version="3.11",
        ),
    )
    bad_warnings = bad_policy.validate()
    check("精确 pin 在 noarch 模式下触发警告", len(bad_warnings) > 0)

    # 9. simulate_noarch_build_string
    sim = simulate_noarch_build_string("20260210", "aa3373a2", "3.11")
    check("noarch string 比 per-python string 短",
          sim["noarch_is_shorter"])
    check("per-python string 含 py311 前缀",
          sim["per_python_has_py_prefix"])
    check("noarch build string 无 py 前缀",
          not sim["noarch_build_string"].startswith("py"))

    # 10. NoarchCIJobSpec
    job = AA3373A_BUILD_NOARCH_JOB
    check("build noarch job 名称正确",
          job.job_name == "python-build-noarch")
    check("build noarch job 依赖 cpp-build",
          "cpp-build" in job.needs)
    check("build noarch job 依赖 python-build",
          "python-build" in job.needs)
    check("build noarch job 阻塞 upload-conda",
          "upload-conda" in job.blocks)
    yaml_frag = job.to_yaml_fragment()
    check("YAML 片段含 pure-conda: true", "pure-conda: true" in yaml_frag)

    # 断点 7：自测汇总
    if _DBG:
        print(f"[DEBUG aa3373a] 自测完成: {passed}/{total}")

    print(f"\n自测结果: {passed}/{total} 通过")
    assert passed == total, f"{total - passed} 项失败"
    print("[PASS] 全部通过\n")


if __name__ == "__main__":
    _self_test()
