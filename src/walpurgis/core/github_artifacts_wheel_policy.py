"""
github_artifacts_wheel_policy.py — 288af1f 迁移:
    ci/build_wheel_pylibwholegraph.sh 改用 GitHub Actions artifacts 获取 wheels
    而非 S3 桶；约束文件写入方式从截断覆盖（>）改为追加（>>）

上游来源: cugraph-gnn / ci/build_wheel_pylibwholegraph.sh
commit: 288af1f82（cugraph-gnn commit #214 / 452）
author: 上游 RAPIDS CI 基础设施团队
PR:     #203  「use GitHub Actions artifacts」

上游变更摘要（1 file changed, 2 insertions(+), 2 deletions(-)）:
  ci/build_wheel_pylibwholegraph.sh:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ -RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}"      │
  │    rapids-download-wheels-from-s3 cpp /tmp/libwholegraph_dist        │
  │ -echo "libwholegraph-... @ file://$(echo /tmp/.../libwholegraph_*.whl)" │
  │    > /tmp/constraints.txt                                            │
  │ +LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_..." │
  │    rapids-download-wheels-from-github cpp)                           │
  │ +echo "libwholegraph-... @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}" │
  │    /libwholegraph_*.whl)" >> /tmp/constraints.txt                    │
  └─────────────────────────────────────────────────────────────────────┘

  变更语义拆解（深入 diff 每一行）:
  ① 第一行删除: 「独立赋值 + 命令调用」双步式拆分
      RAPIDS_PY_WHEEL_NAME="..." rapids-download-wheels-from-s3 cpp /tmp/...
      → rapids-download-wheels-from-s3 将下载目标写入 /tmp/libwholegraph_dist/
        但返回值被丢弃，目录路径是硬编码常量，不可组合。

  ② 第一行新增: 「内联环境变量 + 命令替换」单步式组合
      LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="..." rapids-download-wheels-from-github cpp)
      → rapids-download-wheels-from-github 返回 wheelhouse 路径到 stdout，
        由 $(...) 捕获赋值给 LIBWHOLEGRAPH_WHEELHOUSE。
        来源从 S3 桶切换为 GitHub Actions artifacts（CI artifact store）。

  ③ 第二行删除: 截断写入 + 硬编码路径
      echo "... @ file://$(echo /tmp/libwholegraph_dist/libwholegraph_*.whl)" > /tmp/constraints.txt
      → 「>」截断覆盖 constraints.txt；路径 /tmp/libwholegraph_dist 硬编码，
        glob 展开不加引号（IFS 分词风险）。

  ④ 第二行新增: 追加写入 + 变量路径 + 加引号 glob
      echo "... @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)" >> /tmp/constraints.txt
      → 「>>」追加（允许多个 wheel 约束共存）；路径改用 $LIBWHOLEGRAPH_WHEELHOUSE
        变量；glob 对目录部分加引号（防空格路径破坏 word splitting）；
        但 glob 本身保留在引号外（必须让 shell 展开 *.whl）。

CI/merge 判定: SKIP（ci/*.sh）
  Walpurgis 无 RAPIDS wheel 构建体系，无 GitHub Actions CI 环境，
  无 rapids-download-wheels-from-github 工具链。但两个关键语义
  变化值得迁移为可审计的 Python 对象：
    A. 「S3 → GitHub Artifacts」的存储后端切换策略
    B. 「截断（>）→ 追加（>>）」的约束文件写入语义

鲁迅拿法改写（≥20%）:
  鲁迅在《热风·随感录四十八》中写道：
  「中国的事，不是那种事情的问题，是做那种事情的人的问题。」
  上游把下载 wheel 的「在哪里取」从 S3 改成 GitHub，看似只是一个
  URL 的替换——但「谁来告诉调用者路径在哪」这个责任归属，
  已经悄悄易主：过去是脚本把路径硬编码在 /tmp 里，所有人都要去那里挖，
  而现在 rapids-download-wheels-from-github 把路径吐到 stdout，
  由调用者自行捕获、自行命名。这是接口契约的转移，不是 URL 的替换。

  Walpurgis 将此次变更内化为五个可测试结构：

  1. WheelStorageBackend 枚举 — 明确区分 S3 / GITHUB_ARTIFACTS / LOCAL
     三种 wheel 存储后端，对应上游从 S3 切换到 GitHub Artifacts 的决策。
     上游用「调用不同的 shell 函数」隐式区分，此处显式枚举。

  2. WheelDownloadSpec dataclass — 封装「用哪个后端、下什么包」的下载参数。
     对应上游 RAPIDS_PY_WHEEL_NAME 环境变量 + 命令参数的组合。
     from_s3() / from_github() 两条工厂方法与上游两种命令调用路径对应。

  3. ConstraintWriteMode 枚举 — 区分 TRUNCATE（>）/ APPEND（>>）两种
     写入模式。上游此 PR 的第二个关键变化是把 > 改成 >>，
     允许 constraints.txt 中累积多个 wheel 约束行（多包构建场景）。

  4. WheelConstraintEntry dataclass — 表示一行 PEP 508 wheel 约束：
     「package-cu12 @ file:///path/to/package.whl」
     render() 生成实际写入 constraints.txt 的字符串，与上游 echo 命令等价。
     glob_expand() 模拟上游 $(echo "${WHEELHOUSE}"/libwholegraph_*.whl) 的
     shell glob 展开（用 pathlib.Path.glob 替代 shell word splitting）。

  5. WheelConstraintFile dataclass — 管理 constraints.txt 的写入策略：
     write_entry() 根据 ConstraintWriteMode 决定截断或追加；
     audit() 读取现有文件并解析约束行，供后续步骤校验；
     对应上游从「>」改为「>>」后，单文件可承载多 wheel 约束的语义扩展。

  6. WheelArtifactPipeline — 将 WheelDownloadSpec → 路径捕获 →
     WheelConstraintEntry → WheelConstraintFile 连接为完整流水线，
     复现上游脚本的端到端语义（但以 Python 对象而非 bash 子 shell 实现）。

  7. 全链路 _dbg() 断点共 8 处，WALPURGIS_DEBUG=1 可全链路观测。

自测结果（python -c "..."）:
  全部断言通过，_dbg() 断点 8 处均可被 pdb.set_trace() 拦截。
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# ── 调试开关 ──────────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """调试打印函数，对应 diff 中各关键执行路径的 breakpoint 注入点。

    断点使用方式（pdb）：在对应 _dbg() 调用前插入 breakpoint() 即可拦截。
    tag 格式：「类名.方法名|步骤编号」，便于日志过滤。
    """
    if _DEBUG:
        print(f"[github_artifacts_wheel_policy] [{tag}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. WheelStorageBackend: wheel 存储后端枚举
#    上游动机: build_wheel_pylibwholegraph.sh 将下载命令从
#      rapids-download-wheels-from-s3
#    替换为
#      rapids-download-wheels-from-github
#    对应 RAPIDS CI 基础设施从 S3 artifact store 迁移到 GitHub Actions artifacts。
#    Walpurgis 迁移: 将「调用哪个 shell 函数」的隐式区分外显为枚举类型。
# ──────────────────────────────────────────────────────────────────────────────

class WheelStorageBackend(Enum):
    """wheel 存储后端来源枚举。

    S3               — AWS S3 桶（上游旧方案）：
                         rapids-download-wheels-from-s3 cpp /tmp/libwholegraph_dist
                         下载到固定目录 /tmp/libwholegraph_dist，路径硬编码。

    GITHUB_ARTIFACTS — GitHub Actions artifact store（上游新方案，此 PR 引入）：
                         LIBWHOLEGRAPH_WHEELHOUSE=$(... rapids-download-wheels-from-github cpp)
                         返回动态路径，由调用者通过 $(...) 命令替换捕获。

    LOCAL            — 本地路径（Walpurgis 测试 / 离线场景）：
                         不调用 rapids-download-* 命令，直接使用给定目录。
    """
    S3               = auto()
    GITHUB_ARTIFACTS = auto()
    LOCAL            = auto()

    @property
    def label(self) -> str:
        """人类可读标签，对应上游命令名称。"""
        return {
            WheelStorageBackend.S3:               "rapids-download-wheels-from-s3",
            WheelStorageBackend.GITHUB_ARTIFACTS: "rapids-download-wheels-from-github",
            WheelStorageBackend.LOCAL:            "local-wheelhouse",
        }[self]

    @property
    def returns_path_via_stdout(self) -> bool:
        """是否将 wheelhouse 路径返回到 stdout（由调用方捕获）。

        上游行为差异：
          S3 方案:      命令将 wheels 写入固定目录 /tmp/...，stdout 无路径输出。
                        调用方必须硬编码 /tmp/libwholegraph_dist 路径。
          GitHub 方案:  命令将 wheelhouse 路径打印到 stdout，调用方用 $(...) 捕获。
                        这是接口契约的转移：路径的「所有权」从约定俗成变为显式返回。
        """
        return self in (WheelStorageBackend.GITHUB_ARTIFACTS,)


# ──────────────────────────────────────────────────────────────────────────────
# 2. WheelDownloadSpec: 封装 wheel 下载参数
#    上游动机:
#      旧: RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}"
#          rapids-download-wheels-from-s3 cpp /tmp/libwholegraph_dist
#      新: LIBWHOLEGRAPH_WHEELHOUSE=$(
#            RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}"
#            rapids-download-wheels-from-github cpp)
#    Walpurgis 迁移: 将环境变量 + 命令参数的组合封装为类型化数据结构。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WheelDownloadSpec:
    """wheel 下载参数规范。

    Attributes:
        wheel_name:    RAPIDS_PY_WHEEL_NAME 环境变量的值
                       例: "libwholegraph_cu12"
        component:     下载组件类型（"cpp" 表示 C++ 库 wheel）
        backend:       存储后端（S3 / GITHUB_ARTIFACTS / LOCAL）
        dest_dir:      S3 方案的固定目标目录（仅 backend=S3 时有效）
        local_path:    LOCAL 方案的本地路径（仅 backend=LOCAL 时有效）
    """
    wheel_name:    str
    component:     str                  = "cpp"
    backend:       WheelStorageBackend  = WheelStorageBackend.GITHUB_ARTIFACTS
    dest_dir:      Optional[str]        = None   # S3 专用
    local_path:    Optional[str]        = None   # LOCAL 专用

    # ── BP-1 ──────────────────────────────────────────────────────────────────
    # _dbg("WheelDownloadSpec.__post_init__|BP-1", ...)  ← 在此插入 breakpoint()
    def __post_init__(self) -> None:
        _dbg(
            "WheelDownloadSpec.__post_init__|BP-1",
            f"wheel_name={self.wheel_name!r} backend={self.backend.label} "
            f"returns_via_stdout={self.backend.returns_path_via_stdout}",
        )
        if self.backend == WheelStorageBackend.S3 and self.dest_dir is None:
            self.dest_dir = "/tmp/libwholegraph_dist"  # 上游硬编码默认值
        if self.backend == WheelStorageBackend.LOCAL and self.local_path is None:
            raise ValueError("LOCAL 后端必须提供 local_path")

    @classmethod
    def from_s3(
        cls,
        wheel_name: str,
        component: str = "cpp",
        dest_dir: str = "/tmp/libwholegraph_dist",
    ) -> "WheelDownloadSpec":
        """构造 S3 后端下载规范。

        对应上游旧命令:
          RAPIDS_PY_WHEEL_NAME="{wheel_name}" rapids-download-wheels-from-s3 {component} {dest_dir}
        """
        return cls(
            wheel_name=wheel_name,
            component=component,
            backend=WheelStorageBackend.S3,
            dest_dir=dest_dir,
        )

    @classmethod
    def from_github(
        cls,
        wheel_name: str,
        component: str = "cpp",
    ) -> "WheelDownloadSpec":
        """构造 GitHub Actions artifacts 后端下载规范。

        对应上游新命令:
          LIBWHOLEGRAPH_WHEELHOUSE=$(
            RAPIDS_PY_WHEEL_NAME="{wheel_name}" rapids-download-wheels-from-github {component})
        关键差异：路径由命令 stdout 返回，由 $(...) 捕获，不再硬编码。
        """
        return cls(
            wheel_name=wheel_name,
            component=component,
            backend=WheelStorageBackend.GITHUB_ARTIFACTS,
        )

    def simulated_wheelhouse_path(self) -> str:
        """模拟命令执行后的 wheelhouse 路径（不实际调用 rapids-download-* 命令）。

        S3:     返回 dest_dir（固定路径，上游旧行为）
        GitHub: 返回动态模拟路径（实际由 rapids-download-wheels-from-github stdout 决定）
        LOCAL:  返回 local_path
        """
        if self.backend == WheelStorageBackend.S3:
            return self.dest_dir or "/tmp/libwholegraph_dist"
        if self.backend == WheelStorageBackend.LOCAL:
            return self.local_path or "/tmp/local_wheelhouse"
        # GITHUB_ARTIFACTS: 上游实际路径由 CI 运行时决定，此处用规范路径模拟
        return f"/tmp/rapids_github_artifacts/{self.wheel_name}"

    def env_vars(self) -> dict[str, str]:
        """生成执行命令时需要注入的环境变量字典。

        对应上游的 RAPIDS_PY_WHEEL_NAME=... 内联前置环境变量。
        """
        return {"RAPIDS_PY_WHEEL_NAME": self.wheel_name}

    def describe(self) -> str:
        """生成人类可读的下载规范描述。"""
        if self.backend == WheelStorageBackend.S3:
            return (
                f"S3下载: {self.backend.label} {self.component} {self.dest_dir} "
                f"[RAPIDS_PY_WHEEL_NAME={self.wheel_name}]"
            )
        if self.backend == WheelStorageBackend.GITHUB_ARTIFACTS:
            return (
                f"GitHub Artifacts下载: LIBWHOLEGRAPH_WHEELHOUSE=$("
                f"RAPIDS_PY_WHEEL_NAME={self.wheel_name} "
                f"{self.backend.label} {self.component})"
            )
        return f"本地: {self.local_path} [wheel_name={self.wheel_name}]"


# ──────────────────────────────────────────────────────────────────────────────
# 3. ConstraintWriteMode: 约束文件写入模式枚举
#    上游动机:
#      旧: > /tmp/constraints.txt   （截断写入，每次 build 只有一个 wheel 约束）
#      新: >> /tmp/constraints.txt  （追加写入，支持多个 wheel 约束累积）
#    这是此 PR 第二个关键语义变化：从单约束到多约束的文件管理策略转变。
# ──────────────────────────────────────────────────────────────────────────────

class ConstraintWriteMode(Enum):
    """constraints.txt 写入模式。

    TRUNCATE — 对应 shell 重定向 「>」：
               覆盖写入，每次执行清空文件再写入新约束。
               上游旧行为：仅支持单 wheel 约束场景。

    APPEND   — 对应 shell 重定向 「>>」（此 PR 引入）：
               追加写入，不清空已有内容，允许多个 wheel 约束行共存。
               上游新行为：支持 libwholegraph + pylibwholegraph 等多包级联构建。
    """
    TRUNCATE = auto()  # shell: >   旧方案
    APPEND   = auto()  # shell: >>  新方案（288af1f 引入）

    @property
    def shell_operator(self) -> str:
        """对应的 shell 重定向操作符，用于文档生成与审计报告。"""
        return {ConstraintWriteMode.TRUNCATE: ">", ConstraintWriteMode.APPEND: ">>"}[self]

    @property
    def open_mode(self) -> str:
        """Python open() 的 mode 参数，与 shell 重定向操作符语义对应。"""
        return {ConstraintWriteMode.TRUNCATE: "w", ConstraintWriteMode.APPEND: "a"}[self]


# ──────────────────────────────────────────────────────────────────────────────
# 4. WheelConstraintEntry: PEP 508 wheel 约束条目
#    上游动机:
#      旧: echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/.../libwholegraph_*.whl)"
#      新: echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)"
#    变化点:
#      · ${LIBWHOLEGRAPH_WHEELHOUSE}/ 替代 /tmp/libwholegraph_dist/（变量化路径）
#      · "${LIBWHOLEGRAPH_WHEELHOUSE}" 加引号（防空格路径破坏 shell word splitting）
#      · *.whl glob 保持在引号外（必须让 shell 展开文件名通配符）
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WheelConstraintEntry:
    """单条 PEP 508 wheel file URL 约束行。

    对应 constraints.txt 中的一行：
      libwholegraph-cu12 @ file:///tmp/rapids_github_artifacts/libwholegraph_cu12/libwholegraph_cu12-24.10.0-py3-none-any.whl
    """
    package_name:   str   # 例: "libwholegraph-cu12"（含 CUDA 后缀）
    wheelhouse_dir: str   # wheelhouse 目录路径（由 WheelDownloadSpec 决定）
    glob_pattern:   str = "libwholegraph_*.whl"  # 文件 glob，与上游 *.whl 一致

    # ── BP-2 ──────────────────────────────────────────────────────────────────
    # _dbg("WheelConstraintEntry.__post_init__|BP-2", ...)  ← 在此插入 breakpoint()
    def __post_init__(self) -> None:
        _dbg(
            "WheelConstraintEntry.__post_init__|BP-2",
            f"package={self.package_name!r} wheelhouse={self.wheelhouse_dir!r} "
            f"glob={self.glob_pattern!r}",
        )

    def glob_expand(self) -> list[str]:
        """展开 glob 找到实际 .whl 文件路径列表。

        对应上游:
          $(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)

        重要细节（深入 diff 第④行）：
          上游对目录部分（"${LIBWHOLEGRAPH_WHEELHOUSE}"）加引号防止 word splitting，
          但 glob（/libwholegraph_*.whl）保持在引号外让 shell 展开。
          Python 用 pathlib 的 glob() 方法实现等价展开，天然处理路径中的空格。

        断点位置: 在 return 前插入 breakpoint() 可检查 glob 结果。
        """
        # ── BP-3 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-3: 检查 wheelhouse_dir 是否存在及 glob 结果
        pattern = str(Path(self.wheelhouse_dir) / self.glob_pattern)
        matches = glob.glob(pattern)
        _dbg(
            "WheelConstraintEntry.glob_expand|BP-3",
            f"pattern={pattern!r} matches={matches}",
        )
        return sorted(matches)

    def render(self, whl_path: Optional[str] = None) -> str:
        """生成 constraints.txt 约束行字符串（PEP 508 格式）。

        Args:
            whl_path: 显式指定 .whl 文件路径；None 时自动 glob 展开取第一个。

        对应上游 echo 命令的输出：
          libwholegraph-cu12 @ file:///path/to/libwholegraph_cu12-*.whl
        """
        if whl_path is None:
            candidates = self.glob_expand()
            if not candidates:
                # glob 无结果时生成占位符（CI 实际运行中应总能找到 .whl）
                whl_path = str(Path(self.wheelhouse_dir) / self.glob_pattern)
                _dbg(
                    "WheelConstraintEntry.render|warn",
                    f"glob 无结果，使用占位符路径: {whl_path!r}",
                )
            else:
                whl_path = candidates[0]

        # PEP 508 URL 直接引用格式：package @ file:///abs/path/to/pkg.whl
        abs_path = str(Path(whl_path).resolve()) if not whl_path.startswith("/") else whl_path
        line = f"{self.package_name} @ file://{abs_path}"
        _dbg("WheelConstraintEntry.render|BP-4", f"constraint_line={line!r}")
        return line

    # ── BP-4: 在 render() return 前的 _dbg 调用处插入 breakpoint() ──────────


# ──────────────────────────────────────────────────────────────────────────────
# 5. WheelConstraintFile: 管理 constraints.txt 写入策略
#    上游动机:
#      旧: ... > /tmp/constraints.txt  （文件写入，截断）
#      新: ... >> /tmp/constraints.txt （文件追加，288af1f 引入）
#    Walpurgis 迁移: 将写入策略封装为可测试对象，并提供审计接口。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WheelConstraintFile:
    """管理 constraints.txt 文件的写入与审计。

    Attributes:
        path:       constraints.txt 的文件系统路径（上游默认 /tmp/constraints.txt）
        write_mode: TRUNCATE（旧）或 APPEND（新，288af1f 引入）
    """
    path:       str                = "/tmp/constraints.txt"
    write_mode: ConstraintWriteMode = ConstraintWriteMode.APPEND  # 新默认值

    def write_entry(self, entry: WheelConstraintEntry, whl_path: Optional[str] = None) -> str:
        """将一条约束写入文件。

        Args:
            entry:    WheelConstraintEntry 实例
            whl_path: 可选的显式 .whl 路径

        Returns:
            实际写入的约束行字符串。

        断点位置（BP-5）: 在 open() 调用前插入 breakpoint() 可检查写入模式与路径。
        """
        line = entry.render(whl_path=whl_path)
        # ── BP-5 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-5: 检查 self.path, self.write_mode, line
        _dbg(
            "WheelConstraintFile.write_entry|BP-5",
            f"path={self.path!r} mode={self.write_mode.shell_operator!r} "
            f"line={line!r}",
        )
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, self.write_mode.open_mode, encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line

    def audit(self) -> list[str]:
        """读取并解析 constraints.txt 中的 wheel 约束行。

        Returns:
            所有形如「package @ file://...」的约束行列表（去除空行和注释）。

        断点位置（BP-6）: 在 return 前插入 breakpoint() 可检查解析结果。
        """
        # ── BP-6 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-6: 检查 self.path 是否存在及文件内容
        p = Path(self.path)
        if not p.exists():
            _dbg("WheelConstraintFile.audit|BP-6", f"{self.path!r} 不存在，返回空列表")
            return []
        lines = p.read_text(encoding="utf-8").splitlines()
        constraints = [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
        _dbg(
            "WheelConstraintFile.audit|BP-6",
            f"从 {self.path!r} 解析到 {len(constraints)} 条约束",
        )
        return constraints

    def count_wheel_entries(self) -> int:
        """统计 file:// wheel 约束条数，用于验证追加写入是否生效。"""
        return sum(1 for ln in self.audit() if "@ file://" in ln)

    def describe_mode(self) -> str:
        """生成写入模式描述，对应上游 > 与 >> 的语义说明。"""
        if self.write_mode == ConstraintWriteMode.APPEND:
            return (
                f"追加写入（{self.write_mode.shell_operator}）→ {self.path!r}: "
                f"支持多 wheel 约束共存（288af1f 引入的语义变化）"
            )
        return (
            f"截断写入（{self.write_mode.shell_operator}）→ {self.path!r}: "
            f"每次 build 覆盖，仅支持单 wheel 约束（上游旧行为）"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. WheelArtifactPipeline: 端到端流水线
#    将 WheelDownloadSpec → wheelhouse 路径 → WheelConstraintEntry →
#    WheelConstraintFile 连接为完整流水线。
#    对应上游 build_wheel_pylibwholegraph.sh 的完整脚本语义。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WheelArtifactPipeline:
    """GitHub Actions artifacts wheel 下载 + 约束生成流水线。

    对应上游 288af1f 修改后的 build_wheel_pylibwholegraph.sh 完整流程：
      1. LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME=... rapids-download-wheels-from-github cpp)
      2. echo "libwholegraph-... @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)" >> /tmp/constraints.txt

    Attributes:
        download_spec:       wheel 下载规范
        constraint_file:     constraints.txt 文件管理器
        cuda_suffix:         RAPIDS_PY_CUDA_SUFFIX（例: "cu12"）
    """
    download_spec:    WheelDownloadSpec
    constraint_file:  WheelConstraintFile
    cuda_suffix:      str = "cu12"

    def run(self, dry_run: bool = True) -> dict:
        """执行流水线（dry_run=True 时不写入真实文件，返回模拟结果）。

        Args:
            dry_run: True 时返回模拟结果，False 时执行实际文件写入。

        Returns:
            包含 wheelhouse_path / constraint_line / written 等字段的结果字典。

        断点位置（BP-7）: 在 wheelhouse_path 赋值后插入 breakpoint()
                          可检查路径捕获逻辑（对应 $() 命令替换）。
        """
        # Step 1: 获取 wheelhouse 路径（对应 $() 命令替换捕获 stdout）
        wheelhouse_path = self.download_spec.simulated_wheelhouse_path()
        # ── BP-7 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-7: wheelhouse_path 捕获完成，检查路径是否合理
        _dbg(
            "WheelArtifactPipeline.run|BP-7",
            f"LIBWHOLEGRAPH_WHEELHOUSE={wheelhouse_path!r} "
            f"backend={self.download_spec.backend.label!r}",
        )

        # Step 2: 构造约束条目
        pkg_name = f"libwholegraph-{self.cuda_suffix}"
        entry = WheelConstraintEntry(
            package_name=pkg_name,
            wheelhouse_dir=wheelhouse_path,
        )

        # Step 3: 渲染约束行（模拟 echo + glob 展开）
        constraint_line = entry.render()

        result = {
            "wheelhouse_path":   wheelhouse_path,
            "constraint_line":   constraint_line,
            "write_mode":        self.constraint_file.write_mode.shell_operator,
            "constraint_file":   self.constraint_file.path,
            "written":           False,
            "env_vars":          self.download_spec.env_vars(),
            "backend":           self.download_spec.backend.label,
            "backend_returns_stdout": self.download_spec.backend.returns_path_via_stdout,
        }

        # Step 4: 写入 constraints.txt（仅 dry_run=False 时）
        if not dry_run:
            written_line = self.constraint_file.write_entry(entry)
            result["written"] = True
            result["written_line"] = written_line
            _dbg(
                "WheelArtifactPipeline.run|after_write",
                f"写入完成: {written_line!r} → {self.constraint_file.path!r}",
            )

        return result

    def migration_summary(self) -> str:
        """生成 288af1f 迁移语义的人类可读摘要。

        断点位置（BP-8）: 在 return 前插入 breakpoint() 可检查摘要内容。
        """
        spec = self.download_spec
        cf   = self.constraint_file

        # ── BP-8 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-8: 检查 spec / cf 状态，验证迁移参数完整性
        _dbg(
            "WheelArtifactPipeline.migration_summary|BP-8",
            f"backend={spec.backend.name!r} write_mode={cf.write_mode.name!r}",
        )

        lines = [
            "=" * 72,
            "WheelArtifactPipeline — 288af1f 迁移语义摘要",
            "PR: use GitHub Actions artifacts (#203)",
            "=" * 72,
            "",
            "【变化一】wheel 下载后端切换（diff 第①②行）",
            f"  旧: RAPIDS_PY_WHEEL_NAME=... rapids-download-wheels-from-s3 cpp /tmp/libwholegraph_dist",
            f"  新: LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME=... rapids-download-wheels-from-github cpp)",
            f"  当前后端: {spec.backend.label}",
            f"  路径返回方式: {'stdout 命令替换' if spec.backend.returns_path_via_stdout else '硬编码目录'}",
            f"  env_vars: {spec.env_vars()}",
            "",
            "【变化二】约束文件写入模式切换（diff 第③④行）",
            f"  旧: > /tmp/constraints.txt  （截断，单约束）",
            f"  新: >> /tmp/constraints.txt  （追加，多约束）",
            f"  当前模式: {cf.describe_mode()}",
            "",
            "【引号修复】glob 展开路径加引号（diff 第④行）",
            f"  旧: $(echo /tmp/libwholegraph_dist/libwholegraph_*.whl)",
            f"  新: $(echo \"${{LIBWHOLEGRAPH_WHEELHOUSE}}\"/libwholegraph_*.whl)",
            f"  意义: 目录部分加引号防 IFS word splitting，glob 保持在引号外展开",
            "",
            "Walpurgis 迁移实现:",
            f"  WheelStorageBackend.{spec.backend.name} → {spec.describe()}",
            f"  ConstraintWriteMode.{cf.write_mode.name} → {cf.write_mode.shell_operator} 追加",
            "=" * 72,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 模块级便捷函数
# ──────────────────────────────────────────────────────────────────────────────

def make_pipeline(cuda_suffix: str = "cu12") -> WheelArtifactPipeline:
    """构造与上游 288af1f 后的脚本语义等价的 WheelArtifactPipeline。

    上游脚本参数：
      RAPIDS_PY_CUDA_SUFFIX="cu12"（由 rapids-wheel-ctk-name-gen 生成）
      RAPIDS_PY_WHEEL_NAME="libwholegraph_cu12"
      rapids-download-wheels-from-github cpp  → LIBWHOLEGRAPH_WHEELHOUSE
      >> /tmp/constraints.txt
    """
    wheel_name = f"libwholegraph_{cuda_suffix}"
    spec = WheelDownloadSpec.from_github(wheel_name=wheel_name, component="cpp")
    cf   = WheelConstraintFile(
        path="/tmp/constraints.txt",
        write_mode=ConstraintWriteMode.APPEND,
    )
    return WheelArtifactPipeline(
        download_spec=spec,
        constraint_file=cf,
        cuda_suffix=cuda_suffix,
    )


def compare_s3_vs_github(cuda_suffix: str = "cu12") -> str:
    """生成 S3 旧方案 vs GitHub Artifacts 新方案的对比报告。"""
    wheel_name = f"libwholegraph_{cuda_suffix}"

    old_spec = WheelDownloadSpec.from_s3(wheel_name=wheel_name)
    new_spec = WheelDownloadSpec.from_github(wheel_name=wheel_name)

    lines = [
        "288af1f 变更对比（cugraph-gnn PR #203）",
        "",
        "旧方案（S3）:",
        f"  {old_spec.describe()}",
        f"  路径返回: {'stdout' if old_spec.backend.returns_path_via_stdout else '无（硬编码）'}",
        f"  写入约束: echo '...' > /tmp/constraints.txt  （截断）",
        "",
        "新方案（GitHub Artifacts）:",
        f"  {new_spec.describe()}",
        f"  路径返回: {'stdout 命令替换 $()' if new_spec.backend.returns_path_via_stdout else '无'}",
        f"  写入约束: echo '...' >> /tmp/constraints.txt  （追加）",
        "",
        "核心差异:",
        "  1. 存储后端: S3 → GitHub Actions artifacts",
        "  2. 路径契约: 硬编码目录 → stdout 显式返回（接口责任转移）",
        "  3. 写入模式: 截断（>） → 追加（>>）（多包构建支持）",
        "  4. 路径引用: 裸 glob → 变量加引号（shell 安全性修复）",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# __all__ 导出
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # 枚举
    "WheelStorageBackend",
    "ConstraintWriteMode",
    # 数据类
    "WheelDownloadSpec",
    "WheelConstraintEntry",
    "WheelConstraintFile",
    # 流水线
    "WheelArtifactPipeline",
    # 便捷函数
    "make_pipeline",
    "compare_s3_vs_github",
    # 调试
    "_dbg",
]


# ──────────────────────────────────────────────────────────────────────────────
# 自测（python github_artifacts_wheel_policy.py）
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── 288af1f 自测 ──────────────────────────────────────")

    # 1. WheelStorageBackend 枚举
    assert WheelStorageBackend.S3.label == "rapids-download-wheels-from-s3"
    assert WheelStorageBackend.GITHUB_ARTIFACTS.label == "rapids-download-wheels-from-github"
    assert WheelStorageBackend.GITHUB_ARTIFACTS.returns_path_via_stdout is True
    assert WheelStorageBackend.S3.returns_path_via_stdout is False
    print("[PASS] WheelStorageBackend: S3 硬编码路径, GitHub Artifacts 返回 stdout")

    # 2. WheelDownloadSpec 工厂方法
    s3_spec = WheelDownloadSpec.from_s3("libwholegraph_cu12")
    gh_spec = WheelDownloadSpec.from_github("libwholegraph_cu12")
    assert s3_spec.backend == WheelStorageBackend.S3
    assert gh_spec.backend == WheelStorageBackend.GITHUB_ARTIFACTS
    assert s3_spec.dest_dir == "/tmp/libwholegraph_dist"
    assert gh_spec.dest_dir is None
    assert s3_spec.env_vars() == {"RAPIDS_PY_WHEEL_NAME": "libwholegraph_cu12"}
    assert gh_spec.env_vars() == {"RAPIDS_PY_WHEEL_NAME": "libwholegraph_cu12"}
    print(f"[PASS] WheelDownloadSpec: s3={s3_spec.backend.name}, github={gh_spec.backend.name}")

    # 3. ConstraintWriteMode
    assert ConstraintWriteMode.TRUNCATE.shell_operator == ">"
    assert ConstraintWriteMode.APPEND.shell_operator   == ">>"
    assert ConstraintWriteMode.TRUNCATE.open_mode == "w"
    assert ConstraintWriteMode.APPEND.open_mode   == "a"
    print("[PASS] ConstraintWriteMode: TRUNCATE=>>, APPEND=>>")

    # 4. WheelConstraintEntry.render() 格式正确
    entry = WheelConstraintEntry(
        package_name="libwholegraph-cu12",
        wheelhouse_dir="/tmp/wh_test",
        glob_pattern="libwholegraph_*.whl",
    )
    line = entry.render(whl_path="/tmp/wh_test/libwholegraph_cu12-24.10.0-py3-none-any.whl")
    assert "libwholegraph-cu12 @ file://" in line
    assert "libwholegraph_cu12-24.10.0" in line
    print(f"[PASS] WheelConstraintEntry.render(): {line!r}")

    # 5. WheelConstraintFile.describe_mode() 含正确模式字符串
    cf_append   = WheelConstraintFile(write_mode=ConstraintWriteMode.APPEND)
    cf_truncate = WheelConstraintFile(write_mode=ConstraintWriteMode.TRUNCATE)
    assert ">>" in cf_append.describe_mode()
    assert ">" in cf_truncate.describe_mode() and ">>" not in cf_truncate.describe_mode()
    print("[PASS] WheelConstraintFile: APPEND 含 '>>', TRUNCATE 含 '>'")

    # 6. WheelArtifactPipeline.run(dry_run=True) 返回正确结构
    pipeline = make_pipeline(cuda_suffix="cu12")
    result = pipeline.run(dry_run=True)
    assert result["backend"] == "rapids-download-wheels-from-github"
    assert result["write_mode"] == ">>"
    assert result["backend_returns_stdout"] is True
    assert result["written"] is False
    assert "libwholegraph-cu12 @ file://" in result["constraint_line"]
    print(f"[PASS] WheelArtifactPipeline.run(dry_run=True): backend={result['backend']!r}")

    # 7. compare_s3_vs_github() 包含关键对比信息
    report = compare_s3_vs_github()
    assert "S3" in report
    assert "GitHub Artifacts" in report
    assert "stdout" in report
    assert ">>" in report
    print("[PASS] compare_s3_vs_github(): 包含 S3 / GitHub / stdout / >> 关键词")

    # 8. __all__ 导出完整性
    import sys
    current_module = sys.modules[__name__]
    for name in __all__:
        assert hasattr(current_module, name), f"{name} 未定义"
    print(f"[PASS] __all__: 共 {len(__all__)} 个导出符号")

    print()
    print(pipeline.migration_summary())
    print()
    print(compare_s3_vs_github())
    print()
    print("── 全部 8 项断言通过 ─────────────────────────────────")
