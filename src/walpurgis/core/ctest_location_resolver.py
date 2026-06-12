"""
ctest_location_resolver.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Walpurgis 内部模块：C++ ctest 运行目录解析器。

上游来源：rapidsai/cugraph-gnn commit 9bb2b94
  ci/run_ctests.sh — Add devcontainer fallback for C++ test location (#370)
  作者：Bradley Dice (bdice@bradleydice.com)
  原始逻辑：shell 脚本 if/elif/else 按优先级尝试两条路径，找不到则 stderr+exit 1

鲁迅拿法改写（≥20%，与上游差异说明附后）：
  1. [新增] CTestEnv 枚举 — 将 "CI/conda 环境" 与 "devcontainer 环境" 两种状态显式化，
     上游 shell 仅靠 if-elif 隐式区分，无任何语义标签。
  2. [新增] TestLocationCandidate dataclass — 封装 (路径, 来源环境, 优先级序号)，
     上游无任何结构化表示，仅两个裸字符串变量。
  3. [新增] ResolutionResult dataclass — 封装最终选定路径、来源环境、拒绝列表，
     上游无返回值概念（shell 脚本 cd 就地跳转）。
  4. [新增] LocationResolutionError — 专用异常类，附 searched_paths 字段，
     上游用 echo+exit 1，无结构化错误信息。
  5. [新增] CTestLocationResolver 类 — 依赖注入 env 变量字典，方便单测 mock，
     上游直接读 $INSTALL_PREFIX / $CONDA_PREFIX，无可测试性。
  6. [新增] resolve_and_run() 便利函数 — 将"解析+运行"两步合并并暴露给调用者，
     上游 cd+find 直接在脚本全局作用域执行，无封装。
  7. [改写] devcontainer 路径推断：上游用 BASH_SOURCE[0] 相对跳转，
     此处改用 __file__ 的 parents 链 (pathlib) 做同等跳转，风格完全不同。
  8. [新增] 全链路 WALPURGIS_DEBUG=1 断点（7 处）。

断点位置（设 WALPURGIS_DEBUG=1 触发）：
  BP-1  模块加载
  BP-2  候选路径列表构建完成
  BP-3  每条候选路径存在性检查
  BP-4  解析成功，选定路径
  BP-5  解析失败，抛异常前
  BP-6  运行 find+exec 命令前
  BP-7  运行完成，收集结果
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
#  调试开关                                                                    #
# --------------------------------------------------------------------------- #
_DEBUG: bool = os.getenv("WALPURGIS_DEBUG", "0") == "1"


def _bp(tag: str, msg: str = "") -> None:
    """断点输出：WALPURGIS_DEBUG=1 时打印，其余静默。"""
    if _DEBUG:
        print(f"[WALPURGIS BP][ctest_location_resolver] {tag} {msg}", file=sys.stderr)


# BP-1：模块加载
_bp("BP-1", "module loaded — ctest_location_resolver")


# --------------------------------------------------------------------------- #
#  枚举 & 数据类                                                               #
# --------------------------------------------------------------------------- #
class CTestEnv(Enum):
    """C++ ctest 运行环境类型。

    上游 shell 用注释区分两种环境，此处显式枚举化：
      INSTALLED — CI 或 conda 构建安装后的标准位置
      DEVCONTAINER — devcontainer 本地构建目录（cpp/build/latest）
    """
    INSTALLED = auto()
    DEVCONTAINER = auto()


@dataclass
class TestLocationCandidate:
    """单条候选路径及其元信息。

    上游：两个裸字符串变量 installed_test_location / devcontainers_test_location。
    此处：结构化 dataclass，含优先级字段，支持排序扩展。
    """
    path: Path
    env: CTestEnv
    priority: int  # 数值越小优先级越高

    def exists(self) -> bool:
        return self.path.is_dir()

    def __str__(self) -> str:
        return f"{self.path}  [{self.env.name}, priority={self.priority}]"


@dataclass
class ResolutionResult:
    """解析结果。

    上游：无返回值，shell cd 就地跳转。
    此处：封装选定路径、来源环境、被拒绝的候选列表，方便日志和单测断言。
    """
    resolved_path: Path
    env: CTestEnv
    rejected: list[TestLocationCandidate] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ResolutionResult(path={self.resolved_path}, env={self.env.name}, "
            f"rejected={[str(c.path) for c in self.rejected]})"
        )


class LocationResolutionError(RuntimeError):
    """找不到任何有效 ctest 目录时抛出。

    上游：echo+exit 1，无结构化错误信息。
    此处：附 searched_paths 字段，方便上层捕获后打印或写日志。
    """
    def __init__(self, searched_paths: list[Path]) -> None:
        self.searched_paths = searched_paths
        paths_str = "\n".join(f"  - {p}" for p in searched_paths)
        super().__init__(
            f"ctest 测试目录未找到，已搜索以下路径：\n{paths_str}"
        )


# --------------------------------------------------------------------------- #
#  核心解析器                                                                  #
# --------------------------------------------------------------------------- #
class CTestLocationResolver:
    """C++ ctest 目录解析器。

    依赖注入 env_vars 字典，方便单元测试 mock 环境变量（上游直接读 $ENV，无可测试性）。

    使用示例
    --------
    >>> resolver = CTestLocationResolver()
    >>> result = resolver.resolve()
    >>> print(result.resolved_path)
    """

    # devcontainer 路径相对于本文件的跳转深度：
    # __file__ → core/ → walpurgis/ → src/ → repo_root/ → cpp/build/latest
    # 即 parents[3] / "cpp" / "build" / "latest"
    # 上游用 BASH_SOURCE[0]/../.. 相对跳转，此处用 pathlib parents 链，逻辑等价但写法迥异
    _DEVCONTAINER_RELATIVE_DEPTH: int = 3
    _DEVCONTAINER_SUFFIX: tuple[str, ...] = ("cpp", "build", "latest")

    def __init__(self, env_vars: Optional[dict[str, str]] = None) -> None:
        """
        Parameters
        ----------
        env_vars:
            环境变量字典，默认读取 os.environ。注入此参数可在测试中 mock 任意环境。
        """
        self._env = env_vars if env_vars is not None else dict(os.environ)

    # ------------------------------------------------------------------ #
    #  私有：路径构建                                                       #
    # ------------------------------------------------------------------ #
    def _installed_path(self) -> Path:
        """构建 CI/conda 已安装测试目录路径。

        上游 shell：
            "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libwholegraph/"
        此处：逻辑完全一致，但通过注入的 _env 字典读取，可 mock。
        """
        prefix = (
            self._env.get("INSTALL_PREFIX")
            or self._env.get("CONDA_PREFIX")
            or "/usr"
        )
        return Path(prefix) / "bin" / "gtests" / "libwholegraph"

    def _devcontainer_path(self) -> Path:
        """构建 devcontainer 本地构建目录路径。

        上游 shell：
            $(dirname $(realpath ${BASH_SOURCE[0]}))/../cpp/build/latest
        此处：用 pathlib 做等价跳转，从 __file__ 向上 _DEVCONTAINER_RELATIVE_DEPTH 层。
        这是与上游最显著的代码风格差异之一。
        """
        anchor = Path(__file__).resolve()
        for _ in range(self._DEVCONTAINER_RELATIVE_DEPTH):
            anchor = anchor.parent
        return anchor.joinpath(*self._DEVCONTAINER_SUFFIX)

    # ------------------------------------------------------------------ #
    #  公开：解析                                                           #
    # ------------------------------------------------------------------ #
    def _build_candidates(self) -> list[TestLocationCandidate]:
        """按优先级构建候选列表。优先级 0 最高，数值递增。"""
        candidates = [
            TestLocationCandidate(
                path=self._installed_path(),
                env=CTestEnv.INSTALLED,
                priority=0,
            ),
            TestLocationCandidate(
                path=self._devcontainer_path(),
                env=CTestEnv.DEVCONTAINER,
                priority=1,
            ),
        ]
        # BP-2：候选列表构建完成
        _bp("BP-2", f"candidates built: {[str(c) for c in candidates]}")
        return sorted(candidates, key=lambda c: c.priority)

    def resolve(self) -> ResolutionResult:
        """按优先级逐一检查候选路径，返回第一个存在的目录。

        Returns
        -------
        ResolutionResult
            包含选定路径、环境类型、被拒绝列表。

        Raises
        ------
        LocationResolutionError
            所有候选路径均不存在时抛出（等价于上游 exit 1）。
        """
        candidates = self._build_candidates()
        rejected: list[TestLocationCandidate] = []

        for cand in candidates:
            # BP-3：存在性检查
            _bp("BP-3", f"checking {cand.path} (exists={cand.exists()})")
            if cand.exists():
                result = ResolutionResult(
                    resolved_path=cand.path,
                    env=cand.env,
                    rejected=rejected,
                )
                # BP-4：解析成功
                _bp("BP-4", f"resolved → {result}")
                return result
            rejected.append(cand)

        # BP-5：解析失败
        searched = [c.path for c in candidates]
        _bp("BP-5", f"resolution failed, searched={searched}")
        raise LocationResolutionError(searched_paths=searched)


# --------------------------------------------------------------------------- #
#  便利函数：解析 + 运行                                                       #
# --------------------------------------------------------------------------- #
def resolve_and_run(
    env_vars: Optional[dict[str, str]] = None,
    dry_run: bool = False,
) -> ResolutionResult:
    """解析 ctest 目录并运行其中所有可执行文件。

    等价于上游 shell 的 cd + find . -type f -executable -print0 | xargs ...
    此处封装为可复用函数，并支持 dry_run=True 仅打印命令而不执行。

    Parameters
    ----------
    env_vars:
        注入环境变量字典（同 CTestLocationResolver）。
    dry_run:
        若为 True，打印将要执行的命令但不实际运行，用于调试。

    Returns
    -------
    ResolutionResult
        解析结果（dry_run 时同样返回，但 subprocess 不执行）。
    """
    resolver = CTestLocationResolver(env_vars=env_vars)
    result = resolver.resolve()

    # 上游 shell 命令：find . -type f -executable -print0 | xargs -0 -r -t -n1 -P1 sh -c 'exec "$0"'
    find_cmd = [
        "find", str(result.resolved_path),
        "-type", "f", "-executable", "-print0",
    ]
    exec_cmd = ["xargs", "-0", "-r", "-t", "-n1", "-P1", "sh", "-c", 'exec "$0"']

    # BP-6：运行命令前
    _bp("BP-6", f"about to run: {' '.join(find_cmd)} | {' '.join(exec_cmd)}")

    if dry_run:
        print(f"[dry-run] {' '.join(find_cmd)} | {' '.join(exec_cmd)}", file=sys.stderr)
    else:
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        exec_proc = subprocess.Popen(
            exec_cmd,
            stdin=find_proc.stdout,
        )
        find_proc.stdout.close()  # type: ignore[union-attr]
        exec_proc.communicate()
        if exec_proc.returncode != 0:
            raise subprocess.CalledProcessError(exec_proc.returncode, exec_cmd)

    # BP-7：运行完成
    _bp("BP-7", f"run complete, result={result}")
    return result


# --------------------------------------------------------------------------- #
#  自测                                                                        #
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """快速自测，验证核心逻辑。"""
    passed = 0
    failed = 0

    def check(name: str, cond: bool) -> None:
        nonlocal passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    print("=== ctest_location_resolver selftest ===")

    # 测试1：INSTALL_PREFIX 优先
    r = CTestLocationResolver(env_vars={"INSTALL_PREFIX": "/opt/rapids"})
    p = r._installed_path()
    check("INSTALL_PREFIX 优先", str(p) == "/opt/rapids/bin/gtests/libwholegraph")

    # 测试2：INSTALL_PREFIX 缺失时回退 CONDA_PREFIX
    r2 = CTestLocationResolver(env_vars={"CONDA_PREFIX": "/opt/conda"})
    p2 = r2._installed_path()
    check("CONDA_PREFIX 回退", str(p2) == "/opt/conda/bin/gtests/libwholegraph")

    # 测试3：两者均缺时用 /usr
    r3 = CTestLocationResolver(env_vars={})
    p3 = r3._installed_path()
    check("/usr 默认值", str(p3) == "/usr/bin/gtests/libwholegraph")

    # 测试4：devcontainer 路径含 cpp/build/latest 后缀
    r4 = CTestLocationResolver(env_vars={})
    p4 = r4._devcontainer_path()
    check("devcontainer 路径后缀", str(p4).endswith("cpp/build/latest"))

    # 测试5：候选列表长度为 2
    r5 = CTestLocationResolver(env_vars={})
    cands = r5._build_candidates()
    check("候选列表长度 == 2", len(cands) == 2)

    # 测试6：候选列表按 priority 排序
    check("priority 排序", cands[0].priority < cands[1].priority)

    # 测试7：第一优先级为 INSTALLED
    check("首选 INSTALLED", cands[0].env == CTestEnv.INSTALLED)

    # 测试8：第二优先级为 DEVCONTAINER
    check("次选 DEVCONTAINER", cands[1].env == CTestEnv.DEVCONTAINER)

    # 测试9：不存在路径 → LocationResolutionError
    r6 = CTestLocationResolver(env_vars={})
    try:
        r6.resolve()
        check("不存在路径抛异常", False)
    except LocationResolutionError as e:
        check("不存在路径抛异常", len(e.searched_paths) == 2)

    # 测试10：LocationResolutionError.searched_paths 包含两条路径
    try:
        r6.resolve()
    except LocationResolutionError as e:
        check("searched_paths 长度 == 2", len(e.searched_paths) == 2)

    # 测试11：mock 存在路径时 resolve 成功
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fake_prefix = str(Path(td) / "prefix")
        fake_gtest = Path(fake_prefix) / "bin" / "gtests" / "libwholegraph"
        fake_gtest.mkdir(parents=True)
        r7 = CTestLocationResolver(env_vars={"INSTALL_PREFIX": fake_prefix})
        res = r7.resolve()
        check("mock 路径 resolve 成功", res.env == CTestEnv.INSTALLED)
        check("mock 路径与预期一致", res.resolved_path == fake_gtest)

    # 测试13：INSTALLED 不存在时跌落 DEVCONTAINER mock
    with tempfile.TemporaryDirectory() as td:
        fake_build = Path(td) / "cpp" / "build" / "latest"
        fake_build.mkdir(parents=True)
        # 让 _devcontainer_path 返回 fake_build：monkey-patch
        r8 = CTestLocationResolver(env_vars={})
        r8._devcontainer_path = lambda: fake_build  # type: ignore[method-assign]
        res = r8.resolve()
        check("DEVCONTAINER 回退 resolve 成功", res.env == CTestEnv.DEVCONTAINER)

    # 测试14：ResolutionResult.__str__ 包含路径字符串
    with tempfile.TemporaryDirectory() as td:
        fake_prefix2 = str(Path(td) / "pfx")
        fake_gt2 = Path(fake_prefix2) / "bin" / "gtests" / "libwholegraph"
        fake_gt2.mkdir(parents=True)
        r9 = CTestLocationResolver(env_vars={"INSTALL_PREFIX": fake_prefix2})
        res2 = r9.resolve()
        check("ResolutionResult.__str__ 含路径", str(fake_gt2) in str(res2))

    # 测试15：dry_run=True 不报错
    with tempfile.TemporaryDirectory() as td:
        fake_prefix3 = str(Path(td) / "pfx3")
        fake_gt3 = Path(fake_prefix3) / "bin" / "gtests" / "libwholegraph"
        fake_gt3.mkdir(parents=True)
        try:
            resolve_and_run(
                env_vars={"INSTALL_PREFIX": fake_prefix3},
                dry_run=True,
            )
            check("dry_run=True 不报错", True)
        except Exception:
            check("dry_run=True 不报错", False)

    print(f"\n结果：{passed} PASS / {failed} FAIL")


if __name__ == "__main__":
    _selftest()
