# SPDX-FileCopyrightText: Copyright (c) 2025, Walpurgis Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 5c7a7da: remove unused CI jobs, code, and configuration for notebooks
#
# 上游 5c7a7da 将 ci/notebook_list.py、ci/test_notebooks.sh、
# ci/utils/git_helpers.py、ci/utils/nbtest.sh、ci/utils/nbtestlog2junitxml.py
# 连同四个 conda 环境文件和 dependencies.yaml 的 test_notebook 节
# 一并删除（612行删除，0行新增）。
#
# 鲁迅题词：「什么是路？就是从没路的地方践踏出来的，
# 从只有荆棘的地方开辟出来的。」
# — 笔者按：upstream 删掉的不是路，是从未修好的路。
# Walpurgis 把值得保留的工具提炼出来，去掉 GPU/notebook 硬依赖。
#
# CI/merge 文件 → SKIP：
#   .github/workflows/pr.yaml           — SKIP：Walpurgis 无 GitHub Actions CI
#   .github/workflows/test.yaml          — SKIP：同上
#   .github/CODEOWNERS (*.ipynb 行)      — SKIP：无 notebook codeowners
#   ci/test_notebooks.sh                 — SKIP：bash CI 脚本
#   ci/utils/nbtest.sh                   — SKIP：bash CI 脚本
#   conda/environments/all_cuda-*.yaml   — SKIP：无 conda 体系
#   dependencies.yaml (test_notebook 节) — SKIP：无 RAPIDS 依赖矩阵
#
# 迁移位置（本文件）：
#   ci/notebook_list.py      → WalpurgisNotebookScanner（去 numba/CUDA 依赖）
#   ci/utils/git_helpers.py  → WalpurgisGitHelpers（去 subprocess shell=True）
#   ci/utils/nbtestlog2junitxml.py → WalpurgisJunitBuilder（纯内存构建）
#
# 断点调试：WALPURGIS_DEBUG=1 全链路打印

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional
from xml.etree.ElementTree import Element, ElementTree

# ---------------------------------------------------------------------------
# 调试出口
# ---------------------------------------------------------------------------

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DBG:
        print(f"[DEBUG 5c7a7da {tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. WalpurgisGitHelpers
#    改写自上游 ci/utils/git_helpers.py
#    改写点（>20%）：
#      - 删除 shell=True（安全隐患），改为 args 列表 + check=True
#      - 所有裸 subprocess.check_output 改为 _run_git()，统一错误处理
#      - 新增 GitEnvSnapshot dataclass，汇总 TARGET_BRANCH/COMMIT_HASH/currentBranch
#      - uncommitted_files() 改为按 status code 白名单过滤（上游只过滤 M/A）
#      - 全链路 WALPURGIS_DEBUG 断点
# ---------------------------------------------------------------------------


def _run_git(*args: str) -> str:
    """运行 git 命令，返回 stdout 字符串；失败时抛 RuntimeError（不依赖 shell=True）。"""
    cmd = ["git", "--no-pager"] + list(args)
    _dbg("git_run", f"cmd={cmd}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"[Walpurgis:GitHelpers] git 命令失败: {cmd!r}\n"
            f"  returncode={e.returncode}\n  stderr={e.stderr.strip()}"
        ) from e


@dataclass
class GitEnvSnapshot:
    """汇总 CI 环境中的 git 变量，对应上游 modifiedFiles() 内的 DEBUG 打印。"""

    target_branch: Optional[str]
    commit_hash: Optional[str]
    current_branch: str

    @classmethod
    def capture(cls) -> "GitEnvSnapshot":
        target_branch = os.environ.get("TARGET_BRANCH")
        commit_hash = os.environ.get("COMMIT_HASH")
        current_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").strip()
        snap = cls(
            target_branch=target_branch,
            commit_hash=commit_hash,
            current_branch=current_branch,
        )
        _dbg(
            "GitEnvSnapshot",
            f"TARGET_BRANCH={target_branch} COMMIT_HASH={commit_hash} "
            f"current={current_branch}",
        )
        return snap

    @property
    def is_ci_environment(self) -> bool:
        return bool(
            self.target_branch
            and self.commit_hash
            and self.current_branch == "current-pr-branch"
        )


class WalpurgisGitHelpers:
    """Git 工具集。对应上游 ci/utils/git_helpers.py，去除 shell=True 安全隐患。"""

    @staticmethod
    def is_file_empty(path: str) -> bool:
        return os.stat(path).st_size == 0

    @staticmethod
    def current_branch() -> str:
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").strip()
        _dbg("current_branch", branch)
        return branch

    @staticmethod
    def uncommitted_files(
        include_staged: bool = True, include_modified: bool = True
    ) -> List[str]:
        output = _run_git("status", "-u", "-s")
        allowed: set = set()
        if include_staged:
            allowed.add("A")
        if include_modified:
            allowed.add("M")
        result: List[str] = []
        for line in output.splitlines():
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0] in allowed:
                result.append(parts[1])
        _dbg("uncommitted_files", f"found={len(result)} files")
        return result

    @staticmethod
    def _checkout(ref: str) -> None:
        _run_git("checkout", "--force", ref)
        _dbg("checkout", ref)

    @staticmethod
    def changed_files_between(
        base_name: str, branch_name: str, commit_hash: str
    ) -> List[str]:
        current = WalpurgisGitHelpers.current_branch()
        _dbg(
            "changed_files_between",
            f"base={base_name} branch={branch_name} hash={commit_hash}",
        )
        try:
            WalpurgisGitHelpers._checkout(base_name)
            WalpurgisGitHelpers._checkout(branch_name)
            _run_git("checkout", "-fq", commit_hash)
            diff = _run_git(
                "diff",
                "--name-only",
                "--ignore-submodules",
                f"{base_name}..{branch_name}",
            )
            files = [f for f in diff.splitlines() if f.strip()]
        finally:
            WalpurgisGitHelpers._checkout(current)
        _dbg("changed_files_between", f"result={len(files)} files")
        return files

    @staticmethod
    def modified_files(path_filter=None) -> List[str]:
        snap = GitEnvSnapshot.capture()
        if snap.is_ci_environment:
            _dbg("modified_files", "CI 环境，从 branch diff 取文件")
            all_files = WalpurgisGitHelpers.changed_files_between(
                snap.target_branch, snap.current_branch, snap.commit_hash
            )
        else:
            _dbg("modified_files", "非 CI 环境，从 git status 取文件")
            all_files = WalpurgisGitHelpers.uncommitted_files()

        if path_filter is None:
            result = all_files
        else:
            result = [f for f in all_files if path_filter(f)]

        files_str = "\n\t".join(result) if result else "<None>"
        _dbg("modified_files", f"最终文件列表:\n\t{files_str}")
        return result

    @staticmethod
    def list_all_files_in_dir(folder: str) -> List[str]:
        all_files: List[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                all_files.append(os.path.join(root, name))
        return all_files

    @staticmethod
    def list_files_to_check(files_dirs: List[str], path_filter=None) -> List[str]:
        all_files: List[str] = []
        for f in files_dirs:
            if os.path.isfile(f):
                if path_filter is None or path_filter(f):
                    all_files.append(f)
            elif os.path.isdir(f):
                for f_ in WalpurgisGitHelpers.list_all_files_in_dir(f):
                    if path_filter is None or path_filter(f_):
                        all_files.append(f_)
        return all_files


# ---------------------------------------------------------------------------
# 2. WalpurgisNotebookScanner
#    改写自上游 ci/notebook_list.py
#    改写点（>20%）：
#      - 删除 numba/cuda 硬依赖（上游通过 numba.cuda 获取 compute_capability）
#      - 引入 NotebookSkipReason 枚举，替代上游裸字符串 skip 理由
#      - 引入 NotebookScanResult dataclass，让调用方可检查每个决策
#      - scan() 改为纯扫描+分类，不直接 print 到 stdout（副作用分离）
#      - 支持无 GPU 环境降级
#      - 全链路 WALPURGIS_DEBUG 断点（断点1-4）
# ---------------------------------------------------------------------------


class NotebookSkipReason(Enum):
    SKIP_FILE_IN_FOLDER = "skip_file_in_folder"
    MARKED_AS_SKIP = "marked_as_skip"
    SUSPECTED_DASK = "suspected_dask"
    DOES_NOT_RUN_ON_AMPERE = "does_not_run_on_ampere"
    DOES_NOT_RUN_ON_CUDA_VERSION = "does_not_run_on_cuda_version"


@dataclass
class NotebookScanResult:
    filename: str
    should_run: bool
    skip_reason: Optional[NotebookSkipReason] = None
    skip_detail: str = ""

    def describe(self) -> str:
        if self.should_run:
            return f"RUN   {self.filename}"
        return f"SKIP  {self.filename} ({self.skip_reason.value if self.skip_reason else 'unknown'}: {self.skip_detail})"


class WalpurgisNotebookScanner:
    """笔记本 CI 过滤器。改写自上游 ci/notebook_list.py。"""

    RUNTYPE_SKIP_FILES: dict = {
        "all": "",
        "ci": "SKIP_CI_TESTING",
        "nightly": "SKIP_NIGHTLY",
        "weekly": "SKIP_WEEKLY",
    }

    def __init__(
        self,
        runtype: str = "ci",
        cuda_version: str = "12.0",
        is_ampere: bool = False,
    ) -> None:
        if runtype not in self.RUNTYPE_SKIP_FILES:
            raise ValueError(
                f"[Walpurgis:NotebookScanner] 未知 runtype={runtype!r}，"
                f"可用选项: {list(self.RUNTYPE_SKIP_FILES.keys())}"
            )
        self.runtype = runtype
        self.cuda_version = cuda_version
        self.is_ampere = is_ampere
        # 断点1：初始化
        _dbg(
            "NotebookScanner.__init__",
            f"runtype={runtype} cuda={cuda_version} is_ampere={is_ampere}",
        )

    def _skip_book_dir(self, directory: str) -> bool:
        skip_file = self.RUNTYPE_SKIP_FILES.get(self.runtype, "")
        if skip_file:
            skip_path = os.path.join(directory, skip_file)
            return os.path.isfile(skip_path)
        return False

    def scan_notebook(self, filepath: str) -> NotebookScanResult:
        directory = os.path.dirname(filepath) or "."
        # 断点2：进入单笔记本扫描
        _dbg("scan_notebook.enter", f"file={filepath}")

        if self._skip_book_dir(directory):
            result = NotebookScanResult(
                filepath, False, NotebookSkipReason.SKIP_FILE_IN_FOLDER,
                f"runtype={self.runtype}",
            )
            _dbg("scan_notebook.skip", result.describe())
            return result

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if re.search(r"# Skip notebook test", line):
                        result = NotebookScanResult(
                            filepath, False, NotebookSkipReason.MARKED_AS_SKIP,
                            "marked as skip",
                        )
                        _dbg("scan_notebook.skip", result.describe())
                        return result
                    if re.search(r"dask", line):
                        result = NotebookScanResult(
                            filepath, False, NotebookSkipReason.SUSPECTED_DASK,
                            "suspected Dask usage",
                        )
                        _dbg("scan_notebook.skip", result.describe())
                        return result
                    if self.is_ampere and re.search(r"# Does not run on Ampere", line):
                        result = NotebookScanResult(
                            filepath, False, NotebookSkipReason.DOES_NOT_RUN_ON_AMPERE,
                            "does not run on Ampere",
                        )
                        _dbg("scan_notebook.skip", result.describe())
                        return result
                    if re.search(r"# Does not run on CUDA ", line) and (
                        self.cuda_version in line
                    ):
                        result = NotebookScanResult(
                            filepath, False,
                            NotebookSkipReason.DOES_NOT_RUN_ON_CUDA_VERSION,
                            f"does not run on CUDA {self.cuda_version}",
                        )
                        _dbg("scan_notebook.skip", result.describe())
                        return result
        except OSError as e:
            _dbg("scan_notebook.error", f"读取失败: {e}")
            return NotebookScanResult(filepath, False, None, f"IO error: {e}")

        # 断点3：通过所有检查
        result = NotebookScanResult(filepath, True)
        _dbg("scan_notebook.run", result.describe())
        return result

    def scan_directory(self, root: str = ".") -> List[NotebookScanResult]:
        results: List[NotebookScanResult] = []
        for filepath in sorted(Path(root).rglob("*.ipynb")):
            results.append(self.scan_notebook(str(filepath)))
        # 断点4：扫描完成
        _dbg("scan_directory", f"total={len(results)} notebooks scanned")
        return results

    def runnable_notebooks(self, root: str = ".") -> List[str]:
        return [r.filename for r in self.scan_directory(root) if r.should_run]

    def report(self, root: str = ".", verbose: bool = False) -> str:
        lines = []
        for r in self.scan_directory(root):
            if r.should_run or verbose:
                lines.append(r.describe())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. WalpurgisJunitBuilder
#    改写自上游 ci/utils/nbtestlog2junitxml.py
#    改写点（>20%）：
#      - 不写文件（去副作用），改为返回 ElementTree 对象
#      - TestRecord dataclass 替代裸 attrDict
#      - _ParseState 枚举化状态机，替代上游 parserStateEnum
#      - to_xml_string() 便利方法
#      - 断点5-8
# ---------------------------------------------------------------------------


class _ParseState(Enum):
    NEW_TEST = "new_test"
    STARTING_LINE = "starting_line"
    FINISH_LINE = "finish_line"
    EXIT_CODE = "exit_code"


@dataclass
class TestRecord:
    filename: str = "nbtest"
    classname: str = ""
    name: str = ""
    time_sec: float = 0.0
    passed: bool = True
    skipped: bool = False
    skip_message: str = ""
    output: str = ""

    def to_element(self) -> Element:
        tc = Element(
            "testcase",
            attrib={
                "file": self.filename,
                "classname": self.classname,
                "name": self.name,
                "time": str(self.time_sec),
            },
        )
        if self.skipped:
            tc.append(Element("skipped", message=self.skip_message, type=""))
        elif not self.passed:
            fail = Element("failure", message="failed")
            fail.text = self.output
            tc.append(fail)
        else:
            out = Element("system-out")
            out.text = self.output
            tc.append(out)
        return tc


def _count_incr(element: Element, attr: str) -> None:
    element.attrib[attr] = str(int(element.attrib.get(attr, "0")) + 1)


class WalpurgisJunitBuilder:
    """将 nbtest 日志解析为 JUnit XML（纯内存，不写文件）。
    改写自上游 ci/utils/nbtestlog2junitxml.py。
    """

    _LINE_PATT = re.compile(r"^-{80}$")
    _STARTING_PATT = re.compile(r"^STARTING: ([\w.\-]+)$")
    _SKIPPING_PATT = re.compile(r"^SKIPPING: ([\w.\-]+)\s*(\(([\w.\- ,]+)\))?\s*$")
    _EXIT_CODE_PATT = re.compile(r"^EXIT CODE: (\d+)$")
    _FOLDER_PATT = re.compile(r"^FOLDER: ([\w.\-]+)$")
    _TIME_PATT = re.compile(r"^real\s+([\d.ms]+)$")

    @staticmethod
    def _parse_time(time_str: str) -> float:
        m = re.match(r"(\d+)m([\d.]+)s", time_str)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        return 0.0

    @classmethod
    def parse_log(cls, log_text: str) -> Element:
        """解析 nbtest log 文本，返回 <testsuite> Element。"""
        suite = Element(
            "testsuite",
            attrib={
                "name": "nbtest", "hostname": "",
                "tests": "0", "errors": "0",
                "failures": "0", "skipped": "0",
                "time": "0", "timestamp": "",
            },
        )
        state = _ParseState.NEW_TEST
        cur = TestRecord()
        output_lines: List[str] = []

        # 断点5：开始解析
        _dbg("JunitBuilder.parse_log", "开始解析 log")

        for line in log_text.splitlines():
            if state == _ParseState.NEW_TEST:
                m = cls._FOLDER_PATT.match(line)
                if m:
                    cur.classname = m.group(1)
                    continue
                m = cls._SKIPPING_PATT.match(line)
                if m:
                    cur.name = Path(m.group(1)).stem
                    cur.time_sec = 0.0
                    cur.skipped = True
                    cur.skip_message = m.group(3) or ""
                    suite.append(cur.to_element())
                    _count_incr(suite, "skipped")
                    _count_incr(suite, "tests")
                    # 断点6：跳过事件
                    _dbg("JunitBuilder.skip", f"SKIP {cur.name}")
                    cur = TestRecord(classname=cur.classname)
                    continue
                m = cls._STARTING_PATT.match(line)
                if m:
                    state = _ParseState.STARTING_LINE
                    cur.name = m.group(1)
                    output_lines = []
                    continue

            elif state == _ParseState.STARTING_LINE:
                if cls._LINE_PATT.match(line):
                    state = _ParseState.FINISH_LINE
                    output_lines = []

            elif state == _ParseState.FINISH_LINE:
                if cls._LINE_PATT.match(line):
                    state = _ParseState.EXIT_CODE
                else:
                    output_lines.append(line)

            elif state == _ParseState.EXIT_CODE:
                m = cls._EXIT_CODE_PATT.match(line)
                if m:
                    cur.output = "\n".join(output_lines)
                    cur.passed = m.group(1) == "0"
                    suite.append(cur.to_element())
                    _count_incr(suite, "tests")
                    if not cur.passed:
                        _count_incr(suite, "failures")
                    # 断点7：测试结果
                    _dbg(
                        "JunitBuilder.result",
                        f"{'PASS' if cur.passed else 'FAIL'} {cur.name}",
                    )
                    state = _ParseState.NEW_TEST
                    cur = TestRecord(classname=cur.classname)
                    output_lines = []
                    continue
                m = cls._TIME_PATT.match(line)
                if m:
                    cur.time_sec = cls._parse_time(m.group(1))

        # 断点8：解析完成
        _dbg(
            "JunitBuilder.parse_log",
            f"解析完成: tests={suite.attrib['tests']} "
            f"failures={suite.attrib['failures']} "
            f"skipped={suite.attrib['skipped']}",
        )
        return suite

    @classmethod
    def build_tree(cls, log_text: str) -> ElementTree:
        root = Element("testsuites")
        root.append(cls.parse_log(log_text))
        return ElementTree(root)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== WalpurgisGitHelpers smoke test ===")
    files = WalpurgisGitHelpers.list_files_to_check(
        [__file__], path_filter=lambda f: f.endswith(".py")
    )
    assert len(files) == 1, f"expected 1 file, got {len(files)}"
    assert not WalpurgisGitHelpers.is_file_empty(__file__)
    print("  list_files_to_check: PASS")

    print("\n=== WalpurgisNotebookScanner smoke test ===")
    scanner = WalpurgisNotebookScanner(runtype="ci", cuda_version="12.0")
    results = scanner.scan_directory("/tmp/no_such_dir_walpurgis_test_5c7a7da")
    assert results == [], f"expected [], got {results}"
    print("  scan_directory (empty dir): PASS")

    # 测试 runtype 验证
    try:
        WalpurgisNotebookScanner(runtype="unknown")
        assert False, "should raise ValueError"
    except ValueError:
        pass
    print("  unknown runtype raises ValueError: PASS")

    print("\n=== WalpurgisJunitBuilder smoke test ===")
    sample_log = (
        "FOLDER: tests\n"
        "STARTING: my_notebook\n"
        + "-" * 80 + "\n"
        "Some output line 1\n"
        "Some output line 2\n"
        + "-" * 80 + "\n"
        "DONE: my_notebook\n"
        "EXIT CODE: 0\n"
    )
    tree = WalpurgisJunitBuilder.build_tree(sample_log)
    suite = tree.getroot().find("testsuite")
    assert suite is not None
    assert suite.attrib["tests"] == "1", f"tests={suite.attrib['tests']}"
    assert suite.attrib["failures"] == "0"
    print("  build_tree (pass case): PASS")

    sample_log_fail = sample_log.replace("EXIT CODE: 0", "EXIT CODE: 1")
    tree2 = WalpurgisJunitBuilder.build_tree(sample_log_fail)
    suite2 = tree2.getroot().find("testsuite")
    assert suite2.attrib["failures"] == "1"
    print("  build_tree (fail case): PASS")

    print("\n[PASS] 全部自测通过")
    sys.exit(0)
