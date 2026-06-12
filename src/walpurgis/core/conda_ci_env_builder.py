"""
conda_ci_env_builder.py
=======================
迁移自 upstream cugraph-gnn a11b148 (Kyle Edwards / James Lamb, 2025-02-24)
"Create Conda CI test env in one step (#144)"
Issue: https://github.com/rapidsai/build-planning/issues/22

上游变更摘要 (5 files changed, 115 insertions(+), 65 deletions(-)):
  - ci/build_docs.sh / ci/test_cpp.sh / ci/test_python.sh:
      将「env create + mamba install」两步合并为
      rapids-dependency-file-generator --prepend-channel + env create 一步。
  - ci/release/update-version.sh: DEPENDENCIES 补充 libwholegraph。
  - dependencies.yaml: 新增 test_cugraph_dgl / test_cugraph_pyg /
      test_pylibwholegraph file-key 及四个 depends_on_* 节。

CI/merge → SKIP: 全部 5 个上游文件均为 RAPIDS CI 基础设施文件，
  Walpurgis 无 conda 构建/测试体系（详见 MIGRATION_LOG.md）。

鲁迅拿法改写 (>=20%): 见模块 docstring 及 MIGRATION_LOG.md。
"""

from __future__ import annotations

import os
import pdb
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

_KNOWN_SECTIONS: Set[str] = {
    "cuda_version", "py_version", "docs",
    "test_cpp", "test_python_common", "test_python_pylibwholegraph",
    "depends_on_cugraph", "depends_on_cudf", "depends_on_dgl",
    "depends_on_pytorch", "depends_on_ogb",
    "depends_on_pylibwholegraph", "depends_on_cugraph_dgl",
    "depends_on_cugraph_pyg", "depends_on_mkl",
    "depends_on_libwholegraph", "depends_on_libwholegraph_tests",
    "depends_on_rmm",
}

_CUDA_TO_DGL_CHANNEL: Dict[str, str] = {
    "11": "dglteam/label/th23_cu118",
    "12": "dglteam/label/th23_cu121",
}


class EnvBuildStrategy(Enum):
    """描述 conda CI 测试环境的构建策略。

    TWO_STEP: 旧模式 — env create → mamba install（两次 solver，可能版本覆盖）。
    ONE_STEP: a11b148 — dfg --prepend-channel → env create（单次 solver，一致性保证）。
    上游无此枚举，仅有 shell diff 减号/加号行。
    """
    TWO_STEP = "two_step"
    ONE_STEP  = "one_step"

    def is_idempotent_safe(self) -> bool:
        return self == EnvBuildStrategy.ONE_STEP

    def description(self) -> str:
        if self == EnvBuildStrategy.ONE_STEP:
            return "a11b148 one-step: dfg --prepend-channel → env create (单次 solver)"
        return "pre-a11b148 two-step: dfg → env create → mamba install (两次安装)"


@dataclass
class CondaEnvSpec:
    """结构化描述一个 CI 测试环境的构建规格。

    上游每个 env 参数散落在各自 bash 代码块里，此类集中建模。
    generate_dfg_args() 产出 rapids-dependency-file-generator 可测试参数列表。
    """
    name: str
    file_key: str
    matrix_template: str
    prepend_channels: List[str]
    build_strategy: EnvBuildStrategy
    upstream_commit: str = "a11b148"

    def generate_dfg_args(
        self,
        cuda_version: str = "12.5",
        arch: str = "x86_64",
        py_version: str = "3.11",
    ) -> List[str]:
        if _DEBUG:
            print(f"[DEBUG] CondaEnvSpec.generate_dfg_args: env={self.name!r}")
            pdb.set_trace()  # 断点①
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])
        matrix_str = (
            self.matrix_template
            .replace("{cuda}", cuda_major_minor)
            .replace("{arch}", arch)
            .replace("{py}", py_version)
        )
        args: List[str] = [
            "--output", "conda",
            "--file-key", self.file_key,
            "--matrix", matrix_str,
        ]
        for ch in self.prepend_channels:
            args += ["--prepend-channel", ch]
        return args

    def shell_snippet(
        self,
        cuda_version: str = "12.5",
        arch: str = "x86_64",
        py_version: str = "3.11",
    ) -> str:
        args = self.generate_dfg_args(cuda_version, arch, py_version)
        arg_str = " \\\n    ".join(args)
        return "\n".join([
            f'rapids-logger "({self.name}) Generate Python testing dependencies"',
            "rapids-dependency-file-generator \\",
            f"    {arg_str} \\",
            "  | tee env.yaml",
            "",
            f"rapids-mamba-retry env create --yes -f env.yaml -n {self.name}",
        ])

    def validate(self) -> List[str]:
        if _DEBUG:
            print(f"[DEBUG] CondaEnvSpec.validate: env={self.name!r}")
            pdb.set_trace()  # 断点②
        violations: List[str] = []
        if not self.name:
            violations.append("name 不能为空")
        if not self.file_key:
            violations.append("file_key 不能为空")
        if "{cuda}" not in self.matrix_template:
            violations.append("matrix_template 缺少 {cuda} 占位符")
        if self.build_strategy == EnvBuildStrategy.ONE_STEP and not self.prepend_channels:
            violations.append(f"ONE_STEP 策略的 {self.name!r} 应至少有一个 prepend_channel")
        return violations


@dataclass
class DGLChannelResolver:
    """CUDA major → DGL conda channel 字符串。

    a11b148 将 ci/test_python.sh 中的 if/else 分支前移至脚本顶部。
    此类将映射表显式化，支持扩展至未来 CUDA 版本。
    """
    _channel_map: Dict[str, str] = field(
        default_factory=lambda: dict(_CUDA_TO_DGL_CHANNEL)
    )
    _default_channel: str = "dglteam/label/th23_cu121"

    def resolve(self, cuda_major: str) -> str:
        if _DEBUG:
            print(f"[DEBUG] DGLChannelResolver.resolve: cuda_major={cuda_major!r}")
            pdb.set_trace()  # 断点③
        return self._channel_map.get(cuda_major, self._default_channel)

    def register(self, cuda_major: str, channel: str) -> None:
        self._channel_map[cuda_major] = channel

    def known_cuda_versions(self) -> List[str]:
        return sorted(self._channel_map.keys())


@dataclass
class DependencyFileKeySpec:
    """建模 dependencies.yaml 中一个 files: 条目。

    a11b148 新增三个独立 file-key 并补充 docs/test_cpp 的 includes。
    validate_includes() 检查所有 include 是否在已知 section 集合内。
    上游无任何校验层，仅 YAML 文本。
    """
    key: str
    output: str
    includes: List[str]
    upstream_commit: str = "a11b148"

    def validate_includes(self) -> List[str]:
        if _DEBUG:
            print(f"[DEBUG] DependencyFileKeySpec.validate_includes: key={self.key!r}")
            pdb.set_trace()  # 断点④
        return [
            f"file-key {self.key!r} 的 include {u!r} 不在已知 section 集合内"
            for u in self.includes if u not in _KNOWN_SECTIONS
        ]

    def is_test_env(self) -> bool:
        return self.key.startswith("test_")

    def summary(self) -> str:
        violations = self.validate_includes()
        status = "合规" if not violations else f"违规({len(violations)}处)"
        lines = [f"DependencyFileKeySpec[{self.key!r}] output={self.output} [{status}]"]
        for inc in self.includes:
            mark = "✓" if inc in _KNOWN_SECTIONS else "✗ 未知"
            lines.append(f"  - {inc}  {mark}")
        return "\n".join(lines)


@dataclass
class VersionPinEntry:
    """建模 dependencies.yaml 中一个 depends_on_<pkg> 节。

    a11b148 新增四节：depends_on_libwholegraph / depends_on_libwholegraph_tests /
    depends_on_cugraph_pyg / depends_on_mkl。
    is_upper_bounded() 标识 mkl 类上界约束，is_wildcard_pin() 标识 RAPIDS wildcard。
    上游仅有 YAML 文本，无 Python 层类型区分。
    """
    section_key: str
    conda_package: str
    version_constraint: str
    output_types: List[str]
    upstream_commit: str = "a11b148"

    def is_upper_bounded(self) -> bool:
        return "<" in self.version_constraint and "==" not in self.version_constraint

    def is_wildcard_pin(self) -> bool:
        return ".*" in self.version_constraint

    def conda_spec(self) -> str:
        return f"{self.conda_package}{self.version_constraint}"

    def summary(self) -> str:
        flags = []
        if self.is_upper_bounded():
            flags.append("上界约束")
        if self.is_wildcard_pin():
            flags.append("wildcard-pin")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        return (
            f"{self.section_key}: {self.conda_spec()}"
            f"  output_types={self.output_types}{flag_str}"
        )


@dataclass
class ReleaseVersionTracker:
    """追踪 ci/release/update-version.sh DEPENDENCIES 数组成员。

    a11b148 补充 libwholegraph 到该数组。
    missing_vs_spec() 与 VersionPinEntry 列表交叉验证追踪覆盖完整性。
    上游无程序化交叉验证。
    """
    registered_packages: List[str] = field(default_factory=lambda: [
        "libcugraphops", "libraft", "libraft-headers", "librmm",
        "libwholegraph",   # a11b148 新增
        "pylibcugraph", "pylibwholegraph", "rmm",
    ])

    def is_tracked(self, package: str) -> bool:
        return package in self.registered_packages

    def missing_vs_spec(self, pin_entries: List["VersionPinEntry"]) -> List[str]:
        if _DEBUG:
            print(f"[DEBUG] ReleaseVersionTracker.missing_vs_spec")
            pdb.set_trace()  # 断点⑤
        return [e.conda_package for e in pin_entries if not self.is_tracked(e.conda_package)]


# --- 注册表数据 ---

_ENV_SPECS: Dict[str, CondaEnvSpec] = {
    "docs": CondaEnvSpec(
        name="docs", file_key="docs",
        matrix_template="cuda={cuda};arch={arch};py={py}",
        prepend_channels=["${CPP_CHANNEL}"],
        build_strategy=EnvBuildStrategy.ONE_STEP,
    ),
    "test_cpp": CondaEnvSpec(
        name="test", file_key="test_cpp",
        matrix_template="cuda={cuda};arch={arch}",
        prepend_channels=["${CPP_CHANNEL}"],
        build_strategy=EnvBuildStrategy.ONE_STEP,
    ),
    "test_cugraph_dgl": CondaEnvSpec(
        name="test_cugraph_dgl", file_key="test_cugraph_dgl",
        matrix_template="cuda={cuda};arch={arch};py={py}",
        prepend_channels=[
            "${CPP_CHANNEL}", "${PYTHON_CHANNEL}",
            "pytorch", "conda-forge", "${DGL_CHANNEL}", "nvidia",
        ],
        build_strategy=EnvBuildStrategy.ONE_STEP,
    ),
    "test_cugraph_pyg": CondaEnvSpec(
        name="test_cugraph_pyg", file_key="test_cugraph_pyg",
        matrix_template="cuda={cuda};arch={arch};py={py}",
        prepend_channels=["${CPP_CHANNEL}", "${PYTHON_CHANNEL}", "pytorch"],
        build_strategy=EnvBuildStrategy.ONE_STEP,
    ),
    "test_pylibwholegraph": CondaEnvSpec(
        name="test_pylibwholegraph", file_key="test_pylibwholegraph",
        matrix_template="cuda={cuda};arch={arch};py={py}",
        prepend_channels=["${CPP_CHANNEL}", "${PYTHON_CHANNEL}", "pytorch"],
        build_strategy=EnvBuildStrategy.ONE_STEP,
    ),
}

_FILE_KEY_SPECS: Dict[str, DependencyFileKeySpec] = {
    "test_cugraph_dgl": DependencyFileKeySpec(
        key="test_cugraph_dgl", output="none",
        includes=[
            "cuda_version", "depends_on_cugraph", "depends_on_cudf",
            "depends_on_dgl", "depends_on_pytorch", "depends_on_ogb",
            "py_version", "test_python_common",
            "depends_on_pylibwholegraph", "depends_on_cugraph_dgl",
        ],
    ),
    "test_cugraph_pyg": DependencyFileKeySpec(
        key="test_cugraph_pyg", output="none",
        includes=[
            "cuda_version", "depends_on_cugraph", "depends_on_cudf",
            "depends_on_dgl", "depends_on_pytorch", "depends_on_ogb",
            "py_version", "test_python_common",
            "depends_on_pylibwholegraph", "depends_on_cugraph_pyg",
        ],
    ),
    "test_pylibwholegraph": DependencyFileKeySpec(
        key="test_pylibwholegraph", output="none",
        includes=[
            "cuda_version", "depends_on_cugraph", "depends_on_cudf",
            "depends_on_dgl", "depends_on_pytorch", "depends_on_ogb",
            "py_version", "test_python_common",
            "depends_on_mkl", "depends_on_pylibwholegraph",
            "test_python_pylibwholegraph",
        ],
    ),
}

_VERSION_PIN_ENTRIES: List[VersionPinEntry] = [
    VersionPinEntry(
        section_key="depends_on_libwholegraph",
        conda_package="libwholegraph",
        version_constraint="==25.4.*,>=0.0.0a0",
        output_types=["conda"],
    ),
    VersionPinEntry(
        section_key="depends_on_libwholegraph_tests",
        conda_package="libwholegraph-tests",
        version_constraint="==25.4.*,>=0.0.0a0",
        output_types=["conda"],
    ),
    VersionPinEntry(
        section_key="depends_on_cugraph_pyg",
        conda_package="cugraph-pyg",
        version_constraint="==25.4.*,>=0.0.0a0",
        output_types=["conda"],
    ),
    VersionPinEntry(
        section_key="depends_on_mkl",
        conda_package="mkl",
        version_constraint="<2024.1.0",
        output_types=["conda"],
    ),
]


class CIEnvRegistry:
    """集中注册和校验 a11b148 引入的全部 CI 环境规格。

    上游 5 个 shell 脚本各自硬编码参数，此类提供统一查询层。
    validate_all() 覆盖 env / file_key / release_tracker 三层校验。
    """

    def __init__(self) -> None:
        self._envs = dict(_ENV_SPECS)
        self._file_keys = dict(_FILE_KEY_SPECS)
        self._version_pins = list(_VERSION_PIN_ENTRIES)
        self._release_tracker = ReleaseVersionTracker()
        self._dgl_resolver = DGLChannelResolver()

    def get_env(self, name: str) -> CondaEnvSpec:
        if _DEBUG:
            print(f"[DEBUG] CIEnvRegistry.get_env: name={name!r}")
            pdb.set_trace()  # 断点⑥
        if name not in self._envs:
            raise KeyError(
                f"未知 CI env: {name!r}. 已注册: {sorted(self._envs.keys())}"
            )
        return self._envs[name]

    def list_envs(self) -> List[str]:
        return sorted(self._envs.keys())

    def validate_all(self) -> Dict[str, List[str]]:
        if _DEBUG:
            print("[DEBUG] CIEnvRegistry.validate_all 入口")
            pdb.set_trace()  # 断点⑦
        report: Dict[str, List[str]] = {}
        for name, spec in self._envs.items():
            v = spec.validate()
            if v:
                report[f"env:{name}"] = v
        for key, fk_spec in self._file_keys.items():
            v = fk_spec.validate_includes()
            if v:
                report[f"file_key:{key}"] = v
        missing = self._release_tracker.missing_vs_spec(self._version_pins)
        if missing:
            report["release_tracker:missing"] = [
                f"包 {pkg!r} 在 VersionPinEntry 中但未被 ReleaseVersionTracker 追踪"
                for pkg in missing
            ]
        return report

    def dgl_channel(self, cuda_major: str) -> str:
        return self._dgl_resolver.resolve(cuda_major)

    def version_pin_summary(self) -> str:
        lines = ["=== a11b148 新增 VersionPinEntry ==="]
        for entry in self._version_pins:
            lines.append(f"  {entry.summary()}")
        return "\n".join(lines)


def _selftest() -> None:
    """运行 11 项自测，覆盖全部类。"""
    if _DEBUG:
        print("[DEBUG] _selftest 入口")
        pdb.set_trace()  # 断点⑧

    results: List[tuple] = []

    try:
        assert EnvBuildStrategy.ONE_STEP.is_idempotent_safe()
        assert not EnvBuildStrategy.TWO_STEP.is_idempotent_safe()
        assert "one-step" in EnvBuildStrategy.ONE_STEP.description()
        results.append(("EnvBuildStrategy 语义", "PASS"))
    except AssertionError as e:
        results.append(("EnvBuildStrategy 语义", f"FAIL: {e}"))

    try:
        r = DGLChannelResolver()
        assert r.resolve("11") == "dglteam/label/th23_cu118"
        assert r.resolve("12") == "dglteam/label/th23_cu121"
        assert r.resolve("13") == "dglteam/label/th23_cu121"
        results.append(("DGLChannelResolver CUDA 11/12/fallback", "PASS"))
    except AssertionError as e:
        results.append(("DGLChannelResolver", f"FAIL: {e}"))

    try:
        reg = CIEnvRegistry()
        spec = reg.get_env("test_cugraph_dgl")
        args = spec.generate_dfg_args(cuda_version="12.5", arch="x86_64", py_version="3.11")
        assert "--file-key" in args
        assert "test_cugraph_dgl" in args
        assert "--prepend-channel" in args
        assert "${CPP_CHANNEL}" in args
        results.append(("CondaEnvSpec.generate_dfg_args (test_cugraph_dgl)", "PASS"))
    except (AssertionError, KeyError) as e:
        results.append(("CondaEnvSpec.generate_dfg_args", f"FAIL: {e}"))

    try:
        reg = CIEnvRegistry()
        for env_name in reg.list_envs():
            violations = reg.get_env(env_name).validate()
            assert not violations, f"{env_name} 违规: {violations}"
        results.append(("CondaEnvSpec.validate 全 env 合规", "PASS"))
    except AssertionError as e:
        results.append(("CondaEnvSpec.validate", f"FAIL: {e}"))

    try:
        reg = CIEnvRegistry()
        spec = reg._file_keys["test_cugraph_dgl"]
        violations = spec.validate_includes()
        assert not violations, f"violations: {violations}"
        results.append(("DependencyFileKeySpec.validate_includes (test_cugraph_dgl)", "PASS"))
    except AssertionError as e:
        results.append(("DependencyFileKeySpec.validate_includes", f"FAIL: {e}"))

    try:
        bad_spec = DependencyFileKeySpec(
            key="test_bad", output="none",
            includes=["cuda_version", "nonexistent_section_xyz"]
        )
        violations = bad_spec.validate_includes()
        assert len(violations) == 1
        assert "nonexistent_section_xyz" in violations[0]
        results.append(("DependencyFileKeySpec 未知 include 检出", "PASS"))
    except AssertionError as e:
        results.append(("DependencyFileKeySpec 未知 include 检出", f"FAIL: {e}"))

    try:
        mkl_pin = _VERSION_PIN_ENTRIES[3]
        assert mkl_pin.conda_package == "mkl"
        assert mkl_pin.is_upper_bounded()
        assert not mkl_pin.is_wildcard_pin()
        wg_pin = _VERSION_PIN_ENTRIES[0]
        assert wg_pin.conda_package == "libwholegraph"
        assert wg_pin.is_wildcard_pin()
        assert not wg_pin.is_upper_bounded()
        results.append(("VersionPinEntry 分类 (mkl上界/libwholegraph wildcard)", "PASS"))
    except AssertionError as e:
        results.append(("VersionPinEntry 分类", f"FAIL: {e}"))

    try:
        tracker = ReleaseVersionTracker()
        assert tracker.is_tracked("libwholegraph")
        assert not tracker.is_tracked("not_a_package")
        results.append(("ReleaseVersionTracker.is_tracked", "PASS"))
    except AssertionError as e:
        results.append(("ReleaseVersionTracker.is_tracked", f"FAIL: {e}"))

    try:
        tracker = ReleaseVersionTracker()
        missing = tracker.missing_vs_spec(_VERSION_PIN_ENTRIES)
        assert "cugraph-pyg" in missing or "libwholegraph-tests" in missing
        results.append(("ReleaseVersionTracker.missing_vs_spec 交叉验证", "PASS"))
    except AssertionError as e:
        results.append(("ReleaseVersionTracker.missing_vs_spec", f"FAIL: {e}"))

    try:
        reg = CIEnvRegistry()
        report = reg.validate_all()
        env_v = {k: v for k, v in report.items() if k.startswith("env:")}
        fk_v  = {k: v for k, v in report.items() if k.startswith("file_key:")}
        assert not env_v,  f"env 违规: {env_v}"
        assert not fk_v,   f"file_key 违规: {fk_v}"
        results.append(("CIEnvRegistry.validate_all (env+file_key 合规)", "PASS"))
    except AssertionError as e:
        results.append(("CIEnvRegistry.validate_all", f"FAIL: {e}"))

    try:
        reg = CIEnvRegistry()
        snippet = reg.get_env("test_pylibwholegraph").shell_snippet()
        assert "rapids-dependency-file-generator" in snippet
        assert "test_pylibwholegraph" in snippet
        assert "--prepend-channel" in snippet
        assert "rapids-mamba-retry env create" in snippet
        results.append(("CondaEnvSpec.shell_snippet 关键字段", "PASS"))
    except (AssertionError, KeyError) as e:
        results.append(("CondaEnvSpec.shell_snippet", f"FAIL: {e}"))

    print("\n=== conda_ci_env_builder.py 自测 ===")
    all_pass = True
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] {name}")
        if status != "PASS":
            all_pass = False
    print(f"\n{'全部通过' if all_pass else '存在失败项'} ({len(results)} 项)\n")


if __name__ == "__main__":
    _selftest()
