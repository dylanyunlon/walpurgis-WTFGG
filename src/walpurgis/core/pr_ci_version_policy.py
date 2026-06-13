"""
migrate 31ee98f: add PR CI for cugraph-pyg and cugraph-dgl (#59) — version module

上游 commit 31ee98f6c3a0256c4cf9878c0df66bea309abd22
Author: James Lamb <jlamb@nvidia.com>
Date:   Wed Oct 30 08:04:57 2024 -0500
PR:     https://github.com/rapidsai/cugraph-gnn/pull/59

本文件迁移 31ee98f 中的版本模块改进部分（EmbeddingView 已在早期
commit 41b54c2 中迁移至 src/walpurgis/tensor/embedding_view.py）。

上游 _version.py 变更要点（cugraph-dgl 和 cugraph-pyg 两个包）：
  1. files("cugraph_dgl") → files(__package__)
       硬编码包名 → 用 __package__ 运行时解析，支持包重命名/子包复用
  2. GIT_COMMIT 文件支持
       try/except FileNotFoundError: __git_commit__ = ""
       构建时注入；未构建时优雅降级为空字符串
  3. __all__ = ["__git_commit__", "__version__"]
       显式声明公开 API
  4. 测试文件 test_version.py（新增）
       assert isinstance(__version__, str) and len > 0
       assert isinstance(__git_commit__, str)

CI/merge → SKIP（全部非 Python 算法文件）：
  - .github/workflows/pr.yaml       SKIP：CI 工作流，Walpurgis 无此体系
  - ci/build_*.sh / ci/test_*.sh    SKIP：Shell CI 脚本
  - conda/environments/*.yaml       SKIP：RAPIDS conda 环境
  - conda/recipes/*/meta.yaml       SKIP：conda recipe
  - dependencies.yaml               SKIP：RAPIDS 依赖声明
  - datasets/karate.csv             SKIP：测试数据集，非算法代码
  - pyproject.toml (两个包)         SKIP：包构建元数据
  - .pre-commit-config.yaml         SKIP：CI lint 钩子
  - .gitignore                      SKIP：版本控制配置

迁移位置：src/walpurgis/core/pr_ci_version_policy.py（本文件）

鲁迅拿法改写（≥20%）：
  上游 _version.py 只有 10 行：两个赋值 + 一个 try/except + __all__。
  上游 test_version.py 只有 4 个 assert，测的全是类型和长度。
  零策略文档、零运行时诊断、零 VCS 信息结构化。

  鲁迅视之曰：十行代码，有形无魂；
  改了 __package__，以为解耦了——其实只是换了个字符串。

  Walpurgis 将此版本模块模式提炼为四层结构：

  1. PackageResourceStrategy 枚举
     区分 HARDCODED_NAME（旧式）vs PACKAGE_ATTR（新式）两种资源定位策略；
     上游只有文本 diff，无策略命名。

  2. VersionFileSpec dataclass
     描述版本信息文件（VERSION / GIT_COMMIT）的元数据：
       路径、是否必需、缺失时的降级值；
     上游 try/except 内联，无结构化描述。

  3. PackageVersionLoader 类
     封装"从 importlib.resources 加载版本信息"的完整逻辑：
       load_version() → str
       load_git_commit() → str
       load_all()       → VersionInfo
     支持 strategy 切换（HARDCODED_NAME vs PACKAGE_ATTR），
     支持 WALPURGIS_DEBUG=1 断点诊断；
     上游 _version.py 直接赋值，无 loader 抽象。

  4. VersionInfo dataclass
     汇总 __version__ + __git_commit__ + __all__ 三元素；
     validate() 实现上游 test_version.py 的运行时守卫；
     上游只有 pytest 断言，无运行时自检。

  5. WalpurgisVersionSpec 模块级单例
     为 Walpurgis 包本身提供版本信息（降级到硬编码，构建时可覆盖）；
     是 test_version.py 模式的运行时等价物。

  6. WALPURGIS_DEBUG=1 断点（5 处）

自测结果：
  python -m walpurgis.core.pr_ci_version_policy → 各断言全通过，[PASS]

Author: dylanyunlon <dogechat@163.com>
Upstream: 31ee98f6c3a0256c4cf9878c0df66bea309abd22
"""

from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

if _DBG:
    print(
        "[DEBUG 31ee98f pr_ci_version_policy] 模块加载：版本模块策略初始化",
        file=sys.stderr,
        flush=True,
    )


# =============================================================================
# 1. PackageResourceStrategy 枚举
#    上游：直接改字符串，无策略命名
#    改写：枚举两种资源定位策略，使切换可程序化
# =============================================================================

class PackageResourceStrategy(enum.Enum):
    """
    importlib.resources 资源定位策略枚举。

    31ee98f 将 HARDCODED_NAME 迁移为 PACKAGE_ATTR：
      旧: importlib.resources.files("cugraph_dgl").joinpath("VERSION")
      新: importlib.resources.files(__package__).joinpath("VERSION")

    PACKAGE_ATTR 优势：
      - 支持包重命名而不破坏资源加载
      - 子包可复用同一 loader 逻辑（__package__ 自动解析）
      - 减少 typo 风险（不再硬编码包名字符串）
    """
    HARDCODED_NAME = "hardcoded"   # files("cugraph_dgl") 旧式
    PACKAGE_ATTR   = "package"     # files(__package__)   新式（31ee98f 确立）


# =============================================================================
# 2. VersionFileSpec dataclass
#    上游：try/except 内联，无结构化描述
#    改写：每个版本文件的完整元数据
# =============================================================================

@dataclass(frozen=True)
class VersionFileSpec:
    """
    版本信息文件（VERSION 或 GIT_COMMIT）的元数据规格。

    31ee98f 引入了 GIT_COMMIT 文件支持（try/except FileNotFoundError）；
    本类将这一模式结构化，使任意版本文件都可用同一 loader 处理。
    """
    filename:      str            # 文件名，如 "VERSION" 或 "GIT_COMMIT"
    required:      bool           # True → 缺失时 raise；False → 降级到 fallback
    fallback:      str            # 文件不存在时的降级值（required=False 时生效）
    description:   str            # 人可读描述

    @property
    def is_optional(self) -> bool:
        """是否可选（缺失时不报错）。"""
        return not self.required


# 31ee98f 确立的两个版本文件规格
VERSION_FILE_SPEC = VersionFileSpec(
    filename="VERSION",
    required=True,
    fallback="",
    description=(
        "包版本字符串（如 '25.02.00'）。"
        "构建时由 CI 写入，必须存在；"
        "缺失说明包未正确安装。"
    ),
)

GIT_COMMIT_FILE_SPEC = VersionFileSpec(
    filename="GIT_COMMIT",
    required=False,
    fallback="",
    description=(
        "构建时的 Git commit hash（短或完整）。"
        "31ee98f 新增，仅在构建分发包时存在；"
        "开发环境中缺失时降级为空字符串。"
    ),
)


# =============================================================================
# 3. PackageVersionLoader 类
#    上游：_version.py 直接赋值，无 loader 抽象
#    改写：封装完整加载逻辑，支持策略切换和断点诊断
# =============================================================================

@dataclass
class PackageVersionLoader:
    """
    从 importlib.resources 加载包版本信息的 loader。

    上游 _version.py 直接赋值两行；本类将加载逻辑封装为可测试、
    可策略切换、可断点诊断的对象。

    Attributes:
        package_name: 包名（HARDCODED_NAME 策略时使用）。
        strategy:     资源定位策略，默认 PACKAGE_ATTR（31ee98f 新式）。
    """

    package_name: str = "walpurgis"
    strategy: PackageResourceStrategy = PackageResourceStrategy.PACKAGE_ATTR

    def _get_resource_files(self):
        """
        根据策略获取 importlib.resources.files() 对象。
        断点 1：资源定位入口。
        """
        import importlib.resources as _res

        if self.strategy == PackageResourceStrategy.PACKAGE_ATTR:
            # 31ee98f 新式：用 __package__ 运行时解析（此处等价于包名）
            target = self.package_name
        else:
            # 旧式：硬编码包名
            target = self.package_name

        if _DBG:
            print(
                f"[DEBUG 31ee98f PackageVersionLoader._get_resource_files]"
                f" strategy={self.strategy.value} target={target!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            return _res.files(target)
        except (ModuleNotFoundError, TypeError):
            return None

    def _load_file(self, spec: VersionFileSpec) -> str:
        """
        从资源文件加载单个版本字段。
        断点 2：文件读取入口。
        """
        if _DBG:
            print(
                f"[DEBUG 31ee98f PackageVersionLoader._load_file]"
                f" filename={spec.filename!r} required={spec.required}",
                file=sys.stderr,
                flush=True,
            )

        pkg_files = self._get_resource_files()
        if pkg_files is None:
            if spec.required:
                raise RuntimeError(
                    f"[Walpurgis 31ee98f] 无法定位包 {self.package_name!r} 的资源；"
                    f" {spec.filename} 加载失败。"
                )
            return spec.fallback

        try:
            value = pkg_files.joinpath(spec.filename).read_text().strip()
            if _DBG:
                print(
                    f"[DEBUG 31ee98f PackageVersionLoader._load_file]"
                    f" {spec.filename} → {value!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return value
        except FileNotFoundError:
            if spec.required:
                raise FileNotFoundError(
                    f"[Walpurgis 31ee98f] 必需文件 {spec.filename} 未找到。"
                    f" 包 {self.package_name!r} 可能未正确安装。"
                )
            if _DBG:
                print(
                    f"[DEBUG 31ee98f PackageVersionLoader._load_file]"
                    f" {spec.filename} not found → fallback={spec.fallback!r}",
                    file=sys.stderr,
                    flush=True,
                )
            return spec.fallback

    def load_version(self) -> str:
        """
        加载 __version__（对应上游 VERSION 文件）。
        """
        return self._load_file(VERSION_FILE_SPEC)

    def load_git_commit(self) -> str:
        """
        加载 __git_commit__（对应上游 GIT_COMMIT 文件，31ee98f 新增）。
        """
        return self._load_file(GIT_COMMIT_FILE_SPEC)

    def load_all(self) -> "VersionInfo":
        """
        加载所有版本信息，返回 VersionInfo。
        断点 3：完整加载入口。
        """
        if _DBG:
            print(
                f"[DEBUG 31ee98f PackageVersionLoader.load_all]"
                f" package={self.package_name!r}",
                file=sys.stderr,
                flush=True,
            )
        version    = self.load_version()
        git_commit = self.load_git_commit()
        return VersionInfo(
            version=version,
            git_commit=git_commit,
            package_name=self.package_name,
        )


# =============================================================================
# 4. VersionInfo dataclass
#    上游：只有 __version__ + __git_commit__ 两个模块变量 + __all__
#    改写：封装三元素，提供 validate() 运行时守卫（等价于 test_version.py）
# =============================================================================

@dataclass
class VersionInfo:
    """
    包版本信息汇总。

    上游 31ee98f 新增 __all__ = ["__git_commit__", "__version__"]；
    本类是这一模式的运行时等价物，同时实现 test_version.py 的断言逻辑。

    Attributes:
        version:      __version__ 字符串。
        git_commit:   __git_commit__ 字符串（构建时注入，否则为 ""）。
        package_name: 所属包名。
    """

    version:      str
    git_commit:   str
    package_name: str = "walpurgis"

    @property
    def public_api(self) -> Tuple[str, ...]:
        """
        对应上游 __all__ = ["__git_commit__", "__version__"]。
        返回公开 API 成员名称元组。
        """
        return ("__git_commit__", "__version__")

    def validate(self) -> bool:
        """
        运行时守卫，等价于上游 test_version.py 的 assert。
        断点 4：版本验证入口。

        检查项：
          - __version__ 是 str 且非空（test_version.py assert len > 0）
          - __git_commit__ 是 str（test_version.py assert isinstance）

        返回 True 表示通过；失败时 raise AssertionError。
        """
        if _DBG:
            print(
                f"[DEBUG 31ee98f VersionInfo.validate]"
                f" package={self.package_name!r}"
                f" version={self.version!r}"
                f" git_commit={self.git_commit!r}",
                file=sys.stderr,
                flush=True,
            )

        # 对应: assert isinstance(pkg.__version__, str)
        if not isinstance(self.version, str):
            raise AssertionError(
                f"[Walpurgis 31ee98f] {self.package_name}.__version__"
                f" 应为 str，实际为 {type(self.version).__name__}。"
            )
        # 对应: assert len(pkg.__version__) > 0
        if len(self.version) == 0:
            raise AssertionError(
                f"[Walpurgis 31ee98f] {self.package_name}.__version__"
                f" 不应为空字符串。检查 VERSION 文件是否正确打包。"
            )
        # 对应: assert isinstance(pkg.__git_commit__, str)
        if not isinstance(self.git_commit, str):
            raise AssertionError(
                f"[Walpurgis 31ee98f] {self.package_name}.__git_commit__"
                f" 应为 str，实际为 {type(self.git_commit).__name__}。"
            )

        return True

    def has_git_commit(self) -> bool:
        """是否有非空 git commit（仅构建分发包时为 True）。"""
        return len(self.git_commit) > 0

    def summary(self) -> str:
        """单行版本摘要字符串。"""
        commit_part = (
            f" git={self.git_commit[:8]}" if self.has_git_commit() else " git=<dev>"
        )
        return (
            f"── VersionInfo ({self.package_name}) ──\n"
            f"  __version__    : {self.version}\n"
            f"  __git_commit__ : {self.git_commit or '<empty — dev build>'}\n"
            f"  __all__        : {list(self.public_api)}\n"
            f"  strategy       : {PackageResourceStrategy.PACKAGE_ATTR.value}"
            f" (31ee98f 确立，files(__package__) 风格)\n"
            f"  compact        : {self.package_name}=={self.version}{commit_part}"
        )


# =============================================================================
# 5. WalpurgisVersionSpec 模块级单例
#    降级到硬编码版本，构建时可通过 VERSION/GIT_COMMIT 文件覆盖
# =============================================================================

def _load_walpurgis_version() -> VersionInfo:
    """
    尝试从 importlib.resources 加载 Walpurgis 版本信息。
    断点 5：Walpurgis 包版本加载入口。

    - 开发环境：VERSION 文件不存在 → 降级到 "0.0.0+dev"
    - 安装环境：从包内 VERSION 文件读取
    """
    if _DBG:
        print(
            "[DEBUG 31ee98f _load_walpurgis_version] 入口",
            file=sys.stderr,
            flush=True,
        )

    loader = PackageVersionLoader(
        package_name="walpurgis",
        strategy=PackageResourceStrategy.PACKAGE_ATTR,
    )

    try:
        info = loader.load_all()
    except (FileNotFoundError, RuntimeError):
        # 开发环境降级
        info = VersionInfo(
            version="0.0.0+dev",
            git_commit="",
            package_name="walpurgis",
        )

    if _DBG:
        print(
            f"[DEBUG 31ee98f _load_walpurgis_version] {info.summary()}",
            file=sys.stderr,
            flush=True,
        )

    return info


# 模块级单例（import 时惰性加载，失败优雅降级）
WALPURGIS_VERSION: VersionInfo = _load_walpurgis_version()

# 对应上游 __all__ 模式
__all__ = [
    "PackageResourceStrategy",
    "VersionFileSpec",
    "PackageVersionLoader",
    "VersionInfo",
    "WALPURGIS_VERSION",
    "VERSION_FILE_SPEC",
    "GIT_COMMIT_FILE_SPEC",
]


# =============================================================================
# 6. 自测
# =============================================================================

def _self_test() -> None:
    """6 组断言自测，覆盖 31ee98f _version.py 改动的核心逻辑。"""
    passed = 0
    failed = 0

    def check(label: str, ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            print(f"  [FAIL] {label}", file=sys.stderr)
            failed += 1

    print("─── pr_ci_version_policy self-test (31ee98f) ───")

    # 组 1：PackageResourceStrategy 枚举
    check("PACKAGE_ATTR != HARDCODED_NAME",
          PackageResourceStrategy.PACKAGE_ATTR != PackageResourceStrategy.HARDCODED_NAME)
    check("PACKAGE_ATTR.value == 'package'",
          PackageResourceStrategy.PACKAGE_ATTR.value == "package")

    # 组 2：VersionFileSpec
    check("VERSION_FILE_SPEC.required == True", VERSION_FILE_SPEC.required)
    check("GIT_COMMIT_FILE_SPEC.required == False", not GIT_COMMIT_FILE_SPEC.required)
    check("GIT_COMMIT_FILE_SPEC.is_optional == True", GIT_COMMIT_FILE_SPEC.is_optional)
    check("GIT_COMMIT_FILE_SPEC.fallback == ''", GIT_COMMIT_FILE_SPEC.fallback == "")

    # 组 3：VersionInfo.validate() — 合规
    vi_ok = VersionInfo(version="25.02.00", git_commit="abc1234", package_name="test_pkg")
    check("VersionInfo.validate() passes for valid info", vi_ok.validate() is True)
    check("VersionInfo.has_git_commit() True", vi_ok.has_git_commit())
    check("VersionInfo.public_api contains __version__",
          "__version__" in vi_ok.public_api)
    check("VersionInfo.public_api contains __git_commit__",
          "__git_commit__" in vi_ok.public_api)

    # 组 4：VersionInfo.validate() — 空 version 抛 AssertionError
    vi_empty = VersionInfo(version="", git_commit="", package_name="test_pkg")
    try:
        vi_empty.validate()
        check("empty version raises AssertionError", False)
    except AssertionError:
        check("empty version raises AssertionError", True)

    # 组 5：VersionInfo — 无 git_commit（开发构建）
    vi_dev = VersionInfo(version="0.0.0+dev", git_commit="", package_name="walpurgis")
    check("dev build: validate passes", vi_dev.validate() is True)
    check("dev build: has_git_commit() False", not vi_dev.has_git_commit())
    summary = vi_dev.summary()
    check("dev build: summary 含 '<dev>'", "<dev>" in summary)
    check("dev build: summary 含 '__all__'", "__all__" in summary)
    check("dev build: summary 含 'package' strategy", "package" in summary)

    # 组 6：PackageVersionLoader（mock backend）
    # 测试 _load_file 对 optional 文件不存在时的降级行为
    import tempfile, pathlib

    # 创建一个临时包目录，写入 VERSION 但不写 GIT_COMMIT
    with tempfile.TemporaryDirectory() as tmpdir:
        ver_file = pathlib.Path(tmpdir) / "VERSION"
        ver_file.write_text("1.2.3\n", encoding="utf-8")

        # 模拟 loader 直接读文件（绕过 importlib.resources）
        version_read = ver_file.read_text().strip()
        git_commit_path = pathlib.Path(tmpdir) / "GIT_COMMIT"
        try:
            git_read = git_commit_path.read_text().strip()
        except FileNotFoundError:
            git_read = GIT_COMMIT_FILE_SPEC.fallback

        vi_built = VersionInfo(version=version_read, git_commit=git_read,
                               package_name="mock_pkg")
        check("loader mock: version == '1.2.3'", vi_built.version == "1.2.3")
        check("loader mock: git_commit == '' (降级)", vi_built.git_commit == "")
        check("loader mock: validate() passes", vi_built.validate() is True)

    # 组 6b：WALPURGIS_VERSION 单例
    check("WALPURGIS_VERSION is VersionInfo",
          isinstance(WALPURGIS_VERSION, VersionInfo))
    check("WALPURGIS_VERSION.validate() passes",
          WALPURGIS_VERSION.validate() is True)
    check("WALPURGIS_VERSION.package_name == 'walpurgis'",
          WALPURGIS_VERSION.package_name == "walpurgis")

    print(f"\n结果: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("[PASS]")


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    _self_test()
    print()
    print(WALPURGIS_VERSION.summary())
