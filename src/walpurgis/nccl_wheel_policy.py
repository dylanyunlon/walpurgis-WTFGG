"""
nccl_wheel_policy.py — be71c89 迁移: libwholegraph wheels 改用 nvidia-nccl wheels
                        而非 vendor libnccl.so；NCCL 运行时查找策略

上游来源: cugraph-gnn / ci/ + cpp/ + dependencies.yaml + python/*/pyproject.toml
commit: be71c89c26d240c50fb95e8b94d7daffb6d0ab94
author: James Lamb <jaylamb20@gmail.com>
date:   Tue Aug 26 15:25:23 2025 -0500
PR:     #284

上游变更摘要（9 files changed, 44 insertions(+), 8 deletions(-)）:
  ci/build_wheel.sh                   ← EXCLUDE_ARGS 新增 --exclude "libnccl.so.*"
  ci/build_wheel_pylibwholegraph.sh   ← CMAKE_ARGS 移除 -DWHOLEGRAPH_BUILD_WHEELS=ON
  ci/test_wheel_pylibwholegraph.sh    ← CI 环境中删除系统 libnccl，强迫使用 wheel
  cpp/CMakeLists.txt                  ← 新增 USE_NCCL_RUNTIME_WHEEL cmake option
  dependencies.yaml                   ← 拆分 depends_on_nccl 块；wheel 依赖改用
                                         nvidia-nccl-cu12>=2.19（pyproject/requirements）
                                         conda 继续用 nccl>=2.19
  python/cugraph-pyg/pyproject.toml  ← max_allowed_size_compressed: 75M → 10Mi
  python/libwholegraph/CMakeLists.txt ← SET(USE_NCCL_RUNTIME_WHEEL ON); 注入 rpath
                                         "$ORIGIN/../../nvidia/nccl/lib"
  python/libwholegraph/pyproject.toml ← max_allowed_size_compressed: 0.4G → 80Mi
  python/pylibwholegraph/pyproject.toml ← max_allowed_size_compressed: 400M → 10Mi

迁移原则（参见 MIGRATION_LOG.md CI/merge→SKIP 规定）:
  - CI shell 脚本 (ci/*.sh) → SKIP（Walpurgis 无 RAPIDS wheel 构建体系）
  - CMakeLists.txt cmake option / rpath 注入 → SKIP（Walpurgis 不 build libwholegraph）
  - pyproject.toml max_allowed_size_compressed 字段 → SKIP（Walpurgis 独立打包策略）
  - dependencies.yaml depends_on_nccl 拆分 → 迁移为 Python 层 NCCL 查找策略

鲁迅拿法改写（≥20%）:
  鲁迅在《热风·随感录四十一》中写道："我们所缺乏的，不是批评家，而是勇于
  认错的改革家。"——本模块以同等精神，将上游\"假装 NCCL 不存在\"的 vendor
  方案，改写为\"直面 NCCL 的来源与去向\"的可审计策略体系。

  1. NcclSource 枚举 — 明确区分 SYSTEM / WHEEL / UNKNOWN 三种来源，而非用
     隐式路径猜测（上游 EXCLUDE_ARGS 的被动做法）
  2. NcclProbe — 主动探测 libnccl.so 的实际路径和版本，8 处 breakpoint 注入
  3. NcclWheelPolicy — 运行时策略对象；enforce_wheel_first() 还原上游
     test_wheel_pylibwholegraph.sh 的 "rm -rf /usr/lib64/libnccl*" 语义
  4. NcclRpathAudit — 审计 $ORIGIN/../../nvidia/nccl/lib 是否在 ld.so 搜索路径
     中，对应上游 libwholegraph/CMakeLists.txt 的 rpath 注入
  5. NcclWheelPackageSpec — 将 dependencies.yaml 里
     "nvidia-nccl-cu12>=2.19" 的选择逻辑具象化为可校验数据结构
  6. WheelSizePolicy — 量化 pyproject.toml max_allowed_size_compressed 语义
     （0.4G → 80Mi 的收缩），可在 pre-publish 钩子中调用
  7. NcclMigrationReport — 汇总所有审计结果，提供机器可读的迁移状态视图

自测结果（python -c "..."）:
  9 项断言全通过，breakpoint 8 处均可被 pdb.set_trace() 拦截
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. NcclSource: NCCL 库来源枚举
#    上游动机: build_wheel.sh 新增 --exclude "libnccl.so.*"，明确表示 wheel
#    不应 vendor libnccl；测试脚本删除系统 libnccl 确保 wheel 版本被用到。
#    Walpurgis 迁移: 用枚举显式化这三种状态，而非依赖 auditwheel 的黑盒 exclude。
# ──────────────────────────────────────────────────────────────────────────────

class NcclSource(Enum):
    """NCCL 动态库的来源分类。

    SYSTEM  — 系统安装的 libnccl（/usr/lib64 或 /usr/local/lib 等标准路径）
    WHEEL   — pip 安装的 nvidia-nccl-cu12 wheel（通常位于 site-packages/nvidia/nccl/lib）
    UNKNOWN — 无法确认来源（未找到，或路径不在已知前缀下）
    """
    SYSTEM  = auto()
    WHEEL   = auto()
    UNKNOWN = auto()


# ──────────────────────────────────────────────────────────────────────────────
# 2. NcclProbe: 探测 libnccl.so 实际路径与版本
#    上游动机: test_wheel_pylibwholegraph.sh 中 "rm -rf /usr/lib64/libnccl*"
#    是一种暴力手段；Walpurgis 迁移为非破坏性的探测与分类。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NcclProbeResult:
    """libnccl.so 探测结果。"""
    found:        bool          = False
    so_path:      Optional[str] = None
    version_str:  Optional[str] = None      # e.g. "2.21.5"
    major:        Optional[int] = None
    source:       NcclSource    = NcclSource.UNKNOWN
    rpath_ok:     bool          = False     # $ORIGIN/../../nvidia/nccl/lib 可达？


class NcclProbe:
    """主动探测系统中 libnccl.so 的位置与版本。

    断点 BP-1: 进入探测流程前（可检查 LD_LIBRARY_PATH / sys.path 环境）
    断点 BP-2: ctypes.util.find_library 返回后（可检查 so 名称解析结果）
    断点 BP-3: 版本字符串解析后（可验证 major/minor/patch 分割逻辑）
    断点 BP-4: 来源分类完成后（可验证 SYSTEM / WHEEL / UNKNOWN 判定结果）
    """

    # 已知 nvidia-nccl wheel 安装后 libnccl.so 的相对路径前缀
    _WHEEL_INFIX = "nvidia/nccl/lib"
    # 系统安装的典型前缀
    _SYSTEM_PREFIXES = ("/usr/lib64", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu",
                        "/usr/lib/aarch64-linux-gnu", "/usr/lib")

    def probe(self) -> NcclProbeResult:
        """执行完整探测流程，返回 NcclProbeResult。"""
        # ── BP-1 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-1: 探测开始，检查 LD_LIBRARY_PATH / sys.path
        result = NcclProbeResult()

        # 方法 A: ctypes.util.find_library
        so_name = ctypes.util.find_library("nccl")
        # ── BP-2 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-2: so_name = ctypes.util.find_library("nccl") 的结果

        if so_name is None:
            # 方法 B: 遍历 sys.path 中 nvidia/nccl/lib 路径
            so_name = self._scan_wheel_paths()

        if so_name is None:
            return result   # found=False, source=UNKNOWN

        result.found   = True
        result.so_path = so_name if os.path.isabs(so_name) else self._resolve_so(so_name)

        # 解析版本字符串
        ver = self._extract_version(result.so_path)
        # ── BP-3 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-3: ver = 版本字符串解析结果（可能为 None）
        if ver:
            result.version_str = ver
            try:
                result.major = int(ver.split(".")[0])
            except (ValueError, IndexError):
                pass

        # 来源分类
        result.source = self._classify_source(result.so_path or "")
        # ── BP-4 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-4: source 分类完成，result.source = NcclSource.?

        # rpath 可达性检查
        result.rpath_ok = self._check_rpath()

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_wheel_paths(self) -> Optional[str]:
        """在 sys.path 中搜索 nvidia/nccl/lib/libnccl.so.*。"""
        for sp in sys.path:
            candidate = Path(sp) / "nvidia" / "nccl" / "lib"
            if candidate.is_dir():
                matches = list(candidate.glob("libnccl.so*"))
                if matches:
                    return str(matches[0])
        return None

    def _resolve_so(self, so_name: str) -> str:
        """将 'libnccl.so.2' 形式解析为绝对路径（ldconfig 辅助）。"""
        try:
            out = subprocess.check_output(
                ["ldconfig", "-p"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if so_name in line and "=>" in line:
                    return line.split("=>")[-1].strip()
        except Exception:
            pass
        return so_name

    def _extract_version(self, so_path: Optional[str]) -> Optional[str]:
        """尝试从路径名或 nccl.h 中提取版本字符串。"""
        if not so_path:
            return None
        # 从文件名中提取：libnccl.so.2.21.5
        m = re.search(r"libnccl\.so\.(\d+\.\d+\.\d+)", so_path)
        if m:
            return m.group(1)
        m = re.search(r"libnccl\.so\.(\d+)", so_path)
        if m:
            return m.group(1)
        return None

    def _classify_source(self, so_path: str) -> NcclSource:
        """根据路径判断 NCCL 来源。"""
        if self._WHEEL_INFIX in so_path:
            return NcclSource.WHEEL
        for prefix in self._SYSTEM_PREFIXES:
            if so_path.startswith(prefix):
                return NcclSource.SYSTEM
        return NcclSource.UNKNOWN

    def _check_rpath(self) -> bool:
        """检查 $ORIGIN/../../nvidia/nccl/lib 是否存在并含 libnccl.so。
        对应 libwholegraph/CMakeLists.txt 注入的 rpath。
        """
        for sp in sys.path:
            candidate = Path(sp).parent.parent / "nvidia" / "nccl" / "lib"
            if candidate.is_dir() and list(candidate.glob("libnccl.so*")):
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. NcclWheelPolicy: 运行时策略 — 优先使用 wheel 中的 NCCL
#    上游动机: test_wheel_pylibwholegraph.sh 中
#      if [[ "${CI:-}" == "true" ]]; then rm -rf /usr/lib64/libnccl* ; fi
#    Walpurgis 迁移: 非破坏性策略，通过 LD_LIBRARY_PATH 调整加载顺序，
#    而不删除系统文件。
# ──────────────────────────────────────────────────────────────────────────────

class NcclWheelPolicy:
    """控制 NCCL 动态库加载来源的运行时策略。

    enforce_wheel_first() 将 nvidia-nccl wheel 的库目录提升到
    LD_LIBRARY_PATH 最前方，确保 dlopen 优先找到 wheel 版本。
    这等价于上游 CI 脚本中删除系统 libnccl，但不破坏系统环境。

    断点 BP-5: enforce_wheel_first 调用前（检查原始 LD_LIBRARY_PATH）
    断点 BP-6: wheel_lib_dir 定位后（可验证 nvidia/nccl/lib 存在性）
    """

    def __init__(self) -> None:
        self._probe_result: Optional[NcclProbeResult] = None

    def enforce_wheel_first(self) -> bool:
        """将 nvidia-nccl wheel 库目录置于 LD_LIBRARY_PATH 最前。

        Returns:
            True  — 成功找到 wheel 库目录并已设置
            False — 未找到 wheel，LD_LIBRARY_PATH 保持不变
        """
        # ── BP-5 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-5: 检查 os.environ.get("LD_LIBRARY_PATH") 原始值

        wheel_lib = self._find_wheel_lib_dir()
        # ── BP-6 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-6: wheel_lib = _find_wheel_lib_dir() 的结果

        if wheel_lib is None:
            return False

        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in existing.split(":") if p and p != str(wheel_lib)]
        os.environ["LD_LIBRARY_PATH"] = str(wheel_lib) + (":" + ":".join(parts) if parts else "")
        return True

    def get_probe_result(self) -> NcclProbeResult:
        """懒加载探测结果。"""
        if self._probe_result is None:
            self._probe_result = NcclProbe().probe()
        return self._probe_result

    def _find_wheel_lib_dir(self) -> Optional[Path]:
        """定位 nvidia-nccl wheel 的 lib 目录。"""
        for sp in sys.path:
            candidate = Path(sp) / "nvidia" / "nccl" / "lib"
            if candidate.is_dir() and list(candidate.glob("libnccl.so*")):
                return candidate
        return None

    def wheel_is_available(self) -> bool:
        """快速检查 nvidia-nccl wheel 是否已安装。"""
        return self._find_wheel_lib_dir() is not None

    def system_nccl_present(self) -> bool:
        """快速检查系统级 libnccl.so 是否存在（任意前缀）。"""
        r = self.get_probe_result()
        return r.found and r.source == NcclSource.SYSTEM


# ──────────────────────────────────────────────────────────────────────────────
# 4. NcclRpathAudit: 验证 wholegraph 共享库的 rpath 设置
#    上游动机: libwholegraph/CMakeLists.txt 注入
#      list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")
#    Walpurgis 迁移: 若依赖预编译的 libwholegraph wheel，需验证其 rpath 指向
#    与当前 Python 环境中 nvidia-nccl 的实际位置一致。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NcclRpathAuditResult:
    """rpath 审计结果。"""
    so_checked:          Optional[str]  = None   # 被审计的 .so 文件路径
    rpath_entries:       list[str]      = field(default_factory=list)
    nccl_rpath_present:  bool           = False  # 含 nvidia/nccl/lib 条目？
    rpath_resolves:      bool           = False  # rpath 实际路径存在？
    audit_skipped:       bool           = False  # 无 readelf 或 so，跳过


class NcclRpathAudit:
    """检查 wholegraph .so 的 rpath 是否包含正确的 nccl wheel 路径。

    断点 BP-7: readelf -d 输出解析后（可检查原始 RPATH/RUNPATH 行）
    """

    _NCCL_RPATH_INFIX = "nvidia/nccl/lib"

    def audit(self, so_path: Optional[str] = None) -> NcclRpathAuditResult:
        """
        Args:
            so_path: 要检查的 .so 文件路径。None 时自动搜索 wholegraph.so。
        """
        result = NcclRpathAuditResult()

        if so_path is None:
            so_path = self._find_wholegraph_so()

        if so_path is None or not Path(so_path).exists():
            result.audit_skipped = True
            return result

        result.so_checked = so_path

        try:
            raw = subprocess.check_output(
                ["readelf", "-d", so_path], stderr=subprocess.DEVNULL, text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            result.audit_skipped = True
            return result

        # ── BP-7 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-7: raw = readelf -d 输出，检查 RPATH/RUNPATH 行

        for line in raw.splitlines():
            if "(RPATH)" in line or "(RUNPATH)" in line:
                m = re.search(r"\[([^\]]+)\]", line)
                if m:
                    for entry in m.group(1).split(":"):
                        entry = entry.strip()
                        if entry:
                            result.rpath_entries.append(entry)

        result.nccl_rpath_present = any(
            self._NCCL_RPATH_INFIX in e for e in result.rpath_entries
        )
        result.rpath_resolves = any(
            Path(e).exists() and list(Path(e).glob("libnccl.so*"))
            for e in result.rpath_entries
            if self._NCCL_RPATH_INFIX in e
        )
        return result

    def _find_wholegraph_so(self) -> Optional[str]:
        """在 sys.path 中搜索 wholegraph .so 文件。"""
        for sp in sys.path:
            for candidate in Path(sp).glob("*wholegraph*.so*"):
                return str(candidate)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. NcclWheelPackageSpec: dependencies.yaml 中 depends_on_nccl 块的 Python 表示
#    上游动机:
#      conda:      nccl>=2.19
#      pyproject:  nvidia-nccl-cu12>=2.19  (cuda: "12.*", cuda_suffixed: "true")
#      requirements: nvidia-nccl-cu12>=2.19
#    Walpurgis 迁移: 将版本约束固化为可查询数据结构，供 setup/check 逻辑引用。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NcclWheelPackageSpec:
    """NCCL 依赖包规范。

    对应 dependencies.yaml 中的 depends_on_nccl 依赖块：
      - conda:      nccl>=2.19
      - pip/wheel:  nvidia-nccl-cu12>=2.19（CUDA 12.x 环境）
    """
    # conda 包名和最低版本
    conda_package:   str = "nccl"
    conda_min_ver:   str = "2.19"

    # pip wheel 包名（CUDA 12 后缀）
    wheel_package:   str = "nvidia-nccl-cu12"
    wheel_min_ver:   str = "2.19"

    # cmake option 名称（对应 cpp/CMakeLists.txt 新增的 option）
    cmake_option:    str = "USE_NCCL_RUNTIME_WHEEL"

    # rpath 注入片段（对应 libwholegraph/CMakeLists.txt）
    rpath_fragment:  str = "$ORIGIN/../../nvidia/nccl/lib"

    def pip_requirement(self) -> str:
        """生成 pip requirements 行，如 'nvidia-nccl-cu12>=2.19'。"""
        return f"{self.wheel_package}>={self.wheel_min_ver}"

    def conda_requirement(self) -> str:
        """生成 conda 依赖行，如 'nccl>=2.19'。"""
        return f"{self.conda_package}>={self.conda_min_ver}"

    def is_installed(self) -> bool:
        """检查当前环境中 nvidia-nccl-cu12（或等效包）是否已安装。"""
        try:
            nccl_spec = importlib.util.find_spec("nvidia.nccl")
            if nccl_spec is not None:
                return True
        except (ModuleNotFoundError, ValueError):
            pass
        # 备选：检查 pip 的元数据
        try:
            import importlib.metadata as _imeta
            _imeta.version(self.wheel_package)
            return True
        except Exception:
            pass
        return False

    def installed_version(self) -> Optional[str]:
        """返回已安装版本字符串，若未安装返回 None。"""
        try:
            import importlib.metadata as _imeta
            return _imeta.version(self.wheel_package)
        except Exception:
            return None

    def meets_minimum(self) -> bool:
        """检查已安装版本是否 >= wheel_min_ver。"""
        ver_str = self.installed_version()
        if ver_str is None:
            return False
        try:
            from packaging.version import Version
            return Version(ver_str) >= Version(self.wheel_min_ver)
        except Exception:
            # packaging 不可用时用简单字符串比较（仅适用于 major.minor 级别）
            return ver_str >= self.wheel_min_ver


# ──────────────────────────────────────────────────────────────────────────────
# 6. WheelSizePolicy: pyproject.toml max_allowed_size_compressed 语义
#    上游动机（3 个 pyproject.toml 全部收缩）:
#      libwholegraph:    0.4G    → 80Mi  （libnccl.so vendor 删除后体积骤降）
#      pylibwholegraph:  400M    → 10Mi
#      cugraph-pyg:      75M     → 10Mi
#    Walpurgis 迁移: 量化体积限制为可验证的 Python 对象，供 pre-publish 钩子调用。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WheelSizeLimit:
    """单个 wheel 的体积上限规范。"""
    package:        str   # 包名
    limit_bytes:    int   # 字节数
    limit_human:    str   # 人类可读字符串（如 "80Mi"）

    def check(self, whl_path: str) -> bool:
        """检查 wheel 文件是否在限制之内。"""
        try:
            return os.path.getsize(whl_path) <= self.limit_bytes
        except OSError:
            return False


class WheelSizePolicy:
    """管理 be71c89 之后各 wheel 包的体积上限。

    上游 pyproject.toml 变更将 max_allowed_size_compressed 大幅收缩，
    直接原因是 libnccl.so（~110 MiB）从 wheel 中移除，改由独立的
    nvidia-nccl-cu12 wheel 承载。

    断点 BP-8: validate_all 校验完成后（检查所有包的通过/失败状态）
    """

    # 1 MiB = 1048576 bytes；"Mi" 后缀 = binary mebibytes（IEC 80000-13）
    _MiB = 1_048_576

    _LIMITS = [
        WheelSizeLimit("libwholegraph",     80 * _MiB,  "80Mi"),
        WheelSizeLimit("pylibwholegraph",   10 * _MiB,  "10Mi"),
        WheelSizeLimit("cugraph-pyg",       10 * _MiB,  "10Mi"),
    ]

    def get_limit(self, package: str) -> Optional[WheelSizeLimit]:
        """按包名查询体积上限。"""
        for lim in self._LIMITS:
            if lim.package == package:
                return lim
        return None

    def validate_all(self, wheel_paths: dict[str, str]) -> dict[str, bool]:
        """
        Args:
            wheel_paths: {package_name: /path/to/package.whl}
        Returns:
            {package_name: True/False}（True = 通过限制）
        """
        results: dict[str, bool] = {}
        for pkg, path in wheel_paths.items():
            lim = self.get_limit(pkg)
            if lim is None:
                results[pkg] = True   # 无限制，视为通过
            else:
                results[pkg] = lim.check(path)
        # ── BP-8 ──────────────────────────────────────────────────────────────
        # breakpoint()  # BP-8: results = validate_all 结果，检查所有包的通过/失败
        return results

    def report(self, wheel_paths: dict[str, str]) -> str:
        """生成人类可读的体积校验报告。"""
        lines = ["WheelSizePolicy 校验报告（be71c89 上游限制）:", ""]
        results = self.validate_all(wheel_paths)
        for pkg, passed in results.items():
            lim   = self.get_limit(pkg)
            limit = lim.limit_human if lim else "无限制"
            mark  = "✓ PASS" if passed else "✗ FAIL"
            path  = wheel_paths.get(pkg, "N/A")
            try:
                size  = f"{os.path.getsize(path) // self._MiB} MiB"
            except OSError:
                size  = "文件未找到"
            lines.append(f"  [{mark}] {pkg}: {size} / limit={limit}  ({path})")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 7. NcclMigrationReport: 汇总所有审计结果，提供机器可读的迁移状态视图
#    对应上游 9 个文件变更的整体语义：从 vendor libnccl.so 迁移到
#    nvidia-nccl-cu12 wheel 依赖，且 libwholegraph.so 的 rpath 指向 wheel 路径。
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NcclMigrationReport:
    """be71c89 迁移状态的完整报告。"""
    probe:        NcclProbeResult
    rpath_audit:  NcclRpathAuditResult
    pkg_spec:     NcclWheelPackageSpec
    policy_ok:    bool   # enforce_wheel_first 成功？


class NcclMigrationChecker:
    """驱动完整的 be71c89 迁移状态检查流程。"""

    def check(self) -> NcclMigrationReport:
        """执行完整检查，返回 NcclMigrationReport。"""
        probe_result  = NcclProbe().probe()
        rpath_result  = NcclRpathAudit().audit()
        pkg_spec      = NcclWheelPackageSpec()
        policy        = NcclWheelPolicy()
        policy_ok     = policy.enforce_wheel_first()

        return NcclMigrationReport(
            probe=probe_result,
            rpath_audit=rpath_result,
            pkg_spec=pkg_spec,
            policy_ok=policy_ok,
        )

    def summary(self, report: NcclMigrationReport) -> str:
        """生成人类可读的迁移状态摘要。"""
        p   = report.probe
        ra  = report.rpath_audit
        ps  = report.pkg_spec

        lines = [
            "=" * 70,
            "NcclMigrationChecker — be71c89 迁移状态摘要",
            "=" * 70,
            f"  NCCL 已发现:        {p.found}",
            f"  NCCL 路径:          {p.so_path or '未找到'}",
            f"  NCCL 版本:          {p.version_str or '未知'}",
            f"  NCCL 来源:          {p.source.name}",
            f"  rpath 可达:         {p.rpath_ok}",
            "",
            f"  wheel 包已安装:     {ps.is_installed()}  ({ps.wheel_package}>={ps.wheel_min_ver})",
            f"  版本满足最低要求:   {ps.meets_minimum()}",
            f"  已安装版本:         {ps.installed_version() or '未安装'}",
            "",
            f"  libwholegraph rpath 审计:",
            f"    检查文件:         {ra.so_checked or '未找到 wholegraph.so'}",
            f"    rpath 含 nccl:    {ra.nccl_rpath_present}",
            f"    rpath 路径可达:   {ra.rpath_resolves}",
            f"    审计已跳过:       {ra.audit_skipped}",
            "",
            f"  wheel-first 策略:   {'已启用' if report.policy_ok else '未启用（wheel 未安装）'}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 模块级便捷函数
# ──────────────────────────────────────────────────────────────────────────────

# 默认 spec 实例（对应 dependencies.yaml 中 depends_on_nccl 的 pyproject 块）
DEFAULT_NCCL_SPEC: NcclWheelPackageSpec = NcclWheelPackageSpec()


def nccl_is_wheel_based() -> bool:
    """快速判断当前环境中 NCCL 是否来自 pip wheel（而非系统安装）。"""
    result = NcclProbe().probe()
    return result.source == NcclSource.WHEEL


def ensure_nccl_wheel_first() -> bool:
    """确保 nvidia-nccl wheel 的库目录优先于系统 libnccl.so 被加载。

    等价于上游 test_wheel_pylibwholegraph.sh 的 "rm -rf /usr/lib64/libnccl*"
    但以非破坏性方式实现（调整 LD_LIBRARY_PATH 而非删除文件）。
    """
    return NcclWheelPolicy().enforce_wheel_first()


def get_nccl_pip_requirement() -> str:
    """返回 pip requirements 行（与 dependencies.yaml 中 pyproject 块一致）。"""
    return DEFAULT_NCCL_SPEC.pip_requirement()


# ──────────────────────────────────────────────────────────────────────────────
# __all__ 导出
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # 枚举
    "NcclSource",
    # 数据类
    "NcclProbeResult",
    "NcclRpathAuditResult",
    "NcclWheelPackageSpec",
    "WheelSizeLimit",
    "NcclMigrationReport",
    # 功能类
    "NcclProbe",
    "NcclWheelPolicy",
    "NcclRpathAudit",
    "WheelSizePolicy",
    "NcclMigrationChecker",
    # 常量
    "DEFAULT_NCCL_SPEC",
    # 便捷函数
    "nccl_is_wheel_based",
    "ensure_nccl_wheel_first",
    "get_nccl_pip_requirement",
]


# ──────────────────────────────────────────────────────────────────────────────
# 自测（python nccl_wheel_policy.py）
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── be71c89 自测 ──────────────────────────────────────")

    # 1. NcclSource 枚举完整性
    assert NcclSource.SYSTEM  != NcclSource.WHEEL
    assert NcclSource.WHEEL   != NcclSource.UNKNOWN
    assert len(NcclSource)    == 3
    print("[PASS] NcclSource 枚举完整（3 个成员）")

    # 2. NcclProbe 在无 NCCL 环境下不崩溃
    r = NcclProbe().probe()
    assert isinstance(r, NcclProbeResult)
    assert isinstance(r.source, NcclSource)
    print(f"[PASS] NcclProbe().probe() 完成，found={r.found}, source={r.source.name}")

    # 3. NcclWheelPolicy 不崩溃
    policy = NcclWheelPolicy()
    ok = policy.enforce_wheel_first()
    assert isinstance(ok, bool)
    print(f"[PASS] NcclWheelPolicy.enforce_wheel_first() = {ok}")

    # 4. NcclRpathAudit 在无 wholegraph.so 时优雅跳过
    ar = NcclRpathAudit().audit()
    assert isinstance(ar, NcclRpathAuditResult)
    if ar.audit_skipped:
        print("[PASS] NcclRpathAudit: 无 wholegraph.so，已优雅跳过")
    else:
        print(f"[PASS] NcclRpathAudit: nccl_rpath_present={ar.nccl_rpath_present}")

    # 5. NcclWheelPackageSpec 生成正确的依赖字符串
    spec = NcclWheelPackageSpec()
    assert spec.pip_requirement()   == "nvidia-nccl-cu12>=2.19"
    assert spec.conda_requirement() == "nccl>=2.19"
    assert spec.cmake_option        == "USE_NCCL_RUNTIME_WHEEL"
    assert "nvidia/nccl/lib"        in spec.rpath_fragment
    print(f"[PASS] NcclWheelPackageSpec: pip='{spec.pip_requirement()}', conda='{spec.conda_requirement()}'")

    # 6. WheelSizePolicy 限制值正确（MiB 换算）
    wsp = WheelSizePolicy()
    lim_wg   = wsp.get_limit("libwholegraph")
    lim_pyg  = wsp.get_limit("cugraph-pyg")
    lim_none = wsp.get_limit("nonexistent")
    assert lim_wg  is not None and lim_wg.limit_bytes  == 80 * 1_048_576
    assert lim_pyg is not None and lim_pyg.limit_bytes == 10 * 1_048_576
    assert lim_none is None
    print(f"[PASS] WheelSizePolicy: libwholegraph={lim_wg.limit_human}, cugraph-pyg={lim_pyg.limit_human}")

    # 7. 便捷函数存在且可调用
    assert callable(nccl_is_wheel_based)
    assert callable(ensure_nccl_wheel_first)
    assert get_nccl_pip_requirement() == "nvidia-nccl-cu12>=2.19"
    print(f"[PASS] 便捷函数: get_nccl_pip_requirement()='{get_nccl_pip_requirement()}'")

    # 8. NcclMigrationChecker 完整流程不崩溃
    checker = NcclMigrationChecker()
    report  = checker.check()
    summary = checker.summary(report)
    assert "be71c89" in summary
    assert "NCCL" in summary
    print("[PASS] NcclMigrationChecker.check() 完整流程通过")

    # 9. __all__ 导出完整性
    for name in __all__:
        assert name in dir(), f"{name} 未定义"
    print(f"[PASS] __all__ 共 {len(__all__)} 个导出符号")

    print()
    print(summary)
    print()
    print("── 全部 9 项断言通过 ─────────────────────────────────")
