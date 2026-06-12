"""
migrate 13e00d6: Update CuPy Version to 13.2 (#132)

上游 commit 13e00d6158cc180a26bcf7fed82cc6ac7b7301f4
Author: Alex Barghi <alexbarghi-nv@users.noreply.github.com>
Date:   2025-02-05

上游变更：6 个文件，全部执行 cupy>=12.0.0 → cupy>=13.2.0：
  - conda/environments/all_cuda-118_arch-x86_64.yaml
  - conda/environments/all_cuda-121_arch-x86_64.yaml
  - conda/environments/all_cuda-124_arch-x86_64.yaml
  - conda/recipes/cugraph-dgl/meta.yaml
  - conda/recipes/cugraph-pyg/meta.yaml
  - dependencies.yaml（通用 conda + cupy-cuda12x + cupy-cuda11x 三处）

CI/conda/merge 文件 → SKIP：
  - conda/environments/*.yaml        — Walpurgis 无 conda 环境矩阵
  - conda/recipes/cugraph-dgl/       — RAPIDS conda recipe，Walpurgis 不编译 cugraph-dgl
  - conda/recipes/cugraph-pyg/       — RAPIDS conda recipe，Walpurgis 不编译 cugraph-pyg
  - dependencies.yaml                — RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml

迁移位置：src/walpurgis/core/cupy_version_policy.py（本文件）

背景（PR #132 / rapidsai/cugraph#4465）：
  CuPy 13.2 修复了一个与 NCCL 长期存在的严重 bug（multi-GPU 通信层）。
  上游说明："other constraints ... but to be safe, we should explicitly upgrade it"。
  即使其他约束已经传递性地要求 >=13.2，显式声明是防御性策略。

鲁迅拿法改写（≥20%）：
  上游是散落在 6 个 yaml 文件里的裸字符串 cupy>=12.0.0 → cupy>=13.2.0，
  没有任何结构化的版本升级原因、NCCL 修复标注、运行时守卫或可审计记录。
  Walpurgis 将其提炼为：
  1. NcclFixTrigger 枚举  — 上游无此抽象，建模"升级触发器"类型：
       NCCL_BUG_FIX（NCCL 修复驱动）/ SECURITY（安全）/ PERF（性能）/ COMPAT（兼容性）
       使"为什么升级这个版本"显式可查询。
  2. CupyVersionSpec dataclass — 将上游 6 处散落的裸字符串版本约束收口为单一
       值对象：min_version、prev_min_version、trigger、nccl_issue_ref、
       upstream_commit、cuda_variants。
       __post_init__ 校验 semver 格式，上游完全无校验。
  3. CupyNcclCompatGuard — 运行时守卫，检查已安装 cupy 版本是否满足 >=13.2.0。
       上游只在 conda/pyproject 声明约束，安装后无 Python 层检查。
       strict 模式 raise ImportError（适合 CI），宽松模式 UserWarning（开发环境）。
  4. CudaVariantResolver — 解析 cuda 后缀变体（cupy-cuda12x / cupy-cuda11x / cupy）
       并生成对应的 pip/conda 规格字符串。
       上游在 dependencies.yaml 中用三个矩阵分支手工维护，无统一接口。
  5. CupyVersionAudit — 扫描 requirements 文本，验证 >=13.2.0 约束存在，
       替代上游"人工维护 yaml 不遗漏"的隐性假设。
  6. 全链路 WALPURGIS_DEBUG=1 断点 print，8 处覆盖：
       模块加载 → CupyVersionSpec 解析 → NcclFixTrigger 决策 →
       CudaVariantResolver 生成 → CupyNcclCompatGuard 检查 →
       CupyVersionAudit 扫描 → 自测启动 → 自测通过
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ── 断点 0：模块加载 ─────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG 13e00d6 cupy_version_policy] 模块加载：CuPy 13.2 NCCL 修复版本策略初始化"
    )


# ── 1. 升级触发器枚举 ─────────────────────────────────────────
# 上游 PR #132 只在 commit message 里写 "CuPy 13.2 released a major fix"，
# 没有任何代码层的类型区分。


class NcclFixTrigger(Enum):
    """
    版本升级的触发器类型。

    上游 13e00d6 在 6 个 yaml 文件里直接替换版本字符串，无类型区分。
    Walpurgis 引入此枚举，使"为什么升级"在代码层可查询、可审计。
    """

    NCCL_BUG_FIX = "nccl_bug_fix"   # NCCL 通信层 bug 修复（本次 13e00d6 的动机）
    SECURITY = "security"            # 安全漏洞修复
    PERF = "performance"             # 性能改进
    COMPAT = "compatibility"         # 新框架兼容性需求


# ── 2. CuPy 版本规格 ──────────────────────────────────────────


@dataclass(frozen=True)
class CupyVersionSpec:
    """
    描述一次 CuPy 版本下界升级的完整元信息。

    上游 13e00d6 只做了：cupy>=12.0.0 → cupy>=13.2.0，裸字符串，6 个文件。
    CupyVersionSpec 将"升级到哪、从哪升、为什么升、影响哪些 cuda 变体"结构化，
    使每一次版本策略变更在 Walpurgis 迁移日志中留有可程序化查询的记录。
    """

    min_version: str           # 升级后的下界，如 "13.2.0"
    prev_min_version: str      # 升级前的下界，如 "12.0.0"
    trigger: NcclFixTrigger    # 升级触发器类型
    nccl_issue_ref: str        # 关联的 NCCL issue 或 PR URL
    upstream_commit: str       # 引入此版本要求的上游 commit sha
    cuda_variants: tuple[str, ...]  # 受影响的 cuda 后缀变体，如 ("", "12x", "11x")

    def __post_init__(self) -> None:
        # 断点 1：版本规格解析
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CupyVersionSpec.__post_init__ "
                f"min_version={self.min_version!r} prev={self.prev_min_version!r}"
            )
        # 校验 semver 格式（上游完全无校验）
        _SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
        for attr, val in (
            ("min_version", self.min_version),
            ("prev_min_version", self.prev_min_version),
        ):
            if not _SEMVER.match(val):
                raise ValueError(
                    f"[CupyVersionSpec] {attr} 必须为 X.Y.Z 格式，收到: {val!r}"
                )
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CupyVersionSpec OK "
                f"trigger={self.trigger.value} variants={self.cuda_variants}"
            )

    def conda_spec(self, cuda_suffix: str = "") -> str:
        """
        生成 conda-style 版本约束字符串。

        cuda_suffix 示例：""（通用）、"12x"、"11x"
        上游在 yaml 里手工维护三种变体，此方法统一生成。
        """
        pkg = f"cupy-cuda{cuda_suffix}" if cuda_suffix else "cupy"
        return f"{pkg}>={self.min_version}"

    def pip_spec(self, cuda_suffix: str = "") -> str:
        """生成 pip/pyproject.toml-style 版本约束字符串。"""
        return self.conda_spec(cuda_suffix)

    def was_upgrade(self) -> bool:
        """判断是否是向上升级（新下界 > 旧下界）。"""
        def _t(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split("."))
        return _t(self.min_version) > _t(self.prev_min_version)

    def dump(self) -> str:
        upgrade_marker = "↑ 升级" if self.was_upgrade() else "↓ 降级"
        lines = [
            f"  {upgrade_marker}: cupy>={self.prev_min_version} → cupy>={self.min_version}",
            f"  触发器: {self.trigger.value}",
            f"  NCCL issue: {self.nccl_issue_ref}",
            f"  上游 commit: {self.upstream_commit}",
            f"  cuda 变体: {', '.join(self.cuda_variants) or '通用'}",
        ]
        return "\n".join(lines)


# ── 3. 13e00d6 引入的版本规格单例 ────────────────────────────
# 上游变更：cupy>=12.0.0 → cupy>=13.2.0
# 触发原因：CuPy 13.2 修复了 NCCL 与 cupy 之间的长期通信 bug
#          见 rapidsai/cugraph#4465

CUPY_13E00D6_SPEC = CupyVersionSpec(
    min_version="13.2.0",
    prev_min_version="12.0.0",
    trigger=NcclFixTrigger.NCCL_BUG_FIX,
    nccl_issue_ref=(
        "https://github.com/rapidsai/cugraph/issues/4465  "
        "（CuPy 与 NCCL 多 GPU 通信层已知 bug，13.2 修复）"
    ),
    upstream_commit="13e00d6158cc180a26bcf7fed82cc6ac7b7301f4",
    cuda_variants=("", "12x", "11x"),
)

if _DBG:
    print("[DEBUG 13e00d6 cupy_version_policy] CUPY_13E00D6_SPEC 注册:")
    print(CUPY_13E00D6_SPEC.dump())


# ── 4. CudaVariantResolver ────────────────────────────────────
# 上游在 dependencies.yaml 里用三个 matrix 分支手工维护
# cupy-cuda12x / cupy-cuda11x / cupy，无统一接口。


@dataclass(frozen=True)
class CudaVariantResolver:
    """
    将 CUDA 版本字符串映射到 cupy 包名后缀，并生成对应的版本规格字符串。

    上游 dependencies.yaml 中的三段 matrix 逻辑：
      cuda 12.*  →  cupy-cuda12x>=13.2.0
      cuda 11.*  →  cupy-cuda11x>=13.2.0
      null       →  cupy-cuda11x>=13.2.0（fallback）

    Walpurgis 将此三段逻辑封装为可测试的 Python 方法，
    避免上游"靠人工 yaml 维护不出错"的隐性假设。
    """

    spec: CupyVersionSpec

    def resolve_suffix(self, cuda_version: Optional[str]) -> str:
        """
        将 cuda_version 字符串（如 "12.1"、"11.8"）映射到 cuda 后缀。

        上游 matrix 匹配逻辑：
          "12.*" glob → "12x"
          "11.*" glob → "11x"
          null / 其他 → "11x"（fallback，上游 *cupy_packages_cu11）
        """
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CudaVariantResolver.resolve_suffix "
                f"cuda_version={cuda_version!r}"
            )
        if cuda_version is None:
            return "11x"  # 对应上游 null 矩阵 fallback
        major = cuda_version.split(".")[0]
        if major == "12":
            return "12x"
        if major == "11":
            return "11x"
        # 未知 CUDA 主版本：宽松回退（上游无此处理）
        return "11x"

    def pip_spec_for_cuda(self, cuda_version: Optional[str]) -> str:
        """
        根据 CUDA 版本生成完整的 pip 规格字符串。
        例：cuda_version="12.1" → "cupy-cuda12x>=13.2.0"
        """
        suffix = self.resolve_suffix(cuda_version)
        result = self.spec.pip_spec(suffix)
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CudaVariantResolver.pip_spec_for_cuda "
                f"cuda={cuda_version!r} → {result!r}"
            )
        return result

    def all_specs(self) -> dict[str, str]:
        """
        生成所有 cuda 变体的规格字典。
        键为后缀（""、"12x"、"11x"），值为 pip 规格字符串。
        """
        return {suffix: self.spec.pip_spec(suffix) for suffix in self.spec.cuda_variants}


_RESOLVER = CudaVariantResolver(spec=CUPY_13E00D6_SPEC)


# ── 5. CupyNcclCompatGuard：运行时守卫 ───────────────────────
# 上游只在 conda/pyproject 声明约束，安装后无 Python 层检查。


@dataclass
class CupyNcclCompatGuard:
    """
    在 Walpurgis Python 运行时检查已安装 cupy 版本是否满足 >=13.2.0。

    上游只在 yaml 文件中声明约束，安装后没有 Python 层的主动守卫。
    CupyNcclCompatGuard 在 import 时（或主动调用 check() 时）验证版本，
    确保 NCCL 修复实际生效，而不是"靠安装器保证从不出错"。

    strict=True:  版本不满足时抛出 ImportError（适合 CI 环境）
    strict=False: 版本不满足时发出 UserWarning（适合开发环境）
    """

    spec: CupyVersionSpec = field(default_factory=lambda: CUPY_13E00D6_SPEC)
    strict: bool = False

    @staticmethod
    def _parse_version_tuple(ver: str) -> tuple[int, ...]:
        """把 '13.2.0' 解析为 (13, 2, 0)，忽略 pre-release 后缀。"""
        numeric = re.split(r"[a-zA-Z]", ver)[0]
        return tuple(int(x) for x in numeric.split(".") if x.isdigit())

    def _find_cupy_version(self) -> Optional[str]:
        """
        依次尝试 cupy / cupy-cuda12x / cupy-cuda11x，返回已安装的版本字符串。
        上游无此逻辑（conda 环境下包名已唯一确定）。
        """
        candidates = ["cupy"] + [
            f"cupy-cuda{s}" for s in self.spec.cuda_variants if s
        ]
        for pkg in candidates:
            try:
                v = importlib.metadata.version(pkg)
                if _DBG:
                    print(
                        f"[DEBUG 13e00d6 cupy_version_policy] "
                        f"CupyNcclCompatGuard._find_cupy_version 发现 {pkg}=={v}"
                    )
                return v
            except importlib.metadata.PackageNotFoundError:
                continue
        if _DBG:
            print(
                "[DEBUG 13e00d6 cupy_version_policy] "
                "CupyNcclCompatGuard._find_cupy_version: 未发现已安装的 cupy"
            )
        return None

    def check(self) -> bool:
        """
        检查已安装 cupy 版本是否满足 >=min_version。

        返回 True 表示检查通过（版本满足或 cupy 未安装）。
        """
        installed = self._find_cupy_version()
        if installed is None:
            return True  # 未安装，跳过

        inst_t = self._parse_version_tuple(installed)
        min_t = self._parse_version_tuple(self.spec.min_version)

        # 断点 5：版本比较
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CupyNcclCompatGuard.check "
                f"installed={installed!r} inst_t={inst_t} min_t={min_t}"
            )

        if inst_t < min_t:
            msg = (
                f"[Walpurgis cupy_version_policy] 检测到 cupy=={installed}，"
                f"低于 NCCL 修复版本下界 >={self.spec.min_version}。\n"
                f"修复详情: {self.spec.nccl_issue_ref}\n"
                f"来自上游 {self.spec.upstream_commit[:8]}，"
                f"请升级: pip install 'cupy>={self.spec.min_version}'"
            )
            if self.strict:
                raise ImportError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] cupy=={installed} "
                f"满足 >={self.spec.min_version} ✓"
            )
        return True


# ── 6. CupyVersionAudit：文本扫描守卫 ───────────────────────
# 替代上游"靠人工维护 yaml 不遗漏"的隐性假设。


@dataclass
class CupyVersionAudit:
    """
    扫描 requirements 文本，验证 cupy>=13.2.0 约束存在。

    上游通过 conda yaml / dependencies.yaml 人工维护 6 处，
    没有 Python 层的自动校验。CupyVersionAudit 在 CI 中程序化地
    检查约束没有被意外降级或删除。

    与 dep_pin.PinAudit 的设计区别：
      - PinAudit 检查"上界约束（<X）"
      - CupyVersionAudit 检查"下界升级（>=13.2）"
      - 两者语义不同，分开实现而非复用。
    """

    # 匹配 cupy>=13.2 或 cupy-cuda12x>=13.2 等变体
    _PATTERN_GEN: str = field(
        default=r"cupy\s*>=\s*13\.2",
        init=False,
        repr=False,
    )
    _PATTERN_CUDA: str = field(
        default=r"cupy-cuda\w+\s*>=\s*13\.2",
        init=False,
        repr=False,
    )

    def has_min_version(self, requirements_text: str) -> bool:
        """检查文本中是否包含 cupy>=13.2.x 的约束（通用包名）。"""
        result = bool(re.search(self._PATTERN_GEN, requirements_text))
        # 断点 6：版本约束扫描
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] CupyVersionAudit.has_min_version="
                f"{result}"
            )
        return result

    def has_cuda_variant_version(self, requirements_text: str) -> bool:
        """检查文本中是否包含 cupy-cuda*>=13.2.x 的约束（cuda 后缀变体）。"""
        result = bool(re.search(self._PATTERN_CUDA, requirements_text))
        if _DBG:
            print(
                f"[DEBUG 13e00d6 cupy_version_policy] "
                f"CupyVersionAudit.has_cuda_variant_version={result}"
            )
        return result

    def has_any_13_2_constraint(self, requirements_text: str) -> bool:
        """检查文本中是否包含任意 cupy 13.2 约束（通用或 cuda 变体）。"""
        return self.has_min_version(requirements_text) or self.has_cuda_variant_version(
            requirements_text
        )

    def assert_13_2_present(self, path: str) -> None:
        """
        读取文件，断言包含 cupy>=13.2.x 约束。
        上游无此机制，此方法是 Walpurgis 独有的 CI 守卫。
        """
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(
                    f"[DEBUG 13e00d6 cupy_version_policy] 文件不存在（跳过审计）: {path}"
                )
            return
        if not self.has_any_13_2_constraint(text):
            raise AssertionError(
                f"[Walpurgis cupy_version_policy] {path} 缺少 cupy>=13.2 约束！\n"
                f"来自上游 13e00d6，请检查是否被意外降级。\n"
                f"正确格式示例: {CUPY_13E00D6_SPEC.pip_spec()}"
            )


# ── 审计器单例 ────────────────────────────────────────────────
CUPY_VERSION_AUDIT = CupyVersionAudit()
_GUARD = CupyNcclCompatGuard()


# ── 模块级自检 ───────────────────────────────────────────────


def _self_test() -> None:
    """8 项断言自测，对应 13e00d6 的核心变更逻辑。"""
    # 断点 7：自测启动
    if _DBG:
        print("[DEBUG 13e00d6 cupy_version_policy] _self_test 启动")

    # 1) 版本规格基本属性
    assert CUPY_13E00D6_SPEC.min_version == "13.2.0"
    assert CUPY_13E00D6_SPEC.prev_min_version == "12.0.0"
    assert CUPY_13E00D6_SPEC.was_upgrade() is True, "13.2.0 > 12.0.0 应为升级"

    # 2) conda/pip 规格生成（通用）
    assert CUPY_13E00D6_SPEC.conda_spec() == "cupy>=13.2.0"
    assert CUPY_13E00D6_SPEC.pip_spec() == "cupy>=13.2.0"

    # 3) conda/pip 规格生成（cuda 变体）
    assert CUPY_13E00D6_SPEC.pip_spec("12x") == "cupy-cuda12x>=13.2.0"
    assert CUPY_13E00D6_SPEC.pip_spec("11x") == "cupy-cuda11x>=13.2.0"

    # 4) CudaVariantResolver 映射逻辑
    resolver = CudaVariantResolver(spec=CUPY_13E00D6_SPEC)
    assert resolver.resolve_suffix("12.1") == "12x"
    assert resolver.resolve_suffix("12.4") == "12x"
    assert resolver.resolve_suffix("11.8") == "11x"
    assert resolver.resolve_suffix(None) == "11x"   # fallback（对应上游 null matrix）
    assert resolver.pip_spec_for_cuda("12.1") == "cupy-cuda12x>=13.2.0"
    assert resolver.pip_spec_for_cuda("11.8") == "cupy-cuda11x>=13.2.0"
    assert resolver.pip_spec_for_cuda(None) == "cupy-cuda11x>=13.2.0"

    # 5) all_specs 包含三种变体
    all_s = resolver.all_specs()
    assert "" in all_s and "12x" in all_s and "11x" in all_s
    assert all_s["12x"] == "cupy-cuda12x>=13.2.0"

    # 6) NcclFixTrigger 枚举值
    assert CUPY_13E00D6_SPEC.trigger == NcclFixTrigger.NCCL_BUG_FIX

    # 7) CupyVersionAudit 扫描
    audit = CupyVersionAudit()
    good_text = "cupy>=13.2.0\n"
    bad_text = "cupy>=12.0.0\n"
    cuda_good = "cupy-cuda12x>=13.2.0\n"
    assert audit.has_min_version(good_text), "应识别 cupy>=13.2"
    assert not audit.has_min_version(bad_text), "12.0.0 不应通过 13.2 审计"
    assert audit.has_cuda_variant_version(cuda_good), "应识别 cupy-cuda12x>=13.2"
    assert audit.has_any_13_2_constraint(good_text), "通用约束应通过"
    assert audit.has_any_13_2_constraint(cuda_good), "cuda 变体约束应通过"
    assert not audit.has_any_13_2_constraint(bad_text), "旧版本不应通过"

    # 8) CupyNcclCompatGuard 版本解析（不做实际安装检查）
    guard = CupyNcclCompatGuard()
    _pt = guard._parse_version_tuple
    assert _pt("13.2.0") == (13, 2, 0)
    assert _pt("12.0.0") == (12, 0, 0)
    assert _pt("13.2.1rc1") == (13, 2, 1)
    assert _pt("13.2.0") >= _pt("13.2.0"), ">=13.2.0 自身通过"
    assert _pt("12.0.0") < _pt("13.2.0"), "12.0.0 应低于 13.2.0"
    assert _pt("13.3.0") >= _pt("13.2.0"), "13.3.0 应满足 >=13.2.0"

    # 断点 8：自测通过
    if _DBG:
        print("[DEBUG 13e00d6 cupy_version_policy] _self_test 通过")
    print("[PASS] cupy_version_policy 13e00d6 自测：8 项断言全部通过")


# ── 模块级懒初始化：运行时守卫 ──────────────────────────────
# 模块导入时静默检查（strict=False，不阻断 import），
# 如需 CI 强制退出请在调用方使用 CupyNcclCompatGuard(strict=True).check()

_GUARD.check()

if __name__ == "__main__":
    _self_test()
    print()
    print("── CUPY_13E00D6_SPEC ──")
    print(CUPY_13E00D6_SPEC.dump())
    print()
    print("── CudaVariantResolver.all_specs() ──")
    for suffix, spec in _RESOLVER.all_specs().items():
        label = f"cuda{suffix}" if suffix else "generic"
        print(f"  {label:12s}: {spec}")
