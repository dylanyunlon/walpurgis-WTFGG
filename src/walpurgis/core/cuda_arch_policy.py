"""
cuda_arch_policy.py — 0726fd1 迁移: CUDA 架构集合从硬编码迁移至 RAPIDS 动态策略

上游来源: build.sh + .pre-commit-config.yaml
commit: 0726fd10257544400486b144a88262403fea4637
author: James Lamb <jaylamb20@gmail.com>
date: 2025-09-03
PR: https://github.com/rapidsai/cugraph-gnn/pull/295

上游变更摘要（2 files changed, 5 insertions, 5 deletions）:
  build.sh:
    - NATIVE 分支: 去掉多余的 := 默认值赋值语法，简化为直接赋值 "NATIVE"
    - ALL_ARCH 分支: "70-real;75-real;80-real;86-real;90" → "RAPIDS"
      （从硬编码 semicolon-list 改为 rapids-cmake 动态集合标记）
  .pre-commit-config.yaml:
    - pre-commit/pre-commit-hooks: v5.0.0 → v6.0.0
    - rapidsai/pre-commit-hooks:   v0.4.0 → v0.7.0

上游逻辑（build.sh 核心段，约 6 行）:
  if (( BUILD_ALL_GPU_ARCH == 0 )); then
      WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="NATIVE"
      echo "Building for the architecture of the GPU in the system..."
  else
      WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
      echo "Building for *ALL* supported GPU architectures..."
  fi

上游「RAPIDS」动态集合对应架构（来自 rapids-cmake 0b111489）:
  CUDA 12: 70-real;75-real;80-real;86-real;90a-real;100f-real;120a-real;120
  CUDA 13: 75-real;80-real;86-real;90a-real;100f-real;120a-real;120
  （before: 70-real;75-real;80-real;86-real;90）

Walpurgis 改写 20%（鲁迅拿法）:
1. CudaArchMode 枚举: 替代上游 bash 整型标志位 BUILD_ALL_GPU_ARCH
   上游用 0/1 裸整数判断，无类型约束；枚举使调用路径可读且可 match。
2. RapidsCudaArchSet dataclass: 将 rapids-cmake 动态集合的两个 CUDA 版本
   对应架构列表显式建模，而非藏在 cmake 下游"黑盒"里。
   上游：下游调用 rapids-cmake，Walpurgis 无 CMake，故在 Python 层文档化。
3. CudaArchPolicy 决策类: resolve() 方法返回 CMake-ready 字符串，
   同时暴露 rationale 字段供断点日志消费。
   上游做法：直接 echo + 赋值 bash 变量，无决策记录。
4. PreCommitVersionBump dataclass: 文档化 .pre-commit-config.yaml 版本升级，
   供 hook 版本审计脚本（或 pytest）验证项目内 pre-commit hooks 版本一致性。
   上游做法：直接修改 YAML，无 Python 层可查询记录。
5. 全链路 WALPURGIS_DEBUG=1 断点 print:
   - 架构模式解析入口/出口
   - RAPIDS 集合版本匹配过程
   - resolve() 决策路径（NATIVE / RAPIDS / legacy-fallback）
   - pre-commit 版本核验阶段
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 调试开关（与整个 Walpurgis 体系统一）
# ─────────────────────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """内部调试打印，WALPURGIS_DEBUG=1 时生效。"""
    if _DEBUG:
        print(f"[WPG 0726fd1 {tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# CudaArchMode — 上游 BUILD_ALL_GPU_ARCH 整型标志位的类型化替代
# 上游: (( BUILD_ALL_GPU_ARCH == 0 )) → 裸整数比较，无语义命名
# 改写: 枚举使代码路径一目了然，也可用于 match/case (Python 3.10+)
# ─────────────────────────────────────────────────────────────

@unique
class CudaArchMode(Enum):
    """
    CUDA 架构构建模式，对应上游 build.sh BUILD_ALL_GPU_ARCH 标志。

    NATIVE  — 仅为当前系统 GPU 编译（开发/调试用，速度快）
    ALL     — 为 RAPIDS 完整支持架构集合编译（发布/CI 用，二进制更大）
    """
    NATIVE = "NATIVE"
    ALL    = "RAPIDS"

    @classmethod
    def from_build_all_flag(cls, flag: int) -> "CudaArchMode":
        """
        从上游 BUILD_ALL_GPU_ARCH bash 整型标志转换。

        Args:
            flag: 0 → NATIVE，非 0 → ALL（与上游 if (( == 0 )) 一致）
        """
        _dbg("CudaArchMode.from_build_all_flag", f"flag={flag!r}")
        mode = cls.NATIVE if flag == 0 else cls.ALL
        _dbg("CudaArchMode.from_build_all_flag", f"→ {mode}")
        return mode

    @classmethod
    def from_env(cls, var: str = "WALPURGIS_BUILD_ALL_GPU_ARCH") -> "CudaArchMode":
        """
        从环境变量读取构建模式。

        环境变量未设置或为 '0' → NATIVE；'1' 或其他非零字符串 → ALL。
        """
        raw = os.environ.get(var, "0")
        _dbg("CudaArchMode.from_env", f"var={var!r} raw={raw!r}")
        try:
            flag = int(raw)
        except ValueError:
            _dbg("CudaArchMode.from_env", f"无法解析为整数，raw={raw!r}，回退 NATIVE")
            flag = 0
        return cls.from_build_all_flag(flag)


# ─────────────────────────────────────────────────────────────
# RapidsCudaArchSet — 显式建模 rapids-cmake RAPIDS 架构集合
# 上游：将解析委托给 CMake set_architectures.cmake，Walpurgis 无 CMake 层
# 改写：在 Python 层文档化每个 CUDA 主版本对应的真实架构列表，
#        使版本升级时可追溯、可测试
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RapidsCudaArchSet:
    """
    rapids-cmake 为指定 CUDA 主版本导出的 RAPIDS 架构集合快照。

    来源: rapids-cmake 0b111489d1e6f8400e1fc88297623a2a9915fa77
          rapids-cmake/cuda/set_architectures.cmake

    Attributes:
        cuda_major:    CUDA 主版本号（12 / 13 / …）
        architectures: semicolon-list 格式的架构字符串（CMake-ready）
        note:          与 0726fd1 之前硬编码集合的差异说明
    """
    cuda_major: int
    architectures: str   # CMake WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES 直接使用
    note: str = ""

    def as_list(self) -> List[str]:
        """将 semicolon-list 拆分为 Python list，方便遍历或断言。"""
        return [a.strip() for a in self.architectures.split(";") if a.strip()]

    def __str__(self) -> str:
        return (
            f"RapidsCudaArchSet(cuda{self.cuda_major}: "
            f"{self.architectures!r})"
        )


# 0726fd1 迁移时的 RAPIDS 架构快照（来自 rapids-cmake 0b111489）
# before (hardcoded): "70-real;75-real;80-real;86-real;90"
_RAPIDS_ARCH_SETS: Dict[int, RapidsCudaArchSet] = {
    12: RapidsCudaArchSet(
        cuda_major=12,
        architectures="70-real;75-real;80-real;86-real;90a-real;100f-real;120a-real;120",
        note=(
            "CUDA 12: 新增 90a-real/100f-real/120a-real/120, "
            "删除旧 90（已被 90a-real 取代）"
        ),
    ),
    13: RapidsCudaArchSet(
        cuda_major=13,
        architectures="75-real;80-real;86-real;90a-real;100f-real;120a-real;120",
        note=(
            "CUDA 13: 移除 70-real（Maxwell 不再支持）"
        ),
    ),
}

# 上游硬编码旧值，用于 regression 测试和迁移文档对比
_LEGACY_HARDCODED_ARCHS = "70-real;75-real;80-real;86-real;90"


def get_rapids_arch_set(cuda_major: int) -> Optional[RapidsCudaArchSet]:
    """
    返回指定 CUDA 主版本的 RAPIDS 架构快照。

    Args:
        cuda_major: CUDA 主版本（12 / 13 / …）

    Returns:
        对应的 RapidsCudaArchSet，未知版本返回 None。
    """
    _dbg("get_rapids_arch_set", f"cuda_major={cuda_major}")
    result = _RAPIDS_ARCH_SETS.get(cuda_major)
    _dbg("get_rapids_arch_set", f"→ {result}")
    return result


# ─────────────────────────────────────────────────────────────
# CudaArchPolicy — 核心决策类
# 上游: bash 直接赋值 WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES
# 改写: 封装决策 + 生成 rationale，使构建过程可审计
# ─────────────────────────────────────────────────────────────

@dataclass
class CudaArchPolicy:
    """
    CUDA 架构选择策略，对应 0726fd1 在 build.sh 中的条件分支。

    Attributes:
        mode:       构建模式（NATIVE / ALL）
        cuda_major: 目标 CUDA 主版本，仅 ALL 模式下用于查表；None 时回退旧值
    """
    mode: CudaArchMode
    cuda_major: Optional[int] = None

    def resolve(self) -> Tuple[str, str]:
        """
        解析最终的 WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES 值及决策理由。

        Returns:
            (architectures, rationale):
              architectures — CMake 变量值，直接传入 cmake -D 或写入 build.sh
              rationale     — 决策说明字符串，供日志/断点消费

        决策树（对应 build.sh 0726fd1 逻辑）:
          NATIVE → ("NATIVE", "本机 GPU 编译模式")
          ALL + cuda_major in {12,13} → (rapids_set.architectures, "RAPIDS 动态集合")
          ALL + cuda_major unknown    → (_LEGACY_HARDCODED_ARCHS, "未知版本，回退硬编码")
          ALL + cuda_major is None    → ("RAPIDS", "全架构模式，版本未指定，传 RAPIDS 令牌")
        """
        _dbg("CudaArchPolicy.resolve", f"mode={self.mode} cuda_major={self.cuda_major}")

        if self.mode == CudaArchMode.NATIVE:
            arch = "NATIVE"
            rationale = (
                "NATIVE 模式：仅为本机 GPU 编译 "
                "（对应上游: BUILD_ALL_GPU_ARCH==0）"
            )
            _dbg("CudaArchPolicy.resolve", f"NATIVE 分支 → arch={arch!r}")
            return arch, rationale

        # ALL 模式
        _dbg("CudaArchPolicy.resolve", "ALL 分支：查询 RAPIDS 架构集合")

        if self.cuda_major is None:
            # cuda_major 未指定：直接传 "RAPIDS" 令牌给 CMake，由 rapids-cmake 展开
            arch = "RAPIDS"
            rationale = (
                "ALL 模式（cuda_major 未指定）: "
                "传递 'RAPIDS' 令牌给 CMake，由 rapids-cmake 动态展开 "
                "（0726fd1 上游行为）"
            )
            _dbg("CudaArchPolicy.resolve", f"cuda_major=None → arch={arch!r}")
            return arch, rationale

        rapids_set = get_rapids_arch_set(self.cuda_major)
        if rapids_set is not None:
            arch = rapids_set.architectures
            rationale = (
                f"ALL 模式（CUDA {self.cuda_major}）: "
                f"RAPIDS 动态集合 = {arch!r} "
                f"[{rapids_set.note}] "
                f"（0726fd1: 替换旧硬编码 {_LEGACY_HARDCODED_ARCHS!r}）"
            )
            _dbg("CudaArchPolicy.resolve", f"RAPIDS 查表命中 → arch={arch!r}")
            return arch, rationale

        # 未知 cuda_major：安全回退到旧硬编码，并告警
        import warnings
        warnings.warn(
            f"[WPG 0726fd1] 未知 CUDA 主版本 {self.cuda_major}，"
            f"回退使用旧硬编码架构集合 {_LEGACY_HARDCODED_ARCHS!r}。"
            "请更新 _RAPIDS_ARCH_SETS。",
            stacklevel=2,
        )
        arch = _LEGACY_HARDCODED_ARCHS
        rationale = (
            f"ALL 模式（CUDA {self.cuda_major} 未知）: "
            f"回退旧硬编码 {_LEGACY_HARDCODED_ARCHS!r}，"
            "需更新 _RAPIDS_ARCH_SETS"
        )
        _dbg("CudaArchPolicy.resolve", f"未知版本 → 回退 arch={arch!r}")
        return arch, rationale

    def cmake_var(self) -> str:
        """
        返回 CMake -D 参数字符串，可直接拼入 cmake 命令行。

        示例:
            "-DWHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES=RAPIDS"
        """
        arch, _ = self.resolve()
        return f"-DWHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES={arch}"

    def __str__(self) -> str:
        arch, rationale = self.resolve()
        return (
            f"CudaArchPolicy("
            f"mode={self.mode.name}, "
            f"cuda_major={self.cuda_major}, "
            f"arch={arch!r})"
        )


# ─────────────────────────────────────────────────────────────
# PreCommitVersionBump — .pre-commit-config.yaml 版本升级文档化
# 上游: 直接改 YAML，无 Python 层可查询记录
# 改写: dataclass 存档版本升级，供 hook 版本审计脚本或 pytest 使用
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreCommitHookBump:
    """单个 pre-commit hook 的版本升级记录。"""
    repo: str
    before: str
    after: str
    commit: str = "0726fd1"

    def __str__(self) -> str:
        return (
            f"PreCommitHookBump("
            f"repo={self.repo!r}, "
            f"{self.before!r} → {self.after!r})"
        )


# 0726fd1 中两处 pre-commit 版本升级的显式记录
PRE_COMMIT_BUMPS_0726FD1: Tuple[PreCommitHookBump, ...] = (
    PreCommitHookBump(
        repo="https://github.com/pre-commit/pre-commit-hooks",
        before="v5.0.0",
        after="v6.0.0",
    ),
    PreCommitHookBump(
        repo="https://github.com/rapidsai/pre-commit-hooks",
        before="v0.4.0",
        after="v0.7.0",
    ),
)


def verify_pre_commit_bumps(config_path: str = ".pre-commit-config.yaml") -> List[str]:
    """
    验证项目内 .pre-commit-config.yaml 是否已应用 0726fd1 的版本升级。

    Args:
        config_path: pre-commit 配置文件路径（相对于调用方 cwd）

    Returns:
        问题列表；列表为空表示所有版本已就绪。
    """
    _dbg("verify_pre_commit_bumps", f"config_path={config_path!r}")
    issues: List[str] = []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        _dbg("verify_pre_commit_bumps", f"文件读取成功，{len(content)} 字符")
    except FileNotFoundError:
        _dbg("verify_pre_commit_bumps", "文件不存在，跳过验证")
        issues.append(f"[0726fd1] {config_path} 不存在，无法验证 pre-commit 版本")
        return issues

    for bump in PRE_COMMIT_BUMPS_0726FD1:
        _dbg("verify_pre_commit_bumps", f"检查 {bump}")
        if bump.after in content:
            _dbg("verify_pre_commit_bumps", f"  ✓ {bump.after} 已存在")
        elif bump.before in content:
            issues.append(
                f"[0726fd1] {bump.repo} 仍为旧版 {bump.before!r}，"
                f"应升级至 {bump.after!r}"
            )
            _dbg("verify_pre_commit_bumps", f"  ✗ 旧版本 {bump.before!r} 仍存在")
        else:
            _dbg("verify_pre_commit_bumps", f"  ? 版本信息未找到，跳过")

    _dbg("verify_pre_commit_bumps", f"验证完成，issues={issues}")
    return issues


# ─────────────────────────────────────────────────────────────
# 便捷工厂函数
# ─────────────────────────────────────────────────────────────

def make_policy_from_env(
    mode_var: str = "WALPURGIS_BUILD_ALL_GPU_ARCH",
    cuda_major_var: str = "WALPURGIS_CUDA_MAJOR",
) -> CudaArchPolicy:
    """
    从环境变量构造 CudaArchPolicy，模拟 build.sh 的参数读取方式。

    Args:
        mode_var:       控制 NATIVE/ALL 模式的环境变量名
        cuda_major_var: 目标 CUDA 主版本环境变量名（可选，未设置则为 None）

    Returns:
        CudaArchPolicy 实例

    Example::

        # 模拟 CI 全架构构建 CUDA 12
        os.environ["WALPURGIS_BUILD_ALL_GPU_ARCH"] = "1"
        os.environ["WALPURGIS_CUDA_MAJOR"] = "12"
        policy = make_policy_from_env()
        arch, _ = policy.resolve()
        # arch == "70-real;75-real;80-real;86-real;90a-real;100f-real;120a-real;120"
    """
    _dbg("make_policy_from_env", f"mode_var={mode_var!r} cuda_major_var={cuda_major_var!r}")

    mode = CudaArchMode.from_env(mode_var)

    cuda_major_raw = os.environ.get(cuda_major_var, "")
    cuda_major: Optional[int] = None
    if cuda_major_raw:
        try:
            cuda_major = int(cuda_major_raw)
            _dbg("make_policy_from_env", f"cuda_major={cuda_major}")
        except ValueError:
            _dbg("make_policy_from_env", f"cuda_major_raw={cuda_major_raw!r} 无法解析，设为 None")

    policy = CudaArchPolicy(mode=mode, cuda_major=cuda_major)
    _dbg("make_policy_from_env", f"→ {policy}")
    return policy


# ─────────────────────────────────────────────────────────────
# CLI 自检（WALPURGIS_DEBUG=1 python cuda_arch_policy.py）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("[cuda_arch_policy 0726fd1] 自检启动")
    print(f"  WALPURGIS_DEBUG={os.environ.get('WALPURGIS_DEBUG', '0')}")

    # 1. NATIVE 模式
    p_native = CudaArchPolicy(mode=CudaArchMode.NATIVE)
    arch_n, rat_n = p_native.resolve()
    print(f"\n[NATIVE] arch={arch_n!r}")
    print(f"         cmake_var: {p_native.cmake_var()!r}")
    print(f"         rationale: {rat_n}")

    # 2. ALL 模式 — cuda_major 未指定（上游行为：传 RAPIDS 令牌）
    p_all_nospec = CudaArchPolicy(mode=CudaArchMode.ALL)
    arch_a0, rat_a0 = p_all_nospec.resolve()
    print(f"\n[ALL / no cuda_major] arch={arch_a0!r}")
    print(f"                      rationale: {rat_a0}")

    # 3. ALL 模式 — CUDA 12
    p_all_12 = CudaArchPolicy(mode=CudaArchMode.ALL, cuda_major=12)
    arch_12, rat_12 = p_all_12.resolve()
    print(f"\n[ALL / CUDA 12] arch={arch_12!r}")
    print(f"                cmake_var: {p_all_12.cmake_var()!r}")
    print(f"                as_list: {get_rapids_arch_set(12).as_list()}")  # type: ignore[union-attr]
    print(f"                rationale: {rat_12}")

    # 4. ALL 模式 — CUDA 13
    p_all_13 = CudaArchPolicy(mode=CudaArchMode.ALL, cuda_major=13)
    arch_13, rat_13 = p_all_13.resolve()
    print(f"\n[ALL / CUDA 13] arch={arch_13!r}")
    print(f"                as_list: {get_rapids_arch_set(13).as_list()}")  # type: ignore[union-attr]
    print(f"                rationale: {rat_13}")

    # 5. ALL 模式 — 未知 cuda_major（回退验证）
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        p_all_unk = CudaArchPolicy(mode=CudaArchMode.ALL, cuda_major=99)
        arch_unk, rat_unk = p_all_unk.resolve()
    print(f"\n[ALL / CUDA 99 (unknown)] arch={arch_unk!r}")
    print(f"  caught warning: {caught[0].message if caught else 'none'}")
    assert arch_unk == _LEGACY_HARDCODED_ARCHS, "回退断言失败"

    # 6. from_env 工厂
    os.environ["WALPURGIS_BUILD_ALL_GPU_ARCH"] = "1"
    os.environ["WALPURGIS_CUDA_MAJOR"] = "12"
    p_env = make_policy_from_env()
    arch_env, _ = p_env.resolve()
    assert arch_env == arch_12, f"from_env 结果不匹配: {arch_env!r} != {arch_12!r}"
    print(f"\n[make_policy_from_env] arch={arch_env!r} — PASS")

    # 7. 幂等性验证
    arch_env2, _ = p_env.resolve()
    assert arch_env == arch_env2, "幂等性失败"
    print("  幂等性检查: PASS")

    # 8. pre-commit 版本升级记录打印
    print(f"\n[pre-commit bumps 0726fd1]")
    for bump in PRE_COMMIT_BUMPS_0726FD1:
        print(f"  {bump}")

    print("\n[cuda_arch_policy 0726fd1] 自检完成")
    sys.exit(0)
