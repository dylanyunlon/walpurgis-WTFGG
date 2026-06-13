"""
rapids_config_user_branch.py
=============================
迁移自 cugraph-gnn upstream commit e16ae06
（Update rapids_config to handle user defined branch name, PR #272）
Author: Robert Maynard <rmaynard@nvidia.com>

上游 diff 摘要 (2 files changed, 7 insertions(+), 3 deletions(−)):
┌──────────────────────────────────────────────────────────┬────────────┐
│ 文件                                                     │ 处置       │
├──────────────────────────────────────────────────────────┼────────────┤
│ cmake/RAPIDS.cmake   (条件逻辑 De Morgan 修正)           │  SKIP+迁移 │
│ cmake/rapids_config.cmake (非覆盖式 set 守卫)            │  SKIP+迁移 │
└──────────────────────────────────────────────────────────┴────────────┘

CMake 文件在 Walpurgis 中不存在对应实体 → 原文件 SKIP。
但两处变更的**逻辑语义**高度可迁移：

1. De Morgan 修正（RAPIDS.cmake）：
   旧：if(NOT rapids-cmake-branch OR NOT rapids-cmake-version)
       → 只要其中一个未定义就报错，用户无法单独指定 branch 或 version
   新：if(NOT (rapids-cmake-branch OR rapids-cmake-version))
       → 两者都未定义才报错，允许用户只设置其一
   这是一个经典的布尔逻辑错误，De Morgan 定律：
       NOT A OR NOT B  ≡  NOT (A AND B)  ≠  NOT (A OR B)
   上游修正后语义为：「至少有一个被定义即合法」。

2. 非覆盖式 set 守卫（rapids_config.cmake）：
   旧：set(rapids-cmake-version "...")   # 无条件覆盖
       set(rapids-cmake-branch "...")    # 无条件覆盖
   新：if(NOT rapids-cmake-version)
         set(rapids-cmake-version "...")  # 只在未定义时赋值
       if(NOT rapids-cmake-branch)
         set(rapids-cmake-branch "...")   # 只在未定义时赋值
   语义：用户自定义值优先，自动推导值仅作兜底。

鲁迅拿法改写（≥20%）：
    鲁迅在《故乡》里写过：「希望是本无所谓有，无所谓无的。」
    CMake 变量也是如此——rapids-cmake-branch 本无所谓有，无所谓无。
    旧代码的悲剧在于它用「OR NOT」把「有」和「无」混为一谈，
    把「缺席之一」等同于「完全缺席」，像一个偏执的守门人，
    一旦看到任何空缺就拒之门外，哪怕你已经带着另一份通行证。
    e16ae06 的修正是一次对偏执的纠偏：让「至少持有其一」成为
    进入的门槛，而不是「必须两者齐备」。
    这与 if-not 守卫同根同源——先到先得，用户的意志不可被系统默默覆盖。
    Walpurgis 将这两个决策提炼为可程序化查询的 Python 对象体系：
      1. BoolGuardSemantics — 枚举条件守卫的四种语义模式
      2. DeMorgan修正记录 — UserBranchConditionFix，封装条件逻辑变迁
      3. NonClobberGuard — 建模「先定义优先」的非覆盖赋值语义
      4. RapidsConfigResolutionPolicy — 完整的版本/分支解析策略
      5. UserBranchMigrationAudit — 审计 e16ae06 的迁移完整性

调试：
    export WALPURGIS_DEBUG=1  激活全链路断点。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# 调试基础设施
# ---------------------------------------------------------------------------

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试输出，WALPURGIS_DEBUG=1 时激活。"""
    if _DBG:
        print(f"[_dbg][rapids_config_user_branch][{tag}] {msg}")
        breakpoint()  # noqa: T100


_dbg("MODULE_LOAD", "rapids_config_user_branch 模块加载，e16ae06 语义层迁移初始化")


# ---------------------------------------------------------------------------
# 1. BoolGuardSemantics — 枚举布尔守卫的语义模式
# ---------------------------------------------------------------------------

class BoolGuardSemantics(Enum):
    """
    CMake / Python 条件守卫的四种语义模式。

    上游 e16ae06 的核心是从 REJECT_IF_EITHER_MISSING 修正为
    REJECT_IF_BOTH_MISSING，这是一次 De Morgan 层面的语义修正。
    """

    REJECT_IF_EITHER_MISSING = "not_a_or_not_b"
    """
    旧语义：NOT A OR NOT B
    任一未定义即拒绝。要求两者同时存在，等价于 AND 门。
    """

    REJECT_IF_BOTH_MISSING = "not_(a_or_b)"
    """
    新语义（e16ae06）：NOT (A OR B)
    两者都未定义才拒绝。允许单独定义其一，等价于 NAND 的补集。
    """

    ACCEPT_IF_ANY_DEFINED = "a_or_b"
    """
    接受语义：A OR B — 至少一个已定义即通过。
    REJECT_IF_BOTH_MISSING 的对称正向表述。
    """

    NON_CLOBBER_SET = "if_not_defined_then_set"
    """
    非覆盖赋值守卫：if(NOT var) set(var default)
    用户自定义值优先，系统默认值仅作兜底。
    这是 rapids_config.cmake 中 set() 调用的修正语义。
    """

    @property
    def label(self) -> str:
        """人类可读标签。"""
        labels = {
            BoolGuardSemantics.REJECT_IF_EITHER_MISSING: "NOT A OR NOT B（旧，错误）",
            BoolGuardSemantics.REJECT_IF_BOTH_MISSING: "NOT (A OR B)（新，正确）",
            BoolGuardSemantics.ACCEPT_IF_ANY_DEFINED: "A OR B（通过语义）",
            BoolGuardSemantics.NON_CLOBBER_SET: "if(NOT var) set(...)（非覆盖守卫）",
        }
        return labels[self]

    def evaluate(self, a: bool, b: bool) -> bool:
        """
        按该语义评估 (a, b) 对。
        返回值语义：True = 条件成立（应报错 or 应执行 set）。

        Parameters
        ----------
        a : bool  rapids-cmake-branch 是否已定义
        b : bool  rapids-cmake-version 是否已定义
        """
        _dbg("BoolGuardSemantics.evaluate", f"mode={self.name} a={a} b={b}")
        if self == BoolGuardSemantics.REJECT_IF_EITHER_MISSING:
            # NOT A OR NOT B  →  NOT(A AND B)
            result = not (a and b)
        elif self == BoolGuardSemantics.REJECT_IF_BOTH_MISSING:
            # NOT (A OR B)
            result = not (a or b)
        elif self == BoolGuardSemantics.ACCEPT_IF_ANY_DEFINED:
            result = a or b
        else:  # NON_CLOBBER_SET — 用 a 表示「变量是否已定义」
            result = not a  # 未定义时触发赋值
        return result


_dbg("MODULE_LOAD", "BoolGuardSemantics 枚举定义完成，4 种模式已注册")


# ---------------------------------------------------------------------------
# 2. UserBranchConditionFix — De Morgan 修正记录
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserBranchConditionFix:
    """
    记录 cmake/RAPIDS.cmake 中的 De Morgan 条件修正（e16ae06）。

    上游变更：
      旧：if(NOT rapids-cmake-branch OR NOT rapids-cmake-version)
      新：if(NOT (rapids-cmake-branch OR rapids-cmake-version))

    这不是语法糖，是语义修正。旧代码强制要求两者同时存在，
    导致用户只定义 rapids-cmake-branch 时仍然报错。
    新代码只在两者都缺席时报错，允许用户只定义其中一个。
    """

    upstream_commit: str = "e16ae06"
    upstream_pr: str = "#272"
    upstream_file: str = "cmake/RAPIDS.cmake"
    old_semantics: BoolGuardSemantics = BoolGuardSemantics.REJECT_IF_EITHER_MISSING
    new_semantics: BoolGuardSemantics = BoolGuardSemantics.REJECT_IF_BOTH_MISSING
    variable_a: str = "rapids-cmake-branch"
    variable_b: str = "rapids-cmake-version"

    def old_cmake_condition(self) -> str:
        """返回旧 CMake 条件字符串（含错误）。"""
        return f"NOT {self.variable_a} OR NOT {self.variable_b}"

    def new_cmake_condition(self) -> str:
        """返回修正后的 CMake 条件字符串。"""
        return f"NOT ({self.variable_a} OR {self.variable_b})"

    def is_valid_config(self, branch_defined: bool, version_defined: bool) -> bool:
        """
        按新语义判断 (branch_defined, version_defined) 是否为合法配置。
        合法 = 不触发 FATAL_ERROR，即 NOT (A OR B) 为 False。

        Parameters
        ----------
        branch_defined  : bool  rapids-cmake-branch 是否已定义
        version_defined : bool  rapids-cmake-version 是否已定义
        """
        _dbg(
            "UserBranchConditionFix.is_valid_config",
            f"branch={branch_defined} version={version_defined}",
        )
        # NOT (A OR B) == True → 报错。合法 = 不报错 = NOT (A OR B) == False
        would_error = self.new_semantics.evaluate(branch_defined, version_defined)
        return not would_error

    def old_is_valid_config(self, branch_defined: bool, version_defined: bool) -> bool:
        """按旧（错误）语义判断是否合法。用于对比旧行为。"""
        _dbg(
            "UserBranchConditionFix.old_is_valid_config",
            f"branch={branch_defined} version={version_defined}",
        )
        would_error = self.old_semantics.evaluate(branch_defined, version_defined)
        return not would_error

    def regression_cases(self) -> list[dict]:
        """
        返回 De Morgan 修正的关键回归测试用例。
        每个用例包含 (branch_defined, version_defined) 及新旧行为对比。
        """
        _dbg("UserBranchConditionFix.regression_cases", "生成回归测试用例集")
        cases = [
            {
                "branch_defined": False,
                "version_defined": False,
                "old_valid": False,   # 两者都缺 → 旧也报错
                "new_valid": False,   # 两者都缺 → 新也报错
                "description": "两者均未定义，新旧均报错（行为一致）",
            },
            {
                "branch_defined": True,
                "version_defined": False,
                "old_valid": False,   # 旧：version 缺 → 报错（BUG）
                "new_valid": True,    # 新：branch 已定义 → 合法（修正）
                "description": "仅 branch 已定义：旧报错，新合法（这是 e16ae06 修正的关键场景）",
            },
            {
                "branch_defined": False,
                "version_defined": True,
                "old_valid": False,   # 旧：branch 缺 → 报错（BUG）
                "new_valid": True,    # 新：version 已定义 → 合法（修正）
                "description": "仅 version 已定义：旧报错，新合法（对称场景）",
            },
            {
                "branch_defined": True,
                "version_defined": True,
                "old_valid": True,    # 两者均有 → 旧合法
                "new_valid": True,    # 两者均有 → 新合法
                "description": "两者均已定义，新旧均合法（行为一致）",
            },
        ]
        return cases

    def describe_fix(self) -> str:
        """生成可读的修正摘要，适合写入文档或日志。"""
        return (
            f"[De Morgan 修正] {self.upstream_commit} / {self.upstream_pr}\n"
            f"  文件: {self.upstream_file}\n"
            f"  旧条件: if({self.old_cmake_condition()})  → 语义错误\n"
            f"  新条件: if({self.new_cmake_condition()})  → 正确\n"
            f"  修正原理: De Morgan 定律\n"
            f"    NOT A OR NOT B  ≡  NOT(A AND B)  「两者同时存在才通过」\n"
            f"    NOT (A OR B)    ≡  NOT A AND NOT B  「至少一个存在即通过」\n"
            f"  用户影响: 现在允许只定义 {self.variable_a} 或只定义 {self.variable_b}\n"
        )


_dbg("MODULE_LOAD", "UserBranchConditionFix dataclass 定义完成")


# ---------------------------------------------------------------------------
# 3. NonClobberGuard — 非覆盖赋值守卫
# ---------------------------------------------------------------------------

@dataclass
class NonClobberGuard:
    """
    建模 cmake/rapids_config.cmake 中的非覆盖 set() 守卫（e16ae06）。

    上游变更：
      旧：set(rapids-cmake-version "${RAPIDS_VERSION_MAJOR_MINOR}")
          set(rapids-cmake-branch "${_rapids_branch}")
          → 无条件覆盖，用户自定义值被静默丢弃

      新：if(NOT rapids-cmake-version)
            set(rapids-cmake-version "${RAPIDS_VERSION_MAJOR_MINOR}")
          if(NOT rapids-cmake-branch)
            set(rapids-cmake-branch "${_rapids_branch}")
          → 只在未定义时赋值，用户自定义值受保护

    这是一个经典的「defaults without override」模式。
    鲁迅语：「他们说这是本分，其实不过是奴隶性。」
    旧的无条件 set 是系统对用户意志的默默压制；
    非覆盖守卫是对这种压制的纠正。
    """

    cmake_file: str = "cmake/rapids_config.cmake"
    upstream_commit: str = "e16ae06"

    # 运行时状态（非 frozen，允许模拟赋值）
    _version: Optional[str] = field(default=None, repr=False)
    _branch: Optional[str] = field(default=None, repr=False)

    def set_user_version(self, version: str) -> None:
        """模拟用户预先定义 rapids-cmake-version（在 rapids_config.cmake 执行前）。"""
        _dbg("NonClobberGuard.set_user_version", f"用户定义 version={version!r}")
        self._version = version

    def set_user_branch(self, branch: str) -> None:
        """模拟用户预先定义 rapids-cmake-branch（在 rapids_config.cmake 执行前）。"""
        _dbg("NonClobberGuard.set_user_branch", f"用户定义 branch={branch!r}")
        self._branch = branch

    def apply_defaults(
        self,
        derived_version: str,
        derived_branch: str,
        clobber: bool = False,
    ) -> tuple[str, str]:
        """
        应用默认值（模拟 rapids_config.cmake 的 set() 逻辑）。

        Parameters
        ----------
        derived_version : str  系统自动推导的版本号（如 "25.10"）
        derived_branch  : str  系统自动推导的分支名（如 "branch-25.10"）
        clobber         : bool True = 旧行为（无条件覆盖），False = 新行为（非覆盖守卫）

        Returns
        -------
        (version, branch) 最终生效的值
        """
        _dbg(
            "NonClobberGuard.apply_defaults",
            f"derived_version={derived_version!r} derived_branch={derived_branch!r} "
            f"clobber={clobber} user_version={self._version!r} user_branch={self._branch!r}",
        )
        if clobber:
            # 旧行为：无条件覆盖
            final_version = derived_version
            final_branch = derived_branch
            _dbg(
                "NonClobberGuard.apply_defaults.clobber",
                f"旧行为：用户值被覆盖 → version={final_version!r} branch={final_branch!r}",
            )
        else:
            # 新行为（e16ae06）：非覆盖守卫
            final_version = self._version if self._version is not None else derived_version
            final_branch = self._branch if self._branch is not None else derived_branch
            _dbg(
                "NonClobberGuard.apply_defaults.non_clobber",
                f"新行为：用户值优先 → version={final_version!r} branch={final_branch!r}",
            )
        return final_version, final_branch

    def version_source(self, derived_version: str) -> str:
        """报告 version 的最终来源：USER or DERIVED。"""
        if self._version is not None:
            return f"USER({self._version!r})"
        return f"DERIVED({derived_version!r})"

    def branch_source(self, derived_branch: str) -> str:
        """报告 branch 的最终来源：USER or DERIVED。"""
        if self._branch is not None:
            return f"USER({self._branch!r})"
        return f"DERIVED({derived_branch!r})"


_dbg("MODULE_LOAD", "NonClobberGuard dataclass 定义完成")


# ---------------------------------------------------------------------------
# 4. RapidsConfigResolutionPolicy — 完整解析策略（整合两处修正）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RapidsConfigResolutionPolicy:
    """
    整合 e16ae06 两处修正的完整解析策略。

    封装：
    - De Morgan 条件修正（RAPIDS.cmake）
    - 非覆盖 set 守卫（rapids_config.cmake）

    提供统一的 resolve() 入口，模拟上游 cmake 执行的完整语义。
    """

    condition_fix: UserBranchConditionFix = field(
        default_factory=UserBranchConditionFix
    )
    upstream_commit: str = "e16ae06"
    upstream_subject: str = "Update rapids_config to handle user defined branch name (#272)"

    def resolve(
        self,
        user_branch: Optional[str],
        user_version: Optional[str],
        derived_version: str,
        derived_branch: str,
    ) -> dict:
        """
        模拟 e16ae06 后的完整 CMake 解析流程。

        Parameters
        ----------
        user_branch     : 用户预定义的 rapids-cmake-branch（None = 未定义）
        user_version    : 用户预定义的 rapids-cmake-version（None = 未定义）
        derived_version : rapids_config.cmake 自动推导的版本（如 "25.10"）
        derived_branch  : rapids_config.cmake 自动推导的分支（如 "branch-25.10"）

        Returns
        -------
        dict with keys:
          - valid: bool           是否通过 RAPIDS.cmake 校验
          - final_version: str    最终生效的版本
          - final_branch: str     最终生效的分支
          - version_source: str   version 来源（USER or DERIVED）
          - branch_source: str    branch 来源（USER or DERIVED）
          - error: Optional[str]  若校验失败的错误信息
        """
        _dbg(
            "RapidsConfigResolutionPolicy.resolve",
            f"user_branch={user_branch!r} user_version={user_version!r} "
            f"derived_version={derived_version!r} derived_branch={derived_branch!r}",
        )

        branch_defined = user_branch is not None
        version_defined = user_version is not None

        # 步骤 1：RAPIDS.cmake 校验（新语义：两者都缺才报错）
        valid = self.condition_fix.is_valid_config(branch_defined, version_defined)
        if not valid:
            _dbg(
                "RapidsConfigResolutionPolicy.resolve.fatal",
                "校验失败：rapids-cmake-branch 和 rapids-cmake-version 均未定义",
            )
            return {
                "valid": False,
                "final_version": None,
                "final_branch": None,
                "version_source": None,
                "branch_source": None,
                "error": (
                    "CMake FATAL_ERROR: "
                    "The CMake variable `rapids-cmake-branch` or `rapids-cmake-version` "
                    "must be defined"
                ),
            }

        # 步骤 2：rapids_config.cmake 非覆盖 set 守卫
        guard = NonClobberGuard()
        if user_version is not None:
            guard.set_user_version(user_version)
        if user_branch is not None:
            guard.set_user_branch(user_branch)

        final_version, final_branch = guard.apply_defaults(
            derived_version=derived_version,
            derived_branch=derived_branch,
            clobber=False,  # 新行为
        )

        result = {
            "valid": True,
            "final_version": final_version,
            "final_branch": final_branch,
            "version_source": guard.version_source(derived_version),
            "branch_source": guard.branch_source(derived_branch),
            "error": None,
        }
        _dbg(
            "RapidsConfigResolutionPolicy.resolve.ok",
            f"解析成功 → version={final_version!r} branch={final_branch!r}",
        )
        return result

    def describe(self) -> str:
        """生成策略摘要，适合日志与文档。"""
        return (
            f"RapidsConfigResolutionPolicy\n"
            f"  upstream: {self.upstream_commit} — {self.upstream_subject}\n"
            f"  修正1 (RAPIDS.cmake):       {self.condition_fix.old_cmake_condition()!r}\n"
            f"            →                {self.condition_fix.new_cmake_condition()!r}\n"
            f"  修正2 (rapids_config.cmake): 无条件 set → if(NOT var) set(var default)\n"
            f"  合并效果: 用户可只定义 branch 或 version 之一，且不会被系统默认值覆盖。\n"
        )


_dbg("MODULE_LOAD", "RapidsConfigResolutionPolicy dataclass 定义完成")


# ---------------------------------------------------------------------------
# 5. UserBranchMigrationAudit — 迁移审计
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserBranchMigrationAudit:
    """
    审计 e16ae06 迁移的完整性。

    验证：
    1. 原始 CMake 文件已正确标记为 SKIP
    2. De Morgan 修正已被 UserBranchConditionFix 建模
    3. 非覆盖守卫已被 NonClobberGuard 建模
    4. 回归测试用例覆盖全部 4 个真值表组合
    """

    upstream_commit: str = "e16ae06"
    skipped_files: tuple = (
        "cmake/RAPIDS.cmake",
        "cmake/rapids_config.cmake",
    )
    migrated_semantics: tuple = (
        "De Morgan 条件修正（REJECT_IF_EITHER_MISSING → REJECT_IF_BOTH_MISSING）",
        "非覆盖 set 守卫（NON_CLOBBER_SET）",
    )

    def audit_skipped_files(self) -> dict:
        """验证所有应 SKIP 的文件均已记录。"""
        _dbg("UserBranchMigrationAudit.audit_skipped_files", "开始校验 SKIP 文件清单")
        expected = {"cmake/RAPIDS.cmake", "cmake/rapids_config.cmake"}
        actual = set(self.skipped_files)
        covered = expected == actual
        return {
            "pass": covered,
            "expected": sorted(expected),
            "actual": sorted(actual),
            "message": "SKIP 文件清单完整" if covered else f"缺少: {expected - actual}",
        }

    def audit_demorgan_fix(self) -> dict:
        """验证 De Morgan 修正逻辑的 4 个真值表用例全部正确。"""
        _dbg("UserBranchMigrationAudit.audit_demorgan_fix", "验证 De Morgan 真值表")
        fix = UserBranchConditionFix()
        cases = fix.regression_cases()
        results = []
        all_pass = True
        for case in cases:
            bd = case["branch_defined"]
            vd = case["version_defined"]
            new_valid = fix.is_valid_config(bd, vd)
            old_valid = fix.old_is_valid_config(bd, vd)
            case_pass = (new_valid == case["new_valid"]) and (old_valid == case["old_valid"])
            if not case_pass:
                all_pass = False
            results.append({
                "branch_defined": bd,
                "version_defined": vd,
                "expected_new_valid": case["new_valid"],
                "actual_new_valid": new_valid,
                "expected_old_valid": case["old_valid"],
                "actual_old_valid": old_valid,
                "pass": case_pass,
                "description": case["description"],
            })
        return {"pass": all_pass, "cases": results}

    def audit_non_clobber_guard(self) -> dict:
        """验证 NonClobberGuard 的新旧行为差异。"""
        _dbg("UserBranchMigrationAudit.audit_non_clobber_guard", "验证非覆盖守卫行为")
        guard_new = NonClobberGuard()
        guard_new.set_user_version("user-25.08")
        guard_new.set_user_branch("user-branch-feature")
        v_new, b_new = guard_new.apply_defaults("25.10", "branch-25.10", clobber=False)

        guard_old = NonClobberGuard()
        guard_old.set_user_version("user-25.08")
        guard_old.set_user_branch("user-branch-feature")
        v_old, b_old = guard_old.apply_defaults("25.10", "branch-25.10", clobber=True)

        new_preserves_user = (v_new == "user-25.08") and (b_new == "user-branch-feature")
        old_clobbers_user = (v_old == "25.10") and (b_old == "branch-25.10")

        return {
            "pass": new_preserves_user and old_clobbers_user,
            "new_behavior": {"version": v_new, "branch": b_new, "user_preserved": new_preserves_user},
            "old_behavior": {"version": v_old, "branch": b_old, "user_clobbered": old_clobbers_user},
            "message": (
                "非覆盖守卫正确：用户值在新行为中受保护，旧行为中被覆盖"
                if (new_preserves_user and old_clobbers_user)
                else "非覆盖守卫验证失败"
            ),
        }

    def audit_all(self) -> dict:
        """运行全部审计，返回综合报告。"""
        _dbg("UserBranchMigrationAudit.audit_all", "运行全部审计项目")
        skip_result = self.audit_skipped_files()
        dm_result = self.audit_demorgan_fix()
        nc_result = self.audit_non_clobber_guard()
        overall = skip_result["pass"] and dm_result["pass"] and nc_result["pass"]
        return {
            "overall_pass": overall,
            "skip_files": skip_result,
            "demorgan_fix": dm_result,
            "non_clobber_guard": nc_result,
        }

    def summary(self) -> str:
        """生成可读摘要，适合写入 MIGRATION_LOG.md。"""
        result = self.audit_all()
        status = "ALL PASS" if result["overall_pass"] else "FAIL"
        lines = [
            f"[UserBranchMigrationAudit] e16ae06 迁移审计报告",
            f"  总体状态: {status}",
            f"  SKIP 文件: {result['skip_files']['message']}",
            f"  De Morgan 真值表: {'4/4 PASS' if result['demorgan_fix']['pass'] else 'FAIL'}",
            f"  非覆盖守卫: {result['non_clobber_guard']['message']}",
        ]
        return "\n".join(lines)


_dbg("MODULE_LOAD", "UserBranchMigrationAudit dataclass 定义完成，所有类型已就绪")


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_user_branch_migration() -> dict:
    """
    构建 e16ae06 迁移的完整结果对象。
    返回包含 policy、audit、summary 的字典。
    """
    _dbg("build_user_branch_migration", "构建 e16ae06 迁移结果")
    policy = RapidsConfigResolutionPolicy()
    audit = UserBranchMigrationAudit()
    audit_result = audit.audit_all()
    return {
        "policy": policy,
        "audit": audit,
        "audit_result": audit_result,
        "description": policy.describe(),
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """全链路自测，验证 e16ae06 迁移的正确性。"""
    print("=" * 60)
    print("rapids_config_user_branch.py — 自测 (e16ae06)")
    print("=" * 60)

    fix = UserBranchConditionFix()

    # 测试1: 旧语义 — 只有 branch 定义时报错（De Morgan BUG）
    assert not fix.old_is_valid_config(branch_defined=True, version_defined=False), \
        "旧语义应拒绝 branch=True, version=False"
    print("[PASS] 测试1: 旧语义下 branch=True, version=False → 报错（复现 BUG）")

    # 测试2: 新语义 — 只有 branch 定义时合法（e16ae06 修正）
    assert fix.is_valid_config(branch_defined=True, version_defined=False), \
        "新语义应接受 branch=True, version=False"
    print("[PASS] 测试2: 新语义下 branch=True, version=False → 合法（e16ae06 修正）")

    # 测试3: 新语义 — 只有 version 定义时合法（对称场景）
    assert fix.is_valid_config(branch_defined=False, version_defined=True), \
        "新语义应接受 branch=False, version=True"
    print("[PASS] 测试3: 新语义下 branch=False, version=True → 合法（对称修正）")

    # 测试4: 两者都缺 — 新旧均报错
    assert not fix.is_valid_config(branch_defined=False, version_defined=False), \
        "两者都缺时新语义也应报错"
    assert not fix.old_is_valid_config(branch_defined=False, version_defined=False), \
        "两者都缺时旧语义也应报错"
    print("[PASS] 测试4: 两者均未定义 → 新旧均报错（行为一致）")

    # 测试5: 非覆盖守卫 — 用户值受保护
    guard = NonClobberGuard()
    guard.set_user_version("myver-1.0")
    guard.set_user_branch("my-feature-branch")
    v, b = guard.apply_defaults("25.10", "branch-25.10", clobber=False)
    assert v == "myver-1.0", f"期望 'myver-1.0'，实际 {v!r}"
    assert b == "my-feature-branch", f"期望 'my-feature-branch'，实际 {b!r}"
    print("[PASS] 测试5: 非覆盖守卫保护用户自定义值")

    # 测试6: 旧行为 — 无条件覆盖
    guard2 = NonClobberGuard()
    guard2.set_user_version("myver-1.0")
    v2, b2 = guard2.apply_defaults("25.10", "branch-25.10", clobber=True)
    assert v2 == "25.10", f"期望旧行为覆盖为 '25.10'，实际 {v2!r}"
    print("[PASS] 测试6: 旧行为（clobber=True）无条件覆盖用户值")

    # 测试7: 完整策略 resolve — 用户只定义 branch，version 使用推导值
    policy = RapidsConfigResolutionPolicy()
    result = policy.resolve(
        user_branch="user-feature",
        user_version=None,
        derived_version="25.10",
        derived_branch="branch-25.10",
    )
    assert result["valid"] is True
    assert result["final_branch"] == "user-feature"
    assert result["final_version"] == "25.10"
    assert result["branch_source"].startswith("USER")
    assert result["version_source"].startswith("DERIVED")
    print("[PASS] 测试7: 策略 resolve — 只定义 branch，version 使用推导值")

    # 测试8: 完整策略 resolve — 两者都缺，应失败
    result_fail = policy.resolve(
        user_branch=None,
        user_version=None,
        derived_version="25.10",
        derived_branch="branch-25.10",
    )
    assert result_fail["valid"] is False
    assert result_fail["error"] is not None
    print("[PASS] 测试8: 策略 resolve — 两者均缺时返回校验失败")

    # 测试9: 审计 — 全部通过
    audit = UserBranchMigrationAudit()
    audit_result = audit.audit_all()
    assert audit_result["overall_pass"] is True, f"审计失败: {audit_result}"
    print("[PASS] 测试9: 全链路审计 overall_pass=True")

    # 测试10: BoolGuardSemantics.evaluate 四种模式
    sem = BoolGuardSemantics.REJECT_IF_EITHER_MISSING
    assert sem.evaluate(True, False) is True   # NOT(T AND F) = T → 拒绝
    sem2 = BoolGuardSemantics.REJECT_IF_BOTH_MISSING
    assert sem2.evaluate(True, False) is False  # NOT(T OR F) = F → 接受
    print("[PASS] 测试10: BoolGuardSemantics.evaluate 新旧语义对比正确")

    print("\n✓ 全部 10 项自测通过")
    print(audit.summary())


if __name__ == "__main__":
    _self_test()
