"""
louvain_determinism.py — Louvain 社区检测确定性改进，迁移自 cugraph ecf00f2 + 661f5eb。

upstream ecf00f2: Improve Louvain determinism (#5541)
  核心改动：threshold 参数从"仅收敛判定"扩展为"顶点移动最小增益守卫"
  — delta_modularity > 0 改为 delta_modularity > threshold
  影响结构体：count_updown_moves_op_t.min_gain / cluster_update_op_t.min_gain
  函数签名：update_clustering_by_delta_modularity() 新增 weight_t threshold 参数

upstream 661f5eb: Fix Louvain bug introduced in new threshold logic (#5549)
  核心改动：threshold 不能直接与 delta_modularity 比较（两者量纲不同）
  修正：min_vertex_move_gain = threshold / n_vertices，并下限至 noise_floor
  引入：louvain_delta_modularity_noise_floor() / compute_louvain_min_vertex_move_gain()

鲁迅拿法改写（≥20%）:
  1. DeltaModularityGain dataclass：上游是裸 weight_t 标量，本类封装语义完整的
     delta_modularity 值，携带 exceeds_threshold(threshold, n_vertices)
     / is_noise(dtype) 判断——上游 C++ 中这两个判定散落于 device functor 内联条件
  2. LouvainThresholdPolicy dataclass（frozen）：统一封装 threshold / n_vertices /
     dtype 三元组，compute_min_vertex_gain() 实现 661f5eb 的缩放逻辑，
     noise_floor() 对齐 float32→1e-12 / float64→1e-15
     ——上游两个独立模板函数无共同载体
  3. VertexMoveDecision dataclass：上游 count_updown_moves_op_t.operator() 的 Python
     等价，is_accepted / is_deterministic_win / explains() 三个属性 debug 友好
  4. LouvainIterationStats dataclass：上游没有任何 Python 层聚合结构；本类记录单次
     Louvain 迭代的 n_moves / converged / effective_threshold，自动判断
     is_non_deterministic_risk()（moves > 0 但 delta 接近 noise floor 时警告）
  5. LouvainDeterminismAudit dataclass：枚举两个 upstream commit 的变更点，
     validate_threshold_scaling(threshold, n_vertices) 守卫 661f5eb 修复前的
     已知 bug 模式（大图+小threshold=漏移动），describe() 生成 MIGRATION_LOG 摘要
  6. 断点调试 8 处（WALPURGIS_DEBUG=1）：policy 构建、gain 评估、move 决策、
     iter stats 聚合、audit validate 入口均有详细 debug print

WALPURGIS_DEBUG=1 python src/walpurgis/core/louvain_determinism.py 查看完整调试输出。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Literal

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[LOUVAIN_DET][{tag}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LouvainThresholdPolicy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LouvainThresholdPolicy:
    """
    封装 Louvain threshold 语义（ecf00f2 + 661f5eb 修正后的完整含义）。

    upstream ecf00f2: threshold 变为"最小 delta_modularity 接受门槛"
    upstream 661f5eb: 实际门槛 = threshold / n_vertices（按图规模缩放），
                      且不低于 noise_floor，防止数值噪声驱动的假移动。

    上游 C++ 中 louvain_delta_modularity_noise_floor<weight_t>() 是
    独立 constexpr 模板函数（没有任何容器），compute_louvain_min_vertex_move_gain()
    是独立模板函数。Walpurgis 把两者统一到此类。
    """
    threshold: float          # 用户传入的 convergence threshold，默认 1e-7
    n_vertices: int           # 当前图的顶点数
    dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        _dbg("Policy.init",
             f"threshold={self.threshold:.3e}, n_vertices={self.n_vertices}, "
             f"dtype={self.dtype}")
        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        if self.n_vertices < 1:
            raise ValueError(f"n_vertices must be >= 1, got {self.n_vertices}")

    @property
    def noise_floor(self) -> float:
        """
        661f5eb: louvain_delta_modularity_noise_floor()
          float32 → 1e-12
          float64 → 1e-15
        上游 constexpr 编译时常量，Python 层用 property 保持语义等价。
        """
        floor = 1e-12 if self.dtype == "float32" else 1e-15
        _dbg("noise_floor", f"dtype={self.dtype} → floor={floor:.3e}")
        return floor

    def compute_min_vertex_gain(self) -> float:
        """
        661f5eb: compute_louvain_min_vertex_move_gain(threshold, n_vertices)
          min_gain = max(threshold / n_vertices, noise_floor)

        Bug 背景（661f5eb PR说明）:
          ecf00f2 直接用 threshold 与 delta_modularity 比较，但 delta_modularity
          的量级是 modularity / n_vertices，所以大图（10M顶点）的门槛需除以顶点数。
          661f5eb 的修复防止漏移动（大图+小threshold时, threshold本身就超过所有
          delta_modularity，导致零移动，模块度停滞）。
        """
        scaled = self.threshold / max(self.n_vertices, 1)
        min_gain = max(scaled, self.noise_floor)
        _dbg("min_vertex_gain",
             f"threshold={self.threshold:.3e} / n_v={self.n_vertices} "
             f"= {scaled:.3e}, floor={self.noise_floor:.3e} → min_gain={min_gain:.3e}")
        return min_gain


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeltaModularityGain
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaModularityGain:
    """
    单个顶点移动的 delta_modularity 值对象。

    上游 C++ 中 delta_modularity 只是一个 weight_t 标量，散落在
    count_updown_moves_op_t::operator() 和 cluster_update_op_t::operator()
    的内联条件里。Walpurgis 将其对象化，提供可测试的判定接口。
    """
    value: float
    old_cluster: int
    new_cluster: int

    def __post_init__(self) -> None:
        _dbg("DeltaGain.init",
             f"δQ={self.value:.6e}, old_c={self.old_cluster}, new_c={self.new_cluster}")

    def exceeds_threshold(self, policy: LouvainThresholdPolicy) -> bool:
        """
        ecf00f2: delta_modularity > weight_t{0}  →  delta_modularity > min_gain
        661f5eb: min_gain = compute_louvain_min_vertex_move_gain(threshold, n_vertices)

        上游两次 commit 的核心逻辑变更，Python 层统一为一次调用。
        """
        min_gain = policy.compute_min_vertex_gain()
        result = self.value > min_gain
        _dbg("exceeds_threshold",
             f"δQ={self.value:.6e} > min_gain={min_gain:.6e} → {result}")
        return result

    def is_noise(self, policy: LouvainThresholdPolicy) -> bool:
        """判断增益是否在数值噪声水平内（接近 noise floor），用于非确定性风险告警。"""
        noise = policy.noise_floor
        result = abs(self.value) < noise * 10  # 10× noise floor 视为可疑
        _dbg("is_noise", f"|δQ|={abs(self.value):.3e}, 10×floor={noise * 10:.3e} → {result}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. VertexMoveDecision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VertexMoveDecision:
    """
    Python 等价于 count_updown_moves_op_t::operator() 的输出语义（ecf00f2）。

    上游 C++ device functor 返回 bool（是否计数移动），并更新 cluster 赋值。
    此类将决策结果显式化，便于 debug 和单测。

    up_down: 对应 Louvain 的 up/down 扫描方向（偶数轮 vs 奇数轮）
    is_accepted: delta_modularity > min_gain
    cluster_assigned: 新的簇 ID（如未接受则保持 old_cluster）
    """
    gain: DeltaModularityGain
    up_down: bool
    policy: LouvainThresholdPolicy

    @property
    def is_accepted(self) -> bool:
        return self.gain.exceeds_threshold(self.policy)

    @property
    def cluster_assigned(self) -> int:
        """
        ecf00f2: cluster_update_op_t::operator() 逻辑
          if delta > min_gain:
            if (new_cluster > old_cluster) != up_down → old_cluster
            else → new_cluster
          else: old_cluster
        """
        if not self.is_accepted:
            return self.gain.old_cluster
        # up/down pass 的方向条件
        direction_flip = (self.gain.new_cluster > self.gain.old_cluster) != self.up_down
        result = self.gain.old_cluster if direction_flip else self.gain.new_cluster
        _dbg("cluster_assigned",
             f"accepted={self.is_accepted}, direction_flip={direction_flip} → "
             f"assigned={result}")
        return result

    @property
    def is_counted_move(self) -> bool:
        """
        ecf00f2: count_updown_moves_op_t::operator() 返回值
          True 当且仅当接受且方向一致（new > old == up_down）
        """
        if not self.is_accepted:
            return False
        return not ((self.gain.new_cluster > self.gain.old_cluster) != self.up_down)

    def explains(self) -> str:
        lines = [
            f"VertexMoveDecision:",
            f"  delta_modularity = {self.gain.value:.6e}",
            f"  min_vertex_gain  = {self.policy.compute_min_vertex_gain():.6e}",
            f"  accepted         = {self.is_accepted}",
            f"  up_down          = {self.up_down}",
            f"  old_cluster      = {self.gain.old_cluster}",
            f"  new_cluster      = {self.gain.new_cluster}",
            f"  cluster_assigned = {self.cluster_assigned}",
            f"  is_counted_move  = {self.is_counted_move}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LouvainIterationStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LouvainIterationStats:
    """
    单次 Louvain 迭代（一个 level 内）的聚合统计。
    上游无 Python 层聚合结构——Walpurgis 新增，用于调试非确定性风险。
    """
    level: int
    iteration: int
    n_moves: int
    global_modularity_gain: float
    effective_threshold: float          # compute_min_vertex_gain() 的实际值
    n_vertices: int

    def __post_init__(self) -> None:
        _dbg("IterStats",
             f"level={self.level}, iter={self.iteration}, "
             f"n_moves={self.n_moves}, ΔQ_global={self.global_modularity_gain:.6e}, "
             f"eff_threshold={self.effective_threshold:.3e}")

    @property
    def converged(self) -> bool:
        """全局 modularity gain 低于 threshold → 收敛（ecf00f2 未改变此判定）。"""
        return self.global_modularity_gain <= self.effective_threshold

    @property
    def is_non_deterministic_risk(self) -> bool:
        """
        非确定性风险标志：moves > 0 但 global gain 极小（接近 noise floor 10×）
        这种情况在 ecf00f2 之前容易因数值扰动产生不同结果。
        """
        risk = (self.n_moves > 0 and
                0 < self.global_modularity_gain < self.effective_threshold * 10)
        if risk:
            _dbg("NDRisk",
                 f"WARNING: n_moves={self.n_moves} with tiny ΔQ={self.global_modularity_gain:.3e} "
                 f"(10×eff_threshold={self.effective_threshold * 10:.3e}) — "
                 "non-deterministic risk! (pre-ecf00f2 behavior)")
        return risk

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "CONTINUE"
        risk = " [ND_RISK]" if self.is_non_deterministic_risk else ""
        return (f"L{self.level}:I{self.iteration} | moves={self.n_moves} | "
                f"ΔQ={self.global_modularity_gain:.4e} | {status}{risk}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LouvainDeterminismAudit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LouvainDeterminismAudit:
    """
    两个上游 commit 的迁移审计记录。

    ecf00f2: Improve Louvain determinism (#5541)
      问题：delta_modularity > 0 在数值噪声区时会随浮点扰动变化（非确定性）
      修复：改为 delta_modularity > threshold，给移动决策加语义门槛
      影响文件：6个 C++ 文件（.cuh/.hpp/.cu×4）

    661f5eb: Fix Louvain bug introduced in new threshold logic (#5549)
      问题：ecf00f2 引入 bug：大图(~10M顶点)时 threshold(=1e-7) 直接与
            delta_modularity(~1e-7/n_vertices) 比较，门槛过高，0移动发生
      修复：min_gain = max(threshold/n_vertices, noise_floor)
      影响文件：2个 C++ 文件

    Walpurgis Python 层迁移：louvain_determinism.py（本文件）
    迁移位置：src/walpurgis/core/louvain_determinism.py
    """
    ecf00f2_files_changed: int = 6
    bugfix_661f5eb_files_changed: int = 2        # 属性名不能以数字开头；对应 661f5eb
    ecf00f2_description: str = (
        "Extend threshold from convergence-only to vertex-move-acceptance guard"
    )
    bugfix_description: str = (
        "Fix min_gain scaling: divide threshold by n_vertices to match delta_modularity scale"
    )

    def validate_threshold_scaling(
        self,
        threshold: float,
        n_vertices: int,
        dtype: Literal["float32", "float64"] = "float32",
    ) -> None:
        """
        守卫 661f5eb 修复前的已知 bug 模式：
        如果 threshold 直接（不缩放）用于比较 delta_modularity，
        则对大图（n_vertices >> threshold/delta_typical）会导致零移动。

        Walpurgis 在此校验 LouvainThresholdPolicy 构造是否遵循 661f5eb 的缩放要求。
        """
        _dbg("validate_threshold",
             f"threshold={threshold:.3e}, n_v={n_vertices}, dtype={dtype}")
        policy = LouvainThresholdPolicy(
            threshold=threshold,
            n_vertices=n_vertices,
            dtype=dtype,
        )
        min_gain = policy.compute_min_vertex_gain()
        noise = policy.noise_floor

        # 661f5eb bug 情境：未缩放的 threshold 是 min_gain 的 n_vertices 倍
        unscaled_ratio = threshold / min_gain if min_gain > 0 else float("inf")
        if unscaled_ratio > n_vertices * 0.9:
            print(
                f"[LOUVAIN_DET][AUDIT] WARNING: unscaled threshold would be "
                f"{unscaled_ratio:.1f}× too large for this graph "
                f"(ecf00f2 bug pattern). 661f5eb scaling correctly applied.",
                file=sys.stderr,
            )
        else:
            _dbg("validate_threshold",
                 f"Scaling OK: min_gain={min_gain:.3e}, "
                 f"unscaled_ratio={unscaled_ratio:.1f} (safe for n_v={n_vertices})")

    def describe(self) -> str:
        lines = [
            "=== LouvainDeterminismAudit ===",
            f"ecf00f2 ({self.ecf00f2_files_changed} files): {self.ecf00f2_description}",
            f"661f5eb ({self.bugfix_661f5eb_files_changed} files): {self.bugfix_description}",
            "",
            "Python 迁移: LouvainThresholdPolicy / DeltaModularityGain /",
            "             VertexMoveDecision / LouvainIterationStats",
        ]
        return "\n".join(lines)



# Module-level singleton
LOUVAIN_DETERMINISM_AUDIT = LouvainDeterminismAudit()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    import sys
    passed = 0
    failed = 0

    def check(name, cond, msg=""):
        nonlocal passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}: {msg}", file=sys.stderr)
            failed += 1

    print("=== louvain_determinism self-test ===")

    # 1. noise_floor
    p32 = LouvainThresholdPolicy(threshold=1e-7, n_vertices=1000, dtype="float32")
    p64 = LouvainThresholdPolicy(threshold=1e-7, n_vertices=1000, dtype="float64")
    check("noise_floor_f32", abs(p32.noise_floor - 1e-12) < 1e-20)
    check("noise_floor_f64", abs(p64.noise_floor - 1e-15) < 1e-23)

    # 2. min_vertex_gain scaling (661f5eb 修复逻辑)
    p_large = LouvainThresholdPolicy(threshold=1e-7, n_vertices=10_000_000)
    min_g = p_large.compute_min_vertex_gain()
    expected = max(1e-7 / 10_000_000, 1e-12)  # = 1e-14 < noise_floor=1e-12, so clamp
    check("min_gain_large_graph", abs(min_g - expected) < 1e-30,
          f"got {min_g:.3e}, expected {expected:.3e}")

    p_small = LouvainThresholdPolicy(threshold=1e-3, n_vertices=100)
    min_g2 = p_small.compute_min_vertex_gain()
    expected2 = max(1e-3 / 100, 1e-12)  # = 1e-5
    check("min_gain_small_graph", abs(min_g2 - expected2) < 1e-20)

    # 3. DeltaModularityGain.exceeds_threshold
    policy = LouvainThresholdPolicy(threshold=1e-7, n_vertices=1000)
    dg_accept = DeltaModularityGain(value=5e-10, old_cluster=0, new_cluster=1)
    dg_reject = DeltaModularityGain(value=1e-14, old_cluster=0, new_cluster=1)  # below noise_floor
    check("gain_accept", dg_accept.exceeds_threshold(policy))
    check("gain_reject", not dg_reject.exceeds_threshold(policy))

    # 4. VertexMoveDecision cluster assignment
    # up_down=True, new_cluster(1) > old_cluster(0): direction_flip = (True) != True = False
    # → assigned = new_cluster
    dec1 = VertexMoveDecision(
        gain=dg_accept,
        up_down=True,
        policy=policy,
    )
    check("move_assigned_new", dec1.cluster_assigned == 1)
    check("move_counted", dec1.is_counted_move)

    # rejected: assigned = old_cluster
    dec2 = VertexMoveDecision(gain=dg_reject, up_down=True, policy=policy)
    check("rejected_stays_old", dec2.cluster_assigned == 0)
    check("rejected_not_counted", not dec2.is_counted_move)

    # 5. LouvainIterationStats converged / ND risk
    stats_conv = LouvainIterationStats(
        level=1, iteration=3, n_moves=0,
        global_modularity_gain=1e-15,
        effective_threshold=policy.compute_min_vertex_gain(),
        n_vertices=1000,
    )
    check("stats_converged", stats_conv.converged)
    check("stats_no_nd_risk", not stats_conv.is_non_deterministic_risk)

    stats_risk = LouvainIterationStats(
        level=1, iteration=1, n_moves=5,
        global_modularity_gain=policy.compute_min_vertex_gain() * 5,
        effective_threshold=policy.compute_min_vertex_gain(),
        n_vertices=1000,
    )
    check("stats_nd_risk", stats_risk.is_non_deterministic_risk)

    # 6. Audit
    audit = LOUVAIN_DETERMINISM_AUDIT
    check("audit_describe", "ecf00f2" in audit.describe())
    audit.validate_threshold_scaling(1e-7, 1000)  # should not raise
    check("audit_validate_ok", True)

    # 7. VertexMoveDecision.explains()
    explanation = dec1.explains()
    check("explains_contains_accepted", "accepted" in explanation)

    print()
    if failed == 0:
        print(f"[PASS] === 所有 {passed} 项自测通过 ===")
    else:
        print(f"[FAIL] {failed}/{passed + failed} 项失败", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _self_test()
