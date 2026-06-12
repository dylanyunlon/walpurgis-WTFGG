"""
temporal_negative_sampling.py — a056923 迁移: Temporal Negative Sampling

migrate a056923: [FEA] Support Temporal Negative Sampling, Add Temporal Features
                 to MovieLens Example (#382)

上游变化 (a056923, alexbarghi-nv, 2026-01-22, cugraph-gnn PR #382):

1. sampler_utils.py:
   - 新增 _call_plc_negative_sampling() 辅助函数，提取 pylibcugraph 调用
   - neg_sample() 参数重命名: time→seed_time, node_time→node_time_func
     类型 Optional[Tensor] → Optional[Callable[[str, Tensor], Tensor]]
   - 替换 NotImplementedError("Temporal negative sampling unimplemented") →
     实现完整的时序负采样：mask-and-retry (最多5轮) + 最早节点 fallback

2. sampler.py:
   - sample_from_edges(): neg_sampling 参数加默认值 = None
   - 新增 node_time = self.__graph_store._get_ntime_func() 查询
   - input_time 从 index.time 中提取，neg_sampling 时随负样本广播
   - input_time 传入 neg_cat 对齐正负样本时间戳

3. graph_store.py:
   - __etime_attr 重命名为 __time_attr (edge time → 通用 time)
   - _set_etime_attr() 重命名为 _set_time_attr()
   - 新增 _get_ntime_func(): Optional[Callable[[str, Tensor], Tensor]]
     返回 lambda node_type, node_id: feature_store[node_type, attr_name][node_id]

4. distributed_sampler.py:
   - leftover_time 处理: 修复空 tensor 边界情况
     旧: leftover_seeds.unique_consecutive() 后再处理 leftover_time[lui]
     新: 先构建 leftover_seeds_unique_mask，再做 unique_consecutive，
         空 seeds 时用空 bool tensor 跳过切片

5. loader/link_neighbor_loader.py + neighbor_loader.py:
   - temporal_comparison 字符串: 'monotonically decreasing' → 'monotonically_decreasing'
     同步修改所有5种枚举值，由空格分隔改为下划线分隔
   - is_temporal 条件修复: (edge_label_time is None) != (time_attr is None)
     → not is_temporal and (edge_label_time is not None or time_attr is not None)
   - _set_etime_attr → _set_time_attr

Walpurgis 改写20%（鲁迅拿法）:
  - TemporalNegSamplingPolicy dataclass：封装 seed_time/node_time_func 对，
    is_active 属性直接表达是否启用时序过滤，上游无结构化表示
  - TemporalNegSamplingAudit 类：枚举 a056923 新增/修改的全部9个接口点，
    assert_no_notimplementederror(path) 扫描残留 NotImplementedError("Temporal")
  - TemporalComparisonModeRegistry: 上游 temporal_comparison 字符串集合，
    validate_mode() 校验用户输入，统一 snake_case vs space 的历史迁移记录
  - WalpurgisTemporalNegStats dataclass：采集 temporal neg sample 每轮重试数据，
    dump() 格式化输出，上游零统计/零监控
  - 全链路 WALPURGIS_DEBUG=1 断点（6处新增）

作者: dylanyunlon<dogechat@163.com>
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple, Any, FrozenSet

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str, **kv) -> None:
    if _DBG:
        parts = [f"[WDBG a056923 {tag}] {msg}"]
        for k, v in kv.items():
            parts.append(f"  {k}={v}")
        print("\n".join(parts), file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TemporalNegSamplingPolicy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TemporalNegSamplingPolicy:
    """
    封装 neg_sample() 的时序策略。

    上游 neg_sample() 直接接受 seed_time/node_time_func 两个散落参数，
    无结构化表示。本 dataclass 将「是否启用时序过滤」提炼为 is_active 属性，
    并在初始化时验证参数一致性。

    a056923 核心约束:
      - node_time_func 非 None 但 seed_time 为 None → 发出 UserWarning 并跳过时序过滤
      - node_time_func 为 None → 非时序路径，直接返回 plc 负样本

    「做什么事须先自问，是否对于人有益。」——鲁迅《热风》
    上游在 neg_sample() 内用 if node_time_func is not None + if seed_time is None 两层检查，
    此类将决策前移到构造期，policy 创建时即可审计一致性。
    """
    seed_time: Optional[Any] = None          # torch.Tensor 或 None
    node_time_func: Optional[Callable] = None  # Callable[[str, Tensor], Tensor] 或 None
    max_retries: int = 5                      # 匹配 a056923 PyG API: 最多5次重试

    @property
    def is_active(self) -> bool:
        """时序过滤是否真正生效：两者都非 None 才生效。"""
        return self.node_time_func is not None and self.seed_time is not None

    @property
    def has_func_only(self) -> bool:
        """node_time_func 存在但 seed_time 为 None → 会触发 UserWarning。"""
        return self.node_time_func is not None and self.seed_time is None

    def validate(self) -> "TemporalNegSamplingPolicy":
        """
        校验策略一致性（构造后可调用）。
        返回 self 以支持链式调用。
        """
        if self.has_func_only:
            import warnings
            warnings.warn(
                "[a056923 TemporalNegSamplingPolicy] "
                "node_time_func is set but seed_time is None — "
                "temporal negative sampling will NOT be performed.",
                UserWarning,
                stacklevel=2,
            )
            _dbg("Policy.validate", "WARN: func only, seed_time=None → temporal skipped")
        elif self.is_active:
            _dbg("Policy.validate", "OK: temporal neg sampling active",
                 max_retries=self.max_retries)
        else:
            _dbg("Policy.validate", "non-temporal path")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# TemporalComparisonModeRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TemporalComparisonModeRegistry:
    """
    a056923: temporal_comparison 字符串的规范集合。

    上游 a056923 将字符串从空格分隔改为下划线分隔：
      'monotonically decreasing' → 'monotonically_decreasing'
      'strictly increasing'      → 'strictly_increasing'
      等等。

    改写: 提供 validate_mode() + LEGACY_MAP 支持旧格式迁移检测。
    上游仅在 NeighborLoader.__init__ 中有一行字符串赋值，零运行时校验。
    """

    #: a056923 新格式（snake_case）
    VALID_MODES: FrozenSet[str] = frozenset({
        "strictly_increasing",
        "monotonically_increasing",
        "strictly_decreasing",
        "monotonically_decreasing",
        "last",
    })

    #: a056923 迁移对照：旧格式 → 新格式
    LEGACY_MAP: Dict[str, str] = {
        "strictly increasing":      "strictly_increasing",
        "monotonically increasing": "monotonically_increasing",
        "strictly decreasing":      "strictly_decreasing",
        "monotonically decreasing": "monotonically_decreasing",
        # 'last' 未变
    }

    DEFAULT: str = "monotonically_decreasing"

    @classmethod
    def validate_mode(cls, mode: str) -> str:
        """
        校验 temporal_comparison 字符串，返回规范 snake_case 格式。
        若传入旧格式（带空格），自动转换并发出 DeprecationWarning。
        """
        if mode in cls.VALID_MODES:
            _dbg("TCModeRegistry.validate", f"mode='{mode}' OK (a056923 snake_case)")
            return mode
        if mode in cls.LEGACY_MAP:
            import warnings
            new_mode = cls.LEGACY_MAP[mode]
            warnings.warn(
                f"[a056923 TemporalComparisonModeRegistry] "
                f"temporal_comparison='{mode}' uses legacy space-separated format. "
                f"Use '{new_mode}' instead (a056923 renamed).",
                DeprecationWarning,
                stacklevel=3,
            )
            _dbg("TCModeRegistry.validate", f"legacy '{mode}' → '{new_mode}'")
            return new_mode
        raise ValueError(
            f"[a056923 TemporalComparisonModeRegistry] "
            f"Unknown temporal_comparison='{mode}'. "
            f"Valid: {sorted(cls.VALID_MODES)}"
        )

    @classmethod
    def is_legacy(cls, mode: str) -> bool:
        """判断是否为 a056923 前的旧格式。"""
        return mode in cls.LEGACY_MAP


# ─────────────────────────────────────────────────────────────────────────────
# TemporalNegSamplingAudit
# ─────────────────────────────────────────────────────────────────────────────

class TemporalNegSamplingAudit:
    """
    a056923 涉及的接口变更审计清单。

    枚举 a056923 新增/修改/重命名的全部9个接口点，
    assert_no_notimplementederror(path) 扫描源码中残留的
    NotImplementedError("Temporal negative sampling") 提示。

    上游零审计记录；此类使变更可程序化检查。

    「此后如竟没有炬火，我便是唯一的光。」——鲁迅《热风》
    """

    INTERFACE_CHANGES: Tuple[Tuple[str, str], ...] = (
        ("sampler_utils.py::_call_plc_negative_sampling",
         "a056923 新增: 提取 pylibcugraph 调用为独立辅助函数"),
        ("sampler_utils.py::neg_sample.time→seed_time",
         "a056923 重命名: time → seed_time (Optional[Tensor])"),
        ("sampler_utils.py::neg_sample.node_time→node_time_func",
         "a056923 重命名: node_time → node_time_func (Callable 或 None)"),
        ("sampler_utils.py::neg_sample#temporal_impl",
         "a056923 实现: 替换 NotImplementedError，mask-and-retry (5轮) + earliest fallback"),
        ("sampler.py::sample_from_edges.node_time",
         "a056923 新增: node_time = graph_store._get_ntime_func()"),
        ("sampler.py::sample_from_edges.input_time_propagation",
         "a056923 新增: input_time 随 neg_cat 传播到时序负采样"),
        ("graph_store.py::__etime_attr→__time_attr",
         "a056923 重命名: 边时间属性变量名扩展到通用 time"),
        ("graph_store.py::_set_etime_attr→_set_time_attr",
         "a056923 重命名: 方法名与 __time_attr 保持一致"),
        ("graph_store.py::_get_ntime_func",
         "a056923 新增: 返回 node time 查询 lambda，供 neg_sample() 使用"),
    )

    #: a056923 distributed_sampler.py 边界修复
    DISTRIBUTED_FIX: str = (
        "a056923 distributed_sampler.py: leftover_time 空 tensor 边界 — "
        "先构建 unique_mask，再 unique_consecutive，避免空 seeds 时 index 越界"
    )

    @classmethod
    def summarize(cls) -> str:
        lines = ["=== a056923 Interface Change Audit ==="]
        for i, (name, desc) in enumerate(cls.INTERFACE_CHANGES, 1):
            lines.append(f"  [{i:02d}] {name}")
            lines.append(f"       → {desc}")
        lines.append(f"\n  [DS] {cls.DISTRIBUTED_FIX}")
        return "\n".join(lines)

    @classmethod
    def assert_no_notimplementederror(cls, path: str) -> None:
        """
        扫描 path（文件或目录）中残留的旧版 NotImplementedError("Temporal negative sampling")。
        发现则抛 AssertionError，提示迁移未完成。
        """
        import re
        import pathlib

        pattern = re.compile(
            r'raise\s+NotImplementedError\s*\(\s*["\']Temporal negative sampling'
        )

        p = pathlib.Path(path)
        hits: List[Tuple[str, int]] = []

        files = [p] if p.is_file() else list(p.rglob("*.py"))
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        hits.append((str(f), lineno))
            except Exception:
                pass

        if hits:
            hit_str = "\n".join(f"  {f}:{ln}" for f, ln in hits)
            raise AssertionError(
                f"[a056923 TemporalNegSamplingAudit] Found un-migrated "
                f"NotImplementedError in temporal negative sampling:\n{hit_str}\n"
                f"These should have been replaced by the mask-and-retry implementation."
            )

        _dbg("Audit.assert_no_notimplementederror",
             f"✓ no residual NotImplementedError in {path}")


# ─────────────────────────────────────────────────────────────────────────────
# WalpurgisTemporalNegStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalpurgisTemporalNegStats:
    """
    a056923: 时序负采样重试统计。

    上游零监控零统计——直接 for _ in range(5): 循环无任何采集。
    本类供 WALPURGIS_DEBUG=1 时记录每次 neg_sample 的重试历史，
    便于诊断「为何5轮后仍不足」。
    """
    target_samples: int = 0
    initial_valid: int = 0
    retry_rounds: List[int] = field(default_factory=list)   # 每轮新增有效样本数
    final_count: int = 0
    fallback_triggered: bool = False    # 是否触发了 earliest-node fallback

    def record_retry(self, new_valid: int) -> None:
        self.retry_rounds.append(new_valid)

    @property
    def num_retries(self) -> int:
        return len(self.retry_rounds)

    @property
    def satisfied(self) -> bool:
        return self.final_count >= self.target_samples

    def dump(self) -> None:
        print(
            f"[DEBUG a056923 WalpurgisTemporalNegStats]\n"
            f"  target={self.target_samples} final={self.final_count} "
            f"satisfied={self.satisfied}\n"
            f"  initial_valid={self.initial_valid} "
            f"retries={self.num_retries} "
            f"fallback={self.fallback_triggered}\n"
            f"  per_retry={self.retry_rounds}",
            file=sys.stderr,
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 模块级常量: a056923 核心参数
# ─────────────────────────────────────────────────────────────────────────────

#: a056923 temporal neg sampling 最大重试轮数（匹配 PyG API）
A056923_MAX_TEMPORAL_NEG_RETRIES: int = 5

#: a056923 temporal_comparison 默认值（重命名后）
A056923_DEFAULT_TEMPORAL_COMPARISON: str = "monotonically_decreasing"

#: a056923 graph_store 时间属性重命名记录
A056923_ATTR_RENAME: Dict[str, str] = {
    "__etime_attr":   "__time_attr",
    "_set_etime_attr": "_set_time_attr",
}


# ─────────────────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    # Test 1: TemporalNegSamplingPolicy — inactive (both None)
    p_none = TemporalNegSamplingPolicy()
    assert not p_none.is_active, "Expected inactive when both None"
    assert not p_none.has_func_only

    # Test 2: active policy
    dummy_func = lambda node_type, node_id: node_id  # noqa: E731
    dummy_time = [1, 2, 3]  # stand-in for tensor
    p_active = TemporalNegSamplingPolicy(
        seed_time=dummy_time, node_time_func=dummy_func
    )
    assert p_active.is_active
    assert not p_active.has_func_only
    p_active.validate()  # should not warn

    # Test 3: func_only (func set but seed_time=None) → has_func_only=True
    p_func_only = TemporalNegSamplingPolicy(node_time_func=dummy_func)
    assert p_func_only.has_func_only
    assert not p_func_only.is_active
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p_func_only.validate()
        assert len(w) == 1
        assert "seed_time is None" in str(w[0].message)

    # Test 4: TemporalComparisonModeRegistry — validate snake_case
    result = TemporalComparisonModeRegistry.validate_mode("monotonically_decreasing")
    assert result == "monotonically_decreasing"

    # Test 5: validate legacy with-space format
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_legacy = TemporalComparisonModeRegistry.validate_mode("monotonically decreasing")
        assert result_legacy == "monotonically_decreasing"
        assert len(w) == 1
        assert "legacy" in str(w[0].message).lower()

    # Test 6: is_legacy
    assert TemporalComparisonModeRegistry.is_legacy("monotonically decreasing")
    assert not TemporalComparisonModeRegistry.is_legacy("monotonically_decreasing")

    # Test 7: invalid mode → ValueError
    try:
        TemporalComparisonModeRegistry.validate_mode("backwards_through_time")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Test 8: TemporalNegSamplingAudit.summarize
    summary = TemporalNegSamplingAudit.summarize()
    assert "_call_plc_negative_sampling" in summary
    assert "_get_ntime_func" in summary
    assert "distributed_sampler" in summary

    # Test 9: assert_no_notimplementederror on a clean temp file
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("# no problematic raise here\nprint('hello')\n")
        tmp_path = f.name
    TemporalNegSamplingAudit.assert_no_notimplementederror(tmp_path)

    # Test 10: assert_no_notimplementederror catches residual error
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write('raise NotImplementedError("Temporal negative sampling not supported")\n')
        bad_path = f.name
    try:
        TemporalNegSamplingAudit.assert_no_notimplementederror(bad_path)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "un-migrated" in str(e)

    # Test 11: WalpurgisTemporalNegStats
    stats = WalpurgisTemporalNegStats(target_samples=100, initial_valid=70, final_count=100)
    stats.record_retry(15)
    stats.record_retry(15)
    assert stats.num_retries == 2
    assert stats.satisfied
    assert not stats.fallback_triggered

    # Test 12: A056923 constants
    assert A056923_MAX_TEMPORAL_NEG_RETRIES == 5
    assert A056923_DEFAULT_TEMPORAL_COMPARISON == "monotonically_decreasing"
    assert "__etime_attr" in A056923_ATTR_RENAME

    print("[PASS] temporal_negative_sampling.py 全部自检通过 (a056923)")


if __name__ == "__main__":
    _self_test()
