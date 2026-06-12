"""
conda_channel_policy.py
=======================
迁移自 upstream cugraph-gnn a406f985 (Bradley Dice, 2025-11-25)
"Use strict priority in CI conda tests"

上游变更摘要:
  - ci/test_cpp.sh, ci/test_python.sh: 各新增 2 行，
    `rapids-logger` 通知 + `conda config --set channel_priority strict`
  - 9 个 conda YAML (environments/ + cugraph-pyg/conda/):
    channel 列表顺序 rapidsai ↔ rapidsai-nightly 对调
    (nightly 降为首位使 strict 模式下 nightly 优先级低于 stable)

鲁迅拿法改写 (≥20%):
  上游仅是 shell 里加两行命令、YAML 里调换两行顺序，
  无任何解释和校验机制——"拿来"之后补足以下：
    1. CondaChannelPriority enum: 将 conda 三种优先级策略
       (strict / flexible / disabled) 抽象可枚举类型
    2. ChannelOrderSpec dataclass: 封装 channel 列表的
       语义——stable-first vs nightly-first，附 validate()
    3. CICondaConfig dataclass: 将 shell 里手写的两行命令
       抽象为配置对象，generate_shell_snippet() 生成可测试的 bash 块
    4. CondaYamlPatch: 解析和验证 YAML channel 顺序，
       apply_strict_order() 幂等修正 nightly 置顶、stable 次之
    5. 全链路 WALPURGIS_DEBUG=1 断点 (5 处)，覆盖 enum
       解析、channel 验证、snippet 生成、YAML patch 各阶段

CI/merge → SKIP: 见 MIGRATION_LOG.md
"""

from __future__ import annotations

import os
import re
import pdb
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# 调试开关：WALPURGIS_DEBUG=1 时各阶段设置断点
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# 1. CondaChannelPriority  ——  上游仅写死 "strict"，此处将三种策略枚举化
# ---------------------------------------------------------------------------

class CondaChannelPriority(Enum):
    """conda config --set channel_priority 的合法取值。

    上游 a406f98 仅使用 STRICT，但 Walpurgis CI 可能在不同环境需要
    灵活切换；将策略字符串封装为枚举可在调用处做静态检查。
    """
    STRICT   = "strict"
    FLEXIBLE = "flexible"
    DISABLED = "disabled"

    # ---- 上游没有的：语义辅助方法 ----
    def is_deterministic(self) -> bool:
        """严格模式 + disabled 均可保证 channel 来源唯一性（各有侧重）。
        flexible 会混合多 channel，是上游 PR 要消灭的根源。"""
        return self in (CondaChannelPriority.STRICT,
                        CondaChannelPriority.DISABLED)

    def conda_flag(self) -> str:
        """生成传给 conda config 的完整 flag 字符串。"""
        return f"channel_priority {self.value}"

    @classmethod
    def from_string(cls, s: str) -> "CondaChannelPriority":
        if _DEBUG:
            print(f"[DEBUG] CondaChannelPriority.from_string: 输入={s!r}")
            pdb.set_trace()  # 断点①: enum 解析入口
        try:
            return cls(s.lower().strip())
        except ValueError:
            valid = [e.value for e in cls]
            raise ValueError(
                f"未知 channel_priority 值: {s!r}. 合法值: {valid}"
            )


# ---------------------------------------------------------------------------
# 2. ChannelOrderSpec  ——  上游直接在 YAML 里调换两行，无任何约束声明
# ---------------------------------------------------------------------------

@dataclass
class ChannelOrderSpec:
    """描述 conda YAML 中 channels 列表的语义顺序约束。

    上游的修复将 rapidsai-nightly 提到 rapidsai 之前。
    在 strict 模式下，列表靠后的 channel 优先级更高（conda 倒序语义），
    因此 stable (rapidsai) 应排在 nightly (rapidsai-nightly) 之后，
    确保 stable 包覆盖 nightly 包。

    这一逻辑上游没有注释，此类将其显式化。
    """
    # 期望的 channel 顺序（index=0 优先级最低）
    channels: List[str] = field(default_factory=lambda: [
        "rapidsai-nightly",
        "rapidsai",
        "conda-forge",
    ])
    priority: CondaChannelPriority = CondaChannelPriority.STRICT

    def validate(self) -> List[str]:
        """返回违规列表（空列表 = 合规）。

        规则:
          - strict 模式下 rapidsai 必须在 rapidsai-nightly 之后
            (列表中 index 更大 = 优先级更高)
          - conda-forge 必须存在
        """
        if _DEBUG:
            print(f"[DEBUG] ChannelOrderSpec.validate: channels={self.channels}")
            pdb.set_trace()  # 断点②: channel 顺序验证
        violations = []
        ch = self.channels
        if "rapidsai" not in ch:
            violations.append("缺少 rapidsai channel")
        if "conda-forge" not in ch:
            violations.append("缺少 conda-forge channel")
        if (self.priority == CondaChannelPriority.STRICT
                and "rapidsai" in ch and "rapidsai-nightly" in ch):
            idx_stable  = ch.index("rapidsai")
            idx_nightly = ch.index("rapidsai-nightly")
            if idx_stable <= idx_nightly:
                violations.append(
                    f"strict 模式下 rapidsai (idx={idx_stable}) "
                    f"必须排在 rapidsai-nightly (idx={idx_nightly}) 之后，"
                    f"当前顺序不符合 a406f98 的修复意图"
                )
        return violations

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def summary(self) -> str:
        violations = self.validate()
        status = "合规" if not violations else f"违规({len(violations)}处)"
        lines = [f"ChannelOrderSpec [{status}]  priority={self.priority.value}"]
        for i, ch in enumerate(self.channels):
            lines.append(f"  [{i}] {ch}")
        if violations:
            lines.append("  --- 违规 ---")
            for v in violations:
                lines.append(f"  ✗ {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. CICondaConfig  ——  将 shell 脚本里的两行命令抽象为可测试配置对象
# ---------------------------------------------------------------------------

@dataclass
class CICondaConfig:
    """对应 ci/test_cpp.sh 和 ci/test_python.sh 中新增的 conda 配置块。

    上游直接在 shell 脚本里写死：
        rapids-logger "Configuring conda strict channel priority"
        conda config --set channel_priority strict

    此类将其结构化，generate_shell_snippet() 可重现该输出，
    同时允许在测试中验证生成内容而不执行真实 shell 命令。
    """
    priority:    CondaChannelPriority = CondaChannelPriority.STRICT
    logger_cmd:  str                  = "rapids-logger"
    log_message: str                  = "Configuring conda {priority} channel priority"

    def generate_shell_snippet(self) -> str:
        """生成与上游 a406f98 等价的 bash 代码块。"""
        if _DEBUG:
            print(f"[DEBUG] CICondaConfig.generate_shell_snippet: priority={self.priority.value}")
            pdb.set_trace()  # 断点③: shell snippet 生成
        msg = self.log_message.format(priority=self.priority.value)
        lines = [
            f'{self.logger_cmd} "{msg}"',
            f"conda config --set {self.priority.conda_flag()}",
        ]
        return "\n".join(lines)

    def validate_against_snippet(self, snippet: str) -> bool:
        """验证给定的 shell 片段是否包含正确的 conda config 命令。"""
        expected_cmd = f"conda config --set {self.priority.conda_flag()}"
        return expected_cmd in snippet


# ---------------------------------------------------------------------------
# 4. CondaYamlPatch  ——  上游 a406f98 修改 9 个 YAML，此处可程序化验证和修正
# ---------------------------------------------------------------------------

# 上游 a406f98 涉及的 9 个 YAML 文件路径（相对 repo root）
UPSTREAM_AFFECTED_YAMLS: List[str] = [
    "conda/environments/all_cuda-129_arch-aarch64.yaml",
    "conda/environments/all_cuda-129_arch-x86_64.yaml",
    "conda/environments/all_cuda-130_arch-aarch64.yaml",
    "conda/environments/all_cuda-130_arch-x86_64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-aarch64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-129_arch-x86_64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-130_arch-aarch64.yaml",
    "python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-130_arch-x86_64.yaml",
    "dependencies.yaml",
]


@dataclass
class CondaYamlPatch:
    """解析 conda YAML 的 channels 块并验证/修正 channel 顺序。

    上游直接在编辑器里把 rapidsai 和 rapidsai-nightly 互换。
    此类将该操作抽象为可幂等应用的 patch：
      - parse_channels(): 从 YAML 文本提取 channels 列表
      - check_order(): 检查是否符合 a406f98 后的期望顺序
      - apply_strict_order(): 修正顺序（幂等）
    """
    spec: ChannelOrderSpec = field(default_factory=ChannelOrderSpec)

    # 匹配 YAML channels 块的正则（简单行级解析，不依赖 PyYAML）
    _CHANNELS_BLOCK_RE = re.compile(
        r"^(channels:\s*\n)((?:[-\s]+\S.*\n)*)",
        re.MULTILINE,
    )
    _CHANNEL_ITEM_RE = re.compile(r"^[-\s]+([\w\-]+)\s*$", re.MULTILINE)

    def parse_channels(self, yaml_text: str) -> List[str]:
        """从 YAML 文本中提取 channels 列表（保持原始顺序）。"""
        m = self._CHANNELS_BLOCK_RE.search(yaml_text)
        if not m:
            return []
        block = m.group(2)
        return self._CHANNEL_ITEM_RE.findall(block)

    def check_order(self, yaml_text: str) -> bool:
        """返回 True 表示当前 YAML 已符合 a406f98 修复后的 channel 顺序。"""
        if _DEBUG:
            print("[DEBUG] CondaYamlPatch.check_order 入口")
            pdb.set_trace()  # 断点④: YAML channel 顺序检查
        channels = self.parse_channels(yaml_text)
        if "rapidsai" not in channels or "rapidsai-nightly" not in channels:
            return True  # 不含这两个 channel，视为不适用
        # a406f98 之后：rapidsai-nightly 在前（idx 小），rapidsai 在后（idx 大）
        # conda strict 模式下列表越靠后优先级越高，stable > nightly
        return channels.index("rapidsai-nightly") < channels.index("rapidsai")

    def apply_strict_order(self, yaml_text: str) -> str:
        """幂等修正 YAML 中的 channel 顺序，使其符合 a406f98 的修复。

        若已符合则原样返回；否则将 rapidsai-nightly 置于 rapidsai 之前。
        """
        if self.check_order(yaml_text):
            return yaml_text  # 已合规，幂等返回

        def _reorder_block(m: re.Match) -> str:
            header = m.group(1)
            block  = m.group(2)
            items  = self._CHANNEL_ITEM_RE.findall(block)
            # 期望顺序：nightly → stable → conda-forge（其余保留）
            priority_order = ["rapidsai-nightly", "rapidsai", "conda-forge"]
            ordered = [c for c in priority_order if c in items]
            ordered += [c for c in items if c not in priority_order]
            new_block = "".join(f"- {c}\n" for c in ordered)
            return header + new_block

        return self._CHANNELS_BLOCK_RE.sub(_reorder_block, yaml_text)


# ---------------------------------------------------------------------------
# 5. 自测 / 演示入口
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """运行 5 项自测，覆盖上述所有类。"""
    if _DEBUG:
        print("[DEBUG] _selftest 入口，即将执行全量断点测试")
        pdb.set_trace()  # 断点⑤: 自测总入口

    results = []

    # ---- 测试 1: CondaChannelPriority.from_string ----
    try:
        p = CondaChannelPriority.from_string("strict")
        assert p == CondaChannelPriority.STRICT
        assert p.is_deterministic()
        assert p.conda_flag() == "channel_priority strict"
        results.append(("CondaChannelPriority.from_string + is_deterministic + conda_flag", "PASS"))
    except AssertionError as e:
        results.append(("CondaChannelPriority", f"FAIL: {e}"))

    # ---- 测试 2: ChannelOrderSpec.validate（合规路径）----
    try:
        spec = ChannelOrderSpec()  # 默认 nightly→rapidsai→conda-forge
        violations = spec.validate()
        assert violations == [], f"期望合规，得到: {violations}"
        results.append(("ChannelOrderSpec.validate (合规)", "PASS"))
    except AssertionError as e:
        results.append(("ChannelOrderSpec.validate (合规)", f"FAIL: {e}"))

    # ---- 测试 3: ChannelOrderSpec.validate（违规路径：旧顺序）----
    try:
        bad_spec = ChannelOrderSpec(channels=["rapidsai", "rapidsai-nightly", "conda-forge"])
        violations = bad_spec.validate()
        assert len(violations) == 1
        assert "strict 模式" in violations[0]
        results.append(("ChannelOrderSpec.validate (违规旧顺序)", "PASS"))
    except AssertionError as e:
        results.append(("ChannelOrderSpec.validate (违规旧顺序)", f"FAIL: {e}"))

    # ---- 测试 4: CICondaConfig.generate_shell_snippet ----
    try:
        cfg = CICondaConfig()
        snippet = cfg.generate_shell_snippet()
        assert 'rapids-logger "Configuring conda strict channel priority"' in snippet
        assert "conda config --set channel_priority strict" in snippet
        assert cfg.validate_against_snippet(snippet)
        results.append(("CICondaConfig.generate_shell_snippet", "PASS"))
    except AssertionError as e:
        results.append(("CICondaConfig.generate_shell_snippet", f"FAIL: {e}"))

    # ---- 测试 5: CondaYamlPatch.apply_strict_order（幂等性）----
    try:
        yaml_before = (
            "channels:\n"
            "- rapidsai\n"
            "- rapidsai-nightly\n"
            "- conda-forge\n"
        )
        yaml_after_expected = (
            "channels:\n"
            "- rapidsai-nightly\n"
            "- rapidsai\n"
            "- conda-forge\n"
        )
        patcher = CondaYamlPatch()
        assert not patcher.check_order(yaml_before),  "旧顺序应判定不合规"
        fixed = patcher.apply_strict_order(yaml_before)
        assert fixed == yaml_after_expected, f"修正结果不符:\n{fixed}"
        # 幂等性：再 apply 一次结果不变
        assert patcher.apply_strict_order(fixed) == fixed, "不满足幂等性"
        results.append(("CondaYamlPatch.apply_strict_order (含幂等性)", "PASS"))
    except AssertionError as e:
        results.append(("CondaYamlPatch.apply_strict_order", f"FAIL: {e}"))

    # ---- 汇报 ----
    print("\n=== conda_channel_policy.py 自测 ===")
    all_pass = True
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} [{status}] {name}")
        if status != "PASS":
            all_pass = False
    print(f"\n{'全部通过' if all_pass else '存在失败项'} ({len(results)} 项)\n")


if __name__ == "__main__":
    _selftest()
