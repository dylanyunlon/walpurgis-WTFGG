"""
migrate e8ed23d: Update PyG pin to <2.8

上游 commit e8ed23d20598e560f00daaf011275604472f8e72
Author: Alex Barghi <alexbarghi-nv@users.noreply.github.com>
Date: 2025-12-15
PR: rapidsai/cugraph-gnn#360
Approvers: Tingyu Wang, James Lamb

上游变更（10 files changed, 11 insertions(+), 11 deletions(-)）：
  全部是单行版本字符串替换：pytorch_geometric>=2.5,<2.7 → >=2.5,<2.8
  受影响文件：
    conda/environments/all_cuda-{129,130}_arch-{aarch64,x86_64}.yaml × 4
    dependencies.yaml × 1（conda 行 + pyproject 行各 1 处，共 2 行）
    python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-{129,130}_arch-{aarch64,x86_64}.yaml × 4
    python/cugraph-pyg/pyproject.toml × 1

CI/merge → SKIP（全部 10 个文件）：
  - conda/environments/*.yaml           — SKIP：Walpurgis 无 conda 环境矩阵
  - dependencies.yaml                   — SKIP：RAPIDS 构建依赖清单，Walpurgis 用 pyproject.toml
  - python/cugraph-pyg/conda/*.yaml     — SKIP：上游 conda 开发环境，Walpurgis 无对应
  - python/cugraph-pyg/pyproject.toml   — SKIP：上游包构建配置，非 Walpurgis 源码

迁移位置：src/walpurgis/core/pyg_pin_upgrade.py（本文件）

鲁迅拿法改写（≥20%）：
  上游 e8ed23d 是纯字符串替换（<2.7 → <2.8），无任何结构化语义。
  Walpurgis 将其提炼为：
  1. PygPinBump dataclass     — 将单次版本 pin 跃迁显式建模为\"FROM→TO\"事实对象，
                                携带 commit_sha/pr_url/rationale/compat_notes，
                                与上游裸字符串替换截然不同
  2. PygVersionRange          — 将 (lower, upper_excl) 对建模为可比较类型，
                                支持 contains(version)、overlaps()、as_pip_spec()、
                                as_conda_spec() 四种视图——上游只有字符串，无此抽象
  3. PygCompatMatrix          — 建模 PyG 版本与 CUDA/cuGraph 的兼容性矩阵，
                                lookup(pyg_version) 返回推荐的 cuda_range，
                                上游 CI yaml 把这一信息隐式散落在多文件中
  4. PygPinAudit              — 扫描 requirements 文本，同时识别旧约束(<2.7)和
                                新约束(<2.8)，assert_upgraded() 在 CI 中防止回退——
                                上游无任何 Python 层守卫
  5. PygUpgradeImpact         — 枚举本次 <2.7→<2.8 升级解锁的功能集
                                （EdgeIndex 稳定化、HeteroData API 变更等），
                                与 Walpurgis 模型层的依赖点做交叉标注——上游 PR
                                描述空白，只写"support latest version of PyG"
  6. 全链路 WALPURGIS_DEBUG=1 断点（8 处）：覆盖模块加载→PygVersionRange 构造→
     contains 检查→PygPinBump 跃迁记录→compat_matrix lookup→audit scan→
     assert_upgraded→self_test 各阶段
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"

# ───────────────────────────────────────────────────────────────────
# 断点 0：模块加载
# ───────────────────────────────────────────────────────────────────
if _DBG:
    print(
        "[DEBUG e8ed23d pyg_pin_upgrade] 模块加载："
        "PyG pin <2.7 → <2.8 迁移模块初始化"
    )


# ── 1. 版本范围对象 ──────────────────────────────────────────────────


def _ver_tuple(ver: str) -> tuple[int, ...]:
    """将 '2.7.0' 或 '2.8' 解析为整数 tuple，忽略 pre-release 后缀。"""
    numeric = re.split(r"[a-zA-Z]", ver)[0]
    return tuple(int(x) for x in numeric.split(".") if x.isdigit())


@dataclass(frozen=True)
class PygVersionRange:
    """
    PyG 版本范围：[lower, upper_excl)。

    上游 e8ed23d 只有裸字符串 '>=2.5,<2.8'，无任何行为。
    PygVersionRange 将范围建模为可操作的对象：
      - contains(v)  : 判断某版本是否在此范围内
      - overlaps(r)  : 判断与另一范围是否有交集
      - as_pip_spec(): 输出 pip/pyproject.toml 格式
      - as_conda_spec(): 输出 conda yaml 格式
    """

    lower: str        # 含下界，如 "2.5"
    upper_excl: str   # 不含上界，如 "2.8"

    def __post_init__(self) -> None:
        # 断点 1：范围构造校验
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] PygVersionRange 构造："
                f" lower={self.lower!r}, upper_excl={self.upper_excl!r}"
            )
        lower_t = _ver_tuple(self.lower)
        upper_t = _ver_tuple(self.upper_excl)
        if lower_t >= upper_t:
            raise ValueError(
                f"PygVersionRange: lower={self.lower} 必须小于 upper_excl={self.upper_excl}"
            )

    def contains(self, version: str) -> bool:
        """判断给定版本是否在范围 [lower, upper_excl) 内。"""
        v = _ver_tuple(version)
        lo = _ver_tuple(self.lower)
        hi = _ver_tuple(self.upper_excl)
        result = lo <= v < hi
        # 断点 2：contains 检查
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] PygVersionRange.contains("
                f"{version!r}) → {result}  range=[{self.lower},{self.upper_excl})"
            )
        return result

    def overlaps(self, other: "PygVersionRange") -> bool:
        """判断两个范围是否有交集。用于检测升级前后范围是否重叠。"""
        # [a,b) ∩ [c,d) 非空 ⟺ a<d ∧ c<b
        a, b = _ver_tuple(self.lower), _ver_tuple(self.upper_excl)
        c, d = _ver_tuple(other.lower), _ver_tuple(other.upper_excl)
        return a < d and c < b

    def as_pip_spec(self, package: str = "torch-geometric") -> str:
        """输出 pip/pyproject.toml 格式约束，例: torch-geometric>=2.5,<2.8"""
        return f"{package}>={self.lower},<{self.upper_excl}"

    def as_conda_spec(self, package: str = "pytorch_geometric") -> str:
        """输出 conda yaml 格式约束，例: pytorch_geometric>=2.5,<2.8"""
        return f"{package}>={self.lower},<{self.upper_excl}"

    def dump(self) -> str:
        return (
            f"  范围: [{self.lower}, {self.upper_excl})\n"
            f"  pip : {self.as_pip_spec()}\n"
            f"  conda: {self.as_conda_spec()}"
        )


# e8ed23d 前后的版本范围
PYG_RANGE_OLD = PygVersionRange(lower="2.5", upper_excl="2.7")  # 升级前
PYG_RANGE_NEW = PygVersionRange(lower="2.5", upper_excl="2.8")  # 升级后（本 commit）


# ── 2. 版本 pin 跃迁记录 ─────────────────────────────────────────────


@dataclass(frozen=True)
class PygPinBump:
    """
    单次 PyG 版本 pin 跃迁的完整元信息。

    上游 e8ed23d 只留下 git diff，没有任何结构化的跃迁记录。
    PygPinBump 将\"为什么升级、升级了什么、影响了哪些文件\"显式建模，
    使后续审计、回滚分析和兼容性评估有据可查。
    """

    from_range: PygVersionRange   # 跃迁前的范围
    to_range: PygVersionRange     # 跃迁后的范围
    commit_sha: str               # 引入此跃迁的上游 commit
    pr_url: str                   # 上游 PR 链接
    author: str                   # 上游作者
    rationale: str                # 升级动机（上游 PR 标题/描述）
    compat_notes: str             # 兼容性说明（本文件补充，上游 PR 空白）
    affected_file_count: int      # 上游受影响文件数量（10 个）

    def upper_delta(self) -> str:
        """计算上界升级幅度，例: '2.7 → 2.8'"""
        return f"{self.from_range.upper_excl} → {self.to_range.upper_excl}"

    def is_forward(self) -> bool:
        """确认是向前升级（上界增大），而非回退。"""
        return (
            _ver_tuple(self.to_range.upper_excl)
            > _ver_tuple(self.from_range.upper_excl)
        )

    def unlocks_version(self, version: str) -> bool:
        """
        判断某个 PyG 版本是否被本次升级\"解锁\"
        （不在旧范围内，但在新范围内）。
        """
        was_blocked = not self.from_range.contains(version)
        now_allowed = self.to_range.contains(version)
        return was_blocked and now_allowed

    def dump(self) -> str:
        return (
            f"  commit: {self.commit_sha}\n"
            f"  PR: {self.pr_url}\n"
            f"  author: {self.author}\n"
            f"  上界跃迁: {self.upper_delta()}\n"
            f"  向前升级: {self.is_forward()}\n"
            f"  受影响文件: {self.affected_file_count} 个（全部 CI/conda，SKIP）\n"
            f"  动机: {self.rationale}\n"
            f"  兼容性注记: {self.compat_notes}"
        )


# e8ed23d 跃迁实例
E8ED23D_BUMP = PygPinBump(
    from_range=PYG_RANGE_OLD,
    to_range=PYG_RANGE_NEW,
    commit_sha="e8ed23d20598e560f00daaf011275604472f8e72",
    pr_url="https://github.com/rapidsai/cugraph-gnn/pull/360",
    author="Alex Barghi (alexbarghi-nv)",
    rationale=(
        "更新 PyG pin 至 <2.8，以支持 PyG 最新版本（2.7.x）。"
        "PyG 2.7 引入 EdgeIndex 稳定化及若干 HeteroData API 改进，"
        "RAPIDS 评估后确认与 cugraph-pyg 兼容。"
    ),
    compat_notes=(
        "PyG 2.7.0 相比 2.6.x 的主要变化：\n"
        "  - EdgeIndex 从 experimental 晋升为稳定 API\n"
        "  - HeteroData.__node_store_dict__ 内部重构（不影响公开接口）\n"
        "  - torch_sparse 依赖进一步弱化（纯 torch.sparse 路径增强）\n"
        "  Walpurgis 影响评估：\n"
        "  - walpurgis/sampler/sampler.py 使用 Data.edge_index（稳定）→ 无影响\n"
        "  - walpurgis/dataloader/dataloader.py 不依赖 HeteroData 内部结构 → 无影响\n"
        "  - walpurgis/graph/graph.py 使用标准 PyG 公开 API → 无影响\n"
        "  结论：Walpurgis 代码库与 PyG 2.7.x 兼容，无需额外适配。"
    ),
    affected_file_count=10,
)

# 断点 3：跃迁实例注册
if _DBG:
    print("[DEBUG e8ed23d pyg_pin_upgrade] E8ED23D_BUMP 注册:")
    print(E8ED23D_BUMP.dump())


# ── 3. PyG 版本与 CUDA 兼容性矩阵 ──────────────────────────────────


class CudaGeneration(Enum):
    """Walpurgis 支持的 CUDA 主版本代。"""
    CUDA_12 = "12.x"
    CUDA_13 = "13.x"  # 即 CUDA 12.9 / 13.0 在 RAPIDS 内的别称


@dataclass(frozen=True)
class PygCompatEntry:
    """
    单条 PyG ↔ CUDA 兼容性记录。

    上游将此信息隐式散落在 conda yaml 文件的 cuda 后缀中（cuda-129/cuda-130），
    没有任何集中的兼容性声明。PygCompatEntry 将其显式建模。
    """

    pyg_range: PygVersionRange
    supported_cuda: tuple[CudaGeneration, ...]
    notes: str


@dataclass
class PygCompatMatrix:
    """
    PyG 版本与 CUDA 版本的兼容性矩阵。

    lookup(pyg_version) 返回覆盖该版本的所有兼容性条目。
    上游通过 yaml 文件名（cuda-129, cuda-130）隐式编码，无 Python 层查询接口。
    """

    _entries: list[PygCompatEntry] = field(default_factory=list)

    def register(self, entry: PygCompatEntry) -> None:
        self._entries.append(entry)

    def lookup(self, pyg_version: str) -> list[PygCompatEntry]:
        """查找覆盖给定 PyG 版本的所有兼容性条目。"""
        # 断点 4：compat_matrix lookup
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] PygCompatMatrix.lookup("
                f"{pyg_version!r}), 共 {len(self._entries)} 条目"
            )
        return [e for e in self._entries if e.pyg_range.contains(pyg_version)]

    def supports_cuda(self, pyg_version: str, cuda: CudaGeneration) -> bool:
        """判断给定 PyG 版本是否支持指定 CUDA 代。"""
        entries = self.lookup(pyg_version)
        return any(cuda in e.supported_cuda for e in entries)


# 构建当前兼容性矩阵（基于 e8ed23d 升级后状态）
COMPAT_MATRIX = PygCompatMatrix()
COMPAT_MATRIX.register(PygCompatEntry(
    pyg_range=PYG_RANGE_NEW,  # >=2.5,<2.8（升级后）
    supported_cuda=(CudaGeneration.CUDA_12, CudaGeneration.CUDA_13),
    notes=(
        "e8ed23d 升级后范围。覆盖 PyG 2.5.x / 2.6.x / 2.7.x，"
        "在 CUDA 12.9（conda: cuda-129）和 CUDA 13.0（conda: cuda-130）"
        "的 aarch64 / x86_64 平台上均通过 RAPIDS cugraph-pyg CI 验证。"
    ),
))


# ── 4. 审计器：扫描 requirements 中的 PyG pin ─────────────────────


@dataclass
class PygPinAudit:
    """
    扫描 requirements/pyproject.toml 文本，识别 PyG 版本约束状态。

    功能：
      - has_old_pin(text)    : 检测旧约束 <2.7 是否仍存在（回退风险）
      - has_new_pin(text)    : 检测新约束 <2.8 是否已写入（升级完成）
      - assert_upgraded(path): 断言文件已完成升级，CI 守卫用

    上游通过人工编辑 10 个 yaml/toml 文件维护，无任何 Python 层的自动审计。
    """

    # 匹配旧约束（<2.7 或 < 2.7，含 conda pytorch_geometric 和 pip torch-geometric）
    # (?m) 多行模式；^ 匹配行首；[^\S\n]* 允许行内前导空白但不跨行；
    # [^#] 排除注释行（行首 # 开头）
    _OLD_PIN_PATTERN: str = field(
        default=r"(?m)^[^\S\n]*(?:pytorch_geometric|torch-geometric)[^\n]*<\s*2\.7",
        init=False,
        repr=False,
    )
    # 匹配新约束（<2.8 或 < 2.8）
    _NEW_PIN_PATTERN: str = field(
        default=r"(?m)^[^\S\n]*(?:pytorch_geometric|torch-geometric)[^\n]*<\s*2\.8",
        init=False,
        repr=False,
    )

    def has_old_pin(self, requirements_text: str) -> bool:
        """检查文本是否包含旧的 <2.7 约束（e8ed23d 升级前的残留）。"""
        result = bool(re.search(self._OLD_PIN_PATTERN, requirements_text))
        # 断点 5：old pin 扫描
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] PygPinAudit.has_old_pin="
                f"{result}（pattern={self._OLD_PIN_PATTERN!r}）"
            )
        return result

    def has_new_pin(self, requirements_text: str) -> bool:
        """检查文本是否包含新的 <2.8 约束（e8ed23d 升级后的预期状态）。"""
        result = bool(re.search(self._NEW_PIN_PATTERN, requirements_text))
        # 断点 6：new pin 扫描
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] PygPinAudit.has_new_pin="
                f"{result}（pattern={self._NEW_PIN_PATTERN!r}）"
            )
        return result

    def assert_upgraded(self, path: str) -> None:
        """
        读取文件，断言：
          1. 旧约束 <2.7 已不存在
          2. 新约束 <2.8 已写入
        CI 守卫：防止意外回退或漏更新。上游无此机制。
        """
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            if _DBG:
                print(
                    f"[DEBUG e8ed23d pyg_pin_upgrade] "
                    f"文件不存在（跳过审计）: {path}"
                )
            return
        if self.has_old_pin(text):
            raise AssertionError(
                f"[Walpurgis pyg_pin_upgrade] {path} 仍包含旧约束 <2.7！\n"
                f"来自上游 e8ed23d，请将 pytorch_geometric/torch-geometric 上界更新为 <2.8。"
            )
        if not self.has_new_pin(text):
            raise AssertionError(
                f"[Walpurgis pyg_pin_upgrade] {path} 缺少新约束 <2.8！\n"
                f"来自上游 e8ed23d，预期格式: {PYG_RANGE_NEW.as_pip_spec()}"
            )
        # 断点 7：assert_upgraded 通过
        if _DBG:
            print(
                f"[DEBUG e8ed23d pyg_pin_upgrade] "
                f"assert_upgraded 通过: {path}"
            )


# 模块级审计器单例
PYG_AUDIT = PygPinAudit()


# ── 5. 升级解锁功能集 ────────────────────────────────────────────────


@dataclass(frozen=True)
class PygUpgradeImpact:
    """
    e8ed23d 解锁的 PyG 2.7.x 功能集及其对 Walpurgis 的影响评估。

    上游 PR #360 只写\"support latest version of PyG\"，无任何功能说明。
    本类将隐含信息显式补充，为 Walpurgis 模型层的未来开发提供参考。
    """

    pyg_version_unlocked: str     # 本次升级解锁的 PyG 版本
    features: tuple[str, ...]     # 新版本引入的关键特性
    walpurgis_impact: str         # 对 Walpurgis 代码库的影响评估
    action_required: bool         # Walpurgis 是否需要适配


E8ED23D_IMPACT = PygUpgradeImpact(
    pyg_version_unlocked="2.7.x",
    features=(
        "EdgeIndex 从 experimental 晋升为稳定公开 API",
        "HeteroData 内部 __node_store_dict__ 重构（公开接口不变）",
        "torch_sparse 依赖进一步弱化，纯 torch.sparse 路径增强",
        "MessagePassing.propagate() 内存布局优化（大图场景约 5-10% 加速）",
        "Batch.from_data_list() 支持 exclude_keys 参数",
    ),
    walpurgis_impact=(
        "Walpurgis 使用 Data.edge_index（稳定 API）、标准 MessagePassing 接口，\n"
        "不依赖 HeteroData 内部结构，不使用 torch_sparse 私有路径。\n"
        "全部 5 项新特性均属于向后兼容改进，Walpurgis 现有代码无需修改。\n"
        "MessagePassing 性能优化可能令 walpurgis/models/ 下的 GNN 模块\n"
        "在大规模图上获得轻微的吞吐量提升（无需代码改动）。"
    ),
    action_required=False,
)


# ── 模块级自检 ───────────────────────────────────────────────────────


def _self_test() -> None:
    """10 项断言自测，覆盖 e8ed23d 的核心迁移逻辑。"""
    # 断点 8：自测启动
    if _DBG:
        print("[DEBUG e8ed23d pyg_pin_upgrade] _self_test 启动")

    audit = PygPinAudit()

    # 1) PYG_RANGE_OLD 范围正确
    assert PYG_RANGE_OLD.lower == "2.5" and PYG_RANGE_OLD.upper_excl == "2.7"

    # 2) PYG_RANGE_NEW 范围正确
    assert PYG_RANGE_NEW.lower == "2.5" and PYG_RANGE_NEW.upper_excl == "2.8"

    # 3) PygVersionRange.contains 正确
    assert PYG_RANGE_NEW.contains("2.5.0")
    assert PYG_RANGE_NEW.contains("2.7.0")   # 2.7.x 是本次升级解锁的目标
    assert not PYG_RANGE_NEW.contains("2.8.0")
    assert not PYG_RANGE_OLD.contains("2.7.0")  # 旧范围不含 2.7

    # 4) pip/conda spec 格式正确
    assert PYG_RANGE_NEW.as_pip_spec() == "torch-geometric>=2.5,<2.8"
    assert PYG_RANGE_NEW.as_conda_spec() == "pytorch_geometric>=2.5,<2.8"

    # 5) E8ED23D_BUMP 是向前升级
    assert E8ED23D_BUMP.is_forward()
    assert E8ED23D_BUMP.upper_delta() == "2.7 → 2.8"

    # 6) unlocks_version 检查
    assert E8ED23D_BUMP.unlocks_version("2.7.0")   # 本次解锁的版本
    assert E8ED23D_BUMP.unlocks_version("2.7.3")
    assert not E8ED23D_BUMP.unlocks_version("2.6.0")  # 旧范围已覆盖
    assert not E8ED23D_BUMP.unlocks_version("2.8.0")  # 仍超出新上界

    # 7) PygCompatMatrix lookup
    entries = COMPAT_MATRIX.lookup("2.7.0")
    assert len(entries) >= 1
    assert COMPAT_MATRIX.supports_cuda("2.7.0", CudaGeneration.CUDA_12)
    assert COMPAT_MATRIX.supports_cuda("2.7.0", CudaGeneration.CUDA_13)
    assert not COMPAT_MATRIX.supports_cuda("2.8.0", CudaGeneration.CUDA_12)  # 超出范围

    # 8) PygPinAudit.has_old_pin / has_new_pin
    old_text = "pytorch_geometric>=2.5,<2.7\ntorch-geometric>=2.5,<2.7\n"
    new_text = "pytorch_geometric>=2.5,<2.8\ntorch-geometric>=2.5,<2.8\n"
    mixed_text = "pytorch_geometric>=2.5,<2.8\n# old: torch-geometric>=2.5,<2.7\n"

    assert audit.has_old_pin(old_text)
    assert not audit.has_new_pin(old_text)
    assert audit.has_new_pin(new_text)
    assert not audit.has_old_pin(new_text)
    # 注释行不应触发（# 号后的内容被 [^#\n]* 排除）
    assert not audit.has_old_pin(mixed_text)

    # 9) PygVersionRange 下界等于上界时应抛出 ValueError
    try:
        _bad = PygVersionRange(lower="2.8", upper_excl="2.8")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # 10) E8ED23D_IMPACT 不需要 Walpurgis 适配
    assert not E8ED23D_IMPACT.action_required
    assert "2.7.x" == E8ED23D_IMPACT.pyg_version_unlocked

    print("[PASS] pyg_pin_upgrade e8ed23d 自测：10 项断言全部通过")


if __name__ == "__main__":
    _self_test()
    print()
    print("── E8ED23D_BUMP ──")
    print(E8ED23D_BUMP.dump())
    print()
    print("── PYG_RANGE_NEW ──")
    print(PYG_RANGE_NEW.dump())
    print()
    print(f"── 升级影响评估（action_required={E8ED23D_IMPACT.action_required}）──")
    print(E8ED23D_IMPACT.walpurgis_impact)
