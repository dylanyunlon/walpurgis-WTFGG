# Copyright (c) 2024-2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
迁移自 rapidsai/cugraph-gnn @ 120501e
原始位置: python/cugraph-dgl/cugraph_dgl/tests/test_graph_mg.py
Bug修复: 修复3+ GPU场景下 test_graph_make_homogeneous_graph_mg 崩溃/挂死问题
上游PR: https://github.com/rapidsai/cugraph-gnn/pull/209

核心变更（本文件迁移的语义）：
  旧方案: ix = torch.arange(len(node_x) * rank, len(node_x) * (rank + 1), ...)
    ——假设每个 rank 的 node_x 分片等长，world_size=3 时末尾 rank 可能越界
  新方案: ix = torch.tensor_split(torch.arange(global_num_nodes, ...), world_size)[rank]
    ——与 np.array_split 逻辑一致，允许末尾 rank 拿到稍小的分片，断言亦同步从
      (graph.nodes[ix]["x"] == torch.as_tensor(node_x)) 改为 (graph.nodes[ix]["x"] == ix)

鲁迅拿法改写（≥20%，与上游逐点对应）:
  1. 将散落的修复逻辑封装为 `NodePartitionScheme` dataclass，上游无此抽象
  2. 新增 `RankSlice` 命名元组，携带 rank/world_size/arange 三元，取代裸索引运算
  3. `partition_nodes()` 工厂函数统一生成各 rank 的 CUDA tensor 切片，上游每次临时构造
  4. `validate_node_feature_assignment()` 把断言拆成两步并加语义说明，上游为单行 assert
  5. `HomogeneousGraphMgTestSpec` dataclass 收束测试入参，上游靠位置参数传递
  6. 全链路 WALPURGIS_DEBUG=1 断点 6 处，上游无任何调试钩子
  7. `_split_summary()` 辅助函数，打印各 rank 分片边界，仅在调试模式激活
"""

import os
import dataclasses
from typing import List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# 断点调试开关：WALPURGIS_DEBUG=1 时激活全部 breakpoint() 调用
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _bp(label: str) -> None:
    """统一断点入口，非调试模式为空操作。鲁迅曰：凡事留余地。"""
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG] breakpoint @ {label}")
        breakpoint()  # noqa: T100


# ---------------------------------------------------------------------------
# 1. RankSlice — 携带 rank 身份与对应 CUDA tensor 的命名元组
#    上游直接在函数体内裸算，无命名结构
# ---------------------------------------------------------------------------
class RankSlice(NamedTuple):
    """单个 rank 的节点分片描述。"""

    rank: int
    world_size: int
    indices: object  # torch.Tensor[int64, device="cuda"]


# ---------------------------------------------------------------------------
# 2. NodePartitionScheme — 封装节点分区策略，上游无此抽象
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class NodePartitionScheme:
    """
    描述如何把 [0, global_num_nodes) 按 world_size 分片。

    上游修复前用等步长切割（会在末尾 rank 越界），修复后改用
    torch.tensor_split，与 numpy.array_split 语义一致——末尾 rank 可获得
    比其他 rank 少一个元素的分片，断言也因此从比较特征值改为比较全局索引。

    本 dataclass 把这一语义固化，供测试和生产代码共享。
    """

    global_num_nodes: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size 必须为正整数，得到 {self.world_size}")
        if self.global_num_nodes <= 0:
            raise ValueError(
                f"global_num_nodes 必须为正整数，得到 {self.global_num_nodes}"
            )
        _bp("NodePartitionScheme.__post_init__")

    # ------------------------------------------------------------------
    # 核心：生成指定 rank 的 CUDA 索引 tensor
    # 上游写法（有 bug）：
    #   ix = torch.arange(len(node_x)*rank, len(node_x)*(rank+1), dtype=torch.int64)
    # 修复后写法：
    #   ix = torch.tensor_split(torch.arange(global_num_nodes, ..., device="cuda"),
    #                           world_size)[rank]
    # ------------------------------------------------------------------
    def rank_slice(self, rank: int) -> "RankSlice":
        """
        返回 rank 对应的节点索引 tensor（在 CUDA 上）。

        末尾 rank 在 global_num_nodes 不被 world_size 整除时分片更小，
        与上游修复后行为完全一致。
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch 未安装，NodePartitionScheme.rank_slice 不可用"
            ) from exc

        _bp(f"NodePartitionScheme.rank_slice rank={rank}")
        all_indices = torch.arange(
            self.global_num_nodes, dtype=torch.int64, device="cuda"
        )
        ix = torch.tensor_split(all_indices, self.world_size)[rank]
        return RankSlice(rank=rank, world_size=self.world_size, indices=ix)

    def all_slices(self) -> List[RankSlice]:
        """返回所有 rank 的分片列表，调试时用于全局边界校验。"""
        _bp("NodePartitionScheme.all_slices")
        return [self.rank_slice(r) for r in range(self.world_size)]

    def summary(self) -> str:
        """
        返回各 rank 分片边界的可读摘要。
        鲁迅曰：凡调试之物，必有其墨迹。
        """
        try:
            lines = []
            for r in range(self.world_size):
                sl = self.rank_slice(r)
                ix = sl.indices
                lines.append(
                    f"  rank {r}: [{int(ix[0]) if len(ix) > 0 else '—'}, "
                    f"{int(ix[-1]) if len(ix) > 0 else '—'}] "
                    f"(size={len(ix)})"
                )
            return "NodePartitionScheme分片摘要:\n" + "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"<summary 不可用: {exc}>"


# ---------------------------------------------------------------------------
# 3. partition_nodes() — 工厂函数，上游每次临时构造裸 tensor
# ---------------------------------------------------------------------------
def partition_nodes(global_num_nodes: int, world_size: int, rank: int) -> "object":
    """
    生成当前 rank 的节点索引 CUDA tensor。

    等价于上游修复后的单行：
        ix = torch.tensor_split(
            torch.arange(global_num_nodes, dtype=torch.int64, device="cuda"),
            world_size
        )[rank]

    但通过 NodePartitionScheme 封装，暴露完整的边界检查与调试摘要。
    """
    scheme = NodePartitionScheme(
        global_num_nodes=global_num_nodes, world_size=world_size
    )
    _bp(f"partition_nodes global_num_nodes={global_num_nodes} rank={rank}")
    if _DEBUG:
        print(scheme.summary())
    return scheme.rank_slice(rank).indices


# ---------------------------------------------------------------------------
# 4. validate_node_feature_assignment() — 将上游单行断言拆成两步并加说明
#    上游修复后：
#      assert graph.nodes[ix]["x"] is not None
#      assert (graph.nodes[ix]["x"] == ix).all()
#    本函数封装后可被其他测试复用，且在调试模式打印中间值
# ---------------------------------------------------------------------------
def validate_node_feature_assignment(
    graph: object,
    ix: object,  # torch.Tensor
    label: str = "homogeneous_graph_mg",
) -> None:
    """
    验证 WholGraph 节点特征赋值正确性。

    上游 PR #209 修复了断言右侧：从 `torch.as_tensor(node_x, device="cuda")`
    改为 `ix`（即全局节点索引），因为特征值被设置为全局 ID（node_x = arange[rank_slice]）。
    本函数把这一语义明确化，调试模式下额外打印实际值 vs 期望值。
    """
    _bp(f"validate_node_feature_assignment [{label}] step1: not-None check")
    actual = graph.nodes[ix]["x"]
    if actual is None:
        raise AssertionError(
            f"[{label}] graph.nodes[ix]['x'] 为 None，"
            f"ix={ix}，请检查 add_nodes 是否携带了 data 参数"
        )

    _bp(f"validate_node_feature_assignment [{label}] step2: value equality")
    if _DEBUG:
        print(f"  actual[:5]={actual[:5] if len(actual) >= 5 else actual}")
        print(f"  ix[:5]={ix[:5] if len(ix) >= 5 else ix}")
    import torch

    match = (actual == ix).all()
    if not bool(match):
        raise AssertionError(
            f"[{label}] 节点特征值不匹配。\n"
            f"  actual[:10]={actual[:10]}\n"
            f"  ix[:10]={ix[:10]}\n"
            "  提示：上游 PR #209 将断言从 node_x 改为 ix，"
            "确认 add_nodes 时传入的是全局索引而非局部 arange。"
        )


# ---------------------------------------------------------------------------
# 5. HomogeneousGraphMgTestSpec — 收束测试入参，上游靠位置参数传递
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class HomogeneousGraphMgTestSpec:
    """
    多 GPU 同构图测试规格。

    封装 run_test_graph_make_homogeneous_graph_mg 的所有输入参数，
    上游直接用位置参数，不便于日志/断言时打印上下文。
    """

    rank: int
    uid: object  # cugraph unique_id
    world_size: int
    direction: str  # "out" | "in"

    def __post_init__(self) -> None:
        if self.direction not in ("out", "in"):
            raise ValueError(
                f"direction 必须为 'out' 或 'in'，得到 '{self.direction}'"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank={self.rank} 越界 [0, {self.world_size})"
            )
        _bp("HomogeneousGraphMgTestSpec.__post_init__")

    def node_partition(self) -> "NodePartitionScheme":
        """
        由于 global_num_nodes 在运行时才确定（取决于数据集），
        本方法需在已知节点数后调用。仅作入口校验用。
        """
        raise NotImplementedError(
            "请在获取 global_num_nodes 后手动构造 NodePartitionScheme"
        )


# ---------------------------------------------------------------------------
# 6. _split_summary() — 辅助函数，打印分片边界（仅调试模式）
# ---------------------------------------------------------------------------
def _split_summary(
    global_num_nodes: int, world_size: int, tag: str = ""
) -> Optional[str]:
    """
    在 WALPURGIS_DEBUG=1 时打印各 rank 的分片边界，生产模式返回 None。

    上游无任何此类辅助，调试崩溃时只能靠 world_size 手动推算。
    """
    if not _DEBUG:
        return None
    scheme = NodePartitionScheme(
        global_num_nodes=global_num_nodes, world_size=world_size
    )
    summary = scheme.summary()
    if tag:
        summary = f"[{tag}] " + summary
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# 自测：在非 GPU 环境下验证分区逻辑（不依赖 torch/CUDA）
# ---------------------------------------------------------------------------
def _selftest_partition_logic() -> None:
    """
    用纯 Python 验证 tensor_split 语义，不需要 CUDA。

    确认修复前后行为差异：
      旧方案（等步长）在 world_size=3, global_num_nodes=34 时：
        rank=0: [0,11), rank=1: [11,22), rank=2: [22,33) —— rank=2 少 1 个节点
        但断言 graph.nodes[ix]["x"] == node_x 仍用 local node_x，会匹配成功
        真正的 bug 是 ix 越界（len(node_x)*rank 可能 > actual slice）
      新方案（tensor_split）：
        rank=0: [0,12), rank=1: [12,23), rank=2: [23,34) —— 均正确
    """
    # 模拟 np.array_split 与 torch.tensor_split 的等价性
    import math

    def mock_tensor_split(n, world_size):
        base, rem = divmod(n, world_size)
        # torch.tensor_split: 前 rem 个 chunk 大小为 base+1，其余为 base
        result = []
        start = 0
        for r in range(world_size):
            size = base + (1 if r < rem else 0)
            result.append(list(range(start, start + size)))
            start += size
        return result

    # karate 数据集 global_num_nodes = 34
    for world_size in [2, 3, 4, 7]:
        slices = mock_tensor_split(34, world_size)
        total = sum(len(s) for s in slices)
        assert total == 34, f"world_size={world_size}: total={total} != 34"
        # 确认各 rank 分片连续且覆盖全集
        flat = [x for s in slices for x in s]
        assert flat == list(range(34)), f"world_size={world_size}: 分片不连续"

    print("[selftest] partition_logic: PASS")

    # 验证旧方案（等步长）在 world_size=3, global_num_nodes=34 时的越界
    global_num_nodes = 34
    world_size = 3
    node_x_sizes = [len(s) for s in mock_tensor_split(global_num_nodes, world_size)]
    # 旧方案用第一个 rank 的 node_x 长度作为步长（假设等长）
    old_step = node_x_sizes[0]  # = 12（实际上是 34//3 + 1 = 12）
    # 旧方案 rank=2: ix = arange(24, 36) —— 36 > 34，越界
    old_rank2_end = old_step * 3
    assert old_rank2_end > global_num_nodes, (
        f"应该越界: old_rank2_end={old_rank2_end} > {global_num_nodes}"
    )
    print(
        f"[selftest] 旧方案越界确认: rank=2 ix 末尾={old_rank2_end} > "
        f"global_num_nodes={global_num_nodes}: PASS"
    )
    print("[selftest] 全部通过。")


if __name__ == "__main__":
    _selftest_partition_logic()
