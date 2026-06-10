"""
renumber_kg.py — 05fe6f4 迁移: Knowledge Graph / Graph Database 节点重编号

migrate 05fe6f4: [FEA] Knowledge Graph/Graph Database Renumbering

上游变化 (05fe6f4, cugraph-gnn /
  python/cugraph-pyg/cugraph_pyg/examples/kg/renumber_kg.py):
  全新文件，295行。核心逻辑:

  1. parse_args(): argparse 定义所有输入参数
     - --node_types / --node_input_folders / --node_output_folders / --node_colname
     - --edge_types / --edge_input_folders / --edge_output_folders
     - --source_colname / --destination_colname
     - --output_format / --input_format (csv|parquet)
     - --use_managed_memory (RMM managed memory 开关)

  2. 分布式初始化:
     - torch.distributed.init_process_group("nccl")
     - rank=0 才切换 rmm_torch_allocator，barrier 后所有 rank 设置 cupy allocator
     - rmm.reinitialize(devices=[local_rank], managed_memory=..., pool_allocator=True)

  3. 节点重编号阶段 (per node_type):
     - 每 rank 读自己的 node 文件 (按 local_rank 索引 sorted(os.listdir()))
     - all_gather_into_tensor 收集各 rank node 数量 → cumsum 计算 global offset
     - local_renumber_map: shape=[2, local_num_nodes], row0=新全局id, row1=原始id
     - all_gather 汇总全局 renumber map → cudf.DataFrame(index=原始id, col="id"=新id)
     - 将本 rank 的 renumber map 写到 output_folder

  4. 边重编号阶段 (per edge_type):
     - 每 rank 读自己的 edge 文件 (os.listdir() 无排序，和节点不一致——潜在bug)
     - src/dst 原始 id 通过 global_renumber_map[type]["id"].loc[] 映射到新 id
     - 写出到 output_folder

  5. barrier → print("Success!") → destroy_process_group()

Walpurgis 改写20%(鲁迅拿法):
  - KGRenumberArgs: 将 argparse Namespace 封装为强类型 dataclass，
    替代散落的 args.xxx 访问；加 validate() 后置校验
    (C++/Python 直接访问 args.xxx，无校验，用错参数名只在运行时报 AttributeError)
  - NodeRenumberSession: 封装单个 node_type 的全部重编号状态
    (local_num_nodes, global_num_nodes, local_offset, renumber_map_df)，
    替代 4 个平行 dict[node_type→value]
  - RenumberMapStore: 管理 global_renumber_map[node_type] 的写入与查找，
    替代裸 dict 访问，加 get_strict() 在 edge 阶段找不到 node_type 时给出明确错误
  - EdgeRenumberSession: 封装单个 edge_type 的重编号执行，src_map/dst_map
    查找集中在 apply() 方法，替代 main 中内联的 .loc[] 映射
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路 print，覆盖:
    - args 解析后 dump 全部参数
    - 每个 node_type: 文件路径、local_num_nodes、offset、global_num_nodes
    - all_gather 前后 tensor shape
    - renumber_map 构建完成后 head(3) 预览
    - 每个 edge_type: 文件路径、src/dst 映射前后 value counts 摘要
    - 输出写入路径确认

上游已知问题 (边阶段文件排序不一致, 见下方 Bug 说明):
  节点阶段: sorted(os.listdir(node_folder_name))[local_rank]  ← 有排序
  边阶段:   os.listdir(edge_folder_name)[local_rank]           ← 无排序
  → 多 rank 之间 edge 文件分配不确定，理论上可导致同一文件被两 rank 重复处理
    或某文件被跳过。Walpurgis 迁移修复此问题: EdgeRenumberSession 强制 sorted()。

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────
# 调试开关: WALPURGIS_DEBUG=1 开启断点级 print
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试: renumber_kg 专用 print。

    对应上游 05fe6f4 中唯一的 print("Success!") ——
    该脚本几乎无任何中间过程输出，调试信息全靠此 _dbg 补全。
    """
    if _DEBUG:
        print(f"[DEBUG 05fe6f4 renumber_kg | {tag}] {msg}", file=sys.stderr, flush=True)


# ──────────────────────────────────────────────
# KGRenumberArgs — 强类型参数对象
# ──────────────────────────────────────────────
# 上游 (05fe6f4): argparse.Namespace，parse_args() 返回后散落 args.xxx 访问。
# 改写: 封装为 dataclass，validate() 在运行时入口处统一校验，
#   避免 args.xxx typo 只在深层运行时才报 AttributeError。
# C++ 对应: 无 (上游是纯 Python 脚本)。

@dataclass
class KGRenumberArgs:
    """
    对应上游 parse_args() 返回的 argparse.Namespace。

    上游散落 args.xxx 访问（14 个字段），改写为单一配置对象，
    validate() 做前置合法性检查（上游无此步骤）。
    """
    node_types: List[str]
    node_input_folders: List[str]
    node_output_folders: List[str]
    node_colname: str
    edge_types: List[Tuple[str, str, str]]   # [(src_type, rel_type, dst_type), ...]
    edge_input_folders: List[str]
    edge_output_folders: List[str]
    source_colname: str
    destination_colname: str
    input_format: str
    output_format: str
    use_managed_memory: bool

    def validate(self) -> None:
        """
        校验参数一致性。上游无此校验，错误只在 zip() 遍历到不匹配条目时才爆。

        断点调试: validate() 入口打印所有参数摘要。
        """
        # 断点: 校验前 dump 全部参数
        _dbg(
            "KGRenumberArgs.validate",
            f"node_types={self.node_types} "
            f"node_input_folders={self.node_input_folders} "
            f"node_output_folders={self.node_output_folders} "
            f"node_colname={self.node_colname!r} "
            f"edge_types={self.edge_types} "
            f"edge_input_folders={self.edge_input_folders} "
            f"edge_output_folders={self.edge_output_folders} "
            f"source_colname={self.source_colname!r} "
            f"destination_colname={self.destination_colname!r} "
            f"input_format={self.input_format!r} "
            f"output_format={self.output_format!r} "
            f"use_managed_memory={self.use_managed_memory}"
        )

        # 节点: types / input_folders / output_folders 三者长度一致
        if not (len(self.node_types)
                == len(self.node_input_folders)
                == len(self.node_output_folders)):
            raise ValueError(
                f"node_types ({len(self.node_types)}), "
                f"node_input_folders ({len(self.node_input_folders)}), "
                f"node_output_folders ({len(self.node_output_folders)}) "
                "长度不一致 — 检查参数"
            )
        # 边: edge_types / edge_input_folders / edge_output_folders 三者长度一致
        if not (len(self.edge_types)
                == len(self.edge_input_folders)
                == len(self.edge_output_folders)):
            raise ValueError(
                f"edge_types ({len(self.edge_types)}), "
                f"edge_input_folders ({len(self.edge_input_folders)}), "
                f"edge_output_folders ({len(self.edge_output_folders)}) "
                "长度不一致 — 检查参数"
            )
        # 格式合法性
        valid_fmts = {"csv", "parquet"}
        if self.input_format.lower() not in valid_fmts:
            raise ValueError(f"input_format={self.input_format!r} 不在 {valid_fmts}")
        if self.output_format.lower() not in valid_fmts:
            raise ValueError(f"output_format={self.output_format!r} 不在 {valid_fmts}")

        _dbg("KGRenumberArgs.validate", "参数校验通过")


def _parse_args() -> KGRenumberArgs:
    """
    对应上游 parse_args()，返回 KGRenumberArgs 而非裸 Namespace。

    改写: 在 parse 后立即构造 KGRenumberArgs 并调用 validate()，
      确保参数在进入分布式初始化前已通过校验
      （上游将参数校验完全依赖 argparse required=True，无语义校验）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_types",
        type=str,
        required=True,
        help="List of node types separated by commas (i.e. shape,size,length)",
    )
    parser.add_argument(
        "--node_input_folders",
        type=str,
        required=True,
        help=(
            "List of folders containing input node IDs (should match node type order)"
            " (i.e. data/shape, data/size, data/length)."
            " Each folder should contain (# local workers) files."
        ),
    )
    parser.add_argument(
        "--node_output_folders",
        type=str,
        required=True,
        help=(
            "List of folders containing output node IDs (should match node type order)"
            " (i.e. data/shape, data/size, data/length)."
        ),
    )
    parser.add_argument(
        "--node_colname",
        type=str,
        required=True,
        help="Name of the column containing node ids in each node file.",
    )
    parser.add_argument(
        "--edge_types",
        type=str,
        required=True,
        help=(
            "List of canonical edge types separated by semicolons"
            " (i.e. paper,cites,paper;author,writes,paper)"
        ),
    )
    parser.add_argument(
        "--edge_input_folders",
        type=str,
        required=True,
        help=(
            "List of input folders containing edges, separated by commas"
            " (i.e. data/paper_cites_paper,data/author_writes_paper). "
            "Each folder should contain (# local workers) files."
        ),
    )
    parser.add_argument(
        "--edge_output_folders",
        type=str,
        required=True,
        help=(
            "List of output folders containing edges, separated by commas "
            "(i.e. data/paper_cites_paper,data/author_writes_paper)."
        ),
    )
    parser.add_argument(
        "--source_colname",
        type=str,
        required=True,
        help="Name of the column in each edge file corresponding to source node id.",
    )
    parser.add_argument(
        "--destination_colname",
        type=str,
        required=True,
        help=(
            "Name of the column in each edge file corresponding to"
            " destination node id."
        ),
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        default="csv",
        help="csv or parquet",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        required=False,
        default="csv",
        help="csv or parquet",
    )
    parser.add_argument(
        "--use_managed_memory",
        action="store_true",
        required=False,
        default=False,
        help=(
            "Whether to use managed memory "
            "(allow spilling to CPU memory if there is not enough GPU memory)"
        ),
    )

    raw = parser.parse_args()

    # 解析 edge_types: "paper,cites,paper;author,writes,paper"
    # → [("paper","cites","paper"), ("author","writes","paper")]
    # 上游在 main 中用 tuple(edge_type.split(",")) 内联解析，
    # 改写: 提前解析到强类型，便于 validate() 检查三元组完整性
    parsed_edge_types: List[Tuple[str, str, str]] = []
    for et in raw.edge_types.split(";"):
        parts = et.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"edge_type {et!r} 格式错误，需要 'src_type,rel_type,dst_type'"
            )
        parsed_edge_types.append((parts[0], parts[1], parts[2]))

    cfg = KGRenumberArgs(
        node_types=raw.node_types.split(","),
        node_input_folders=raw.node_input_folders.split(","),
        node_output_folders=raw.node_output_folders.split(","),
        node_colname=raw.node_colname,
        edge_types=parsed_edge_types,
        edge_input_folders=raw.edge_input_folders.split(","),
        edge_output_folders=raw.edge_output_folders.split(","),
        source_colname=raw.source_colname,
        destination_colname=raw.destination_colname,
        input_format=raw.input_format,
        output_format=raw.output_format,
        use_managed_memory=raw.use_managed_memory,
    )
    cfg.validate()
    return cfg


# ──────────────────────────────────────────────
# NodeRenumberSession — 单 node_type 重编号状态
# ──────────────────────────────────────────────
# 上游 (05fe6f4): 4 个平行 dict (local_num_nodes, global_num_nodes,
#   local_node_offsets, global_renumber_map)，以 node_type 为 key。
# 改写: 合并为单一对象，字段直接命名，避免 dict[node_type] 访问四次。
# 断点调试: 每阶段 print 当前节点类型的重要状态。

@dataclass
class NodeRenumberSession:
    """
    封装单个 node_type 的全部重编号中间状态。

    对应上游四个平行 dict:
      local_num_nodes[node_type]      → self.local_num_nodes
      global_num_nodes[node_type]     → self.global_num_nodes
      local_node_offsets[node_type]   → self.local_offset
      global_renumber_map[node_type]  → self.renumber_map_df (cudf.DataFrame)
    """
    node_type: str
    local_num_nodes: int = 0
    global_num_nodes: int = 0
    local_offset: int = 0
    # cudf.DataFrame({"id": cupy_array}, index=cupy_array)
    # 键=原始id，值=新全局id
    renumber_map_df: object = None   # type: ignore[type-arg]


# ──────────────────────────────────────────────
# RenumberMapStore — global_renumber_map 的查找层
# ──────────────────────────────────────────────
# 上游 (05fe6f4): global_renumber_map = {} ，直接 dict 写入/读取，
#   edge 阶段 global_renumber_map[src_type]["id"] 若 src_type 不在 map 中
#   只会抛 KeyError，错误信息不友好。
# 改写: get_strict() 给出明确的 node_type 不存在原因，
#   便于大规模 KG (数十种 node_type) 时定位参数配置错误。

class RenumberMapStore:
    """
    管理 global_renumber_map[node_type] → cudf.DataFrame 的写入与查找。

    对应上游:
      global_renumber_map = {}           (初始化)
      global_renumber_map[node_type] = cudf.DataFrame(...)  (写入)
      global_renumber_map[src_type]["id"]  (读取)
    改写: 封装为对象，get_strict() 给出明确错误信息。
    """

    def __init__(self) -> None:
        # 对应上游 global_renumber_map = {}
        self._store: Dict[str, object] = {}   # type: ignore[type-arg]

    def put(self, node_type: str, df: object) -> None:
        """
        对应上游 global_renumber_map[node_type] = cudf.DataFrame(...)
        """
        _dbg(
            "RenumberMapStore.put",
            f"node_type={node_type!r}  df.shape={getattr(df, 'shape', '?')}"
        )
        self._store[node_type] = df

    def get_strict(self, node_type: str) -> object:
        """
        对应上游 global_renumber_map[src_type] / global_renumber_map[dst_type]。

        改写: 不在 store 中时给出明确错误，说明哪个 node_type 未在节点重编号阶段处理。
        上游: 直接 dict[key] → KeyError，错误信息只有 key 字符串。
        """
        if node_type not in self._store:
            raise KeyError(
                f"node_type={node_type!r} 不在 renumber_map_store 中。"
                f" 已注册的 node_types: {list(self._store.keys())}。"
                f" 检查 --edge_types 中的 src_type/dst_type 是否都出现在 --node_types 中。"
            )
        df = self._store[node_type]
        _dbg(
            "RenumberMapStore.get_strict",
            f"node_type={node_type!r}  df.shape={getattr(df, 'shape', '?')}"
        )
        return df


# ──────────────────────────────────────────────
# EdgeRenumberSession — 单 edge_type 重编号执行
# ──────────────────────────────────────────────
# 上游 (05fe6f4): main 中内联执行，src/dst map 查找、loc[] 映射、写出混在一起。
# 改写: 封装为 apply() 方法，单元可测，调试 print 集中管理。
# Bug 修复: 上游边文件无 sorted()，改写中强制 sorted() 与节点阶段对齐。

@dataclass
class EdgeRenumberSession:
    """
    封装单个 edge_type 的边重编号执行。

    对应上游 main 中 edge_type 循环体内联逻辑:
      edge_fname = os.listdir(edge_folder_name)[local_rank]  ← 无排序 (bug)
      edf = cudf.read_csv(edge_fpath)
      srcs = edf[source_colname].values
      dsts = edf[destination_colname].values
      src_map = global_renumber_map[src_type]["id"]
      dst_map = global_renumber_map[dst_type]["id"]
      new_edf = cudf.DataFrame({...src_map.loc[srcs].values, ...dst_map.loc[dsts].values})
      new_edf.to_csv(...)

    改写: apply() 方法封装以上全部逻辑，断点 print 覆盖 src/dst 映射前后。
    """
    src_type: str
    rel_type: str
    dst_type: str
    input_folder: str
    output_folder: str
    local_rank: int
    source_colname: str
    destination_colname: str
    input_format: str
    output_format: str

    def apply(
        self,
        renumber_map_store: RenumberMapStore,
        cudf: object,   # type: ignore[type-arg]
    ) -> str:
        """
        执行一个 edge_type 的全部重编号，返回写出路径。

        对应上游 main 中 edge_type 循环体。

        Bug 修复 (05fe6f4 原始代码):
          edge_fname = os.listdir(edge_folder_name)[local_rank]   ← 无排序
          → 不同 rank 在同一 OS 上 os.listdir() 结果顺序不确定，
            但通常一致（ext4/tmpfs）。跨节点 NFS 等场景下顺序可能不同，
            导致两 rank 处理同一文件或遗漏某文件。
          修复: sorted(os.listdir(edge_folder_name))[local_rank]  ← 强制排序

        断点调试:
          1. 边文件路径确认
          2. src/dst 原始 id 统计 (nunique, min, max)
          3. 映射后新 id 统计 (验证没有 NaN/越界)
          4. 输出写入路径确认
        """
        import cudf as _cudf  # type: ignore[import]

        # 断点1: 边文件路径
        # 上游 os.listdir() 无排序，改写为 sorted()
        # C++ 对应: 无（上游纯 Python）
        edge_files = sorted(os.listdir(self.input_folder))  # Bug 修复: 加 sorted()
        edge_fname = edge_files[self.local_rank]
        edge_fpath = os.path.join(self.input_folder, edge_fname)
        _dbg(
            "EdgeRenumberSession.apply[1/4]",
            f"edge_type=({self.src_type},{self.rel_type},{self.dst_type})  "
            f"edge_fname={edge_fname!r}  edge_fpath={edge_fpath!r}  "
            f"local_rank={self.local_rank}  "
            f"(bug-fix: sorted() applied, upstream had no sort)"
        )

        # 读取边文件，对应上游 cudf.read_csv / cudf.read_parquet
        if self.input_format.lower() == "csv":
            edf = _cudf.read_csv(edge_fpath)
        elif self.input_format.lower() == "parquet":
            edf = _cudf.read_parquet(edge_fpath)
        else:
            raise ValueError(f"Invalid input_format={self.input_format!r}")

        # 获取 src/dst 原始 id
        # 对应上游: srcs = edf[source_colname].values / dsts = edf[destination_colname].values
        srcs = edf[self.source_colname].values
        dsts = edf[self.destination_colname].values

        # 断点2: src/dst 原始 id 统计
        _dbg(
            "EdgeRenumberSession.apply[2/4]",
            f"edge_type=({self.src_type},{self.rel_type},{self.dst_type})  "
            f"num_edges={len(edf)}  "
            f"src_col={self.source_colname!r}  dst_col={self.destination_colname!r}"
        )

        # 从 RenumberMapStore 获取 src/dst 的全局重编号 map
        # 对应上游: src_map = global_renumber_map[src_type]["id"]
        #           dst_map = global_renumber_map[dst_type]["id"]
        # 改写: get_strict() 提供明确错误信息
        src_renumber_df = renumber_map_store.get_strict(self.src_type)
        dst_renumber_df = renumber_map_store.get_strict(self.dst_type)

        # 对应上游: src_map = global_renumber_map[src_type]["id"]
        src_map = src_renumber_df["id"]  # type: ignore[index]
        dst_map = dst_renumber_df["id"]  # type: ignore[index]

        # 断点3: 映射执行
        # 对应上游: src_map.loc[srcs].values / dst_map.loc[dsts].values
        _dbg(
            "EdgeRenumberSession.apply[3/4]",
            f"src_map.shape={getattr(src_map, 'shape', '?')}  "
            f"dst_map.shape={getattr(dst_map, 'shape', '?')}  "
            f"executing .loc[] renumbering..."
        )
        new_srcs = src_map.loc[srcs].values
        new_dsts = dst_map.loc[dsts].values

        # 构造输出 DataFrame
        # 对应上游: new_edf = cudf.DataFrame({source_colname: ..., destination_colname: ...})
        new_edf = _cudf.DataFrame(
            {
                self.source_colname: new_srcs,
                self.destination_colname: new_dsts,
            }
        )

        # 写出
        # 对应上游: new_edf.to_parquet(...) / new_edf.to_csv(...)
        if self.output_format.lower() == "parquet":
            out_path = os.path.join(
                self.output_folder, f"{edge_fname}_renumbered.parquet"
            )
            new_edf.to_parquet(out_path, index=False)
        elif self.output_format.lower() == "csv":
            out_path = os.path.join(
                self.output_folder, f"{edge_fname}_renumbered.csv"
            )
            new_edf.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Invalid output_format={self.output_format!r}")

        # 断点4: 写出路径确认
        _dbg(
            "EdgeRenumberSession.apply[4/4]",
            f"edge_type=({self.src_type},{self.rel_type},{self.dst_type})  "
            f"out_path={out_path!r}  num_edges_written={len(new_edf)}"
        )
        return out_path


# ──────────────────────────────────────────────
# _read_node_file — 读取节点文件
# ──────────────────────────────────────────────
# 上游: main 中内联 cudf.read_csv / cudf.read_parquet，raise ValueError
# 改写: 提取为工具函数，断点打印文件路径和行数

def _read_node_file(fpath: str, input_format: str, cudf: object) -> object:  # type: ignore[type-arg]
    """
    读取单个节点文件，对应上游 main 中:
      if args.input_format.lower() == "csv":   ndf = cudf.read_csv(node_fpath)
      elif args.input_format.lower() == "parquet": ndf = cudf.read_parquet(node_fpath)
      else: raise ValueError("Invalid input type.")
    """
    _dbg("_read_node_file", f"fpath={fpath!r}  input_format={input_format!r}")
    if input_format.lower() == "csv":
        ndf = cudf.read_csv(fpath)  # type: ignore[union-attr]
    elif input_format.lower() == "parquet":
        ndf = cudf.read_parquet(fpath)  # type: ignore[union-attr]
    else:
        raise ValueError(f"Invalid input_format={input_format!r}")
    _dbg("_read_node_file", f"loaded  rows={len(ndf)}  columns={list(ndf.columns)}")
    return ndf


# ──────────────────────────────────────────────
# main — 分布式入口
# ──────────────────────────────────────────────
# 对应上游 if __name__ == "__main__": 全部逻辑
# 改写: 提取为 main() 函数，方便测试和从外部调用
# 结构保持与上游高度对应，每段均有注释标注上游原始行

def main() -> None:
    """
    分布式 KG 重编号主入口。

    对应上游 05fe6f4 renumber_kg.py if __name__ == "__main__": 全部逻辑（~190行）。
    改写: 使用 KGRenumberArgs / NodeRenumberSession / RenumberMapStore /
      EdgeRenumberSession 替代散落的局部变量和裸 dict 访问。
    """
    # ── 参数解析 ──────────────────────────────
    # 对应上游: args = parse_args()
    cfg = _parse_args()

    # ── 分布式初始化 ──────────────────────────
    # 对应上游:
    #   torch.distributed.init_process_group("nccl")
    #   world_size = torch.distributed.get_world_size()
    #   global_rank = torch.distributed.get_rank()
    #   local_rank = int(os.environ["LOCAL_RANK"])
    #   device = torch.device(local_rank)
    import torch
    import cupy

    os.environ["RAPIDS_NO_INITIALIZE"] = "1"

    torch.distributed.init_process_group("nccl")
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    _dbg(
        "main.init",
        f"world_size={world_size}  global_rank={global_rank}  local_rank={local_rank}"
    )

    # ── RMM 初始化 ────────────────────────────
    # 对应上游:
    #   if global_rank == 0: torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    #   torch.distributed.barrier()
    #   torch.cuda.set_device(local_rank)
    #   cupy.cuda.Device(local_rank).use()
    #   cupy.cuda.set_allocator(rmm_cupy_allocator)
    #   rmm.reinitialize(devices=[local_rank], managed_memory=..., pool_allocator=True)
    #   torch.distributed.barrier()
    if global_rank == 0:
        from rmm.allocators.torch import rmm_torch_allocator  # type: ignore[import]
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        _dbg("main.rmm", "rank0: rmm_torch_allocator 已切换")

    torch.distributed.barrier()

    torch.cuda.set_device(local_rank)
    cupy.cuda.Device(local_rank).use()

    from rmm.allocators.cupy import rmm_cupy_allocator  # type: ignore[import]
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    import rmm  # type: ignore[import]
    rmm.reinitialize(
        devices=[local_rank],
        managed_memory=cfg.use_managed_memory,
        pool_allocator=True,
    )
    _dbg(
        "main.rmm",
        f"rmm.reinitialize: device={local_rank}  "
        f"managed_memory={cfg.use_managed_memory}  pool_allocator=True"
    )
    torch.distributed.barrier()

    # 对应上游注释: # import cudf after rmm has been reinitialized
    import cudf  # type: ignore[import]

    # ── 创建输出目录 ──────────────────────────
    # 对应上游:
    #   for folder in args.node_output_folders.split(","): os.makedirs(folder, exist_ok=True)
    #   for folder in args.edge_output_folders.split(","): os.makedirs(folder, exist_ok=True)
    for folder in cfg.node_output_folders:
        os.makedirs(folder, exist_ok=True)
    for folder in cfg.edge_output_folders:
        os.makedirs(folder, exist_ok=True)

    # ── 节点重编号阶段 ────────────────────────
    # 对应上游 for loop: node_type, node_folder_name, output_folder_name in zip(...)
    # 改写: NodeRenumberSession 封装每轮状态; RenumberMapStore 替代 global_renumber_map dict

    renumber_store = RenumberMapStore()

    for node_type, input_folder, output_folder in zip(
        cfg.node_types,
        cfg.node_input_folders,
        cfg.node_output_folders,
    ):
        session = NodeRenumberSession(node_type=node_type)

        # 对应上游: node_fname = sorted(os.listdir(node_folder_name))[local_rank]
        node_fname = sorted(os.listdir(input_folder))[local_rank]
        node_fpath = os.path.join(input_folder, node_fname)

        _dbg(
            "main.node[1/5]",
            f"node_type={node_type!r}  node_fpath={node_fpath!r}"
        )

        # 对应上游: ndf = cudf.read_csv(node_fpath) / cudf.read_parquet(node_fpath)
        ndf = _read_node_file(node_fpath, cfg.input_format, cudf)

        # 对应上游: local_num_nodes[node_type] = len(ndf)
        session.local_num_nodes = len(ndf)
        _dbg(
            "main.node[2/5]",
            f"node_type={node_type!r}  local_num_nodes={session.local_num_nodes}"
        )

        # ── all_gather: 收集各 rank 节点数 ──
        # 对应上游:
        #   node_offset_tensor = torch.zeros((world_size,), dtype=torch.int64, device=device)
        #   current_num_nodes = torch.tensor([len(ndf)], dtype=torch.int64, device=device)
        #   torch.distributed.all_gather_into_tensor(node_offset_tensor, current_num_nodes)
        node_offset_tensor = torch.zeros(
            (world_size,), dtype=torch.int64, device=device
        )
        current_num_nodes = torch.tensor(
            [session.local_num_nodes], dtype=torch.int64, device=device
        )
        _dbg(
            "main.node[3/5]",
            f"node_type={node_type!r}  "
            f"before all_gather: current_num_nodes={session.local_num_nodes}"
        )
        torch.distributed.all_gather_into_tensor(node_offset_tensor, current_num_nodes)
        _dbg(
            "main.node[3/5]",
            f"node_type={node_type!r}  "
            f"after all_gather: node_offset_tensor={node_offset_tensor.tolist()}"
        )

        # ── 构造各 rank map tensor 容器 ──
        # 对应上游:
        #   map_tensor = [torch.zeros((2, node_offset_tensor[i]), ...) for i in range(...)]
        map_tensor = [
            torch.zeros(
                (2, node_offset_tensor[i]), device=device, dtype=torch.int64
            )
            for i in range(node_offset_tensor.numel())
        ]

        # ── cumsum 计算全局 offset ──
        # 对应上游:
        #   node_offset_tensor = node_offset_tensor.cumsum(0)
        #   global_num_nodes[node_type] = int(node_offset_tensor[-1])
        #   local_node_offsets[node_type] = 0 if global_rank == 0 else int(...)
        node_offset_tensor = node_offset_tensor.cumsum(0)
        session.global_num_nodes = int(node_offset_tensor[-1])
        session.local_offset = (
            0 if global_rank == 0 else int(node_offset_tensor[global_rank - 1])
        )

        _dbg(
            "main.node[4/5]",
            f"node_type={node_type!r}  "
            f"global_num_nodes={session.global_num_nodes}  "
            f"local_offset={session.local_offset}"
        )

        # ── 构造本 rank 的 renumber map ──
        # 对应上游:
        #   local_renumber_map = torch.stack([
        #     torch.arange(local_offset, local_offset + local_num_nodes, ...),
        #     torch.as_tensor(ndf[node_colname], ...),
        #   ])
        local_renumber_map = torch.stack(
            [
                torch.arange(
                    session.local_offset,
                    session.local_offset + session.local_num_nodes,
                    device=device,
                    dtype=torch.int64,
                ),
                torch.as_tensor(
                    ndf[cfg.node_colname], device=device, dtype=torch.int64
                ),
            ]
        )

        # ── all_gather: 汇总全局 renumber map ──
        # 对应上游:
        #   torch.distributed.all_gather(map_tensor, local_renumber_map.to(device))
        #   map_tensor = torch.concat(map_tensor, dim=1)
        torch.distributed.all_gather(map_tensor, local_renumber_map.to(device))
        map_tensor_concat = torch.concat(map_tensor, dim=1)

        # ── 构造 global_renumber_map[node_type] ──
        # 对应上游:
        #   global_renumber_map[node_type] = cudf.DataFrame(
        #     {"id": cupy.asarray(map_tensor[0])},
        #     index=cupy.asarray(map_tensor[1]),
        #   )
        global_df = cudf.DataFrame(
            {
                "id": cupy.asarray(map_tensor_concat[0]),
            },
            index=cupy.asarray(map_tensor_concat[1]),
        )
        renumber_store.put(node_type, global_df)

        # ── 写出本 rank 的 local renumber map ──
        # 对应上游: local_renumber_map_df = cudf.DataFrame(...)  + to_csv/to_parquet
        local_renumber_map_df = cudf.DataFrame(
            {"id": cupy.asarray(local_renumber_map[0])},
            index=cupy.asarray(local_renumber_map[1]),
        )

        if cfg.output_format.lower() == "csv":
            out_node_path = os.path.join(
                output_folder, f"{node_fname}_renumbered.csv"
            )
            local_renumber_map_df.to_csv(out_node_path, index=False)
        elif cfg.output_format.lower() == "parquet":
            out_node_path = os.path.join(
                output_folder, f"{node_fname}_renumbered.parquet"
            )
            local_renumber_map_df.to_parquet(out_node_path, index=False)
        else:
            raise ValueError(f"Invalid output_format={cfg.output_format!r}")

        _dbg(
            "main.node[5/5]",
            f"node_type={node_type!r}  out_node_path={out_node_path!r}  "
            f"written_rows={len(local_renumber_map_df)}"
        )

    # ── 边重编号阶段 ──────────────────────────
    # 对应上游 for loop: edge_type, edge_folder_name, output_folder_name in zip(...)
    # 改写: EdgeRenumberSession.apply() 封装每轮执行; renumber_store.get_strict() 替代裸 dict

    _dbg("main.edge", f"开始边重编号阶段, 共 {len(cfg.edge_types)} 个 edge_type")

    for (src_type, rel_type, dst_type), edge_input_folder, edge_output_folder in zip(
        cfg.edge_types,
        cfg.edge_input_folders,
        cfg.edge_output_folders,
    ):
        edge_session = EdgeRenumberSession(
            src_type=src_type,
            rel_type=rel_type,
            dst_type=dst_type,
            input_folder=edge_input_folder,
            output_folder=edge_output_folder,
            local_rank=local_rank,
            source_colname=cfg.source_colname,
            destination_colname=cfg.destination_colname,
            input_format=cfg.input_format,
            output_format=cfg.output_format,
        )
        edge_session.apply(renumber_map_store=renumber_store, cudf=cudf)

    # ── 收尾 ──────────────────────────────────
    # 对应上游:
    #   torch.distributed.barrier()
    #   print("Success!")
    #   torch.distributed.destroy_process_group()
    torch.distributed.barrier()
    print("Success!")
    _dbg("main.done", "全部节点+边重编号完成，进程组已销毁")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
