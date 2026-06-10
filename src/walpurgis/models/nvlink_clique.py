"""
nvlink_clique.py — 824a809 迁移: NVLink Clique UUID 识别层

migrate 824a809: fix mnnvl issue with using nvlink clique uuid

上游变化 (824a809):
  1. communicator.cpp exchange_rank_info():
     - 新增 #include <string>
     - std::set<int> clique_ids{} → std::set<std::string> clique_uuids{}
       (根因: cliqueId 是 int，在多套 NVLink fabric 拓扑下可能跨 clique 重复；
        clusterUuid 是 NVML_GPU_FABRIC_UUID_LEN 字节的二进制串，全局唯一)
     - clique_ids.insert(cliqueId) →
       clique_uuids.insert(std::string(reinterpret_cast<const char*>(clusterUuid),
                                       NVML_GPU_FABRIC_UUID_LEN))
       (用 clusterUuid 二进制 blob 作 set key，而非 int cliqueId)
     - wm_comm->clique_info.clique_num = clique_uuids.size()
     - 提取本 rank 自身的 uuid 字符串到局部变量 std::string uuid
     - for 循环改用 clique_uuid 与 uuid 做字符串相等比较
       (原先用 cliqueId == ri.fabric_info.cliqueId 做整数比较)
  2. 末尾 namespace wholememory 注释补 newline (无逻辑变化)

Bug 根因:
  MNNVL (Multi-Node NVLink) 环境下, 每个 GPU Fabric 有独立的 cliqueId 编号空间。
  不同物理 clique 可能被 NVML 分配相同的 cliqueId 整数值。
  若将 int cliqueId 作为全局唯一标识符存入 set, 会导致本属于不同 clique 的 GPU
  被误判为同一 clique, clique_num 偏少, clique_id 赋值错误 → MNNVL 通信拓扑错误。
  修复: 改用 clusterUuid (128-bit 二进制 blob，NVML 保证全局唯一) 作 set 键。

Walpurgis 改写20%(鲁迅拿法):
  - CliqueUUID: 将 std::string(reinterpret_cast<const char*>(uuid_bytes), LEN)
    封装为 Python bytes 对象 + 比较方法，替代 C++ std::string 直接比较。
    改写: 加 hex_str 属性方便调试打印；C++ 直接打印二进制串不可读，Python改写后
    可打印形如 "a3f8..." 的十六进制摘要。
  - CliqueRegistry: 对应 std::set<std::string> clique_uuids + clique_num/clique_id
    两段逻辑, 合并为单一对象管理 uuid 集合 + id 分配。
    改写: 用 dict[uuid, int] 替代 set + id 计数器; 插入即分配 id, 无需二次遍历。
    (C++: 先 set.insert 全部，再 for 遍历 set 找 id; 我们改写为一次 O(1) 查表)
  - WalpurgisCliqueInfo: 对应 wm_comm->clique_info 结构体字段赋值。
    改写: 封装为 dataclass，clique_id 赋值前做合法性校验 (C++ 无校验，直接赋值)。
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - 每次 uuid 插入打印 hex 摘要 + 是否新增
    - clique_id 赋值时打印 uuid → id 映射
    - 最终 clique_info 完整 dump

作者: dylanyunlon<dogechat@163.com>
"""
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'

# 对应 C++ NVML_GPU_FABRIC_UUID_LEN (16 字节)
NVML_GPU_FABRIC_UUID_LEN: int = 16


def _dbg_clique(tag: str, msg: str) -> None:
    """断点调试: NVLink clique 专用 print"""
    if _DBG:
        print(f"[DEBUG 824a809 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── CliqueUUID: 对应 std::string(reinterpret_cast<const char*>(clusterUuid), LEN) ──
# C++ (824a809 communicator.cpp):
#   clique_uuids.insert(
#     std::string(reinterpret_cast<const char*>(p_rank_info.get()[r].fabric_info.clusterUuid),
#                 NVML_GPU_FABRIC_UUID_LEN));
#
# 改写: 封装为对象，加 hex_str 调试属性；C++ std::string 二进制 blob 调试不可读，
#   Python 版本改写后每次打印均为十六进制摘要。
class CliqueUUID:
    """
    对应 C++ std::string 存储的 clusterUuid 二进制 blob。

    C++ 用 std::string 存储原始字节 (length=NVML_GPU_FABRIC_UUID_LEN)。
    Python 改写: 用 bytes 对象存储，提供 hex_str 属性方便调试，
    __eq__/__hash__ 保证在 set/dict 中正确去重。

    根因修复核心:
      cliqueId (int) 在多 fabric 拓扑下不唯一 → bug
      clusterUuid (bytes[16]) 全局唯一 → fix
    """

    __slots__ = ('_raw',)

    def __init__(self, uuid_bytes: bytes) -> None:
        """
        对应 C++:
          std::string(reinterpret_cast<const char*>(clusterUuid), NVML_GPU_FABRIC_UUID_LEN)
        """
        if len(uuid_bytes) != NVML_GPU_FABRIC_UUID_LEN:
            # 断点调试: 长度不对立即报警，C++ 直接截断，Python 改写加校验
            _dbg_clique(
                "CliqueUUID.__init__",
                f"uuid_bytes length={len(uuid_bytes)} != expected {NVML_GPU_FABRIC_UUID_LEN} "
                f"— truncating/padding (mirrors C++ std::string(ptr, len) behavior)"
            )
        # 对齐 C++ std::string(ptr, NVML_GPU_FABRIC_UUID_LEN): 截断或右补零
        if len(uuid_bytes) >= NVML_GPU_FABRIC_UUID_LEN:
            self._raw: bytes = uuid_bytes[:NVML_GPU_FABRIC_UUID_LEN]
        else:
            self._raw = uuid_bytes + b'\x00' * (NVML_GPU_FABRIC_UUID_LEN - len(uuid_bytes))

    @property
    def raw(self) -> bytes:
        return self._raw

    @property
    def hex_str(self) -> str:
        """断点调试用: 打印可读的 hex 摘要，C++ 打印二进制 blob 不可读"""
        return self._raw.hex()

    @property
    def short_hex(self) -> str:
        """8字符短摘要, 用于日志"""
        return self._raw.hex()[:8] + "..."

    def is_zero(self) -> bool:
        """
        对应 C++ 824a809 前的 UUID 零值检查:
          (((long*)ri.fabric_info.clusterUuid)[0] | ((long*)ri.fabric_info.clusterUuid)[1]) == 0
        零 UUID 表示无 MNNVL fabric 信息，is_in_clique=0
        """
        return self._raw == b'\x00' * NVML_GPU_FABRIC_UUID_LEN

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CliqueUUID):
            return self._raw == other._raw
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._raw)

    def __repr__(self) -> str:
        return f"CliqueUUID({self.short_hex})"

    @classmethod
    def from_int_clique_id(cls, clique_id: int) -> 'CliqueUUID':
        """
        兼容旧接口: 将 int cliqueId 转为 CliqueUUID。
        仅用于测试/对比，不应在生产路径使用 —— 这正是 824a809 修复的 bug。

        断点调试: 调用此方法时打印警告，提示使用了已知有 bug 的 int cliqueId 路径
        """
        _dbg_clique(
            "CliqueUUID.from_int_clique_id",
            f"WARNING: using int cliqueId={clique_id} as UUID — "
            f"this is the PRE-824a809 bug path! "
            f"int cliqueId is NOT globally unique across fabrics."
        )
        # 将 int 编码为 16 字节小端，仅用于测试兼容
        raw = clique_id.to_bytes(NVML_GPU_FABRIC_UUID_LEN, byteorder='little', signed=False)
        return cls(raw)


# ─── CliqueRegistry: 合并 std::set<std::string> 插入 + clique_id 分配两段逻辑 ──
# C++ (824a809 communicator.cpp) 原先两段代码:
#
# [第一段, loop 内]:
#   clique_uuids.insert(std::string(...clusterUuid..., NVML_GPU_FABRIC_UUID_LEN));
#
# [第二段, loop 后]:
#   wm_comm->clique_info.clique_num = clique_uuids.size();
#   std::string uuid = std::string(...ri.fabric_info.clusterUuid..., NVML_GPU_FABRIC_UUID_LEN);
#   int id = 0;
#   for (auto clique_uuid : clique_uuids) {
#     if (clique_uuid == uuid) { wm_comm->clique_info.clique_id = id; }
#     id++;
#   }
#
# 改写: 合并为 CliqueRegistry, 插入即分配 id (O(1) 查表代替 O(N) 二次遍历)
# 说明: std::set 是有序集合, 遍历时按字典序, id=0 对应字典序最小 uuid;
#   Python dict (3.7+) 保留插入顺序, id 按首次插入顺序分配，与 std::set 字典序不同。
#   如需完全对齐 C++ std::set 顺序, 使用 sorted_id() 方法。
class CliqueRegistry:
    """
    对应 C++ std::set<std::string> clique_uuids + 后续 clique_id 赋值逻辑。

    核心 bug 修复 (824a809):
      旧: std::set<int> clique_ids — int cliqueId 非全局唯一, 多 fabric 下碰撞
      新: std::set<std::string> clique_uuids — clusterUuid 二进制 blob 全局唯一

    改写: 用 dict[CliqueUUID, int] 插入即分配 id, 无需二次 for 循环。
    C++ 先 insert 全部再 for 遍历是因为 std::set 无 O(1) 查找自己的 rank;
    Python dict 可 O(1) 查找, 改写去掉二次遍历。
    """

    def __init__(self) -> None:
        # 改写: dict[CliqueUUID, int] 替代 std::set<std::string> + 计数器
        # key=CliqueUUID, value=分配的 clique_id (按插入顺序, 从 0 开始)
        self._uuid_to_id: Dict[CliqueUUID, int] = {}

        _dbg_clique("CliqueRegistry.__init__", "empty registry created")

    def insert(self, uuid: CliqueUUID) -> Tuple[int, bool]:
        """
        对应 C++:
          clique_uuids.insert(std::string(...clusterUuid..., NVML_GPU_FABRIC_UUID_LEN))

        改写: 插入同时分配 id, 返回 (assigned_id, is_new)。
        C++ set.insert 返回 pair<iterator, bool>, 我们对齐这个语义。

        断点调试: 打印 uuid hex + 是否新增 + 当前 registry 大小
        """
        is_new = uuid not in self._uuid_to_id
        if is_new:
            new_id = len(self._uuid_to_id)
            self._uuid_to_id[uuid] = new_id
            _dbg_clique(
                "CliqueRegistry.insert",
                f"NEW uuid={uuid.short_hex} assigned clique_id={new_id} "
                f"registry_size={len(self._uuid_to_id)}"
            )
        else:
            existing_id = self._uuid_to_id[uuid]
            _dbg_clique(
                "CliqueRegistry.insert",
                f"DUP uuid={uuid.short_hex} already exists with clique_id={existing_id} "
                f"(same clique as a previous rank)"
            )
        return self._uuid_to_id[uuid], is_new

    def clique_num(self) -> int:
        """
        对应 C++: wm_comm->clique_info.clique_num = clique_uuids.size()
        """
        n = len(self._uuid_to_id)
        _dbg_clique("CliqueRegistry.clique_num", f"clique_num={n}")
        return n

    def get_id(self, uuid: CliqueUUID) -> Optional[int]:
        """
        对应 C++ 第二段 for 循环:
          for (auto clique_uuid : clique_uuids) {
            if (clique_uuid == uuid) { wm_comm->clique_info.clique_id = id; }
            id++;
          }

        改写: O(1) dict lookup 代替 O(N) for 循环。
        断点调试: 打印 uuid hex → clique_id 映射
        """
        result = self._uuid_to_id.get(uuid)
        _dbg_clique(
            "CliqueRegistry.get_id",
            f"uuid={uuid.short_hex} → clique_id={result} "
            f"({'FOUND' if result is not None else 'NOT FOUND — uuid not in registry'})"
        )
        if result is None:
            # C++ 此情况下 clique_id 不会被赋值, 保持初始值 (通常是 -1 或 0)
            # Python 改写: 明确返回 None，调用方决定如何处理
            print(
                f"[WARN 824a809 CliqueRegistry.get_id] "
                f"uuid={uuid.short_hex} not found in registry — "
                f"this rank's clique_id will remain unset (mirrors C++ uninitialized path)",
                file=sys.stderr
            )
        return result

    def sorted_id(self, uuid: CliqueUUID) -> Optional[int]:
        """
        按 C++ std::set 字典序重新分配 id，完全对齐 C++ 行为。

        C++ std::set<std::string> 按字典序排列 uuid，for 循环 id=0 对应最小 uuid。
        Python dict 按插入顺序，sorted_id() 提供与 C++ 完全对齐的 id 计算。

        断点调试: 打印 sorted uuid 列表 + 对应 id
        """
        sorted_uuids = sorted(self._uuid_to_id.keys(), key=lambda u: u.raw)
        _dbg_clique(
            "CliqueRegistry.sorted_id",
            f"sorted uuid order: {[u.short_hex for u in sorted_uuids]}"
        )
        for idx, u in enumerate(sorted_uuids):
            _dbg_clique(
                "CliqueRegistry.sorted_id",
                f"  id={idx} uuid={u.short_hex}"
            )
            if u == uuid:
                return idx
        return None

    def dump(self) -> None:
        """断点调试: 打印 CliqueRegistry 完整状态"""
        print(
            f"[DEBUG 824a809 CliqueRegistry.dump] "
            f"clique_num={self.clique_num()} entries:",
            file=sys.stderr
        )
        for uuid, cid in self._uuid_to_id.items():
            print(f"  clique_id={cid} uuid={uuid.hex_str}", file=sys.stderr)


# ─── WalpurgisCliqueInfo: 对应 wm_comm->clique_info 结构体字段 ─────────────
# C++ communicator.cpp (涉及 824a809 的字段):
#   int clique_first_rank;  // 当前 clique 第一个 rank 的 world rank
#   int clique_rank;        // 本 rank 在 clique 内的序号
#   int clique_rank_num;    // 本 clique 内 rank 总数
#   int clique_num;         // 全局 clique 数量  ← 824a809 修复
#   int clique_id;          // 本 rank 所属 clique 的 id  ← 824a809 修复
#   int is_in_clique;       // 是否在任意 clique 中
#
# 改写: 封装为 dataclass, clique_id 赋值前做合法性校验
# C++ 无校验直接赋值, 若 cliqueId 整数碰撞 (pre-824a809 bug), 赋值错误也无报警
@dataclass
class WalpurgisCliqueInfo:
    """
    对应 C++ wholememory::communicator 中的 clique_info 结构体。

    824a809 修复的两个字段: clique_num + clique_id
    改写: dataclass 替代 C struct, 加 validate() 方法在赋值后做一致性校验
    """
    clique_first_rank: int = -1    # 对应 C++ clique_info.clique_first_rank
    clique_rank: int = -1          # 对应 C++ clique_info.clique_rank
    clique_rank_num: int = 0       # 对应 C++ clique_info.clique_rank_num
    clique_num: int = 0            # 对应 C++ clique_info.clique_num (824a809 修复)
    clique_id: int = -1            # 对应 C++ clique_info.clique_id  (824a809 修复)
    is_in_clique: int = 0          # 对应 C++ clique_info.is_in_clique

    def validate(self) -> bool:
        """
        对应 C++ 无任何校验的直接赋值。
        改写: Python 版本加后置校验，捕捉 824a809 前的整数碰撞类 bug。

        断点调试: 打印完整 clique_info + 每项校验结果
        """
        ok = True

        # clique_id 必须在 [0, clique_num) 范围内
        if self.is_in_clique and self.clique_id < 0:
            print(
                f"[ERROR 824a809 WalpurgisCliqueInfo.validate] "
                f"is_in_clique=1 but clique_id={self.clique_id} < 0 — "
                f"clique_id was never assigned. "
                f"Pre-824a809 bug: int cliqueId碰撞导致 set 里找不到自己的 uuid",
                file=sys.stderr
            )
            ok = False

        if self.is_in_clique and self.clique_num <= 0:
            print(
                f"[ERROR 824a809 WalpurgisCliqueInfo.validate] "
                f"is_in_clique=1 but clique_num={self.clique_num} <= 0",
                file=sys.stderr
            )
            ok = False

        if self.is_in_clique and self.clique_id >= self.clique_num:
            print(
                f"[ERROR 824a809 WalpurgisCliqueInfo.validate] "
                f"clique_id={self.clique_id} >= clique_num={self.clique_num} — out of range",
                file=sys.stderr
            )
            ok = False

        _dbg_clique(
            "WalpurgisCliqueInfo.validate",
            f"clique_id={self.clique_id} clique_num={self.clique_num} "
            f"is_in_clique={self.is_in_clique} "
            f"clique_rank={self.clique_rank}/{self.clique_rank_num} "
            f"validate={'OK' if ok else 'FAIL'}"
        )
        return ok

    def dump(self) -> None:
        """断点调试: 打印完整 clique_info 结构"""
        print(
            f"[DEBUG 824a809 WalpurgisCliqueInfo]\n"
            f"  clique_id={self.clique_id}  clique_num={self.clique_num}\n"
            f"  is_in_clique={self.is_in_clique}\n"
            f"  clique_rank={self.clique_rank}  clique_rank_num={self.clique_rank_num}\n"
            f"  clique_first_rank={self.clique_first_rank}",
            file=sys.stderr
        )


# ─── exchange_rank_clique_info: 对应 exchange_rank_info() 中 clique 相关逻辑 ─
# C++ exchange_rank_info() (824a809 修复的核心路径):
#
#   // [loop 内, #if CUDA_VERSION >= 12030]
#   if (same clusterUuid AND same cliqueId) {
#     clique_rank_num++; clique_rank=...; clique_first_rank=...;
#   }
#   clique_uuids.insert(std::string(reinterpret_cast<const char*>(clusterUuid),
#                                   NVML_GPU_FABRIC_UUID_LEN));
#
#   // [loop 后, #if CUDA_VERSION >= 12030]
#   clique_num = clique_uuids.size();
#   uuid = std::string(reinterpret_cast<const char*>(ri.fabric_info.clusterUuid), LEN);
#   id = 0;
#   for (auto clique_uuid : clique_uuids) {
#     if (clique_uuid == uuid) { clique_id = id; }
#     id++;
#   }
#
# 改写: 提取为独立函数，输入为每个 rank 的 uuid 列表 + 本 rank world_rank,
#   输出为 WalpurgisCliqueInfo。无需依赖 wholememory_comm_t 结构体。
def exchange_rank_clique_info(
    rank_uuids: List[bytes],    # 每个 rank 的 clusterUuid (len=NVML_GPU_FABRIC_UUID_LEN)
    rank_clique_ids: List[int], # 每个 rank 的 cliqueId (int, 仅用于 clique membership 判断)
    world_rank: int,            # 本 rank 的 world rank
    use_sorted_id: bool = True, # True = 对齐 C++ std::set 字典序; False = 插入顺序
) -> WalpurgisCliqueInfo:
    """
    对应 C++ exchange_rank_info() 中 CUDA_VERSION >= 12030 的 clique 计算路径。

    824a809 核心修复:
      旧路径: std::set<int> clique_ids, key=int cliqueId → 多 fabric 下碰撞
      新路径: std::set<std::string> clique_uuids, key=clusterUuid binary → 全局唯一

    断点调试: WALPURGIS_DEBUG=1 时打印每个 rank 的 uuid + 最终 clique_info
    """
    world_size = len(rank_uuids)
    assert world_size == len(rank_clique_ids), (
        f"rank_uuids length={world_size} != rank_clique_ids length={len(rank_clique_ids)}"
    )
    assert 0 <= world_rank < world_size, (
        f"world_rank={world_rank} out of range [0, {world_size})"
    )

    _dbg_clique(
        "exchange_rank_clique_info",
        f"world_size={world_size} world_rank={world_rank} use_sorted_id={use_sorted_id}"
    )

    # 本 rank 自身的 uuid (对应 C++ ri.fabric_info.clusterUuid)
    self_uuid = CliqueUUID(rank_uuids[world_rank])
    _dbg_clique(
        "exchange_rank_clique_info",
        f"self uuid={self_uuid.short_hex} "
        f"is_zero={self_uuid.is_zero()} "
        f"(zero=no MNNVL fabric)"
    )

    # is_in_clique: 对应 824a809 前的 clusterUuid 零值检查
    is_in_clique = 0 if self_uuid.is_zero() else 1

    # ─── loop: 对应 C++ for (int r = 0; r < world_size; r++) ─────────────
    registry = CliqueRegistry()
    clique_info = WalpurgisCliqueInfo(is_in_clique=is_in_clique)

    for r in range(world_size):
        r_uuid = CliqueUUID(rank_uuids[r])
        r_clique_id_int = rank_clique_ids[r]

        _dbg_clique(
            "exchange_rank_clique_info.loop",
            f"r={r} uuid={r_uuid.short_hex} int_cliqueId={r_clique_id_int}"
        )

        # 对应 C++ if (same clusterUuid AND same cliqueId) — clique membership 判断
        # 注意: 这里 clique membership (clique_rank) 仍用 clusterUuid + cliqueId 双重判断
        # 而 clique_num/clique_id 的修复是改用纯 clusterUuid (824a809 的核心)
        same_uuid = (r_uuid == self_uuid)
        same_clique_id = (r_clique_id_int == rank_clique_ids[world_rank])

        if same_uuid and same_clique_id:
            # 对应 C++:
            #   if (r == world_rank) clique_rank = clique_rank_num;
            #   if (clique_rank_num == 0) clique_first_rank = r;
            #   clique_rank_num++;
            if r == world_rank:
                clique_info.clique_rank = clique_info.clique_rank_num
                _dbg_clique(
                    "exchange_rank_clique_info.loop",
                    f"r={r} == world_rank: clique_rank={clique_info.clique_rank}"
                )
            if clique_info.clique_rank_num == 0:
                clique_info.clique_first_rank = r
                _dbg_clique(
                    "exchange_rank_clique_info.loop",
                    f"clique_first_rank={r} (first rank in clique)"
                )
            clique_info.clique_rank_num += 1

        # 824a809 核心修复:
        # 旧: clique_ids.insert(r_clique_id_int)  ← int 碰撞 bug
        # 新: clique_uuids.insert(std::string(clusterUuid, LEN))  ← uuid 全局唯一
        registry.insert(r_uuid)

    # ─── loop 后: clique_num + clique_id 赋值 ──────────────────────────────
    # 对应 C++:
    #   clique_num = clique_uuids.size();
    #   uuid = std::string(ri.fabric_info.clusterUuid, LEN);
    #   id = 0;
    #   for (auto clique_uuid : clique_uuids) {
    #     if (clique_uuid == uuid) clique_id = id;
    #     id++;
    #   }
    clique_info.clique_num = registry.clique_num()

    # 改写: O(1) dict lookup 代替 O(N) for 循环
    # use_sorted_id=True 时对齐 C++ std::set 字典序
    if use_sorted_id:
        clique_id = registry.sorted_id(self_uuid)
    else:
        clique_id = registry.get_id(self_uuid)

    if clique_id is not None:
        clique_info.clique_id = clique_id
        _dbg_clique(
            "exchange_rank_clique_info",
            f"clique_id={clique_id} assigned for self_uuid={self_uuid.short_hex}"
        )
    else:
        # C++ 此路径下 clique_id 字段不会被赋值 (保持 0 或未初始化)
        # Python 改写: 保持 -1 (dataclass 初始值), validate() 会报警
        _dbg_clique(
            "exchange_rank_clique_info",
            f"self_uuid={self_uuid.short_hex} NOT found in registry — "
            f"clique_id remains unset (={clique_info.clique_id})"
        )

    if _DBG:
        registry.dump()
        clique_info.dump()

    # 改写: 赋值后立即 validate, C++ 无此步骤
    clique_info.validate()

    return clique_info


# ─── pre_824a809_buggy_exchange (仅用于对比测试，演示 bug) ──────────────────
def pre_824a809_buggy_exchange(
    rank_clique_ids: List[int],   # 仅用 int cliqueId (pre-824a809 bug)
    world_rank: int,
) -> Tuple[int, int]:
    """
    对应 C++ 824a809 前的旧路径:
      std::set<int> clique_ids{};
      ...
      clique_ids.insert(cliqueId);
      ...
      clique_num = clique_ids.size();
      for (auto clique_id : clique_ids) {
        if (clique_id == ri.fabric_info.cliqueId) clique_info.clique_id = id;
        id++;
      }

    返回 (buggy_clique_num, buggy_clique_id)。
    仅用于回归测试对比，证明 int cliqueId 在多 fabric 下会碰撞。

    断点调试: 打印 int cliqueId set + 与 uuid-based 结果的差异
    """
    _dbg_clique(
        "pre_824a809_buggy_exchange",
        f"WARNING: running PRE-FIX buggy path! "
        f"int cliqueIds={rank_clique_ids}"
    )

    # 旧: std::set<int>
    clique_id_set = sorted(set(rank_clique_ids))  # sorted 对齐 std::set 字典序
    buggy_clique_num = len(clique_id_set)

    self_clique_id_int = rank_clique_ids[world_rank]
    buggy_clique_id = -1
    for idx, cid in enumerate(clique_id_set):
        if cid == self_clique_id_int:
            buggy_clique_id = idx
            break

    _dbg_clique(
        "pre_824a809_buggy_exchange",
        f"buggy_clique_num={buggy_clique_num} buggy_clique_id={buggy_clique_id} "
        f"(based on int cliqueId, may be wrong in multi-fabric topology)"
    )
    return buggy_clique_num, buggy_clique_id
