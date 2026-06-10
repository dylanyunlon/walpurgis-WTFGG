"""
temporal_sampler.py — d4b52c9 迁移: Temporal Sampling 接口层

migrate d4b52c9: Enable Temporal Sampling in cuGraph-PyG

上游变化 (d4b52c9):
  1. graph_store.py: __etime_attr + _set_etime_attr() + __get_etime_tensor()
     GraphStore记录(feature_store, attr_name)对, 调用时从feature_store按attr_name
     和edge_index取出每条边的时间戳, 拼成跨edge-type的etime tensor加入edgelist_dict
  2. distributed_sampler.py: _func_table 8-entry dispatch dict
     key=(homo/hetero, uniform/biased, temporal:bool) → pylibcugraph采样函数
     temporal=True时额外设置 func_kwargs["temporal_property_name"]="time"
  3. neighbor_loader.py / link_neighbor_loader.py:
     - 移除: if time_attr is not None: raise ValueError("Temporal sampling unsupported")
     - 新增: is_temporal = time_attr is not None
     - 新增: if is_temporal: graph_store._set_etime_attr((feature_store, time_attr))
     - 新增: warnings.warn("Temporal sampling currently only forward in time...")
  4. node_loader.py / link_loader.py: 移除 input_time unsupported 报错

Walpurgis 改写20%(鲁迅拿法):
  - 无 pylibcugraph 依赖: TemporalSamplerDispatch 用 enum + dict[key, Callable]
    替代 Python _func_table 的硬编码 module.attr 引用
  - is_temporal_session() 替代 Python is_temporal 局部变量,
    改写为 TemporalSamplerSession 对象携带整个采样配置
  - WalpurgisEtimeStore 替代 Python (feature_store, attr_name) tuple,
    改写: 加 edge_type_name 字段 + get_etimes() 方法封装索引
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - 每次 set_etime_attr 打印 old/new 对比
    - 每次 select_sampler 打印 key + func name
    - 每次 get_etime_tensor 打印 per-type etime 范围

作者: dylanyunlon<dogechat@163.com>
"""
import sys
import os
from enum import Enum
from typing import Optional, Tuple, List, Dict, Callable, Any
import warnings

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_temporal(tag: str, msg: str) -> None:
    """断点调试: temporal sampling专用print"""
    if _DBG:
        print(f"[DEBUG d4b52c9 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── SamplerFunc: 对应 _func_table 的值 ────────────────────────────────────
# Python: 值是 pylibcugraph.homogeneous_uniform_temporal_neighbor_sample 等函数
# 改写: 用 enum 替代函数引用, 无 pylibcugraph 依赖; 实际dispatch在采样引擎中处理
class SamplerFunc(Enum):
    HOMO_UNIFORM_NONTEMPORAL   = "homogeneous_uniform_neighbor_sample"
    HOMO_BIASED_NONTEMPORAL    = "homogeneous_biased_neighbor_sample"
    HETERO_UNIFORM_NONTEMPORAL = "heterogeneous_uniform_neighbor_sample"
    HETERO_BIASED_NONTEMPORAL  = "heterogeneous_biased_neighbor_sample"
    # d4b52c9 新增的4个temporal路径 (上游commit的核心贡献)
    HOMO_UNIFORM_TEMPORAL      = "homogeneous_uniform_temporal_neighbor_sample"
    HOMO_BIASED_TEMPORAL       = "homogeneous_biased_temporal_neighbor_sample"
    HETERO_UNIFORM_TEMPORAL    = "heterogeneous_uniform_temporal_neighbor_sample"
    HETERO_BIASED_TEMPORAL     = "heterogeneous_biased_temporal_neighbor_sample"

    def is_temporal(self) -> bool:
        return "temporal" in self.value

    def is_heterogeneous(self) -> bool:
        return "heterogeneous" in self.value

    def is_biased(self) -> bool:
        return "biased" in self.value


# ─── TemporalSamplerDispatch: 对应 _func_table ──────────────────────────────
# Python (d4b52c9 distributed_sampler.py:59-108):
#   _func_table = {
#       ("homogeneous", "uniform", True):  pylibcugraph.homogeneous_uniform_temporal_...,
#       ("homogeneous", "uniform", False): pylibcugraph.homogeneous_uniform_...,
#       ... (8 total entries)
#   }
# 改写: 用 Tuple[bool, bool, bool] 键替代 Tuple[str, str, bool],
#   整数索引避免字符串比较, linear layout = O(1) lookup
class TemporalSamplerDispatch:
    """
    8-entry dispatch table mirroring DistributedNeighborSampler._func_table.

    Key: (heterogeneous: bool, biased: bool, temporal: bool)
    Index: temporal*4 + heterogeneous*2 + biased*1  → [0, 7]
    """

    # 按 index() 顺序排列的8个函数枚举
    _TABLE: List[SamplerFunc] = [
        SamplerFunc.HOMO_UNIFORM_NONTEMPORAL,    # 000
        SamplerFunc.HOMO_BIASED_NONTEMPORAL,     # 001
        SamplerFunc.HETERO_UNIFORM_NONTEMPORAL,  # 010
        SamplerFunc.HETERO_BIASED_NONTEMPORAL,   # 011
        SamplerFunc.HOMO_UNIFORM_TEMPORAL,       # 100
        SamplerFunc.HOMO_BIASED_TEMPORAL,        # 101
        SamplerFunc.HETERO_UNIFORM_TEMPORAL,     # 110
        SamplerFunc.HETERO_BIASED_TEMPORAL,      # 111
    ]

    @staticmethod
    def _index(heterogeneous: bool, biased: bool, temporal: bool) -> int:
        return (4 if temporal else 0) | (2 if heterogeneous else 0) | (1 if biased else 0)

    @classmethod
    def select(cls, heterogeneous: bool, biased: bool, temporal: bool) -> SamplerFunc:
        """
        对应 Python:
          self.__func = self._func_table[
              ("heterogeneous" if heterogeneous else "homogeneous",
               "uniform" if not biased else "biased",
               temporal)
          ]
        """
        idx = cls._index(heterogeneous, biased, temporal)
        func = cls._TABLE[idx]
        _dbg_temporal(
            "TemporalSamplerDispatch.select",
            f"hetero={heterogeneous} biased={biased} temporal={temporal} "
            f"-> {func.value}"
        )
        return func

    @classmethod
    def dump_all(cls) -> None:
        """打印全部8条路径 — 断点调试用"""
        print("[DEBUG d4b52c9 TemporalSamplerDispatch.dump_all] 8-entry dispatch table:",
              file=sys.stderr)
        for i, func in enumerate(cls._TABLE):
            is_t = bool(i & 4)
            print(f"  [{i}] {func.value}  {'<-- TEMPORAL PATH' if is_t else ''}",
                  file=sys.stderr)


# ─── WalpurgisEtimeStore: 对应 (feature_store, attr_name) tuple ─────────────
# Python (d4b52c9 graph_store.py):
#   self.__etime_attr = (feature_store, attr_name)  # attr_name = "time"
# 改写: 封装为对象, 加 edge_type_name + get_etimes() 方法
class WalpurgisEtimeStore:
    """
    对应 graph_store.py __etime_attr Tuple[FeatureStore, str].

    Python stores (feature_store, attr_name), then __get_etime_tensor() calls:
        etime = feature_store[et, attr_name][ix]
    We encapsulate this lookup in get_etimes().

    改写: 加 edge_type_name 字段 + 独立 get_etimes() 方法 (Python是在GraphStore.edgelist_dict
    构建时内联调用, 我们提取到独立方法, 更易测试)
    """
    def __init__(
        self,
        attr_name: str,                          # e.g., "time"
        etime_data: Dict[str, Any],              # {edge_type_name: etime_tensor/array}
        edge_type_names: Optional[List[str]] = None,  # 改写: 支持多edge type
    ):
        self.attr_name = attr_name
        self._data = etime_data  # keyed by edge_type_name
        self.edge_type_names = edge_type_names or list(etime_data.keys())

        _dbg_temporal(
            "WalpurgisEtimeStore.__init__",
            f"attr_name='{attr_name}' edge_types={self.edge_type_names} "
            f"n_types={len(self.edge_type_names)}"
        )

    @property
    def is_valid(self) -> bool:
        return bool(self._data) and bool(self.attr_name)

    def get_etimes(
        self,
        edge_type: str,
        start_offset: int,
        count: int,
    ):
        """
        对应 d4b52c9 __get_etime_tensor 中的单edge-type查询:
            ix = torch.arange(start_offsets[i], start_offsets[i] + num_edges_t[i])
            etime = feature_store[et, attr_name][ix]
            if etime is None: raise ValueError("Time property must be present...")

        断点调试: 打印 edge_type, offset, count, etime range
        """
        if edge_type not in self._data:
            _dbg_temporal(
                "WalpurgisEtimeStore.get_etimes",
                f"edge_type='{edge_type}' NOT FOUND in store. "
                f"Available: {list(self._data.keys())}"
            )
            raise ValueError(
                f"Time property must be present for all edge types. "
                f"Missing: '{edge_type}' (mirrors d4b52c9 ValueError)"
            )

        etime_all = self._data[edge_type]
        etime_slice = etime_all[start_offset:start_offset + count]

        _dbg_temporal(
            "WalpurgisEtimeStore.get_etimes",
            f"edge_type='{edge_type}' start={start_offset} count={count} "
            f"etime_range=[{etime_slice[0] if len(etime_slice) > 0 else 'N/A'}, "
            f"{etime_slice[-1] if len(etime_slice) > 0 else 'N/A'}]"
        )
        return etime_slice

    def get_etime_tensor(
        self,
        sorted_edge_types: List[str],
        start_offsets: List[int],
        num_edges: List[int],
    ):
        """
        对应 d4b52c9 graph_store.py __get_etime_tensor():
            etimes = []
            for i, et in enumerate(sorted_keys):
                ix = torch.arange(start_offsets[i], start_offsets[i]+num_edges_t[i])
                etime = feature_store[et, attr_name][ix]
                if etime is None: raise ValueError(...)
                etimes.append(etime)
            return torch.concat(etimes)
        """
        _dbg_temporal(
            "WalpurgisEtimeStore.get_etime_tensor",
            f"attr='{self.attr_name}' num_edge_types={len(sorted_edge_types)}"
        )

        all_etimes = []
        for i, et in enumerate(sorted_edge_types):
            start = start_offsets[i] if i < len(start_offsets) else 0
            count = num_edges[i] if i < len(num_edges) else 0
            chunk = self.get_etimes(et, start, count)  # raises ValueError if missing
            all_etimes.extend(chunk)

        _dbg_temporal(
            "WalpurgisEtimeStore.get_etime_tensor",
            f"concat result: total_etimes={len(all_etimes)}"
        )
        return all_etimes

    def dump(self) -> None:
        """断点调试: 打印 EtimeStore 完整状态"""
        print(
            f"[DEBUG d4b52c9 WalpurgisEtimeStore] attr='{self.attr_name}' "
            f"edge_types={self.edge_type_names} valid={self.is_valid}",
            file=sys.stderr
        )
        for et, data in self._data.items():
            n = len(data) if hasattr(data, '__len__') else '?'
            first = data[0] if hasattr(data, '__getitem__') and len(data) > 0 else '?'
            last = data[-1] if hasattr(data, '__getitem__') and len(data) > 0 else '?'
            print(f"  edge_type='{et}' n={n} range=[{first}, {last}]",
                  file=sys.stderr)


# ─── TemporalSamplerSession: 对应 NeighborLoader/LinkNeighborLoader 的完整配置 ──
# Python (d4b52c9 neighbor_loader.py):
#   is_temporal = time_attr is not None
#   if is_temporal:
#       graph_store._set_etime_attr((feature_store, time_attr))
#       warnings.warn("Temporal sampling ... currently only forward in time ...")
#   sampler = DistributedNeighborSampler(..., temporal=is_temporal, ...)
#
# 改写: 封装为对象, 携带整个配置; Python是局部变量+函数调用,
#       我们改写为可序列化的配置对象(便于多进程采样共享配置)
class TemporalSamplerSession:
    """
    Encapsulates the temporal sampling configuration for one NeighborLoader session.
    Mirrors d4b52c9 is_temporal + _set_etime_attr + DistributedNeighborSampler init.

    改写比Python更结构化: Python仅有is_temporal局部变量 + 零散的_set_etime_attr调用,
    我们改写为单一对象, 所有配置集中管理.
    """

    FORWARD_IN_TIME_WARNING = (
        "Temporal sampling in cuGraph-PyG is currently only forward in time "
        "instead of the expected backward in time. "
        "This will be fixed in a future release."
        " (d4b52c9 FIXME, mirrors upstream warning)"
    )

    def __init__(
        self,
        time_attr: Optional[str] = None,         # e.g., "time"; None = non-temporal
        edge_label_time=None,                     # for LinkNeighborLoader (d4b52c9 link_loader.py)
        weight_attr: Optional[str] = None,        # for biased sampling
        heterogeneous: bool = False,
        etime_store: Optional[WalpurgisEtimeStore] = None,
        emit_forward_warning: bool = True,        # matches d4b52c9 warnings.warn
    ):
        # d4b52c9 neighbor_loader.py: is_temporal = time_attr is not None
        # d4b52c9 link_neighbor_loader.py: is_temporal = (edge_label_time is not None) and (time_attr is not None)
        if edge_label_time is not None:
            # LinkNeighborLoader path
            self.is_temporal = (time_attr is not None)
        else:
            # NeighborLoader path
            self.is_temporal = (time_attr is not None)

        self.time_attr = time_attr
        self.edge_label_time = edge_label_time
        self.weight_attr = weight_attr
        self.heterogeneous = heterogeneous
        self.etime_store = etime_store
        self.biased = (weight_attr is not None)

        _dbg_temporal(
            "TemporalSamplerSession.__init__",
            f"time_attr={time_attr!r} edge_label_time={'<set>' if edge_label_time is not None else 'None'} "
            f"is_temporal={self.is_temporal} biased={self.biased} hetero={self.heterogeneous}"
        )

        # d4b52c9: warnings.warn("Temporal sampling ... only forward in time ...")
        if self.is_temporal and emit_forward_warning:
            warnings.warn(self.FORWARD_IN_TIME_WARNING, UserWarning, stacklevel=2)
            _dbg_temporal("TemporalSamplerSession.warning", self.FORWARD_IN_TIME_WARNING)

        # d4b52c9: if is_temporal: graph_store._set_etime_attr((feature_store, time_attr))
        if self.is_temporal and etime_store is not None:
            _dbg_temporal(
                "TemporalSamplerSession._set_etime_attr",
                f"attr='{time_attr}' store.valid={etime_store.is_valid}"
            )
            etime_store.dump()

        # d4b52c9: select sampler function from _func_table
        self.sampler_func: SamplerFunc = TemporalSamplerDispatch.select(
            self.heterogeneous, self.biased, self.is_temporal
        )

        # d4b52c9: if temporal: func_kwargs["temporal_property_name"] = "time"
        self.func_kwargs: Dict[str, Any] = {}
        if self.is_temporal:
            self.func_kwargs["temporal_property_name"] = "time"
            _dbg_temporal(
                "TemporalSamplerSession",
                f"func_kwargs[temporal_property_name] = 'time' (d4b52c9)"
            )

    def validate(self) -> bool:
        """
        Validate session consistency.
        断点调试: 打印完整配置摘要.
        """
        _dbg_temporal(
            "TemporalSamplerSession.validate",
            f"sampler={self.sampler_func.value} "
            f"is_temporal={self.is_temporal} "
            f"func_kwargs={self.func_kwargs} "
            f"etime_store={'valid' if (self.etime_store and self.etime_store.is_valid) else 'None/invalid'}"
        )

        if self.is_temporal and self.etime_store is None:
            # d4b52c9: is_temporal=True requires etime_store set
            print(
                f"[WARN d4b52c9 TemporalSamplerSession.validate] "
                f"is_temporal=True but etime_store is None — "
                f"temporal constraint cannot be applied",
                file=sys.stderr
            )
            return False

        if self.is_temporal and not self.sampler_func.is_temporal():
            print(
                f"[ERROR d4b52c9 TemporalSamplerSession.validate] "
                f"is_temporal=True but sampler_func={self.sampler_func.value} "
                f"is not temporal! Dispatch bug.",
                file=sys.stderr
            )
            return False

        return True

    def dump_state(self) -> None:
        """断点调试: 打印 TemporalSamplerSession 完整状态"""
        print(
            f"[DEBUG d4b52c9 TemporalSamplerSession.dump_state]\n"
            f"  is_temporal={self.is_temporal}\n"
            f"  time_attr={self.time_attr!r}\n"
            f"  edge_label_time={'<set>' if self.edge_label_time is not None else 'None'}\n"
            f"  biased={self.biased}  heterogeneous={self.heterogeneous}\n"
            f"  sampler_func={self.sampler_func.value}\n"
            f"  func_kwargs={self.func_kwargs}\n"
            f"  etime_store={'valid' if (self.etime_store and self.etime_store.is_valid) else 'None/invalid'}",
            file=sys.stderr
        )
        if self.etime_store:
            self.etime_store.dump()


# ─── Convenience builder: 对应 NeighborLoader.__init__ 中 is_temporal 路径 ──
def make_temporal_session_from_loader_args(
    time_attr: Optional[str],
    etime_data: Optional[Dict[str, Any]] = None,
    edge_label_time=None,
    weight_attr: Optional[str] = None,
    heterogeneous: bool = False,
    emit_warning: bool = True,
) -> TemporalSamplerSession:
    """
    建立 TemporalSamplerSession, 对应 d4b52c9 NeighborLoader/LinkNeighborLoader
    __init__ 中从 loader args 到 sampler 配置的转换.

    Usage (mirrors d4b52c9 neighbor_loader.py):
        session = make_temporal_session_from_loader_args(
            time_attr="time",
            etime_data={("paper","cites","paper"): tme_cite_tensor},
            heterogeneous=True,
        )
        assert session.is_temporal
        assert session.sampler_func == SamplerFunc.HETERO_UNIFORM_TEMPORAL

    断点调试: WALPURGIS_DEBUG=1 时打印完整dispatch decision
    """
    _dbg_temporal(
        "make_temporal_session_from_loader_args",
        f"time_attr={time_attr!r} has_etime_data={etime_data is not None} "
        f"has_edge_label_time={edge_label_time is not None} "
        f"weight_attr={weight_attr!r} hetero={heterogeneous}"
    )

    etime_store = None
    if time_attr is not None and etime_data is not None:
        etime_store = WalpurgisEtimeStore(
            attr_name=time_attr,
            etime_data=etime_data,
        )

    session = TemporalSamplerSession(
        time_attr=time_attr,
        edge_label_time=edge_label_time,
        weight_attr=weight_attr,
        heterogeneous=heterogeneous,
        etime_store=etime_store,
        emit_forward_warning=emit_warning,
    )

    if _DBG:
        session.dump_state()

    return session
