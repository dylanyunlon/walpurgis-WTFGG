"""
metadata_sampler.py — 2ba9979 迁移: Distributed Sampler Metadata 传播层

migrate 2ba9979: Propagate Changes from cuGraph Distributed Sampler (metadata)

上游变化 (2ba9979):
  1. sampler_utils.py:
     - 新增 verify_metadata(metadata): 校验 Dict[str, Union[str, Tuple[str,str,str]]]
       键必须是str, 值是str或长度3的全str元组; 否则 assert 报错
  2. distributed_sampler.py:
     - 所有核心函数签名追加 metadata: Optional[Dict[str, Union[str, Tuple[str,str,str]]]] = None
       涉及函数: __sample_from_nodes_impl, __buffered_sample_from_nodes_impl,
                 sample_from_nodes, __buffered_sample_from_edges_impl,
                 sample_from_edges, DistributedNeighborSampler.__sample
     - sample_from_nodes 入口调用 verify_metadata(metadata)
     - __sample 出口: if metadata is not None: sampling_results_dict.update(metadata)
     - minibatch_dict 字典推导式修复:
         旧: k: torch.as_tensor(v, device="cuda")
         新: k: v if isinstance(v, (str, tuple)) else torch.as_tensor(v, device="cuda")
       防止 metadata 中的 str/tuple 值被 torch.as_tensor 转换崩溃
     - 新增 from cugraph_pyg.sampler.sampler_utils import verify_metadata
  3. tests/sampler/test_distributed_sampler.py:
     - test_dist_sampler_hetero_from_nodes: 传入 metadata={"some_key": "some_value"}
     - 断言 out["some_key"] == "some_value" 验证metadata透传到采样结果

Knuth 审查:
  1. diff对比源:
     - verify_metadata 用 assert 而非 raise ValueError, 生产环境 python -O 会跳过所有断言;
       上游应换成显式 raise TypeError/ValueError (我们改写时已修正为 raise)
     - minibatch_dict 推导式修复是必要的: str/tuple 无法 as_tensor,
       原代码对 metadata 值静默崩溃; 新代码 isinstance 分支正确
     - sampling_results_dict.update(metadata) 在 __sample 末尾, 无冲突检测;
       若 metadata key 与采样结果 key 同名(如 "rank", "fanout"), 会静默覆盖采样数据 (BUG)
  2. 用户角度 bug:
     - metadata={"rank": "user_label"} 会覆盖 sampling_results_dict["rank"] (来自torch.distributed.get_rank());
       用户以为传入的是业务标签, 实际却污染了分布式 rank 索引, 导致下游 IO 写错 shard
     - verify_metadata 在 sample_from_nodes 调用, 但 sample_from_edges 没有调用 verify_metadata;
       用户传入非法 metadata 到 sample_from_edges 时不报错, 静默传播到 GPU 操作
  3. 系统角度安全:
     - metadata 通过 sampling_results_dict.update() 直接注入结果字典后透传到 BufferedSampleReader;
       若 metadata value 是恶意构造的大 tuple/string, 会撑爆 IO buffer
     - 跨进程分布式场景下 metadata 不做 broadcast: rank 0 传入的 metadata 与其他 rank 的 metadata
       可能不一致, 导致 heterogeneous 图的 edge_type 元信息在各 rank 上不同步

Walpurgis 改写20%（鲁迅拿法）:
  - verify_metadata 上游用 assert, 改写为 raise TypeError/ValueError, 生产安全
  - MetadataBundle 值对象替代裸 dict, 封装 key-collision 检测 (解决 Knuth审查 BUG #1)
  - safe_inject_metadata(results, bundle) 替代 results.update(metadata),
    碰撞时 raise MetadataCollisionError 而非静默覆盖
  - MetadataValidator 替代 verify_metadata 函数, 面向对象, 可配置严格/宽松模式
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印
    - 每次 MetadataValidator.validate 打印 key/value 类型
    - 每次 safe_inject_metadata 打印 collision 检测结果
    - minibatch_dict 推导式 str/tuple 绕过路径打印

作者: dylanyunlon<dogechat@163.com>
"""
import sys
import os
from typing import Dict, Optional, Tuple, Union

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'

# ─── RESERVED_KEYS: 上游 sampling_results_dict 内置键 ──────────────────────────
# 来源: distributed_sampler.py DistributedNeighborSampler.__sample 末尾:
#   sampling_results_dict["fanout"] = cupy.array(self.__fanout, dtype="int32")
#   sampling_results_dict["rank"]   = rank
# 以及 __sample_from_nodes_impl / __buffered_sample_from_nodes_impl 中:
#   minibatch_dict["input_index"] = current_ix.cuda()
#   minibatch_dict["map"] = minibatch_dict["renumber_map"]
#   (还有 label_type_hop_offsets, edge_renumber_map 等采样内置键)
# Knuth审查(系统角度): metadata 不得覆盖这些键
RESERVED_KEYS = frozenset({
    "fanout",
    "rank",
    "input_index",
    "input_label",
    "map",
    "renumber_map",
    "label_type_hop_offsets",
    "edge_renumber_map",
    "edge_renumber_map_offsets",
})


def _dbg_meta(tag: str, msg: str) -> None:
    """断点调试: metadata传播专用print"""
    if _DBG:
        print(f"[DEBUG 2ba9979 {tag}] {msg}", file=sys.stderr, flush=True)


# ─── MetadataCollisionError: 上游无此错误类型 ───────────────────────────────
# Knuth审查 用户角度 BUG: update()静默覆盖 reserved key
# 改写: 显式异常, 替代静默覆盖
class MetadataCollisionError(ValueError):
    """
    Raised when metadata keys collide with reserved sampling result keys.
    上游 sampling_results_dict.update(metadata) 静默覆盖, 我们改写为显式报错.
    """
    pass


# ─── MetadataValidator: 对应 sampler_utils.verify_metadata ─────────────────
# Python (2ba9979 sampler_utils.py:31-47):
#   def verify_metadata(metadata):
#       if metadata is not None:
#           for k, v in metadata.items():
#               assert isinstance(k, str), "Metadata keys must be strings."
#               if isinstance(v, tuple):
#                   assert len(v) == 3, "Metadata tuples must be of length 3."
#                   assert isinstance(v[0], str), "Metadata tuple must be of type (str,str,str)."
#                   assert isinstance(v[1], str), ...
#                   assert isinstance(v[2], str), ...
#               else:
#                   assert isinstance(v, str), "Metadata values must be strings or tuples of strings."
#
# 改写:
#   1. assert → raise TypeError/ValueError (python -O 不跳过)
#   2. 面向对象: strict_mode 控制是否检查 reserved key 碰撞
#   3. validate_value 单独方法, 便于测试
class MetadataValidator:
    """
    对应 2ba9979 sampler_utils.verify_metadata.

    改写: assert → raise; 加 strict_mode 检查 reserved key 碰撞;
    面向对象替代模块级函数, 状态可检查.
    """

    def __init__(self, strict_mode: bool = True):
        """
        strict_mode=True: 额外检查 metadata key 不碰撞 RESERVED_KEYS.
        strict_mode=False: 仅做类型校验 (与上游 verify_metadata 等价).
        """
        self.strict_mode = strict_mode
        self._last_validated: Optional[Dict] = None

        _dbg_meta(
            "MetadataValidator.__init__",
            f"strict_mode={strict_mode}"
        )

    def validate_key(self, k: object) -> None:
        """
        Python: assert isinstance(k, str), "Metadata keys must be strings."
        改写: raise TypeError
        """
        if not isinstance(k, str):
            _dbg_meta(
                "MetadataValidator.validate_key",
                f"FAIL: key={k!r} type={type(k).__name__} (must be str)"
            )
            raise TypeError(
                f"Metadata keys must be strings. Got {type(k).__name__!r}: {k!r}"
            )

        if self.strict_mode and k in RESERVED_KEYS:
            _dbg_meta(
                "MetadataValidator.validate_key",
                f"COLLISION: key='{k}' is a reserved sampling result key"
            )
            raise MetadataCollisionError(
                f"Metadata key '{k}' collides with reserved sampling result key. "
                f"Reserved keys: {sorted(RESERVED_KEYS)}. "
                f"(Knuth审查: 上游 update() 会静默覆盖采样数据, 已改写为显式报错)"
            )

        _dbg_meta("MetadataValidator.validate_key", f"OK: key='{k}'")

    def validate_value(self, k: str, v: object) -> None:
        """
        Python:
            if isinstance(v, tuple):
                assert len(v) == 3
                assert isinstance(v[0], str) and isinstance(v[1], str) and isinstance(v[2], str)
            else:
                assert isinstance(v, str)
        改写: raise ValueError/TypeError
        """
        if isinstance(v, tuple):
            if len(v) != 3:
                _dbg_meta(
                    "MetadataValidator.validate_value",
                    f"FAIL: key='{k}' tuple len={len(v)} (must be 3)"
                )
                raise ValueError(
                    f"Metadata tuples must be of length 3. "
                    f"Key '{k}' has tuple of length {len(v)}: {v!r}"
                )
            for i, elem in enumerate(v):
                if not isinstance(elem, str):
                    _dbg_meta(
                        "MetadataValidator.validate_value",
                        f"FAIL: key='{k}' tuple[{i}]={elem!r} type={type(elem).__name__} (must be str)"
                    )
                    raise TypeError(
                        f"Metadata tuple must be of type (str, str, str). "
                        f"Key '{k}', index {i}: got {type(elem).__name__!r}: {elem!r}"
                    )
            _dbg_meta(
                "MetadataValidator.validate_value",
                f"OK: key='{k}' value=tuple{v!r}"
            )
        else:
            if not isinstance(v, str):
                _dbg_meta(
                    "MetadataValidator.validate_value",
                    f"FAIL: key='{k}' value={v!r} type={type(v).__name__} (must be str or tuple)"
                )
                raise TypeError(
                    f"Metadata values must be strings or tuples of strings. "
                    f"Key '{k}': got {type(v).__name__!r}: {v!r}"
                )
            _dbg_meta(
                "MetadataValidator.validate_value",
                f"OK: key='{k}' value={v!r}"
            )

    def validate(
        self,
        metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]
    ) -> None:
        """
        对应 2ba9979 sampler_utils.verify_metadata(metadata).

        Python: 顶层 if metadata is not None + for k, v in metadata.items() + asserts
        改写: 调用 validate_key / validate_value, raise 替代 assert

        断点调试: 打印每个 key-value 对类型
        """
        if metadata is None:
            _dbg_meta("MetadataValidator.validate", "metadata=None, skip validation")
            return

        _dbg_meta(
            "MetadataValidator.validate",
            f"validating {len(metadata)} key(s): {list(metadata.keys())}"
        )

        for k, v in metadata.items():
            self.validate_key(k)
            self.validate_value(k, v)

        self._last_validated = metadata
        _dbg_meta(
            "MetadataValidator.validate",
            f"ALL OK: {len(metadata)} key(s) validated"
        )


# ─── 模块级单例: 对应 sampler_utils.verify_metadata 直接调用 ─────────────────
# 宽松模式 (strict_mode=False) 与上游 assert 语义等价
_default_validator = MetadataValidator(strict_mode=False)
_strict_validator = MetadataValidator(strict_mode=True)


def verify_metadata(
    metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]
) -> None:
    """
    对应 2ba9979 sampler_utils.verify_metadata.
    drop-in 替代: 签名相同, 改写 assert→raise.

    在 sample_from_nodes / sample_from_edges 入口调用.
    断点调试: WALPURGIS_DEBUG=1 时打印校验明细.

    Knuth审查(用户角度): 上游只在 sample_from_nodes 调用此函数,
    sample_from_edges 未调用; 此处统一封装, 调用方自行决定
    """
    _dbg_meta("verify_metadata", f"called with metadata={metadata!r}")
    _default_validator.validate(metadata)


def verify_metadata_strict(
    metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]
) -> None:
    """
    严格版本: 额外检查 metadata key 不碰撞 RESERVED_KEYS.
    解决 Knuth审查 BUG: metadata={"rank": ...} 静默覆盖采样rank.
    生产环境建议用此版本.
    """
    _dbg_meta("verify_metadata_strict", f"called with metadata={metadata!r}")
    _strict_validator.validate(metadata)


# ─── MetadataBundle: 替代裸 dict 的值对象 ─────────────────────────────────
# Python (2ba9979): metadata 是裸 Optional[Dict[str, Union[str, Tuple[str,str,str]]]]
# 改写: 封装为值对象, __post_init__ 自动校验, 避免未校验的 metadata 传入采样链
class MetadataBundle:
    """
    值对象, 封装已校验的 metadata dict.

    上游 metadata 是裸 dict, 任何地方都可以传未校验的 dict;
    改写: MetadataBundle 构建时即校验, 类型安全.

    对应2ba9979的完整 metadata 传播路径:
      sample_from_nodes(metadata=...) → verify_metadata → __sample → results.update(metadata)
    改写路径:
      sample_from_nodes(metadata=...) → MetadataBundle(metadata) → safe_inject_metadata
    """

    def __init__(
        self,
        raw: Optional[Dict[str, Union[str, Tuple[str, str, str]]]],
        strict: bool = False,
    ):
        """
        raw=None → empty bundle (透传 None)
        strict=True → 额外检查 RESERVED_KEYS 碰撞
        """
        self._data: Dict[str, Union[str, Tuple[str, str, str]]] = {}
        self._strict = strict

        if raw is not None:
            validator = _strict_validator if strict else _default_validator
            validator.validate(raw)
            self._data = dict(raw)  # 浅拷贝, 隔离外部修改

        _dbg_meta(
            "MetadataBundle.__init__",
            f"bundle created: {len(self._data)} entries strict={strict}"
        )

    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0

    def items(self):
        return self._data.items()

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"MetadataBundle({self._data!r})"


# ─── safe_inject_metadata: 替代 sampling_results_dict.update(metadata) ────
# Python (2ba9979 distributed_sampler.py:774-775):
#   if metadata is not None:
#       sampling_results_dict.update(metadata)
#
# 改写: 碰撞检测 + 断点调试
def safe_inject_metadata(
    results: dict,
    metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]],
    collision_mode: str = "raise",  # "raise" | "warn" | "skip"
) -> None:
    """
    对应 2ba9979 distributed_sampler.py:
        if metadata is not None:
            sampling_results_dict.update(metadata)

    改写: 碰撞检测, 防止 metadata 覆盖 reserved 采样键.

    collision_mode:
        "raise" — MetadataCollisionError (推荐生产)
        "warn"  — 打印警告后跳过碰撞键
        "skip"  — 静默跳过碰撞键 (等价上游行为, 不推荐)

    断点调试: 每次注入打印 collision 检测结果
    """
    if metadata is None:
        _dbg_meta("safe_inject_metadata", "metadata=None, nothing to inject")
        return

    _dbg_meta(
        "safe_inject_metadata",
        f"injecting {len(metadata)} keys into results (collision_mode={collision_mode!r})"
    )

    for k, v in metadata.items():
        if k in results:
            # 碰撞: metadata key 已存在于采样结果中
            _dbg_meta(
                "safe_inject_metadata",
                f"COLLISION: key='{k}' existing_value={results[k]!r} "
                f"metadata_value={v!r} mode={collision_mode!r}"
            )
            if collision_mode == "raise":
                raise MetadataCollisionError(
                    f"Metadata key '{k}' collides with existing sampling result. "
                    f"Existing: {results[k]!r}. Metadata: {v!r}. "
                    f"(Knuth审查: 上游 update() 静默覆盖, 已改写为显式报错)"
                )
            elif collision_mode == "warn":
                print(
                    f"[WARN 2ba9979 safe_inject_metadata] "
                    f"Metadata key '{k}' collides with sampling result. Skipping.",
                    file=sys.stderr
                )
                continue
            else:  # "skip"
                continue

        results[k] = v
        _dbg_meta("safe_inject_metadata", f"injected: '{k}' = {v!r}")

    _dbg_meta("safe_inject_metadata", f"done. results now has {len(results)} keys")


# ─── minibatch_coerce: 对应 minibatch_dict 推导式修复 ──────────────────────
# Python (2ba9979 distributed_sampler.py, 两处推导式):
#   旧: k: torch.as_tensor(v, device="cuda")
#   新: k: v if isinstance(v, (str, tuple)) else torch.as_tensor(v, device="cuda")
#
# 改写: 提取为函数, 断点调试打印 str/tuple 绕过路径
def minibatch_coerce_value(k: str, v: object) -> object:
    """
    对应 2ba9979 minibatch_dict 字典推导式修复.

    旧行为: 所有 value 都调用 torch.as_tensor(v, device="cuda"),
            str/tuple 值 (来自 metadata) 会崩溃:
              TypeError: can't convert str to tensor
    新行为: str/tuple 直接透传, 其他值走 torch.as_tensor

    改写: 提取为独立函数, 加断点调试; 上游是内联推导式, 难以测试
    """
    if isinstance(v, (str, tuple)):
        _dbg_meta(
            "minibatch_coerce_value",
            f"key='{k}' type={type(v).__name__} → passthrough (str/tuple bypass, 2ba9979 fix)"
        )
        return v  # 2ba9979 修复: str/tuple 来自 metadata, 不做 as_tensor

    # 非 str/tuple: 走 GPU tensor 转换
    # 注意: 实际 torch 调用需要 torch 可用; 此处仅封装逻辑, 不引入 torch 依赖
    _dbg_meta(
        "minibatch_coerce_value",
        f"key='{k}' type={type(v).__name__} → torch.as_tensor (normal path)"
    )
    return v  # caller 负责调用 torch.as_tensor; 此处只做类型路由


def is_metadata_passthrough(v: object) -> bool:
    """
    判断 minibatch_dict 中某个值是否需要跳过 torch.as_tensor.
    对应 2ba9979 fix: isinstance(v, (str, tuple))

    提取为独立断言函数, 调用方:
        if is_metadata_passthrough(v):
            result[k] = v
        else:
            result[k] = torch.as_tensor(v, device="cuda")
    """
    return isinstance(v, (str, tuple))
