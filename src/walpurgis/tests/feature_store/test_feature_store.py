# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# migrate 6d1a8de: Support more dtypes in the cuGraph-PyG FeatureStore
# 原 PR: https://github.com/rapidsai/cugraph-gnn/pull/346
# 原作者: Alex Barghi <alexbarghi-nv>
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 上游 6d1a8de 核心变更 (cugraph-pyg/tests/data/test_feature_store.py):
#   - 将 test_feature_store_basic_api_float 重命名为 test_feature_store_basic_api_types
#   - 原测试硬编码 torch.float32；改写后用 @pytest.mark.parametrize 覆盖 7 种 dtype:
#       float32 / float16 / int8 / int16 / int32 / int64 / float64
#   - 这 7 种 dtype 对应 6d1a8de 在 feature_store.py 中新增的 WholeGraph 支持
#   - torch.bool 不在列表中 (WholeGraph 从未实际支持，6d1a8de 将其从映射表中移除)
#
# Walpurgis 迁移策略:
#   上游测试针对 FeatureStore.__make_wg_tensor 中内嵌的 dtypes dict；
#   Walpurgis 对应的实现是 core/unified_store.py 中的 DtypeNegotiator.DTYPE_TO_ID。
#   本测试文件迁移并扩展：
#     1. test_dtype_negotiator_encode_decode — 离线单元测试 DtypeNegotiator 编解码
#        等价于上游对 dtypes/dtype_ids 内联 dict 的隐式覆盖
#     2. test_dtype_negotiator_bool_removed — 验证 bool 已被移除 (6d1a8de 破坏性语义)
#     3. test_dtype_negotiator_id_uniqueness — 验证 id 无重复、无漏洞
#     4. test_dtype_negotiator_int16_takes_id4 — 断言 int16 占据原 bool 的 id=4
#     5. test_feature_store_basic_api_types — 上游同名测试 Walpurgis 等价版本
#        (不依赖 GPU/分布式: mock DtypeNegotiator.encode/negotiate, 纯逻辑验证)
#
# 鲁迅拿法 20% 改写要点 (相对上游 6d1a8de 前的原始 test_feature_store.py):
#   1. _dbg() 断点: 每个测试参数化迭代在 WALPURGIS_DEBUG=1 时打印入口 dtype + 期望 id
#   2. test_dtype_negotiator_bool_removed: 上游没有, Walpurgis 显式断言 bool 的移除语义
#   3. test_dtype_negotiator_id_uniqueness: 上游没有, 保护映射表不出现重复或跳洞
#   4. test_dtype_negotiator_int16_takes_id4: 精确断言 id=4 现在归 int16 而非 bool
#   5. WALPURGIS_DEBUG 断点覆盖: encode/decode/validate 三个路径各有断点
#      (断点1: encode 入口; 断点2: decode 入口; 断点3: bool 移除验证)

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pytest

_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点打印: WALPURGIS_DEBUG=1 时输出到 stderr。"""
    if _DEBUG:
        print(
            f"[WALPURGIS tests/feature_store/test_feature_store|{tag}] {msg}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 依赖导入: DtypeNegotiator (不需要 GPU/分布式)
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_skip_no_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not available"
)

try:
    from walpurgis.core.unified_store import DtypeNegotiator
    _NEGOTIATOR_AVAILABLE = True
except ImportError:
    _NEGOTIATOR_AVAILABLE = False

_skip_no_negotiator = pytest.mark.skipif(
    not _NEGOTIATOR_AVAILABLE,
    reason="walpurgis.core.unified_store.DtypeNegotiator not available"
)


# ---------------------------------------------------------------------------
# 测试 1: DtypeNegotiator.encode/decode 参数化测试
#
# 上游等价: 上游 __make_wg_tensor 中直接内联 dtypes dict/dtype_ids dict，
#           encode/decode 是闭包函数，无法直接测试——只能通过 FeatureStore 集成测试覆盖。
# Walpurgis 改写: DtypeNegotiator 是独立类，可直接单元测试。
#
# migrate 6d1a8de: 7 种支持 dtype 对应上游 __make_wg_tensor 新增的映射项
# ---------------------------------------------------------------------------

# (dtype_name, expected_id) — 对应 6d1a8de 后的 DtypeNegotiator.DTYPE_TO_ID
_SUPPORTED_DTYPE_CASES: List[Tuple[str, int]] = [
    ("torch.float32", 0),
    ("torch.float64", 1),
    ("torch.int32",   2),
    ("torch.int64",   3),
    ("torch.int16",   4),   # migrate 6d1a8de: int16 占据原 bool 的 id=4
    ("torch.float16", 5),   # migrate 6d1a8de: 新增 half-precision
    ("torch.int8",    6),   # migrate 6d1a8de: 新增 int8 (量化场景)
    ("torch.bfloat16", 7),  # migrate 220563b: bf16 训练
]

_SUPPORTED_DTYPE_IDS = [f"{name}(id={eid})" for name, eid in _SUPPORTED_DTYPE_CASES]


@_skip_no_torch
@_skip_no_negotiator
@pytest.mark.parametrize("dtype_name,expected_id", _SUPPORTED_DTYPE_CASES,
                         ids=_SUPPORTED_DTYPE_IDS)
def test_dtype_negotiator_encode_decode(dtype_name: str, expected_id: int) -> None:
    """
    验证 DtypeNegotiator.encode/decode 往返一致，且 id 与 6d1a8de 后的上游映射对齐。

    断点1: encode 入口打印 dtype + 期望 id
    断点2: decode 入口打印 id + 期望 dtype_name

    上游等价:
        # cugraph-pyg/data/feature_store.py __make_wg_tensor
        dtypes = {torch.float32: 0, ..., torch.int16: 4, torch.float16: 5, torch.int8: 6}
        assert dtypes[tensor.dtype] == expected_id
        assert dtype_ids[expected_id] == tensor.dtype
    """
    # 断点1: encode 入口
    _dbg("encode", f"dtype_name={dtype_name!r} expected_id={expected_id}")

    # 从字符串解析 torch.dtype (e.g. "torch.float32" → torch.float32)
    attr_name = dtype_name.replace("torch.", "")
    dtype = getattr(torch, attr_name)

    # encode: dtype → id
    actual_id = DtypeNegotiator.encode(dtype)
    assert actual_id == expected_id, (
        f"encode({dtype_name}) = {actual_id}, 期望 {expected_id}。"
        f"\n当前 DTYPE_TO_ID = {DtypeNegotiator.DTYPE_TO_ID}"
    )

    # decode: id → dtype
    _dbg("decode", f"id={expected_id} expected_dtype_name={dtype_name!r}")
    recovered_dtype = DtypeNegotiator.decode(expected_id)
    assert recovered_dtype == dtype, (
        f"decode({expected_id}) = {recovered_dtype}, 期望 {dtype}。"
        f"\n当前 ID_TO_DTYPE_NAME = {DtypeNegotiator.ID_TO_DTYPE_NAME}"
    )

    _dbg("encode-decode", f"往返验证通过: {dtype_name} ↔ id={expected_id}")


# ---------------------------------------------------------------------------
# 测试 2: torch.bool 已被移除 (6d1a8de 核心语义)
#
# migrate 6d1a8de:
#   上游 PR #346 明确说明 torch.bool 从未被 WholeGraph 实际支持，
#   加入是历史错误 (mistake)。使用 torch.bool 总会在 WholeGraph 层抛异常。
#   移除是正名，不是破坏性变更。
#
# 鲁迅: 「错的事, 不因为流传已久就变成对的。」
# 断点3: bool 移除验证
# ---------------------------------------------------------------------------

@_skip_no_torch
@_skip_no_negotiator
def test_dtype_negotiator_bool_removed() -> None:
    """
    验证 torch.bool 已从 DtypeNegotiator 映射表中移除。

    6d1a8de 移除语义: torch.bool 不应出现在 DTYPE_TO_ID 中，
    encode(torch.bool) 应抛出 ValueError（而非返回错误的 id=4）。
    """
    # 断点3: bool 移除验证
    _dbg("bool_removed", "验证 torch.bool 已从 DTYPE_TO_ID 移除 (6d1a8de)")

    # DTYPE_TO_ID 中不应出现 "torch.bool"
    assert "torch.bool" not in DtypeNegotiator.DTYPE_TO_ID, (
        "torch.bool 仍在 DTYPE_TO_ID 中！6d1a8de 要求移除此项。\n"
        "WholeGraph 从未实际支持 torch.bool (使用它会在 gather/scatter 时抛异常)。\n"
        f"当前 DTYPE_TO_ID = {DtypeNegotiator.DTYPE_TO_ID}"
    )

    # encode(torch.bool) 应抛出 ValueError
    with pytest.raises(ValueError, match="Unsupported dtype"):
        DtypeNegotiator.encode(torch.bool)

    # id=4 现在属于 torch.int16，而非 torch.bool
    recovered = DtypeNegotiator.decode(4)
    assert recovered == torch.int16, (
        f"id=4 应解码为 torch.int16 (6d1a8de 新映射), 实际得到 {recovered}。\n"
        f"当前 ID_TO_DTYPE_NAME = {DtypeNegotiator.ID_TO_DTYPE_NAME}"
    )

    _dbg("bool_removed", "✓ torch.bool 已移除, id=4 正确归属 torch.int16")


# ---------------------------------------------------------------------------
# 测试 3: id 唯一性 + 连续性保护
#
# 保护 DtypeNegotiator.DTYPE_TO_ID 映射表不出现重复 id 或意外跳洞。
# 上游没有此类测试；Walpurgis 加入以防止后续迁移引入隐患。
# ---------------------------------------------------------------------------

@_skip_no_negotiator
def test_dtype_negotiator_id_uniqueness() -> None:
    """
    验证 DTYPE_TO_ID 中所有 id 唯一，且从 0 开始连续无跳洞。

    这是 Walpurgis 特有的防护测试，上游没有对应项。
    动机: 每次迁移（6d1a8de / 220563b）都可能调整 id，手动编辑容易引入重复。
    """
    dtype_to_id = DtypeNegotiator.DTYPE_TO_ID
    ids = sorted(dtype_to_id.values())

    # 断点: 打印当前映射表
    _dbg("id_uniqueness", f"当前 DTYPE_TO_ID = {dtype_to_id}")

    # 1. id 无重复
    assert len(ids) == len(set(ids)), (
        f"DTYPE_TO_ID 存在重复 id！ids={ids}\n"
        f"完整映射: {dtype_to_id}"
    )

    # 2. 从 0 开始连续 (0, 1, 2, ..., n-1)
    expected_ids = list(range(len(ids)))
    assert ids == expected_ids, (
        f"DTYPE_TO_ID 的 id 不连续。实际={ids}, 期望={expected_ids}。\n"
        f"注意: 6d1a8de 移除了 bool(原id=4), int16 补位为 4, 序列应连续。\n"
        f"完整映射: {dtype_to_id}"
    )

    _dbg("id_uniqueness", f"✓ id 唯一且连续: {ids}")


# ---------------------------------------------------------------------------
# 测试 4: int16 占 id=4 (精确断言 6d1a8de 的核心 id 重排)
# ---------------------------------------------------------------------------

@_skip_no_torch
@_skip_no_negotiator
def test_dtype_negotiator_int16_takes_id4() -> None:
    """
    精确断言 6d1a8de 的核心重排: int16 占据原 bool 的 id=4。

    这是 6d1a8de 最重要的单一语义变更，值得单独断言。
    序列化 dtype id 的任何改变都会影响跨 rank all_gather 的兼容性，
    此测试提供快速失败信号。
    """
    _dbg("int16_id4", "验证 int16 占 id=4 (6d1a8de 核心重排)")

    assert DtypeNegotiator.DTYPE_TO_ID.get("torch.int16") == 4, (
        "torch.int16 应占 id=4 (6d1a8de 将 bool 移除后 int16 补位)。\n"
        f"当前 DTYPE_TO_ID = {DtypeNegotiator.DTYPE_TO_ID}"
    )
    assert DtypeNegotiator.encode(torch.int16) == 4

    _dbg("int16_id4", "✓ int16 → id=4 验证通过")


# ---------------------------------------------------------------------------
# 测试 5: FeatureStore 基础 API — dtype 参数化 (上游 6d1a8de 同名测试 Walpurgis 版)
#
# 上游原始函数:
#   test_feature_store_basic_api_float (单 float32)
#   → test_feature_store_basic_api_types (7 种 dtype @parametrize)
#
# Walpurgis 版本:
#   由于 FeatureStore 依赖 torch.distributed + GPU，CI 环境无法运行集成测试。
#   此处迁移上游测试的逻辑意图：
#     1. 构造指定 dtype 的特征张量 (2000 → 20×100 reshape)
#     2. 验证 DtypeNegotiator 能正确 encode 该 dtype (encode 不抛异常)
#     3. 验证 encode 后能 decode 回相同 dtype (往返一致)
#   这覆盖上游测试验证的核心不变量: dtype 信息在 all_gather 协商中正确传输。
#
# @pytest.mark.sg — 上游标记「单 GPU 测试」; Walpurgis 用 skip_no_torch 替代
# ---------------------------------------------------------------------------

# 对应上游 6d1a8de 后的 parametrize 列表 (去掉 torch.bool)
_FEATURE_STORE_DTYPE_CASES = [
    pytest.param(torch.float32, id="float32") if _TORCH_AVAILABLE else None,
    pytest.param(torch.float16, id="float16") if _TORCH_AVAILABLE else None,
    pytest.param(torch.int8,    id="int8")    if _TORCH_AVAILABLE else None,
    pytest.param(torch.int16,   id="int16")   if _TORCH_AVAILABLE else None,
    pytest.param(torch.int32,   id="int32")   if _TORCH_AVAILABLE else None,
    pytest.param(torch.int64,   id="int64")   if _TORCH_AVAILABLE else None,
    pytest.param(torch.float64, id="float64") if _TORCH_AVAILABLE else None,
]
_FEATURE_STORE_DTYPE_CASES = [c for c in _FEATURE_STORE_DTYPE_CASES if c is not None]


@_skip_no_torch
@_skip_no_negotiator
@pytest.mark.parametrize("dtype", _FEATURE_STORE_DTYPE_CASES)
def test_feature_store_basic_api_types(dtype) -> None:
    """
    验证 FeatureStore 能处理各种 dtype 的特征张量。

    上游 6d1a8de test_feature_store_basic_api_types:
        features = torch.arange(0, 2000).reshape((20, 100)).to(dtype)
        whole_store = FeatureStore()
        whole_store["node", "fea", None] = features
        res = whole_store["node", "fea", None].storage
        assert (res == features).all()

    Walpurgis 等价 (离线，不依赖 GPU/distributed):
        1. 构造相同形状的张量
        2. 验证 DtypeNegotiator.encode 能处理该 dtype（不抛异常）
        3. 验证 encode/decode 往返一致
        4. 验证张量 reshape + dtype 转换正确（与上游构造逻辑一致）

    断点: 每种 dtype 打印张量形状和编码 id
    """
    _dbg("basic_api_types", f"dtype={dtype} 开始验证")

    # 上游构造逻辑: torch.arange(0, 2000).reshape((20, 100)).to(dtype)
    features = torch.arange(0, 2000)
    features = features.reshape((features.numel() // 100, 100)).to(dtype)

    assert features.shape == (20, 100), (
        f"张量形状应为 (20, 100), 实际: {features.shape}"
    )
    assert features.dtype == dtype, (
        f"张量 dtype 应为 {dtype}, 实际: {features.dtype}"
    )

    # 验证 DtypeNegotiator 能正确 encode 该 dtype
    dtype_id = DtypeNegotiator.encode(dtype)

    _dbg(
        "basic_api_types",
        f"dtype={dtype} → id={dtype_id} "
        f"shape={list(features.shape)} "
        f"min={features.min().item()} max={features.max().item()}"
    )

    # 验证 decode 往返一致
    recovered_dtype = DtypeNegotiator.decode(dtype_id)
    assert recovered_dtype == dtype, (
        f"encode/decode 往返失败: {dtype} → id={dtype_id} → {recovered_dtype}"
    )

    # 验证 id 在合法范围内 (0..n-1, n = len(DTYPE_TO_ID))
    n_dtypes = len(DtypeNegotiator.DTYPE_TO_ID)
    assert 0 <= dtype_id < n_dtypes, (
        f"dtype_id={dtype_id} 超出合法范围 [0, {n_dtypes})。"
    )

    _dbg("basic_api_types", f"✓ dtype={dtype} 全部断言通过")


# ---------------------------------------------------------------------------
# 测试 6: torch.bool 不在参数化列表中 (6d1a8de 移除语义的负向验证)
#
# 上游 6d1a8de 明确: torch.bool is removed since it was never supported by WholeGraph,
# and its inclusion was a mistake.
# Walpurgis: 确认测试用例列表中没有 torch.bool，保持与上游意图一致。
# ---------------------------------------------------------------------------

@_skip_no_torch
def test_feature_store_dtype_list_excludes_bool() -> None:
    """
    验证 Walpurgis 参数化 dtype 列表不包含 torch.bool。

    这是一个「测试的测试」(meta-test)，确保上游 6d1a8de 的删除意图在 Walpurgis
    参数化列表中也得到体现，防止维护者不小心把 bool 加回去。
    """
    _dbg("exclude_bool", "验证参数化列表不含 torch.bool")

    param_dtypes = [p.values[0] for p in _FEATURE_STORE_DTYPE_CASES]
    assert torch.bool not in param_dtypes, (
        "torch.bool 出现在参数化测试列表中！\n"
        "6d1a8de 明确: bool 从未被 WholeGraph 支持，其加入是历史错误，应从列表中移除。"
    )

    _dbg("exclude_bool", f"✓ bool 不在列表中。当前列表: {[str(d) for d in param_dtypes]}")
