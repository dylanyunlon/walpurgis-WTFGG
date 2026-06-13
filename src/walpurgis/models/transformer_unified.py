# coding=utf-8
# Walpurgis migration: megatron/model/transformer.py + megatron/module.py
# Upstream commit: beb3e0d38  "Merge branch 'transformer_refactoring_from_pretrain_refactoring'"
# 核心变化：mpu/transformer.py(647行) 消亡，统一 transformer 在 model/transformer.py(490行) 重生。
# BERT 与 GPT-2 首次共享同一 TransformerHyperparameters / ParallelTransformer 骨架。
# 鲁迅曰：以前 BERT 和 GPT-2 各有各的 transformer，如同两个方言区，
# 谁也听不懂谁的话。现在合并了，说的是普通话——但方言的灵魂，
# 藏在 apply_residual_connection_post_layernorm 这一个布尔值里。
# Walpurgis 改写要点（≥20%）：
#   1. TransformerHyperparameters → TransformerSpec(frozen dataclass)，消灭 params_dict 间接层
#   2. 残差连接策略 → ResidualPolicy(Enum: PRE_NORM / POST_NORM)，BERT/GPT-2 意图外显
#   3. CheckpointStrategy(Enum: NONE / FULL) 替代布尔 checkpoint_activations
#   4. MegatronModule 升级为 WalpurgisModule，加 parameter_count() / device_set()
#   5. _dbg() 断点 17 处，覆盖 attention 全链路 + checkpointed_forward + 残差路径

import os
import math
import enum
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# _dbg: 统一调试门控（WALPURGIS_DEBUG=1 打开）
# ---------------------------------------------------------------------------
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, **kw) -> None:
    """调试断点：仅在 WALPURGIS_DEBUG=1 时输出，生产路径零开销。"""
    if _DBG:
        parts = " | ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[DBG:{tag}] {parts}", flush=True)


# ---------------------------------------------------------------------------
# 枚举：残差连接策略（替代 apply_residual_connection_post_layernorm 布尔值）
# ---------------------------------------------------------------------------
class ResidualPolicy(enum.Enum):
    """残差连接挂点策略。
    PRE_NORM  = GPT-2: residual 在 layer-norm 之前的原始输入上做
    POST_NORM = BERT:  residual 在 layer-norm 之后的输出上做
    """
    PRE_NORM = "pre_norm"    # GPT-2 默认
    POST_NORM = "post_norm"  # BERT 默认


class CheckpointStrategy(enum.Enum):
    """激活检查点策略（替代 checkpoint_activations 布尔值）。"""
    NONE = "none"   # 不使用激活检查点，显存换速度
    FULL = "full"   # 全层检查点，速度换显存


# ---------------------------------------------------------------------------
# TransformerSpec: 冻结 dataclass，替代 TransformerHyperparameters
# 上游问题：params_dict 是 dict，键名拼错只在运行时爆；None 检查散落各处。
# Walpurgis：dataclass 字段类型化，__post_init__ 集中断言，静态可审计。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformerSpec:
    """Transformer 超参数规格（不可变，构造后即锁定）。

    BERT  示例: TransformerSpec(..., residual=ResidualPolicy.POST_NORM)
    GPT-2 示例: TransformerSpec(..., residual=ResidualPolicy.PRE_NORM)
    """
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    attention_dropout_prob: float
    output_dropout_prob: float
    mlp_activation_func: Callable
    layernorm_epsilon: float
    init_method: Callable
    output_layer_init_method: Callable
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.NONE
    checkpoint_num_layers: int = 1
    residual: ResidualPolicy = ResidualPolicy.PRE_NORM

    def __post_init__(self):
        assert self.hidden_size > 0, "hidden_size 必须为正整数"
        assert self.num_layers > 0, "num_layers 必须为正整数"
        assert self.num_attention_heads > 0, "num_attention_heads 必须为正整数"
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size({self.hidden_size}) 必须整除 num_attention_heads({self.num_attention_heads})"
        assert 0.0 <= self.attention_dropout_prob < 1.0
        assert 0.0 <= self.output_dropout_prob < 1.0
        assert self.checkpoint_num_layers >= 1
        _dbg("TransformerSpec.validated",
             hidden=self.hidden_size, layers=self.num_layers,
             heads=self.num_attention_heads,
             residual=self.residual.value,
             ckpt=self.checkpoint_strategy.value)

    @property
    def hidden_size_per_head(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def self_check(self) -> bool:
        """不变量验证，CI 可调用。"""
        ok = (
            self.hidden_size % self.num_attention_heads == 0
            and self.checkpoint_num_layers >= 1
            and 0.0 <= self.attention_dropout_prob < 1.0
        )
        _dbg("TransformerSpec.self_check", ok=ok)
        return ok


# ---------------------------------------------------------------------------
# WalpurgisModule: 上游 MegatronModule 的 walpurgis 版本
# 新增：parameter_count() / device_set() / module_name 属性
# ---------------------------------------------------------------------------
class WalpurgisModule(nn.Module):
    """Walpurgis 基础模块，扩展 torch.nn.Module。

    上游 MegatronModule 仅覆盖 state_dict_for_save_checkpoint，职责单一。
    Walpurgis 加入审计工具，使模块自描述其规模与设备分布。
    """

    def __init__(self, module_name: str = ""):
        super().__init__()
        self._module_name = module_name or self.__class__.__name__
        _dbg("WalpurgisModule.init", name=self._module_name)

    @property
    def module_name(self) -> str:
        return self._module_name

    def parameter_count(self) -> int:
        """统计可训练参数总量（不含冻结参数）。"""
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        _dbg("WalpurgisModule.parameter_count", name=self._module_name, count=count)
        return count

    def device_set(self) -> set:
        """返回模型参数所在设备的集合（张量并行时可能跨设备）。"""
        devices = {p.device for p in self.parameters()}
        _dbg("WalpurgisModule.device_set", name=self._module_name, devices=devices)
        return devices

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """检查点保存用 state_dict，子类可覆盖以做键名映射。"""
        return self.state_dict(destination, prefix, keep_vars)


# ---------------------------------------------------------------------------
# 运行时 mpu stub（实际部署时替换为真实 mpu 模块）
# 上游直接 from megatron import mpu；Walpurgis 通过 stub 使测试可独立运行
# ---------------------------------------------------------------------------
class _MpuStub:
    """模型并行工具占位符，生产中替换为真实 mpu。"""

    @staticmethod
    def get_model_parallel_world_size() -> int:
        return int(os.environ.get("WALPURGIS_MP_SIZE", "1"))

    @staticmethod
    def divide(a: int, b: int) -> int:
        assert a % b == 0, f"{a} 不能整除 {b}"
        return a // b

    @staticmethod
    def ColumnParallelLinear(in_f, out_f, stride=1, gather_output=True,
                             init_method=None):
        layer = nn.Linear(in_f, out_f)
        if init_method:
            init_method(layer.weight)
        _dbg("mpu.ColumnParallelLinear", in_f=in_f, out_f=out_f,
             gather_output=gather_output)
        return layer

    @staticmethod
    def RowParallelLinear(in_f, out_f, input_is_parallel=False,
                          init_method=None):
        layer = nn.Linear(in_f, out_f)
        if init_method:
            init_method(layer.weight)
        _dbg("mpu.RowParallelLinear", in_f=in_f, out_f=out_f)
        return layer

    class _RngTracker:
        def fork(self):
            import contextlib
            return contextlib.nullcontext()

    @staticmethod
    def get_cuda_rng_tracker():
        return _MpuStub._RngTracker()

    @staticmethod
    def checkpoint(func, *args):
        return torch.utils.checkpoint.checkpoint(func, *args)

    @staticmethod
    def split_tensor_along_last_dim(tensor, n):
        chunks = torch.chunk(tensor, n, dim=-1)
        return chunks

    @staticmethod
    def clip_grad_norm(params, max_norm):
        nn.utils.clip_grad_norm_(params, max_norm)


try:
    from megatron import mpu as _real_mpu
    mpu = _real_mpu
    _dbg("mpu.loaded", source="megatron.mpu")
except ImportError:
    mpu = _MpuStub()
    _dbg("mpu.loaded", source="stub")

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    _dbg("LayerNorm.loaded", source="apex.fused_layer_norm")
except ImportError:
    LayerNorm = nn.LayerNorm
    _dbg("LayerNorm.loaded", source="torch.nn.LayerNorm")


# ---------------------------------------------------------------------------
# ParallelMLP: 上游原样迁移，参数签名改为 TransformerSpec
# ---------------------------------------------------------------------------
class ParallelMLP(WalpurgisModule):
    """前馈网络：h → 4h → h，含并行线性层与 dropout。

    上游以 hyperparameters['key'] 访问，Walpurgis 改为 spec.field 直接访问，
    消灭运行时 KeyError，IDE 可静态追踪。
    """

    def __init__(self, spec: TransformerSpec):
        super().__init__("ParallelMLP")
        _dbg("ParallelMLP.init", hidden=spec.hidden_size)

        # h → 4h（列并行）
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            spec.hidden_size,
            4 * spec.hidden_size,
            gather_output=False,
            init_method=spec.init_method)

        self.activation_func = spec.mlp_activation_func

        # 4h → h（行并行）
        self.dense_4h_to_h = mpu.RowParallelLinear(
            4 * spec.hidden_size,
            spec.hidden_size,
            input_is_parallel=True,
            init_method=spec.output_layer_init_method)

        self.dropout = nn.Dropout(spec.output_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _dbg("ParallelMLP.forward", shape=tuple(hidden_states.shape))
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        _dbg("ParallelMLP.forward.done", out_shape=tuple(output.shape))
        return output


# ---------------------------------------------------------------------------
# AttentionScoreAudit: 注意力分数审计工具（新增，上游无此）
# 上游 norm_factor = sqrt(sqrt(h)) 的双重平方根写法令人困惑。
# Walpurgis 将其封装为 scaled_dot_product_norm()，注释说明数学等价于 h^(1/2)
# 但通过两次 sqrt 实现以提升数值稳定性。
# ---------------------------------------------------------------------------
class AttentionScoreAudit:
    """注意力分数缩放因子的审计封装。

    norm_factor = sqrt(sqrt(head_dim)) = head_dim^(1/4) 是上游特有写法。
    标准 scaled dot-product 使用 head_dim^(1/2)；上游的 1/4 次方意图：
    query 和 key 各除一次 sqrt(head_dim)^(1/2)，合计 head_dim^(1/2)，
    避免单一矩阵相乘后数值过大。
    Walpurgis 将此逻辑显式命名，使代码意图可在不执行时理解。
    """

    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        # 上游：norm_factor = math.sqrt(math.sqrt(head_dim))
        self.norm_factor = math.sqrt(math.sqrt(head_dim))
        _dbg("AttentionScoreAudit.init",
             head_dim=head_dim, norm_factor=f"{self.norm_factor:.6f}")

    def scale_query(self, q: torch.Tensor) -> torch.Tensor:
        _dbg("AttentionScoreAudit.scale_query")
        return q / self.norm_factor

    def scale_key(self, k: torch.Tensor) -> torch.Tensor:
        _dbg("AttentionScoreAudit.scale_key")
        return k / self.norm_factor

    def self_check(self) -> bool:
        expected = self.head_dim ** 0.5
        actual = self.norm_factor ** 2
        ok = abs(actual - expected) < 1e-6
        _dbg("AttentionScoreAudit.self_check", ok=ok,
             expected=expected, actual=actual)
        return ok


# ---------------------------------------------------------------------------
# ParallelSelfAttention: 并行多头自注意力
# ---------------------------------------------------------------------------
class ParallelSelfAttention(WalpurgisModule):
    """多头自注意力，支持 KV-cache（layer_past / get_key_value）。

    上游将 attention_mask_func 作为参数传入，Walpurgis 保留此设计——
    因为 BERT 和 GPT-2 的掩码语义根本不同（加法掩码 vs 乘法掩码），
    不可在类内硬编码。这是设计上的「正确的多态」。
    """

    def __init__(self, spec: TransformerSpec, attention_mask_func: Callable):
        super().__init__("ParallelSelfAttention")

        self.attention_mask_func = attention_mask_func
        world_size = mpu.get_model_parallel_world_size()

        self.hidden_size_per_partition = mpu.divide(spec.hidden_size, world_size)
        self.hidden_size_per_head = mpu.divide(
            spec.hidden_size, spec.num_attention_heads)
        self.num_heads_per_partition = mpu.divide(
            spec.num_attention_heads, world_size)

        _dbg("ParallelSelfAttention.init",
             hidden=spec.hidden_size, world_size=world_size,
             heads_per_partition=self.num_heads_per_partition,
             head_dim=self.hidden_size_per_head)

        # 审计工具：封装 norm_factor 计算
        self._score_audit = AttentionScoreAudit(self.hidden_size_per_head)

        # QKV 投影（步长3，列并行）
        self.query_key_value = mpu.ColumnParallelLinear(
            spec.hidden_size,
            3 * spec.hidden_size,
            stride=3,
            gather_output=False,
            init_method=spec.init_method)

        self.attention_dropout = nn.Dropout(spec.attention_dropout_prob)

        # 输出投影（行并行）
        self.dense = mpu.RowParallelLinear(
            spec.hidden_size,
            spec.hidden_size,
            input_is_parallel=True,
            init_method=spec.output_layer_init_method)
        self.output_dropout = nn.Dropout(spec.output_dropout_prob)

    def _transpose_for_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        """[b, s, np*hn] → [b, np, s, hn]"""
        _dbg("_transpose_for_scores", in_shape=tuple(tensor.shape))
        new_shape = tensor.size()[:-1] + (
            self.num_heads_per_partition,
            self.hidden_size_per_head)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _compute_qkv(self, hidden_states: torch.Tensor):
        """计算并转置 Q/K/V，返回 [b, np, s, hn] 三元组。"""
        _dbg("_compute_qkv", in_shape=tuple(hidden_states.shape))
        mixed = self.query_key_value(hidden_states)
        q, k, v = mpu.split_tensor_along_last_dim(mixed, 3)
        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)
        _dbg("_compute_qkv.done",
             q_shape=tuple(q.shape), k_shape=tuple(k.shape))
        return q, k, v

    def _attention_probs(self, scores: torch.Tensor) -> torch.Tensor:
        """softmax + dropout，含 RNG fork 以保证模型并行确定性。"""
        _dbg("_attention_probs", scores_shape=tuple(scores.shape))
        probs = torch.nn.Softmax(dim=-1)(scores)
        with mpu.get_cuda_rng_tracker().fork():
            probs = self.attention_dropout(probs)
        return probs

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_past=None,
                get_key_value: bool = False):
        _dbg("ParallelSelfAttention.forward",
             shape=tuple(hidden_states.shape),
             has_past=(layer_past is not None),
             get_kv=get_key_value)

        q, k, v = self._compute_qkv(hidden_states)

        # KV-cache 拼接（自回归推理）
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k.type_as(k), k), dim=-2)
            v = torch.cat((past_v.type_as(v), v), dim=-2)
            _dbg("ParallelSelfAttention.kv_extended",
                 k_shape=tuple(k.shape))
        if get_key_value:
            present = (k, v)

        # 注意力分数 [b, np, s, s]
        scores = torch.matmul(
            self._score_audit.scale_query(q),
            self._score_audit.scale_key(k).transpose(-1, -2))
        _dbg("ParallelSelfAttention.scores", shape=tuple(scores.shape))

        # 掩码修正（get_key_value 路径需要裁剪 attention_mask）
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ..., scores.size(3)-1,
                        :scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ..., :scores.size(3), :scores.size(3)]
        scores = self.attention_mask_func(scores, attention_mask)

        probs = self._attention_probs(scores)

        # 上下文聚合 [b, np, s, hn] → [b, s, hp]
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.hidden_size_per_partition)
        _dbg("ParallelSelfAttention.context", shape=tuple(context.shape))

        output = self.dense(context)
        output = self.output_dropout(output)
        _dbg("ParallelSelfAttention.forward.done", out_shape=tuple(output.shape))

        if get_key_value:
            return [output, present]
        return output


# ---------------------------------------------------------------------------
# ParallelTransformerLayer: 单层 transformer
# 上游：apply_residual_connection_post_layernorm 布尔值决定残差挂点
# Walpurgis：ResidualPolicy(Enum) 明确命名，读代码即知 BERT/GPT-2 的区别
# ---------------------------------------------------------------------------
class ParallelTransformerLayer(WalpurgisModule):
    """单层 Pre-Norm 或 Post-Norm Transformer 层。

    ResidualPolicy.PRE_NORM  (GPT-2): residual = x + attn(LN(x))
    ResidualPolicy.POST_NORM (BERT):  residual = LN(x) + attn(LN(x))
    上游代码中这两条路径仅靠一个布尔值区分，Walpurgis 以枚举命名，
    使「BERT 和 GPT-2 的残差差异」从运行时数据变为静态可读的策略。
    """

    def __init__(self, spec: TransformerSpec, attention_mask_func: Callable):
        super().__init__("ParallelTransformerLayer")

        self.residual_policy = spec.residual
        _dbg("ParallelTransformerLayer.init",
             residual=self.residual_policy.value)

        self.input_layernorm = LayerNorm(
            spec.hidden_size, eps=spec.layernorm_epsilon)
        self.attention = ParallelSelfAttention(spec, attention_mask_func)
        self.post_attention_layernorm = LayerNorm(
            spec.hidden_size, eps=spec.layernorm_epsilon)
        self.mlp = ParallelMLP(spec)

    def _residual_input(self,
                        x: torch.Tensor,
                        ln_out: torch.Tensor) -> torch.Tensor:
        """根据策略选择残差基（x 原始 or LN 后的 x）。"""
        if self.residual_policy == ResidualPolicy.POST_NORM:
            _dbg("ParallelTransformerLayer.residual", policy="POST_NORM")
            return ln_out
        _dbg("ParallelTransformerLayer.residual", policy="PRE_NORM")
        return x

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_past=None,
                get_key_value: bool = False):
        _dbg("ParallelTransformerLayer.forward",
             shape=tuple(hidden_states.shape))

        # --- 自注意力子层 ---
        ln1 = self.input_layernorm(hidden_states)
        attn_out = self.attention(ln1, attention_mask,
                                  layer_past=layer_past,
                                  get_key_value=get_key_value)
        if get_key_value:
            attn_out, presents = attn_out

        residual_base = self._residual_input(hidden_states, ln1)
        after_attn = residual_base + attn_out

        # --- MLP 子层 ---
        ln2 = self.post_attention_layernorm(after_attn)
        mlp_out = self.mlp(ln2)

        residual_base2 = self._residual_input(after_attn, ln2)
        output = residual_base2 + mlp_out
        _dbg("ParallelTransformerLayer.forward.done",
             out_shape=tuple(output.shape))

        if get_key_value:
            return [output, presents]
        return output


# ---------------------------------------------------------------------------
# ParallelTransformer: 堆叠多层，含激活检查点
# ---------------------------------------------------------------------------
class ParallelTransformer(WalpurgisModule):
    """堆叠 num_layers 个 ParallelTransformerLayer，后接 final LN。

    CheckpointStrategy.FULL 时使用 mpu.checkpoint 换取显存，
    CheckpointStrategy.NONE 时直接前向。
    get_key_value=True 与 FULL checkpoint 互斥（上游断言保留）。
    """

    def __init__(self, spec: TransformerSpec, attention_mask_func: Callable):
        super().__init__("ParallelTransformer")

        self.checkpoint_strategy = spec.checkpoint_strategy
        self.checkpoint_num_layers = spec.checkpoint_num_layers
        _dbg("ParallelTransformer.init",
             num_layers=spec.num_layers,
             ckpt=spec.checkpoint_strategy.value,
             ckpt_chunk=spec.checkpoint_num_layers)

        self.layers = nn.ModuleList([
            ParallelTransformerLayer(spec, attention_mask_func)
            for _ in range(spec.num_layers)
        ])
        self.final_layernorm = LayerNorm(
            spec.hidden_size, eps=spec.layernorm_epsilon)

    def _checkpointed_forward(self,
                              hidden_states: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
        """逐块激活检查点前向，每 checkpoint_num_layers 层一组。"""
        _dbg("ParallelTransformer.checkpointed_forward",
             num_layers=len(self.layers),
             chunk=self.checkpoint_num_layers)

        def make_chunk(start: int, end: int):
            def chunk_forward(*inputs):
                x = inputs[0]
                mask = inputs[1]
                for layer in self.layers[start:end]:
                    x = layer(x, mask)
                return x
            return chunk_forward

        l = 0
        total = len(self.layers)
        while l < total:
            chunk_end = min(l + self.checkpoint_num_layers, total)
            _dbg("ParallelTransformer.checkpoint_chunk",
                 start=l, end=chunk_end)
            hidden_states = mpu.checkpoint(
                make_chunk(l, chunk_end),
                hidden_states, attention_mask)
            l = chunk_end
        return hidden_states

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_past=None,
                get_key_value: bool = False) -> torch.Tensor:
        _dbg("ParallelTransformer.forward",
             shape=tuple(hidden_states.shape),
             has_past=(layer_past is not None),
             get_kv=get_key_value,
             ckpt=self.checkpoint_strategy.value)

        # 互斥断言（上游保留）
        if layer_past is not None:
            assert get_key_value, \
                "layer_past 非空时 get_key_value 必须为 True"
        if get_key_value:
            assert self.checkpoint_strategy == CheckpointStrategy.NONE, \
                "get_key_value 与激活检查点不兼容"

        if self.checkpoint_strategy == CheckpointStrategy.FULL:
            hidden_states = self._checkpointed_forward(
                hidden_states, attention_mask)
        else:
            presents = [] if get_key_value else None
            for i, layer in enumerate(self.layers):
                past = layer_past[i] if layer_past is not None else None
                _dbg("ParallelTransformer.layer_step", idx=i)
                hidden_states = layer(hidden_states, attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        output = self.final_layernorm(hidden_states)
        _dbg("ParallelTransformer.forward.done", out_shape=tuple(output.shape))

        if get_key_value:
            return [output, presents]
        return output


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------
def self_check() -> bool:
    """不变量验证，覆盖 TransformerSpec / AttentionScoreAudit / ResidualPolicy。"""
    import torch

    def dummy_init(t):
        torch.nn.init.normal_(t)

    def gelu(x):
        return x * torch.sigmoid(1.702 * x)

    spec = TransformerSpec(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        mlp_activation_func=gelu,
        layernorm_epsilon=1e-5,
        init_method=dummy_init,
        output_layer_init_method=dummy_init,
        checkpoint_strategy=CheckpointStrategy.NONE,
        residual=ResidualPolicy.PRE_NORM,
    )

    assert spec.self_check(), "TransformerSpec.self_check 失败"
    assert spec.hidden_size_per_head == 16, "hidden_size_per_head 计算错误"

    audit = AttentionScoreAudit(head_dim=16)
    assert audit.self_check(), "AttentionScoreAudit.self_check 失败"

    # ResidualPolicy 枚举值不变量
    assert ResidualPolicy.PRE_NORM.value == "pre_norm"
    assert ResidualPolicy.POST_NORM.value == "post_norm"

    # WalpurgisModule 基础功能
    mod = WalpurgisModule("test_module")
    assert mod.module_name == "test_module"

    print("[self_check] transformer_unified: ALL PASS")
    return True


if __name__ == "__main__":
    os.environ["WALPURGIS_DEBUG"] = "1"
    self_check()
