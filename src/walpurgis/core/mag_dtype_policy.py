"""
mag_dtype_policy.py
迁移自 upstream 81b7074 ([FEA] Update MAG example to show fp16/bf16 support #464)

原上游：mag_lp_mnmg.py 新增 --dtype 命令行参数，将模型、特征存储、embedding
        全部切换到用户指定 dtype（float32/float16/bfloat16）。

改写：将「dtype 解析与验证」逻辑抽取为独立模块，
      并将 81b7074 中散落在示例脚本各处的 .to(dtype) 转换点建模为策略对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── 1. MagDtype：上游 _DTYPE_CHOICES 强类型化 ────────────────────────────────
class MagDtype(Enum):
    """
    上游 _DTYPE_CHOICES = ("float32", "float16", "bfloat16")
    此枚举使 dtype 选择在类型层面可见，避免字符串到 torch.dtype 的延迟转换失误。
    """
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @classmethod
    def from_str(cls, name: str) -> "MagDtype":
        for member in cls:
            if member.value == name:
                return member
        raise ValueError(
            f"不支持的 dtype: {name!r}。"
            f"可选值: {[m.value for m in cls]}"
        )

    @property
    def is_reduced_precision(self) -> bool:
        """fp16/bf16 为降精度，训练时可降显存但需注意数值稳定性"""
        return self in (MagDtype.FLOAT16, MagDtype.BFLOAT16)

    def describe_tradeoff(self) -> str:
        """返回该 dtype 的显存/精度权衡说明（供日志/文档使用）"""
        if self == MagDtype.FLOAT32:
            return "全精度：数值稳定，显存开销最大"
        if self == MagDtype.FLOAT16:
            return "半精度 FP16：显存减半，动态范围较窄，需梯度缩放"
        return "BF16：显存减半，动态范围与 FP32 相同，推荐训练默认值"


# ── 2. MagDtypeConversionMap：81b7074 各转换点的显式注册 ────────────────────
@dataclass(frozen=True)
class ConversionPoint:
    """
    上游在 mag_lp_mnmg.py 各处插入 .to(dtype)，
    此处将每个转换点命名化，便于审计和回溯。
    """
    location: str     # 代码位置描述
    tensor_name: str  # 被转换的张量/模块
    reason: str       # 转换原因


# 上游 81b7074 新增的所有 .to(dtype) 转换点（鲁迅刀法：每刀皆有命名）
MAG_CONVERSION_POINTS: tuple[ConversionPoint, ...] = (
    ConversionPoint(
        location="Classifier.__init__: DistributedEmbedding",
        tensor_name="embedding dtype 参数",
        reason="原硬编码 torch.float32，改为参数化以支持 bf16/fp16 embedding 训练",
    ),
    ConversionPoint(
        location="Classifier.forward: paper_lin 输入",
        tensor_name="batch['paper'].x",
        reason="x_paper = paper_lin(batch['paper'].x.to(w_dtype))：对齐 Linear 权重 dtype",
    ),
    ConversionPoint(
        location="Classifier.forward: author/institution/fos zeros",
        tensor_name="torch.zeros(..., dtype=x_paper.dtype)",
        reason="原硬编码 device='cuda'，改为从 x_paper 推断 device 和 dtype，消除显式 cuda 绑定",
    ),
    ConversionPoint(
        location="__main__: feature_store paper.x",
        tensor_name="data.x_dict['paper'].to(dtype)",
        reason="特征存储载入时即转换，避免训练循环中反复转换",
    ),
    ConversionPoint(
        location="__main__: betweeness centrality features",
        tensor_name="feature_store[etype.edge_type, 'x', None].to(dtype)",
        reason="边特征与节点特征保持一致 dtype，原变量名 dtype 与循环变量冲突（修复为 dst_type）",
    ),
    ConversionPoint(
        location="__main__: model.to(device, dtype)",
        tensor_name="model",
        reason="原 .to(device) 改为 .to(device, dtype)，参数化模型精度",
    ),
    ConversionPoint(
        location="__main__: output embedding tensor",
        tensor_name="local_x0, local_x1",
        reason="输出拼接前 .to(torch.float32)：cudf/parquet 暂不支持 bf16 写入",
    ),
)


# ── 3. MagDtypePolicy：聚合策略对象 ─────────────────────────────────────────
@dataclass(frozen=True)
class MagDtypePolicy:
    """
    聚合「训练 dtype」与「输出 dtype」的策略对象。
    上游 81b7074 将输出统一 .to(torch.float32) 写 parquet，
    此处显式建模该「训练精度 ≠ 输出精度」约定。

    上游默认值：--dtype bfloat16（推荐训练默认）
    """
    train_dtype: MagDtype = MagDtype.BFLOAT16
    output_dtype: MagDtype = MagDtype.FLOAT32  # cudf 写入固定 fp32

    def validate(self) -> None:
        """
        输出 dtype 须为 FLOAT32（cudf/parquet 当前约束）。
        若上游解除此限制，此处将明确失败而非静默通过。
        """
        if self.output_dtype != MagDtype.FLOAT32:
            raise ValueError(
                f"输出 dtype 须为 FLOAT32（cudf/parquet 写入约束），"
                f"当前设置: {self.output_dtype}"
            )

    def summary(self) -> str:
        return (
            f"训练: {self.train_dtype.value} "
            f"({self.train_dtype.describe_tradeoff()})\n"
            f"输出: {self.output_dtype.value} (cudf/parquet 写入约束)"
        )

    def conversion_points(self) -> tuple[ConversionPoint, ...]:
        """返回所有需要 .to(train_dtype) 的转换点"""
        return MAG_CONVERSION_POINTS


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # MagDtype
    assert MagDtype.from_str("bfloat16") == MagDtype.BFLOAT16
    assert MagDtype.from_str("float32") == MagDtype.FLOAT32
    assert MagDtype.FLOAT16.is_reduced_precision is True
    assert MagDtype.FLOAT32.is_reduced_precision is False
    assert "BF16" in MagDtype.BFLOAT16.describe_tradeoff()

    try:
        MagDtype.from_str("int8")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    # MagDtypePolicy
    policy = MagDtypePolicy()
    policy.validate()   # 默认值应通过
    assert "bfloat16" in policy.summary()
    assert len(policy.conversion_points()) == 7  # 上游 7 处转换点

    # ConversionPoint
    pts = policy.conversion_points()
    assert all(isinstance(p, ConversionPoint) for p in pts)
    assert any("paper_lin" in p.reason for p in pts)

    # 错误 output_dtype 应被拒绝
    bad_policy = MagDtypePolicy(
        train_dtype=MagDtype.BFLOAT16,
        output_dtype=MagDtype.FLOAT16
    )
    try:
        bad_policy.validate()
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    print("mag_dtype_policy.py 自测：9 项断言全部 PASS")
