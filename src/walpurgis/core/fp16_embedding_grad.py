"""
fp16_embedding_grad.py
迁移自 upstream 5909ae8 (Fp16 embedding train #462) + 662a6d9 (fix shm permission #463)

原上游：exchange_embeddings_nccl_func.cu — float-only DedupIndiceAndGradientsKernel
改写：将 CUDA C++ 泛型模板逻辑在 Python 层建模，同时收录 shm 安全权限规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── 1. GradDtype：强类型化上游 BF16_HALF_FLOAT dispatch 宏 ─────────────────────
class GradDtype(Enum):
    """
    上游通过 REGISTER_DISPATCH_TWO_TYPES(…, SINT3264, BF16_HALF_FLOAT) 扩展支持。
    原实现裸用 wholememory_dtype_t 整型，此处强类型化，
    消除调用层「float*」假设，使非 float32 路径在类型层面可见。
    """
    FLOAT32 = "float32"
    FLOAT16 = "float16"      # CUDA half
    BFLOAT16 = "bfloat16"   # CUDA __nv_bfloat16

    @classmethod
    def from_torch_dtype_name(cls, name: str) -> "GradDtype":
        """从 torch.dtype.__str__ 风格名称构造，e.g. 'torch.float16' → FLOAT16"""
        mapping = {
            "torch.float32": cls.FLOAT32,
            "float32": cls.FLOAT32,
            "torch.float16": cls.FLOAT16,
            "float16": cls.FLOAT16,
            "torch.bfloat16": cls.BFLOAT16,
            "bfloat16": cls.BFLOAT16,
        }
        try:
            return mapping[name]
        except KeyError:
            raise ValueError(
                f"不支持的梯度 dtype: {name!r}。"
                f"支持列表: {list(mapping.keys())}"
            )

    @property
    def is_reduced_precision(self) -> bool:
        """fp16 / bf16 均属降精度，dedup 路径须 static_cast<float> 累加"""
        return self in (GradDtype.FLOAT16, GradDtype.BFLOAT16)


# ── 2. DedupGradSpec：将上游 dedup_indice_and_gradients 签名变更建模 ──────────
@dataclass(frozen=True)
class DedupGradSpec:
    """
    上游 5909ae8 将 dedup_indice_and_gradients() 第三参数
    从 `const float* grads` 改为 `const void* grads`，
    同时 dedup_grads 输出固定为 WHOLEMEMORY_DT_FLOAT。

    此 spec 对象将该约定显式化，供 Python 层校验与文档化。

    字段含义（鲁迅刀法：每刀皆有命名）：
      grad_dtype    — 输入梯度实际精度（可以是 fp16/bf16/fp32）
      output_dtype  — dedup 后输出固定 float32（上游 embedding.cpp L243 修订）
      embedding_dim — embedding 向量维度，用于内存估算
    """
    grad_dtype: GradDtype
    output_dtype: GradDtype = GradDtype.FLOAT32   # 上游硬编码 WHOLEMEMORY_DT_FLOAT
    embedding_dim: int = 0

    def validate(self) -> None:
        """
        断言 output_dtype 固定为 FLOAT32（上游约定）。
        若将来上游改变此约定，此处会明确报错而非静默通过。
        """
        if self.output_dtype != GradDtype.FLOAT32:
            raise ValueError(
                f"dedup_grads 输出 dtype 必须为 FLOAT32，"
                f"当前: {self.output_dtype}（上游 embedding.cpp 硬编码）"
            )

    def cast_annotation(self) -> str:
        """
        返回用于日志/注释的 cast 说明字符串。
        上游 DedupIndiceAndGradientsKernel 中 static_cast<float>(current_grads_ptr[dim])。
        """
        if self.grad_dtype.is_reduced_precision:
            return (
                f"static_cast<float>({self.grad_dtype.value}) → accumulate in float32"
            )
        return "float32 → float32 (no-op cast)"


# ── 3. ShmPermissionRule：662a6d9 shm 安全权限规则建模 ────────────────────────
class ShmPermissionRule(Enum):
    """
    上游 662a6d9 将 shmget() 权限从 0644 → 0600。
    含义：组/其他用户本不应能 attach 同一 IPC shm segment，
    0644 允许组读取，存在跨用户访问 GPU 共享内存的安全漏洞。

    此枚举将「安全权限」vs「宽松权限」两种策略建模，
    便于在测试/文档层面验证正确性，而非散落于 C++ 魔术数字。
    """
    SECURE = 0o600      # owner rw only — 修复后值
    PERMISSIVE = 0o644  # owner rw, group/other r — 修复前值（存在安全风险）

    def is_secure(self) -> bool:
        return self == ShmPermissionRule.SECURE

    def describe(self) -> str:
        if self.is_secure():
            return (
                "0600: 仅 owner 可读写。"
                "防止同节点其他用户通过 IPC key 猜测 attach 同一 shm segment。"
                "（修复 #463，对应 global_mapped_host_wholememory_impl）"
            )
        return (
            "0644: owner rw + 组/其他 r。"
            "存在安全风险：同 UID 组内其他用户可 attach。"
        )


@dataclass
class ShmAllocationRecord:
    """
    记录一次 shmget 调用的参数与安全决策，便于审计。
    上游直接在 C++ 散落三处 shmget(…, 0600 | IPC_CREAT | IPC_EXCL) 调用，
    此处集中建模，消除重复魔术数字。
    """
    shm_key: int
    alloc_size: int
    permission: ShmPermissionRule
    flags: list[str] = field(default_factory=list)

    def validate_permission(self) -> None:
        if not self.permission.is_secure():
            raise ValueError(
                f"SHM 权限不安全: {oct(self.permission.value)}。"
                f"应使用 {oct(ShmPermissionRule.SECURE.value)}。"
                f"\n{self.permission.describe()}"
            )

    def as_shmget_mode(self) -> int:
        """返回 shmget 第三参数的数值（permission | flag bits）"""
        flag_map = {"IPC_CREAT": 0o1000, "IPC_EXCL": 0o2000}
        mode = self.permission.value
        for f in self.flags:
            mode |= flag_map.get(f, 0)
        return mode


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # GradDtype
    assert GradDtype.from_torch_dtype_name("torch.float16") == GradDtype.FLOAT16
    assert GradDtype.from_torch_dtype_name("bfloat16") == GradDtype.BFLOAT16
    assert GradDtype.FLOAT16.is_reduced_precision is True
    assert GradDtype.FLOAT32.is_reduced_precision is False

    # DedupGradSpec
    spec_fp16 = DedupGradSpec(grad_dtype=GradDtype.FLOAT16, embedding_dim=128)
    spec_fp16.validate()
    assert "static_cast" in spec_fp16.cast_annotation()

    spec_fp32 = DedupGradSpec(grad_dtype=GradDtype.FLOAT32, embedding_dim=64)
    spec_fp32.validate()
    assert "no-op" in spec_fp32.cast_annotation()

    # ShmPermissionRule
    assert ShmPermissionRule.SECURE.is_secure() is True
    assert ShmPermissionRule.PERMISSIVE.is_secure() is False

    rec = ShmAllocationRecord(
        shm_key=12345,
        alloc_size=1024 * 1024,
        permission=ShmPermissionRule.SECURE,
        flags=["IPC_CREAT", "IPC_EXCL"],
    )
    rec.validate_permission()
    assert rec.as_shmget_mode() == (0o600 | 0o1000 | 0o2000)

    try:
        bad = ShmAllocationRecord(
            shm_key=99, alloc_size=1024,
            permission=ShmPermissionRule.PERMISSIVE
        )
        bad.validate_permission()
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

    print("fp16_embedding_grad.py 自测：10 项断言全部 PASS")
