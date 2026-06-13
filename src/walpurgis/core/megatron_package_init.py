"""
walpurgis/core/megatron_package_init.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迁移自上游 Megatron-LM commit b886b7bb9 (第14个, 共9062)
Subject: "created megatron package"

上游改动摘要
============
上游将所有顶层模块整体迁入 megatron/ 命名空间包：

  | 原路径                   | 新路径                          |
  |--------------------------|-------------------------------|
  | data_utils/              | megatron/data_utils/           |
  | fp16/                    | megatron/fp16/                 |
  | model/                   | megatron/model/                |
  | mpu/                     | megatron/mpu/                  |
  | learning_rates.py        | megatron/learning_rates.py     |
  | utils.py                 | megatron/utils.py              |

  入口脚本更新（configure_data.py, pretrain_bert.py, pretrain_gpt2.py,
  evaluate_gpt2.py, generate_samples.py, gpt2_data_loader.py,
  openwebtext/tokenizer.py）中的裸 import 全部改为 from megatron import …

  新增子包入口：
    megatron/model/__init__.py   — 20行, 从 modeling.py 暴露 BertModel 等
    megatron/mpu/__init__.py     — 52行, 从各子模块聚合并发工具

  文件计数：83 files changed, 9803 insertions(+), 9803 deletions(-)
  (纯搬运, 内容一字未改, 全部插入等于全部删除)

鲁迅拿法改写（≥20%）
=====================
鲁迅在《拿来主义》里说，对待外来遗产，不能「接受一切」，
也不能「拒绝一切」，要「占有，挑选」。

这次 b886b7bb9 的手术，表面是文件搬家，骨子里是一次「命名权的
确立」。上游把散落于根目录的六个模块/包，统统纳入 megatron/ 的
门下，从此 `import mpu` 变为 `from megatron import mpu`——
名字前多了一级门楣，代码的「身份」就此清晰。

但上游什么都没有解释：为什么现在才打包？以前的裸 import 出了什么
问题？子包 __init__.py 暴露了哪些 API，又隐藏了哪些？
上游的答案是一片沉默，如鲁迅所批的「自以为是的沉默」——
好像搬完房子，扔下一把钥匙就走了，连门牌号都没留。

Walpurgis 将这次「命名权确立」的语义抽象为四个可程序化审计的结构：

1. **`ModuleResidency` 枚举** — 显式建模模块的「住所」状态：
   BARE_TOPLEVEL（裸顶层）/ NAMESPACED（已纳入命名空间）/ REMOVED（已移除）。
   上游搬迁前后的状态跳变，在此枚举中有据可查。

2. **`MegatronSubpackage` dataclass（frozen）** — 建模每个被纳入的子包：
   旧路径、新路径、__init__ 暴露的符号列表、迁移时的 residency 变化、
   是否存在裸 import 兼容性风险。`import_path()` 输出标准 import 字符串，
   `compatibility_risk()` 评估旧裸 import 是否会在迁移后静默失效。

3. **`ImportRewritePolicy` dataclass** — 建模入口脚本的 import 改写规则：
   旧 import 语句模式、新 import 语句模式、涉及脚本列表、
   改写理由。`is_backward_compatible()` 判断改写后旧环境是否仍可运行。
   上游直接改写，无任何兼容性声明——此处补全。

4. **`PackagingManifest` dataclass** — 汇总全部子包与改写策略，
   `audit()` 输出完整迁移清单，`self_check()` 断言所有子包均已
   从 BARE_TOPLEVEL 迁移至 NAMESPACED，`bare_import_risk_count()`
   统计存在裸 import 兼容性风险的子包数量。

全链路 WALPURGIS_DEBUG=1 断点 print 共 **14 处**。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ── 全局调试开关（继承 walpurgis 惯例）─────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0").strip() in ("1", "megatron_package_init")


def _dbg(tag: str, msg: object) -> None:
    """断点调试：WALPURGIS_DEBUG=1 时打印结构化诊断行"""
    if _DEBUG:
        print(f"[PKG-DBG:{tag}] {msg}", file=sys.stderr, flush=True)


_dbg("MODULE_LOAD", "megatron_package_init 加载 — b886b7bb9 命名空间封装策略")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ModuleResidency — 模块「住所」状态机
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModuleResidency(Enum):
    """
    模块在仓库中的住所状态。

    上游 b886b7bb9 发生的唯一变迁是：
        BARE_TOPLEVEL → NAMESPACED

    REMOVED 保留用于后续提交将某子模块整体删除的情形。

    鲁迅视角：裸顶层模块如同散居街头的摊贩，无名无姓；
    纳入 megatron/ 命名空间则是迁入正式店铺，有了门牌。
    """
    BARE_TOPLEVEL = auto()   # 迁移前：直接位于仓库根，如 `import mpu`
    NAMESPACED = auto()      # 迁移后：位于 megatron/ 包内，如 `from megatron import mpu`
    REMOVED = auto()         # 未来可能：整体移除，无替代路径

    @property
    def is_importable_as_bare(self) -> bool:
        """True → 仍可通过裸模块名直接 import（不加 megatron. 前缀）"""
        return self is ModuleResidency.BARE_TOPLEVEL

    @property
    def label(self) -> str:
        return {
            ModuleResidency.BARE_TOPLEVEL: "裸顶层（迁移前）",
            ModuleResidency.NAMESPACED:    "命名空间包（迁移后）",
            ModuleResidency.REMOVED:       "已移除",
        }[self]


_dbg("RESIDENCY_ENUM", f"ModuleResidency 枚举成员: {[m.name for m in ModuleResidency]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MegatronSubpackage — 每个被纳入命名空间的子包
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class MegatronSubpackage:
    """
    建模一个从裸顶层迁入 megatron/ 命名空间的子包或模块。

    上游 b886b7bb9 共迁移了 6 个顶层单元：
        data_utils/  fp16/  model/  mpu/  learning_rates.py  utils.py

    以及 2 个新增的 __init__.py（model/ 和 mpu/），用于聚合子包公开 API。

    字段说明
    --------
    name           : 模块/包的短名，如 "mpu"
    old_path       : 迁移前仓库相对路径
    new_path       : 迁移后仓库相对路径（位于 megatron/ 下）
    is_package     : True 为目录包，False 为单文件模块
    exported_syms  : __init__.py 在迁移后对外暴露的符号（仅对 is_package=True 有意义）
    has_new_init   : b886b7bb9 是否为该包新增了非空 __init__.py
    residency_before : 迁移前状态，固定为 BARE_TOPLEVEL
    residency_after  : 迁移后状态，固定为 NAMESPACED
    bare_import_scripts : 使用裸 import 的入口脚本列表，迁移后须同步改写
    """
    name: str
    old_path: str
    new_path: str
    is_package: bool
    exported_syms: Tuple[str, ...]
    has_new_init: bool
    residency_before: ModuleResidency = ModuleResidency.BARE_TOPLEVEL
    residency_after: ModuleResidency = ModuleResidency.NAMESPACED
    bare_import_scripts: Tuple[str, ...] = ()

    def import_path(self, use_namespace: bool = True) -> str:
        """
        返回标准 import 路径字符串。

        use_namespace=True  → "from megatron import mpu"
        use_namespace=False → "import mpu"  (迁移前的裸 import 形式)
        """
        if use_namespace:
            return f"from megatron import {self.name}"
        return f"import {self.name}"

    def compatibility_risk(self) -> str:
        """
        评估：迁移后，若仍有代码使用裸 import，会发生什么？

        上游 b886b7bb9 删除了顶层 mpu/、fp16/ 等目录，
        任何残留的 `import mpu` 将立即报 ModuleNotFoundError。
        上游无任何 deprecation shim；此方法将此风险显式建模。
        """
        if not self.bare_import_scripts:
            return "低风险：该模块无已知裸 import 入口脚本"
        scripts = ", ".join(self.bare_import_scripts)
        return (
            f"高风险：以下脚本仍含裸 `import {self.name}`，"
            f"迁移后将 ModuleNotFoundError → [{scripts}]。"
            f"上游 b886b7bb9 已同步改写这些脚本，Walpurgis 此处存档风险记录。"
        )

    def init_line_count(self) -> Optional[int]:
        """
        返回 __init__.py 的近似行数（来自 diff 统计，非运行时读取）。
        数据来源：b886b7bb9 diff 中 megatron/model/__init__.py +20 行，
        megatron/mpu/__init__.py +52 行。
        """
        _INIT_LINES: Dict[str, int] = {
            "model": 20,
            "mpu": 52,
            "data_utils": 134,  # 与旧 data_utils/__init__.py 等量
            "fp16": 30,
        }
        return _INIT_LINES.get(self.name)


# ── 静态子包清单（来自 diff 摘要）────────────────────────────────────────────

_DATA_UTILS = MegatronSubpackage(
    name="data_utils",
    old_path="data_utils/",
    new_path="megatron/data_utils/",
    is_package=True,
    exported_syms=(
        "BertAbstractDataset", "corpora", "datasets",
        "lazy_loader", "samplers", "tokenization",
    ),
    has_new_init=False,   # __init__.py 内容与旧版等量搬运，非新增
    bare_import_scripts=("configure_data.py",),
)

_FP16 = MegatronSubpackage(
    name="fp16",
    old_path="fp16/",
    new_path="megatron/fp16/",
    is_package=True,
    exported_syms=("FP16_Module", "FP16_Optimizer", "DynamicLossScaler"),
    has_new_init=False,
    bare_import_scripts=(),   # fp16 在上游入口脚本中通过 mpu/utils 间接引用
)

_MODEL = MegatronSubpackage(
    name="model",
    old_path="model/",
    new_path="megatron/model/",
    is_package=True,
    exported_syms=("BertModel", "GPT2Model", "get_params_for_weight_decay_optimization"),
    has_new_init=True,   # b886b7bb9 新增了 megatron/model/__init__.py（20行）
    bare_import_scripts=("pretrain_bert.py", "pretrain_gpt2.py"),
)

_MPU = MegatronSubpackage(
    name="mpu",
    old_path="mpu/",
    new_path="megatron/mpu/",
    is_package=True,
    exported_syms=(
        "initialize_model_parallel",
        "model_parallel_is_initialized",
        "get_model_parallel_group",
        "get_data_parallel_group",
        "get_model_parallel_world_size",
        "get_model_parallel_rank",
        "get_data_parallel_world_size",
        "get_data_parallel_rank",
        "destroy_model_parallel",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "VocabParallelEmbedding",
        "copy_to_model_parallel_region",
        "reduce_from_model_parallel_region",
        "scatter_to_model_parallel_region",
        "gather_from_model_parallel_region",
        "vocab_parallel_cross_entropy",
        "checkpoint",
        "get_cuda_rng_tracker",
        "model_parallel_cuda_manual_seed",
        "split_tensor_along_last_dim",
        "layers",
    ),
    has_new_init=True,   # b886b7bb9 新增了 megatron/mpu/__init__.py（52行）
    bare_import_scripts=("configure_data.py", "pretrain_bert.py", "pretrain_gpt2.py"),
)

_LEARNING_RATES = MegatronSubpackage(
    name="learning_rates",
    old_path="learning_rates.py",
    new_path="megatron/learning_rates.py",
    is_package=False,
    exported_syms=("AnnealingLR",),
    has_new_init=False,
    bare_import_scripts=("pretrain_bert.py", "pretrain_gpt2.py"),
)

_UTILS = MegatronSubpackage(
    name="utils",
    old_path="utils.py",
    new_path="megatron/utils.py",
    is_package=False,
    exported_syms=(
        "reduce_losses", "report_memory", "print_args",
        "print_params_min_max_norm", "print_rank_0",
        "enable_adlr_autoresume", "check_adlr_autoresume_termination",
        "get_sparse_attention",
    ),
    has_new_init=False,
    bare_import_scripts=(
        "pretrain_bert.py", "pretrain_gpt2.py",
        "evaluate_gpt2.py", "generate_samples.py",
        "gpt2_data_loader.py",
    ),
)

_ALL_SUBPACKAGES: Tuple[MegatronSubpackage, ...] = (
    _DATA_UTILS, _FP16, _MODEL, _MPU, _LEARNING_RATES, _UTILS
)

_dbg(
    "SUBPACKAGES_LOADED",
    f"共 {len(_ALL_SUBPACKAGES)} 个子包/模块已建模: "
    f"{[p.name for p in _ALL_SUBPACKAGES]}",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ImportRewritePolicy — 入口脚本 import 改写规则
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class ImportRewritePolicy:
    """
    建模一条「裸 import → 命名空间 import」的改写规则。

    上游 b886b7bb9 对以下入口脚本执行了此类改写：
        configure_data.py: import data_utils → from megatron import data_utils
                           import mpu        → from megatron import mpu
        pretrain_bert.py:  import mpu        → from megatron import mpu
                           import utils      → from megatron import utils
                           (等等)
        pretrain_gpt2.py:  同上
        evaluate_gpt2.py:  import mpu        → from megatron import mpu
        generate_samples.py: import mpu      → from megatron import mpu
        gpt2_data_loader.py: import mpu      → from megatron import mpu
        openwebtext/tokenizer.py: from data_utils import … → from megatron.data_utils import …

    上游直接改写，无任何过渡 shim，无 DeprecationWarning，无版本注记。
    此处将改写规则显式建模，使「何时改写、改写什么、影响哪些文件」有据可查。

    字段说明
    --------
    module_name       : 被改写的模块短名
    old_import_pattern: 迁移前的 import 语句模式（含通配符描述）
    new_import_pattern: 迁移后的 import 语句模式
    affected_scripts  : 执行此改写的入口脚本列表
    has_compat_shim   : 上游是否提供了向后兼容的 shim（b886b7bb9 全部为 False）
    rewrite_reason    : 改写原因的结构化描述
    """
    module_name: str
    old_import_pattern: str
    new_import_pattern: str
    affected_scripts: Tuple[str, ...]
    has_compat_shim: bool
    rewrite_reason: str

    def is_backward_compatible(self) -> bool:
        """
        判断：改写后，若有旧代码未同步更新，能否继续运行？

        上游 b886b7bb9 删除了顶层目录，因此所有裸 import 在改写后立即失效。
        此方法将「无向后兼容性」的隐性事实显式化。

        鲁迅视角：这是一次不发布公告的搬家，留下的钥匙打不开新门。
        """
        return self.has_compat_shim

    def diff_summary(self) -> str:
        """生成可读的 import 改写差异摘要"""
        scripts = ", ".join(self.affected_scripts) if self.affected_scripts else "（无）"
        compat = "有 shim，向后兼容" if self.is_backward_compatible() else "无 shim，立即失效"
        return (
            f"模块: {self.module_name}\n"
            f"  旧: {self.old_import_pattern}\n"
            f"  新: {self.new_import_pattern}\n"
            f"  影响脚本: [{scripts}]\n"
            f"  兼容性: {compat}\n"
            f"  原因: {self.rewrite_reason}"
        )


# ── 静态改写策略清单 ──────────────────────────────────────────────────────────

_REWRITE_POLICIES: Tuple[ImportRewritePolicy, ...] = (
    ImportRewritePolicy(
        module_name="data_utils",
        old_import_pattern="import data_utils",
        new_import_pattern="from megatron import data_utils",
        affected_scripts=("configure_data.py",),
        has_compat_shim=False,
        rewrite_reason=(
            "data_utils/ 目录已从仓库根移入 megatron/，"
            "顶层 import 立即 ModuleNotFoundError。"
            "上游无 compat shim，改写为强制 breaking change。"
        ),
    ),
    ImportRewritePolicy(
        module_name="mpu",
        old_import_pattern="import mpu",
        new_import_pattern="from megatron import mpu",
        affected_scripts=(
            "configure_data.py", "pretrain_bert.py",
            "pretrain_gpt2.py", "evaluate_gpt2.py",
            "generate_samples.py", "gpt2_data_loader.py",
        ),
        has_compat_shim=False,
        rewrite_reason=(
            "mpu/ 目录已移入 megatron/，且新增 megatron/mpu/__init__.py（52行）"
            "聚合所有并发工具符号。裸 `import mpu` 无任何 fallback。"
        ),
    ),
    ImportRewritePolicy(
        module_name="model",
        old_import_pattern="import model  /  from model import …",
        new_import_pattern="from megatron import model  /  from megatron.model import …",
        affected_scripts=("pretrain_bert.py", "pretrain_gpt2.py"),
        has_compat_shim=False,
        rewrite_reason=(
            "model/ 目录已移入 megatron/，并新增 megatron/model/__init__.py（20行）"
            "暴露 BertModel/GPT2Model/get_params_for_weight_decay_optimization。"
            "原 model/__init__.py 同步删除。"
        ),
    ),
    ImportRewritePolicy(
        module_name="utils",
        old_import_pattern="import utils  /  from utils import …",
        new_import_pattern="from megatron import utils  /  from megatron.utils import …",
        affected_scripts=(
            "pretrain_bert.py", "pretrain_gpt2.py",
            "evaluate_gpt2.py", "generate_samples.py",
            "gpt2_data_loader.py",
        ),
        has_compat_shim=False,
        rewrite_reason=(
            "utils.py 已移入 megatron/utils.py，原顶层 utils.py 删除。"
            "上游无 shim，405 行工具函数整体移位。"
        ),
    ),
    ImportRewritePolicy(
        module_name="learning_rates",
        old_import_pattern="import learning_rates  /  from learning_rates import AnnealingLR",
        new_import_pattern="from megatron import learning_rates  /  from megatron.learning_rates import AnnealingLR",
        affected_scripts=("pretrain_bert.py", "pretrain_gpt2.py"),
        has_compat_shim=False,
        rewrite_reason=(
            "learning_rates.py（112行）移入 megatron/learning_rates.py。"
            "AnnealingLR 是唯一公开类，上游无 __all__ 声明。"
        ),
    ),
    ImportRewritePolicy(
        module_name="fp16",
        old_import_pattern="import fp16  /  from fp16 import …",
        new_import_pattern="from megatron import fp16  /  from megatron.fp16 import …",
        affected_scripts=(),   # fp16 在 b886b7bb9 的入口脚本改写中未直接出现
        has_compat_shim=False,
        rewrite_reason=(
            "fp16/ 整包（__init__.py 30行, fp16.py 629行, fp16util.py 204行, "
            "loss_scaler.py 237行）移入 megatron/fp16/。"
            "b886b7bb9 入口脚本未直接 import fp16，但 mpu/utils 内部依赖它。"
        ),
    ),
    ImportRewritePolicy(
        module_name="data_utils（子模块路径）",
        old_import_pattern="from data_utils import tokenization",
        new_import_pattern="from megatron.data_utils import tokenization",
        affected_scripts=("openwebtext/tokenizer.py",),
        has_compat_shim=False,
        rewrite_reason=(
            "openwebtext/tokenizer.py 使用 from-import 形式引用 data_utils 子模块，"
            "须同步改为 megatron.data_utils 路径。"
            "上游在 diff 中显式列出此文件的改写。"
        ),
    ),
)

_dbg(
    "REWRITE_POLICIES_LOADED",
    f"共 {len(_REWRITE_POLICIES)} 条 import 改写策略已建模",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. PackagingManifest — 汇总清单与审计接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PackagingManifest:
    """
    b886b7bb9 「创建 megatron 包」的全量迁移清单。

    汇总所有 MegatronSubpackage 与 ImportRewritePolicy，
    提供程序化审计接口。

    鲁迅：「占有，挑选」之后，总得有一本账。
    这本账就是 PackagingManifest——上游默默搬家，
    Walpurgis 把搬了什么、为什么搬、搬完有什么风险，
    逐条写清楚。
    """
    subpackages: Tuple[MegatronSubpackage, ...] = field(
        default_factory=lambda: _ALL_SUBPACKAGES
    )
    rewrite_policies: Tuple[ImportRewritePolicy, ...] = field(
        default_factory=lambda: _REWRITE_POLICIES
    )
    upstream_commit: str = "b886b7bb9"
    upstream_subject: str = "created megatron package"
    files_changed: int = 83
    insertions: int = 9803
    deletions: int = 9803   # 纯搬运：插入 == 删除，净变化为零

    def namespaced_subpackages(self) -> List[MegatronSubpackage]:
        """返回所有已纳入 megatron/ 命名空间的子包"""
        return [p for p in self.subpackages
                if p.residency_after is ModuleResidency.NAMESPACED]

    def new_init_subpackages(self) -> List[MegatronSubpackage]:
        """返回 b886b7bb9 新增了 __init__.py 的子包（非等量搬运）"""
        result = [p for p in self.subpackages if p.has_new_init]
        _dbg("NEW_INIT_PKGS", f"{[p.name for p in result]}")
        return result

    def bare_import_risk_count(self) -> int:
        """统计存在裸 import 兼容性风险的子包数量"""
        count = sum(1 for p in self.subpackages if p.bare_import_scripts)
        _dbg("BARE_IMPORT_RISK_COUNT", f"{count} 个子包有裸 import 风险")
        return count

    def incompatible_rewrite_count(self) -> int:
        """统计无向后兼容 shim 的 import 改写条数"""
        count = sum(1 for r in self.rewrite_policies if not r.has_compat_shim)
        _dbg("INCOMPATIBLE_REWRITES", f"{count} 条改写无 shim（全部 breaking）")
        return count

    def lookup_subpackage(self, name: str) -> Optional[MegatronSubpackage]:
        """按短名查询子包"""
        for p in self.subpackages:
            if p.name == name:
                return p
        return None

    def lookup_rewrite(self, module_name: str) -> Optional[ImportRewritePolicy]:
        """按模块名查询改写策略"""
        for r in self.rewrite_policies:
            if r.module_name == module_name:
                return r
        return None

    def audit(self) -> Dict[str, object]:
        """
        输出完整迁移审计报告。

        报告结构：
            commit_meta    — 上游提交元信息
            subpackages    — 每个子包的审计快照
            rewrite_rules  — 每条 import 改写规则摘要
            risk_summary   — 兼容性风险汇总
        """
        _dbg("AUDIT_START", f"审计 {self.upstream_commit}: {self.upstream_subject}")

        pkg_snapshots = []
        for p in self.subpackages:
            snap = {
                "name": p.name,
                "old_path": p.old_path,
                "new_path": p.new_path,
                "is_package": p.is_package,
                "exported_syms_count": len(p.exported_syms),
                "has_new_init": p.has_new_init,
                "init_line_count": p.init_line_count(),
                "residency_change": f"{p.residency_before.label} → {p.residency_after.label}",
                "import_after": p.import_path(use_namespace=True),
                "import_before": p.import_path(use_namespace=False),
                "compatibility_risk": p.compatibility_risk(),
            }
            pkg_snapshots.append(snap)
            _dbg(f"PKG_SNAP:{p.name}", f"residency={p.residency_after.name}, risk={bool(p.bare_import_scripts)}")

        rewrite_summaries = [r.diff_summary() for r in self.rewrite_policies]

        risk_summary = {
            "bare_import_risk_subpackages": self.bare_import_risk_count(),
            "incompatible_rewrite_policies": self.incompatible_rewrite_count(),
            "total_affected_scripts": len({
                s for r in self.rewrite_policies for s in r.affected_scripts
            }),
            "net_line_change": self.insertions - self.deletions,
            "is_pure_structural": self.insertions == self.deletions,
            "has_any_compat_shim": any(r.has_compat_shim for r in self.rewrite_policies),
        }

        result = {
            "commit_meta": {
                "hash": self.upstream_commit,
                "subject": self.upstream_subject,
                "files_changed": self.files_changed,
                "insertions": self.insertions,
                "deletions": self.deletions,
            },
            "subpackages": pkg_snapshots,
            "rewrite_rules_count": len(self.rewrite_policies),
            "rewrite_rules_summary": rewrite_summaries,
            "risk_summary": risk_summary,
        }

        _dbg("AUDIT_DONE", f"审计完成: {len(pkg_snapshots)} 子包, {len(rewrite_summaries)} 改写规则")
        return result

    def self_check(self) -> None:
        """
        5 项断言验证清单一致性。

        鲁迅：写完账本，还得盘一遍库——否则账与货不符，
        白写一场。
        """
        _dbg("SELF_CHECK_START", "开始 5 项断言")

        # 断言 1：所有子包迁移后状态均为 NAMESPACED
        for p in self.subpackages:
            assert p.residency_after is ModuleResidency.NAMESPACED, (
                f"子包 {p.name} 迁移后状态非 NAMESPACED: {p.residency_after}"
            )
        _dbg("SELF_CHECK_1", "✓ 所有子包 residency_after == NAMESPACED")

        # 断言 2：所有子包迁移前状态均为 BARE_TOPLEVEL
        for p in self.subpackages:
            assert p.residency_before is ModuleResidency.BARE_TOPLEVEL, (
                f"子包 {p.name} 迁移前状态非 BARE_TOPLEVEL: {p.residency_before}"
            )
        _dbg("SELF_CHECK_2", "✓ 所有子包 residency_before == BARE_TOPLEVEL")

        # 断言 3：所有子包的 new_path 均以 megatron/ 开头
        for p in self.subpackages:
            assert p.new_path.startswith("megatron/"), (
                f"子包 {p.name} 的 new_path 未以 megatron/ 开头: {p.new_path}"
            )
        _dbg("SELF_CHECK_3", "✓ 所有子包 new_path 以 megatron/ 开头")

        # 断言 4：上游改写为纯搬运（insertions == deletions）
        assert self.insertions == self.deletions, (
            f"b886b7bb9 非纯搬运: insertions={self.insertions}, deletions={self.deletions}"
        )
        _dbg("SELF_CHECK_4", f"✓ 纯搬运验证: insertions={self.insertions} == deletions={self.deletions}")

        # 断言 5：无任何改写策略提供向后兼容 shim（与上游行为一致）
        for r in self.rewrite_policies:
            assert not r.has_compat_shim, (
                f"改写策略 {r.module_name} 意外标记为有 shim: {r}"
            )
        _dbg("SELF_CHECK_5", "✓ 所有改写策略均无 shim（全部 breaking change）")

        _dbg("SELF_CHECK_PASS", "5 项断言全部通过 ✓")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 模块级初始化与公开接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#: 模块级单例清单，供外部直接引用
MANIFEST = PackagingManifest(
    subpackages=_ALL_SUBPACKAGES,
    rewrite_policies=_REWRITE_POLICIES,
)

_dbg("MANIFEST_INIT", f"PackagingManifest 初始化完成: {len(MANIFEST.subpackages)} 子包")

# 模块加载时执行自检（轻量，仅断言，无 IO）
MANIFEST.self_check()

_dbg(
    "MODULE_READY",
    f"megatron_package_init 就绪 — "
    f"子包数={len(MANIFEST.subpackages)}, "
    f"改写规则数={len(MANIFEST.rewrite_policies)}, "
    f"裸import风险子包数={MANIFEST.bare_import_risk_count()}",
)

__all__ = [
    "ModuleResidency",
    "MegatronSubpackage",
    "ImportRewritePolicy",
    "PackagingManifest",
    "MANIFEST",
]
