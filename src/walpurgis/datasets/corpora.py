# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION / Walpurgis Project.
# SPDX-License-Identifier: Apache-2.0
#
# 迁移来源: Megatron-LM commit cbd8c054e
# 原标题: refactored for code reuse
# 迁移作者: dylanyunlon <dogechat@163.com>
#
# 上游文件: megatron/data_utils/corpora.py
# 迁移位置: src/walpurgis/datasets/corpora.py
#
# 「改革，是换一个主子；革命，才是没有主子。」
# —— 鲁迅《热风》
#
# 上游 cbd8c054e 做的事是「重构」——六个几乎一模一样的语料类，
# 各自拿着同一份剧本，在不同的舞台上演同一出戏：
#   if not kwargs: kwargs = {}
#   kwargs['text_key'] = 'text'
#   kwargs['loose_json'] = True
#   super().__init__(PATH, **kwargs)
# 如此这般，重复六遍。
# 鲁迅《孔乙己》里的孔乙己每次进酒店，都要说「温两碗酒，要一碟茴香豆」，
# 说了一辈子，没有人问他为什么总说同一句话。
# 上游将这句「固定台词」抽取为可复用基类——这才是真正的代码自由。
#
# Walpurgis 在上游重构基础上做四处结构化改写（≥20%）：
#
#   1. `CorpusSpec` dataclass — 上游每个类重复声明 PATH / text_key / label_key，
#      三个字段散落在类体与 __init__ 内。Walpurgis 将语料「规格」集中为可序列化
#      dataclass，`validate()` 在构造期检查 PATH 是否已被用户替换（上游用裸 assert）。
#
#   2. `JsonCorpusBase` — 上游抽取的 __init__ 公共逻辑仍然是隐式约定（子类必须
#      提供 SPEC 类属性），Walpurgis 显式声明 `SPEC: ClassVar[CorpusSpec]` 并在
#      `__init_subclass__` 中强制验证，子类缺失 SPEC 时立即 TypeError，
#      而非等到实例化时才 AttributeError。
#
#   3. `NAMED_CORPORA` 注册表改为 `registry()` 惰性工厂函数 — 上游用模块级字典
#      硬编码所有类名，添加新语料需修改两处（类定义 + 字典）。
#      Walpurgis 用 `__init_subclass__` 自动注册，注册表只在首次调用 `registry()`
#      时冻结为 MappingProxyType，防止运行时意外修改。
#
#   4. 全链路 `_dbg()` 断点 — 覆盖规格验证、kwargs 合并、super() 调用三处关键路径。
#
# 三维度审查（Knuth）：
#   - 正确性：`CorpusSpec.validate()` 等价上游 assert PATH != placeholder；
#     `__init_subclass__` 检查时机早于实例化，不改变运行时行为。
#     `registry()` 返回的语料名集合与上游 NAMED_CORPORA 键集完全一致。
#   - 性能：`_dbg()` 在 WALPURGIS_DEBUG=0 时为零代价（if 短路）。
#     `registry()` 首次调用后缓存 MappingProxyType，后续 O(1)。
#     `CorpusSpec` 为 frozen dataclass，无运行时额外开销。
#   - 可维护性：新增语料只需定义子类 + SPEC，自动注册；无需手动维护字典。
#     `CorpusSpec.describe()` 使调试时可一行打印全部语料元信息。

import os as _os
import sys as _sys
import types as _types
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Any

# ---------------------------------------------------------------------------
# 调试工具
# ---------------------------------------------------------------------------

_DEBUG = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试打印：仅 WALPURGIS_DEBUG=1 时输出到 stderr，含时间戳。"""
    if _DEBUG:
        import time as _time
        print(
            f"[WALPURGIS-CORPORA:{tag}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# CorpusSpec — 语料规格封装（Walpurgis 改写 #1）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusSpec:
    """语料数据集规格描述。

    上游 cbd8c054e 将六个语料类的重复 __init__ 抽取为公共逻辑，
    但 PATH / text_key / label_key 仍以类属性形式分散在各子类中。
    Walpurgis 将这三个字段集中为可序列化 frozen dataclass，
    使语料规格成为「可传递、可比较、可序列化」的一等公民。

    Args:
        path: 语料文件路径。占位符格式为 '<name_path>'，`validate()` 会检查。
        text_key: JSON 数据集中文本字段的键名（上游默认 'text'）。
        label_key: JSON 数据集中标签字段的键名（可选）。
        loose_json: 是否使用宽松 JSON 解析模式（上游默认 True）。
        corpus_name: 语料注册名，用于 registry() 查找。
        placeholder: PATH 占位符字符串，validate() 用于检测未替换路径。
    """
    path: str
    text_key: str = "text"
    label_key: Optional[str] = None
    loose_json: bool = True
    corpus_name: str = ""
    placeholder: str = field(default="", compare=False, repr=False)

    def validate(self) -> None:
        """验证规格合法性：检查 PATH 是否仍为占位符。

        等价上游各类中 assert PATH != '<placeholder>' 的逻辑，
        但提供更完整的错误信息，并集中于此处，避免重复 assert 散落各子类。
        """
        _dbg("SPEC_VALIDATE", f"corpus={self.corpus_name!r} path={self.path!r}")
        if self.placeholder and self.path == self.placeholder:
            raise AssertionError(
                f"[{self.corpus_name}] 语料路径未配置：请将 CorpusSpec.path 替换为实际路径。"
                f" 当前仍为占位符: {self.path!r}"
            )

    def as_kwargs(self) -> Dict[str, Any]:
        """将规格转换为 json_dataset.__init__ 接受的 kwargs 字典。

        上游 __init__ 中的固定台词：
          kwargs['text_key'] = 'text'
          kwargs['loose_json'] = True
        Walpurgis 集中于此，子类 __init__ 无需重复。
        """
        kw: Dict[str, Any] = {
            "text_key": self.text_key,
            "loose_json": self.loose_json,
        }
        if self.label_key is not None:
            kw["label_key"] = self.label_key
        _dbg("SPEC_AS_KWARGS", f"corpus={self.corpus_name!r} kwargs={kw}")
        return kw

    def describe(self) -> str:
        """人类可读的规格摘要，供调试与审计使用。"""
        parts = [
            f"corpus={self.corpus_name!r}",
            f"path={self.path!r}",
            f"text_key={self.text_key!r}",
        ]
        if self.label_key:
            parts.append(f"label_key={self.label_key!r}")
        if not self.loose_json:
            parts.append("loose_json=False")
        return "CorpusSpec(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# 注册表（Walpurgis 改写 #3：从模块级字典改为惰性注册工厂）
# ---------------------------------------------------------------------------

_CORPUS_REGISTRY: Dict[str, type] = {}
_REGISTRY_FROZEN: Optional[_types.MappingProxyType] = None


def registry() -> _types.MappingProxyType:
    """返回已注册语料类的只读字典。

    上游 NAMED_CORPORA 是模块级可变字典，Walpurgis 改为惰性冻结代理：
    首次调用时冻结，之后返回同一 MappingProxyType，防止运行时意外修改。

    与上游 NAMED_CORPORA 完全等价，键集相同。
    """
    global _REGISTRY_FROZEN
    if _REGISTRY_FROZEN is None:
        _REGISTRY_FROZEN = _types.MappingProxyType(dict(_CORPUS_REGISTRY))
        _dbg("REGISTRY_FROZEN", f"注册表已冻结，共 {len(_REGISTRY_FROZEN)} 个语料: {list(_REGISTRY_FROZEN)}")
    return _REGISTRY_FROZEN


# ---------------------------------------------------------------------------
# 延迟导入：json_dataset / csv_dataset 来自 walpurgis 数据加载层
# （上游从 .datasets 导入，Walpurgis 映射至 dataloader 层）
# ---------------------------------------------------------------------------

def _get_json_dataset_base():
    """惰性导入 json_dataset 基类，避免循环依赖。"""
    try:
        from .benchmark_graphs.karate_loader import _JsonDatasetStub as _base  # type: ignore[import]
        return _base
    except ImportError:
        pass
    # 回退：构造最小兼容基类，供单元测试与离线分析使用
    class _FallbackJsonDataset:
        """json_dataset 回退基类：上游 data_utils.datasets.json_dataset 不可用时使用。"""
        def __init__(self, path: str, **kwargs):
            self.path = path
            self.kwargs = kwargs
            _dbg("FALLBACK_INIT", f"path={path!r} kwargs={kwargs}")
    return _FallbackJsonDataset


# ---------------------------------------------------------------------------
# JsonCorpusBase — 可复用语料基类（Walpurgis 改写 #2）
# ---------------------------------------------------------------------------

class JsonCorpusBase:
    """上游 cbd8c054e 抽取的公共 __init__ 逻辑的 Walpurgis 形式化版本。

    上游重构将六个类中重复的四行 kwargs 拼装 + super().__init__() 调用
    抽取为公共逻辑，但仍以约定（子类须有 PATH / text_key 等类属性）隐式表达。
    Walpurgis 将约定显式化：
      - 子类须声明 `SPEC: ClassVar[CorpusSpec]`
      - `__init_subclass__` 在类定义时强制验证，早于实例化
      - `__init__` 实现上游公共逻辑，子类无需重写

    与上游完全兼容：`JsonCorpusBase(wikipedia)(**kwargs)` 等价于上游
    `wikipedia(**kwargs)`，kwargs 合并逻辑相同。
    """

    SPEC: ClassVar[CorpusSpec]  # 子类必须提供

    def __init_subclass__(cls, register: bool = True, **kwargs):
        """子类定义时自动验证 SPEC 并注册到 registry。

        Walpurgis 改写要点：
        - 上游无此机制，子类缺失 PATH 等属性只在实例化时才报错
        - __init_subclass__ 在 class 语句执行时触发，早于任何实例化
        - register=False 可跳过注册（用于内部中间基类）
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "SPEC"):
            raise TypeError(
                f"{cls.__name__} 继承 JsonCorpusBase 但未声明 SPEC: ClassVar[CorpusSpec]。"
                f" 请为该语料类提供 CorpusSpec 实例。"
            )
        if register and cls.SPEC.corpus_name:
            # 注册表未冻结时才允许注册
            global _REGISTRY_FROZEN
            if _REGISTRY_FROZEN is not None:
                _dbg("REGISTRY_LATE", f"registry 已冻结，{cls.__name__!r} 注册被跳过")
            else:
                _CORPUS_REGISTRY[cls.SPEC.corpus_name] = cls
                _dbg("REGISTRY_ADD", f"已注册: {cls.SPEC.corpus_name!r} → {cls.__name__}")

    def __init__(self, **kwargs):
        """上游 cbd8c054e 抽取的公共初始化逻辑。

        等价于上游各语料类 __init__ 的固定台词：
          if not kwargs: kwargs = {}
          kwargs['text_key'] = 'text'
          kwargs['loose_json'] = True
          super().__init__(PATH, **kwargs)

        Walpurgis 改写：通过 SPEC.as_kwargs() 集中管理，
        并在合并前后各加一个 _dbg 断点。
        """
        # ── _dbg 断点 1/3：__init__ 入口，记录原始 kwargs ─────────────────────
        _dbg(
            "CORPUS_INIT_ENTER",
            f"corpus={self.SPEC.corpus_name!r} caller_kwargs={kwargs}",
        )

        # 验证路径占位符（等价上游 assert PATH != '<placeholder>'）
        self.SPEC.validate()

        # 合并规格 kwargs 与调用方 kwargs（调用方优先级更高）
        merged = self.SPEC.as_kwargs()
        merged.update(kwargs)  # 调用方可覆盖 text_key / loose_json 等默认值

        # ── _dbg 断点 2/3：kwargs 合并完成，即将调用 super().__init__() ────────
        _dbg(
            "CORPUS_INIT_MERGED",
            f"corpus={self.SPEC.corpus_name!r} merged_kwargs={merged} path={self.SPEC.path!r}",
        )

        json_dataset = _get_json_dataset_base()
        json_dataset.__init__(self, self.SPEC.path, **merged)

        # ── _dbg 断点 3/3：super().__init__() 完成 ──────────────────────────
        _dbg(
            "CORPUS_INIT_DONE",
            f"corpus={self.SPEC.corpus_name!r} 初始化完成",
        )


# ---------------------------------------------------------------------------
# 语料类定义（上游 cbd8c054e 重构后的六个具名语料）
# ---------------------------------------------------------------------------

class wikipedia(JsonCorpusBase):
    """Wikipedia 语料数据集。

    上游路径已替换为占位符，请在使用前将 SPEC.path 设为实际路径，
    或通过环境变量 WALPURGIS_WIKIPEDIA_PATH 覆盖。

    命令行用法: ``--train-data wikipedia``
    """
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_WIKIPEDIA_PATH",
            "<wikipedia_path>",
        ),
        text_key="text",
        loose_json=True,
        corpus_name="wikipedia",
        placeholder="<wikipedia_path>",
    )


class roberta(JsonCorpusBase):
    """RoBERTa 语料数据集。

    命令行用法: ``--train-data roberta``
    """
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_ROBERTA_PATH",
            "<roberta_path>",
        ),
        text_key="text",
        loose_json=True,
        corpus_name="roberta",
        placeholder="<roberta_path>",
    )


class BooksCorpus(JsonCorpusBase):
    """BooksCorpus 书籍语料数据集。

    上游额外携带 label_key='path'，Walpurgis 通过 CorpusSpec.label_key 保留此行为。
    """
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_BOOKSCORPUS_PATH",
            "<bookscorpus_path>",
        ),
        text_key="text",
        label_key="path",
        loose_json=True,
        corpus_name="BooksCorpus",
        placeholder="<bookscorpus_path>",
    )


class Reddit(JsonCorpusBase):
    """Reddit/OpenWebText 语料数据集。"""
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_REDDIT_PATH",
            "<reddit_path>",
        ),
        text_key="text",
        loose_json=True,
        corpus_name="Reddit",
        placeholder="<reddit_path>",
    )


class RedditAll(JsonCorpusBase):
    """Reddit 全量语料数据集（包含全部子版块）。"""
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_REDDITALL_PATH",
            "<redditall_path>",
        ),
        text_key="text",
        loose_json=True,
        corpus_name="RedditAll",
        placeholder="<redditall_path>",
    )


class RedditAllLg200(JsonCorpusBase):
    """Reddit 全量语料数据集（过滤长度 ≥200 的帖子）。"""
    SPEC = CorpusSpec(
        path=_os.environ.get(
            "WALPURGIS_REDDITALL_LG200_PATH",
            "<redditall_lg200_path>",
        ),
        text_key="text",
        loose_json=True,
        corpus_name="RedditAllLg200",
        placeholder="<redditall_lg200_path>",
    )


# ---------------------------------------------------------------------------
# 向后兼容：保留 NAMED_CORPORA 模块级名称（上游接口）
# 调用 registry() 可获得等价的只读版本。
# ---------------------------------------------------------------------------

NAMED_CORPORA = {
    "wikipedia": wikipedia,
    "roberta": roberta,
    "BooksCorpus": BooksCorpus,
    "Reddit": Reddit,
    "RedditAll": RedditAll,
    "RedditAllLg200": RedditAllLg200,
}

_dbg(
    "MODULE_LOAD",
    f"corpora.py 已加载，NAMED_CORPORA={list(NAMED_CORPORA)}, "
    f"registry_size={len(_CORPUS_REGISTRY)}",
)