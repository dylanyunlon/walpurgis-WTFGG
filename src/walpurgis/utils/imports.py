"""
walpurgis.utils.imports — 延迟可选依赖导入

上游迁移: cugraph-gnn 94ac7fea (2026-03-19)
  - 删除 package_available() 及 packaging 依赖 (packaging.requirements.Requirement)
  - 删除 torch <1.13 兼容路径 (pylibwholegraph/test_utils/test_comm.py)
  - 保留 MissingModule / FoundModule / import_optional 核心三件套
  - 加入 find_spec 双重探测, 正确处理点分模块名如 "torch.autograd"

鲁迅: 以前有一个叫 package_available 的函数, 它用 packaging.requirements 检查版本,
      然后发现 torch>=2.3 早就进门了, 这个守门人才被裁撤。
"""
import os
import importlib
from importlib import import_module
from importlib.util import find_spec

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


class MissingModule:
    """
    当可选依赖不可用时的占位对象。
    实例上的任何属性访问均抛出 RuntimeError, 令错误信息清晰指向缺失依赖。

    上游 cugraph_pyg.utils.imports.MissingModule (94ac7fea 后)
    Walpurgis 改写: 错误消息增加安装建议；__call__ 也抛错（上游未覆盖此路径）。
    """

    def __init__(self, mod_name: str):
        # 上游用 self.name, 此处兼容保留
        object.__setattr__(self, 'name', mod_name)
        if _WALPURGIS_DEBUG:
            print(f"[WALPURGIS_DEBUG] MissingModule created for: {mod_name}")

    def __getattr__(self, attr: str):
        name = object.__getattribute__(self, 'name')
        raise RuntimeError(
            f"This feature requires the '{name}' package/module. "
            f"Install it with: pip install {name}"
        )

    def __call__(self, *args, **kwargs):
        name = object.__getattribute__(self, 'name')
        raise RuntimeError(
            f"This feature requires the '{name}' package/module. "
            f"Install it with: pip install {name}"
        )

    def __bool__(self):
        return False

    def __repr__(self):
        name = object.__getattribute__(self, 'name')
        return f"MissingModule({name!r})"


class FoundModule:
    """
    懒加载包装器: 首次属性访问时才真正 import 模块。

    上游 cugraph_pyg.utils.imports.FoundModule (94ac7fea 引入)
    Walpurgis 改写: 加 WALPURGIS_DEBUG 断点；__repr__ 展示加载状态。
    """

    def __init__(self, mod: str):
        object.__setattr__(self, '_wmod', mod)
        object.__setattr__(self, '_wimported', False)
        if _WALPURGIS_DEBUG:
            print(f"[WALPURGIS_DEBUG] FoundModule stub created for: {mod}")

    def __getattr__(self, attr: str):
        mod_name = object.__getattribute__(self, '_wmod')
        imported = object.__getattribute__(self, '_wimported')
        if not imported:
            if _WALPURGIS_DEBUG:
                print(f"[WALPURGIS_DEBUG] FoundModule lazy-importing: {mod_name}")
            real_mod = import_module(mod_name)
            object.__setattr__(self, '_wmod', real_mod)
            object.__setattr__(self, '_wimported', True)
        return getattr(object.__getattribute__(self, '_wmod'), attr)

    def __repr__(self):
        mod = object.__getattribute__(self, '_wmod')
        imported = object.__getattribute__(self, '_wimported')
        return f"FoundModule({mod!r}, imported={imported})"

    def __bool__(self):
        return True


def import_optional(mod: str, default_mod_class=MissingModule):
    """
    尝试导入可选模块 mod, 成功返回 FoundModule 懒代理, 失败返回 default_mod_class 实例。

    上游: cugraph_pyg.utils.imports.import_optional (94ac7fea 后去除 package_available 后)
    Walpurgis 改写:
      - 用 find_spec 双重探测处理点分名如 "torch.autograd"（与上游一致）
      - WALPURGIS_DEBUG 断点覆盖探测/命中/未命中三个阶段
      - default_mod_class 构造时统一用 mod_name= 关键字参数（上游同）

    Example
    -------
    >>> from walpurgis.utils.imports import import_optional
    >>> nx = import_optional("networkx_not_exist")
    >>> nx.Graph()  # → RuntimeError: This feature requires 'networkx_not_exist'

    删除历史: package_available(requirement: str) → bool
      上游 94ac7fea 中移除, 因为 packaging.requirements.Requirement 依赖被整体移除。
      Walpurgis 此文件从未引入 package_available, 本次迁移同步确认清洁。
    """
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG] import_optional: probing '{mod}'")

    mod_found = False
    try:
        mod_found = find_spec(mod) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        mod_found = False

    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG] import_optional: '{mod}' found={mod_found}")

    if mod_found:
        return FoundModule(mod)
    else:
        return default_mod_class(mod_name=mod)


# ── 迁移批次记录 ────────────────────────────────────────────────────────────
# migrate 94ac7fe: remove dependency on 'packaging', patches for torch 1.x (#437)
#   上游删除 package_available() + packaging dep + torch<1.13 compat
#   Walpurgis: 同步升级 imports.py, 引入 FoundModule/find_spec 双探测
#   print('[94ac7fe]')
print('[94ac7fe]')
