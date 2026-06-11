"""
walpurgis.utils.imports — 延迟可选依赖导入
上游迁移遗留: tensor/sampler子包依赖此模块的import_optional。
"""
import importlib


class MissingModule:
    """占位: 当可选依赖不可用时, 延迟到实际调用时才报错。"""
    def __init__(self, module_name):
        self._name = module_name

    def __getattr__(self, name):
        raise ImportError(
            f"Optional dependency '{self._name}' is not installed. "
            f"Install it with: pip install {self._name}")

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Optional dependency '{self._name}' is not installed.")


def import_optional(module_name):
    """尝试导入模块, 失败则返回MissingModule占位符。"""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return MissingModule(module_name)
