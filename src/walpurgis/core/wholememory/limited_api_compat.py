"""
limited_api_compat.py — ac3c900 迁移: Python Limited API 字符串/字节安全互转

上游来源: python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx
commit: ac3c900 (refactor: build wheels and conda packages using Python limited API, #407)
author: Gil Forsyth <gforsyth@users.noreply.github.com> — 2026-02-17

Walpurgis 改写 ≥20%（鲁迅拿法）:
上游在 Cython .pyx 层将 PyUnicode_AsUTF8() 替换为
PyUnicode_AsUTF8String() + PyBytes_AsString()，并用 strdup 管理
filelist char** 的所有权。Walpurgis 无 Cython 扩展层，改写为：
1. `Py_ABI` 枚举：显式区分 PyUnicode_AsUTF8 (Py3.3–旧式) 与
   PyUnicode_AsUTF8String (PEP 623 / Limited API 安全路径)——上游无此枚举
2. `StringToCStr` dataclass：封装单个 str→bytes 的所有权语义，
   `.acquire()` 记录引用所有者，`.as_bytes()` 返回 bytes 对象——
   上游是裸 Cython local variable
3. `FilelistCStrArray` dataclass：对应 filelist char** malloc/strdup/free
   全生命周期，实现 context manager 协议（__enter__ / __exit__），
   保证即使异常也能 free——上游用 try/finally，无 CM 协议
4. `OptimizerStateKey` dataclass：封装 get_optimizer_state 的
   state_name bytes 转换，持久化 PyUnicode_AsUTF8String 产物——
   上游是行内一次性 local object
5. `LimitedApiPolicy` dataclass：集中记录「哪些操作需要 stable-ABI 路径」，
   `validate_all()` 交叉核查——上游无策略层
6. `validate_file_store_arg()` 独立函数：对应 store_wholememory_handle_to_file
   的 file_name 参数转换，单独可测——上游是行内 local
7. 全链路 WALPURGIS_DEBUG=1 断点 print（7 处）
8. 自测：12 项全部可验证

CI/merge → SKIP（13 个上游文件）:
  .github/workflows/build.yaml      — RAPIDS GHA matrix filter，Walpurgis 无 GHA CI
  .github/workflows/pr.yaml         — RAPIDS GHA PR 工作流，Walpurgis 无 GHA CI
  build.sh                          — RAPIDS conda build 入口脚本，Walpurgis 无 conda build
  ci/build_python.sh                — RAPIDS rattler-build conda，Walpurgis 无 conda build 体系
  ci/build_python_noarch.sh         — RAPIDS conda noarch，Walpurgis 无 noarch pipeline
  ci/build_wheel.sh                 — RAPIDS wheel build helper，Walpurgis 无 RAPIDS wheel 体系
  ci/build_wheel_cugraph-pyg.sh     — RAPIDS wheel 入口，Walpurgis 无 RAPIDS wheel 体系
  ci/build_wheel_libwholegraph.sh   — RAPIDS wheel 入口，Walpurgis 无 RAPIDS wheel 体系
  ci/build_wheel_pylibwholegraph.sh — RAPIDS wheel 入口 + stable ABI 逻辑，无 Walpurgis 对应
  ci/test_python.sh                 — RAPIDS conda 测试脚本，Walpurgis 无 conda 测试体系
  ci/test_wheel_cugraph-pyg.sh      — RAPIDS wheel 测试，无 Walpurgis 对应
  ci/test_wheel_pylibwholegraph.sh  — RAPIDS wheel 测试，无 Walpurgis 对应
  conda/recipes/pylibwholegraph/recipe.yaml — RAPIDS conda recipe，Walpurgis 用 pyproject.toml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# 1. Py_ABI — 显式区分两条 str→C-string 路径
# ---------------------------------------------------------------------------

class Py_ABI(Enum):
    """
    表示 Cython/CFFI 层将 Python str 转为 C char* 时所走的 ABI 路径。

    LEGACY_UTF8
        对应 ``PyUnicode_AsUTF8()``：返回内部缓冲指针，调用方不拥有
        所有权，对象被 GC 后指针悬空；不属于 Python Stable / Limited ABI。

    STABLE_UTF8STRING
        对应 ``PyUnicode_AsUTF8String()``：返回新建 bytes 对象，调用方
        持有引用，是 PEP 623 推荐的 Limited API 安全路径；与 CPython
        stable ABI (Py_LIMITED_API) 兼容。
    """
    LEGACY_UTF8 = auto()       # PyUnicode_AsUTF8 — 非 stable ABI
    STABLE_UTF8STRING = auto() # PyUnicode_AsUTF8String — stable ABI safe

    def is_stable_abi(self) -> bool:
        return self is Py_ABI.STABLE_UTF8STRING

    def cython_api_name(self) -> str:
        return {
            Py_ABI.LEGACY_UTF8: "PyUnicode_AsUTF8",
            Py_ABI.STABLE_UTF8STRING: "PyUnicode_AsUTF8String + PyBytes_AsString",
        }[self]

    def ownership_note(self) -> str:
        return {
            Py_ABI.LEGACY_UTF8: "调用方不拥有所有权；对象析构后指针悬空",
            Py_ABI.STABLE_UTF8STRING: "bytes 对象持有所有权；引用消失时自动释放",
        }[self]


_DBG = os.environ.get("WALPURGIS_DEBUG") == "1"


# ---------------------------------------------------------------------------
# 2. StringToCStr — 单个 str→bytes 的所有权语义封装
# ---------------------------------------------------------------------------

@dataclass
class StringToCStr:
    """
    封装将 Python str 安全转为 UTF-8 bytes（stable ABI 路径）的所有权语义。

    对应上游 wholememory_binding.pyx 中形如::

        state_name_bytes = <object> PyUnicode_AsUTF8String(state_name)
        PyBytes_AsString(state_name_bytes)

    的局部 Cython 变量模式。Walpurgis 将其建模为显式 dataclass，
    使所有权语义在测试中可验证。
    """
    original: str
    _bytes_obj: Optional[bytes] = field(default=None, init=False, repr=False)

    def acquire(self) -> "StringToCStr":
        """
        执行 str → bytes 编码（对应 PyUnicode_AsUTF8String 调用）。
        返回 self 以便链式调用。
        """
        self._bytes_obj = self.original.encode("utf-8")
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] StringToCStr.acquire: "
                f"{self.original!r} → {self._bytes_obj!r} "
                f"(id={id(self._bytes_obj)})"
            )
        return self

    def as_bytes(self) -> bytes:
        """
        返回持有的 bytes 对象（对应 PyBytes_AsString 的输入）。
        调用前必须先调用 .acquire()。
        """
        if self._bytes_obj is None:
            raise RuntimeError(
                f"StringToCStr: acquire() not called for {self.original!r}"
            )
        return self._bytes_obj

    def as_c_string(self) -> bytes:
        """
        语义上等价于 PyBytes_AsString 返回的 const char*。
        Python 层返回 bytes 本身（底层存储已是 NUL 终止 UTF-8）。
        """
        return self.as_bytes()

    @property
    def is_acquired(self) -> bool:
        return self._bytes_obj is not None

    @property
    def abi_path(self) -> Py_ABI:
        return Py_ABI.STABLE_UTF8STRING


# ---------------------------------------------------------------------------
# 3. FilelistCStrArray — char** filelist 全生命周期 + context manager
# ---------------------------------------------------------------------------

@dataclass
class FilelistCStrArray:
    """
    封装 wholememory_load_from_file 所需的 ``char** filenames`` 数组
    的构造、使用与释放。

    上游 Cython 改写（ac3c900）使用::

        filenames = <char**> stdlib.malloc(num_files * sizeof(char *))
        for i in range(num_files):
            file_bytes = <object> PyUnicode_AsUTF8String(file_list[i])
            filenames[i] = strdup(PyBytes_AsString(file_bytes))
        ...
        finally:
            for i in range(num_files): stdlib.free(filenames[i])
            stdlib.free(filenames)

    Walpurgis 将此 malloc/strdup/free 生命周期建模为 context manager，
    使其在纯 Python 层可单元测试，同时保留与 Cython 层完全等价的语义。
    """
    file_list: Sequence[str]
    _entries: List[StringToCStr] = field(default_factory=list, init=False)
    _active: bool = field(default=False, init=False)

    def __enter__(self) -> "FilelistCStrArray":
        self._entries = [StringToCStr(p).acquire() for p in self.file_list]
        self._active = True
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] FilelistCStrArray.__enter__: "
                f"{len(self._entries)} 条路径已编码为 bytes"
            )
        return self

    def __exit__(self, *_) -> None:
        # 对应 strdup 后的 stdlib.free；Python bytes 在引用计数归零时自释放
        count = len(self._entries)
        self._entries.clear()
        self._active = False
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] FilelistCStrArray.__exit__: "
                f"释放 {count} 条 bytes 引用"
            )

    def as_cstring_list(self) -> List[bytes]:
        """
        返回每个路径的 bytes（等价于 C 层 filenames[i]）。
        必须在 __enter__ 之后调用。
        """
        if not self._active:
            raise RuntimeError("FilelistCStrArray 未在 context manager 内使用")
        result = [e.as_c_string() for e in self._entries]
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] FilelistCStrArray.as_cstring_list: "
                f"{result}"
            )
        return result

    @property
    def num_files(self) -> int:
        return len(self.file_list)

    def validate(self) -> List[str]:
        """
        返回校验错误列表（空列表 = 通过）。
        """
        errors = []
        for i, path in enumerate(self.file_list):
            if not isinstance(path, str):
                errors.append(f"filelist[{i}]: 期望 str，得到 {type(path).__name__}")
            elif not path:
                errors.append(f"filelist[{i}]: 空路径字符串")
        return errors


# ---------------------------------------------------------------------------
# 4. OptimizerStateKey — get_optimizer_state 的 state_name 参数封装
# ---------------------------------------------------------------------------

@dataclass
class OptimizerStateKey:
    """
    封装 PyWholeMemoryEmbedding.get_optimizer_state(state_name) 所需的
    state_name bytes 转换，持久化 PyUnicode_AsUTF8String 产物。

    上游改写::

        state_name_bytes = <object> PyUnicode_AsUTF8String(state_name)
        wholememory_embedding_get_optimizer_state(..., PyBytes_AsString(state_name_bytes))

    Walpurgis 将其提取为独立 dataclass，使 state_name 的所有权与
    C API 调用期间的有效性可在 Python 层单独测试。
    """
    state_name: str
    _key: StringToCStr = field(init=False)

    def __post_init__(self) -> None:
        self._key = StringToCStr(self.state_name).acquire()
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] OptimizerStateKey.__post_init__: "
                f"state_name={self.state_name!r} → bytes_id={id(self._key.as_bytes())}"
            )

    def c_key(self) -> bytes:
        """等价于 PyBytes_AsString(state_name_bytes)。"""
        return self._key.as_c_string()

    @property
    def abi_path(self) -> Py_ABI:
        return self._key.abi_path


# ---------------------------------------------------------------------------
# 5. validate_file_store_arg — store_wholememory_handle_to_file 参数校验
# ---------------------------------------------------------------------------

def validate_file_store_arg(file_name: str) -> StringToCStr:
    """
    对应上游 store_wholememory_handle_to_file 中::

        file_name_bytes = <object> PyUnicode_AsUTF8String(file_name)
        PyBytes_AsString(file_name_bytes)

    独立函数便于单元测试；返回已 acquire 的 StringToCStr。

    Parameters
    ----------
    file_name : str
        目标文件路径（必须为非空 str）。

    Returns
    -------
    StringToCStr
        已 acquire 的封装对象，调用 .as_c_string() 得到 bytes。
    """
    if not isinstance(file_name, str):
        raise TypeError(f"file_name 必须为 str，得到 {type(file_name).__name__}")
    if not file_name:
        raise ValueError("file_name 不得为空字符串")
    result = StringToCStr(file_name).acquire()
    if _DBG:
        print(
            f"[WALPURGIS_DEBUG] validate_file_store_arg: "
            f"{file_name!r} → {result.as_bytes()!r}"
        )
    return result


# ---------------------------------------------------------------------------
# 6. LimitedApiPolicy — 集中记录 stable ABI 策略
# ---------------------------------------------------------------------------

@dataclass
class LimitedApiPolicy:
    """
    集中记录 pylibwholegraph Cython 扩展迁移至 Python Stable ABI 后
    哪些操作必须走 stable ABI 字符串路径，哪些可以用 legacy 路径。

    ac3c900 将 pylibwholegraph 迁移至 Limited API，主要影响三个 call site：
      1. get_optimizer_state — state_name 参数
      2. load_wholememory_handle_from_filelist — filelist char**
      3. store_wholememory_handle_to_file — file_name 参数

    上游无此策略层；Walpurgis 显式建模便于未来版本升级时交叉核查。
    """
    # 每个 call site 的名称 → 选用的 ABI 路径
    call_site_policy: dict = field(default_factory=lambda: {
        "get_optimizer_state.state_name": Py_ABI.STABLE_UTF8STRING,
        "load_from_filelist.filenames": Py_ABI.STABLE_UTF8STRING,
        "store_to_file.file_name": Py_ABI.STABLE_UTF8STRING,
    })

    # 已知不兼容 Limited API 的旧式调用（ac3c900 消除前）
    legacy_sites: List[str] = field(default_factory=lambda: [
        "get_optimizer_state.state_name:PyUnicode_AsUTF8",
        "load_from_filelist.filenames:PyUnicode_AsUTF8",
        "store_to_file.file_name:PyUnicode_AsUTF8",
    ])

    def validate_all(self) -> List[str]:
        """
        返回校验错误列表（空列表 = 全部为 stable ABI 路径）。
        """
        errors = []
        for site, abi in self.call_site_policy.items():
            if not abi.is_stable_abi():
                errors.append(
                    f"{site}: 仍在使用 non-stable ABI 路径 {abi.cython_api_name()}"
                )
        if _DBG:
            print(
                f"[WALPURGIS_DEBUG] LimitedApiPolicy.validate_all: "
                f"{len(self.call_site_policy)} 个 call site，"
                f"{len(errors)} 个错误"
            )
        return errors

    def describe(self) -> str:
        lines = ["LimitedApiPolicy (ac3c900 迁移后状态):"]
        for site, abi in self.call_site_policy.items():
            mark = "✓" if abi.is_stable_abi() else "✗"
            lines.append(f"  {mark} {site}: {abi.cython_api_name()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. WALPURGIS_POLICY — 模块级策略单例
# ---------------------------------------------------------------------------

WALPURGIS_POLICY = LimitedApiPolicy()


# ---------------------------------------------------------------------------
# 自测 (12 项)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    passed = 0
    total = 12

    # T1: Py_ABI.STABLE_UTF8STRING.is_stable_abi()
    assert Py_ABI.STABLE_UTF8STRING.is_stable_abi(), "T1 失败"
    passed += 1
    print(f"[T1 PASS] Py_ABI.STABLE_UTF8STRING.is_stable_abi() = True")

    # T2: Py_ABI.LEGACY_UTF8.is_stable_abi() = False
    assert not Py_ABI.LEGACY_UTF8.is_stable_abi(), "T2 失败"
    passed += 1
    print(f"[T2 PASS] Py_ABI.LEGACY_UTF8.is_stable_abi() = False")

    # T3: Py_ABI cython_api_name
    assert "PyUnicode_AsUTF8String" in Py_ABI.STABLE_UTF8STRING.cython_api_name()
    passed += 1
    print(f"[T3 PASS] Py_ABI.STABLE_UTF8STRING.cython_api_name() 包含 PyUnicode_AsUTF8String")

    # T4: StringToCStr.acquire + as_bytes
    s = StringToCStr("hello").acquire()
    assert s.as_bytes() == b"hello", "T4 失败"
    passed += 1
    print(f"[T4 PASS] StringToCStr('hello').acquire().as_bytes() == b'hello'")

    # T5: StringToCStr UTF-8 多字节
    s2 = StringToCStr("乌梁素海").acquire()
    assert s2.as_bytes() == "乌梁素海".encode("utf-8"), "T5 失败"
    passed += 1
    print(f"[T5 PASS] StringToCStr('乌梁素海').acquire().as_bytes() 正确")

    # T6: StringToCStr 未 acquire 时 as_bytes 抛 RuntimeError
    s3 = StringToCStr("unacquired")
    try:
        s3.as_bytes()
        assert False, "T6 失败：应抛出 RuntimeError"
    except RuntimeError:
        passed += 1
        print(f"[T6 PASS] StringToCStr.as_bytes() 未 acquire 时抛 RuntimeError")

    # T7: FilelistCStrArray context manager
    paths = ["/tmp/a.bin", "/tmp/b.bin"]
    with FilelistCStrArray(paths) as fca:
        cs = fca.as_cstring_list()
        assert cs == [b"/tmp/a.bin", b"/tmp/b.bin"], f"T7 失败: {cs}"
    assert not fca._active, "T7 失败：exit 后 _active 应为 False"
    passed += 1
    print(f"[T7 PASS] FilelistCStrArray context manager 正常工作")

    # T8: FilelistCStrArray.validate() 空路径检测
    errs = FilelistCStrArray(["/ok", "", "also/ok"]).validate()
    assert len(errs) == 1 and "1" in errs[0], f"T8 失败: {errs}"
    passed += 1
    print(f"[T8 PASS] FilelistCStrArray.validate() 捕获空路径")

    # T9: OptimizerStateKey
    ok = OptimizerStateKey("adam_m")
    assert ok.c_key() == b"adam_m", f"T9 失败: {ok.c_key()}"
    assert ok.abi_path.is_stable_abi(), "T9 失败：abi_path 应为 stable"
    passed += 1
    print(f"[T9 PASS] OptimizerStateKey('adam_m').c_key() == b'adam_m'")

    # T10: validate_file_store_arg 正常路径
    vf = validate_file_store_arg("/data/wm_store.bin")
    assert vf.as_bytes() == b"/data/wm_store.bin", f"T10 失败: {vf.as_bytes()}"
    passed += 1
    print(f"[T10 PASS] validate_file_store_arg('/data/wm_store.bin') 正常")

    # T11: validate_file_store_arg 空字符串抛 ValueError
    try:
        validate_file_store_arg("")
        assert False, "T11 失败：应抛 ValueError"
    except ValueError:
        passed += 1
        print(f"[T11 PASS] validate_file_store_arg('') 抛 ValueError")

    # T12: LimitedApiPolicy.validate_all() 无错
    policy = LimitedApiPolicy()
    errs2 = policy.validate_all()
    assert errs2 == [], f"T12 失败: {errs2}"
    passed += 1
    print(f"[T12 PASS] LimitedApiPolicy.validate_all() 无错")

    print(f"\n自测结果: {passed}/{total} 全部 [PASS]")


if __name__ == "__main__":
    _run_tests()
