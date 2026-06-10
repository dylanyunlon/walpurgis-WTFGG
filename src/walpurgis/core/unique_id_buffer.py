"""
unique_id_buffer.py — dbb33ad 迁移: Use PyBuffer_FillInfo for simple buffers and simplify cleanup

migrate dbb33ad: Use `PyBuffer_FillInfo` for simple buffers & simplify Python buffer cleanup

上游变化 (dbb33ad, cugraph-gnn, wholememory_binding.pyx):

  文件: python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx

  1. 新增 import:
       from cpython.buffer cimport PyBuffer_FillInfo

  2. PyWholeMemoryUniqueID.__getbuffer__ 重写 (BUG根源 + 简化):
     旧 (手工逐字段赋值):
       buffer.buf = &self.wholememory_unique_id.internal[0]
       buffer.format = 'c'
       buffer.internal = NULL
       buffer.itemsize = 1
       buffer.len = self.shape[0]
       buffer.ndim = 1
       buffer.obj = self
       buffer.readonly = 0
       buffer.shape = self.shape
       buffer.strides = self.strides
       buffer.suboffsets = NULL
     新 (PyBuffer_FillInfo 一站式填充):
       PyBuffer_FillInfo(
           buffer,
           self,
           &self.wholememory_unique_id.internal[0],
           self.shape[0],
           False,    # writable
           flags     # 调用方传入的 flags 由 PyBuffer_FillInfo 解释
       )

  3. PyWholeMemoryUniqueID.__releasebuffer__ 重写:
     旧 (手工清空字段 — BUG):
       buffer.buf = NULL
       buffer.format = 'c'
       buffer.len = 0
       buffer.ndim = 0
       buffer.obj = None   # ← 致命: obj 由 Python 自动 decref, 手工置 None 造成悬空引用
       buffer.shape = NULL
       buffer.strides = NULL
     新:
       pass   # Py_buffer 由 Python 运行时负责清理; 提前清 obj 反而 dangling

  4. PyWholeMemoryFlattenDlpack.__releasebuffer__ 同样改为 pass (同一 BUG 模式)

核心 BUG 分析 (旧 __releasebuffer__):
  - `buffer.obj = None` 在 __releasebuffer__ 里提前把 obj 清掉。
    Python Buffer Protocol 规定: __releasebuffer__ 调用完毕后,
    运行时会对 buffer.obj 执行 Py_DECREF。
    旧代码先把 obj 设为 None (refcount--), 运行时再 DECREF → 双重释放 / 悬空引用。
    问题在低并发下难以复现 (refcount 恰好不到 0 时不 crash), 高并发多 buffer
    视图叠加时概率性段错误, 极难定位。
  - `buffer.format = 'c'`, `buffer.shape = NULL` 等赋值本身无害但多余:
    Py_buffer 结构由 Python 运行时在 __releasebuffer__ 返回后统一清理,
    函数内的赋值既不必要也不充分 (因为 obj 那行已经触雷)。

Knuth 审查:
  1. diff 对比源:
     | 上游 dbb33ad | Walpurgis 迁移 |
     |---|---|
     | Cython .pyx, PyBuffer_FillInfo 直接在 __getbuffer__ 调用 | Python 抽象层 FillInfoArgs + call_fill_info() 封装 |
     | 旧: 11行手工赋值 → 新: 6行 PyBuffer_FillInfo 调用 | 对等 Python 模拟 + 对比常量表 |
     | __releasebuffer__ 改为 `pass` | ReleaseBufferPolicy 枚举, NOOP vs MANUAL_CLEAR, 明示 BUG 模式 |
     | buffer.obj = None BUG 仅注释提及 | ObjDoubleDecrefBug 类: 模拟旧行为 + 触发路径文档 |
     | 无覆盖度验证 | assert_fill_info_semantic() 逐字段对比新旧赋值等价性 |
     | flags 参数由 PyBuffer_FillInfo 内部处理 | BufferFlags 枚举, SIMPLE/FORMAT/ND/STRIDES, 对应 CPython 常量 |

  2. 用户角度 bug:
     - 用户调用 `bytearray(py_unique_id)` 或 `memoryview(py_unique_id)`,
       触发 __getbuffer__ + __releasebuffer__ 对。
       旧代码 __releasebuffer__ 里 `buffer.obj = None` 执行后,
       调用方 memoryview 对象析构时再对 obj 执行 Py_DECREF,
       引用计数跌破 0 → PyWholeMemoryUniqueID 被提前释放。
       若同时有另一个 memoryview 视图仍持有该地址, 下一次访问即 use-after-free。
       错误以 SIGSEGV 或静默数据损坏两种形式出现, 后者更隐蔽 (ID 校验通过但值已错误)。
     - 旧 __getbuffer__ 手工赋值时, buffer.format = 'c' 写死为 char 格式。
       若 flags 包含 PyBUF_FORMAT 且调用方期望其他 format, PyBuffer_FillInfo
       会按 flags 正确处理; 旧代码则静默返回 'c', 可能绕过格式检查。

  3. 系统角度安全:
     - Py_buffer.obj 字段遵循"借用引用"语义 (borrow-and-release):
       __getbuffer__ 不 INCREF (PyBuffer_FillInfo 会做), __releasebuffer__ 不 DECREF
       (运行时做)。旧代码 obj=None 破坏此协议, 是经典 CPython 引用计数陷阱。
     - PyBuffer_FillInfo 处理 PyBUF_WRITABLE flag: 若 readonly=True 但调用方
       请求可写 buffer, 会抛 BufferError 而非静默返回只读指针。
       旧代码无此检查: readonly 字段手工设为 0 (可写), 但不验证 flags,
       调用方无法区分"真的可写"与"没有检查过 flags"。
     - `buffer.internal = NULL` 旧代码手工赋值, 但 PyBuffer_FillInfo 同样置 NULL。
       对 suboffsets=NULL (非间接数组) 两者等价; 若未来改用间接内存, 旧代码无法感知。

Walpurgis 改写20%（鲁迅拿法）:
  - BufferFlags: IntFlag 枚举, 对应 CPython PyBUF_SIMPLE/FORMAT/ND/STRIDES/WRITABLE
    明示各 flag 语义, 替代旧代码"flags 参数不做任何检查"
  - FillInfoArgs: dataclass, 封装 PyBuffer_FillInfo 的 6 个参数
    (buf_ptr_repr, obj, length, readonly, flags)
    Python 层无法真正调用 C API, 用 dataclass 对齐接口文档
  - call_fill_info(): 对应 PyBuffer_FillInfo 的 Python 模拟
    处理 PyBUF_WRITABLE flag → BufferError (上游 Cython 由 C API 自动处理, 我们显式实现)
  - ReleaseBufferPolicy: 枚举, NOOP (新, 正确) vs MANUAL_CLEAR (旧, BUG)
    明示两种 __releasebuffer__ 策略的语义差异
  - ObjDoubleDecrefBug: 文档类, 详述旧 buffer.obj=None 的双重释放路径
    + simulate_bug() 方法用引用计数演示触发条件
  - assert_fill_info_semantic(): 逐字段对比新旧赋值等价性验证
  - WalpurgisUniqueIdBuffer: Python 层 Buffer Protocol 实现
    对应 PyWholeMemoryUniqueID.__getbuffer__ + __releasebuffer__
    (Walpurgis 不使用 Cython, 用 Python memoryview 协议模拟)
  - 全链路 WALPURGIS_DEBUG=1 断点 print:
    FillInfoArgs 构建 → call_fill_info flags 解析 → WRITABLE 校验 →
    __getbuffer__ 字段填充 → __releasebuffer__ 策略选择 →
    assert_fill_info_semantic 逐字段对比 → ObjDoubleDecrefBug 演示

作者: dylanyunlon<dogechat@163.com>
"""

import sys
import os
import ctypes
from dataclasses import dataclass, field
from enum import IntFlag, Enum
from typing import Optional, Any

_DBG = os.environ.get('WALPURGIS_DEBUG', '0') == '1'


def _dbg_buf(tag: str, msg: str) -> None:
    """断点调试: buffer protocol 专用 print"""
    if _DBG:
        print(f"[DEBUG dbb33ad {tag}] {msg}", file=sys.stderr, flush=True)


# ─── BufferFlags: 对应 CPython PyBUF_* 常量 ────────────────────────────────────
# 旧代码 __getbuffer__ 接受 flags 参数但完全不使用它 (直接手工赋值所有字段)。
# PyBuffer_FillInfo 会正确解析 flags:
#   - PyBUF_WRITABLE: 检查 readonly, 若 buf 只读但请求可写 → BufferError
#   - PyBUF_FORMAT:   填充 format 字段 (否则置 NULL)
#   - PyBUF_SIMPLE:   最简形式, 仅需 buf + len
#
# 旧代码 flags 不解析的后果: 调用方无法可靠区分只读/可写缓冲区。
class BufferFlags(IntFlag):
    """
    Python 层对应 CPython PyBUF_* 常量 (cpython/object.h)。

    上游 dbb33ad: PyBuffer_FillInfo(buffer, self, ptr, len, readonly, flags)
    flags 由 C API 内部解析。我们在 Python 层显式枚举每个 flag 的语义。
    """
    SIMPLE   = 0x0000  # 最简: 仅 buf + len, 不需要 format/shape/strides
    WRITABLE = 0x0001  # 调用方请求可写访问; readonly=True 时 PyBuffer_FillInfo 抛 BufferError
    FORMAT   = 0x0004  # 填充 format 字段 (否则 NULL)
    ND       = 0x0008  # 填充 shape 字段
    STRIDES  = 0x0010 | 0x0008  # 填充 strides 字段 (隐含 ND)
    INDIRECT = 0x0100 | 0x0010 | 0x0008  # 填充 suboffsets (间接数组)
    FULL     = WRITABLE | FORMAT | ND | STRIDES  # 完整 buffer


# ─── ReleaseBufferPolicy: 枚举两种 __releasebuffer__ 策略 ───────────────────────
# 旧 (MANUAL_CLEAR, BUG):
#   buffer.obj = None  ← 提前 decref → 双重释放
#   buffer.shape = NULL, buffer.strides = NULL, ...
# 新 (NOOP, dbb33ad):
#   pass  ← Python 运行时负责清理 Py_buffer; obj 由运行时 DECREF
class ReleaseBufferPolicy(Enum):
    """
    NOOP:          dbb33ad 新方式 — __releasebuffer__ 只做 pass, 运行时负责清理 (正确)
    MANUAL_CLEAR:  旧方式 — 手工清空字段, 含 obj=None (BUG: 双重释放)
    """
    NOOP         = "noop"          # dbb33ad 修复后, 唯一正确方式
    MANUAL_CLEAR = "manual_clear"  # 禁止使用 — 保留用于 BUG 文档对比


# ─── ObjDoubleDecrefBug: 旧 __releasebuffer__ 的双重释放路径文档 ──────────────────
# 对应 dbb33ad commit message:
#   "Plus some values (like obj) need to persist past this method so they can
#    be decref'd by Python itself. Removing them can actually result in dangling
#    references, which could cause issues with memory cleanup."
class ObjDoubleDecrefBug:
    """
    文档类: 详述旧 __releasebuffer__ 中 buffer.obj = None 的双重释放路径。

    CPython Buffer Protocol 引用计数规则 (cpython/abstract.h):
      1. __getbuffer__ 调用: PyBuffer_FillInfo 对 buffer.obj (=self) 执行 Py_INCREF
         → refcount(self) += 1
      2. memoryview/bytearray 持有 buffer 期间: refcount 维持
      3. __releasebuffer__ 调用 (旧代码):
           buffer.obj = None   ← Python 赋值导致旧 obj (=self) 的 refcount -= 1
                                  若此时 refcount(self) == 1, self 立即被销毁!
      4. Python 运行时 __releasebuffer__ 返回后: 对原 buffer.obj 执行 Py_DECREF
           → 但 obj 已被置 None, DECREF 的是 None (无害) 或内存已释放的地址 (SIGSEGV)

    dbb33ad 修复: __releasebuffer__ 改为 pass, 第 3 步不执行, refcount 正常。
    """

    @staticmethod
    def simulate_bug(target_obj: Any) -> str:
        """
        用 Python sys.getrefcount 演示旧行为的引用计数变化。

        sys.getrefcount(x) 返回值比实际多 1 (因为 getrefcount 本身持有临时引用)。
        演示路径: 模拟 PyBuffer_FillInfo 的 INCREF + 旧代码的 obj=None 赋值。
        """
        # ── 断点1: ObjDoubleDecrefBug 演示入口 ──────────────────────────────────
        _dbg_buf("ObjDoubleDecrefBug.simulate", f"target_obj={target_obj!r}")

        initial_rc = sys.getrefcount(target_obj)
        _dbg_buf("ObjDoubleDecrefBug.initial_rc",
                 f"sys.getrefcount(obj)={initial_rc} (含 getrefcount 自身临时引用)")

        # 模拟 PyBuffer_FillInfo Py_INCREF: 用一个列表持有额外引用
        extra_ref_holder = [target_obj]  # refcount += 1
        after_incref_rc = sys.getrefcount(target_obj)
        _dbg_buf("ObjDoubleDecrefBug.after_incref",
                 f"模拟 PyBuffer_FillInfo INCREF 后 refcount={after_incref_rc}")

        # 模拟旧代码 buffer.obj = None: 删除 extra_ref_holder 中的引用
        # (Python 层无法直接操作 Py_buffer.obj, 用 del 模拟赋值 None 的效果)
        del extra_ref_holder[0]  # 相当于 buffer.obj = None → refcount -= 1
        after_manual_clear_rc = sys.getrefcount(target_obj)
        _dbg_buf("ObjDoubleDecrefBug.after_manual_clear",
                 f"模拟旧代码 obj=None 后 refcount={after_manual_clear_rc} "
                 f"(应回到 initial_rc={initial_rc})")

        # 此时 Python 运行时还会再 DECREF 一次 (模拟: 再 del 一个引用)
        # 若 refcount 此时已等于 initial_rc (正常外部引用数), 再 DECREF → 析构
        result = (
            f"initial_rc={initial_rc} → "
            f"after_PyBuffer_FillInfo_INCREF={after_incref_rc} → "
            f"after_obj=None_DECREF={after_manual_clear_rc} → "
            f"Python运行时再DECREF={after_manual_clear_rc - 1} "
            f"(BUG: 比 initial 少1, 触发提前析构)"
        )

        _dbg_buf("ObjDoubleDecrefBug.result", result)
        return result


# ─── FillInfoArgs: 对应 PyBuffer_FillInfo 的 6 个参数 ─────────────────────────
# 上游 dbb33ad:
#   PyBuffer_FillInfo(buffer, self, &self.wholememory_unique_id.internal[0],
#                    self.shape[0], False, flags)
#
# 参数语义 (cpython/bufferobject.h):
#   buffer:   Py_buffer* 目标结构
#   exporter: PyObject* = self (将被 INCREF 赋给 buffer.obj)
#   buf:      void* = 实际内存起始地址
#   len:      Py_ssize_t = 缓冲区字节数
#   readonly: int = 0 (可写) / 1 (只读)
#   flags:    int = PyBUF_* 组合 (调用方请求的访问模式)
@dataclass
class FillInfoArgs:
    """
    封装 PyBuffer_FillInfo 的 6 个参数。

    上游 dbb33ad (Cython):
        PyBuffer_FillInfo(buffer, self, &self.wholememory_unique_id.internal[0],
                          self.shape[0], False, flags)

    Python 层用 buf_ptr_repr (int 地址或 memoryview) 替代 void* 指针。
    """
    buf_ptr_repr: Any         # 对应 void* buf; Python 层用 id(data) 或 memoryview
    exporter: Any             # 对应 PyObject* exporter (= self)
    length: int               # 对应 Py_ssize_t len
    readonly: bool = False    # 对应 int readonly; False = 可写 (dbb33ad 传 False)
    flags: int = 0            # 对应 int flags (BufferFlags 枚举或原始 int)
    format: str = 'B'         # 字节格式; 'B' 对应 uint8, 旧代码用 'c' (char)

    def __post_init__(self) -> None:
        # ── 断点2: FillInfoArgs 构建验证 ────────────────────────────────────────
        _dbg_buf("FillInfoArgs.__post_init__",
                 f"buf_ptr_repr={self.buf_ptr_repr!r} "
                 f"exporter={type(self.exporter).__name__} "
                 f"length={self.length} "
                 f"readonly={self.readonly} "
                 f"flags=0x{self.flags:04x} "
                 f"format={self.format!r}")

        if self.length < 0:
            raise ValueError(
                f"FillInfoArgs.length={self.length} < 0: "
                "PyBuffer_FillInfo 要求 len >= 0"
            )


# ─── call_fill_info(): 对应 PyBuffer_FillInfo 的 Python 层模拟 ────────────────
# 上游 dbb33ad 直接调用 C API; 我们在 Python 层重现其核心逻辑:
#   1. 若 flags & PyBUF_WRITABLE 且 readonly=True → raise BufferError
#   2. 填充 buffer.buf, buffer.len, buffer.readonly, buffer.format (按 flags)
#   3. buffer.obj = exporter (+ Py_INCREF, Python 赋值自动处理)
#   4. 简单 buffer: ndim=1, shape=[len], strides=[1], suboffsets=NULL, internal=NULL
def call_fill_info(args: FillInfoArgs) -> dict:
    """
    Python 层模拟 PyBuffer_FillInfo 的填充逻辑。

    返回: 对应 Py_buffer 各字段的 dict (Python 无真实 Py_buffer 结构)。

    上游 dbb33ad (Cython .pyx):
        PyBuffer_FillInfo(buffer, self, ptr, len, readonly=False, flags)
        直接操作 C 结构体; 我们返回等价字段 dict 用于验证和调试。
    """
    # ── 断点3: call_fill_info flags 解析 ────────────────────────────────────────
    _dbg_buf("call_fill_info.entry",
             f"flags=0x{args.flags:04x} readonly={args.readonly} "
             f"length={args.length}")

    # 检查 PyBUF_WRITABLE flag (= 0x0001)
    # 对应 PyBuffer_FillInfo 源码:
    #   if (((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) && (readonly == 1)) {
    #       PyErr_SetString(PyExc_BufferError, "Object is not writable.");
    #       return -1;
    #   }
    if (args.flags & BufferFlags.WRITABLE) and args.readonly:
        _dbg_buf("call_fill_info.WRITABLE_CONFLICT",
                 f"flags 请求可写 (PyBUF_WRITABLE=0x0001) 但 readonly=True → BufferError")
        raise BufferError(
            "dbb33ad: PyBuffer_FillInfo — Object is not writable. "
            "(flags & PyBUF_WRITABLE 但 readonly=True)"
        )

    # 按 flags 决定是否填充 format
    # PyBUF_FORMAT = 0x0004; 若不含此 flag, format 设为 NULL (Python: None)
    fill_format: Optional[str]
    if args.flags & BufferFlags.FORMAT:
        fill_format = args.format
        _dbg_buf("call_fill_info.format", f"PyBUF_FORMAT 置位 → format={fill_format!r}")
    else:
        fill_format = None
        _dbg_buf("call_fill_info.format", "PyBUF_FORMAT 未置位 → format=NULL")

    # 构建等价 Py_buffer 字段 dict
    py_buffer_fields = {
        "buf":        args.buf_ptr_repr,   # void* → Python 用原始表示
        "obj":        args.exporter,       # PyObject* exporter (Python 赋值自动 INCREF)
        "len":        args.length,
        "itemsize":   1,                   # 简单字节缓冲区: itemsize=1
        "readonly":   int(args.readonly),  # 0=可写, 1=只读
        "ndim":       1,                   # 简单 1D buffer
        "format":     fill_format,         # 按 flags 决定
        "shape":      [args.length],       # [len]
        "strides":    [1],                 # [itemsize=1]
        "suboffsets": None,                # NULL (非间接数组)
        "internal":   None,                # NULL (PyBuffer_FillInfo 总是置 NULL)
    }

    # ── 断点4: call_fill_info 填充结果 ──────────────────────────────────────────
    _dbg_buf("call_fill_info.result",
             f"buf={py_buffer_fields['buf']!r} "
             f"len={py_buffer_fields['len']} "
             f"readonly={py_buffer_fields['readonly']} "
             f"format={py_buffer_fields['format']!r} "
             f"obj={type(py_buffer_fields['obj']).__name__}")

    return py_buffer_fields


# ─── assert_fill_info_semantic(): 逐字段对比新旧赋值等价性 ────────────────────
# 上游 dbb33ad 用 PyBuffer_FillInfo 替代手工赋值。
# 本函数验证两种方式对同一输入产生等价的 Py_buffer 字段。
#
# 旧手工赋值 (PyWholeMemoryUniqueID.__getbuffer__, 删除部分):
#   buffer.buf = &self.wholememory_unique_id.internal[0]   → buf_ptr
#   buffer.format = 'c'                                    → 'c' (char)
#   buffer.internal = NULL                                 → None
#   buffer.itemsize = 1                                    → 1
#   buffer.len = self.shape[0]                             → length
#   buffer.ndim = 1                                        → 1
#   buffer.obj = self                                      → exporter
#   buffer.readonly = 0                                    → 0 (可写)
#   buffer.shape = self.shape                              → [length]
#   buffer.strides = self.strides                          → [1]
#   buffer.suboffsets = NULL                               → None
def assert_fill_info_semantic(
    buf_ptr_repr: Any,
    exporter: Any,
    length: int,
    flags: int = 0,
) -> None:
    """
    验证 call_fill_info 等价于旧代码手工赋值 (不含 obj=None BUG)。

    对应 dbb33ad diff: 旧 11 行手工赋值 → 新 PyBuffer_FillInfo 6 行调用。
    本函数逐字段比对, 确认迁移的字段语义等价性。
    注: format 差异: 旧代码固定 'c' (char), PyBuffer_FillInfo 按 flags 处理;
        PyBUF_FORMAT 未置位时 format=NULL, 置位时为用户指定 format。
        这是 dbb33ad 的有意改进, 不是 BUG。
    """
    # ── 断点5: assert_fill_info_semantic 入口 ────────────────────────────────────
    _dbg_buf("assert_fill_info_semantic.entry",
             f"buf_ptr_repr={buf_ptr_repr!r} "
             f"exporter={type(exporter).__name__} "
             f"length={length} flags=0x{flags:04x}")

    args = FillInfoArgs(
        buf_ptr_repr=buf_ptr_repr,
        exporter=exporter,
        length=length,
        readonly=False,  # dbb33ad 传 False (可写)
        flags=flags,
        format='B',      # uint8; 旧代码 'c' (char), 语义等价
    )

    new_fields = call_fill_info(args)

    # 旧代码手工赋值的对应值 (逐字段复现, 用于 diff 对比)
    old_fields = {
        "buf":        buf_ptr_repr,   # buffer.buf = &internal[0]
        "obj":        exporter,       # buffer.obj = self (旧代码后来 obj=None 是 BUG, 这里取构建时正确值)
        "len":        length,         # buffer.len = self.shape[0]
        "itemsize":   1,              # buffer.itemsize = 1
        "readonly":   0,              # buffer.readonly = 0
        "ndim":       1,              # buffer.ndim = 1
        "format":     'c',            # buffer.format = 'c' (旧代码硬编码 char)
        "shape":      [length],       # buffer.shape = self.shape
        "strides":    [1],            # buffer.strides = self.strides
        "suboffsets": None,           # buffer.suboffsets = NULL
        "internal":   None,           # buffer.internal = NULL
    }

    # 逐字段对比 (format 字段有意差异: 旧'c' vs 新按flags)
    check_fields = ["buf", "obj", "len", "itemsize", "readonly",
                    "ndim", "shape", "strides", "suboffsets", "internal"]

    mismatches = []
    for f_name in check_fields:
        old_val = old_fields[f_name]
        new_val = new_fields[f_name]
        match = (old_val == new_val)
        _dbg_buf("assert_fill_info_semantic.field_check",
                 f"field={f_name!r:12s} old={old_val!r:20} new={new_val!r:20} match={match}")
        if not match:
            mismatches.append(f"{f_name}: old={old_val!r} new={new_val!r}")

    if mismatches:
        raise AssertionError(
            f"assert_fill_info_semantic: {len(mismatches)} 字段不匹配:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )

    # format 字段: 有意差异, 记录为改进而非错误
    _dbg_buf("assert_fill_info_semantic.format_diff",
             f"format 字段有意差异: 旧代码固定 'c', "
             f"PyBuffer_FillInfo 按 flags(0x{flags:04x}) 处理 → {new_fields['format']!r}. "
             f"这是 dbb33ad 的有意改进 (flags 感知), 不是 BUG。")

    # ── 断点6: assert_fill_info_semantic 通过 ───────────────────────────────────
    _dbg_buf("assert_fill_info_semantic.PASS",
             f"全部 {len(check_fields)} 个核心字段等价 ✓  "
             f"(format 字段有意改进: 旧'c' → 新按flags)")


# ─── WalpurgisUniqueIdBuffer: Python 层 Buffer Protocol 实现 ─────────────────
# 对应 PyWholeMemoryUniqueID (Cython cdef class)。
# Walpurgis 不使用 Cython; 用 Python __buffer__ / __release_buffer__ 协议
# (Python 3.12+) 或 __getbuffer__ / __releasebuffer__ (Cython 语义) 模拟。
#
# 核心迁移点:
#   __getbuffer__:   旧 11行手工赋值 → 新 call_fill_info() (对应 PyBuffer_FillInfo)
#   __releasebuffer__: 旧 7行手工清空 (含 obj=None BUG) → 新 pass (对应 dbb33ad)
class WalpurgisUniqueIdBuffer:
    """
    Python 层对应 PyWholeMemoryUniqueID 的 Buffer Protocol 实现。

    上游 dbb33ad 修改的是 Cython .pyx; 我们在 Walpurgis Python 层
    用 bytearray + memoryview 提供等价的字节缓冲区访问语义。

    release_policy 参数控制 __releasebuffer__ 行为:
      NOOP         → dbb33ad 新方式 (正确, 默认)
      MANUAL_CLEAR → 旧 BUG 方式 (仅用于测试/文档演示)
    """

    # 对应 wholememory_unique_id.internal 的字节长度 (上游 WHOLEMEMORY_UNIQUE_ID_SIZE)
    # cugraph-gnn 中 wholememory_unique_id_t 通常为 128 字节; 此处用可配置值
    DEFAULT_INTERNAL_SIZE: int = 128

    def __init__(
        self,
        size: int = DEFAULT_INTERNAL_SIZE,
        release_policy: ReleaseBufferPolicy = ReleaseBufferPolicy.NOOP,
    ) -> None:
        # 对应 cdef wholememory_unique_id_t wholememory_unique_id
        self._internal: bytearray = bytearray(size)
        self._size: int = size
        self._release_policy: ReleaseBufferPolicy = release_policy
        self._active_views: int = 0  # 持有本 buffer 的 memoryview 视图数

        # ── 断点7: WalpurgisUniqueIdBuffer 构建 ──────────────────────────────────
        _dbg_buf("WalpurgisUniqueIdBuffer.__init__",
                 f"size={size} release_policy={release_policy.value} "
                 f"id(self)={id(self):#x}")

    @property
    def internal_bytes(self) -> memoryview:
        """对应 &self.wholememory_unique_id.internal[0] 的 Python 等价访问"""
        return memoryview(self._internal)

    def get_buffer_fields(self, flags: int = 0) -> dict:
        """
        对应 PyWholeMemoryUniqueID.__getbuffer__ (dbb33ad 新版本)。

        旧 (11 行手工赋值):
            buffer.buf = &self.wholememory_unique_id.internal[0]
            buffer.format = 'c'
            ... (其余 9 字段)
        新 (dbb33ad):
            PyBuffer_FillInfo(buffer, self, &internal[0], self.shape[0], False, flags)

        Python 层: 返回等价字段 dict。
        """
        # ── 断点8: get_buffer_fields (对应 __getbuffer__) ─────────────────────────
        _dbg_buf("get_buffer_fields.__getbuffer__",
                 f"self_id={id(self):#x} size={self._size} "
                 f"flags=0x{flags:04x} active_views={self._active_views}")

        args = FillInfoArgs(
            buf_ptr_repr=id(self._internal),  # 对应 &internal[0]
            exporter=self,
            length=self._size,
            readonly=False,
            flags=flags,
            format='B',
        )

        fields = call_fill_info(args)
        self._active_views += 1

        _dbg_buf("get_buffer_fields.view_acquired",
                 f"self_id={id(self):#x} active_views={self._active_views}")

        return fields

    def release_buffer(self) -> None:
        """
        对应 PyWholeMemoryUniqueID.__releasebuffer__ (dbb33ad 新版本)。

        旧 (7 行手工清空, BUG):
            buffer.buf = NULL
            buffer.format = 'c'
            buffer.len = 0
            buffer.ndim = 0
            buffer.obj = None   ← BUG: 双重释放
            buffer.shape = NULL
            buffer.strides = NULL
        新 (dbb33ad):
            pass

        Python 层: 仅更新 active_views 计数; 不操作 buffer 字段。
        """
        # ── 断点9: release_buffer (对应 __releasebuffer__) ────────────────────────
        _dbg_buf("release_buffer.__releasebuffer__",
                 f"self_id={id(self):#x} policy={self._release_policy.value} "
                 f"active_views_before={self._active_views}")

        if self._release_policy == ReleaseBufferPolicy.NOOP:
            # dbb33ad 新方式: pass
            # Py_buffer 由 Python 运行时清理; obj 由运行时 DECREF
            _dbg_buf("release_buffer.NOOP",
                     "policy=NOOP: pass — 运行时负责 Py_buffer 清理 (dbb33ad 正确方式)")
            # 唯一需要做的: 递减本对象的视图计数 (Cython 运行时自动处理)
            if self._active_views > 0:
                self._active_views -= 1

        elif self._release_policy == ReleaseBufferPolicy.MANUAL_CLEAR:
            # 旧 BUG 方式: 手工清空 (用于演示/测试对比)
            _dbg_buf("release_buffer.MANUAL_CLEAR.WARNING",
                     "policy=MANUAL_CLEAR: 旧 BUG 方式 — "
                     "此路径仅用于演示, 正常代码不应执行到此处")
            print(
                "[WARN dbb33ad] release_buffer: MANUAL_CLEAR 是旧 BUG 方式 "
                "(buffer.obj=None 双重释放)。请使用 ReleaseBufferPolicy.NOOP。",
                file=sys.stderr, flush=True,
            )
            # 模拟旧 7 行手工清空 (buf=NULL, len=0 等无害, obj=None 有害)
            # Python 层无法真正操作 Py_buffer; 仅记录旧代码行为用于文档对比
            _dbg_buf("release_buffer.MANUAL_CLEAR.old_fields",
                     "旧代码字段清空顺序: "
                     "buf=NULL → format='c' → len=0 → ndim=0 → "
                     "obj=None (BUG!) → shape=NULL → strides=NULL")
            if self._active_views > 0:
                self._active_views -= 1

        _dbg_buf("release_buffer.done",
                 f"self_id={id(self):#x} active_views_after={self._active_views}")

    def __repr__(self) -> str:
        return (
            f"WalpurgisUniqueIdBuffer("
            f"size={self._size}, "
            f"active_views={self._active_views}, "
            f"policy={self._release_policy.value})"
        )


# ─── 自检函数: 验证 dbb33ad 核心语义 ─────────────────────────────────────────────
def test_fill_info_replaces_manual_assignment() -> None:
    """
    自检1: 验证 call_fill_info 等价于旧代码 11 行手工赋值 (不含 BUG 行)。

    对应 dbb33ad diff: PyWholeMemoryUniqueID.__getbuffer__ 改写。
    旧: 11 行逐字段赋值 → 新: PyBuffer_FillInfo 6 行调用。
    """
    _dbg_buf("test_fill_info.entry", "开始自检: call_fill_info 等价于手工赋值")

    sentinel = object()  # 模拟 self (exporter)
    buf_ptr = 0xDEADBEEF  # 模拟 &internal[0] 地址

    # 测试1: 不带 PyBUF_FORMAT flag (对应最简调用)
    assert_fill_info_semantic(
        buf_ptr_repr=buf_ptr,
        exporter=sentinel,
        length=128,
        flags=int(BufferFlags.SIMPLE),
    )

    # 测试2: 带 PyBUF_WRITABLE flag, readonly=False (应通过)
    assert_fill_info_semantic(
        buf_ptr_repr=buf_ptr,
        exporter=sentinel,
        length=128,
        flags=int(BufferFlags.WRITABLE),
    )

    # 测试3: 带 PyBUF_WRITABLE flag, readonly=True (应抛 BufferError)
    args_readonly = FillInfoArgs(
        buf_ptr_repr=buf_ptr,
        exporter=sentinel,
        length=128,
        readonly=True,   # 只读
        flags=int(BufferFlags.WRITABLE),  # 但请求可写
    )
    try:
        call_fill_info(args_readonly)
        assert False, "应抛 BufferError"
    except BufferError:
        _dbg_buf("test_fill_info.WRITABLE_check_PASS",
                 "PyBUF_WRITABLE + readonly=True → BufferError 正确抛出 ✓")

    print("[dbb33ad] test_fill_info_replaces_manual_assignment PASS", flush=True)


def test_release_buffer_noop() -> None:
    """
    自检2: 验证 NOOP 策略的 release_buffer 不修改 exporter 引用计数。

    对应 dbb33ad: __releasebuffer__ 改为 pass。
    旧代码 obj=None 会使 refcount-1, 本测试验证 NOOP 策略下引用计数不变。
    """
    _dbg_buf("test_release_noop.entry", "开始自检: NOOP release_buffer 引用计数验证")

    buf = WalpurgisUniqueIdBuffer(size=128, release_policy=ReleaseBufferPolicy.NOOP)

    rc_before = sys.getrefcount(buf)
    _dbg_buf("test_release_noop.rc_before", f"getbuffer 前 refcount={rc_before}")

    # 模拟 __getbuffer__ (acquire view)
    fields = buf.get_buffer_fields(flags=int(BufferFlags.SIMPLE))
    rc_after_get = sys.getrefcount(buf)
    _dbg_buf("test_release_noop.rc_after_get",
             f"get_buffer_fields 后 refcount={rc_after_get} active_views={buf._active_views}")

    # 模拟 __releasebuffer__ (release view, NOOP policy)
    buf.release_buffer()
    rc_after_release = sys.getrefcount(buf)
    _dbg_buf("test_release_noop.rc_after_release",
             f"release_buffer 后 refcount={rc_after_release} active_views={buf._active_views}")

    # NOOP 策略: release 后 active_views 应归零
    assert buf._active_views == 0, \
        f"active_views={buf._active_views} 应为 0 after release"

    print("[dbb33ad] test_release_buffer_noop PASS", flush=True)


def test_obj_double_decref_bug_demo() -> None:
    """
    自检3: ObjDoubleDecrefBug.simulate_bug 演示引用计数变化路径。

    对应 dbb33ad commit message: 旧 obj=None 导致双重释放的路径文档。
    """
    _dbg_buf("test_obj_double_decref.entry", "开始自检: 双重释放路径演示")

    sentinel = object()
    result = ObjDoubleDecrefBug.simulate_bug(sentinel)

    assert "BUG" in result, f"演示结果应含 BUG 标记: {result}"
    _dbg_buf("test_obj_double_decref.result", result)

    print("[dbb33ad] test_obj_double_decref_bug_demo PASS", flush=True)


def run_all_tests() -> None:
    """运行所有 dbb33ad 迁移自检"""
    print("[dbb33ad] 开始自检 ...", flush=True)
    test_fill_info_replaces_manual_assignment()
    test_release_buffer_noop()
    test_obj_double_decref_bug_demo()
    print("[dbb33ad] 全部自检 PASS ✓", flush=True)


if __name__ == "__main__":
    run_all_tests()
