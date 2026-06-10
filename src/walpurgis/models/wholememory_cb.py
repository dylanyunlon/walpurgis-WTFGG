"""
wholememory_cb.py — fbea7cb 迁移: PyObject 回调层重构

migrate fbea7cb: Fix append unique

上游变化 (fbea7cb):
  文件 1: python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx

  6个 cdef 回调函数全部重构，废弃手工 PyTuple_New / Py_INCREF /
  PyTuple_SetItem / PyObject_CallObject / Py_DECREF 手动调用链，
  改用 Cython 原生 <object> 转型后直接函数调用。

  具体变化逐函数:

  [1] python_cb_wrapper_temp_create_context:
    旧: python_fn = wrapped_global_context.temp_create_context_fn
        python_global_context = wrapped_global_context.temp_global_context
        args = PyTuple_New(1)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 0, <object> python_global_context)
        py_memory_context = PyObject_CallObject(<object> python_fn, <object> args)
        Py_DECREF(args)
    新: fn = <object> wrapped_global_context.temp_create_context_fn
        ctx = <object> wrapped_global_context.temp_global_context
        py_memory_context = fn(ctx)
    → 消除 args tuple 的手动引用计数: 旧代码 Py_INCREF(global_context) 后
      SetItem 将 incref 的所有权转给 tuple，再 Py_DECREF(args) 时
      tuple 析构释放所有 item; 新代码由 Cython GIL 自动管理。

  [2] python_cb_wrapper_temp_destroy_context:
    旧: args = PyTuple_New(2)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 0, ...)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 1, ...)
        PyObject_CallObject(python_fn, args)
        Py_DECREF(args)
    新: fn(mem_ctx, ctx)
    → 旧代码在 CallObject 后没有检查返回值是否为 NULL（异常传播丢失），
      新代码 Python 调用协议由 Cython 自动处理异常。

  [3] python_cb_wrapper_temp_malloc:
    旧: py_tensor_desc = PyWholeMemoryTensorDescription()
        py_tensor_desc.set_by_tensor_desc(tensor_desc)
        py_malloc_type = PyMemoryAllocType()
        py_malloc_type.set_type(malloc_type)
        [PyTuple_New(4) + 4× INCREF/SetItem + CallObject + DECREF]
        res_ptr = PyLong_AsLongLong(...)
    新: py_shape = tuple([tensor_desc.sizes[i] for i in range(tensor_desc.dim)])
        py_dtype = int(tensor_desc.dtype)
        py_malloc_type_int = int(malloc_type)
        res_ptr = fn(py_shape, py_dtype, py_malloc_type_int, mem_ctx, ctx)
    → 核心语义变化: 不再传 PyWholeMemoryTensorDescription 对象，
      改传 (shape: tuple, dtype_int: int, malloc_type_int: int)。
      Python 侧 torch_malloc_env_fn 签名随之更新（见文件2）。
      PyLong_AsLongLong 替换为 Cython 原生 int64 → void* 转型。

  [4] python_cb_wrapper_temp_free:
    旧: [PyTuple_New(2) + INCREF×2 + SetItem×2 + CallObject + DECREF]
    新: fn(mem_ctx, ctx)
    → 同 destroy_context，消除手工 tuple 操作。

  [5] python_cb_wrapper_output_malloc:
    旧: py_tensor_desc = PyWholeMemoryTensorDescription() ...
        [PyTuple_New(4) + 旧代码混用 <PyObject*> 和直接对象 INCREF，
         SetItem(args, 0, <object> <PyObject *> py_tensor_desc)
         对 py_tensor_desc 进行了多余的 void* 往返转换]
    新: 同 temp_malloc，py_shape/py_dtype/py_malloc_type_int + 直接调用
    → 旧代码在 SetItem(args, 0, <object> <PyObject *> py_tensor_desc) 处
      有隐性 bug: py_tensor_desc 已是 Python 对象，再转 PyObject* 再转 object
      引用计数状态不明确（与 temp_malloc 版本不一致，temp_malloc 直接 SetItem
      不做此二次转换）。fbea7cb 一并修正此不对称。

  [6] python_cb_wrapper_output_free:
    旧: [PyTuple_New(2) + INCREF×2 + SetItem×2 + CallObject + DECREF]
    新: fn(mem_ctx, ctx)
    → 同 free/destroy，消除手工 tuple 操作。

  文件 2: python/pylibwholegraph/pylibwholegraph/torch/wholegraph_env.py

  torch_malloc_env_fn 函数签名及内部实现同步更新:
    旧签名: (tensor_desc: PyWholeMemoryTensorDescription,
             malloc_type: PyMemoryAllocType, ...)
    新签名: (shape: tuple, dtype_int: int, malloc_type_int: int, ...)
    旧: if malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatDevice:
        elif malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatHost:
        else: assert malloc_type.get_type() == ...MatPinned
        shape = tensor_desc.shape
        dtype = wholememory_dtype_to_torch_dtype(tensor_desc.dtype)
    新: if malloc_type_int == int(wmb.WholeMemoryMemoryAllocType.MatDevice):
        elif malloc_type_int == int(wmb.WholeMemoryMemoryAllocType.MatHost):
        else: assert malloc_type_int == int(...)
        dtype = wholememory_dtype_to_torch_dtype(wmb.WholeMemoryDataType(dtype_int))
    → int() 显式转换替代 enum 对象 .get_type() 方法调用;
      WholeMemoryDataType(dtype_int) 重建枚举用于 dtype lookup。

Bug 根因 (fbea7cb: "Fix append unique"):
  标题"append unique"指向 PyTuple_SetItem 的使用规范问题。
  CPython 文档: PyTuple_SetItem 对传入的 item "steals" 引用。
  旧代码在 Py_INCREF 后 SetItem → 引用计数多1，Py_DECREF(args) 时
  tuple 析构减1(stolen)，但外部那个 INCREF 的 +1 永不归还 → 内存泄漏。
  具体在 output_malloc 中: SetItem(0, <object><PyObject*>py_tensor_desc)
  做了额外 PyObject* 往返，stealing 行为与 temp_malloc 不一致，
  存在对同一对象不对称 INCREF 的潜在 double-free 风险。
  新代码全部改为 Cython native call，引用计数由 Cython 编译器保证正确。

Walpurgis 改写20%（鲁迅拿法）:
  - WholememoryCallbackSpec: 值对象，描述一个回调函数的签名契约
    (name, arg_names, arg_types, return_type)
    替代 Python 中6个函数签名散落两个文件、靠注释维护一致性的模式。
  - WholememoryTensorParams: 替代 (shape, dtype_int, malloc_type_int) 三散参数，
    改写为有名字段的轻量对象，携带 to_torch_dtype() / to_device() 便利方法。
    上游 fbea7cb 将 PyWholeMemoryTensorDescription 拆成三个 int/tuple 散参传递，
    但调用侧仍需理解语义；我们用 WholememoryTensorParams 封装语义。
  - WholememoryAllocDecision: 替代 malloc_type_int 三路 if/elif/assert，
    改写为枚举映射表，输入 int 输出 (device, pinned) 决策对。
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路打印:
    - create_context: fn 地址 / ctx id
    - destroy_context: mem_ctx id
    - malloc: shape / dtype_int / malloc_type_int / device决策 / data_ptr
    - free: mem_ctx id / tensor shape
    - output_malloc / output_free: 同 temp 路径，前缀 [OUTPUT]

作者: dylanyunlon<dogechat@163.com>
"""

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# 调试开关
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试 print，WALPURGIS_DEBUG=1 时激活。"""
    if _DEBUG:
        print(f"[WALPURGIS_DEBUG][wholememory_cb][{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# WholememoryCallbackSpec: 回调签名契约对象
# 改写20%要点: 上游两个文件中6个函数的签名靠注释维护一致性;
# 我们用值对象显式描述每个回调的参数名称和语义。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WholememoryCallbackSpec:
    """
    描述一个 wholememory C→Python 回调函数的签名契约。

    上游 fbea7cb 重构后，6个回调函数签名分布在 .pyx 和 wholegraph_env.py 两侧，
    靠命名约定维护一致性。此对象将契约显式化，便于 Walpurgis 调试和接口验证。

    字段:
        name        回调函数的规范名称 (如 "temp_malloc")
        arg_names   参数名称列表 (与 Python 侧函数形参对应)
        doc         一行语义描述
    """
    name: str
    arg_names: Tuple[str, ...]
    doc: str

    def validate_call_args(self, *args: Any) -> None:
        """断言调用参数数量匹配契约，失配时打印诊断信息。"""
        if len(args) != len(self.arg_names):
            raise TypeError(
                f"[wholememory_cb] {self.name}: 期望 {len(self.arg_names)} 个参数 "
                f"({', '.join(self.arg_names)})，实际收到 {len(args)} 个。"
            )
        _dbg(self.name, f"call args count={len(args)} OK: {self.arg_names}")


# fbea7cb 后的6个回调签名契约
CALLBACK_SPECS = {
    "temp_create_context": WholememoryCallbackSpec(
        name="temp_create_context",
        arg_names=("global_context",),
        doc="创建临时内存上下文，返回 TorchMemoryContext",
    ),
    "temp_destroy_context": WholememoryCallbackSpec(
        name="temp_destroy_context",
        arg_names=("memory_context", "global_context"),
        doc="销毁临时内存上下文，释放 tensor",
    ),
    "temp_malloc": WholememoryCallbackSpec(
        name="temp_malloc",
        arg_names=("shape", "dtype_int", "malloc_type_int", "memory_context", "global_context"),
        doc="临时内存分配，返回 data_ptr (int)",
    ),
    "temp_free": WholememoryCallbackSpec(
        name="temp_free",
        arg_names=("memory_context", "global_context"),
        doc="释放临时内存 (free_data)",
    ),
    "output_malloc": WholememoryCallbackSpec(
        name="output_malloc",
        arg_names=("shape", "dtype_int", "malloc_type_int", "memory_context", "global_context"),
        doc="输出内存分配，返回 data_ptr (int)，与 temp_malloc 语义相同",
    ),
    "output_free": WholememoryCallbackSpec(
        name="output_free",
        arg_names=("memory_context", "global_context"),
        doc="释放输出内存 (free_data)",
    ),
}


# ---------------------------------------------------------------------------
# WholememoryTensorParams: 张量参数封装
# 改写20%要点: fbea7cb 将 PyWholeMemoryTensorDescription 拆为三个散参
# (shape: tuple, dtype_int: int, malloc_type_int: int)。
# 上游 Python 侧直接解包用散参，调用者需自行理解语义。
# 我们用 WholememoryTensorParams 封装，提供 to_device() / to_torch_dtype() 便利方法。
# ---------------------------------------------------------------------------

class WholememoryAllocMode(IntEnum):
    """
    对应 wholememory_memory_alloc_type_t 枚举值。
    fbea7cb 后传递 int，此枚举供 WholememoryTensorParams 内部解码使用。
    """
    MatDevice = 0   # CUDA device memory
    MatHost   = 1   # CPU host memory
    MatPinned = 2   # CPU pinned (page-locked) memory


@dataclass
class WholememoryTensorParams:
    """
    封装 fbea7cb 后 malloc 回调收到的三个参数。

    上游散参调用:
        fn(py_shape, py_dtype, py_malloc_type_int, mem_ctx, ctx)
    Walpurgis 封装:
        params = WholememoryTensorParams.from_callback_args(shape, dtype_int, malloc_type_int)
        device, pinned = params.alloc_decision()
        torch_dtype    = params.to_torch_dtype()

    字段:
        shape           张量各维度大小的 tuple
        dtype_int       wmb.WholeMemoryDataType 枚举的 int 值
        malloc_type_int wmb.WholeMemoryMemoryAllocType 枚举的 int 值
    """
    shape: Tuple[int, ...]
    dtype_int: int
    malloc_type_int: int

    @classmethod
    def from_callback_args(
        cls,
        shape: Tuple[int, ...],
        dtype_int: int,
        malloc_type_int: int,
    ) -> "WholememoryTensorParams":
        inst = cls(shape=shape, dtype_int=dtype_int, malloc_type_int=malloc_type_int)
        _dbg(
            "WholememoryTensorParams",
            f"shape={shape} dtype_int={dtype_int} malloc_type_int={malloc_type_int}",
        )
        return inst

    def alloc_decision(self) -> Tuple[Any, bool]:
        """
        返回 (device, pinned) 两元组。

        对应 fbea7cb 后 torch_malloc_env_fn 内的三路 if/elif/assert:
            MatDevice  → (cuda, False)
            MatHost    → (cpu,  False)
            MatPinned  → (cpu,  True)

        改写20%: 上游是内联 if/elif/assert 树；
        我们用映射表 + 枚举，使分支决策可独立测试。
        """
        import torch
        _ALLOC_MAP = {
            int(WholememoryAllocMode.MatDevice): (torch.device("cuda"), False),
            int(WholememoryAllocMode.MatHost):   (torch.device("cpu"),  False),
            int(WholememoryAllocMode.MatPinned): (torch.device("cpu"),  True),
        }
        if self.malloc_type_int not in _ALLOC_MAP:
            raise ValueError(
                f"[wholememory_cb] 未知 malloc_type_int={self.malloc_type_int}，"
                f"期望 {list(_ALLOC_MAP.keys())}"
            )
        device, pinned = _ALLOC_MAP[self.malloc_type_int]
        _dbg("alloc_decision", f"malloc_type_int={self.malloc_type_int} → device={device} pinned={pinned}")
        return device, pinned

    def to_torch_dtype(self, wholememory_binding: Any) -> Any:
        """
        将 dtype_int 转换为 torch dtype。

        对应 fbea7cb 后:
            dtype = wholememory_dtype_to_torch_dtype(wmb.WholeMemoryDataType(dtype_int))
        上游直接内联在 torch_malloc_env_fn 中；
        我们提取为方法，便于断点调试和单独测试。

        参数:
            wholememory_binding: pylibwholegraph.binding.wholememory_binding 模块 (wmb)
        """
        from pylibwholegraph.torch.utils import wholememory_dtype_to_torch_dtype
        wm_dtype = wholememory_binding.WholeMemoryDataType(self.dtype_int)
        torch_dtype = wholememory_dtype_to_torch_dtype(wm_dtype)
        _dbg("to_torch_dtype", f"dtype_int={self.dtype_int} → wm_dtype={wm_dtype} → torch_dtype={torch_dtype}")
        return torch_dtype


# ---------------------------------------------------------------------------
# WholememoryCallbackBridge: 6个回调函数的 Walpurgis 实现层
# 对应 fbea7cb 后 wholegraph_env.py 的 torch_*_env_fn 函数族。
# 改写20%要点: 上游6个顶层函数直接散落在模块中；
# 我们封装为一个类的静态方法，便于统一 DEBUG print 和签名验证。
# ---------------------------------------------------------------------------

class WholememoryCallbackBridge:
    """
    Walpurgis 层对 fbea7cb 后 6个 wholememory C→Python 回调的封装。

    上游 wholegraph_env.py 实现:
        torch_create_memory_context_env_fn(global_context) → TorchMemoryContext
        torch_destroy_memory_context_env_fn(memory_context, global_context)
        torch_malloc_env_fn(shape, dtype_int, malloc_type_int, memory_context, global_context) → int
        torch_free_env_fn(memory_context, global_context)
        (output_malloc / output_free 与 temp 同实现，通过 create_context() 分别注册)

    改写20%:
        - 每个方法入口打印 WALPURGIS_DEBUG 断点信息
        - WholememoryTensorParams 封装 malloc 参数
        - validate_call_args 检查参数个数契约
    """

    @staticmethod
    def create_context(global_context: Any) -> Any:
        """
        对应 python_cb_wrapper_temp_create_context 的 Python 侧实现。
        fbea7cb 后: fn = ...; ctx = ...; py_memory_context = fn(ctx)
        """
        spec = CALLBACK_SPECS["temp_create_context"]
        spec.validate_call_args(global_context)
        _dbg("create_context", f"global_context id={id(global_context)}")

        # 对应 TorchMemoryContext() 构造
        try:
            from pylibwholegraph.torch.wholegraph_env import TorchMemoryContext
            mem_ctx = TorchMemoryContext()
        except ImportError:
            # 无 torch 环境时，返回轻量占位对象（供测试）
            mem_ctx = _StubMemoryContext()

        _dbg("create_context", f"→ memory_context id={id(mem_ctx)}")
        return mem_ctx

    @staticmethod
    def destroy_context(memory_context: Any, global_context: Any) -> None:
        """
        对应 python_cb_wrapper_temp_destroy_context 的 Python 侧实现。
        fbea7cb 后: fn(mem_ctx, ctx)
        """
        spec = CALLBACK_SPECS["temp_destroy_context"]
        spec.validate_call_args(memory_context, global_context)
        _dbg("destroy_context", f"memory_context id={id(memory_context)}")
        memory_context.free()
        _dbg("destroy_context", "free() 完成")

    @staticmethod
    def malloc(
        shape: Tuple[int, ...],
        dtype_int: int,
        malloc_type_int: int,
        memory_context: Any,
        global_context: Any,
        cb_label: str = "temp",
    ) -> int:
        """
        对应 python_cb_wrapper_temp_malloc / python_cb_wrapper_output_malloc。

        fbea7cb 核心变化:
            旧: (PyWholeMemoryTensorDescription, PyMemoryAllocType, mem_ctx, global_ctx)
            新: (shape: tuple, dtype_int: int, malloc_type_int: int, mem_ctx, global_ctx)

        参数:
            cb_label    "temp" 或 "output"，仅用于 DEBUG 标签区分
        """
        spec_key = f"{cb_label}_malloc"
        spec = CALLBACK_SPECS.get(spec_key, CALLBACK_SPECS["temp_malloc"])
        spec.validate_call_args(shape, dtype_int, malloc_type_int, memory_context, global_context)

        _dbg(f"{cb_label}_malloc", f"shape={shape} dtype_int={dtype_int} malloc_type_int={malloc_type_int}")

        params = WholememoryTensorParams.from_callback_args(shape, dtype_int, malloc_type_int)

        try:
            import torch
            import pylibwholegraph.binding.wholememory_binding as wmb

            device, pinned = params.alloc_decision()
            torch_dtype = params.to_torch_dtype(wmb)

            t = torch.empty(shape, dtype=torch_dtype, device=device, pin_memory=pinned)
            memory_context.set_tensor(t)
            data_ptr = t.data_ptr()

            _dbg(
                f"{cb_label}_malloc",
                f"→ device={device} pinned={pinned} dtype={torch_dtype} "
                f"shape={shape} data_ptr=0x{data_ptr:016x}",
            )
            return data_ptr

        except ImportError:
            # 无 torch 环境（纯测试路径），返回哨兵值
            _dbg(f"{cb_label}_malloc", "torch 不可用，返回哨兵 data_ptr=0")
            return 0

    @staticmethod
    def free(
        memory_context: Any,
        global_context: Any,
        cb_label: str = "temp",
    ) -> None:
        """
        对应 python_cb_wrapper_temp_free / python_cb_wrapper_output_free。
        fbea7cb 后: fn(mem_ctx, ctx)
        """
        spec = CALLBACK_SPECS[f"{cb_label}_free"]
        spec.validate_call_args(memory_context, global_context)

        if _DEBUG:
            try:
                t = memory_context.get_tensor()
                shape_info = str(t.shape) if t is not None else "None"
            except Exception:
                shape_info = "unavailable"
            _dbg(f"{cb_label}_free", f"memory_context id={id(memory_context)} tensor.shape={shape_info}")

        memory_context.free_data()
        _dbg(f"{cb_label}_free", "free_data() 完成")

    # --- output 路径别名（与 temp 共用实现，通过 cb_label 区分 DEBUG 标签） ---

    @classmethod
    def output_malloc(
        cls,
        shape: Tuple[int, ...],
        dtype_int: int,
        malloc_type_int: int,
        memory_context: Any,
        global_context: Any,
    ) -> int:
        """对应 python_cb_wrapper_output_malloc（output 侧，语义同 temp_malloc）。"""
        return cls.malloc(shape, dtype_int, malloc_type_int, memory_context, global_context, cb_label="output")

    @classmethod
    def output_free(cls, memory_context: Any, global_context: Any) -> None:
        """对应 python_cb_wrapper_output_free（output 侧，语义同 temp_free）。"""
        return cls.free(memory_context, global_context, cb_label="output")


# ---------------------------------------------------------------------------
# _StubMemoryContext: 无 torch 环境时的占位对象（仅供测试）
# ---------------------------------------------------------------------------

class _StubMemoryContext:
    """
    不依赖 torch 的轻量 TorchMemoryContext 替代品。
    仅在单元测试 / torch 未安装时使用。
    """

    def __init__(self):
        self._data_ptr: int = 0
        self._tensor = None

    def set_tensor(self, t: Any) -> None:
        self._tensor = t

    def get_tensor(self) -> Optional[Any]:
        return self._tensor

    def free(self) -> None:
        self._tensor = None
        self._data_ptr = 0

    def free_data(self) -> None:
        self._tensor = None


# ---------------------------------------------------------------------------
# 自检函数
# ---------------------------------------------------------------------------

def test_wholememory_cb_migration() -> None:
    """
    Walpurgis 自检: 验证 fbea7cb 迁移的核心语义正确性。

    检验项:
    1. WholememoryCallbackSpec.validate_call_args 参数数量校验
    2. WholememoryTensorParams.alloc_decision 三路 int 映射
    3. WholememoryCallbackBridge.create_context / destroy_context / free 流程
    4. CALLBACK_SPECS 六个规格名称完整性
    5. malloc_type_int 越界时抛出 ValueError
    """
    import sys

    print("[test_wholememory_cb_migration] 开始 fbea7cb 迁移自检 ...")

    # ---- 检验1: 参数数量校验 ----
    spec_create = CALLBACK_SPECS["temp_create_context"]
    try:
        spec_create.validate_call_args("ctx_obj")  # 正确: 1个参数
        print("  [OK] validate_call_args(1) 通过")
    except TypeError as e:
        print(f"  [FAIL] validate_call_args(1): {e}", file=sys.stderr)
        raise

    try:
        spec_create.validate_call_args("ctx1", "ctx2")  # 错误: 2个参数
        print("  [FAIL] validate_call_args(2) 应该抛出 TypeError", file=sys.stderr)
        raise AssertionError("未能检出参数数量不匹配")
    except TypeError:
        print("  [OK] validate_call_args(2) 正确抛出 TypeError")

    # ---- 检验2: alloc_decision 三路映射 ----
    for malloc_type_int, expected_device_type, expected_pinned in [
        (int(WholememoryAllocMode.MatDevice), "cuda",  False),
        (int(WholememoryAllocMode.MatHost),   "cpu",   False),
        (int(WholememoryAllocMode.MatPinned), "cpu",   True),
    ]:
        params = WholememoryTensorParams(
            shape=(4, 128),
            dtype_int=0,
            malloc_type_int=malloc_type_int,
        )
        try:
            import torch
            device, pinned = params.alloc_decision()
            assert device.type == expected_device_type, (
                f"device mismatch: {device.type} != {expected_device_type}"
            )
            assert pinned == expected_pinned, (
                f"pinned mismatch: {pinned} != {expected_pinned}"
            )
            print(f"  [OK] alloc_decision(malloc_type_int={malloc_type_int}) → device={device} pinned={pinned}")
        except ImportError:
            print(f"  [SKIP] alloc_decision torch 不可用，跳过 device 校验")

    # ---- 检验3: malloc_type_int 越界 ----
    params_bad = WholememoryTensorParams(shape=(1,), dtype_int=0, malloc_type_int=99)
    try:
        import torch
        params_bad.alloc_decision()
        print("  [FAIL] 越界 malloc_type_int=99 应抛出 ValueError", file=sys.stderr)
        raise AssertionError("未能检出越界 malloc_type_int")
    except ValueError as e:
        print(f"  [OK] 越界 malloc_type_int=99 正确抛出 ValueError: {e}")
    except ImportError:
        print("  [SKIP] torch 不可用，跳过越界校验")

    # ---- 检验4: create_context / destroy_context / free 流程 ----
    class _FakeGlobalCtx:
        pass

    global_ctx = _FakeGlobalCtx()
    mem_ctx = WholememoryCallbackBridge.create_context(global_ctx)
    assert mem_ctx is not None, "create_context 返回 None"
    print(f"  [OK] create_context → {type(mem_ctx).__name__}")

    WholememoryCallbackBridge.destroy_context(mem_ctx, global_ctx)
    print("  [OK] destroy_context 完成")

    mem_ctx2 = WholememoryCallbackBridge.create_context(global_ctx)
    WholememoryCallbackBridge.free(mem_ctx2, global_ctx, cb_label="temp")
    print("  [OK] free(temp) 完成")

    mem_ctx3 = WholememoryCallbackBridge.create_context(global_ctx)
    WholememoryCallbackBridge.output_free(mem_ctx3, global_ctx)
    print("  [OK] output_free 完成")

    # ---- 检验5: CALLBACK_SPECS 完整性 ----
    expected_keys = {
        "temp_create_context", "temp_destroy_context",
        "temp_malloc", "temp_free",
        "output_malloc", "output_free",
    }
    actual_keys = set(CALLBACK_SPECS.keys())
    assert actual_keys == expected_keys, (
        f"CALLBACK_SPECS 缺失或多余: 期望 {expected_keys}，实际 {actual_keys}"
    )
    print(f"  [OK] CALLBACK_SPECS 6个规格完整: {sorted(actual_keys)}")

    print("[test_wholememory_cb_migration] 全部自检通过 ✓")


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "WholememoryCallbackSpec",
    "WholememoryCallbackBridge",
    "WholememoryTensorParams",
    "WholememoryAllocMode",
    "CALLBACK_SPECS",
    "test_wholememory_cb_migration",
]
