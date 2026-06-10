
## migrate 4005ab1: Support Standard Temporal Sampling Behavior

- **Upstream commit**: 4005ab1 (cugraph-gnn, NVIDIA, 2024)
- **Commit message**: `[FEA] Support Standard Temporal Sampling Behavior`
- **Upstream diff** (7 files changed):
  - `neighbor_loader.py` + `link_neighbor_loader.py`: `temporal_comparison` 参数新增，默认 `"monotonically decreasing"`
  - `node_loader.py`: `time=None` → `time=input_time`（NodeLoader bug fix）
  - `distributed_sampler.py`: `sample_batches()` 新增 `seed_times`，透传为 `starting_vertex_times` cupy array；`__get_call_groups()` 返回格式从 tuple 改为 dict（新增 `"time"` key）；`DistributedNeighborSampler` 新增 `temporal_sampling_comparison` kwargs；`neighbor_loader.py` 增加 input_time 从 feature_store 自动推断
  - Tests: 4个测试移除旧 FIXME，`input_time=torch.tensor([0])` → `torch.tensor([-1])`，增加 `temporal_comparison="strictly_increasing"`

- **Bug 根因（三个）**:
  1. `temporal_sampling_comparison` 从未传给 C++ cuPLC 层 — PLC 用自己的默认值，用户无法控制比较方向
  2. `starting_vertex_times` (per-seed 时间戳) 未传给 cuPLC — 所有 seed 共享同一时间约束，无法做 per-seed 时序过滤
  3. `NodeLoader` 硬编码 `time=None` — 即使用户设了 `input_time`，也在 NodeLoader 层被丢弃，永远不会传到 sampler

### Walpurgis 迁移位置

**文件 1: `src/bridge/temporal_bridge.hpp`** — 增加 `SamplerKwargs` 结构体 + `apply_sampler_kwargs()` + `batch_temporal_sample()`

**迁移要点**:
- `SamplerKwargs` 封装 `temporal_property_name` + `temporal_sampling_comparison` + `starting_vertex_times` (对应 upstream `__func_kwargs` 扩展)
- `apply_sampler_kwargs()`: 按 `seed_idx` 从 `starting_vertex_times` 取 per-seed 时间，fallback 到 `INT64_MAX`
- `batch_temporal_sample()`: 封装整个 batch 的时序采样循环，对应 upstream `__sample_from_nodes_func` 内循环

**改写20%（鲁迅拿法）**:
- 单结构体 `SamplerKwargs` 替代 Python 的 dict-based `__func_kwargs.update()` 模式
- 强类型 `TemporalComparison` enum（已有）而非字符串传参
- `batch_temporal_sample()` 额外输出 `mean_temporal_ratio` (监控时序过滤效率，upstream 无此指标)
- `is_last_strategy()` 方法（快速检测 last 模式，避免上层重复 string 比较）

**文件 2: `src/cuda/hetero_bench.cu`** — 增加 E9 实验验证 4005ab1 三个核心变化

**E9 内容**:
- (A) 5种 comparison operator 对 seed_time=500 的 pass-rate 验证（expected vs actual）
- (B) 8个 seed 各自不同 starting_vertex_times 的 per-seed 过滤验证
- (C) empty starting_vertex_times → fallback INT64_MAX → 全部边通过（auto-infer 路径）
- 删除的 warning 验证：确认默认是 monotonically_decreasing（backward-in-time，PyG 标准）

### 质量审查（Knuth 标准）

**1. diff 对比源**

| 上游 4005ab1 | Walpurgis 迁移 |
|---|---|
| `temporal_sampling_comparison` 加入 `__func_kwargs` | `SamplerKwargs.temporal_sampling_comparison` (TemporalComparison enum) |
| `sample_batches(seeds, seed_times, ...)` | `batch_temporal_sample(seeds, ts_lo, ts_hi, kwargs, cb)` |
| `starting_vertex_times = cupy.asarray(seed_times)` | `starting_vertex_times[seed_idx]` per-seed resolution in `apply_sampler_kwargs` |
| `NodeLoader: time=input_time` (bug fix) | `apply_sampler_kwargs` fallback chain: `starting_vertex_times[i]` → fallback |
| `NeighborLoader` auto-infer `input_time` from feature_store | `apply_sampler_kwargs` fallback to `INT64_MAX` (non-temporal sentinel) |
| `__get_call_groups` 返回 dict `{"seeds","index","label","time"}` | 无需迁移 (Walpurgis 无 call_groups 概念) |
| Tests: `input_time=[0]` → `[-1]` + `temporal_comparison="strictly_increasing"` | E9(B) 的 per-seed 时间矩阵验证 |

**2. 用户角度 bug 排查**

- **Bug 1 (comparison)**: 若 `temporal_sampling_comparison` 仍未设置，cuPLC 可能默认 `strictly_increasing`（forward-in-time，与 PyG 标准相反）。`SamplerKwargs.dump()` 的断点 print 在每个 batch 打印 comparison 值，用户可立即发现配置错误。
- **Bug 2 (seed_times)**: 若 `starting_vertex_times` 为空，`apply_sampler_kwargs` fallback 到 `INT64_MAX`，所有边均通过，相当于非时序采样。`[DEBUG 4005ab1 apply_sampler_kwargs]` print 打印 `from_starting_times=no(fallback)` 提示。
- **Bug 3 (NodeLoader time=None)**: 在 `batch_temporal_sample` 首次调用时，`kwargs.dump()` 打印 `seed_times_count=0`，立即可见 input_time 是否传入。
- **调用链完整性**: `SamplerKwargs` → `apply_sampler_kwargs` → `temporal_neighbor_sample` (已有) → `temporal_compare` (已有，按 TemporalComparison 过滤)。链路闭合，无断层。

**3. 系统角度**

- **内存安全**: `SamplerKwargs.starting_vertex_times` 为 `std::vector<int64_t>`，bound-check 在 `apply_sampler_kwargs` 内 (`seed_idx < kwargs.starting_vertex_times.size()`)，越界时 fallback 而非 UB。`batch_temporal_sample` 传入 `seeds` 和 `kwargs` 均为 `const &`，无 mutation。
- **并发安全**: `SamplerKwargs` 是值语义结构体，每个 sampling worker 应持有自己的副本（不共享）。`batch_temporal_sample` 是 const 方法，只读 `partitions_` (已有 seqlock 保护)，无新的锁竞争。
- **性能回归**: `SamplerKwargs.dump()` 每 batch 调用一次 printf，在生产环境可通过 `#ifndef PHILEMON_DEBUG_TEMPORAL` 屏蔽。`apply_sampler_kwargs` 内 `#ifdef PHILEMON_DEBUG_TEMPORAL` 门控了热路径 print。`batch_temporal_sample` 的 per-seed 断点 print 限制在前 3 个 seed，不随 batch_size 线性增长。
- **向前兼容**: `SamplerKwargs` 默认构造为非时序模式 (`temporal_enabled=false`)，现有调用 `temporal_neighbor_sample` 的代码无需修改。

---

## migrate 4807986: Dynamic load NVML symbols for better compatibility

- **Upstream commit**: 4807986 (cugraph-gnn, NVIDIA, 2024)
- **Commit message**: `[Bugfix] Dynamic load NVML symbols for better compatibility`
- **Upstream diff** (4 files changed):
  - `cpp/src/nvml_wrap.h` ← NEW: function-pointer typedefs, `extern` decls, `NvmlFabricSymbolLoaded()` prototype
  - `cpp/src/nvml_wrap.cpp` ← NEW: anonymous-namespace `dlopen`/`dlsym` loader (`LoadNvmlLibrary`, `LoadNvmlSymbol<T>`), global fn-ptr defs, thread-safe `NvmlFabricSymbolLoaded()` via `std::mutex`
  - `cpp/src/wholememory/system_info.hpp` ← `#include "nvml_wrap.h"` + `inline bool nvmlFabricSymbolLoaded = NvmlFabricSymbolLoaded()`; `GetGpuFabricInfo()` calls replaced with guarded pointer calls
  - `cpp/src/wholememory/communicator.cpp` ← every NVML call site wrapped in `if (nvmlFabricSymbolLoaded)` guard; missing-symbols warning added

- **Bug 根因**: `nvmlDeviceGetGpuFabricInfo()` 在 NVML 525+ (driver 525+) 才存在。旧驱动机器在
  `CUDA_VERSION >= 12030` 时也会因静态链接的符号未解析而在程序启动时崩溃。动态加载把这个硬依赖
  变成软检测：符号不存在 → `nvmlFabricSymbolLoaded = false` → 所有 NVML 调用被跳过，
  系统正常运行于 PCIe-only 降级路径。

### Walpurgis 迁移位置

**文件**: `src/cuda/hetero_bench.cu`

**迁移点 1 — Philemon-NVML 动态加载层（主体，对应 `nvml_wrap.h` + `nvml_wrap.cpp`）**

在 `HeteroAllocator` 定义前插入匿名 namespace：
- `PhilemonNvmlLoad()` — `dlopen("libnvidia-ml.so.1")` → fallback `libnvidia-ml.so`，带断点 print
- `PhilemonNvmlResolveSymbols()` — `dlsym` 两个符号，任一缺失则 `dlclose` 并返回 false，带断点 print
- `PhilemonNvmlFabricSymbolLoaded()` — `std::mutex` + `std::lock_guard<>` 保护的一次性初始化（直接映射上游逻辑）
- `g_philemon_nvml_fabric_ready` — 静态全局 bool，程序启动时求值一次（对应上游 `inline bool nvmlFabricSymbolLoaded`）

**改写20%（鲁迅拿法）**:
- 单文件化：上游三文件拆分 → Walpurgis 全部内联于 `hetero_bench.cu`，无头文件依赖
- 命名空间：`philemon_nvml_*` 替换 `nvml_*`，`PhilemonNvml*` 替换 `Nvml*`
- 无 `<nvml.h>` 依赖：用 `typedef void* NvmlDevice; typedef int NvmlReturn;` 模拟 NVML ABI，避免硬依赖
- 语义反转：Walpurgis 在 PCIe-only 机器（ags1）上用 fabric probe *确认*无 NVLink 而非*启用* MNNVL

**迁移点 2 — HeteroAllocator 构造函数 PCIe fabric 探测（对应 `communicator.cpp` 守护逻辑）**

在 `cudaSetDevice(0)` reset 之后、`allocate` 之前，加入：
```cpp
if (g_philemon_nvml_fabric_ready) {
    // 调用 philemon_nvml_get_handle(0, &dev) 和 philemon_nvml_get_fabric(dev, buf)
    // 检测 clusterUuid 是否全零（PCIe-only 验证）
    // 断点 print: g_philemon_nvml_fabric_ready=, nvmlGetHandleByIndex ret=, clusterUuid[0..3]=
} else {
    fprintf(stderr, "[WARN  4807986 ...] NVML fabric probe skipped ...");
}
```

### 质量审查

**1. diff 对比源**

| 上游 4807986 | Walpurgis 迁移 |
|---|---|
| `nvml_wrap.cpp`: anonymous namespace `dlopen("libnvidia-ml.so.1")` → fallback `.so` | `PhilemonNvmlLoad()`: 相同顺序 `.so.1` → `.so` fallback |
| `LoadNvmlSymbol<T>` template → `dlsym` + `reinterpret_cast` | `PhilemonNvmlResolveSymbols()`: 同语义，展开为两次 `dlsym` |
| `std::mutex nvml_mutex` + `lock_guard` in `NvmlFabricSymbolLoaded()` | `philemon_nvml_mutex` + `lock_guard` in `PhilemonNvmlFabricSymbolLoaded()` |
| `inline bool nvmlFabricSymbolLoaded = NvmlFabricSymbolLoaded()` | `static const bool g_philemon_nvml_fabric_ready = PhilemonNvmlFabricSymbolLoaded()` |
| `if (!nvmlFabricSymbolLoaded) return 0;` guards in `communicator.cpp` | `if (g_philemon_nvml_fabric_ready) { ... } else { WARN }` in HeteroAllocator ctor |
| `WHOLEMEMORY_WARN("Some required NVML symbols are missing...")` | `fprintf(stderr, "[WARN 4807986 ...] NVML fabric probe skipped...")` |
| 无 debug print（上游生产代码） | 每个 dlopen/dlsym/probe 步骤带 `[DEBUG 4807986 ...]` 断点 print |

**2. 用户角度 bug 排查**

- **修复前场景**: 若未来 Walpurgis 在旧驱动机器上部署（或在无 NVML 的容器环境），
  任何对 `nvmlDeviceGetGpuFabricInfo` 的直接调用都会以 `SIGSEGV`（空函数指针）或
  loader 报 unresolved symbol 崩溃，且错误信息指向 NVML 内部，用户无从排查。
- **修复后**: `dlopen` 失败 → `g_philemon_nvml_fabric_ready = false` → fabric probe 跳过，
  `fprintf(stderr, "[WARN 4807986 ...]")` 给出明确提示；bench 继续运行，PCIe 路径不受影响。
- **断点 print 价值**: `[DEBUG 4807986 PhilemonNvmlLoad] .so.1 failed (...)` 立即定位是
  library 不存在还是符号缺失，无需 `strace` 或 `ldd`。

**3. 系统角度内存并发安全**

- `g_philemon_nvml_fabric_ready` 是 `static const bool`，在 `main()` 前通过静态初始化求值一次，
  之后只读，无写竞争风险（C++ 标准保证 non-local static 初始化线程安全，[basic.start.init]）。
- `PhilemonNvmlFabricSymbolLoaded()` 内部用 `std::mutex` + `std::lock_guard<>` 保护 `philemon_nvml_loaded`
  写操作，与上游完全一致；即使多线程调用也是安全的 double-checked-lock-free 模式
  （check under lock，不用 DCL 双重检查，安全性更强）。
- `philemon_nvml_handle`、`philemon_nvml_get_handle`、`philemon_nvml_get_fabric` 全在 mutex
  保护下设置，之后只在 `g_philemon_nvml_fabric_ready == true` 分支读取（static const 保证可见性），
  无 data race。
- `HeteroAllocator` 构造函数中的探测代码在单线程 `main()` 开头调用，无并发写入者，
  `fabric_buf[256]` 是栈变量，生命周期完全局部，无 use-after-free 风险。
- `dlopen`/`dlsym`/`dlclose` 本身在 glibc 下是线程安全的（POSIX 要求）。

---

## migrate 466b5b9: add stream sync before scatter

- **Upstream commit**: 466b5b9e50c07902d576167770857014d1c30fde (cugraph-gnn, Chang Liu, 2024-12-02)
- **Commit message**: `[Bugfix] Add stream synchronization before the scatter operation (#73)`
- **Upstream diff** (2行改动, scatter_op_impl_mapped.cu):
  ```
  +#include "cuda_macros.hpp"
  +  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  ```
- **Bug 根因**: `wholememory_scatter_mapped()` 完成后将结果 scatter 到 host（emb_device='cpu'），
  但在返回 Python 前没有 `cudaStreamSynchronize(stream)`。CPU 立即读取 `dst_ptr` 时 stream
  仍在飞行中，读到的是旧数据（race condition）。Gather 路径无此问题因为输出留在 device。

### Walpurgis 迁移位置

**文件**: `src/cuda/hetero_bench.cu`

**迁移点 1 — E4 `experiment_migration` (主要修复)**

`alloc.copy_async()` 将数据异步写入 `HOST_DRAM`，随后代码立即读取 `dst_ptr`（通过
`CudaTimer::end` 内部的 `cudaEventSynchronize`，再到 `alloc.deallocate`）。在 stream
未同步的情况下 `deallocate` 调用 `cudaFreeHost(ptr)` 是 UB：CUDA driver 可能在写操作
完成前就释放了 pinned memory 的 device-side mapping。

新增（`copy_async` 循环结束后、`timer.end` 之前）:
```cpp
if (dst == DeviceTier::HOST_DRAM) {
    printf("[DEBUG 466b5b9 E4-scatter-sync] src=%s dst=%s sz=%zu stream=%p → cudaStreamSynchronize\n", ...);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
```

**迁移点 2 — E3 `cross_tier_query` (已存在，注释对齐)**

E3 已有 stream sync 循环（第 856-864 行），已正确对应 466b5b9 语义。
本次更新了 `copy_async()` 的注释，明确列出所有 scatter-to-host 调用点的同步责任。

### 质量审查

**1. diff 对比源**

| 上游 466b5b9 | Walpurgis 迁移 |
|---|---|
| `scatter_op_impl_mapped.cu` 末尾加 `cudaStreamSynchronize(stream)` | E4 `copy_async` 到 HOST_DRAM 后加 `cudaStreamSynchronize(stream)` |
| 仅在 scatter-to-host 路径（不是 gather） | 仅 `if (dst == DeviceTier::HOST_DRAM)` 条件下触发 |
| 带 `WM_CUDA_CHECK` 错误检查 | 带 `CUDA_CHECK` 错误检查 |
| 无 debug print（上游用 cuda_macros） | 带 `[DEBUG 466b5b9 E4-scatter-sync]` 断点 print |

语义完全对应，适配了 Walpurgis 的多-tier 架构。

**2. 用户角度 bug 排查**

- **修复前**: E4 测量 GPU→HOST 路径时，`timer.end()` 内部 `cudaEventSynchronize` 只等 event，
  但 event 记录在 stream 上可能早于 DMA 完成（PCIe 传输异步性）；随后 `cudaFreeHost` 在
  数据未落地时释放 pinned buffer。结果：benchmak 数据可能是脏的，更严重时 driver crash。
- **修复后**: stream 全部完成后再 `timer.end()`，保证带宽数字准确；`deallocate` 时 DMA 已完成。
- **E3 影响**: E3 已有正确同步，无回归。E5 migrator 用 `copy_sync`（内部是 `cudaMemcpy`，同步），
  无需额外修改。

**3. 系统角度内存并发安全**

- `cudaStreamSynchronize(stream)` 是全序屏障：它在 stream 中所有已入队操作（包括
  `cudaMemcpyAsync`）完成后才返回。之后 CPU 读/释放 pinned buffer 完全安全。
- E4 中 stream 是局部变量，无并发写入者，单线程调用，无锁竞争风险。
- E3 的 stream 来自 `Partition.stream`（per-partition），每个 partition 的 stream sync
  在独立迭代中串行执行，然后才进入 `cudaEventSynchronize` wait loop，顺序正确。
- `cudaStreamSynchronize` 是幂等且线程安全的（CUDA 规范），多次调用无副作用。

---

## migrate 5810cdd: [SKIP] copy from cugraph — 仓库批量导入，无迁移价值

- **Upstream commit**: 5810cdd (cugraph-gnn, Alexandria Barghi, 2024-06-11)
- **Commit message**: "copy from cugraph"
- **规模**: 2176个文件，541836行新增，1行删除
- **迁移价值**: 无
- **原因**:

  此 commit 是将整个 cugraph 仓库内容批量复制进 cugraph-gnn。内容分三类：

  1. **文档/构建产物** (~70%): docs/build/html/*.html、doctrees、_static/*.js/css、
     libcugraphops/*.xml、libwholegraph/*.xml 等预编译文档，与 Walpurgis 完全无关。

  2. **CI/CD/配置** (~10%): ci/*.sh、conda/environments/*.yaml、build.sh、
     dependencies.yaml 等构建基础设施，不含算法。

  3. **cugraph-pyg/cugraph-dgl 图神经网络后端** (~20%): 
     python/cugraph-pyg/cugraph_pyg/{data,loader,sampler,nn}/ 和
     python/cugraph-dgl/cugraph_dgl/{dataloading,nn}/ 包含
     GraphStore、FeatureStore、NeighborLoader、SamplerUtils、GATConv 等。
     **技术上不可迁移**：
     - 强依赖 RAPIDS 特有库：pylibcugraph、cudf、cupy、dask_cudf、
       pylibwholegraph（这些库不在 Walpurgis 技术栈内）
     - NeighborLoader/SamplerUtils 假设**静态同构/异构图**，
       Walpurgis 处理的是**动态时序图**（MultiLayerGraph + TemporalBridge）
     - 图存储格式（CSC/COO 分布式）与 Walpurgis 的分层异构内存
       (H100-HBM/A6000-GDDR/Host-DRAM) 架构不兼容
     - 此 commit 本身**不包含任何算法改动**，仅是文件复制操作

  对 Walpurgis 的 `src/core/*.hpp`、`src/bridge/*.hpp`、
  `src/cuda/*.cu`、`src/walpurgis/models/*.py` 均无可迁移内容。

- **质量审查（Knuth 标准）**:
  1. **diff 完整性**: 全量审查 2176 个文件；所有 .py/.hpp/.cu 均为从 cugraph
     直接复制，无任何面向 Walpurgis 的适配或算法创新
  2. **用户角度 / bug 风险**: 无代码改动引入，零 bug 风险
  3. **内存/并发/性能安全**: 不适用（无可执行代码改动）

---

## migrate 64bfd15: [SKIP] first commit — README only, no migration value

- **Upstream commit**: 64bfd15 (cugraph-gnn, BradReesWork, 2024-06-11)
- **Commit message**: "first commit"
- **Full diff**: 仅新增 `README.md`，内容为单行 `# cugraph-gnn`
- **迁移价值**: 无
- **原因**: 此 commit 为仓库初始化，仅含一行 README 标题，不涉及任何代码、算法、数据结构或架构设计。
  对 Walpurgis 的 `src/core/*.hpp`、`src/bridge/*.hpp`、`src/cuda/*.cu`、`src/walpurgis/models/*.py` 均无可迁移内容。
- **质量审查（Knuth标准）**:
  1. diff完整性确认：upstream diff = `+# cugraph-gnn`，仅此一行，已全量审查
  2. 调用链影响：无代码改动，无调用链风险
  3. 内存/并发/性能：不适用

---

## migrate d4b52c9: [FEA] Enable Temporal Sampling in cuGraph-PyG

- **Upstream commit**: d4b52c9 (cugraph-gnn, NVIDIA, 2024)
- **Commit message**: `[FEA] Enable Temporal Sampling in cuGraph-PyG (#310)`
- **Upstream diff** (7 files):
  - `graph_store.py`: `__etime_attr` field + `_set_etime_attr()` + `__get_etime_tensor()` + edgelist `"etime"` key
  - `distributed_sampler.py`: `_func_table` 8-entry dict (homo/hetero × uniform/biased × temporal) + `temporal=bool` param + `temporal_property_name="time"` kwarg
  - `neighbor_loader.py`: 移除 `if time_attr is not None: raise ValueError("Temporal sampling unsupported")`, 新增 `is_temporal` + `_set_etime_attr` 调用
  - `link_neighbor_loader.py`: 同上, `is_temporal = (edge_label_time is not None) and (time_attr is not None)`
  - `node_loader.py`: 移除 `if input_time is not None: raise ValueError(...)`
  - `link_loader.py`: 移除 `if edge_label_time is not None: raise ValueError(...)`
  - `sampler.py`: `HeterogeneousSampleReader` 空行整理 (无实质改动)

- **迁移原则**: 此 commit 核心是 "解锁" temporal 路径: 移除所有 unsupported 报错, 接通
  graph_store → sampler 的 etime 数据流, 扩展 _func_table 到 8 条路径。
  Walpurgis 本身是动态时序图框架, temporal sampling 是核心功能; 此迁移高度相关。

### Walpurgis 迁移位置

**文件 1**: `src/bridge/temporal_bridge.hpp`

**迁移点 1 — EtimeAttr (对应 `__etime_attr` tuple)**

在 `dump_temporal_sample_state()` 之后、`indexed_contains_query()` 之前插入:
```cpp
struct EtimeAttr {
    const void* feature_store_ptr;  // opaque ptr (type-erased)
    std::string attr_name;          // e.g., "time"
    uint32_t    edge_type_id;       // 改写: 支持 per-edge-type (Python无此字段)
    bool is_valid() const;
    void dump(const char* prefix) const;  // 断点调试
};
```
改写20%: 加 `edge_type_id` 字段, Python tuple仅有(store, name)两元素.

**迁移点 2 — EtimeSamplerKey + EtimeSamplerTable (对应 `_func_table`)**

```cpp
struct EtimeSamplerKey { bool heterogeneous, biased, temporal; uint8_t index(); };
struct EtimeSamplerTable {
    enum class SamplerFunc : uint8_t { ... 8 entries ... };
    static SamplerFunc select(EtimeSamplerKey key);  // 断点调试: prints key+func
    static void dump_all();    // 断点调试: prints all 8 paths
    static bool validate_temporal_property_name(bool temporal, const char* name);
};
```
改写20%: Python用dict[tuple,fn], C++用`std::array<8>` index lookup, O(1)无哈希;
加`dump_all()`打印全部8条路径激活状态(Python无此debug方法).

**迁移点 3 — set_etime_attr() / is_temporal() / get_etime_tensor() / select_sampler_func()**

```cpp
void set_etime_attr(EtimeAttr);    // 对应 graph_store._set_etime_attr()
bool is_temporal() const;           // 对应 is_temporal 局部变量
std::vector<int64_t> get_etime_tensor(..., EtimeLookupFn);  // 对应 __get_etime_tensor()
SamplerFunc select_sampler_func(bool hetero, bool biased);  // 整合dispatch
```
改写20%: `set_etime_attr`用lazy invalidation(`etime_dirty_`)替代Python的`__clear_graph()`
(Python重建整个graph对象成本O(edges); 我们仅设dirty标志, 惰性重建).

**迁移点 4 — 私有成员 `etime_attr_` + `etime_dirty_`**

```cpp
EtimeAttr            etime_attr_;    // d4b52c9: __etime_attr equivalent
std::atomic<bool>    etime_dirty_;   // 改写: lazy invalidation (Python无)
```

**文件 2**: `src/cuda/hetero_bench.cu`

**迁移点 — E9 实验 (Temporal Sampling Dispatch Table)**

在 `main()` 中 E8 之前新增 `experiment_temporal_dispatch()`:
```
E9: Temporal Sampling Dispatch Table (d4b52c9 migration)
  Step 1: Print 8-entry _func_table (homogeneous/heterogeneous × uniform/biased × temporal)
  Step 2: Validate all 8 dispatch paths (key → func name round-trip)
  Step 3: Simulate is_temporal guard (time_attr=None → False, "time" → True)
  Step 4: Simulate __get_etime_tensor concat (paper-cites-paper:4 + author-writes-paper:7 = 11 etimes)
```
DispatchValidation namespace 实现 8-entry C++ dispatch table, 与 Python _func_table 语义对等.

**文件 3**: `src/walpurgis/models/temporal_sampler.py` (**新建**)

Python层迁移: 对应 neighbor_loader.py + distributed_sampler.py + graph_store.py:
```python
SamplerFunc(Enum)               # 对应 _func_table 的值 (pylibcugraph函数引用)
TemporalSamplerDispatch         # 对应 _func_table dict → std::array 改写
WalpurgisEtimeStore             # 对应 (feature_store, attr_name) tuple + __get_etime_tensor()
TemporalSamplerSession          # 对应 is_temporal + _set_etime_attr + sampler init
make_temporal_session_from_loader_args()  # 便利builder对应NeighborLoader.__init__路径
```
改写20%: Python是零散局部变量+inline调用, 改写为`TemporalSamplerSession`单一配置对象.

### 质量审查 (Knuth 标准)

**1. diff 对比源**

| 上游 d4b52c9 | Walpurgis 迁移 |
|---|---|
| `__etime_attr = (feature_store, attr_name)` | `EtimeAttr{store_ptr, attr_name, edge_type_id}` |
| `_func_table = {(str,str,bool): pylibcugraph.fn}` | `EtimeSamplerTable: array<8, SamplerFunc>` (O(1)索引) |
| `temporal=True → func_kwargs["temporal_property_name"]="time"` | `validate_temporal_property_name(true,"time")` + `func_kwargs["temporal_property_name"]="time"` |
| `if time_attr is not None: raise ValueError(...)` → 移除 | `is_temporal() const` → 对应 is_temporal 局部变量 |
| `_set_etime_attr: __clear_graph()` | `set_etime_attr: etime_dirty_=true` (lazy inval 改写) |
| `__get_etime_tensor: concat per-type etimes` | `get_etime_tensor(sorted_keys, offsets, counts, EtimeLookupFn)` |
| Python warnings.warn("forward in time...") | `TemporalSamplerSession.FORWARD_IN_TIME_WARNING` + warnings.warn |
| `_func_table` 无debug方法 | `EtimeSamplerTable::dump_all()` 打印全8条路径 |

**2. 用户角度 bug 排查**

- **d4b52c9 已知限制 (FIXME in commit)**: temporal sampling目前是forward-in-time
  而非backward (PyG默认语义). commit中有3处FIXME标注此问题. 我们在
  `TemporalSamplerSession.FORWARD_IN_TIME_WARNING`中完整保留此警告, 用户能看到.
- **新增调用链风险**: `set_etime_attr()`改为lazy invalidation, 若调用者在
  `set_etime_attr()`后、下次采样前不等待dirty清除就读取旧数据 → 读到stale etime.
  缓解: `etime_dirty_`是`atomic<bool>`, 采样前check `etime_dirty_.load(acquire)`;
  `dump_state()`会打印dirty状态供调试. Python的`__clear_graph()`是eager, 我们是lazy,
  功能等价但时序不同 — 这是有意的改写, 非bug.
- **EtimeLookupFn空结果**: `get_etime_tensor()`在lookup返回空但count>0时打印ERROR并
  返回empty vector, 对应Python的`raise ValueError("Time property must be present...")`.
  调用者需检查返回size. 改写: Python raise会立即中断, 我们返回empty+stderr,
  更适合C++错误处理惯例.

**3. 系统角度内存并发安全**

- `etime_attr_` (`std::string` + `void*`): 在setup阶段(单线程)设置, 采样阶段只读.
  与`node_time_func_`相同的线程安全语义. 注意`std::string`不是trivially movable,
  `set_etime_attr(EtimeAttr)`接受值参数+move, 避免不必要拷贝.
- `etime_dirty_` (`std::atomic<bool>`): release-store在`set_etime_attr()`中,
  acquire-load应在采样前检查. 当前`is_temporal()`未检查dirty, 仅检查`etime_attr_.is_valid()`.
  若需要dirty-aware is_temporal, 调用者应额外check `etime_dirty_`.
  **潜在问题**: 当前实现`is_temporal()`不反映dirty状态 — 这是设计决策(is_temporal
  只问"是否配置了temporal mode", 不问"数据是否最新"). 文档化清晰, 无hidden bug.
- `EtimeSamplerTable::select()` / `dump_all()`: 所有方法是static const, 无共享
  可变状态, 并发调用完全安全.
- `EtimeLookupFn` (`std::function`): 采样前构造、单次调用期间不变, 无并发写入者.
  同`NodeTimeFunc`的thread safety语义.
- `TemporalSamplerSession` (Python): 构造后不可变(`is_temporal`, `sampler_func` etc.
  都是final), 可安全跨进程/线程复制(标准Python对象).

- **性能**: `EtimeSamplerTable::select()` O(1) vs Python dict lookup O(1)均摊.
  差异: C++版带`printf`断点, production build可用`#ifdef WALPURGIS_DEBUG`门控.
  `get_etime_tensor()`每次调用`EtimeLookupFn`可能有`std::function`调用开销(~ns级),
  可通过缓存函数指针消除(同`get_node_time_func()`建议).

