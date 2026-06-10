
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
