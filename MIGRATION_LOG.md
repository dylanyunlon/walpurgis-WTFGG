
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
