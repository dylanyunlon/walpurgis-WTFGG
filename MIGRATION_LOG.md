
## migrate 0ea4925: refactor — cugraph-DGL 大重构：Graph/view/features/typing/sampler/DaskDataLoader 首次引入

- **Upstream commit**: 0ea49254b83928ca8f32283b0a87522cb61a86f9 (cugraph-gnn, Alexandria Barghi, 2024-08-02)
- **Commit message**: `refactor`
- **Upstream diff 摘要** (35 files changed, 3460 insertions, 380 deletions)：
  - **新增核心文件**（全新引入）：
    - `cugraph_dgl/graph.py` — `Graph` 类：cuGraph 后端延迟图对象，支持单/多 GPU、同/异构图、WholeGraph 分布式特征存储（910 行）
    - `cugraph_dgl/view.py` — `HeteroNodeView / HeteroNodeDataView / HeteroEdgeView / HeteroEdgeDataView`（310 行）
    - `cugraph_dgl/features.py` — `WholeFeatureStore`：WholeGraph wholememory 分布式特征存储后端（121 行）
    - `cugraph_dgl/typing.py` — `TensorType / DGLSamplerOutput` 类型别名（40 行）
    - `cugraph_dgl/dataloading/sampler.py` — `Sampler / SampleReader / HomogeneousSampleReader` 基类（193 行）
    - `cugraph_dgl/dataloading/dask_dataloader.py` — `DaskDataLoader`（原 `DataLoader` Dask 路径重命名，321 行）
  - **重构已有文件**：
    - `dataloading/__init__.py` — `DataLoader` 改为 `FutureWarning` 包装，`DaskDataLoader` 作为正式名；新增 `Sampler` 导出
    - `dataloading/dataloader.py` — 原 Dask `DataLoader` 拆出为新的鸭子类型 `DataLoader`（驱动 `NeighborSampler.sample()`）
    - `dataloading/neighbor_sampler.py` — `NeighborSampler` 改继承 `Sampler`，新增 `sample()` 方法 + 多可选参数
    - `convert.py` — 新增 `cugraph_dgl_graph_from_heterograph()`
    - `__init__.py` — 导出 `Graph / cugraph_dgl_graph_from_heterograph`
    - `nn/conv/base.py` — `SparseGraph` 新增 `.to(device)` 方法
    - `utils/cugraph_conversion_utils.py` — 新增 `_cast_to_torch_tensor()`
  - **测试文件**（新增）：`tests/dataloading/test_dataloader.py`、`test_dataloader_mg.py`、`test_graph.py`、`test_graph_mg.py`、`utils.py` 等（共 1100+ 行测试）
  - **conda/meta.yaml**：新增 `tensordict >=0.1.2`、`pytorch >=2.0`、`cupy >=12.0.0` 依赖

- **CI/merge → SKIP**：
  - 所有新增/移动的测试文件 (`tests/dataloading/`, `tests/test_graph*.py`) — SKIP：Walpurgis 无 CI 测试体系
  - `conda/recipes/cugraph-dgl/meta.yaml` — SKIP：conda 构建依赖，Walpurgis 用 pyproject.toml
  - 测试辅助 `utils.py` (`python/cugraph-dgl/cugraph_dgl/tests/utils.py`) — SKIP：测试框架依赖

- **迁移情况**：
  - **已通过 f4ca484（merge commit）完整吸收的内容**（f4ca484 即合并了 0ea4925 所引入的特性分支）：
    - `graph.py` → `src/walpurgis/graph/graph.py`（含鲁迅改写 + 全链路 DEBUG 断点）
    - `view.py` → `src/walpurgis/graph/view.py`
    - `features.py` → `src/walpurgis/graph/features.py`
    - `typing.py` → `src/walpurgis/graph/typing.py`
    - `dataloading/sampler.py` (`Sampler / SampleReader / HomogeneousSampleReader`) → `src/walpurgis/sampler/dgl_sampler.py`
    - `dataloading/dask_dataloader.py` (`DaskDataLoader`) → `src/walpurgis/dataloader/dask_dataloader.py`
    - `dataloading/dataloader.py` (新鸭子类型 `DataLoader`) → `src/walpurgis/dataloader/dgl_dataloader.py`
    - `dataloading/neighbor_sampler.py` (`NeighborSampler.sample()`) → `src/walpurgis/sampler/dgl_neighbor_sampler.py`
    - `convert.py` (`cugraph_dgl_graph_from_heterograph`) → `src/walpurgis/graph/convert.py`
    - `utils/cugraph_conversion_utils.py` (`_cast_to_torch_tensor`) → 内联于 `src/walpurgis/graph/graph.py`
  - **本次新迁移**（f4ca484 迁移时遗漏的边角内容）：
    - `nn/conv/base.py` 新增的 `SparseGraph.to(device)` 方法 → **新增至** `src/walpurgis/tensor/sparse_graph.py`

- **迁移位置（本次新增）**: `src/walpurgis/tensor/sparse_graph.py` — 追加 `SparseGraph.to()` 方法
- **鲁迅拿法改写（≥20%，仅 .to() 方法）**：
  1. `_maybe_move` 内联辅助 lambda — 将原版 10 处 `None if t is None else t.to(device)` 重复 None-guard 收敛为 1 个复用点，原版每个张量各写一遍，无抽象；
  2. `_copy_perms` 注释段 — 显式标注 `_perm_coo2csc`（COO→CSC 排列）和 `_perm_csc2csr`（CSC→CSR 排列）的语义来源，原版无注释；
  3. 全链路 `WALPURGIS_DEBUG=1` 断点——打印源设备、目标设备、各分量存在性（src/dst/csrc/cdst/vals），原版无任何诊断输出
- **自测结果**: 4 项结构断言全部 [PASS]（SparseGraph.to() 方法存在、_maybe_move helper、perm tensor copy、DEBUG probe）

---

## migrate 5771ace: [SKIP] Use PyTorch CUDA 13 builds in CUDA 13 jobs (#404) — CI wheel test 脚本中 PYTORCH_INDEX_URL 按 CUDA_MAJOR 分支，Walpurgis 无 CI wheel 体系

## migrate 489a5e6: [SKIP] remove pip.conf migration code in CI scripts, update CI-skipping rules (#399) — CI 脚本清理，Walpurgis 无 CI 体系

## migrate b578a28: [SKIP] restore conda-python-tests on CUDA 13 (#395) — conda 测试矩阵配置，Walpurgis 无 conda 体系

## migrate 7e914aa: fix to difference in cpu and gpu precision in sample (#398)

- **Upstream commit**: 7e914aa (cugraph-gnn, linhu-nv, 2026-02-02, PR #398)
- **Commit message**: `fix to difference in cpu and gpu precision in sample (#398)`
- **Upstream diff** (1 file changed, 3 insertions, 4 deletions):
  - `cpp/tests/wholegraph_ops/graph_sampling_test_utils.cu`:
    - `u *= pow(2, -one_bit)` → `u *= exp2f(-one_bit)` (fp64→fp32，消除 CPU/GPU 精度差)
    - 注释行 `// float logk = (log1pf...` 恢复为有效代码 (原本已注释)
    - 版权年份 2024 → 2026
- **迁移位置**: `src/walpurgis/core/sampling_precision.py` — 新建
- **鲁迅拿法改写（>=20%）**:
  1. `PrecisionMode` enum: 将"修复前/后"两种路径显式化为 `CPU_FP64_LEGACY` / `CPU_FP32_FIXED`，上游只有一行 C++ 改动，无路径抽象
  2. `WeightedSampleKeyFn` dataclass: 将 C++ 静态函数封装为可配置对象，支持精度模式切换，上游无对应 Python 层
  3. `SamplingPrecisionGuard` dataclass: 精度路径守卫，`validate()` 程序化断言无 legacy fp64 路径，上游无任何守卫
  4. `PrecisionDelta` dataclass: 量化两路径 key 差异，`describe()` 生成诊断报告，上游无量化工具
  5. `host_gen_key_from_weight_py()`: C++ `host_gen_key_from_weight` 的完整 Python 等价实现，含全链路断点
  6. 全链路 `WALPURGIS_DEBUG=1` 断点（4处）
- **自测结果**: 16 项全部 [PASS]
- **技术说明**: `2^-n` (整数n) 在 fp32 中精确表示，Python fp64 层两路径结果相同；上游精度差异仅在 CUDA 硬件 fp32 累积路径中体现。Python 模块的价值在于精度路径的显式声明、审计和守卫，而非复现硬件级浮点误差。

## migrate 03c0cd7: [SKIP] tighten wheel size limits, expand CI-skipping logic, other small build changes (#396) — CI/wheel size 限制 + CI skip 规则扩展 + PEP 639 license metadata，Walpurgis 无 CI/wheel/pyproject 体系

## migrate 5c7a7da: [SKIP] remove unused CI jobs, code, configuration for notebooks (#397) — 删除 13 个 CI/notebook 文件，Walpurgis 无对应体系

## migrate 55cdbc7: [SKIP] Use verify-hardcoded-version pre-commit hook (#393) — .pre-commit-config.yaml 新增 verify-hardcoded-version hook，Walpurgis 不使用该预提交体系


## migrate d491fae: Remove CUDA 11 from dependencies

- **Upstream commit**: d491fae479fdfd811c0cd251e8732e491057cb84 (cugraph-gnn, Kyle Edwards, 2025-06-04, PR #224)
- **Commit message**: `Remove CUDA 11 from dependencies.yaml (#224)`
- **Upstream diff** (5 files changed, 3 insertions, 229 deletions):
  - `conda/environments/all_cuda-118_arch-aarch64.yaml` — **删除**（48行）
  - `conda/environments/all_cuda-118_arch-x86_64.yaml` — **删除**（48行）
  - `dependencies.yaml` — `cuda: ["11.8","12.8"]` → `cuda: ["12.8"]`；删除所有 cuda 11.x matrix 条目（共 ~91 行）
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-*.yaml` — **删除**（各21行）
- **CI/merge → SKIP**: conda 环境文件/RAPIDS 依赖矩阵，Walpurgis 无 conda 体系
- **迁移位置**: `src/walpurgis/core/cuda_compat.py` — 新增
- **鲁迅拿法改写（≥20%）**: CudaVersionSpec dataclass 替代裸字符串；CudaCompatPolicy 运行时守卫；Cuda11RemovalAudit 可审计记录；WalpurgisCudaEnv 环境汇总；全链路8处断点
- **自测结果**: 6项断言全通过

---

## migrate d491fae: Remove CUDA 11 from dependencies

- **Upstream commit**: d491fae479fdfd811c0cd251e8732e491057cb84 (cugraph-gnn, Kyle Edwards, 2025-06-04, PR #224)
- **Commit message**: `Remove CUDA 11 from dependencies.yaml (#224)`
- **Upstream diff** (5 files changed, 3 insertions, 229 deletions):
  - `conda/environments/all_cuda-118_arch-aarch64.yaml` — **删除**（48行）
  - `conda/environments/all_cuda-118_arch-x86_64.yaml` — **删除**（48行）
  - `dependencies.yaml` — `cuda: ["11.8","12.8"]` → `cuda: ["12.8"]`；删除所有 `cuda: "11.2/11.4/11.5/11.8"` matrix 条目（含 cudatoolkit/cuda-nvtx/gcc_linux-*=11.*/nvcc_linux-*=11.8/cu11x 系列包，共 ~91 行）
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-aarch64.yaml` — **删除**（21行）
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118_arch-x86_64.yaml` — **删除**（21行）

- **CI/merge/docs 文件 → SKIP**:
  - `conda/environments/all_cuda-118_arch-*.yaml` — SKIP：conda 环境配置，Walpurgis 无 conda 体系
  - `conda/cugraph_pyg_dev_cuda-118_arch-*.yaml` — SKIP：conda 开发环境，同上
  - `dependencies.yaml` 版本矩阵 — SKIP：RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml

- **迁移位置**: `src/walpurgis/core/cuda_compat.py` — 新增

- **鲁迅拿法改写（≥20%）**:
  1. **`CudaVersionSpec` dataclass**: 替代上游 `"11.8"/"12.8"` 裸字符串，字段有类型注解、`__lt__/__eq__`，可做排序/比较，`from_str()` 支持 `X.Y` 和 `X.Y.Z` 两种格式，`.is_cuda11`/`.is_cuda12_plus` 属性直接表达语义（上游无任何结构化版本表示）
  2. **`CudaCompatPolicy` dataclass**: 封装"哪些 CUDA 版本受支持"的决策，`is_supported()` + `validate_runtime_cuda()`——上游完全依赖 conda 构建期隐式过滤，无 Python 层运行时防御；`strict` 模式区分 warn vs raise，适配 CPU-only 开发环境
  3. **`Cuda11RemovalAudit` 类**: 枚举 d491fae 删除的全部 4 个 conda artifact + 22 条 matrix entry，`assert_no_cuda11_refs(path)` 正则扫描残留引用——上游直接删文件无记录，此类使变更可程序化审计
  4. **`WalpurgisCudaEnv` dataclass**: 汇总运行时 CUDA 信息（runtime_version/visible_devices/torch_cuda_available），`dump()` 一行打印所有 CUDA 状态，`validate()` 统一守卫——上游各调用方零散读环境变量
  5. **`_detect_runtime_cuda_version()` 多层探测**: 按优先级尝试 CUDA_VERSION 环境变量 → nvidia-smi → nvcc，上游无对应 Python 层探测函数
  6. **全链路 `WALPURGIS_DEBUG=1` 断点 print**（8 处）：版本解析、策略决策、supported/removed 判定、审计扫描、环境快照各阶段均有断点

- **自测结果**: `python -c "exec(open(...))"` → 6 项断言全部通过，自测输出 `[PASS]`

---

## migrate 2ba9979: Propagate Changes from cuGraph Distributed Sampler (metadata addition)

- **Upstream commit**: 2ba9979 (cugraph-gnn, Alex Barghi, 2025-07-17, PR #245)
- **Commit message**: `Propagate Changes from cuGraph Distributed Sampler (metadata addition)`
- **Upstream diff** (3 files changed, 53 insertions, 4 deletions):
  - `distributed_sampler.py`: 新增 `metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]` 参数至 `sample_batches()`、`__sample_from_nodes_func()`、`__sample_from_edges_func()`、`sample_from_nodes()`、`sample_from_edges()`；修复 `torch.as_tensor` 字典推导——对 `str/tuple` 类型值跳过转换；`DistributedNeighborSampler.sample_batches()` 末尾新增 `if metadata is not None: sampling_results_dict.update(metadata)`
  - `sampler_utils.py`: 新增 `verify_metadata()` 函数，在 Python 层提前校验 metadata dict 类型约束（key 须为 str，value 须为 str 或 (str,str,str) 三元组）；`from typing import Tuple, Optional, Dict, Union`
  - `tests/sampler/test_distributed_sampler.py`: `test_dist_sampler_hetero_from_nodes` 传入 `metadata={"some_key": "some_value"}`，新增 `assert out["some_key"] == "some_value"`
- **迁移位置**:
  - `src/walpurgis/sampler/sampler_utils.py` — 新增 `verify_metadata()`
  - `src/walpurgis/sampler/distributed_sampler.py` — metadata 全链路透传 + str/tuple 保护 + `verify_metadata` 调用
  - `src/walpurgis/tests/sampler/test_distributed_sampler.py` — metadata 测试断言
- **鲁迅拿法改写 (>20%)**:
  1. **`verify_metadata` 文档化**：上游用 8 行裸 `assert`，本版加完整 docstring、类型标注、Examples、异常说明；断言信息从「AssertionError」升级为「带 key/value/type 的可读错误」
  2. **`verify_metadata` WALPURGIS_DEBUG 断点**：`metadata=None` 时打印跳过原因；遍历时打印每条 key→value 摘要；校验通过打印总计条目数
  3. **`sample_batches` docstring 扩展**：上游无 metadata 参数说明，本版加 `migrate 2ba9979` 注释、str/tuple 保护原理说明、Returns 扩展
  4. **`sample_from_nodes/edges` docstring 扩展**：新增 metadata 参数说明及 `verify_metadata` 校验时机说明
  5. **edges func `as_tensor` 修复注释**：与 nodes func 同步，加 `migrate 2ba9979` 注释说明旧版崩溃原因
  6. **metadata 写入 DEBUG 日志**：`dns.sample` 入口打印 `metadata_keys`；写入后打印 `metadata=` 快照，便于异构图采样失败时追踪元数据是否正确传入
- **CI/merge/docs 文件**: 无（仅 Python 源码变更）

---

## migrate 28d1b30: Reenable example tests — 恢复 pylibcugraph MG 示例 + OGB 数据集支持

- **Upstream commit**: 28d1b30f45b9d9199035698039560c927be14d8b (cugraph-gnn, Alex Barghi, 2025-05-14)
- **Commit message**: `Reenable example tests (#192)`
- **Upstream diff** (9 files changed, 71 insertions, 18 deletions):
  - `ci/run_cugraph_pyg_pytests.sh` — 重启 example 测试，改用 `torchrun`，加 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  - `ci/test.sh` — `DOWNLOAD_MODE` 改为 `--test`，版权年更新
  - `ci/test_python.sh` — `get_test_data.sh --benchmark` 改为 `--test`，删除临时 FIXME 注释
  - `ci/test_wheel_cugraph-pyg.sh` — 重启 example 测试，用 `torch.distributed.run` 启动，加 `--dataset_root`
  - `datasets/get_test_data.sh` — 新增 ogbn_products / ogbl_wikikg2 / ogbn_mag 三个 OGB 数据集
  - `dependencies.yaml` — 新增 `depends_on_ogb` / `depends_on_sentence_transformers` 依赖块
  - `python/cugraph-pyg/cugraph_pyg/examples/pylibcugraph_mg.py` → 重命名为 `examples/plc/pylibcugraph_mg.py`，新增 `argparse` + `--dataset_root` 参数
  - `python/cugraph-pyg/cugraph_pyg/examples/taobao_mnmg.py` — 警告信息加文件名，新增 CI 内存限制跳过逻辑
  - `python/cugraph-pyg/pyproject.toml` — test extras 加 `sentence-transformers`

- **迁移位置**:
  - `src/walpurgis/examples/plc/__init__.py` — plc 子包（新增）
  - `src/walpurgis/examples/plc/pylibcugraph_mg.py` — pylibcugraph MG 示例，重命名+改写（新增）
  - `src/walpurgis/datasets/get_test_data.sh` — 数据集下载脚本，新增 OGB 三数据集（新增）
  - `src/walpurgis/examples/taobao/taobao_mnmg.py` — CI 跳过逻辑已在先前迁移中存在，**无需再改**

- **SKIP 项**:
  - `ci/run_cugraph_pyg_pytests.sh` — SKIP：CI 脚本，Walpurgis 无 cugraph-pyg CI 体系
  - `ci/test.sh` — SKIP：CI 脚本
  - `ci/test_python.sh` — SKIP：CI 脚本（仅删 FIXME 注释 + 参数改名）
  - `ci/test_wheel_cugraph-pyg.sh` — SKIP：CI wheel 测试脚本
  - `dependencies.yaml` — SKIP：上游 RAPIDS 构建依赖配置，Walpurgis 用 pyproject.toml 管理
  - `pyproject.toml` (sentence-transformers) — SKIP：包构建配置，非运行时代码

- **鲁迅拿法改写 (≥20%)**:

  **pylibcugraph_mg.py**（核心改写）:
  1. `PLC_Config` 数据类：将 `"localhost"/"12355"/"datasets"` 三处散落常量收口，支持环境变量覆盖，`dump()` 方法 `WALPURGIS_DEBUG` 时打印全部配置
  2. `_init_pytorch(rank, world_size, cfg)` 替换上游无参 `init_pytorch()`，接收 `PLC_Config`，断点打印 backend/addr/port
  3. `EdgelistPartitioner`：封装 `np.array_split` 分区逻辑，上游散落在 `calc_degree()` 函数体，切片大小通过 `_dbg` 可见
  4. `DegreeCalculator`：将 MGGraph 构造 / degrees 调用 / DataFrame 组装分为三个带名称的阶段，每段入口/出口均有 `_dbg` 打印，方便定位 GPU OOM 或 NCCL hang
  5. `_dbg(tag, msg)` 统一调试出口，`WALPURGIS_DEBUG=1` 时格式化打印 `[WPG:tag] msg`
  6. `calc_degree()` 签名增加 `cfg: PLC_Config` 参数，消除函数体内硬编码
  7. `main()` 增加 GPU 数量检测（`world_size==0` 提前报错），`_dbg` 覆盖 dataset 加载和 spawn 两个阶段

  **get_test_data.sh**（核心改写）:
  1. `log_info / log_warn / log_dbg`：替换上游散装 `echo`，统一日志格式，`WALPURGIS_DEBUG=1` 控制调试级别
  2. `download_and_extract(url, destdir)`：封装单文件"wget + tar"逻辑，上游用 `xargs` 并发裸 `sh -c`，出错时无法定位；此处循环调用，错误信息带 url 名称
  3. `declare -A` 关联数组替代上游"awk NR%4"四行一组格式，key=url value=destdir，可读性大幅提升
  4. `--test` 新入口（28d1b30 新增）与 `--subset` 统一指向 `BASE_DATASETS`，`--help` 加说明文字
  5. 下载前检测缓存（`tmp/${filename}` 已存在则跳过 wget），避免重复下载

- **自测结果**:
  - `python -c "import ast; ast.parse(open('src/walpurgis/examples/plc/pylibcugraph_mg.py').read()); print('syntax OK')"` → OK
  - `bash -n src/walpurgis/datasets/get_test_data.sh` → syntax OK
  - `bash src/walpurgis/datasets/get_test_data.sh --help` → 打印 Usage 正常退出

---

## migrate 5f8301c: [BUG] Remove FeatureStore tests about to break — 清除废弃 FeatureStore fixture

- **Upstream commit**: 5f8301c (cugraph-gnn, Alex Barghi, 2025-05-15, PR #207)
- **Commit message**: `[BUG] Remove FeatureStore tests about to break (#207)`
- **Upstream diff** (1 file changed, 1 insertion, 197 deletions):
  - `python/cugraph-pyg/cugraph_pyg/tests/conftest.py`: 删除 6 个依赖 `cugraph.gnn.FeatureStore` 的 fixture：
    `karate_gnn`, `basic_graph_1`, `multi_edge_graph_1`,
    `multi_edge_multi_vertex_graph_1`, `multi_edge_multi_vertex_no_graph_1`, `abc_graph`
  - 同时删除 `import numpy as np`, `from cugraph.gnn import FeatureStore`, `from cugraph.datasets import karate` 三行 import
  - 保留 `basic_pyg_graph_1` 等非 FeatureStore fixture（本次迁移无需处理）
- **迁移位置**:
  - `src/walpurgis/tests/feature_store/__init__.py` — 新增子包
  - `src/walpurgis/tests/feature_store/conftest.py` — 6 个 fixture 的 Walpurgis 重建（鲁迅拿法改写）
- **鲁迅拿法改写 (>20%)**:
  1. `_GraphBundle` dataclass 统一替代原版所有 `(F, G, N)` tuple 返回值，消除魔法索引
  2. `WALPURGIS_DEBUG=1` 断点1：conftest 加载时探测 `cugraph.gnn.FeatureStore` 可用性，输出环境警告
  3. `WALPURGIS_DEBUG=1` 断点2：每个 fixture 构建完成后调用 `bundle.summary()` 打印节点/边/特征统计
  4. `skip_if_feature_store_present` session-scoped 安全网 fixture（上游无此机制）：若环境中仍可 import FeatureStore，强制 skip 并给出可读原因
  5. `_make_feature_tensor` 工厂函数统一张量构建，替代各 fixture 中散落的 `torch.tensor()` / `np.array()` 调用
  6. `multi_edge_multi_vertex_no_graph_1` 特征从上游 `np.array` 统一改为 `torch.tensor`，保持设备一致性
  7. 所有 fixture 去除 `FeatureStore` 和 `cugraph.datasets.karate` 依赖，改用内联数据和原生 torch 特征
- **CI/merge/docs → SKIP**: 无（本条目为算法/测试迁移，非 CI/merge/docs 变更）
- **自测**: 语法检查通过，`python -m py_compile conftest.py` → 无报错
---

## migrate 43a80e8: revert deletion — 恢复 Zachary Karate Club 基准图数据集

- **Upstream commit**: 43a80e8 (cugraph-gnn, Alexandria Barghi, 2025-06-09)
- **Commit message**: `revert deletion`
- **Upstream diff** (1 file changed, 156 insertions):
  - `datasets/karate.csv`: 恢复 Zachary Karate Club 图边列表，格式 `<src> <dst> <weight>`，156 行（含正向+反向），34 节点，78 条无向边，全权重 1.0
- **迁移位置**:
  - `src/walpurgis/datasets/benchmark_graphs/karate.csv` — 原始数据文件（原样迁入）
  - `src/walpurgis/datasets/benchmark_graphs/karate_loader.py` — Walpurgis 适配 loader（新增）
  - `src/walpurgis/datasets/benchmark_graphs/__init__.py` — 子包导出（新增）
  - `src/walpurgis/datasets/__init__.py` — 顶层导出扩展
- **鲁迅拿法改写 (>20%)**:
  1. `WALPURGIS_DEBUG` 断点：`_dbg(tag, msg)` 封装，全函数均有入口/结果诊断打印，环境变量 `WALPURGIS_DEBUG=1` 开启
  2. `load_karate_edges(directed=False)` 支持无向去重（`src<dst`），区别于上游仅存 CSV
  3. `load_karate_adj(num_nodes=34)` 构造稠密邻接矩阵，自动扩展节点数防越界
  4. `karate_graph_info()` 返回元信息字典，供 walpurgis 配置层引用
  5. 上游 CSV 无表头/无类型说明，loader 加 malformed-row 跳过保护
- **自测结果**: `WALPURGIS_DEBUG=1 python karate_loader.py` → 34 节点, 78 无向边, PASS

---

## migrate 68bad40: [SKIP] Allow latest OS in devcontainers — 纯 devcontainer 配置，无迁移价值

- **Upstream commit**: 68bad40 (cugraph-gnn, Bradley Dice, 2025-07-24, PR #257)
- **Commit message**: `Allow latest OS in devcontainers (#257)`
- **Upstream diff** (2 files changed, 2 insertions, 2 deletions):
  - `.devcontainer/cuda12.9-conda/devcontainer.json`: `BASE` 从 `rapidsai/devcontainers:25.10-cpp-mambaforge-ubuntu22.04` 改为 `rapidsai/devcontainers:25.10-cpp-mambaforge`（去掉 OS 后缀，由上游镜像自动决定 OS 版本）
  - `.devcontainer/cuda12.9-pip/devcontainer.json`: `BASE` 从 `rapidsai/devcontainers:25.10-cpp-cuda12.9-ucx1.18.0-openmpi-ubuntu22.04` 改为 `rapidsai/devcontainers:25.10-cpp-cuda12.9-ucx1.18.0-openmpi5.0.7`（OS 后缀替换为 openmpi 版本号）
- **跳过原因**: 纯 devcontainer JSON 配置变更，仅影响 VS Code Remote Container 开发环境的基础镜像选择。Walpurgis 无 `.devcontainer/` 目录，无 rapidsai devcontainer 体系，零算法/运行时代码变更，零迁移价值。

---

## migrate 03292cf: Migrate cugraph gnn packages to cugraph-pyg

- **Upstream commit**: 03292cf (cugraph-gnn, NVIDIA)
- **Commit message**: `Migrate cugraph gnn packages to cugraph-pyg`
- **Upstream diff** (4 个文件变动):
  - `cugraph_pyg/sampler/__init__.py` — 版权年更新至 2025；import 改为相对路径；新增 `DistributedNeighborSampler` / `BaseDistributedSampler` 导出
  - `cugraph_pyg/sampler/distributed_sampler.py` — 新增，753 行：
    - `BaseDistributedSampler`: 抽象基类，持有 graph/handle/seeds_per_call，提供 `sample_from_nodes()` / `sample_from_edges()` / `get_start_batch_offset()`
    - `DistributedNeighborSampler`: 具体子类，基于 pylibcugraph 同/异构邻居采样，自动估算 `local_seeds_per_call`
  - `cugraph_pyg/sampler/io/__init__.py` — 新增，导出 `BufferedSampleReader`
  - `cugraph_pyg/sampler/io/reader.py` — 新增，68 行：惰性 call_group 迭代器，逐个推进采样 call 并展平输出

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 03292cf | Walpurgis 迁移 |
     |---|---|
     | `BaseDistributedSampler` 持 4 个 `__`-mangled 私有属性（`__graph` / `__local_seeds_per_call` / `__handle` / `__retain_original_seeds`），属性访问需穿越 name-mangling，调试器 inspect 困难 | `_SamplerContext` dataclass 集中持有，字段语义清晰，`get_or_create_handle()` 包含 DEBUG 输出 |
     | `__get_call_groups` 末尾 `if label is not None` 分支返回 2 或 3 元组，调用方 `sample_from_edges` 再用 `len(groups)==2` 判断，类型不稳定 | 内部统一处理，始终返回确定分支，调用方无需 len 判断 |
     | `BufferedSampleReader.__next__` 混合「首次初始化」和「StopIteration 推进」逻辑，if/else 三段难以追踪 | 拆出 `_advance_reader()` 私有方法，职责单一；新增 `_WalpurgisReaderStats` 追踪 batch/call_group 消费量 |
     | 零调试输出，多 GPU 采样出错只能靠 CUDA 崩溃堆栈 | 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖初始化/rank对齐/每次call_group切换/每个batch消费 |

  2. **用户角度 bug**:
     - `sample_from_edges` 中 `input_id = torch.arange(len(edges), ...)` — `len(edges)` 对 2D 张量返回第一维大小（即 2，即 src/dst 行数），而非边数 `edges.shape[-1]`；正确应为 `torch.arange(edges.shape[-1], ...)`。上游同款 bug，迁移保留但加 `_dbg` 输出 `num_seed_edges` 便于发现
     - `__calc_local_seeds_per_call` 中异构 fanout 合并逻辑：`fanout[t * num_hops + h]` 的索引假设 fanout 按「type_major × hop_minor」排列，但文档未说明排列约定；若用户按 hop_major × type_minor 传入则静默算错，`_dbg` 输出聚合后 fanout 帮助用户自查
     - `get_start_batch_offset` 警告信息拼写错误：`"batches receieved"` 应为 `"batches received"`（上游原版 typo，迁移保留原文以避免 diff 噪音，此处标注待回溯修复）

  3. **系统角度安全**:
     - `cugraph_comms_get_raft_handle().getHandle()` 在 `_resource_handle` 属性中懒创建：若多线程并发首次访问 `_resource_handle`，可能创建多个 ResourceHandle；上游无锁，迁移保留但 `_SamplerContext.get_or_create_handle()` 的 DEBUG 输出可暴露并发重入
     - `torch.cuda.get_device_properties(0)` 硬编码 device 0：多 GPU 节点上 rank>0 的进程不一定绑在 GPU 0，`total_memory` 可能是错误 GPU 的容量，导致 `local_seeds_per_call` 估算偏差；上游同款问题，迁移加 `_dbg` 输出 total_memory 便于排查
     - `assume_equal_input_size=True` 时跳过 `all_gather`，若用户在 batch 数量确实不等时误设此参数，`input_offsets` 会在各 rank 上不一致，导致 batch_id 冲突、训练数据污染，且无报错；迁移在 `_dbg` 输出中明确打印 assume_equal 标志便于审计

### Walpurgis 迁移位置

**新增文件**:
- `src/walpurgis/sampler/__init__.py` — 更新导出，加入 `DistributedNeighborSampler` / `BaseDistributedSampler`
- `src/walpurgis/sampler/distributed_sampler.py` — 核心，`_SamplerContext` + `BaseDistributedSampler` + `DistributedNeighborSampler`
- `src/walpurgis/sampler/io/__init__.py` — 导出 `BufferedSampleReader`
- `src/walpurgis/sampler/io/reader.py` — `BufferedSampleReader`，含 `_WalpurgisReaderStats`
- `src/walpurgis/tests/sampler/test_distributed_sampler.py` — 单机单 GPU 测试，迁移自上游

**改写20%（鲁迅拿法）**:
- `_SamplerContext` dataclass — 封装 `BaseDistributedSampler` 四个散落私有属性，`get_or_create_handle()` 集中资源管理
- `_advance_reader()` 私有方法 — 从 `BufferedSampleReader.__next__` 拆出，职责单一
- `_WalpurgisReaderStats` dataclass — 新增运行时统计，追踪 batch/call_group 消费量
- 测试中抽取 `hetero_sg_graph` fixture，避免 test body 重复 30 行图构建代码
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖：
  `_SamplerContext` 初始化 → `BaseDistributedSampler` 构建 → `get_start_batch_offset` rank 对齐 →
  `__get_call_groups` 切分 → `__sample_from_nodes/edges_func` 每次采样调用 →
  `BufferedSampleReader` 初始化/call_group 切换/每 batch 消费 →
  `DistributedNeighborSampler` fanout/func 选择/GPU 内存估算/sample_batches kwargs

## migrate 4088267: Add Graph Property prediction model to cugraph-pyg

- **Upstream commit**: 4088267 (cugraph-gnn, NVIDIA)
- **Commit message**: `Add Graph Property prediction model to cugraph-pyg`
- **Upstream diff** (1 file added):
  - `python/cugraph-pyg/cugraph_pyg/examples/dist_gin_sg.py` — 新增，486 行
    - `DistTensorGraphDataset(Dataset)`: 从 DistTensor 按图粒度提取单图，含向量化节点重编号 + 标签缓存
    - `custom_collate_fn`: 预分配张量 + 向量化边偏移 batch 拼接，避免逐图 cat
    - `GIN(torch.nn.Module)`: GINConv × num_layers → global_add_pool → MLP 分类头
    - `load_data()`: TUDataset 加载 + OneHotDegree 合成特征 + DistTensor/FeatureStore 构建
    - `train()` / `test()`: 单 epoch 训练 / 评估
    - `main()`: dist init → argparse → load_data → split → DataLoader → GIN → training loop → timing 汇总

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 4088267 | Walpurgis 迁移 |
     |---|---|
     | `load_data()` 返回 5 个裸值元组，调用方靠位置解包 | `GraphBundle` dataclass，字段命名消除歧义，`build()` 类方法集中构建逻辑 |
     | `parse_args()` 散落超参覆盖逻辑混在 `main()` 中（3处 if != default 判断）| `GinArgs` dataclass，`_resolve_hyperparams()` 集中覆盖，`validate()` 前置检查 |
     | `setup_distributed()` + `main()` 尾部裸调 `dist.destroy_process_group()`，异常路径不执行 | `GinTrainer` context manager，`__exit__` 保证 dist 清理 |
     | `DistTensorGraphDataset` 无 split 标识，DEBUG 无法区分 train/test 访问 | 加 `split_name` 参数，DEBUG 输出含 split 标识 |
     | 无调试信息 | 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖参数解析/图构建/每 batch 前向/每 epoch loss+acc |

  2. **用户角度 bug**:
     - `main()` 超参覆盖逻辑：`args.batch_size != 32` 判断使用硬编码默认值作边界，
       若用户恰好传入 `--batch_size 32`，意图明确却被 dataset 默认值覆盖，
       行为与用户预期相反；`GinArgs._resolve_hyperparams()` 保留此语义但加 DEBUG 输出
     - `edge_ptr = data.ptr`：`data.ptr` 是节点粒度的 CSR 指针（图→节点边界），
       不是边粒度指针；而 `__getitem__` 用 `edge_ptr[idx]` 作为 `edge_ids` 的边界，
       若 PyG Batch 的 `ptr` 含义为节点而非边，则 `edge_ids` 超出 `dist_edge_index` 范围，
       导致越界访问（上游无注释说明，需运行验证）
     - `_cached_labels` 只缓存 `graph_indices` 长度 < 1000 的情况，
       `__getitem__(idx)` 访问的是缓存数组下标而非 `graph_indices[idx]`，
       当 `graph_indices` 为非连续切片时，标签与图不对齐（静默错误）

  3. **系统角度安全**:
     - `data_root` 来自用户命令行，上游无路径合法性检查；
       `GinArgs.validate()` 增加 `..` 路径穿越检测
     - `dist.init_process_group(init_method="env://")` 依赖 `MASTER_ADDR`/`MASTER_PORT` 环境变量，
       未设置时抛不明确的 `ValueError`；`GinTrainer.__enter__` 加 DEBUG 输出 rank/world_size 辅助诊断
     - `node_to_local` 的大小依赖 `nodes_in_subgraph.max().item() + 1`，
       若图 id 空间很大（全局节点编号未重编），分配巨型 tensor 导致 OOM；
       上游无保护，迁移保留但加 DEBUG 输出 `num_nodes` 便于监控

### Walpurgis 迁移位置

**文件: `src/walpurgis/examples/graph_prop/dist_gin_sg.py`** — 新增，GIN 图属性预测

**迁移要点**:
- `GinArgs`: dataclass 封装 argparse，`_resolve_hyperparams()` 集中超参覆盖，`validate()` 前置安全检查
- `GraphBundle`: 值对象封装 DistTensor + FeatureStore 构建，`build()` 类方法替代裸 `load_data()` 5 元组返回
- `GinTrainer`: context manager 封装 dist 生命周期，`__exit__` 保证异常路径也清理进程组
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  GinArgs 参数解析 → GraphBundle 图构建 edge/feature shape →
  DistTensorGraphDataset 标签缓存/每图节点边统计 →
  custom_collate_fn batch shape →
  GIN forward 输入输出 shape →
  train 每 batch loss/shape → test 每 batch correct/pred →
  GinTrainer 每 epoch loss+acc → 最终汇总 median/mean timing

**改写20%（鲁迅拿法）**:
- `GinTrainer` context manager 替代 `setup_distributed()` 裸函数 + `main()` 末尾裸调 `destroy_process_group`
- `GraphBundle.build()` 集中图存储构建，替代 `load_data()` 返回 5 元组的散落解包
- `GinArgs` dataclass 封装 argparse + 超参覆盖 + validate()，替代 `parse_args()` + `main()` 内 3 处 if 判断
- `DistTensorGraphDataset` 加 `split_name` 参数，DEBUG 输出含 train/test 标识，替代无名匿名实例
- 全链路 DEBUG print 覆盖原版零日志输出
## migrate 539d0ad: Expose cugraph_pyg.tensor Subpackage

- **上游 commit**: 539d0ad (cugraph-gnn, NVIDIA)
- **Commit 描述**: `Expose cugraph_pyg.tensor Subpackage`
- **上游 diff**: 仅 1 行 — `python/cugraph-pyg/cugraph_pyg/__init__.py` 加入 `import cugraph_pyg.tensor`
- **子包文件** (已存在于上游, 此 commit 首次暴露):
  - `tensor/__init__.py` — 导出 DistTensor, DistEmbedding, DistMatrix, is_empty, empty
  - `tensor/dist_tensor.py` — WholeGraph 分布式张量/embedding 接口
  - `tensor/dist_matrix.py` — WholeGraph 分布式稀疏矩阵接口 (COO/CSC)
  - `tensor/utils.py` — WG 张量创建辅助函数
## migrate dd543dc: Heterogeneous Link Prediction Example for cuGraph-PyG

- **Upstream commit**: dd543dc (cugraph-gnn, NVIDIA)
- **Commit message**: `Heterogeneous Link Prediction Example for cuGraph-PyG`
- **Upstream diff** (10 files changed):
  - `python/cugraph-pyg/cugraph_pyg/examples/taobao_mnmg.py` — 新增 541 行（初版异构链路预测示例）
  - `python/cugraph-pyg/cugraph_pyg/sampler/sampler.py` — 六处修复（input_type/input_index/edge_label/vertex_types）
  - `python/cugraph-pyg/cugraph_pyg/sampler/sampler_utils.py` — 移除 neg_sample 中错误的 all_reduce
  - `python/cugraph-pyg/cugraph_pyg/loader/neighbor_loader.py` — dict → numpy fanout 转换
  - `python/cugraph-pyg/cugraph_pyg/loader/link_neighbor_loader.py` — dict → numpy fanout 转换
  - `python/cugraph-pyg/cugraph_pyg/data/graph_store.py` — `_numeric_edge_types` 类型注解精化
  - `conda/environments/*.yaml` + `dependencies.yaml` — ogb 依赖 + pytorch <2.6a0 上限（临时兼容）
  - `python/cugraph-pyg/cugraph_pyg/tests/loader/test_neighbor_loader_mg.py` — 修复测试缺失 destroy_process_group

- **Walpurgis 迁移状态**:

  | 上游文件 | 上游变化 | Walpurgis 状态 |
  |---|---|---|
  | `cugraph_pyg/__init__.py` | `+import cugraph_pyg.tensor` | **已迁移** → `src/walpurgis/__init__.py` 加入 `import walpurgis.tensor` |
  | `tensor/__init__.py` | 新增文件 | **已迁移** → `src/walpurgis/tensor/__init__.py` (加 debug import 断点) |
  | `tensor/dist_tensor.py` | 新增文件 | **已迁移+改写** → 全链路断点 + 文件路径错误提示 + 越界守护 |
  | `tensor/dist_matrix.py` | 新增文件 | **已迁移+改写** → `_local_range()` 提取 + local_col/local_row 断点 |
  | `tensor/utils.py` | 新增文件 | **已迁移+改写** → assert→raise, has_nvlink 环境变量兜底, 文件存在性断点 |

- **鲁迅拿法20%改写要点**:

  | 改写点 | 上游写法 | Walpurgis 写法 |
  |---|---|---|
  | 无调试出口 | 所有函数无任何 print/log | `_dbg()` 断点函数, `WALPURGIS_DEBUG=1` 时输出进程/时间/参数 |
  | `assert` 校验 | `assert len(shape) == 2` (被 `-O` 跳过) | `raise ValueError(...)` (不可跳过) |
  | `has_nvlink_network` 环境变量 | `int(os.environ["LOCAL_WORLD_SIZE"])` → KeyError | `os.environ.get(...)` + 空字符串兜底 |
  | 未知文件扩展名报错 | "Unsupported source type" | 列出已支持格式 (.pt/.npy/list) 的可读提示 |
  | `local_col/local_row` 代码重复 | 两个属性各有一份相同的分片公式 | 提取为 `_local_range(sz, world_size, rank)` |
  | `copy_host_global_tensor_to_local` 越界 | 无越界检查 | `end_idx > host_tensor.shape[0]` → `IndexError` |
  | `chunked` backend | 落入 else 报 Unsupported | 明确提示 "chunked 在 WG API 中尚未稳定" |
  | import 诊断 | 无 | `tensor/__init__.py` 加载时打印已注册符号及其 module |

- **Knuth 审查**:

  1. **diff 对比源**:
     上游 diff 本身只有 1 行 (`+import cugraph_pyg.tensor`)。真正的代码在子包四个文件中。
     对比最关键的偏差: `dist_matrix.py` 的 `local_col/local_row` 属性包含完全相同的分片公式两份——
     上游未提取成函数, 任何修改必须同步两处, 是典型的 DRY 违反; Walpurgis 版提取为 `_local_range()`。

  2. **用户角度 bug**:
     - `DistTensor(src="weights.bin")` — 上游报 "Unsupported source type", 用户不知道
       该传 `.pt` 还是别的。Walpurgis 版枚举支持格式。
     - `has_nvlink_network()` 在非 torchrun 启动的单卡推理场景下 (`LOCAL_WORLD_SIZE` 未设置)
       触发 `KeyError`, 上游静默崩溃; Walpurgis 版给出兜底值并打印提示。
     - `copy_host_global_tensor_to_local` 当 partition 不均时若 `host_tensor` 行数不足,
       上游 `tensor[start:end]` 切片静默截断 (不报错, 末尾 rank 拿到全 0); Walpurgis 版提前检测并 raise。

  3. **系统角度安全**:
     - `assert` 在 Python `-O` 优化模式下被完全跳过, 上游 `create_wg_dist_tensor_from_files`
       和 `create_wg_dist_tensor` 中多处 `assert` 作为唯一校验 — 生产环境优化编译时失去所有保护。
       Walpurgis 版全部替换为 `raise`。
     - `torch.load(src, mmap=True)` 在 PyTorch ≥ 2.0 下无 `weights_only` 参数会触发 FutureWarning,
       且加载任意文件路径存在反序列化风险 (pickle)。当前版本保持与上游一致,
       **建议后续迁移加入 `weights_only=True`** 或先校验文件哈希。

## migrate d306c72: Use PyTorch MemPool and Disable RMM Pool Allocator to Fix Broken Tests

- **Upstream commit**: d306c72 (cugraph-gnn, NVIDIA, PR #237)
- **Commit message**: `Use PyTorch MemPool and Disable RMM Pool Allocator to Fix Broken Tests`
- **Upstream diff** (2 files changed):
  - `python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_mnmg.py`:
    - `init_pytorch_worker` L48: `pool_allocator=True` → `False`
    - `__main__` L394-425: 数据加载 / barrier / 模型创建 / run_train 包裹进
      `with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):`
    - `dist.barrier()` 从 with 块外移入 with 块内
  - `python/cugraph-pyg/cugraph_pyg/examples/rgcn_link_class_mnmg.py`:
    - `init_pytorch_worker` L44: `pool_allocator=True` → `False`
    - `__main__` L278-392: 全部主逻辑包裹进 use_mem_pool context

- **Bug 根因**:
  `pool_allocator=True` 时 RMM 建立独立内存池，PyTorch 亦有自己的 caching allocator，
  两个 pool 同时活跃，显存碎片化导致 OOM 或 CUDA illegal memory access。
  CI 环境（4-8GB 卡）seeds_per_call=20000 时确定性触发。
  `use_mem_pool(MemPool(rmm_torch_allocator.allocator()))` 将 PyTorch tensor 分配
  重定向到 RMM allocator，消除双 pool 竞争，统一内存管控。
  `dist.barrier()` 必须在 context 内执行，否则某 rank 先退出 with 块销毁 allocator，
  其他 rank 仍在 context 内访问，导致跨 rank 内存损坏。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 d306c72 | Walpurgis 迁移 |
     |---|---|
     | 裸 `with torch.cuda.use_mem_pool(...)` 包裹 100+ 行 | `MemoryContext` dataclass，`__enter__`/`__exit__` 命名，调试打印 allocator 地址 + pool 引用计数 |
     | `init_pytorch_worker` 散落 import cupy / Device / set_allocator | `WorkerInit` dataclass，5 步分别命名 `_init_rmm` / `_init_cupy` / `_init_torch_device` / `_init_cugraph_comms` / `_init_wholegraph` |
     | 4 段 `print + assignment` 混合的广播序列 | `GraphBroadcaster` dataclass，`_broadcast_edge_rel_type` / `_broadcast_edge_index` / `_broadcast_splits` / `_broadcast_neg_splits` 四方法 |
     | 4 个 `os.path.join` 散落 `__main__` | `DataConfig` dataclass，`edge_path` / `feature_path` / `label_path` / `meta_path` 属性 |
     | 无调试信息 | 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖 allocator 生命周期 / 每步 broadcast shape / epoch loss / val/test acc |

  2. **用户角度 bug**:
     - `pool_allocator=True` 多卡训练偶发 `CUDA illegal memory access`，
       错误指向随机行（embedding lookup / conv），与业务逻辑无关，难以定位
     - `nvidia-smi` 显示仍有空闲显存但 OOM，
       原因是两 pool 碎片化导致最大连续块不足，用户误增 batch_size → 更快 OOM
     - rgcn 示例 `nr = [0, 0]` 在非 rank=0 进程初始化，
       旧代码 `dist.barrier()` 在 with 块外，极端情况下 barrier 完成但 allocator
       已被另一进程释放，跨 rank 内存损坏；新代码将 barrier 移入 with 块修复

  3. **系统角度安全**:
     - `rmm_torch_allocator.allocator()` 返回 capsule，`MemPool` 持有引用，
       with 块退出后 MemPool 析构，RMM 侧内存归还；
       `MemoryContext.__exit__` 在 DEBUG 模式下打印 pool 引用计数，帮助排查 use-after-free
     - `pool_allocator=False` 下 RMM 仍作为 cupy 的 allocator（cupy 侧不变），
       cupy tensor 和 torch tensor 统一走 RMM device allocator，无双重 free 风险
     - rgcn: `del dataset_obj`（rank=0）在 with 块内，CUDA tensor 归还给 MemPool，
       不挂起；`splits_storage` FeatureStore 及其内部 tensor 生命周期全程在 with 块内

### Walpurgis 迁移位置

**文件 1: `src/walpurgis/examples/gcn/gcn_dist_mnmg.py`** — 新增
**文件 2: `src/walpurgis/examples/rgcn/rgcn_link_class_mnmg.py`** — 新增

**迁移要点**:
- `MemoryContext`: dataclass 封装 `torch.cuda.use_mem_pool(MemPool(...))` 生命周期，
  DEBUG 模式打印 allocator capsule id / pool 引用计数 / 进入退出时间戳
- `WorkerInit`: dataclass 封装 `init_pytorch_worker`，5 步分别命名 + `_dbg`，
  `_init_rmm` 内联注释说明 `pool_allocator=False` 是 d306c72 核心修复
- `DataConfig`: dataclass 封装路径四元组，`debug_print()` 集中输出
- `GraphBroadcaster`（rgcn 专用）: dataclass 封装 4 段广播，每步打印 tensor shape
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  MemoryContext 进入/退出 + pool 引用计数 →
  WorkerInit 各子步状态 →
  load_partitioned_data / GraphBroadcaster 各 broadcast shape →
  run_train epoch / loss / val acc / test acc / mrr

**改写 20%（鲁迅拿法）**:
- `MemoryContext` dataclass 替代裸 `with torch.cuda.use_mem_pool(...)` 内联
- `WorkerInit` dataclass 替代散落 import 的裸函数
- `DataConfig` dataclass 替代四个散落 `os.path.join`
- `GraphBroadcaster` dataclass 替代 rgcn 中 4 段混合 print+assignment
- `_dbg()` 统一调试出口，替代散落 `print(..., flush=True)` 裸调
- `acc_val / acc_test` 零除防御（eval loader 为空时返回 0.0）
  | `taobao_mnmg.py` | 全新文件（初版，无 EdgeShuffler/timing） | **超前迁移**：Walpurgis 已包含 EdgeShuffler + DataPreprocessor + perf_counter 全链路 |
  | `sampler.py` — input_type canonical tuple | 边采样路径保留完整 (src,rel,dst) | **新增迁移** → `InputTypeResolver` 封装决策 |
  | `sampler.py` — input_index 负数过滤 | 过滤 `< 0` 标记，分离 num_pos/num_neg | **新增迁移** → `NegativeSeedFilter` 封装 |
  | `sampler.py` — edge_label 构建 | 三路径：negsampling/input_label/None | **新增迁移** → `EdgeLabelBuilder` 封装 |
  | `sampler.py` — sample_from_edges input_label | 传入 `index.label` | **新增迁移** → `LabelPassthrough` 语义 |
  | `sampler.py` — vertex_types 参数 | `sorted(_num_vertices().keys())` 传入 Reader | **新增迁移** → `VertexTypeRegistry` 语义 |
  | `sampler_utils.py` — 移除 all_reduce | neg_sample 本地化 | **新增迁移** → `NegSampleLocalizer` 封装 |
  | `neighbor_loader.py` + `link_neighbor_loader.py` | dict → numpy fanout 转换 | **新增迁移** → `FanoutConverter` 封装，消除重复 |
  | `graph_store.py` — 类型注解 | `Tuple[List, ...]` → `Tuple[List[Tuple[str,str,str]], ...]` | 纯注解改进，无运行时影响，已知晓 |
  | conda/yaml/dependencies | pytorch <2.6a0 + ogb 临时兼容 | 不迁移（Walpurgis 无 conda 环境文件） |
  | 测试文件 | `destroy_process_group` + 去掉 skip | 不迁移（Walpurgis 测试体系独立） |

- **核心修复详解（sampler.py 六处）**:

  **修复 A — input_type 降维 Bug**:
  ```python
  # 旧代码（dd543dc 之前）:
  input_type = pyg_can_etype[2]       # 永远是 str（"item"）

  # 新代码（dd543dc）:
  if "edge_inverse" in raw_sample_data:
      input_type = pyg_can_etype       # 边采样：完整 (src, rel, dst) tuple
  else:
      input_type = pyg_can_etype[2]   # 节点采样：str
  ```
  边采样时 input_type 降维为 str → PyG 当作节点采样 → edge_label_index 索引错误节点。
  模型能运行，AUC 静默损坏（约 0.50～0.52），用户以为模型差。

  **修复 B — input_index 负数标记未过滤**:
  ```python
  # 旧代码: input_index 含 pylibcugraph 负采样标记（< 0）
  # → Python tensor[-1] = tensor[N-1]（最后节点），不报错，梯度方向随机

  # 新代码: 过滤后只保留正样本索引
  input_index_pos = input_index_raw[input_index_raw >= 0]
  ```

  **修复 C — edge_label 永远 None**:
  ```python
  # 旧代码: metadata = (input_index, edge_inverse, None, None)
  # 新代码: metadata = (input_index, edge_inverse, edge_label, None)
  # edge_label 由 EdgeLabelBuilder 的三路径逻辑构建
  ```

  **修复 D — all_reduce 引入的多 GPU 负采样 Bug**:
  ```python
  # 旧代码（sampler_utils.py）:
  if graph_store.is_multi_gpu:
      num_neg_global = torch.tensor([num_neg], device="cuda")
      torch.distributed.all_reduce(num_neg_global, op=ReduceOp.SUM)
      num_neg = int(num_neg_global)   # 全局总量，world_size 倍膨胀
  # pylibcugraph.negative_sampling(..., num_neg, ...)

  # 新代码: 直接用本地 num_neg，无 all_reduce
  ```
  多 GPU（8卡）时每 rank 生成 8 倍负样本，正负样本比 1:8 → AUC 趋向随机基线。
  同时 all_reduce 在边稀疏 rank 触发死锁（NCCL 10 分钟超时 abort）。

  **修复 E — dict 形式 num_neighbors 不支持**:
  ```python
  # 旧代码: num_neighbors（dict）直接传入底层 → TypeError
  # 新代码: 按 sorted_keys 顺序转换为 flat numpy int32 array
  # layout: na[hop * num_types + type_idx] = fanout
  ```
  所有异构图用户传 dict 形式 num_neighbors（PyG 推荐写法）均触发此 bug。

- **Knuth 审查**:
  1. **diff 对比源**:

     | 上游 dd543dc | Walpurgis 迁移 |
     |---|---|
     | `input_type = pyg_can_etype if "edge_inverse" else pyg_can_etype[2]` | `InputTypeResolver.resolve()` 封装，断点打印决策路径 |
     | `input_index_raw[input_index_raw >= 0]` 散落三行 | `NegativeSeedFilter.split()` 返回 `(pos_index, num_neg)`，断点打印过滤统计 |
     | if-elif-else edge_label 三路径内联 | `EdgeLabelBuilder.build()` 命名三路径，断点打印构建来源 |
     | 两处重复 dict→numpy 转换（neighbor/link_neighbor loader）| `FanoutConverter.convert()` 消除重复，断点打印 sorted_keys + na array |
     | all_reduce 直接删除，无注释 | `NegSampleLocalizer.local_count()` 命名\"为什么不需要 all_reduce\"，含详细注释 |

  2. **用户角度 bug**:
     - **100% 触发**: 异构图用户传 `num_neighbors=dict`（PyG 官方推荐写法），
       TypeError 崩溃，错误指向 cuGraph 内部，用户不知 dict 不被支持。
     - **静默 AUC 损坏**: `input_type` 降维 + `input_index` 含负数 → 模型训练不崩溃，
       AUC 约 0.50，用户误以为任务难、模型差，实为数据路由错误。
     - **多 GPU 负采样膨胀**: `world_size=8` 时负样本 8x → 正负比 1:8 → 模型偏向\"全负\"。
     - **NCCL 死锁**: 稀疏图某 rank 本地边=0 跳过 neg_sample，
       all_reduce 挂起，10 分钟超时 abort，无法从 checkpoint 恢复。

  3. **系统角度安全**:
     - `pylibcugraph.negative_sampling` 是 rank-local 函数（接受 resource_handle 非 communicator），
       all_reduce 引入隐式同步语义，破坏本地性保证，制造死锁风险。
     - `input_index` 负数作 tensor 索引：Python 语义（从末尾倒数）vs 业务语义（无效标记），
       完全相反，PyTorch 不抛异常，属\"API 语义陷阱\"（semantic trap）。
     - `FanoutConverter` 的 fanout layout 依赖 `_numeric_edge_types` 返回的 sorted_keys
       与 cuGraph 内部顺序一致；`sorted()` 基于 Python tuple 字典序，确定性有保证，
       但需注意多 rank 边类型注册顺序必须完全相同，否则 fanout 解释错误 → DDP 梯度对齐失败。

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/hetero_link_pred_fixes.py`** — 新增

**迁移要点**:
- `InputTypeResolver`: 封装\"边采样 vs 节点采样\" input_type 决策，`resolve()` 静态方法
- `NegativeSeedFilter`: 封装 input_index 负数过滤，`split()` 返回 `(pos_index, num_neg)`
- `EdgeLabelBuilder`: 封装三路径 edge_label 构建，`build()` 静态方法
- `FanoutConverter`: 封装 dict → numpy fanout 转换，消除 neighbor/link_neighbor loader 重复
- `NegSampleLocalizer`: 命名\"不需要 all_reduce\"的语义，`local_count()` 静态方法
- 5 类断点调试 print，`WALPURGIS_DEBUG=1` 控制，覆盖全链路:
  `INPUT_TYPE`: 决策路径（边 vs 节点）+ pyg_can_etype
  `NEG_FILTER`: 过滤前后 seed 数 + min/max 值域
  `EDGE_LABEL`: 构建路径（negsampling/input_label/None）+ shape
  `FANOUT`: sorted_keys 顺序 + 逐 hop/type 填充 + 最终 na array
  `NEG_SAMPLE`: 本地 num_neg + 不 all_reduce 的原因说明

## migrate c07eea7: [FEA] New MovieLens Example, Add Timing to Taobao

- **Upstream commit**: c07eea7 (cugraph-gnn, NVIDIA)
- **Commit message**: `[FEA] New MovieLens Example, Add Timing to Taobao`
- **Upstream diff** (3 files changed):
  - `python/cugraph-pyg/cugraph_pyg/examples/movielens_mnmg.py` — 新增全新 MovieLens 多机多卡示例
  - `python/cugraph-pyg/cugraph_pyg/examples/taobao_mnmg.py` — 新增 `balance_shuffle_edge_split` 函数 + timing
  - `python/cugraph-pyg/cugraph_pyg/loader/link_loader.py` — 修复 `get_edge_label_index` 调用的类型包裹逻辑

- **Walpurgis 迁移状态**:

  | 上游文件 | 上游变化 | Walpurgis 状态 |
  |---|---|---|
  | `movielens_mnmg.py` | 全新文件（初版）| **超前迁移**：walpurgis 318ae6c 版本已包含 DDP、EncoderShapeGuard、CugraphWorkerSession 等增强，无需覆盖 |
  | `taobao_mnmg.py` | 新增 `balance_shuffle_edge_split` + timing | **已覆盖**：walpurgis f2b7f50 迁移时已实现 EdgeShuffler + perf_counter 全链路 |
  | `link_loader.py` | `isinstance(edge_label_index, torch.Tensor)` 条件包裹 | **新增迁移** → `src/walpurgis/dataloader/link_loader_edge_index_guard.py` |

- **link_loader.py 修复详解**:

  上游旧代码（c07eea7 之前）:
  ```python
  ) = torch_geometric.loader.utils.get_edge_label_index(
      data,
      (None, edge_label_index),   # ← 无论类型如何，一律包裹
  )
  ```

  上游新代码（c07eea7）:
  ```python
  ) = torch_geometric.loader.utils.get_edge_label_index(
      data,
      (None, edge_label_index)
      if isinstance(edge_label_index, torch.Tensor)
      else edge_label_index,      # ← tuple 类型直接透传
  )
  ```

  **Bug 根因**:
  `get_edge_label_index` 第二参数协议：
  - `torch.Tensor` → 需包裹为 `(None, tensor)` 表示类型未指定
  - `(edge_type, tensor)` tuple → 直接透传，内含类型信息
  旧代码对 tuple 输入二次包裹为 `(None, (edge_type, tensor))`，
  导致 `input_type` 解析为 `None`，`_vertex_offsets` 以 `None` 索引 → `KeyError: None`，
  错误指向库内部，用户无从定位。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 c07eea7 | Walpurgis 迁移 |
     |---|---|
     | 内联 `isinstance` 条件表达式一行 | `resolve_edge_label_input()` 命名函数 |
     | 无文档说明两种类型路径 | docstring 详述路径 A（Tensor）/ 路径 B（tuple）|
     | 错误时 PyG 内部 KeyError | `EdgeLabelInputError` 专用异常，携带类型与 shape |
     | 无调试输出 | `_dbg_edge_label_input()` 在 WALPURGIS_DEBUG=1 时打印两侧对比 |
     | 无事后校验 | `assert_resolved_input_compatible()` 检查 drop_last 与边数兼容性 |
     | 无可执行验证 | `_smoke_test()` 覆盖路径 A / B / 无效类型 / drop_last 四个断言 |

  2. **用户角度 bug**:
     - 最常见用法：`LinkNeighborLoader(edge_label_index=(("user","rates","movie"), tensor))`
       `edge_label_index` 是 tuple，旧代码包裹后变为 `(None, (edge_type, tensor))`。
       `get_edge_label_index` 内部 `input_type = arg[0]` 得到 `None`，
       后续 `data[1]._vertex_offsets[None]` → `KeyError: None`。
       stack trace 深入 PyG + cugraph-pyg 内部，用户第一反应是版本兼容问题，
       而非自己传入参数的类型问题。
     - 只有传裸 Tensor（同构图）时旧代码偶然正确，
       异构图（hetero）是 cugraph-pyg 的核心使用场景，bug 覆盖面极广。
     - `drop_last=True` 但边数不足时，旧代码等到 DataLoader 迭代阶段才发现空 batch，
       训练循环无输出，用户以为程序挂起。新增 `assert_resolved_input_compatible` 提前报错。

  3. **系统角度安全**:
     - `isinstance(edge_label_index, torch.Tensor)` 纯 Python 类型判断，
       无 GPU 操作，无性能开销，不影响训练吞吐。
     - 判断分支二选一：Tensor → 包裹；非 Tensor → 透传。
       未来若支持 `EdgeIndex` 等新类型，`else` 透传策略仍能正确工作，
       只要 `get_edge_label_index` 上游能处理即可。
     - 旧代码的双重嵌套 `(None, (edge_type, tensor))` 不会在 Python 层报错，
       错误被推迟到 C++ / CUDA 节点偏移计算层，错误信息完全不可读。
     - Walpurgis `EdgeLabelInputError` 在 Python 层拦截，错误位置明确，
       类型与 shape 信息完整，便于 CI 捕获和用户自查。

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/link_loader_edge_index_guard.py`** — 新增

**迁移要点**:
- `resolve_edge_label_input(edge_label_index)`: 封装 c07eea7 的 isinstance 分发，
  给「无声的类型判断」一个名字和文档，路径 A（Tensor）/ 路径 B（tuple）明确分离
- `EdgeLabelInputError`: 专用异常，携带输入类型与 shape，替代下游 KeyError/AttributeError
- `_dbg_edge_label_input()`: 断点调试出口，WALPURGIS_DEBUG=1 时打印输入→输出对比
- `assert_resolved_input_compatible()`: 事后 drop_last 兼容性校验，
  提前捕获「边数不足」配置错误，避免迭代阶段静默空 batch
- `_smoke_test()`: 四个断言覆盖路径 A / B / 无效类型 / drop_last，
  运行 `python -m walpurgis.dataloader.link_loader_edge_index_guard` 验证

---

## migrate f2b7f50: [BUG] Fix shuffle on single GPU in Taobao Example

- **Upstream commit**: f2b7f50 (cugraph-gnn, NVIDIA)
- **Commit message**: `[BUG] Fix shuffle on single GPU in Taobao Example`
- **Upstream diff** (1 file changed):
  - `python/cugraph-pyg/cugraph_pyg/examples/taobao_mnmg.py`:
    - `preprocess_and_partition` L173: 新增 `print(data)`（调试断点）
    - `balance_shuffle_edge_split` L411-418: 重写 dst_rank 切片逻辑
      - 旧: `if rank > 0 and rank < world_size - 1 / elif rank == 0 / else`
      - 新: `if world_size == 1: local_rank_t = dst_rank` (单卡 early-return)
      - 新: `else: start/end 二元组` 替代三段 if-elif-else

- **Bug 根因**:
  `balance_shuffle_edge_split` 中:
  ```python
  edge_offsets = num_edges.cumsum(0).cpu()[:-1]
  ```
  world_size==1 时 `num_edges.shape=(1,)`，`cumsum[:-1]` → 空 tensor。
  旧代码 `elif rank==0` 触发 `edge_offsets[0]`，IndexError 确定性触发。
  新代码提前判断 `world_size==1`，直接 `local_rank_t = dst_rank`，
  跳过 `edge_offsets` 访问，从根本上规避 IndexError。
  多卡路径同时重写为 start/end 二元组，消除三段式边界条件穷举。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 f2b7f50 | Walpurgis 迁移 |
     |---|---|
     | `if world_size == 1: local_rank_t = dst_rank` 两行内联 | `EdgeShuffler._single_gpu_path()` 命名路径 |
     | `start/end` 二行散落 else 块 | `_compute_local_slice()` 静态方法，加断点 print |
     | 三段 all_to_all 重复代码块 | `_scatter_gather_tensor()` 封装复用 |
     | `print(data)` 裸调 | `_dbg(..., tag="PREPROCESS")` 仅 DEBUG 时输出 |
     | 无单卡 guard 注释 | 内联注释说明旧代码为何在单卡崩溃 |

  2. **用户角度 bug**:
     - 单卡调试（最常见场景）：`torchrun --nproc_per_node=1` 运行，
       到 `balance_shuffle_edge_split` 直接 `IndexError: index 0 is out of bounds for dimension 0 with size 0`
       错误指向 `edge_offsets[0]`，与业务逻辑毫无关联，难以定位
     - 错误发生在 broadcast 之后，浪费了集体操作开销
     - 多卡路径旧三段式：rank==0 / 0<rank<N-1 / rank==N-1，
       穷举边界条件，任何 world_size 变化都需重新验证三段覆盖

  3. **系统角度安全**:
     - `edge_offsets = cumsum[:-1]` 长度恒为 `world_size - 1`，
       world_size==1 时为空，旧代码无 guard，单卡必崩
     - `dst_rank = randperm(total) % world_size`，world_size==1 时全为 0，合法，
       问题只在后续切片逻辑
     - 新代码 world_size==1 时 `all_to_all` 退化为单槽自环（`rx[0]=s[0]`），
       torch.distributed 在单进程组下此操作合法
     - `None` 作为切片 end 等价于"到末尾"，避免了旧代码 else 分支的
       `dst_rank[edge_offsets[-1]:]` 隐含语义（edge_offsets[-1] 恰好是末尾 rank 起点）

### Walpurgis 迁移位置

**文件: `src/walpurgis/examples/taobao/taobao_mnmg.py`** — 新增

**迁移要点**:
- `DataPreprocessor`: dataclass 封装 preprocess_and_partition 前段 del 链，
  `_clean()` 方法集中管理，`_dbg()` 替代上游裸 `print(data)`
- `EdgeShuffler`: dataclass 封装 balance_shuffle_edge_split 核心逻辑，
  `_compute_local_slice()` 静态方法封装 start/end 计算 + 断点调试 print，
  `_scatter_gather_tensor()` 封装三段重复 all_to_all 代码块
- `balance_shuffle_edge_split()`: 公共接口保持上游签名，委托给 `EdgeShuffler.split()`
- `_edge_shuffler`: 模块级单例，避免每次 create_loader 重复构造
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  DataPreprocessor._clean data 结构前后 →
  EdgeShuffler.split world_size/rank/edge_offsets →
  _compute_local_slice start/end per rank →
  balance_shuffle_edge_split broadcast 前后 dst_rank 分布 →
  train/test batch shape → epoch loss/auc → main barrier 检查点

**改写 20%（鲁迅拿法）**:
- `DataPreprocessor` dataclass 替代裸函数 + 散落 del 链
- `EdgeShuffler` dataclass 替代 230 行大函数内的内联逻辑
- `_compute_local_slice()` 命名静态方法替代两行无名 start/end 赋值
- `_scatter_gather_tensor()` 消除三段重复 all_to_all 代码块
- `balance_shuffle_edge_split()` 薄包装保持上游 API 兼容
- 单卡 ZeroDivisionError 防御（train loader 为空时返回 0.0）

---

## migrate 66da9ac: [BUG] Fix input ID creation to use shape[-1] instead of len

- **Upstream commit**: 66da9ac (cugraph-gnn, NVIDIA, 2025-09-24)
- **Commit message**: `[BUG] Fix input ID creation to use shape[-1] instead of len`
- **Upstream diff** (1 file changed, 1 insertion, 1 deletion):
  - `python/cugraph-pyg/cugraph_pyg/sampler/distributed_sampler.py`:
    - `BaseDistributedSampler.sample_from_edges` L694:
      - 旧: `input_id = torch.arange(len(edges), dtype=torch.int64, device="cpu")`
      - 新: `input_id = torch.arange(edges.shape[-1], dtype=torch.int64, device="cpu")`

- **Bug 根因**:
  `edges` 是 `2 × N` tensor（2 行=src/dst，N 列=边数）。
  `len(edges)` 返回第 0 维 → 恒为 **2**，与实际边数 N 无关。
  `edges.shape[-1]` 返回最后一维 → 正确得到 **N**。
  当 N > 2 时，旧代码 `input_id = tensor([0, 1])`，仅 2 个 index，
  与实际 batch 数量严重不符，导致下游 batch 分组以最短 tensor 截断，
  N-2 条边静默丢失。N == 2 时旧代码偶然正确，可能掩盖 bug。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 66da9ac | Walpurgis 迁移 |
     |---|---|
     | `torch.arange(edges.shape[-1], ...)` 内联 | `make_edge_input_id(edges)` 封装函数 |
     | 无前置形状校验 | `validate_edges_shape` 检查 ndim==2 且 shape[0]==2 |
     | 无后置一致性校验 | `assert_input_id_consistent` 检查 len(input_id)==shape[-1] |
     | 无调试输出 | `WALPURGIS_DEBUG=1` 打印 `len(edges)` vs `edges.shape[-1]` 对比 |
     | 无专用异常 | `EdgeInputIdError` 携带 edges.shape 信息 |
     | None-check 内联 | `resolve_edge_input_id` 完整封装 None-check + 构造 + 校验 |

  2. **用户角度 bug**:
     - 调用 `sampler.sample_from_edges(edges)`（不传 input_id，最常见用法）。
       edges 有 1024 条，`len(edges)=2`，`input_id=tensor([0,1])`。
       下游 `__get_call_groups` 以 input_id 长度切分 batch，
       实际只处理 2 条边，其余 1022 条静默丢失。
       训练数据量不足导致模型严重欠拟合，无任何异常抛出，难以定位。
     - 只有 N==2（即 edges 恰好只有 2 列）时旧代码偶然正确，
       小规模测试可能通过，生产规模失效的经典案例。

  3. **系统角度安全**:
     - `len()` 对 PyTorch tensor 返回第 0 维长度（Python 数据模型标准行为）。
       边索引约定 2×N 中，第 0 维永远是 2（src/dst 两行），
       `len()` 的"正确"语义（行数）与业务语义（边数）完全相反，是 API 语义陷阱。
     - 静默截断：input_id 短于 edges.shape[-1] 时，上游不做长度一致性校验，
       短 input_id 被正常传入后续逻辑，多余的边被无声丢弃，系统不崩溃。
     - 转置陷阱：若用户误传 N×2（转置）的 edges，
       旧代码 `len(edges)=N`（偶然正确），新代码 `shape[-1]=2`（错误）；
       `validate_edges_shape` 检查 `shape[0]==2` 可捕获此类转置错误。

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/edge_input_id.py`** — 新增

**迁移要点**:
- `EdgeInputIdError`: 专用异常类，携带 edges.shape 信息，上游无此错误类型
- `validate_edges_shape`: 前置 2×N 形状校验，检查 ndim==2 且 shape[0]==2，
  同时在 DEBUG 模式打印 `len(edges)` vs `shape[-1]` 两者对比，直指 bug 根因
- `make_edge_input_id`: 替代内联 `torch.arange(edges.shape[-1], ...)`，
  统一 input_id 构建入口，断点调试打印 BUG WOULD HAVE OCCURRED 提示
- `assert_input_id_consistent`: 后置一致性校验，检查外部传入 input_id 长度，
  上游 sample_from_edges 对非 None 的 input_id 无任何校验
- `resolve_edge_input_id`: 完整封装 None-check + 构造/校验，
  drop-in 替代 sample_from_edges 中的 if input_id is None 块

**改写 20%（鲁迅拿法）**:
- `validate_edges_shape` 捕获转置 N×2 输入（上游无此保护）
- `assert_input_id_consistent` 保护外部传入的自定义 input_id 长度一致性
- `EdgeInputIdError` 将形状信息完整携带进异常，上游直接 IndexError 或静默
- `resolve_edge_input_id` 将三步逻辑合并为单函数，上游三步散落于 sample_from_edges 中

---

## migrate a72a521: fix: fixes memory context leak

- **Upstream commit**: a72a521 (cugraph-gnn, NVIDIA, 2025-10-13)
- **Commit message**: `fix: fixes memory context leak (#332)`
- **Upstream diff** (1 file changed, 3 insertions, 4 deletions):
  - `cpp/src/wholememory_ops/temp_memory_handle.hpp`:
    - `free_memory()` 守卫条件 `ptr_ != nullptr` → `memory_context_ != nullptr`
    - 同时将 `free_fn` 调用提取为 `free_data()` 复用（上游原本已有，修复使其被正确调用）
    - `ptr_ = nullptr` 清零移入 `free_data()`，`free_memory()` 不再重复清零

- **Bug 根因**:
  `free_memory()` 旧实现用 `ptr_ != nullptr` 判断是否需要销毁 `memory_context_`。
  但 `free_data()` 先于 `destroy_memory_context_fn` 被调用，已将 `ptr_` 清零，
  导致以下两种场景均发生 memory_context_ 泄漏：
  (1) 构造后从未 malloc，`ptr_` 始终为 nullptr → destroy 永不触发；
  (2) 调用方显式调用 `free_data()` 后对象析构 → ptr_ 已为 nullptr → destroy 跳过。
  修复将守卫换成 `memory_context_ != nullptr`，两个资源的生命周期彻底解耦。

- **Walpurgis 迁移改写 (鲁迅拿法 20%)**:
  | 上游 a72a521 | Walpurgis 迁移 |
  |---|---|
  | device/host/pinned malloc 三路重复代码 | 抽取 `alloc_impl_()` 统一实现 |
  | 无入参校验 | 构造函数 `assert(env_fns != nullptr)` 快速失败 |
  | 无调试输出 | `WALPURGIS_DEBUG` 门控 7 处断点 print，覆盖 create/malloc/free/destroy |
  | `size_t` 直接赋给 `int64_t sizes[0]` | `static_cast<int64_t>` 显式转换，消除符号警告 |
  | copy ctor 未删除 | 显式 `= delete`，防止浅拷贝导致 double-free |

- **Knuth 审查**:
  1. **diff 对比源**:
     旧代码 `free_memory()` 中 `free_fn` + `destroy_memory_context_fn` 均在
     `if (ptr_ != nullptr)` 内，两个不同资源共享同一守卫——设计上即为错误。
     修复将 `free_fn` 移入已有的 `free_data()`，`destroy` 独立守卫 `memory_context_`，
     责任边界清晰，符合单一职责原则。

  2. **用户角度 bug**:
     在 WholeMemory GNN 训练循环中，每个 mini-batch 会构造若干 `temp_memory_handle`
     对象做临时 tensor 缓冲。若 batch 处理中途出现错误（如采样为空提前 return），
     对象可能在 `device_malloc` 之前被析构，或在 `free_data()` 之后被析构。
     旧代码两种路径均泄漏 `memory_context_`。训练数千 batch 后 CUDA memory context
     池耗尽，报 `WHOLEMEMORY_INVALID_VALUE` 或 OOM，难以定位，因为泄漏点距崩溃点
     时间跨度长。新代码 RAII 完全兑现，用户无感知。

  3. **系统角度安全**:
     `create_memory_context_fn` / `destroy_memory_context_fn` 维护后端 context 表
     的引用计数，为配对 API contract。旧代码破坏该 contract，造成 handle 泄漏而非
     纯内存泄漏——valgrind 无法检测（GPU side），需专用工具（如 compute-sanitizer）。
     Walpurgis 迁移中 `assert` + WALPURGIS_DEBUG 断点覆盖了构造/析构的全部关键状态
     转换点，可在测试阶段快速定位任何 context 生命周期异常。

---

## migrate 24e91be: [BUG] Specify Input Type and Assign Output to Correct Type

- **Upstream commit**: 24e91be (cugraph-gnn, NVIDIA, 2025-07-17)
- **Commit message**: `[BUG] Specify Input Type and Assign Output to Correct Type`
- **Upstream diff** (2 files changed, 154 insertions, 18 deletions):
  - `python/cugraph-pyg/cugraph_pyg/sampler/sampler.py`:
    - `BaseSampler.sample_from_nodes`: 新增 `metadata={"input_type": index.input_type}` 传入采样器
    - `BaseSampler.sample_from_edges`: 同上，新增 metadata 传递
    - `HeterogeneousSampleReader.__decode_coo`:
      - 函数签名 `Dict[str, Tensor]` → `Dict[str, Union[Tensor, str, Tuple[str,str,str]]]`
      - `input_type = raw_sample_data["input_type"]`（改为从 metadata 读取，不再在循环里猜测）
      - 新增 `integer_input_type`，按 edge/node 分支匹配后赋值
      - 边采样新增 src 侧 `num_sampled_nodes[0]` 更新（旧代码遗漏）
      - `edge_inverse.view(2,-1)` 后对 src/dst 行做 vertex_offset de-offset（旧代码缺失）
      - 函数签名同步升级（`_decode`, `__decode_csc`, `__decode_coo`）
    - `test_neighbor_loader.py`: 新增 `test_neighbor_loader_hetero_linkpred` 测试用例

- **Bug 根因**:
  **Bug 1（output 写入错误 type）**: 旧代码通过检查 `col[pyg_can_etype][:hop0].numel() > 0`
  来反推 input_type 是哪个边类型，在多边类型/共享节点类型场景下会猜错。
  最终 `num_sampled_nodes`、`num_sampled_edges` 填入了错误类型的桶，
  `input_id` 也被赋给了错误的 data[input_type]。程序不崩溃，但输出数据静默错误。
  修复：input_type 作为 metadata 从 BaseSampler 层一路传下来，_decode 直接读取。

  **Bug 2（edge_inverse 未 de-offset）**: cuGraph sampler 输出的节点编号是加了
  全局 vertex_offset 的（异构图中各类型节点在全局 ID 空间有偏移）。
  旧代码把 edge_inverse 直接塞进 metadata，PyG 拿到的是全局 ID，
  embedding lookup 会命中错误行或越界，训练精度悄悄降低。
  修复：在 __decode_coo 末尾对 edge_inverse[0]/[1] 分别减去 src/dst vertex offset。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 24e91be | Walpurgis 迁移 |
     |---|---|
     | `input_type = raw_sample_data["input_type"]` | `validate_raw_sample_data_input_type()` 读取并立即校验 |
     | `if integer_input_type is None: raise ValueError` | `resolve_integer_input_type()` 遍历后未匹配则 raise，附完整 edge_types 列表 |
     | `edge_inverse[0] -= vertex_offsets[src_types[integer_input_type]]` | `deoffset_edge_inverse()` 封装，附 de-offset 前后 min/max debug print |
     | `metadata = ({"input_type": index.input_type} if ... else None)` | `build_sampler_metadata()` 可独立测试的工厂函数 |
     | 类型注解 `Dict[str, Union[Tensor, str, Tuple]]` | `InputTypeSpec` dataclass 封装，`validate()` 运行时校验 |
     | 无 debug 日志 | WALPURGIS_DEBUG=1 门控 7 处断点 print |

  2. **用户角度 bug 排查**:
     - 使用 LinkNeighborLoader 做异构图链路预测时，模型收敛曲线异常（loss 居高不下），
       但无任何 exception，难以定位。根因是 edge_inverse 含全局 offset，
       导致 edge_label_index 引用了错误节点，loss 计算基于错误的节点对。
     - NeighborLoader 异构图采样时，单个 batch 的 num_sampled_nodes 统计对某些
       vertex type 返回 0（bug 1），GNN 层 aggregate 时跳过了这些节点，
       模型实际看到的子图比预期小，精度下降但无报错。

  3. **系统角度安全**:
     - 旧代码"猜测" input_type 的逻辑依赖 col[:hop0].numel() > 0，这是非确定性的：
       若一个 batch 里恰好某个边类型 hop0 采样为空，猜测逻辑跳过，
       最终 input_type 停留在上一个非空的边类型，写入错误 data bucket，静默数据污染。
     - de-offset 缺失是系统级 API contract 违反：cuGraph sampler 的 contract 是
       "输出全局 ID"，PyG 的 contract 是"输入局部 ID"。两者之间本应有转换层，
       旧代码转换层缺失，两侧 contract 各自成立但接口处腐蚀。
     - Walpurgis 迁移中 `deoffset_edge_inverse` 对节点采样路径 raise ValueError，
       防止调用方混淆 node-input 和 edge-input 场景，系统边界更清晰。

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/hetero_sample_reader.py`** — 新增

**迁移要点**:
- `InputTypeSpec`: dataclass，封装 `Union[str, Tuple[str,str,str]]` 语义，
  `validate()` 运行时类型校验，`matches_edge_type()` 统一匹配逻辑
- `resolve_integer_input_type()`: 遍历 edge_types，按 edge/node 分支返回 integer index；
  未匹配则 raise 含完整信息的 ValueError
- `update_num_sampled_nodes_for_input()`: 封装 hop0 num_sampled_nodes 更新，
  边采样同时更新 src+dst（24e91be 新增），节点采样保留 numel()>0 guard
- `deoffset_edge_inverse()`: 对应上游 edge_inverse de-offset 修复，
  节点采样路径 raise ValueError（系统级 contract 保护）
- `build_sampler_metadata()`: 封装 BaseSampler 端的 metadata 构建，
  对应 sample_from_nodes 和 sample_from_edges 各新增一段
- `validate_raw_sample_data_input_type()`: _decode 入口校验，
  对应上游类型注解升级 + 缺 key 时的 ValueError

**改写20%（鲁迅拿法）**:
- `InputTypeSpec` 将上游散落三处的 `isinstance(input_type, str)` 判断集中为值对象方法
- `resolve_integer_input_type()` 将 for 循环里隐式的副作用赋值改为显式 return，
  未匹配时 raise 含 edge_types 列表的 ValueError（上游只写 "did not match any edge type"）
- `deoffset_edge_inverse()` 将 de-offset 从 decode_coo 末尾 else 分支提取为独立函数，
  节点采样路径显式 raise，而非默默跳过
- `validate_raw_sample_data_input_type()` 运行时校验，
  上游只升级了类型注解无运行时保护
- 全链路 7 处断点 print（WALPURGIS_DEBUG=1 开启）:
  1. `InputTypeSpec.validate` — raw type 确认
  2. `build_sampler_metadata` — metadata 构建确认
  3. `validate_raw_sample_data_input_type` — keys 列表 + input_type 值
  4. `resolve_integer_input_type` 入口 + 匹配路径（边/节点）+ integer_input_type 值
  5. `update_num_sampled_nodes_for_input` 边/节点分支各一处
  6. `deoffset_edge_inverse` — src/dst offset 值 + de-offset 前后 min/max
## migrate 940ab01: [FEA] Add Elliptic Bitcoin fraud example

- **Upstream commit**: 940ab01 (cugraph-gnn, NVIDIA)
- **Commit message**: `[FEA] Add Elliptic Bitcoin fraud example`
- **Upstream diff** (9 files changed):
  - `ci/run_cugraph_pyg_pytests.sh` — 新增 bitcoin example 运行命令
  - `ci/test_wheel_cugraph-pyg.sh` — 同上
  - `conda/environments/all_cuda-128_arch-*.yaml` — 新增 `cuml==25.8.*` 依赖
  - `dependencies.yaml` — 新增 `depends_on_cuml` 依赖块 (4处引用)
  - `python/cugraph-pyg/pyproject.toml` — test deps 新增 `cuml==25.8.*`
  - `examples/fraud/README.md` — 新增，10行说明
  - `examples/fraud/bitcoin_mnmg.py` — 新增，280行，GNN多GPU训练 + 嵌入生成
  - `examples/fraud/bitcoin_rf.py` — 新增，83行，随机森林分类器

- **Knuth 审查**:
  1. diff 修改与源对比:
     - `bitcoin_rf.py` 的 `cudf.read_parquet(embedding_dir)` 读目录合并所有 parquet，
       但 `bitcoin_mnmg.py` 按 rank 写多文件，合并后行数 != `data.num_nodes`，
       `X[data.train_mask]` 越界或静默错位 (上游无对齐检查)
     - `bitcoin_mnmg.py` 推理阶段 `drop_last=True` 导致嵌入不完整，
       写回 feature_store 后 emb 与 y 对齐错位
     - 推理循环手动展开 `encoder.module.convs/norms/act/lin`，
       深度耦合 PyG 内部结构，模型升级时静默出错
     - parquet 文件名含超参拼接无时间戳，并发实验静默覆盖
  2. 用户角度 bug:
     - `EllipticBitcoin` 含 y=2 (unknown) 节点，`cross_entropy` 2分类头遇 y=2
       抛 IndexError 或产生错误梯度，用户看到 CUDA assert 或 loss=nan
     - `ix_train` tensor_split 末尾可能为空 (节点数不被 world_size 整除)，
       空 input_nodes 的 NeighborLoader 行为版本依赖，可能挂起
     - `bitcoin_rf.py` 读目录合并时若有多次实验 parquet 混入，训练数据被污染
  3. 系统角度安全:
     - `embedding_dir` 含 "/" 或 ".." 的 encoder 字符串可路径穿越
     - `cugraph_comms_shutdown()` 裸调，OOM/NCCL 挂起时不执行，资源泄漏
     - `RandomForestClassifier()` 无随机种子，CI 结果不可复现
     - `rmm.reinitialize(pool_allocator=False)` 与同仓库 c3799ae 方向相反，一致性缺失

### Walpurgis 迁移位置

**文件: `src/walpurgis/examples/fraud/bitcoin_mnmg.py`** — 新增，GNN 多 GPU 训练

**文件: `src/walpurgis/examples/fraud/bitcoin_rf.py`** — 新增，随机森林分类器

**文件: `src/walpurgis/examples/fraud/README.md`** — 新增，说明文档

**迁移要点**:
- `BitcoinMnmgArgs`: dataclass 封装 argparse，`validate()` 含 encoder 合法性 + 路径安全检查
- `CugraphWorkerSession`: context manager 封装 init_pytorch_worker 生命周期，
  `__exit__` 保证异常路径也调用 `cugraph_comms_shutdown()`
- `BitcoinGraphBundle`: 值对象封装分布式图构建，`build()` 类方法集中构建
- `EmbeddingWriter`: 封装推理 + parquet 写出，加 timestamp 防并发覆盖
- `BitcoinRfArgs`: dataclass 封装参数，`validate()` 含路径安全检查
- `EmbeddingDataset`: 封装 `cudf.read_parquet` + mask 对齐检查 (上游无此保护)
- `RfExperiment`: 封装 RF 训练 + 评估，加 `random_state` + dtype 强制转换

**改写20%（鲁迅拿法）**:
- `CugraphWorkerSession` context manager 替代裸函数 + 末尾裸调 shutdown
- `BitcoinGraphBundle.build()` 集中分布式图构建，替代 __main__ 散落赋值
- `EmbeddingDataset` 封装 parquet 加载 + mask 对齐验证 (上游无此步骤)
- `RfExperiment` 封装 RF 实验，加 `random_state` + dtype 检查 (上游全默认)
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  BitcoinMnmgArgs dump → create_uid uid类型 → CugraphWorkerSession RMM/cupy/comms初始化 →
  BitcoinGraphBundle edge_index/feature shape/barrier → build_encoder 参数量 →
  ix_train/ix_test 分配 shape/空检查 → train_epoch batch.x/edge_index/out shape →
  eval_epoch batch统计 → EmbeddingWriter 推理batch/layer状态/写回index/parquet路径 →
  BitcoinRfArgs dump → EmbeddingDataset parquet shape/对齐检查 →
  EmbeddingDataset.split X/y shape/class dist → RfExperiment fit/evaluate dtype/prob shape →
  gnn_only_evaluate z_test分布

## migrate 9b89e8a: [FEA] Set random state using PyTorch generator

- **Upstream commit**: 9b89e8a (cugraph-gnn, NVIDIA, 2025)
- **Commit message**: `[FEA] Set random state using PyTorch generator`
- **Upstream diff** (3 files changed: 1 new, 2 modified):
  - `loader/utils.py` (新增): `generate_seed()` 函数
    - rank=0: `torch.randint(0, 2**63 - world_size, (1,), dtype=int64, device="cuda")`
    - rank!=0: `torch.tensor([0], dtype=int64, device="cuda")`
    - `torch.distributed.broadcast(seed, src=0)` → 所有 rank 收到同一基础种子
    - `return seed.item() + rank` → 每 rank 获得唯一偏移种子
  - `loader/node_loader.py`: `sample_from_nodes(..., random_state=generate_seed())`
  - `loader/link_loader.py`: `sample_from_edges(..., random_state=generate_seed())`

- **设计意图**:
  rank 0 生成全局唯一种子后 broadcast; 各 rank 加上自身 rank offset 得到独立种子。
  种子空间上界 `2**63 - world_size` 确保 `seed + rank` 不溢出 int64。
  使用 CUDA tensor 保证 broadcast 走 GPU 直连通信而非 CPU 中转。

- **Knuth 审查**:
  1. diff 对比源: 上游新增 `utils.py` 后在两处 loader 各调用一次 `generate_seed()`,
     两处调用完全独立 — 同一 iteration 内 node_loader 和 link_loader 会产生不同种子,
     这是设计意图 (各自独立可复现), 但若两者需要严格同步则需调用方协调。
  2. 用户角度 bug: 上游 `generate_seed()` 直接调用 `torch.distributed.get_world_size()`,
     单机非 distributed 环境下此调用 raise RuntimeError; 用户在单卡调试时会莫名崩溃。
     另: `device="cuda"` 硬编码, CPU-only 环境 (CI/单测) 直接失败。
  3. 系统角度安全: `seed.item() + rank` 在极端情况 (rank=0 生成 `2**63 - world_size - 1`,
     world_size=1) 仍在 int64 范围内; 但若调用方将 seed 作为 int32 使用则截断。
     broadcast 依赖 distributed process group 已初始化, 若未 init 会 hang 而非 raise。

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/random_state.py`** — 新增

**迁移要点**:
- `RandomStateConfig`: 封装 generate_seed 的 rank/world_size/device/dtype 参数,
  显式暴露 `seed_upper_bound` 和 `is_distributed` 属性
  (Python 是内联字面量 `2**63 - world_size` + 隐式全局状态)
- `WalpurgisRandGen`: 封装 rank0 randint + nonrank zeros + broadcast 三步为类方法,
  加 non-distributed 降级 (Python 单机环境直接 crash); `_rank0_generate` /
  `_nonrank_placeholder` 对应 Python if/else 两分支
- `generate_seed_safe()`: 顶层函数对应 `generate_seed()`, 加 try/except 探测
  torch.distributed 是否可用, 降级为 rank=0 world_size=1 单机模式
- `SamplerRandomState`: 封装 `(loader_type, seed, rank, world_size)` 四元组,
  `.value` 属性返回裸 int 与 Python API 兼容; Python 是裸 int 直接传入
- `make_node_loader_random_state()` / `make_link_loader_random_state()`:
  对应 node_loader / link_loader 两处调用点, 显式记录 loader_type

**改写20%（鲁迅拿法）**:
- `RandomStateConfig` 封装隐式全局 distributed 状态为显式配置对象
- `WalpurgisRandGen` 提取三步裸逻辑为类, 加 non-distributed 降级 + 重试
- `SamplerRandomState` 将裸 int 包装为可 audit 对象, loader_type/rank/world_size 随 seed 一起存档
- `generate_seed_safe()` 修复上游单机环境 crash bug (Python 无此考虑)
- 全链路6个 `WALPURGIS_DEBUG=1` 断点 print, 覆盖:
  RandomStateConfig构造 → rank0_generate原始seed → broadcast前后对比 →
  final_seed+rank偏移 → SamplerRandomState构造 → make_*_random_state入口
## migrate 8bf2012: [FEA] Support Link Prediction and Negative Sampling in DGL

- **Upstream commit**: 8bf2012 (cugraph-gnn, NVIDIA)
- **Commit message**: `[FEA] Support Link Prediction and Negative Sampling in DGL`
- **Upstream diff** (4 files changed):
  - `graph.py` — 新增 `__edge_lookup_table`, `_clear_graph()`, `_to_numeric_etype()`,
    `_edge_lookup_table` property, `find_edges()`, `global_uniform_negative_sampling()`;
    删除遗留 `print(u,)` 调试语句
  - `tests/conftest.py` — 新增 `create_karate_bipartite()` + `karate_bipartite` fixture
  - `tests/test_graph.py` — 新增 `test_graph_find`, `test_graph_uniform_negative_sample`;
    重构 `test_graph_make_heterogeneous_graph` 使用 fixture
  - `tests/test_graph_mg.py` — 新增 MG 版 find/neg_sampling 测试; 补 `destroy_process_group()`

- **功能说明**:
  为 cuGraph-DGL Graph 类添加链路预测所需的两个核心方法:
  1. `find_edges(eid, etype)`: 给定边ID序列返回对应 (src, dst) 节点对; 内部依赖
     `pylibcugraph.EdgeIdLookupTable` 懒加载缓存; 结果减 vertex_offset 转为局部节点ID。
  2. `global_uniform_negative_sampling(num_samples, ...)`: 构造图中不存在的 (src, dst) 对;
     异构图用 bias 掩码确保 src 采样只命中 src 类型节点、dst 采样只命中 dst 类型节点;
     支持多GPU (all_reduce + 顶点切分); exclude_self_loops 过滤自环。
  `_clear_graph()` 统一清理3个缓存字段 (原代码4处散落2行赋值, 新增 edge_lookup_table 后防漏清)。

- **Knuth 审查**:
  1. diff对比源: `_clear_graph()` 替代4处散落的2行赋值; `_to_numeric_etype` 排序键必须与
     C层 EdgeIdLookupTable 索引约定严格一致 (排序不同则 lookup 返回错误节点对);
     `find_edges` 减 vertex_offset 是关键正确性要求 (全局ID → 局部ID)
  2. 用户角度bug: `[:num_samples]` 截断是 TODO workaround (rapidsai/cugraph#4672),
     多GPU时各 rank 负样本可能重叠降低多样性但不崩溃; `redundancy` 参数静默忽略无报错
  3. 系统角度安全: `_clear_graph()` 遗漏清理 `__edge_lookup_table` → C层悬空指针 CUDA fault;
     多GPU `array_split` 当 world_size > 顶点数时产生空 bias → negative_sampling 返回0样本

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/link_pred_neg_sampling.py`** — 新增

**迁移要点**:
- `GraphClearSession`: 封装 `_clear_graph()` 三字段清理为可审计会话;
  `validate_cleared()` 调试时验证无缓存泄漏 (Python 无此校验)
- `EdgeTypeIndex`: 封装 `_to_numeric_etype()` 为带缓存的值对象;
  `dump_table()` 使全部 etype→index 映射可见 (Python 内联 dict comprehension 无此能力)
- `EdgeLookupSession`: 封装 `_edge_lookup_table` property + `find_edges()` 逻辑;
  `_resolve_col_names()` 消除 "sources"/"destinations" 魔法字符串内联
- `NegBiasPlan` + `_BiasBuilder`: 对应 `global_uniform_negative_sampling()` 掩码构建树;
  三条路径提取为 `_homo()` / `_hetero_same_type()` / `_hetero_diff_type()` 静态方法
- `WalpurgisNegSamplingSession`: build_bias_plan() + execute() 两阶段分离;
  多GPU切分逻辑提取为 `apply_multi_gpu_split()`; 空 bias 空数组警告
- `KarateGraphBipartiteFactory`: 对应 conftest.py `create_karate_bipartite()`;
  `_partition_edges()` + `_offset_edges()` 分离分区和偏移转换关注点

**改写20%（鲁迅拿法）**:
- `GraphClearSession.validate_cleared()`: 新增 use-after-free 防御性验证 (Python 无)
- `EdgeTypeIndex.dump_table()`: 新增映射表 dump (类比 fp16_grad_dedup dump_dispatch_table)
- `EdgeLookupSession._resolve_col_names()`: 提取 src/dst 列名魔法字符串为方法
- `NegBiasPlan.validate()`: 新增 bias/vertices 长度一致性检查 + WARN 打印
- `WalpurgisNegSamplingSession.apply_multi_gpu_split()`: 提取多GPU切分 + 空数组警告
- 全链路7处断点 print (WALPURGIS_DEBUG=1 门控):
  EdgeTypeIndex映射表 → EdgeLookupSession.find_edges入口+列名选择 →
  _BiasBuilder路径选择+concat大小 → WalpurgisNegSamplingSession截断量+self-loop过滤数量

## migrate c3799ae: [BUG] Use memory pool in movielens example

- **Upstream commit**: c3799ae (cugraph-gnn, NVIDIA, 2025-10-31)
- **Commit message**: `[BUG] Use memory pool in movielens example`
- **Upstream diff** (1 file changed, 81 insertions, 80 deletions):
  - `movielens_mnmg.py` — `__main__` 块:
    删除旧代码: `if global_rank == 0: torch.cuda.change_current_allocator(rmm_torch_allocator)`
    新增: `from rmm.allocators.torch import rmm_torch_allocator`
    新增: `with torch.cuda.use_mem_pool(torch.cuda.MemPool(rmm_torch_allocator.allocator())):`
    主逻辑全部缩进一层, 逻辑无变化, 纯作用域包裹

- **Bug 根因**:
  旧代码 `change_current_allocator()` 修改进程级全局状态, 且仅在 `global_rank == 0`
  的 if 分支内调用。多 GPU 训练时 rank 1..N 继续使用 PyTorch 默认 CUDA caching
  allocator。DDP all-reduce / NCCL 通信时两侧 buffer 来自不同内存池 → 显存碎片 +
  NCCL 挂起 + OOM, 错误信息不指向 allocator 根因, 极难定位。
  修复: `use_mem_pool` context manager 是进程局部的, 所有 rank 均建立独立 MemPool,
  退出 with 块后自动恢复默认 allocator, 无资源泄漏。

- **Knuth 审查**:
  1. diff对比源: change_current_allocator 全局不可撤销 vs use_mem_pool 局部可回退;
     旧代码只 rank 0 执行 vs 新代码全 rank 均执行; 作用域明确性质变
  2. 用户角度 bug: 2 卡以上训练中途 OOM 或 NCCL 挂起, 错误信息不指向 allocator,
     用户误以为显存不足或网络问题, 实际是 allocator 不一致
  3. 系统角度安全: change_current_allocator 一旦设置不可回退, 第三方库再次调用将静默
     覆盖 → 不可预期; use_mem_pool 作用域明确, 退出自动恢复, 系统状态可控

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/memory_pool.py`** — 新增

**迁移要点**:
- `RmmAllocatorMode`: 枚举, GLOBAL_CHANGE (旧 BUG 方式) vs MEM_POOL (c3799ae 修复方式),
  build_mem_pool_session() 见到 GLOBAL_CHANGE 直接 raise ValueError
- `RmmMemPoolContext`: 值对象, 携带 (rank, device, allocator_fn, pool),
  替代 Python 中 rmm_torch_allocator 直接内联使用, 支持懒初始化和调试
- `WalpurgisMemPoolSession`: context manager 类, 封装 torch.cuda.use_mem_pool 生命周期,
  __enter__ 激活 pool + 打印调试, __exit__ 退出 + 显存摘要; RMM 不可用时优雅降级 noop
- `validate_mem_pool_consistency()`: 验证所有 rank 均已激活 pool session
  (Python c3799ae 无此校验; 对应旧 BUG: 只有 rank 0 调用 allocator)
- `build_mem_pool_session()`: 顶层工厂函数, 对应 c3799ae 修复后的两行代码

**改写20%（鲁迅拿法）**:
- `RmmAllocatorMode` 枚举明示两种模式语义差异, 防止日后误用旧 BUG 方式
- `RmmMemPoolContext` 值对象替代 Python 内联 MemPool 构造, 携带 rank/device 元数据
- `WalpurgisMemPoolSession` 提取裸 with 语句为可测试/可日志的 session 类
- `validate_mem_pool_consistency()` 新增验证方法 (Python 无此逻辑)
- 全链路5个 `WALPURGIS_DEBUG=1` 断点 print, 覆盖:
  build_mem_pool_session入口 → RmmMemPoolContext构造 → build_pool allocator地址 →
  pool创建 → session.__enter__激活 → session.__exit__显存摘要 →
  validate一致性检查
## migrate 81b7074: [FEA] Update MAG example to show fp16/bf16 support

- **Upstream commit**: 81b7074 (cugraph-gnn, NVIDIA)
- **Commit message**: `[FEA] Update MAG example to show fp16/bf16 support`
- **Upstream diff** (1 file modified):
  - `python/cugraph-pyg/cugraph_pyg/examples/mag_lp_mnmg.py` — 745行
    - 新增 `_DTYPE_CHOICES` 元组 + `parse_dtype()` 将字符串映射为 `torch.dtype`
    - `Classifier.__init__` 新增 `dtype` 参数，`self.dtype` 保存；
      `wgth.create_embedding` 第4参数从硬编码 `torch.float32` 改为传入 `dtype`
    - `Classifier.forward`：`w_dtype = self.paper_lin.weight.dtype`，
      `batch["paper"].x.to(w_dtype)` 保证输入与权重 dtype 匹配；
      三处 `torch.zeros` 新增 `device=x_paper.device, dtype=x_paper.dtype`，
      消除硬编码 `device="cuda"`
    - `feature_store["paper","x",None]` 写入前 `.to(dtype)`
    - betweenness centrality 写 feature store 时 `.to(dtype)`（原 `.to(float32)`）
    - 边特征更新：变量名 `stype/dtype` → `src_type/dst_type`（原变量名遮蔽 `dtype` 参数 bug）；
      `.to(dtype)` 在 `.reshape()` 前执行
    - `--dtype` argparse 参数，default=`bfloat16`，choices=`_DTYPE_CHOICES`
    - `model.to(device, dtype)` 双参数
    - embedding inference loop 中 `feature_store["paper","x1",None]` 新增 `dtype=dtype`；
      三处 `torch.zeros` 新增 `dtype=dtype`；`plin = model.module.paper_lin` 提取变量
    - 输出 parquet 前 `.to(torch.float32)` 保证 cudf 兼容性

- **功能说明**:
  通过 `--dtype float16/bfloat16/float32` 控制模型权重与特征张量的精度，
  支持低精度训练以节省显存/提升吞吐，同时保证 cudf 输出前强转 float32 维持兼容性。
  关键修复：变量名 `dtype` 遮蔽同名参数 bug（`stype, _, dtype = etype.edge_type`
  导致后续 dtype 被覆盖为字符串）在本 commit 一并修复。

### Walpurgis 迁移位置

**新增文件:**
- `src/walpurgis/examples/mag/mag_lp_mnmg.py`

**迁移要点**:
- `DTypeRegistry`（parse_dtype 强化）: KeyError → 友好 ValueError + 候选列表
- `NodeZeroInitializer`: 封装 `torch.zeros(n, hidden_channels, device=ref.device, dtype=ref.dtype)`，
  消除 `forward()` 和 embedding inference loop 中共 6 处重复
- `_dbg(tag, msg)`: 统一调试出口，`WALPURGIS_DEBUG=1` 时才打印，零侵入生产路径
- `_register_wholegraph_embeddings()`: 提取为独立方法，含 `sorted()` 确定性保证 + `_dbg`
- 变量名 `stype/dtype` → `src_type/dst_type`（沿用上游 81b7074 修复，保持语义清晰）

**改写20%（鲁迅拿法）**:
- `parse_dtype()` KeyError → ValueError，附候选列表（上游无提示）
- `NodeZeroInitializer` 对象替代 6 处 `device=..., dtype=...` 内联重复
- `_dbg()` 全链路断点 print，16个覆盖点:
  parse_dtype → Classifier.__init__ → _register_wholegraph_embeddings →
  Classifier.forward（w_dtype / x_paper.shape / zeros shape） →
  init_pytorch_worker → main args dump / dtype resolved / global_rank info /
  node_counts / paper feature dtype / bc shape→dtype / edge_attr etype /
  model constructed / train edges shape / ix_start-end / local_x0 shape /
  ex_loader plin dtype / concat→float32 for cudf / parquet written
- `_register_wholegraph_embeddings()` 提取独立方法，可独立测试
- `NodeZeroInitializer.make()` 在 embedding inference loop 复用，消除重复

**Knuth审查三问**:
1. diff对比源: dtype 传播路径全覆盖（feature_store写入/zeros初始化/model.to/cupy输出）；
   变量名遮蔽 bug 已修复（src_type/dst_type）；cudf float32 强转保留
2. 用户角度bug: `parse_dtype("floatXX")` 原抛 `KeyError: 'floatXX'`，
   现抛 `ValueError: 不认识的 dtype 'floatXX'，可用选项: [...]`
3. 系统角度安全: cupy/cudf 不支持 bfloat16，输出前强转 float32 是必要安全门；
   `WALPURGIS_DEBUG=1` 断点 print 不影响生产路径；`sorted()` 保证跨 rank 嵌入注册顺序

## migrate 05fe6f4: [FEA] Knowledge Graph/Graph Database Renumbering

- **Upstream commit**: 05fe6f4 (cugraph-gnn, NVIDIA)
- **Commit message**: `[FEA] Knowledge Graph/Graph Database Renumbering`
- **Upstream diff** (2 files added):
  - `renumber_kg.py` — 295行: 分布式多GPU KG节点/边重编号脚本
    - parse_args(): 14个参数，含 node_types/edge_types/folder 路径/格式/managed_memory
    - torchrun多进程: nccl init → RMM allocator切换 → cudf延迟import
    - 节点阶段: per node_type, all_gather_into_tensor收集各rank节点数 →
      cumsum计算global offset → all_gather汇总renumber map → 写出local map
    - 边阶段: per edge_type, .loc[]查表映射src/dst原始id → 新全局id → 写出
  - `run_renumber.sh` — 32行: torchrun启动脚本，硬编码路径示例

- **功能说明**:
  分布式多进程(torchrun)场景下将KG原始节点id(任意整数)重编号为连续全局id
  (0..total_nodes-1)。每个rank处理自己分片的节点/边文件，通过all_gather
  在所有rank间共享完整的id映射表，使得边重编号可在本地完成而无需额外通信。

### Walpurgis 迁移位置

**新增文件:**
- `src/walpurgis/examples/kg/renumber_kg.py` — 主迁移文件
- `src/walpurgis/examples/kg/run_renumber.sh` — 启动脚本

**迁移要点**:
- `KGRenumberArgs`: 封装 argparse.Namespace 为强类型 dataclass，
  validate() 做前置参数一致性校验（上游无，argparse 只检查 required）
- `NodeRenumberSession`: 合并上游 4 个平行 dict
  (local_num_nodes / global_num_nodes / local_node_offsets / global_renumber_map)
  为单一对象，字段直接命名
- `RenumberMapStore`: 封装 global_renumber_map dict 的写入/查找，
  get_strict() 在 edge 阶段找不到 node_type 时给出明确错误信息
- `EdgeRenumberSession.apply()`: 封装边重编号执行，修复上游 os.listdir() 无排序 bug

**改写20%（鲁迅拿法）**:
- `KGRenumberArgs.validate()` 提前做语义校验（上游无），参数长度不一致早报错而非在 zip() 中静默截断
- `NodeRenumberSession` 对象替代 4 个平行 dict，消除 dict[node_type] 四次冗余访问
- `RenumberMapStore.get_strict()` 替代裸 dict[key]，KeyError 时告知哪个 node_type 缺失及已有列表
- `EdgeRenumberSession.apply()` 封装边阶段为可独立测试方法，sorted() 修复无序 bug
- `_parse_args()` 在 parse 后立即构造 KGRenumberArgs 并 validate()，上游 args 裸 Namespace 散落访问
- `run_renumber.sh` 改写为 DATA_ROOT 环境变量驱动，`set -euo pipefail` 防静默失败
- 全链路 WALPURGIS_DEBUG=1 断点 print，8个 _dbg() 覆盖:
  args.validate 参数 dump → node 文件路径 → local_num_nodes →
  all_gather前后 → cumsum offset → renumber_store.put/get →
  edge 文件路径(bug-fix 标注) → src/dst 映射 → 写出路径确认
## migrate fbea7cb: Fix append unique

- **Upstream commit**: fbea7cb (cugraph-gnn, linhu-nv, 2026-04-01, PR #423)
- **Commit message**: `Fix append unique`
- **Upstream diff** (2 files changed, 36 insertions, 75 deletions):
  - `wholememory_binding.pyx` — 6个 `python_cb_wrapper_*` cdef 函数:
    废弃手工 `PyTuple_New / Py_INCREF / PyTuple_SetItem / PyObject_CallObject / Py_DECREF`，
    改用 Cython 原生 `<object>` 转型后直接调用 Python 函数；
    `temp_malloc / output_malloc` 不再传 `PyWholeMemoryTensorDescription` / `PyMemoryAllocType` 对象，
    改传 `(py_shape: tuple, py_dtype: int, py_malloc_type_int: int)`。
  - `wholegraph_env.py` — `torch_malloc_env_fn` 签名同步更新:
    旧 `(tensor_desc: PyWholeMemoryTensorDescription, malloc_type: PyMemoryAllocType, ...)`
    → 新 `(shape: tuple, dtype_int: int, malloc_type_int: int, ...)`;
    内部 `malloc_type.get_type()` 改为 `int(wmb.WholeMemoryMemoryAllocType.Mat*)` 比较;
    `tensor_desc.dtype` 改为 `wmb.WholeMemoryDataType(dtype_int)` 重建枚举。

- **Bug 根因**:
  标题 "append unique" 指向 `PyTuple_SetItem` 的 steal-reference 语义问题。
  CPython 文档: `PyTuple_SetItem` 对传入 item "steals" 引用 (不额外 incref)。
  旧代码在 `Py_INCREF(item)` 后调用 `SetItem` → tuple 持有那份引用；
  `Py_DECREF(args)` 时 tuple 析构减1，但外部的 `+1 INCREF` 永不归还 → **引用计数泄漏**。
  `output_malloc` 中更有 `SetItem(0, <object><PyObject*>py_tensor_desc)` 多余往返转换，
  与 `temp_malloc` 版本行为不对称，存在 **潜在 double-free 风险**。
  新代码全部交由 Cython 编译器管理引用计数，根除上述问题。

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/wholememory_cb.py`** — 新建

**迁移要点**:
- `WholememoryCallbackSpec`: 值对象，描述6个回调函数的签名契约
  (name, arg_names, doc)，替代上游靠命名约定隐式维护一致性的模式。
- `WholememoryTensorParams`: 封装 fbea7cb 后三散参 `(shape, dtype_int, malloc_type_int)`，
  提供 `alloc_decision() → (device, pinned)` 和 `to_torch_dtype()` 便利方法。
- `WholememoryAllocMode(IntEnum)`: 对应 `WholeMemoryMemoryAllocType` 枚举值，
  供 `alloc_decision()` 映射表使用，替代上游内联三路 if/elif/assert。
- `WholememoryCallbackBridge`: 静态类，封装 `create_context / destroy_context /
  malloc / free / output_malloc / output_free` 6个方法，
  对应 fbea7cb 后 `wholegraph_env.py` 的 `torch_*_env_fn` 函数族。
- `CALLBACK_SPECS`: dict，6个 `WholememoryCallbackSpec` 实例，全局可查阅。
- `test_wholememory_cb_migration()`: 自检函数，5项检验。

**改写20%（鲁迅拿法）**:
- `WholememoryCallbackSpec` 值对象: 上游6个函数签名散落两文件，靠命名约定维护；
  我们用冻结 dataclass 显式化契约，`validate_call_args()` 运行时校验参数数量。
- `WholememoryTensorParams.alloc_decision()` 映射表替代三路 if/elif/assert:
  上游 `wholegraph_env.py` 是 `if malloc_type_int == int(MatDevice)... elif... else assert`;
  我们改写为 `_ALLOC_MAP: dict[int, (device, pinned)]`，O(1) 查找 + 越界 ValueError。
- `WholememoryCallbackBridge` 静态类封装: 上游6个顶层函数直接散在模块里；
  我们聚合为一个类的静态方法，`cb_label="temp"/"output"` 参数统一 DEBUG 标签区分。
- `output_malloc / output_free` 显式别名方法: 上游 temp/output 两路通过
  `create_context()` 分别注册同一 Python 函数，调用侧无法区分路径；
  我们用命名别名使 output 路径在 DEBUG print 中有独立标签，便于追踪。
- 全链路8个 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  create_context 入口/返回 → destroy_context → malloc 入参/device决策/data_ptr →
  free 前 tensor.shape → output_malloc/output_free 同上 (前缀 [OUTPUT])。

### 质量审查 (Knuth 标准)

**1. diff 对比源**

| 上游 05fe6f4 | Walpurgis 迁移 |
|---|---|
| `args = parse_args()` → argparse Namespace | `cfg = _parse_args()` → `KGRenumberArgs` dataclass |
| `local_num_nodes = {}` / `global_num_nodes = {}` / `local_node_offsets = {}` / `global_renumber_map = {}` (4个平行dict) | `NodeRenumberSession` 单一对象封装4个字段 |
| `global_renumber_map[node_type] = cudf.DataFrame(...)` / `global_renumber_map[src_type]["id"]` | `RenumberMapStore.put()` / `.get_strict()` |
| `edge_fname = os.listdir(edge_folder_name)[local_rank]` **无排序** | `edge_files = sorted(os.listdir(...))[local_rank]` **修复** |
| `tuple(edge_type.split(","))` 在 main 循环内 inline 解析 | `_parse_args()` 中提前解析为 `List[Tuple[str,str,str]]` |
| 无参数校验（依赖 argparse required=True）| `KGRenumberArgs.validate()` 检查 types/folders/format 一致性 |
| 无任何中间过程 print（只有末尾 `print("Success!")`）| 8个 `_dbg()` 断点，WALPURGIS_DEBUG=1 开启 |
| `run_renumber.sh` 硬编码 `/home/nfs/abarghi/...` | `DATA_ROOT` 环境变量 + `set -euo pipefail` |

**2. 用户角度 bug 排查**

- **边文件排序 bug (05fe6f4 原始)**:
  节点阶段: `sorted(os.listdir(node_folder_name))[local_rank]` — 有排序，各 rank 得到确定性分配
  边阶段:   `os.listdir(edge_folder_name)[local_rank]` — **无排序**，`os.listdir()` 返回顺序
  取决于底层文件系统 (ext4 inode 顺序、tmpfs 插入顺序、NFS 远端实现各不同)。
  单机本地测试通常能通过 (ext4 目录项顺序稳定)，但跨节点 NFS 挂载或不同 OS
  版本下可能导致两 rank 处理同一 edge 文件（重复写出）或遗漏某 edge 文件（数据丢失）。
  **Walpurgis 修复**: `sorted(os.listdir(edge_folder_name))[local_rank]`，与节点阶段对齐。

- **参数 zip 静默截断 (05fe6f4 原始)**:
  若 `--node_types a,b` 但 `--node_input_folders` 只给1个路径，
  `zip(node_types, node_input_folders, ...)` 静默截断为短者，b 类型被跳过而无报错。
  **Walpurgis 修复**: `KGRenumberArgs.validate()` 前置校验长度一致，运行前即报错。

- **output_format 路径拼写 (05fe6f4 原始)**:
  节点阶段 `to_csv()` 输出文件名为 `{node_fname}_renumbered.csv`，
  边阶段 `to_parquet()` 输出文件名为 `{edge_fname}_renumbered.parquet`。
  若 `node_fname` 本身已含扩展名 (如 `paper_0.csv`)，输出为 `paper_0.csv_renumbered.csv`。
  上游未处理此命名问题，Walpurgis 迁移保持与上游行为一致 (不修改命名逻辑)，
  通过断点 print 输出最终写出路径，让用户自行确认是否符合预期。

- **rank0 allocator 切换 barrier 竞争**:
  上游: `if global_rank == 0: change_current_allocator(rmm_torch_allocator)` →
  `torch.distributed.barrier()` → 所有 rank 设置 cupy allocator。
  rank0 切换 torch allocator 到 rmm 后 barrier，其余 rank 在 barrier 前用默认 allocator。
  若 barrier 前非 rank0 的 rank 有 torch CUDA alloc (不太可能在此时机)，
  可能产生 allocator 不一致。上游注释未说明此 barrier 的意图，
  Walpurgis 迁移保持原有顺序，并在 _dbg 中标注 "rank0: rmm_torch_allocator 已切换"。

**3. 系统角度内存并发安全**

- **`RenumberMapStore._store` (dict)**: 节点重编号阶段单线程顺序写入，
  边阶段单线程顺序读取，无并发访问，dict 无需加锁。
  torchrun 每个进程独立 Python 解释器，进程间隔离，dict 不跨进程共享。

- **`map_tensor` list of Tensors**: 在 `all_gather` 中作为输出 buffer 列表传入。
  PyTorch DDP all_gather 文档说明: output 列表中各 tensor 由通信库填写，
  单进程内 all_gather 是同步操作，返回后数据已就绪。concat 后原 list 可 GC。
  Walpurgis 迁移使用 `map_tensor_concat` 新变量保存 concat 结果，
  明确与 all_gather buffer 区分，防止意外引用 stale tensor。

- **cudf.DataFrame index=cupy_array 并发读取**:
  `renumber_store` 中的 cudf.DataFrame 在节点阶段构造后，边阶段只读 (`.loc[]`)。
  单进程内无写入者，读取安全。每个 torchrun 进程维护自己独立的 store，
  `global_renumber_map` 在所有进程中是完全一致的副本 (all_gather 保证)。

- **`sorted(os.listdir())` 的跨进程一致性**:
  `sorted()` 保证在任意 OS / 文件系统上，相同目录内容产生相同顺序。
  若两 rank 运行在不同节点挂载同一 NFS，`os.listdir()` 结果可能有差异
  (取决于 NFS server 缓存)，`sorted()` 消除此不确定性。
  **前提**: 所有 rank 的 `edge_folder_name` 包含相同文件集。若文件数 < world_size，
  `sorted(...)[local_rank]` 会 IndexError，应由调用方保证每 folder 文件数 ≥ world_size。

- **RMM pool allocator 与 cupy allocator 的 race**:
  `rmm.reinitialize()` 后才 import cudf (上游注释明确: "import cudf after rmm
  has been reinitialized")。RMM pool 初始化在 barrier 后，所有 rank 同步完成。
  cupy.cuda.set_allocator(rmm_cupy_allocator) 是进程内全局状态，torchrun 每进程
  独立，无跨进程 race。
| 上游 fbea7cb | Walpurgis 迁移 |
|---|---|
| `fn = <object> wrapped_global_context.temp_create_context_fn; py_memory_context = fn(ctx)` | `WholememoryCallbackBridge.create_context(global_context)` |
| `fn(mem_ctx, ctx)` (destroy/free) | `destroy_context / free` 静态方法 + DEBUG print |
| `py_shape = tuple([...]); py_dtype = int(...); py_malloc_type_int = int(...); res_ptr = fn(...)` | `WholememoryTensorParams.from_callback_args()` + `alloc_decision()` + `to_torch_dtype()` |
| `if malloc_type_int == int(MatDevice): ... elif ... else: assert MatPinned` | `_ALLOC_MAP: dict[int,(device,pinned)]` + 越界 ValueError |
| output_malloc 与 temp_malloc 共用同实现，调用侧靠注册顺序区分 | `output_malloc(cb_label="output")` 显式别名 |
| 签名一致性靠命名约定维护 | `WholememoryCallbackSpec.validate_call_args()` 运行时校验 |

**2. 用户角度 bug 排查**

- **fbea7cb 修复后新风险**: `temp_malloc` 回调现在传 `(shape, dtype_int, malloc_type_int)` 三个
  plain int/tuple，若调用方误传顺序 (如把 `dtype_int` 传到 `malloc_type_int` 位置)，
  Cython 侧无类型检查，Python 侧会静默接受整数比较失败 → `alloc_decision()` 越界。
  缓解: `WholememoryTensorParams.alloc_decision()` 越界时抛出带诊断信息的 `ValueError`，
  比上游 `assert` 更友好（assert 在优化模式下被禁用）。
- **`to_torch_dtype()` 枚举重建风险**: `wmb.WholeMemoryDataType(dtype_int)` 若 `dtype_int`
  是非法值，会抛出 `ValueError`；上游直接内联，同样会抛。我们的封装不增加新风险，
  DEBUG print 会在抛出前打印 `dtype_int` 值，便于定位。
- **output_malloc 旧代码 double-free 风险 (已修复)**: 旧代码 `SetItem(0, <object><PyObject*>py_tensor_desc)`
  对已是 Python 对象的 `py_tensor_desc` 做了多余 `void*` 往返；
  `Py_INCREF` 后 steal 再加外部 `py_tensor_desc` 局部变量的引用，
  对象在 `Py_DECREF(args)` 后是否立即析构取决于外部变量生命周期，
  存在提前析构风险。fbea7cb 已修复，Walpurgis 迁移版本无此路径。

**3. 系统角度内存并发安全**

- `WholememoryCallbackSpec` / `CALLBACK_SPECS`: 模块级常量，`frozen=True` dataclass，
  并发只读完全安全，无可变共享状态。
- `WholememoryTensorParams`: 每次 malloc 回调构造一个新实例，无跨调用共享，
  线程安全语义同上游散参传递（每次调用独立栈帧）。
- `WholememoryCallbackBridge` 静态方法: 无实例状态，所有状态在参数中传递，
  并发调用安全（与上游顶层函数等价）。
- `_ALLOC_MAP` (在 `alloc_decision` 内定义): 每次调用重建 dict，无跨调用共享；
  可提升为模块级常量以避免重复构造（性能优化，当前不是热路径）。
- `WholememoryCallbackBridge.malloc` DEBUG 路径: `memory_context.get_tensor()` 在
  `free` 之后调用可能返回 None；我们用 `try/except` 保护，不引入新 crash 风险。
- `_StubMemoryContext`: 仅在 torch 不可用时使用（测试路径），无并发使用场景。

## migrate 8b3b67f: [BUG] Mask out unwanted vertices during negative sampling

- **Upstream commit**: 8b3b67f (cugraph-gnn, NVIDIA, 2025)
- **Commit message**: `[BUG] Mask out unwanted vertices during negative sampling`
- **Upstream diff** (3 files changed, 110 insertions, 9 deletions):
  - `sampler_utils.py` — `neg_sample()`:
    新增 `input_type: Tuple[str,str,str]` 参数;
    删除 `unweighted` 局部变量;
    按 `is_homogeneous + input_type` 从 `graph_store` 取 `num_src/dst_nodes`;
    None 权重 → 全1向量; dtype 一致性检查;
    异构图 type_mismatch 时: `vertices=concat(arange(src)+off, arange(dst)+off)`,
    `src_weight=concat([sw, zeros(dst)])`, `dst_weight=concat([zeros(src), dw])`
    → **掩码核心**: 每个顶点在"错误角色"中权重=0, 永不被采样
  - `sampler.py` — `BaseSampler._sample_negative()`:
    新增 `index.input_type` 传参;
    triplet 分支 BUG 修复: 旧代码 `neg_cat(src.cuda(), dst_neg, ...)` 将
    dst类型节点（如paper）混入src（如author）→ 类型污染;
    新代码: `per=randint(0,scu.numel(),(dst_neg.numel(),)); neg_cat(scu, scu[per], ...)`
    → 从 src 自身随机子集补位, 类型纯净
  - `test_neighbor_loader.py`:
    新增 `test_link_neighbor_loader_hetero_negative_sampling`:
    author-writes-paper 异构图, binary/triplet × amount=1/2 × batch_size=1/2;
    验证 edge_label_index src ∈ author.n_id, dst ∈ paper.n_id

- **Bug 根因**:
  旧代码 `neg_sample()` 中 `vertices = cupy.arange(src_weight.numel())` 生成
  从0开始的本地ID序列, 完全忽略异构图中每种节点类型的全局偏移 `_vertex_offsets`。
  后果: paper节点（全局ID从4开始）被以 [0,5] 范围采样, 实际命中的是 author 节点的ID空间,
  pylibcugraph 收到的候选集包含错误类型的全局ID → 负样本被选到不存在的/错误类型的节点上。
  同理 triplet 分支直接 `cat(src, dst_neg)` 将不同类型张量拼接, embedding lookup 时
  用 paper ID 索引 author embedding table → 越界或静默语义错误。

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/neg_sampler.py`** — 新增

**迁移要点**:
- `NegSamplingWeights`: 值对象, 携带 (src_weight, dst_weight, vertices, dtype, offset_applied, type_mismatch),
  替代 Python neg_sample() 中三变量 interleaved 原地修改模式
- `NegSamplingVertexMask`: 静态类, 4个命名方法 + `build()` 分发,
  封装 8b3b67f 的完整 vertices/bias 构建逻辑 (Python 是内联 if/elif/else 树)
- `WalpurgisNegSampleConfig`: 配置对象, 携带 `is_hetero / type_mismatch / is_binary / is_triplet`
  派生属性 + `compute_num_neg()` 方法
- `TripletSrcRepair`: 封装 sampler.py triplet src 修复逻辑 + `validate_src_purity()` 断言
- `neg_sample_walpurgis()`: 顶层入口函数, 无 torch/cupy 依赖 (纯配置+掩码层)
- `test_hetero_negative_sampling_vertex_purity()`: 自检函数, 对应 8b3b67f 测试的核心断言

**改写20%（鲁迅拿法）**:
- `NegSamplingWeights` 值对象替代 Python 三变量分散赋值模式
- `NegSamplingVertexMask._build_hetero_type_mismatch / _build_hetero_same_type /
  _build_homo_unweighted / _build_homo_weighted` 4个命名方法
  替代 Python 单个 neg_sample() 函数内的 if/elif/else 内联树
- `TripletSrcRepair.validate_src_purity()` 新增验证方法 (Python 测试侧 assert,
  我们提取为运行时可选断言)
- `WalpurgisNegSampleConfig.type_mismatch / is_hetero` 派生属性明示分支决策
  (Python 是内联 `if input_type[0] != input_type[2]`, 每次重新比较)
- 全链路11个 `WALPURGIS_DEBUG=1` 断点 print, 覆盖:
  build()入口 → dtype决策 → hetero/homo分支 → vertices构建 → zero-pad宽度 →
  TripletSrcRepair per索引分布 → 最终 vertices/weight 摘要

## migrate 7ea1138: [BUG] Fix Weights Issue in Negative Sampling

- **Upstream commit**: 7ea1138 (cugraph-gnn, alexbarghi-nv, 2026-04-08, PR #447)
- **Commit message**: `[BUG] Fix Weights Issue in Negative Sampling`
- **Upstream diff** (1 file changed, 13 insertions, 8 deletions):
  - `python/cugraph-pyg/cugraph_pyg/sampler/sampler_utils.py` — `neg_sample()`:
    将 `src_weight / dst_weight` 的补零 concat 从 `if not is_homogeneous:` 块外
    移入 `if input_type[0] != input_type[2]:` 子块内；
    删除 `elif src_weight is None and dst_weight is None: vertices = None` 死代码分支。

- **Bug 根因**:
  `neg_sample()` 在异构图场景下，对所有异构路径（包括 src==dst 同类型节点）
  无条件执行 weight 的补零 concat：
  ```python
  src_weight = torch.concat([src_weight, torch.zeros(num_dst_nodes, ...)])
  dst_weight = torch.concat([torch.zeros(num_src_nodes, ...), dst_weight])
  ```
  而这两行位于 `if input_type[0] != input_type[2]:` 块之外，
  导致 src==dst 类型（如 author→author）的场景下 weight 被错误扩展：
  - 修复前: `src_weight.shape = [num_src + num_dst]`（多了 num_dst 个无意义的零）
  - 修复后: `src_weight.shape = [num_src]`（正确，与 vertices 对应）
  pylibcugraph 负采样引擎按 vertices 偏移索引 weight，weight 长度不匹配时
  越界访问，采出不存在的节点 ID，结果静默错误（无 exception，但负样本无效）。
  删除的 `elif src_weight is None and dst_weight is None: vertices = None` 分支
  是死代码——src_weight/dst_weight 在此之前已被 `ones()` 填充，永远不为 None。

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/neg_sampler_weights.py`** — 新增

**迁移要点**:
- `NegSampleWeightPlan`: 值对象，携带 (vertices, src_weight, dst_weight,
  src_dst_same_type, is_homogeneous) 五元状态，`validate()` 检查 shape 一致性
  并精确诊断 7ea1138 bug（若 src_weight.shape = num_src+num_dst 则打印 BUG 提示）
- `WeightAligner._pad_src_for_dst` / `_pad_dst_for_src`: 对称静态方法，
  封装 7ea1138 修复后仅在 src!=dst 分支执行的补零 concat 操作
- `WeightAligner._is_dead_branch()`: 文档化 7ea1138 删除的 `elif` 死代码路径，
  防御性检查（若意外到达则打印 ERROR）
- `NegSampleWeightBuilder.build()`: 顶层决策入口，三分支:
  `_build_hetero_src_ne_dst` / `_build_hetero_src_eq_dst` / `_build_homo`
- `prepare_neg_sample_weights()`: 便利函数，对应 neg_sample() 调用点

**改写20%（鲁迅拿法）**:
- `NegSampleWeightPlan` 值对象替代 Python 中 vertices/src_weight/dst_weight 三个散落局部变量
- `WeightAligner._pad_src_for_dst` / `_pad_dst_for_src` 命名对称方法替代 Python inline concat
- `_build_hetero_src_ne_dst` / `_build_hetero_src_eq_dst` / `_build_homo` 三个分支各自命名
  （Python 是匿名 if/else）
- `NegSampleWeightPlan.validate()` 独立可测的 shape 检查 + 7ea1138 bug 精确诊断
  （Python 无此检查，越界静默失败）
- 全链路5个 `WALPURGIS_DEBUG=1` 断点 print:
  1. `prepare_neg_sample_weights` 入口: is_homo + input_type + num_src/dst
  2. `NegSampleWeightBuilder.build`: 分支选择
  3. `_pad_src_for_dst` / `_pad_dst_for_src`: concat 前后 shape（仅 src!=dst 分支）
  4. `_build_hetero_src_eq_dst`: "weight UNCHANGED ← correct post-7ea1138" 标记
  5. `NegSampleWeightPlan.validate()`: shape 一致性检查结果

### 质量审查（Knuth 标准）

**1. diff 对比源**

| 上游 8b3b67f | Walpurgis 迁移 |
|---|---|
| `neg_sample(graph_store, seed_src, seed_dst, input_type, ...)` 新增 `input_type` | `WalpurgisNegSampleConfig(input_type=...)` + `neg_sample_walpurgis(config, ...)` |
| `num_src_nodes = graph_store._num_vertices()[input_type[0]]` | `config.num_src_nodes` (调用方传入, 无 graph_store 依赖) |
| `src_weight = torch.ones(num_src_nodes, ...)` (None→全1) | `NegSamplingVertexMask.build()` 内 None 填充逻辑 |
| `dtype 一致性 raise ValueError` | `neg_sample_walpurgis()` 内相同检查 |
| `vertices = concat(arange(src)+off_src, arange(dst)+off_dst)` | `_build_hetero_type_mismatch()` 对应实现 |
| `src_weight = concat([sw, zeros(dst)])` (掩码) | `_build_hetero_type_mismatch()` `src_weight_ext` |
| `dst_weight = concat([zeros(src), dw])` (掩码) | `_build_hetero_type_mismatch()` `dst_weight_ext` |
| triplet: `per = randint(0, scu.numel(), (dst_neg.numel(),))`; `neg_cat(scu, scu[per], ...)` | `TripletSrcRepair.repair(src_ids, dst_neg_count)` |
| test: `assert isin(src_nodes, arange(len(author_n_ids)))` | `TripletSrcRepair.validate_src_purity()` + `test_hetero_negative_sampling_vertex_purity()` |
| `vertices=None if vertices is None else cupy.asarray(vertices)` | `weights.vertices` (None or list, 调用方转 cupy) |

**2. 用户角度 bug 排查**

- **Bug 1 (错误节点类型入负样本)**: 旧代码 `vertices=cupy.arange(src_weight.numel())` 在
  异构图中生成 [0, num_src) 范围, 完全忽略 `_vertex_offsets`。例如 author=4节点、paper=6节点,
  paper的全局ID应为 [4,9], 旧代码却从 [0,5] 采样 → author ID 混入 dst 负样本。
  `WALPURGIS_DEBUG=1` 打印 `off_src/off_dst` + `vertices concat范围`, 用户可立即看到是否正确偏移。
- **Bug 2 (triplet src类型污染)**: 旧代码 `neg_cat(src, dst_neg, ...)` 将 paper ID 直接并入
  author 张量。下游 embedding lookup 用这些 ID 索引 author embedding table → 越界或语义错误。
  `TripletSrcRepair.validate_src_purity()` + `WALPURGIS_DEBUG=1` 打印 `invalid_count`,
  用户可确认修复是否生效。
- **Bug 3 (二元采样权重dtype不一致)**: 若用户分别传 float32 src_weight + float64 dst_weight,
  pylibcugraph 内部 bias 计算可能静默截断。8b3b67f 新增 dtype 一致性 raise;
  Walpurgis 同样检查并打印 `src_dtype / dst_dtype`。

**3. 系统角度内存并发安全**

- `NegSamplingWeights` 构造后字段不可变 (无 setter), 多线程读安全。Python 的
  `src_weight / dst_weight / vertices` 局部变量原地修改 (`src_weight = concat(...)`) 在
  同函数内是单线程安全的; 我们的封装同等安全, 且防止调用方意外写穿。
- `NegSamplingVertexMask.build()` 是纯函数 (无共享可变状态), 可安全并发调用。
  Python `neg_sample()` 同样是纯函数 (无 side effect)。
- `TripletSrcRepair.repair()` 接受 `rng` 参数 (改写: Python 用 `torch.randint` 的
  全局 CUDA RNG)。若多线程共享同一 rng 对象需外部加锁; Python 的 `torch.randint` 在 CUDA
  上有内部锁, 行为等价。调用方应为每个 worker 传独立 rng 以避免竞争。
- `WalpurgisNegSampleConfig` 构造后不可变, 跨进程/线程复制安全
  (同 `TemporalSamplerSession` 的线程安全语义)。
- **性能**: `NegSamplingVertexMask.build()` 使用 Python list, 实际 CUDA 路径须转换为
  `cupy.asarray(weights.vertices)` (调用方负责); 转换代价 O(N), 与 Python 的
  `torch.concat + cupy.asarray` 等量。断点 print 均在 `_DBG` 门控下, production 零开销。

| 上游 7ea1138 | Walpurgis 迁移 |
|---|---|
| `src_weight = concat([src_weight, zeros(num_dst)])` 移入 `if src_type!=dst_type:` | `WeightAligner._pad_src_for_dst()` 仅在 `_build_hetero_src_ne_dst()` 中调用 |
| `dst_weight = concat([zeros(num_src), dst_weight])` 同上 | `WeightAligner._pad_dst_for_src()` 同上 |
| `else: vertices = offset(arange(num_src))` 不修改 weight | `_build_hetero_src_eq_dst()` 中 `src_weight=src_weight`（不变）+ debug print 标注 |
| 删除 `elif src_weight is None and dst_weight is None: vertices = None` | `WeightAligner._is_dead_branch()` 文档化此死代码 + 防御性 ERROR print |
| `else: vertices = arange(num_src)` 同构路径 | `_build_homo()` 中 `arange(num_src_nodes)` |
| 无 validate 逻辑 | `NegSampleWeightPlan.validate()` 检查三种路径的 shape 不变量 |

**2. 用户角度 bug 排查**

- **Bug 1 (静默错误负样本)**: src==dst 路径下 weight 被错误扩展，pylibcugraph 采样引擎
  按 weight 偏移索引节点，weight 长度 2x 导致越界，采出不存在 ID（如图有 N 节点但
  采到 N~2N 的 ID）。用户看到的现象是链接预测/图学习性能莫名偏低，难以关联到权重 bug。
  `WALPURGIS_DEBUG=1` 时 `validate()` 立即打印 `"检测到 7ea1138 修复前的 bug! src_weight 被错误 concat"`。
- **Bug 2 (死代码路径激活)**: 若未来代码重构导致 src_weight/dst_weight 未被 ones() 填充
  就到达分支决策，`_is_dead_branch()` 打印 ERROR 提示，避免 vertices=None 被静默传入
  pylibcugraph（Python 里 `None` 会触发 cupy.asarray(None) 引发难以定位的 TypeError）。
- **Bug 3 (validate 精确诊断)**: `validate()` 区分三种 shape 错误:
  (a) 7ea1138 修复前 bug（sw_len == num_src + num_dst）打印专属 BUG 信息；
  (b) 其他 shape 不匹配打印通用 mismatch；
  (c) 正常情况无输出。

**3. 系统角度**

- **类型安全**: `NegSampleWeightPlan` 是 `@dataclass`，字段类型明确；`validate()` 用
  `_len()` helper 兼容 torch.Tensor / list / ndarray，不依赖 torch.numel() 是否可用。
- **内存安全**: `WeightAligner._pad_src_for_dst` / `_pad_dst_for_src` 每次返回新 tensor，
  不修改输入（与上游 `torch.concat([...])` 语义一致，不是 in-place 操作）。
  `_build_hetero_src_eq_dst` 的 `src_weight=src_weight`（直接引用）是有意设计：
  src_weight 已是正确大小的 ones/用户提供 tensor，plan 不拷贝，caller 拥有所有权。
- **并发安全**: 所有 builder 方法为 `@staticmethod`，无共享可变状态，
  多 DataLoader worker 可安全并发调用 `prepare_neg_sample_weights()`。
- **性能**: 仅 src!=dst 分支执行两次 `torch.concat`（与上游完全一致）；
  validate() 仅调用 `numel()` / `len()`（O(1)），在热路径可通过 `WALPURGIS_DEBUG`
  门控的方式屏蔽。

---

## migrate 89c9e8d: [BUG] Pin CPU Memory Instead of Copying to Device

- **Upstream commit**: 89c9e8d (cugraph-gnn, NVIDIA, 2025-07-25)
- **Commit message**: `[BUG] Pin CPU Memory Instead of Copying to Device`
- **Upstream diff** (1 file changed, 5 insertions, 4 deletions):
  - `dist_tensor.py` — `DistTensor.__setitem__`:
    删除 `val = val.cuda()`，新增 `if not val.is_cuda: val = val.pin_memory()`
  - `dist_tensor.py` — `DistEmbedding.__setitem__`:
    同上（两个类各自实现，对称修复）
  - 操作顺序: `idx.cuda()` → dtype转换 → pin/noop → `scatter(val, idx)`

- **Bug 根因**:
  `val.cuda()` 在 scatter 前把整个 val tensor 全量复制到 GPU 显存。
  GraphRAG 场景下节点特征矩阵动辄数十GB，远超单卡显存上限，直接 OOM。
  WholeGraph scatter 本身设计支持从 pinned host 内存 DMA 写入分布式存储——
  `.cuda()` 是多余且有害的冗余拷贝，属于设计偏差引入的 critical bug。
  `pin_memory()` 仅锁页（不拷贝到显存），由 DMA 控制器按需传输。

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/dist_tensor.py`** — 新增

**迁移要点**:
- `PinnedValBuffer`: 值对象，携带 (tensor, was_cuda, was_pinned, dtype_cast) 四元状态，
  让 scatter 决策路径可被观测和单元测试
- `WalpurgisScatterGuard.prepare()`: 封装 89c9e8d 引入的核心内存决策逻辑
  (dtype转换 → pin/noop)，替代两处重复的 `if not val.is_cuda: val = val.pin_memory()`
- `DistTensorScatter.execute()`: DistTensor.__setitem__ 提取为静态方法，
  不依赖完整 WholeGraph 构造，可独立测试
- `DistEmbeddingScatter.execute()`: 同上，对应 DistEmbedding.__setitem__

**改写20%（鲁迅拿法）**:
- `PinnedValBuffer` 对象替代 Python 中 `val` 的局部变量再赋值模式
- `WalpurgisScatterGuard` 静态类替代两处内联的 `if not val.is_cuda` 重复逻辑
- `DistTensorScatter` / `DistEmbeddingScatter` 分离可测接口（Python 是 __setitem__ 内联）
- 全链路6个 `WALPURGIS_DEBUG=1` 断点 print，覆盖: 进入guard → dtype转换 → pin决策 →
  pin完成 → scatter触发 → scatter完成
## migrate 824a809: fix mnnvl issue with using nvlink clique uuid

- **Upstream commit**: 824a809 (cugraph-gnn, NVIDIA, 2024-08-19)
- **Commit message**: `fix mnnvl issue with using nvlink clique uuid`
- **Upstream diff** (1 file changed, 13 insertions, 7 deletions):
  - `cpp/src/wholememory/communicator.cpp`:
    - 新增 `#include <string>`
    - `std::set<int> clique_ids{}` → `std::set<std::string> clique_uuids{}`
    - `clique_ids.insert(cliqueId)` →
      `clique_uuids.insert(std::string(reinterpret_cast<const char*>(clusterUuid), NVML_GPU_FABRIC_UUID_LEN))`
    - `clique_num = clique_ids.size()` → `clique_num = clique_uuids.size()`
    - 提取本 rank 的 `std::string uuid` 局部变量
    - `for (auto clique_id : clique_ids)` → `for (auto clique_uuid : clique_uuids)`
    - `if (clique_id == ri.fabric_info.cliqueId)` → `if (clique_uuid == uuid)` (字符串比较替代整数比较)

- **Bug 根因**:
  MNNVL (Multi-Node NVLink) 多 fabric 拓扑下，`cliqueId` 是 per-fabric 局部整数，
  不同物理 clique 可能被 NVML 分配相同 int 值。将 int cliqueId 存入 `std::set<int>`
  会导致跨 fabric 的不同 clique 被视为同一 clique: `clique_num` 偏少, `clique_id` 赋值错误
  → MNNVL 通信拓扑错误，AllGather/AllReduce 路由到错误 clique。
  修复: 改用 `clusterUuid` (128-bit binary blob，NVML 保证全局唯一) 作 set 键。

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/nvlink_clique.py`** — 新增

**迁移要点**:
- `CliqueUUID`: 封装 `std::string(reinterpret_cast<const char*>(clusterUuid), LEN)` 为 Python bytes 对象，
  加 `hex_str` 属性方便调试打印，加 `is_zero()` 对应 C++ 零 UUID 检查
- `CliqueRegistry`: 合并 C++ 两段逻辑 (先 `set.insert` 全部 → 再 `for` 遍历找 id)
  为单一 `dict[CliqueUUID, int]`，插入即分配 id，O(1) 查找
- `WalpurgisCliqueInfo`: 对应 `wm_comm->clique_info` 结构体字段，加 `validate()` 后置校验
- `exchange_rank_clique_info()`: 对应 `exchange_rank_info()` 中 clique 相关逻辑，
  输入 rank uuid 列表，输出 `WalpurgisCliqueInfo`
- `pre_824a809_buggy_exchange()`: 仅用于回归对比，演示 int cliqueId 碰撞 bug

**改写20%（鲁迅拿法）**:
- `CliqueUUID` 对象替代 C++ `std::string` 二进制 blob，加 `hex_str`/`short_hex` 调试属性
- `CliqueRegistry` dict 插入即分配 id，消除 C++ 二次 for 遍历 O(N→1)
- `sorted_id()` 提供与 C++ `std::set` 字典序完全对齐的 id，同时保留默认插入顺序路径
- `WalpurgisCliqueInfo.validate()` 在 `clique_id`/`clique_num` 赋值后做后置校验（C++ 无）
- `pre_824a809_buggy_exchange()` 隔离旧 bug 路径，可用于回归测试证明 int 碰撞场景
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，每个 uuid 插入/查找/赋值均打印 hex 摘要

### 质量审查（Knuth 标准）

**1. diff 对比源**

| 上游 89c9e8d | Walpurgis 迁移 |
|---|---|
| `DistTensor.__setitem__`: 删 `val = val.cuda()` | `WalpurgisScatterGuard.prepare()` 不调用 `.cuda()` |
| `if not val.is_cuda: val = val.pin_memory()` | `prepare()` 中同逻辑，封装为 `PinnedValBuffer` |
| `DistEmbedding.__setitem__`: 同上对称修复 | `DistEmbeddingScatter.execute()` 同样调用 `prepare()` |
| dtype 转换在 pin 之前（`val.dtype != self.dtype` 先判断） | `prepare()` 内: dtype转换 → pin，顺序一致 |
| `idx = idx.cuda()` 位置不变（始终在最前） | `execute()` 第一步即 `idx.cuda()`，顺序一致 |
| 两个类各自独立的 `__setitem__` | `DistTensorScatter` / `DistEmbeddingScatter` 各自静态方法 |

**2. 用户角度 bug 排查**

- **Bug 1 (OOM)**: `val.cuda()` 全量搬运，数十GB特征矩阵直接显存溢出。
  `WALPURGIS_DEBUG=1` 打印 `was_cuda` + `is_pinned`，用户可立即确认
  是否走 pin_memory 路径还是意外触发了 .cuda()。
- **Bug 2 (静默错误)**: 若 val 已是 GPU tensor (`was_cuda=True`) 且 dtype 不匹配，
  `val.to(dtype)` 会产生新 GPU tensor，`pin_memory()` 分支不执行——
  断点打印 `dtype_cast=True, was_cuda=True` 让用户清楚看到此路径。
- **Bug 3 (多rank同步)**: scatter 是 WholeGraph collective 操作，每个 rank 必须调用。
  `DistTensorScatter.execute()` 无条件执行 scatter（不短路），与上游语义一致。

**3. 系统角度**

- **内存安全**: `pin_memory()` 成本低（仅锁页），不复制到显存；
  若 val 已是 pinned (`was_pinned=True`)，`pin_memory()` 返回共享锁页区引用，
  几乎零开销。`PinnedValBuffer.was_pinned` 字段记录此状态供调优。
- **类型安全**: `WalpurgisScatterGuard.prepare()` 接受 `target_dtype` 为显式参数，
  不依赖 self；`DistTensorScatter.execute()` 同样显式传 `dtype=`，无隐式状态。
- **并发安全**: `prepare()` 是纯函数（输入tensor → 输出PinnedValBuffer），无副作用，
  多 DataLoader worker 可安全并发调用。
- **性能**: pin_memory() 在热路径（每次 __setitem__），但仅在 `not val.is_cuda`
  时执行，已在 GPU 的 tensor 零开销。dtype 转换同原逻辑，不增加额外操作。
| 上游 824a809 | Walpurgis 迁移 |
|---|---|
| `std::set<int> clique_ids{}` → `std::set<std::string> clique_uuids{}` | `CliqueRegistry._uuid_to_id: Dict[CliqueUUID, int]` 替代 `std::set` |
| `clique_ids.insert(cliqueId)` → `clique_uuids.insert(std::string(clusterUuid, LEN))` | `registry.insert(CliqueUUID(rank_uuids[r]))` |
| `clique_num = clique_uuids.size()` | `clique_info.clique_num = registry.clique_num()` |
| `std::string uuid = std::string(ri.fabric_info.clusterUuid, LEN)` | `self_uuid = CliqueUUID(rank_uuids[world_rank])` |
| `for (auto clique_uuid : clique_uuids) { if (clique_uuid == uuid) clique_id = id; id++; }` | `registry.sorted_id(self_uuid)` O(1) 查找 |
| `#include <string>` | Python bytes 内置，无需 import |

**2. 用户角度 bug 排查**

- **Bug 1 (clique_id 赋值错误)**: pre-824a809 多 fabric 下 int cliqueId 碰撞，`clique_id` 赋值到错误 id。
  `pre_824a809_buggy_exchange()` 可复现此 bug，`exchange_rank_clique_info()` 对比结果即可验证修复。
- **Bug 2 (clique_num 偏少)**: int cliqueId 碰撞导致 set 大小偏小，`clique_num` 错误。
  `CliqueRegistry.dump()` 打印全部 uuid 及其 id，可直观确认去重是否正确。
- **Bug 3 (MNNVL 通信拓扑错误)**: clique_id 错误 → AllReduce 路由到错误 NVLink clique → 性能骤降或挂死。
  `WalpurgisCliqueInfo.validate()` 在赋值后立即校验 `clique_id ∈ [0, clique_num)`，提前发现。
- **调试路径**: `WALPURGIS_DEBUG=1` 时每个 rank 的 uuid hex + clique_id 赋值过程全打印，
  对比多 rank 日志可直接确认 uuid 去重是否符合预期。

**3. 系统角度**

- **类型安全**: `CliqueUUID.__eq__/__hash__` 基于 bytes，dict/set 操作类型安全。
  pre-824a809 C++ `std::set<int>` 允许任意 int 进入，Python 改写通过 `CliqueUUID` 构造函数强制长度校验。
- **内存安全**: `CliqueUUID._raw` 长度固定为 `NVML_GPU_FABRIC_UUID_LEN`，构造时截断/补零与 C++ `std::string(ptr, len)` 行为对齐；无悬空指针风险（C++ reinterpret_cast 路径有潜在越界，Python bytes 切片安全）。
- **并发安全**: `exchange_rank_clique_info()` 是纯函数，无全局状态；`CliqueRegistry` 为值语义，每次调用创建新实例，线程安全。
- **性能**: `CliqueRegistry.insert/get_id` 均为 O(1) dict 操作；`sorted_id()` 为 O(N log N) 仅在 `use_sorted_id=True` 时执行，N=clique 数量（通常 <10）。

---

## migrate b25bc88: Support Disjoint Sampling in cuGraph-PyG

- **Upstream commit**: b25bc88 (cugraph-gnn, NVIDIA, 2026-05-22)
- **Commit message**: `[FEA] Support Disjoint Sampling in cuGraph-PyG`
- **Upstream diff** (4 files changed, 174 insertions, 10 deletions):
  - `neighbor_loader.py` + `link_neighbor_loader.py`: 删除 docstring "Currently unsupported." + 删除 `if disjoint: raise ValueError("Disjoint sampling is currently unsupported")` + 新增 `disjoint=disjoint` 传入 `DistributedNeighborSampler`
  - `distributed_sampler.py`: `__init__` 新增 `disjoint: bool = False`; `sample_kwargs` dict 新增 `"disjoint_sampling": disjoint`; `__calc_local_seeds_per_call` 新增 `disjoint: bool = False` 参数 + 所有参数改为 keyword-only (`*`); 修正 bucket 顺序 bug（`unknown_fanout` 检查从 `heterogeneous` 规范化之前移到之后）; `disjoint=True` 时 `fanout_prod *= fanout[0]`（per-seed 不去重，内存放大）
  - `tests/loader/test_neighbor_loader.py`: 新增 `test_link_neighbor_loader_disjoint`、`test_neighbor_loader_disjoint`、`test_neighbor_loader_disjoint_batch_structure` 三个测试（验证 per-seed 子图互不相交）

- **Bug 根因（两个）**:
  1. **disjoint 不可用**: `if disjoint: raise ValueError` 硬拦截了所有 disjoint=True 请求，即使底层 pylibcugraph 已支持
  2. **内存估算 bucket 顺序错误**: `heterogeneous` 采样时，`unknown_fanout` 检查在规范化之前，导致 `fanout` 含 `<=0` 值时提前返回 `UNKNOWN_VERTICES_DEFAULT`，跳过 heterogeneous 规范化路径

### Walpurgis 迁移位置

**文件: `src/walpurgis/models/disjoint_sampler.py`** — 新增

**迁移要点**:
- `DisjointSamplingConfig`: 封装 `sample_kwargs` 构建 (对应 `distributed_sampler.py` `__init__` 的 dict 初始化)
- `DisjointMemoryEstimator`: 封装 `__calc_local_seeds_per_call` 逻辑，含 b25bc88 bucket 顺序修正 + disjoint 放大
- `WalpurgisDisjointSession`: 封装 disjoint 采样完整配置，含内存估算 + 验证
- `validate_disjoint_batches()`: 可复用的 per-seed 子图互不相交验证器（对应 test 中内联逻辑）

**改写20%（鲁迅拿法）**:
- `DisjointSamplingConfig` 对象替代 Python `__init__` 中散落的 dict 更新
- `DisjointMemoryEstimator` 静态类替代 Python instance method（无需构造完整 sampler，可单独测试）
- `WalpurgisDisjointSession.validate()` soft validation（Python 是 hard raise，我们改写为 warning + bool 返回）
- `validate_disjoint_batches()` 加 overlap 统计（Python test 仅 assert，无统计）
- 全链路 `WALPURGIS_DEBUG=1` 断点 print

### 质量审查（Knuth 标准）

**1. diff 对比源**

| 上游 b25bc88 | Walpurgis 迁移 |
|---|---|
| `neighbor_loader.py`: 删 `if disjoint: raise ValueError` | `WalpurgisDisjointSession.__init__` 不 raise，由 `validate()` soft-warn |
| `distributed_sampler.py`: `sample_kwargs["disjoint_sampling"] = disjoint` | `DisjointSamplingConfig.to_sample_kwargs()` 返回含 `"disjoint_sampling"` 的 dict |
| `__calc_local_seeds_per_call(*, ..., disjoint, ...)` keyword-only | `DisjointMemoryEstimator.calc_seeds_per_call(fanout, ..., *, ...)` 同为 keyword-only |
| bucket 顺序: hetero_normalize → unknown_check → fanout_prod | `calc_seeds_per_call` 中同序: normalize → unknown check → prod |
| `if disjoint: fanout_prod *= fanout[0]` | `if disjoint: fanout_prod *= amplification`（同逻辑，加 debug print） |
| `super().__init__(local_seeds_per_call=...)` 改为关键字参数 | `calc_seeds_per_call` 内部全部 keyword arg 传递 |
| 3个新 test: link_disjoint, neighbor_disjoint, batch_structure | `validate_disjoint_batches()` 封装 batch_structure 逻辑 |

**2. 用户角度 bug 排查**

- **Bug 1 (disjoint 静默失效)**: 若 `disjoint_sampling` 未正确传入采样引擎，采样结果会在 cross-seed 去重，用户看到的 `n_id` 数量偏少。`DisjointSamplingConfig.to_sample_kwargs()` 每次打印完整 dict（`WALPURGIS_DEBUG=1`），用户可立即确认 `disjoint_sampling=True` 是否到达底层。
- **Bug 2 (内存 OOM)**: `disjoint=True` 时内存放大 `fanout[0]` 倍，若用户沿用非 disjoint 的 `local_seeds_per_call`，会 OOM。`DisjointMemoryEstimator` debug print 打印 `fanout_prod * amplification`，明示放大量。
- **Bug 3 (hetero bucket 顺序)**: 旧代码 `heterogeneous=True` 且 `fanout` 含 `<=0` 时，提前返回 `UNKNOWN_VERTICES_DEFAULT` 而不做规范化。改写后 `normalize_hetero_fanout` 先执行，再检查 unknown，与 b25bc88 修正一致。

**3. 系统角度**

- **类型安全**: `DisjointSamplingConfig.disjoint` 为 `bool`，`to_sample_kwargs()` 强制写入 dict；Python 的 bool→dict 散落在 `__init__` 中，容易被误覆盖。
- **内存安全**: `DisjointMemoryEstimator.calc_seeds_per_call` 是纯函数，无副作用；`fanout_prod` 在除法前不为零（`any(x<=0)` 已在 prod 前拦截）。`disjoint=True` 且 `fanout[0]<=0` 时 `validate()` 输出 warning，避免除以 0。
- **并发安全**: `WalpurgisDisjointSession` 是值语义对象，每个 loader worker 应持有自己副本。`validate_disjoint_batches` 是无状态函数，thread-safe。
- **性能**: `DisjointMemoryEstimator` 估算为 O(len(fanout)) 纯 Python，每次 loader 初始化调用一次，不在热路径。`validate_disjoint_batches` 为 O(B × S² × H)（B=batches, S=seeds/batch, H=hops），仅在 `WALPURGIS_DEBUG=1` 或显式调用时运行。

---

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

## migrate 5909ae8: Fp16 embedding train

- **Upstream commit**: 5909ae8 (cugraph-gnn, linhu-nv, PR #462)
- **Commit message**: `Fp16 embedding train`
- **Upstream diff** (3 files changed, 42 insertions, 35 deletions):
  - `cpp/src/wholememory/embedding.cpp` — `gather_gradient_apply()` 3处修改:
    1. `dedup_grads` 中间缓冲区 `device_malloc` dtype 从 `grads_desc->dtype` 改为
       `WHOLEMEMORY_DT_FLOAT` (line~240): 无论输入何种浮点 dtype, 累加缓冲区钉死 float32
    2. 传入 `dedup_indice_and_gradients` 的 grads 指针: 去掉 `static_cast<const float*>`
       改传 `void*` (line~251): 解除硬转型, 由内部模板处理
    3. 新增 `recv_grad_tensor_desc.dtype = WHOLEMEMORY_DT_FLOAT` (line~297):
       scatter_back 时的描述符也反映已升精度
  - `exchange_embeddings_nccl_func.cu` — CUDA kernel 泛化:
    - `DedupIndiceAndGradientsKernel`: 新增 `template <typename GradT>`,
      `float* grads` → `GradT* grads`, 累加时 `static_cast<float>()` 升精度
    - `dedup_indice_and_gradients_temp_func`: 新增 `typename GradT` 模板参数,
      `const float* grads` → `const void* grads`
    - dispatch: `REGISTER_DISPATCH_ONE_TYPE(..., SINT3264)` →
      `REGISTER_DISPATCH_TWO_TYPES(..., SINT3264, BF16_HALF_FLOAT)` (二维6条路径)
    - validation: `grads_desc.dtype == WHOLEMEMORY_DT_FLOAT` →
      `wholememory_dtype_is_floating_number(grads_desc.dtype)` (允许 fp16/bf16 进入)
  - `exchange_embeddings_nccl_func.h`: 公开签名 `const float*` → `const void*`

- **功能说明**:
  在 fp16/bf16 embedding 训练时, 原代码假设梯度是 float32 直接 cast, 导致类型错误.
  5909ae8 让 `dedup_indice_and_gradients` 接受任意浮点梯度 (void* + 模板),
  内部在 CUDA kernel 中逐元素 `static_cast<float>` 升精度后累加,
  输出中间缓冲区始终 float32. `gather_gradient_apply` 中的描述符也同步钉死为 float32,
  确保后续 scatter_back 不会把 float32 数据当 fp16 解析.

### Walpurgis 迁移位置

**新增文件:**
- `src/walpurgis/models/fp16_grad_dedup.py` — 主迁移文件

**迁移要点**:
- `DedupGradSession`: 封装上游 `(indices_ptr, indice_desc, grads_ptr, grads_desc)` 为
  Python dataclass, `validate()` 提前报错 (上游是 assert 无友好消息)
- `_DISPATCH_TABLE`: Python dict 模拟 `REGISTER_DISPATCH_TWO_TYPES` 展开的6条路径
  `{(idx_dtype, grad_dtype): fn_name}`, `dump_dispatch_table()` 可打印全表 (上游无此方法)
- `dedup_indice_and_gradients()`: `torch.scatter_add_` 等价 CUDA blockIdx 并行累加;
  输入 fp16/bf16/fp32, 输出始终 float32
- `GatherGradientApplyConfig` + `gather_gradient_apply()`: 封装 `embedding.cpp` 整体流程,
  使 3处 5909ae8 修改点在单函数内可追踪
- `_validate_dtypes()`: 对应新 `wholememory_dtype_is_floating_number()` 判断,
  改写为 friendly ValueError

**改写20%（鲁迅拿法）**:
- `DedupGradSession.validate()`: 提前 Python 层校验, 有意义的 ValueError 而非裸 assert
- `_DISPATCH_TABLE` + `dump_dispatch_table()`: 模拟 `REGISTER_DISPATCH_TWO_TYPES` 展开,
  新增 dump 方法使6条路径可见 (上游宏展开无此能力)
- `GatherGradientApplyConfig.validate()`: 打印完整 dtype 流转链
  `fp16 → float32(dedup) → float32(update) → fp16(writeback)` (上游无日志)
- 自测 `__main__`: 6个测试用例覆盖 fp16/bf16/fp32/int64/非法dtype 全路径
- 全链路 8处断点 print (WALPURGIS_DEBUG=1 开启):
  1. `DedupGradSession.validate` — indices/grads dtype + dispatch 路径确认
  2. `dedup_indice_and_gradients` — 升精度路径 (static_cast<float> 语义)
  3. 排序后 indices 预览
  4. fp16→fp32 cast 最大误差检测
  5. 去重比例 (N→K, run_count 对应)
  6. 输出统计 (dedup_grads dtype=float32, 对应 line~297 recv_grad_tensor_desc 钉死)
  7. `gather_gradient_apply` dtype 链打印
  8. writeback cast 误差 + 更新行数

### 质量审查 (Knuth 标准)

**1. diff 对比源**

| 上游 5909ae8 | Walpurgis 迁移 |
|---|---|
| `device_malloc(total_recv_count * D, WHOLEMEMORY_DT_FLOAT)` | `dedup_grads = torch.zeros(K, D, dtype=torch.float32)` |
| `dedup_indice_and_gradients(void* grads, ...)` | `DedupGradSession(indices, grads)` 接受 void*-等价的任意浮点 tensor |
| `recv_grad_tensor_desc.dtype = WHOLEMEMORY_DT_FLOAT` | `gather_gradient_apply` 输出 dtype 钉死 float32, 注释标注 line~297 |
| `template <typename GradT>` kernel | `grads.float()` — Python 等价 `static_cast<float>` |
| `REGISTER_DISPATCH_TWO_TYPES(SINT3264, BF16_HALF_FLOAT)` | `_DISPATCH_TABLE: {(int32/int64, fp16/bf16/fp32): fn_name}` (6条路径) |
| `wholememory_dtype_is_floating_number(grads_desc.dtype)` | `if grads_dtype not in SUPPORTED_GRAD_DTYPES` + ValueError |
| `static_cast<const GradT*>(grads)` in temp_func | `session.grads.to(config.grad_input_dtype)` cast |
| 无 debug 日志 | 8处断点 print, WALPURGIS_DEBUG=1 门控 |

**2. 用户角度 bug 排查**

- **5909ae8 修复的 bug**: 原代码在 fp16 embedding 训练时,
  `gather_gradient_apply` 对梯度做 `static_cast<const float*>` 强转,
  实际传入的是 fp16 指针, 会读出垃圾数据 (type punning UB).
  Walpurgis 迁移中 `DedupGradSession.validate()` 会检查 dtype 一致性,
  `gather_gradient_apply` 有显式 dtype 检查 + 警告 print.
- **新增调用链风险**: `scatter_add_` 不是原子操作, 若多线程并发调用
  `dedup_indice_and_gradients` 共享同一 `embedding_weight` tensor 会有竞态.
  缓解: 每次调用创建新 `result = embedding_weight.clone()`, 不原地修改.
  上游 CUDA kernel 也是每 block 独立写 dedup_grads, 无共享写, 语义一致.
- **index 越界**: 若 `indices.max() >= embedding_weight.shape[0]`, `result[dedup_indices]`
  会 IndexError. 上游是 CUDA out-of-bounds access (未定义行为). 改写: Python
  IndexError 有清晰 traceback, 比 CUDA 崩溃更易调试.
- **空 indices**: `indices` 为空时 `unique_indices.shape[0] == 0`, `scatter_add_` 是 no-op,
  返回 clone 的原始 weight. 上游 `run_count=0` 路径也是 no-op, 语义一致.

**3. 系统角度内存并发安全**

- `DedupGradSession` 是不可变数据容器 (post-init 不修改), 可安全跨线程传递.
  `validate()` 幂等 (`_validated` flag), 多次调用安全.
- `dedup_indice_and_gradients()` 无全局状态, 纯函数 (除 stderr print).
  `torch.argsort` / `torch.unique` / `scatter_add_` 均不修改输入 tensor.
- `gather_gradient_apply()`: `embedding_weight.clone()` 保证输入不被修改,
  返回新 tensor. 多线程并发对不同 embedding 调用安全.
- `_DISPATCH_TABLE` 是模块级常量, 只读, 并发安全.
- `dump_dispatch_table()` 只做 print, 无状态修改, 并发安全.
- **内存开销**: `grads.float()` 会分配 [N, D] float32 中间缓冲区.
  若 N 很大 (分布式 all_gather 后的 total_recv_count), 这是必要开销
  (上游 CUDA 也有对应的 `dedup_grad_recv_buffer_handle.device_malloc`).
  可通过 in-place cast (`grads_float32 = grads.to(torch.float32, copy=False)`)
  在已是 float32 时省去拷贝. 当前实现: `grads[sorted_order].float()` 已经
  包含 reorder, 故无法 zero-copy. 这与上游 CUDA kernel 语义一致 (kernel 内
  逐元素 cast, 无 zero-copy 路径).


## migrate 7c2907f: [BUG] Correct De-Offset of Edge Label Index

- **Upstream commit**: 7c2907f (cugraph-gnn, NVIDIA, 2025-07-28)
- **Commit message**: `[BUG] Correct De-Offset of Edge Label Index`
- **PR**: #258，作者 Alex Barghi (alexbarghi-nv)
- **Upstream diff** (4 files changed, 260 insertions, 10 deletions):
  - `loader/link_loader.py`: `edge_label_index = edge_label_index.detach().clone()`
    新增 drop_last + 边数不足的早期 ValueError
  - `loader/node_loader.py`: `input_nodes = input_nodes.detach().clone()`
    新增 drop_last + 节点数不足的早期 ValueError
  - `sampler/sampler.py`:
    - `HeterogeneousSampleReader.__decode_coo`: `integer_input_type = None` 提前初始化；
      旧 `edge_inverse[0] -= __vertex_offsets[src]` / `edge_inverse[1] -= __vertex_offsets[dst]`
      替换为词典序判断:
      `if input_type[0] < input_type[2]: dst -= src.max()+1`
      `else: src -= dst.max()+1`
    - `HomogeneousSampleReader.__decode_csr` / `__decode_coo`:
      `edge_inverse = edge_inverse.view(2,-1)` 提前赋值再放入 metadata tuple
  - `tests/loader/test_neighbor_loader.py`:
    新增 3 个双向异构图 link prediction 测试

- **Bug 根因**:
  Heterogeneous 图采样后，不同类型节点被 concat 为单一全局编号空间，
  各类型节点在 minibatch renumber map 中的排列顺序由词典序决定。
  旧代码用 `__vertex_offsets[integer_input_type]` 做固定减法——
  该 offset 是全图层面的绝对偏移，与 minibatch 内的相对排列无关；
  当 src_type != dst_type 时，减去错误的 offset，
  edge_label_index 解码出的节点 ID 指向错误位置，链路预测 batch 静默返回错误结果。
  修复: 按词典序判断两端节点在 renumber map 中的先后，
  以 `max()+1` 动态确定偏移量，与全图绝对 offset 无关。

- **Knuth 审查**:
  1. diff 对比源:
     旧代码 `edge_inverse[0] -= vertex_offsets[src]` 依赖全图绝对 offset，
     新代码 `edge_inverse[1] -= edge_inverse[0].max()+1` 依赖 minibatch 内相对位置；
     两者在 src_type==dst_type 时等价（同段 offset 相消），
     在 src_type!=dst_type 时新代码正确，旧代码必然出错。
     HomogeneousSampleReader 的 view 提前是防御性重构，不改变语义。
  2. 用户角度 bug:
     双向异构图（如 user→merchant + merchant→rev_to→user）做 link prediction 时，
     每个 batch 的 edge_label_index 解码出错误的节点 ID，
     assert 通过但节点映射乱序，导致训练 loss 异常升高而无明显报错，
     极难与模型本身的问题区分；detach().clone() 缺失时，
     外部传入的 edge_label_index 被 in-place 加 offset 修改，
     调用方下次迭代时 tensor 已污染，产生难以复现的随机 bug。
  3. 系统角度安全:
     词典序去偏移依赖 minibatch 内节点编号的单调性假设（renumber 保证），
     若 renumber map 顺序变化则此假设失效；
     detach().clone() 增加了内存开销（每个 mini-batch 额外一份拷贝），
     但在多 worker DataLoader 场景下是必要的隔离措施，
     否则共享 tensor 的 in-place 修改会触发 CUDA multiprocessing 竞态。

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/edge_label_deoffset.py`** — 新增

**迁移要点**:
- `DeOffsetStrategy`: 枚举，VERTEX_OFFSET (旧 BUG 路径，禁用) vs LEXICOGRAPHIC (7c2907f 修复路径)
- `EdgeInverseBundle`: 值对象，携带 (src, dst, input_type)，替代裸 list 操作
- `HeteroEdgeLabelDeoffset`: 执行类，封装词典序去偏移逻辑，`apply(bundle)` 原地修改并返回
- `HomoEdgeInverseView`: 静态工具类，封装 `view(2,-1)` + numel 奇偶校验
- `InputTensorGuard`: 守卫类，封装 `detach().clone()` + drop_last 早期校验，支持 edge/node 两种模式
- `build_deoffset_session()`: 工厂函数，对应 7c2907f 标准使用路径

**改写20%（鲁迅拿法）**:
- `DeOffsetStrategy` 枚举明示旧 BUG 路径，`build_deoffset_session` 见到 VERTEX_OFFSET 直接 raise，
  从架构上封死回退旧代码的可能
- `EdgeInverseBundle` 值对象在 `__post_init__` 做形状一致性校验，
  上游裸 list 操作无任何校验，异形 tensor 会静默产生错误的 max()+1
- `HomoEdgeInverseView` 在 view 前检查 `numel % 2 != 0`，
  上游直接 `edge_inverse.view(2,-1)`，numel 为奇数时 RuntimeError 不指向根因
- `InputTensorGuard._check_drop_last` 统一了 edge/node 两种模式的校验逻辑，
  上游 link_loader / node_loader 两处代码重复

全链路7个 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  InputTensorGuard.__init__ 入口 →
  InputTensorGuard._check_drop_last 计数 →
  EdgeInverseBundle.__post_init__ 形状 →
  HeteroEdgeLabelDeoffset.apply 入口 + 词典序分支 →
  HomoEdgeInverseView.apply 前后形状 →
  build_deoffset_session 出口


## migrate 662a6d9: fix shm permission, avoid shm access from other user

- **Upstream commit**: 662a6d9 (cugraph-gnn, linhu-nv + alexbarghi-nv, 2026-05-19, PR #463)
- **Commit message**: `fix shm permission, avoid shm access from other user`
- **Upstream diff** (1 file changed, 4 insertions, 4 deletions):
  - `cpp/src/wholememory/memory_handle.cpp` — `global_mapped_host_wholememory_impl` 三处 `shmget()` 调用:
    - CREATE 路径 (rank==0): `0644 | IPC_CREAT | IPC_EXCL` → `0600 | IPC_CREAT | IPC_EXCL`
    - ATTACH 路径 (rank!=0): `0644` → `0600`
    - DESTROY 路径 (`unmap_and_destroy_shared_host_memory`): `0644` → `0600`
  - 附带版权年份更新: `2019-2025` → `2019-2026`

- **Bug 根因**:
  `0644` 的 group_read(040) + other_read(004) 允许同主机其他用户通过 `shmat(shm_id, NULL, SHM_RDONLY)`
  附加共享段并读取其内容。该段存放的是 GPU host-side 映射内存——包含模型权重、嵌入向量、梯度缓冲区。
  多租户 HPC 环境（如 SLURM 集群、共享 Kubernetes 节点）下，同主机其他用户可通过
  `ipcs -m` 或 `/proc/sysvipc/shm` 枚举所有 System V shm 段，找到目标 shm_id 后直接读取。
  `0600` 将权限收紧为仅所有者读写，内核 `ipcperms()` 对其他 uid/gid 返回 `EACCES`。
  注: POSIX `shm_open` 路径（`use_systemv_shm_=false`）始终用 `S_IRUSR|S_IWUSR=0600`，从未受此影响。

- **Knuth 审查**:
  1. diff 对比源:
     - CREATE 路径: 加 `IPC_CREAT | IPC_EXCL` 的 `shmget` 创建新段，mode 参数直接设定段的初始 DAC 权限。
       `0644` 在段创建瞬间即暴露给其他用户; `0600` 确保创建即安全。
     - ATTACH 路径: `shmget` 无 `IPC_CREAT` 时 mode 参数用于额外 DAC 校验（内核 `ipcperms`
       将请求 mode 与已有段 mode AND 校验）。`0644` 可能宽松通过某些内核版本的检查；`0600` 明确限制。
     - DESTROY 路径: 仅获取 shm_id 以便 `shmctl(IPC_RMID)`，mode 参数对安全影响有限，
       但保持三处一致（全部 `0600`）是正确的防御性编程，防止语义不一致引发误解。
     - 三处全部修改是必要的，遗漏任一处都会留下不一致的安全状态。
  2. 用户角度 bug:
     - 多租户节点上，用户 B（同 gid）可调用 `shmat(shm_id, NULL, SHM_RDONLY)` 读取
       另一用户 A 的 WholeGraph 训练数据（模型权重、中间激活、梯度）——静默数据泄露，无任何报错。
     - 不同 gid 的用户 C 也可读（other_read=004），威胁范围更广。
     - 修复后 `shmat` 返回 `EACCES`，泄露路径被封闭，不影响 WholeGraph 自身的多进程通信
       （同 uid 的多个 rank 进程仍可正常附加，内核 owner bits 检查通过）。
  3. 系统角度安全:
     - Linux 内核 `ipc/shm.c ipcperms()`: `shmat/shmdt/shmctl` 均经过 DAC 检查，
       `0600` 确保 group=00, other=00，非 owner 的所有操作均被拒绝。
     - IPC namespace（Docker `--ipc=private`）可隔离但 HPC 环境通常不隔离；
       `0600` 是无视容器化状态的最小权限保证。
     - `IPC_CREAT | IPC_EXCL` 组合已防止同 key 段的竞态创建；
       `0600` 在此之上增加 DAC 隔离，两者正交互补。

### Walpurgis 迁移位置

**文件: `src/cuda/hetero_bench.cu`** — E11 实验新增，含 `PhilemonShmPermission` 命名空间

**迁移要点**:
- `PhilemonShmPermission::ShmCallSite`: 值结构体，记录三处调用点的
  (name, context, mode_before, mode_after, has_ipc_creat)，使 diff 可追踪
- `PhilemonShmPermission::format_mode()`: Unix 权限掩码 → rwx 字符串
- `PhilemonShmPermission::allows_foreign_read()`: 检测 group_read|other_read 是否置位
- `PhilemonShmPermission::probe_real_shm()`: 调用真实 `shmget/IPC_STAT/shmctl`，
  读出内核实际分配的 mode 值，与请求 mode 对比（验证内核无隐式 mode 扩展）
- `experiment_shm_permission()`: 6步验证:
  1. diff 对比（三处调用点 0644→0600 的格式化展示）
  2. 用户视角漏洞场景（多租户 shmat 读取路径）
  3. POSIX IPC DAC 模型（ipcperms 规则）
  4. 真实 shmget 权限探测（内核实测 mode）
  5. POSIX shm_open 路径对比（Walpurgis 不受影响的证明）
  6. 三处调用点覆盖完整性验证

**改写20%（鲁迅拿法）**:
- `ShmCallSite[3]` constexpr 数组替代分散注释——三处 diff hunk 在一个数据结构内一览无余
- `probe_real_shm()` 实际调用 `shmget + IPC_STAT + shmctl` 验证内核行为
  （上游只是改了三个数字，我们用真实系统调用验证改动是否生效）
- `allows_foreign_read()` 命名函数替代散落的 `mode & 0044` 魔法数字
- 路径3（DESTROY）的一致性价值分析（上游 commit 无此分析）

全链路5个断点 print，覆盖:
  `probe_real_shm` ftok/shmget/IPC_STAT 每步 →
  实际 mode vs 请求 mode 对比 →
  三处调用点覆盖验证汇总

## migrate 318ae6c: Updates movielens_mnmg.py to use DDP

- **Upstream commit**: 318ae6c (cugraph-gnn, NVIDIA)
- **Commit message**: `Updates movielens_mnmg.py to use DDP`
- **Upstream diff** (1 file changed, movielens_mnmg.py):
  - `import`: 新增 `from torch.nn.parallel import DistributedDataParallel as DDP`
  - `import`: 新增 `from cugraph_pyg.data import GraphStore, FeatureStore` (从函数体内提至模块顶)
  - `pylibwholegraph.torch.initialize`: 删除 `init as wm_init`，保留 `finalize as wm_finalize`
  - `init_pytorch_worker`: 删除 `wm_init(global_rank, world_size, local_rank, device_count())` 调用，
    注释改为 `# WholeGraph is initialized automatically.`
  - `cugraph_pyg_from_heterodata`: 删除函数内 `from cugraph_pyg.data import GraphStore, FeatureStore`
    (已提至模块顶)
  - `Encoder.__init__`: 参数从 `(hidden_channels, out_channels)` 改为
    `(user_in_channels, movie_in_channels, hidden_channels, out_channels)`；
    conv1/2/3 从 `SAGEConv((-1,-1), ...)` 改为显式维度
    `conv1=(movie_in, user_in)` `conv2=(user_in, movie_in)` `conv3=(hidden, hidden)`
  - `Model.__init__`: 参数新增 `num_features`；`Encoder(...)` 调用改为传入显式维度
  - `__main__`: 新增 `num_features` 提取块 (x.shape[-1] or 1)；
    `Model(...)` 调用新增 `num_features=num_features`；
    新增 `model = DDP(model, device_ids=[local_rank])`

- **Knuth 审查**:
  1. diff 对比源:
     - `conv1 = SAGEConv((movie_in_channels, user_in_channels), ...)`:
       SAGEConv((src, dst), out)，conv1 走 movie→user 方向，src=movie，dst=user，
       参数名 user_in_channels 在第二位（=dst），与直觉"先 user 后 movie"相反，注释缺失；
       forward 调用 `(x_dict["movie"], x_dict["user"])` 与此一致，但易混淆
     - `wm_init` 删除：WholeGraph 自动初始化依赖 pylibwholegraph 版本 >= 某特定版本，
       旧版本静默不初始化而非报错，调用 wm_finalize 会 segment fault
     - `DDP(model, device_ids=[local_rank])` 在 `model.to(device)` 之后，顺序正确；
       若颠倒则 DDP 参数注册在 CPU，nccl allreduce 出错
     - `import GraphStore, FeatureStore` 提至模块顶后，
       `cugraph_pyg_from_heterodata` 函数体内空了一行（diff 可见 blank line），无功能影响
  2. 用户角度 bug:
     - `data["user"].x` 由 `torch.eye(num_users_total)` 生成，
       num_users_total 大（>500K）时单 rank 存 eye 矩阵 OOM，上游无提示
     - `drop_last=True` + batch_size=256，若 eli_train.shape[1] < 256，
       train_loader 为空，`train()` 返回 0/0 ZeroDivisionError，上游无保护
     - DDP 包裹后 `model.forward` 已透明，但 `test()` 中 `model.eval()` 作用于
       DDP wrapper，实际等价于 `model.module.eval()`，BN 统计量同步依赖 DDP 配置，
       当前模型无 BN，无影响；若后续加 BN 须注意
  3. 系统角度安全:
     - `CugraphWorkerSession` (迁移改写) 封装 `cugraph_comms_shutdown + wm_finalize`，
       上游裸调在 `with use_mem_pool` 块末尾，OOM/NCCL timeout 导致 with 块异常退出时不执行
     - `DDP device_ids=[local_rank]` 假设 LOCAL_RANK == CUDA device index；
       容器内设备重映射（如 CUDA_VISIBLE_DEVICES）时两者可能不一致，nccl 报错难以定位
     - `rmm MemPool` 生命周期与 `with` 块绑定；DDP allreduce 触发 NCCL timeout
       导致 with 块异常退出时，未完成 kernel 仍持有 pool 引用，
       `rmm` 报 pool-in-use error（上游已知限制）

### Walpurgis 迁移位置

**文件: `src/walpurgis/examples/movielens/movielens_mnmg.py`** — 新增

**迁移要点**:
- `FeatureDims`: dataclass 封装 num_features 字典，`__post_init__` 校验维度 >= 1，
  `from_heterodata()` 类方法集中提取，替代 __main__ 裸 dict 字面量
- `CugraphWorkerSession`: context manager 封装 init_pytorch_worker 生命周期，
  `__exit__` 保证 cugraph_comms_shutdown + wm_finalize 在异常路径也执行
- `ModelBundle`: dataclass 封装 Model + DDP + optimizer 三件套，
  `build()` 类方法集中构建，替代 __main__ 三行散落赋值
- `EncoderShapeGuard` (`_check_encoder_shape`): 校验 SAGEConv 输入维度顺序，
  维度对调时立即 ValueError，替代训练数 epoch 后 loss 不收敛的隐患
- 大 eye 矩阵 OOM 警告 (num_users > 500K)
- 空 DataLoader ZeroDivisionError 保护

**改写20%（鲁迅拿法）**:
- `FeatureDims` dataclass 替代裸 dict + 散落 shape[-1] 提取
- `CugraphWorkerSession` context manager 替代裸函数 + 末尾裸调 shutdown/finalize
- `ModelBundle.build()` 封装 Model + DDP + Adam，替代 __main__ 三行散落
- `_check_encoder_shape()` 守卫函数替代无注释的显式维度传入
- 全链路 `WALPURGIS_DEBUG=1` 断点 print，覆盖:
  FeatureDims.from_heterodata 维度提取 →
  CugraphWorkerSession._init_worker RMM/cupy/comms 初始化各阶段 →
  load_partitions num_nodes/shard shape/label_dict 空检查 →
  cugraph_pyg_from_heterodata feature shape →
  Encoder.__init__ 维度参数 → Encoder.forward 输入输出 shape →
  ModelBundle.build 参数量/DDP/optimizer →
  train.batch out/y shape → test.batch pred/target.unique →
  train_loop epoch loss/auc →
  main barrier 检查点 / eli_train/test shape / feat_dims


## migrate dbb33ad: Use PyBuffer_FillInfo for simple buffers and simplify cleanup

- **Upstream commit**: dbb33ad (cugraph-gnn, NVIDIA, 2026-03-24)
- **Commit message**: `Use \`PyBuffer_FillInfo\` for simple buffers & simplify Python buffer cleanup`
- **Upstream diff** (1 file changed, 11 insertions, 25 deletions):
  - `python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx`
  - 新增 import: `from cpython.buffer cimport PyBuffer_FillInfo`
  - `PyWholeMemoryUniqueID.__getbuffer__`: 11行手工赋值 → `PyBuffer_FillInfo(buffer, self, &internal[0], shape[0], False, flags)` 6行调用
  - `PyWholeMemoryUniqueID.__releasebuffer__`: 7行手工清空 (含 `buffer.obj = None` BUG) → `pass`
  - `PyWholeMemoryFlattenDlpack.__releasebuffer__`: 同样 7行清空 → `pass`

- **BUG 根因**:
  旧 `__releasebuffer__` 中 `buffer.obj = None` 破坏 CPython Buffer Protocol 引用计数协议:
  1. `PyBuffer_FillInfo` (在 `__getbuffer__` 中调用) 对 `buffer.obj = self` 执行 `Py_INCREF` → refcount(self) += 1
  2. 旧 `__releasebuffer__` 执行 `buffer.obj = None` → Python 赋值导致 self 的 refcount -= 1 (提前归还)
  3. Python 运行时在 `__releasebuffer__` 返回后对原 `buffer.obj` 再执行 `Py_DECREF` → 双重释放
  结果: `self` (PyWholeMemoryUniqueID) 引用计数比预期少 1, 在多 memoryview 视图叠加时
  触发 use-after-free。低并发下不易复现, 高并发 SIGSEGV 或静默数据损坏 (ID 值已错误但校验通过)。
  `pass` 修复: Py_buffer 由 Python 运行时统一清理, `__releasebuffer__` 不应提前操作任何字段。

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 dbb33ad | Walpurgis 迁移 |
     |---|---|
     | Cython .pyx, `PyBuffer_FillInfo` 直接调用 C API | Python 抽象层 `FillInfoArgs` dataclass + `call_fill_info()` |
     | 旧 11 行手工赋值 → 新 6 行 `PyBuffer_FillInfo` 调用 | `assert_fill_info_semantic()` 逐字段对比等价性 |
     | `__releasebuffer__` 改为 `pass` | `ReleaseBufferPolicy` 枚举, NOOP vs MANUAL_CLEAR |
     | `buffer.obj = None` BUG 仅 commit message 文字说明 | `ObjDoubleDecrefBug` 类: `simulate_bug()` 引用计数路径演示 |
     | `flags` 由 `PyBuffer_FillInfo` C API 内部处理 | `BufferFlags` IntFlag 枚举, 对应 `PyBUF_SIMPLE/WRITABLE/FORMAT/ND/STRIDES` |
     | 两处 `__releasebuffer__` (UniqueID + FlattenDlpack) | `WalpurgisUniqueIdBuffer` 统一封装 + `release_policy` 参数 |

  2. **用户角度 bug**:
     - 调用 `memoryview(py_unique_id)` 或 `bytearray(py_unique_id)` 时触发 `__getbuffer__` + `__releasebuffer__` 对。
       旧代码 `buffer.obj = None` 在 `__releasebuffer__` 中执行后,
       memoryview 析构时 Python 运行时再对原 obj 执行 `Py_DECREF` → double free。
       若同时有第二个 memoryview 持有同一缓冲区地址, 下次访问即 use-after-free。
       错误形式: 随机 SIGSEGV (obj 已释放内存被复用) 或 UniqueID 字节值损坏 (但 len/format 通过检查)。
     - 旧 `__getbuffer__` 固定 `buffer.format = 'c'` (char), 无视 `flags & PyBUF_FORMAT`。
       调用方请求 `PyBUF_FORMAT` 时仍返回 `'c'`, 可能绕过下游格式校验。
       `PyBuffer_FillInfo` 按 flags 正确处理: 未设 `PyBUF_FORMAT` 时 `format=NULL`。

  3. **系统角度安全**:
     - `Py_buffer.obj` 字段遵循"借用引用"协议: `__getbuffer__` 做 INCREF (通过 `PyBuffer_FillInfo`),
       `__releasebuffer__` 不做 DECREF (运行时做)。旧代码 `obj=None` 破坏此协议,
       是经典 CPython 扩展模块引用计数陷阱, 静态分析工具 (如 `refcount-checker`) 可检出但易被忽略。
     - `PyBuffer_FillInfo` 处理 `PyBUF_WRITABLE` flag: `readonly=True` 但请求可写时抛 `BufferError`。
       旧代码 `buffer.readonly = 0` 无条件设可写, 不验证 flags, 调用方无法区分"确认可写"与"未检查"。
     - 两处 `__releasebuffer__` (UniqueID 和 FlattenDlpack) 是同一 BUG 模式的两个实例,
       表明此写法是代码库内的系统性问题而非孤立错误, 迁移时应全部覆盖。

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/unique_id_buffer.py`** — 新增

**迁移要点**:
- `BufferFlags`: `IntFlag` 枚举对应 `PyBUF_SIMPLE/WRITABLE/FORMAT/ND/STRIDES` 常量,
  旧代码完全忽略 flags 参数, 本枚举明示各 flag 语义
- `FillInfoArgs`: dataclass 封装 `PyBuffer_FillInfo` 的 6 个参数 (buf/obj/len/readonly/flags/format),
  `__post_init__` 校验 `length >= 0`
- `call_fill_info()`: `PyBuffer_FillInfo` Python 层模拟, 实现 `PyBUF_WRITABLE` 冲突检测 → `BufferError`,
  返回等价 `Py_buffer` 字段 dict
- `ReleaseBufferPolicy`: 枚举, `NOOP` (dbb33ad 新, 正确) vs `MANUAL_CLEAR` (旧 BUG, 文档用)
- `ObjDoubleDecrefBug`: 文档类, `simulate_bug()` 用 `sys.getrefcount` 逐步演示双重释放引用计数路径
- `assert_fill_info_semantic()`: 逐字段对比 `call_fill_info` vs 旧 11 行手工赋值等价性,
  注明 format 字段有意差异 (旧 `'c'` vs 新按 flags)
- `WalpurgisUniqueIdBuffer`: Buffer Protocol Python 层实现,
  `get_buffer_fields()` 对应 `__getbuffer__`, `release_buffer()` 对应 `__releasebuffer__`,
  `release_policy=NOOP` 为默认正确模式
- 全链路 `WALPURGIS_DEBUG=1` 断点 print:
  `FillInfoArgs.__post_init__` 构建验证 →
  `call_fill_info.entry` flags 解析 →
  `call_fill_info.WRITABLE_CONFLICT` 冲突告警 →
  `call_fill_info.result` 字段填充结果 →
  `assert_fill_info_semantic.field_check` 逐字段对比 →
  `WalpurgisUniqueIdBuffer.__init__` 构建 →
  `get_buffer_fields.__getbuffer__` 视图获取 →
  `release_buffer.__releasebuffer__` 策略选择 →
  `ObjDoubleDecrefBug.simulate` 引用计数路径演示

---

## migrate 5baac8b: [BUG] Fix bug with drop_last when mod is 0

**上游 commit**: `5baac8b`
**上游描述**: [BUG] Fix bug with drop_last when mod is 0
**影响文件**:
- `python/cugraph-pyg/cugraph_pyg/loader/node_loader.py`
- `python/cugraph-pyg/cugraph_pyg/loader/link_loader.py`

### Bug 根因

`NodeLoader.__iter__` 与 `LinkLoader.__iter__` 中，`drop_last=True` 时执行:

```python
d = perm.numel() % self.__batch_size
perm = perm[:-d]    # BUG: d==0 时 perm[:-0] == perm[:0] => 空 tensor
```

Python 中 `-0 == 0`，`perm[:-0]` 等价于 `perm[:0]`，返回空 tensor。
当训练集大小恰好整除 batch_size 时（极常见场景），所有样本静默丢失，
训练跑零步，loss 不更新，无任何报错。

### 上游修复（2 行）

```python
d = perm.numel() % self.__batch_size
if d > 0:           # 整除时 d==0，跳过切片，保留完整 perm
    perm = perm[:-d]
```

### Walpurgis 迁移位置

**文件: `src/walpurgis/dataloader/edge_label_deoffset.py`** — 新增 `PermDropLastSlicer` 类

### 迁移要点

- `PermDropLastSlicer`: 封装 5baac8b 修复为独立可测试类
  - `__init__(batch_size)`: 预存 batch_size，校验 > 0
  - `apply(perm, drop_last)`: 纯函数语义，d==0 跳过切片，d>0 裁尾
  - NodeLoader 与 LinkLoader 共享同一实现，消除上游两处重复代码

### 改写20%（鲁迅拿法）

- 上游: 2 文件各一处裸 if 条件，逻辑相同但分散
- Walpurgis: 提取为 `PermDropLastSlicer.apply()`，单一职责，独立测试
- `__init__` 校验 batch_size > 0，防止 d 计算出现除零静默行为

### 调试断点 (WALPURGIS_DEBUG=1)

1. `PermDropLastSlicer.__init__`: batch_size 记录
2. `PermDropLastSlicer.apply` 入口: perm.numel(), batch_size, d 值
3. `PermDropLastSlicer.apply` 决策: skip（d==0）或 slice（d>0）
4. `PermDropLastSlicer.apply` 出口: 裁剪后 perm.numel()

### Knuth 审查

1. **diff 对比源**: 上游两处 `perm = perm[:-d]` 改为 `if d > 0: perm = perm[:-d]`，
   改动最小，语义精确。Walpurgis 封装消除重复，保持与上游修复等价。

2. **用户角度 bug**: 整除场景（1024 样本 / batch_size=32 等）
   静默产生空迭代器，train_loop 跑零步，无报错无 warning，
   用户只能靠 loss 曲线异常或 acc=nan 来猜测根因，极难排查。

3. **系统角度安全**: 空 perm 导致后续 `input_id[perm]`、`node[perm]` 均为空 tensor；
   进入 CUDA 采样 kernel 时 grid_size=0 行为未定义；
   分布式场景各 rank perm 长度不一致，allreduce shape mismatch
   直接崩溃 NCCL，且错误信息指向 collective 而非根因 `perm[:-0]`。

---

## migrate 07ce63f: [FEA] Support Unified WholeGraph FeatureStore and GraphStore

- **Upstream commit**: 07ce63f (cugraph-gnn, NVIDIA, 2025-04-22)
- **Commit message**: `[FEA] Support Unified WholeGraph FeatureStore and GraphStore`
- **Upstream diff** (19 files changed, 2267 insertions, 296 deletions):
  - `tensor/__init__.py` (新增): 导出 DistTensor, DistEmbedding, DistMatrix, is_empty, empty
  - `tensor/dist_tensor.py` (新增, 545行): WholeGraph 分布式 tensor 封装，1D/2D，scatter/gather，nccl/vmm 后端
  - `tensor/dist_matrix.py` (新增, 148行): COO 格式稀疏分布式矩阵，内部用两个 DistTensor(_col, _row)
  - `tensor/utils.py` (新增, 216行): create_wg_dist_tensor / copy_host_global_tensor_to_local / has_nvlink_network / is_empty / empty
  - `data/feature_store.py`: WholeFeatureStore → FeatureStore，memory_type 废弃，__make_wg_tensor 全重构，自动 all_gather + 分片
  - `data/graph_store.py`: 新增 NewGraphStore (434行)，基于 DistMatrix，支持 SG/SNMG/MNMG
  - `data/__init__.py`: GraphStore 变为工厂函数，is_multi_gpu=True → wgth.init + NewGraphStore
  - `loader/*.py` (4文件): isinstance 检查扩展为 (GraphStore, NewGraphStore)
  - `examples/gcn_dist_mnmg.py`: 废弃 wg_mem_type/in_memory 参数，WholeFeatureStore → FeatureStore
  - `examples/rgcn_link_class_mnmg.py`: 去掉磁盘分区，改用 WholeGraph 广播，节点 0 持有完整数据 → all ranks
  - `tests/`: 全部补 os.environ["LOCAL_WORLD_SIZE"]，新增 test_dist_tensor_mg.py / test_dist_matrix_mg.py

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 07ce63f | Walpurgis 迁移 |
     |---|---|
     | `__backend = "vmm" if ... else "nccl"` 散落在两处 __init__ | `BackendSelector.select()` 统一封装，携带 reason 字段 |
     | `__features = {}` / `__edge_indices = {}` 裸 dict | `UnifiedStoreRegistry` 带 debug 打印 + 键格式校验 |
     | `__make_wg_tensor` 里 `if dim==1 / elif dim==2` 内嵌分支 | `TensorDimStrategy.build()` 携带 dim_type + 构造参数 |
     | `_encode_dtype/_decode_dtype` 内嵌函数 + 手工 all_gather | `DtypeNegotiator.encode/decode/negotiate()` 可单独测试 |
     | `GraphStore()` 工厂函数内联 path 判断 | `FeatureStoreFactory.resolve()` 返回 (path, store)，路径可观测 |
     | `__get_edgelist()` 私有方法，逻辑混合偏移计算 | `EdgelistBuilder.build()` 静态方法，断点打印各 edge-type 局部边数 |
     | 无 WALPURGIS_DEBUG 断点 | 14 个断点，覆盖后端选择/dtype协商/维度决策/scatter/edgelist构建 |

  2. **用户角度 bug**:
     - `int(os.environ["LOCAL_WORLD_SIZE"])` 在裸 `python` 运行时抛 `KeyError`，
       而非友好提示"请用 torchrun 运行"。测试代码通过 `os.environ["LOCAL_WORLD_SIZE"]=str(world_size)`
       绕过，但生产用户容易踩坑。`BackendSelector.from_env()` 在 KeyError 时
       可包装为更清晰的错误信息。
     - `FeatureStore._put_tensor` 对 `attr.index` 的处理：
       `if attr.is_set("index") and attr.index is not None` 与旧代码
       `if attr.is_set("index")` 存在语义差异——旧代码 index 设为 None 仍触发分支，
       新代码要求 index 非 None。若用户显式传 `attr.index=None`，
       新代码走整体 scatter 路径（正确），旧代码走异常/忽略路径，行为改变是有意设计。
     - `rgcn` 例子用 `empty(dim=2)` 给空 rank 传占位符，依赖 `is_empty()` 检测，
       若下游不识别空 tensor 形状 `(0, 1)` 可能静默错误。

  3. **系统角度安全**:
     - `NewGraphStore.__get_edgelist` 按 `sorted(keys)` 排序 edge-type，
       依赖 Python tuple 字典序，需保证所有 rank 排序结果完全一致；
       若不同 rank 注册了不同的 edge-type 子集，sorted_keys 可能不同，
       导致 edge_type_array 编号不一致 → graph 构建静默错误，属隐患。
     - `_num_vertices` 的 `edge_attr.edge_type[2]` 应为目标顶点类型，
       但在 `else` 分支里出现 `num_vertices[edge_attr.edge_type[1]]`（关系类型名）
       而非 `[2]`（目标类型），疑似笔误，上游代码 L540 可能是 bug：
       `num_vertices[edge_attr.edge_type[1]]` 用关系名作顶点类型键，
       后续 `_vertex_offsets` 排序时会引入非顶点类型的键，导致偏移计算混乱。
     - `has_nvlink_network()` 内部调用 `wgth.comm.get_global_communicator("nccl")`，
       若 WholeGraph 尚未初始化会抛异常，而此函数在 `__init__` 中被调用，
       即 NewGraphStore 构造前必须先初始化 WholeGraph，否则构造函数崩溃；
       工厂函数 `GraphStore(is_multi_gpu=True)` 已在构造前调用 `wgth.initialize.init()`，
       但直接实例化 `NewGraphStore()` 则没有此保护。

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/unified_store.py`** — 新增

**迁移要点**:
- `BackendSelector`: `select(local_world_size, world_size)` 封装 vmm/nccl 决策，
  携带 reason 字段，`from_env()` 自动读取环境变量
- `UnifiedStoreRegistry`: 封装 `__features` / `__edge_indices` dict，
  put/get/remove 带 debug 打印
- `TensorDimStrategy`: `build(tensor_dim, global_row_count, trailing_dim)` 封装
  1D→DistTensor / 2D→DistEmbedding 分支，`instantiate()` 创建实例
- `DtypeNegotiator`: `encode/decode/negotiate()` 封装 all_gather dtype 协商，
  空 rank 过滤（sizes>0 mask）
- `FeatureStoreFactory`: `resolve(is_multi_gpu, args, kwargs)` 封装工厂三条路径，
  返回 (path_name, store_instance)
- `EdgelistBuilder`: `build(edge_indices, vertex_offsets, is_multi_gpu, weight_attr)` 静态方法，
  按 sorted edge-type 构造 {src/dst/eid/etp/wgt} dict，注明 PyG vs cuGraph src/dst 约定反转
- 14 个 WALPURGIS_DEBUG=1 断点，覆盖全链路:
  断点1-2: BackendSelector 决策 →
  断点3-5: UnifiedStoreRegistry put/get/remove →
  断点6-8: TensorDimStrategy build/instantiate →
  断点9-10: DtypeNegotiator encode/negotiate →
  断点11-12: FeatureStoreFactory resolve 路径选择 →
  断点13-14: EdgelistBuilder 各 edge-type sizes + 最终 shape/eid range

## migrate 757992f: Improving the taobao_mnmg example

- **Upstream commit**: 757992f (cugraph-gnn, NVIDIA, 2025)
- **Commit message**: `Improving the taobao_mnmg example`
- **Upstream diff** (1 file changed, 6 insertions, 3 deletions):
  - `python/cugraph-pyg/cugraph_pyg/examples/taobao_mnmg.py`:
    - `train()` 新增 `max_iter=None` 参数，循环内 `if max_iter is not None and i >= max_iter: break`
    - 进度打印 `{time}` → `{time:.4f}` 精度 4 位
    - `total_loss += float(loss)` → `float(loss.detach())` 切断计算图，修复隐式显存泄漏
    - `argparse` 新增 `--max_iter` 参数
    - 主循环 `train(model, optimizer, train_loader)` → `train(..., args.max_iter)`

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 757992f | Walpurgis 迁移 |
     |---|---|
     | `train(..., max_iter=None)` | `train(..., cfg: TrainConfig)` |
     | `if max_iter is not None and i >= max_iter: break` 内联 | `IterGuard(cfg.max_iter).should_stop(i)` 封装 |
     | `float(loss.detach())` 裸调 | `safe_loss_item(loss)` 带值域断言 [0, +inf) |
     | `return total_loss / total_examples` 标量 | `return TrainResult` 数据类，保留各分量 |
     | `argparse.Namespace` 裸访问 | `TrainConfig` dataclass + `__post_init__` 校验 |
     | `:.4f` 进度打印直接 print | `_dbg()` 统一调试出口，`WALPURGIS_DEBUG=1` 开关 |

  2. **用户角度 bug**:
     - `--max_iter 0` 时上游行为：`i=0` 立即 break，`total_examples=0`，
       `return 0/0` → `ZeroDivisionError`，无任何提示。
       `TrainConfig.__post_init__` 增加 `max_iter >= 1 或 None` 校验。
     - `loss.detach()` 后若产生 `NaN`（学习率过大 + 梯度爆炸），
       `total_loss` 累加 `NaN`，最终打印 `Loss: nan`，上游无报警。
       `safe_loss_item()` 每步断言 `0 <= v < inf`，提前定位发散。
     - `max_iter` 分布式一致性：各 rank batch 数量不同时，
       rank-A 提前截断而 rank-B 仍在迭代，DDP AllReduce 导致 NCCL 超时。
       上游同样有此隐患，`IterGuard` TODO 注明修复思路（截断前广播 stop 信号）。

  3. **系统角度安全**:
     - `float(loss)` 隐式持有整个前向计算图，每步累加一个图节点引用，
       epoch 越长显存线性增长，最终 OOM。`loss.detach()` 零拷贝断开 `grad_fn`，
       GC 在每步 `backward()` 后立即回收，是大规模分布式训练的必要做法。
     - `safe_loss_item()` 值域断言 `[0, +inf)` 保证 NaN/Inf 不静默进入累加器。

### Walpurgis 迁移位置

**文件: `src/walpurgis/examples/taobao/taobao_mnmg.py`** — 新增

**迁移要点**:
- `TrainConfig`: dataclass 封装超参，`__post_init__` 校验 `max_iter >= 1 or None`，`lr > 0`，`epochs >= 1`
- `TrainResult`: dataclass 封装单 epoch 结果，`.avg_loss` property 含 ZeroDivisionError guard
- `IterGuard`: 封装 max_iter 截断逻辑，`should_stop(i)` 可单元测试，TODO 注明分布式一致性隐患
- `safe_loss_item()`: `loss.detach()` + 值域断言 `[0, +inf)`，NaN/Inf 立即 `RuntimeError`
- `_dbg()`: `WALPURGIS_DEBUG=1` 调试出口，覆盖全链路断点 print

## migrate 18222fa: Remove inheritance from deprecated unary_function

- **Upstream commit**: 18222fa (cugraph-gnn, NVIDIA, 2025-11-10)
- **Commit message**: `Remove inheritance from deprecated unary_function`
- **Upstream diff** (1 file changed, 4 insertions, 7 deletions):
  - `cpp/src/wholememory_ops/register.hpp`:
    - 版权年从 2024 → 2025
    - `one_wmt_hash`: 移除 `: public std::unary_function<wholememory_dtype_t, std::size_t>`
    - `two_wmt_hash`: 移除 `: public std::unary_function<std::tuple<dtype,dtype>, std::size_t>`
    - `three_wmt_hash`: 移除跨行三行 `: public std::unary_function<tuple<dtype,dtype,dtype>, size_t>`
    - `operator()` 签名与哈希逻辑完全不变，只去掉继承，无功能差异

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 18222fa | Walpurgis 迁移 |
     |---|---|
     | `struct one_wmt_hash {` 裸 struct，无标记 | `// [WMT-HASH-POLICY]` 注释标记，grep 可定位 |
     | `return static_cast<size_t>(k)` 裸 cast | 同 + `_WMT_DBG_HASH_1(k)` 断点宏前置 |
     | REGISTER 宏内 `emplace` 直接调用 | `fprintf(stderr, "[WALPURGIS][REGISTER_N]...")` 断点 print |
     | `static_cast<size_t>(k)` 无类型宽度保证 | `static_assert(sizeof(dtype) <= sizeof(size_t))` 防截断 |
     | `#include <functional>` 隐式依赖（旧版） | 移除后 `<unordered_map>` 已足够，include 更干净 |

  2. **用户角度 bug**:
     - `std::unary_function` 在 C++17 被彻底移除（不只是弃用）。
       编译器升到 GCC 13 / Clang 16 + `-std=c++17` 后，
       包含旧版 register.hpp 的翻译单元**直接编译失败**，
       错误信息指向 `<functional>` 内部模板实例化，
       与 register.hpp 的业务逻辑毫无关联，难以定位。
     - 新代码无继承依赖，`-std=c++11/14/17/20` 均合法通过。
     - `WHOLEMEMORY_DT_COUNT` 枚举值将来若超过 `size_t` 上界（理论上），
       旧 `static_cast<size_t>(k)` 静默截断产生哈希碰撞，
       `unordered_map::find` 返回错误条目，`DISPATCH_*` 宏触发
       `WHOLEMEMORY_CHECK_NOTHROW` 失败但报错键值已错位，
       `static_assert` 在编译期暴露此风险。

  3. **系统角度安全**:
     - `std::unary_function` 提供 `argument_type` / `result_type` 两个 typedef，
       `std::unordered_map` 的 Hash named requirement **不需要**这两个 typedef，
       继承纯属历史包袱（C++98 时代 `std::bind1st`/`bind2nd` 需要）。
       移除后不影响 `unordered_map` 任何功能路径。
     - `__attribute__((constructor))` 注册函数在动态链接时自动调用，
       若注册过程抛异常（`WHOLEMEMORY_FAIL_NOTHROW` 调用 `abort`），
       整个进程在 `main()` 之前终止。
       `static_assert` 在编译期而非运行期拦截 dtype 宽度问题，
       避免 `.so` 加载时 abort 难以定位根因。
     - 三个 `fprintf(stderr, "[WALPURGIS][REGISTER_N]...")` 断点 print
       在 `WALPURGIS_DEBUG_HASH` 未定义时由预处理器消除，
       生产路径零开销，不改变分发表注册时序和内存布局。

### Walpurgis 迁移位置

**文件: `src/wholememory_ops/register.hpp`** — 新增

**迁移要点**:
- `one_wmt_hash` / `two_wmt_hash` / `three_wmt_hash`: 移除 `std::unary_function` 继承，`operator()` 逻辑原样保留
- `[WMT-HASH-POLICY]` 注释标记三个 hash functor，统一 grep 入口
- `_WMT_DBG_HASH_{1,2,3}` 宏：`WALPURGIS_DEBUG_HASH=1` 时 `fprintf(stderr)` 打印 dtype 入参，追踪 lookup miss
- `[断点]` 注释标注每个调试打印位置，方便阅读时定位
- `REGISTER_DISPATCH_{ONE,TWO,THREE}_TYPES` 宏内增加 `static_assert(sizeof(dtype) <= sizeof(size_t))` 防截断
- 注册 helper `Register##NAME##Map{1,2,3}FuncHelper0` 内增加 `fprintf` 断点，打印每个 dtype 注册事件
- 文件头部设计注释（鲁迅拿法）：参照 slab_allocator.hpp 风格，标注上游改写原因、cppreference 参考、Hash named requirement 依据

**改写 20%（鲁迅拿法）**:
- `[WMT-HASH-POLICY]` 标记 + 文件头 design rationale 注释替代裸 struct 定义
- `_WMT_DBG_HASH_{1,2,3}` 条件编译调试宏替代上游无调试能力
- `REGISTER_DISPATCH_*` 宏内 `fprintf` 断点 print 替代裸 emplace
- `static_assert` 编译期 dtype 宽度防御替代裸 cast
- 注释标注 `std::unary_function` 弃用原因 + C++17 删除时间线

---

## migrate d43e6c1: [BUG] Fix warnings, fix MNMG graph store test, Matrix Accessors

- **Upstream commit**: d43e6c1 (cugraph-gnn, NVIDIA, 2025-2026)
- **Commit message**: `[BUG] Fix warnings, fix MNMG graph store test, Matrix Accessors`
- **Upstream diff** (3 files changed):
  - `python/cugraph-pyg/cugraph_pyg/tensor/dist_matrix.py`:
    - `local_col` / `local_row`: 删除 `get_local_tensor()` 调用，改为手工按 rank 切片:
      `q = sz // world_size; r = sz % world_size`
      `ix = arange(q*rank+rank, q*(rank+1)+rank+1) if rank < r else arange(q*rank+r, q*(rank+1)+r)`
    - `local_coo`: `torch.stack(self.get_local_tensor())` → `torch.stack([self.local_col, self.local_row])`
    - 版权年份 2025 → 2025-2026
  - `python/cugraph-pyg/cugraph_pyg/data/graph_store.py`:
    - `num_vertices` 4 处赋值增加 `int()` 强制转换，防止 numpy 标量污染 dict
    - `num_vertices[edge_attr.edge_type[1]]` → `num_vertices[edge_attr.edge_type[2]]`
      **关键 bug**: edge_type[1] 是关系名（如 "knows"），edge_type[2] 才是 dst 顶点类型（如 "person"）
  - `python/cugraph-pyg/cugraph_pyg/tests/data/test_graph_store_mg.py`:
    - `src/dst` 增加 `.to(torch.int64)` 强转，防止 int32 NCCL 类型不匹配
    - 验证逻辑从局部直接比较改为 `all_gather sizes → all_gather rei → concat → assert`
    - 版权年份 2024-2025 → 2024-2026

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 d43e6c1 | Walpurgis 迁移 |
     |---|---|
     | `local_col/row` 内联 `q, r, ix` 公式（重复两次）| `SlicePartitioner.compute_indices()` 策略对象，一次提取 |
     | `local_coo: torch.stack([self.local_col, self.local_row])` | 委托 `WalpurgisDistMatrix` property，复用 SlicePartitioner |
     | `num_vertices[t] = int(max(...)) if t in ... else int(...)` | `VertexCountRegistry.update_from_size()` 封装 + 类型断言 |
     | `num_vertices[edge_type[2]] = int(...)` 裸写 | `VertexCountRegistry.get_dst_type_key()` 明确提取 [2] |
     | 测试内联 all_gather 3 步逻辑 | `GatheredEdgeVerifier.gather_edge_index()` + `verify_against()` |

  2. **用户角度 bug**:
     - `edge_type[1]` vs `edge_type[2]` 是 silent typo，编译无错、运行时
       `num_vertices["knows"] = 34` 写入关系名为键，后续 `num_vertices["person"]` KeyError，
       错误信息完全看不出是哪里写错了键，排查成本极高。
     - `get_local_tensor()` 是 WholeGraph 私有 API，文档无保证，若上游改变其返回
       （如增加 padding row），`local_coo` 静默返回错误维度，GNN 训练 loss 异常
       但无报错，用户只能从 metric 下降中猜测。
     - 旧测试局部比较：rank-0 与 rank-1 各自验证自己的切片正确，但若
       all_gather 后全局顺序错位（如 edge_type[1] bug 导致 src/dst 混淆），
       两 rank 都通过，bug 静默进入生产。

  3. **系统角度安全**:
     - `int()` 强制转换是 numpy 2.x 兼容的必要防御：`numpy.int64.__hash__` 在 2.x 中
       与 `int.__hash__` 不同，dict 键查找 `num_vertices["person"]` 在 3.9+ numpy 2.x
       下可能找不到由 `numpy.int64("person")` 写入的键（虽然字符串键此处无 numpy 问题，
       但值若是 numpy 标量则 max() 返回 numpy.int64，后续 `num_vertices[t] = numpy.int64(34)`
       与 `if num_vertices[t] > 0` 的 Python int 比较在极端情况下行为改变）。
     - 手工切片公式数学可证（总长 = r*(q+1) + (world_size-r)*q = q*world_size + r = sz ✓），
       与 WholeGraph 版本升级解耦，是更安全的长期做法。

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/dist_matrix.py`** — 新增

**迁移要点**:
- `SlicePartitioner`: `compute_indices(sz, rank, world_size)` 提取 d43e6c1 切片公式，
  `slice_tensor()` 一步到位，消除 local_col / local_row 的代码重复
- `VertexCountRegistry`: `update_from_size()` 封装 int() 转换 + max 更新，
  `get_dst_type_key()` 明确提取 edge_type[2] 防止 [1] 混淆，`update_from_index()` 处理边索引推断
- `GatheredEdgeVerifier`: `gather_edge_index()` 封装 all_gather 三步，
  `verify_against()` 全局断言，替代旧测试局部比较盲区
- `WalpurgisDistMatrix`: `local_col` / `local_row` / `local_coo` property 委托 SlicePartitioner，
  语义清晰，与上游 DistMatrix 接口兼容
- 12 个 WALPURGIS_DEBUG=1 断点，覆盖全链路:
  断点1-3: SlicePartitioner 切片计算 →
  断点4-6: VertexCountRegistry 类型安全更新 + dst_key 提取 →
  断点7-9: GatheredEdgeVerifier all_gather + 全局验证 →
  断点10-12: WalpurgisDistMatrix local_col / local_row / local_coo property

---

## migrate 2d545b9: Deprecate TensorDictFeatureStore in cuGraph-PyG

- **Upstream commit**: 2d545b9 (cugraph-gnn, NVIDIA, 2025)
- **Commit message**: `Deprecate TensorDictFeatureStore in cuGraph-PyG`
- **Upstream diff** (1 file changed):
  - `python/cugraph-pyg/cugraph_pyg/data/__init__.py` — 12 行新增，1 行替换
    - 原裸 import `TensorDictFeatureStore` 改为 `TensorDictFeatureStore as DEPRECATED__TensorDictFeatureStore`
    - 新增 wrapper 函数 `TensorDictFeatureStore(*args, **kwargs)`:
      触发 FutureWarning("TensorDictFeatureStore is deprecated. Consider changing your
      workflow to launch using 'torchrun' and store data in the faster and more
      memory-efficient WholeFeatureStore instead.")，透传构造 `DEPRECATED__TensorDictFeatureStore`
    - 设计目标: 统一 API 战略（Unified API / Project #159）要求所有特征存储走
      WholeGraph；TensorDictFeatureStore 是单机 dict 存储，不再作为可选路径维护

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 2d545b9                              | Walpurgis 迁移                               |
     |------------------------------------------|----------------------------------------------|
     | wrapper 函数，无调试信息，无调用统计       | `DeprecationGate` 类，call_count + 断点 print |
     | DEPRECATED__ 别名直接暴露在 module 命名空间| `_DEPRECATED__` 封装在模块内，不外泄          |
     | isinstance(obj, TensorDictFeatureStore) 崩溃 | `InstanceCheckGuard` 修补 isinstance        |
     | 每个废弃类重复 warnings.warn 模板          | `DeprecationPolicy.register()` 统一注册      |
     | __init__.py 内联 wrapper 不可单测          | `feature_store_deprecation.py` 独立可测模块  |

  2. **用户角度 bug**:
     - 上游 wrapper 函数化使 `TensorDictFeatureStore` 不再是 type，
       `isinstance(my_store, TensorDictFeatureStore)` 会抛 `TypeError`，
       这是 silent breaking change（用户类型检查代码无警告直接 crash）；
       `InstanceCheckGuard` 提供修补路径，`isinstance(obj, InstanceCheckGuard(gate))` 安全透传
     - `FutureWarning` 被 Python 默认 warning filter 静默（非 `-W error` 时不可见），
       用户可能完全看不到废弃提示；`DeprecationGate` 同时写 WALPURGIS_DEBUG stderr，
       调试模式下强制可见
     - wrapper 函数每次调用均触发 `warnings.warn`，Python filter 策略 `once/location`
       意味着同一调用点只显示一次；多处构造 TensorDictFeatureStore 的代码
       可能只看到第一处警告，`call_count` 暴露真实调用频次

  3. **系统角度安全**:
     - wrapper 函数与 `import as` 别名共享 module 命名空间，mypy/pyright
       静态分析会报 `TensorDictFeatureStore is not a type`，影响 type annotation；
       `DeprecationGate` 通过 `__name__`/`__qualname__` 属性模拟类名，减轻静态分析影响
     - 多进程 spawn 模式下 wrapper 函数 pickle 时查找的是 module 全局名称，
       `TensorDictFeatureStore` 指向函数而非类，pickle 后 unpickle 会找到 wrapper
       而非原始类，若子进程直接调用会再次触发 FutureWarning（double-warn）；
       这是上游已知的废弃期 trade-off，`DeprecationGate` 保持同等行为，
       `call_count` 可帮助诊断多进程场景下的调用频次异常

### Walpurgis 迁移位置

**文件: `src/walpurgis/core/feature_store_deprecation.py`** — 新增

**迁移要点**:
- `DeprecationGate`: `_wrapped_cls` + `_warning_msg` + `call_count`，
  `__call__` 等同上游 wrapper 函数，额外 call_count 统计 + 断点 print；
  `get_wrapped_class()` 为 isinstance 检查提供逃生舱
- `InstanceCheckGuard`: 修补 wrapper 函数化破坏 `isinstance` 的问题，
  `__instancecheck__` 委托给 `_wrapped_cls`，完全透明
- `DeprecationPolicy`: `register(name, cls, msg)` 统一注册，
  `get(name)` 返回 `DeprecationGate`，`call_all_summary()` 批量统计；
  替代 `__init__.py` 中重复的 `warnings.warn` 模板代码
- 模块级 `_POLICY` 实例预注册 `TensorDictFeatureStore`（上游 2d545b9 唯一变更），
  ImportError 时降级为 stub 模式，保持模块可导入（适用纯 Walpurgis 环境）
- 6 个 WALPURGIS_DEBUG=1 断点，覆盖全链路:
  断点1: DeprecationGate.__call__ 入口（参数类型摘要）→
  断点2: FutureWarning 触发时机（call_count）→
  断点3: 构造完成（result_type + id）→
  断点4: DeprecationPolicy.register（注册事件）→
  断点5: DeprecationPolicy.get（查找事件）→
  断点6: DeprecationPolicy.call_all_summary（批量调用统计柱状图）

## migrate 0c1b857: [SKIP] remove docs — 纯文档删除，无迁移价值

**上游 commit**: `0c1b857` (Alexandria Barghi, abarghi@nvidia.com)
**上游描述**: remove docs

**diff 分析**:
- 变更范围: 1854 个文件，508700 行删除，0 行新增
- 全部位于 `docs/cugraph/` 目录下
- 删除内容分三类:
  1. `docs/cugraph/build/doctrees/` — Sphinx 构建产物，`.doctree` 二进制缓存文件（cugraph/cugraph-dgl/cugraph-pyg/wholegraph/cugraph-ops/plc 各模块 API 文档树）
  2. `docs/cugraph/libwholegraph/*.xml` — Doxygen 生成的 C++ 源码 XML，含 wholegraph CSR 采样、WholeMemory 内存管理 API 的完整符号树（`wholememory_8h.xml`、`wholegraph__op_8h.xml` 等约百个文件）
  3. `docs/cugraph/source/` — Sphinx RST/MD 源文档，含 wholegraph 安装指南、API 索引、配置说明

**Knuth 审查**:
1. **diff 对比源**: 无任何 `.py`/`.cu`/`.cpp`/`.h` 源码改动，全部是文档产物
2. **用户角度 bug**: 无运行时影响，文档删除不影响任何功能行为
3. **系统安全**: 无安全隐患，删除本地 docs 目录不影响库的正确性

**迁移决策**: SKIP — 纯文档目录清理，Walpurgis 有独立文档体系，无任何可迁移内容

## migrate cf71bc7: [SKIP] remove pycache — 纯编译缓存清理，无迁移价值

**上游 commit**: `cf71bc7` (remove pycache)

**diff 分析**:
- 变更范围: 共 47 个文件，全部删除，0 行新增
- 全部位于 `python/cugraph-dgl/` 和 `python/cugraph-pyg/` 的各级 `__pycache__/` 目录
- 删除内容: CPython 3.11 编译生成的 `.pyc` 字节码缓存文件，以及 pytest-8.2.0 生成的测试 `.pyc`
  - `cugraph-dgl`: `__init__`、`convert`、`cugraph_storage`、`dataloading`、`nn/conv/base`、`utils/*` 等模块的 `.pyc`
  - `cugraph-pyg`: `__init__`、`_version`、`data/*`、`loader/*`、`nn/conv/*`（gat/gatv2/hetero_gat/rgcn/sage/transformer）、`sampler/*`、全套 `tests/**/*.pyc`

**Knuth 审查**:
1. **diff 对比源**: 无任何 `.py`/`.cu`/`.cpp` 源码改动，全部是二进制构建产物；`.gitignore` 应早已屏蔽 `__pycache__/`，此 commit 是补救性清理
2. **用户角度 bug**: 无运行时影响；但若 `.gitignore` 未同步修复，开发者每次运行 Python 后这些文件会重新污染 `git status`，制造持续噪音
3. **系统安全**: `.pyc` 文件头部嵌有编译时绝对路径（开发者本机路径/用户名），入库属轻微信息泄露；此 commit 正确清除，安全性略有提升

**迁移决策**: SKIP — 纯 `__pycache__` 清理，Walpurgis 项目无此类缓存文件入库问题，无任何可迁移内容

## migrate 62d487d: [SKIP] copy gitignore — 上游基础设施产物，与 Walpurgis 项目无关

**上游 commit**: `62d487d` (Alexandria Barghi, abarghi@nvidia.com)
**上游描述**: copy gitignore

**diff 分析**:
- 变更范围: 1 个文件，107 行新增，0 行删除
- 全部位于 `.gitignore`，为 cugraph-gnn 仓库第一个 commit（从 `/dev/null` 新建）
- 内容来自 RAPIDS cugraph 主仓库 `.gitignore`，面向 C++/Python 混合构建体系：
  1. **C++ 构建产物**: `CMakeFiles/`、`cpp/build/`、`*.a`/`*.o`/`*.so`/`*.dylib`、IPC 生成头文件
  2. **Python 构建产物**: `_skbuild/`、`pylibcugraph.egg-info`、`python/cugraph/bindings/*.cpp`、`dask-worker-space/`
  3. **文档产物**: `docs/cugraph/lib*`、`docs/cugraph/api/*`、`cpp/doxygen/html`
  4. **数据集白名单**: `datasets/*` + `!datasets/cyber.csv` 等 5 个例外（cugraph 专属数据集名）
  5. **IDE/OS 通用**: `.vscode`、`.idea/`、`.DS_Store`、`*.swp` 等

**Knuth 审查**:
1. **diff 对比源**:
   | 上游 62d487d `.gitignore` | Walpurgis 现有 `.gitignore` |
   |---|---|
   | 包含 `CMakeFiles/`、`cpp/build/`、`*.dylib`（C++ 构建产物）| Walpurgis 纯 Python 项目，无 CMake/C++ 构建，这些规则全部无效 |
   | 包含 `python/cugraph/bindings/*.cpp`、`pylibcugraph.egg-info`（cugraph 专属路径）| Walpurgis 无 `cugraph` 子包路径，规则永远不会匹配 |
   | `datasets/*` + 白名单仅含 cugraph 专属 CSV（`cyber.csv`、`karate-data.csv`）| Walpurgis 有 `!datasets/sensor_graph/*.pkl`、`!datasets/SYNTH/*.pkl` 例外，覆盖真实数据集文件 |
   | 无 `*.pt`/`*.pth`/`*.pkl`/`*.h5` 等 ML 模型/数据规则 | Walpurgis 已有完整模型检查点 + 大数据文件忽略规则 |
   | 无 `submodel_*.txt`、`.claude-hk-config`（Walpurgis 运行产物）| 已在 Walpurgis `.gitignore` 中覆盖 |

2. **用户角度 bug**:
   - 若直接合并上游 `.gitignore`，`datasets/*` 规则覆盖 Walpurgis 现有白名单；
     `!datasets/sensor_graph/*.pkl` 例外需在新文件中重复声明，否则 `.pkl` 传感器图文件将被意外忽略，
     导致 `git status` 不显示这些关键数据文件的变更，调试数据版本问题时极难发现
   - 上游白名单 `!datasets/cyber.csv` 等 5 个文件名在 Walpurgis 项目中不存在，
     这些例外规则形同噪音，增加未来维护者的认知负担
   - `*.diff`/`*.orig`/`*.rej`（Patching 分类）在 Walpurgis 迁移工作流中 `.diff` 文件有时需要暂存检查，
     若合并上游规则会导致迁移 patch 文件被 git 忽略，污染迁移工作流

3. **系统角度安全**:
   - 纯 `.gitignore` 配置，无运行时代码，无安全影响
   - 唯一潜在风险：若合并后 `.gitignore` 规则冲突导致敏感文件（如含 API key 的配置文件）
     意外不被忽略而提交入库——但 Walpurgis 现有规则已覆盖 `.claude-hk-config`，风险低

**迁移决策**: SKIP — 上游 `.gitignore` 是 RAPIDS cugraph C++/GPU 全栈基础设施的历史产物，
与 Walpurgis 纯 Python GNN 研究项目技术栈完全不匹配。Walpurgis 已有针对自身项目特征
（ML 模型检查点、传感器图数据集、实验日志、迁移工具产物）精心设计的 `.gitignore`，
强行合并上游规则只会引入无效噪音并破坏现有数据集白名单逻辑。无任何可迁移内容。

## migrate 793bc03: [SKIP] update readme — 上游纯文档删除，Walpurgis 无对应结构

**上游 commit**: `793bc03` (Alexandria Barghi, abarghi@nvidia.com)  
**上游描述**: update readme  
**日期**: 2024-06-11

**diff 分析**:
- 变更范围: 6 个文件，188 行删除，0 行新增
- 全部位于 `readme_pages/` 目录，无任何 `.py`/`.cu`/`.cpp` 源码改动
- 删除文件清单:
  1. `readme_pages/TRANSITIONGUIDE.md` — cuGraph 0.11/0.12 Python/C++ API 迁移指南（67 行）
  2. `readme_pages/cugraph_ops.md` — cuGraphOps 闭源 GNN 原语库介绍（17 行）
  3. `readme_pages/cugraph_python.md` — cuGraph Python 包概述，含顶点 ID 重编号说明（24 行）
  4. `readme_pages/cugraph_service.md` — Graph-as-a-Service 目标与架构图（28 行）
  5. `readme_pages/data_types.md` — cuGraph 支持数据类型列表（46 行），末尾有空白节点 `##`（未完成内容）
  6. `readme_pages/libcugraph.md` — libcugraph C/C++ 底层库简介（6 行）

**Knuth 审查**:
1. **diff 对比源**: 6 个文件均为纯 Markdown 文档，内容面向 cuGraph 0.11/0.12 时代 CUDA C++ 用户；Walpurgis 项目无 `readme_pages/` 目录，无 cuGraph Python/C++ API 封装层，无 `gdf_column`/`cugraph::Graph` 等 C++ 类型；所删文档与 Walpurgis 代码库在结构和技术层面均无交集
2. **用户角度 bug**: 纯文档删除，无运行时代码，不影响任何用户行为；`data_types.md` 中存在空白节点 `## NetworkX Graph Objects`（无内容）和裸 `##`（无标题无内容）——上游删除这些未完成内容是合理的技术债清理，与 Walpurgis 无关
3. **系统角度安全**: 无安全影响；文档中无硬编码密钥、路径、凭证或敏感信息；删除操作仅减少冗余历史文档

**迁移决策**: SKIP — 纯 `readme_pages/` 文档删除，涉及 cuGraph 0.x 时代 C++ API 过渡说明，与 Walpurgis 时空图 GNN 研究项目无任何关联，无可迁移内容

## migrate f19fc8c: [SKIP] remove build/ci for packages — 纯 CI/conda 基础设施删除，Walpurgis 无对应结构

**上游 commit**: `f19fc8c` (Alexandria Barghi, abarghi@nvidia.com)  
**上游描述**: remove build/ci for packages  
**日期**: 2024-06-11

**diff 分析**:
- 变更范围: 36 个文件，1063 行全量删除，0 行新增
- 两类删除：
  1. `ci/` 目录下 16 个 shell 脚本 — `build_cpp.sh`、`build_docs.sh`、`build_wheel_*.sh`、`run_ctests.sh`、`run_*_pytests.sh`、`test_wheel_*.sh`、`wheel_smoke_test_*.py`，全部是 RAPIDS CI 流水线的 conda/wheel 构建与测试入口
  2. `conda/recipes/` 下 20 个文件 — `libcugraph`、`cugraph`、`pylibcugraph`、`nx-cugraph`、`cugraph-equivariant`、`cugraph-service` 六个 conda 包的 `meta.yaml`、`build.sh`、`conda_build_config.yaml`、安装脚本，包含大量 CUDA 版本矩阵（CUDA 11/12 双分支）、`sccache` S3 配置、`rapids-conda-retry mambabuild` 调用

**Knuth 三维审查**:

1. **diff 对比源**: 所有被删文件均为 RAPIDS cugraph 全栈基础设施（C++/CUDA 多 GPU 图算法库）的 CI 运维脚本，涵盖：`rapids-configure-conda-channels`、`rapids-generate-version`、`rapids-upload-conda-to-s3`、CMake Ninja 构建、openmpi 多 GPU 测试、DGL/PyG conda channel 配置。Walpurgis 是纯 Python 时空图 GNN 研究项目，无 `ci/` 目录，无 `conda/recipes/`，无 C++ 编译步骤，无 RAPIDS 版本管理体系；`wheel_smoke_test_nx-cugraph.py` 中的 `nxcg.betweenness_centrality` 与 Walpurgis 的 `WalpurgisModel` 无任何调用关系；`wheel_smoke_test_pylibcugraph.py` 中的 `SGGraph`/`pagerank` cuPy 测试与 Walpurgis PyG 数据流无交集

2. **用户角度 bug**: 纯基础设施删除，无运行时逻辑，不影响任何用户可感知行为。`build_docs.sh` 中有一处防御性检查 `python -c "import cugraph; print(f'Using cugraph: {cugraph}')"` 保证文档构建前 import 可用——此模式在 Walpurgis 现有 `train_walpurgis.py` 已有对应实践，无需补充。`ci/run_ctests.sh` 的 `ctest --no-tests=error` flag 阻止测试目录缺失时静默通过——Walpurgis 使用 pytest，该细节不适用

3. **系统角度安全**: `ci/*.sh` 脚本中大量引用 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`、`SCCACHE_BUCKET`、`SCCACHE_REGION` 等敏感环境变量——删除操作实为正向安全收益，减少 CI 凭证暴露面。`conda/recipes/libcugraph/meta.yaml` 中 `SCCACHE_S3_KEY_PREFIX` 含架构标注注释 `# [aarch64]` / `# [linux64]`，是 conda-build 条件渲染语法而非实际路径泄露。Walpurgis 不引入任何此类 CI secret，无安全继承风险

**迁移决策**: SKIP — 36 个文件全为 RAPIDS cugraph CUDA C++ 全栈的 CI/conda 打包基础设施历史存档，与 Walpurgis 纯 Python GNN 研究项目技术栈在每一层均无交集。Walpurgis 项目无 `ci/`、无 `conda/recipes/`、无 CMake/C++ 构建、无 RAPIDS 版本管理，强行迁移任何文件均为引入无效噪音。无任何可迁移内容。
## migrate fc5c0e6: [SKIP] remove property graph page — 上游纯文档删除，Walpurgis 无对应结构

**上游 commit**: `fc5c0e6` (Alexandria Barghi, abarghi@nvidia.com)  
**上游描述**: remove property graph page  
**日期**: 2024-06-11

**diff 分析**:
- 变更范围: 1 个文件，54 行删除，0 行新增
- 文件路径: `readme_pages/property_graph.md`
- 全部为纯 Markdown 文档内容，无任何 `.py`/`.cu`/`.cpp` 源码改动

**diff 逐行深读**:

文件由三部分构成：

1. **头部图文块**（第 1-9 行）: 居中 HTML `<img>` 引用 `../img/pg_example.png`，左对齐 H1 "Property Graph"；纯展示内容，无逻辑

2. **概念描述段**（第 11-14 行）: 解释 PropertyGraph 是数据模型而非图类型，是 cuGraph 生态中封装所有图类型的 meta-graph，链接 Dataversity 外部定义。**注意**：描述中将 PropertyGraph 定性为"originally created for database systems"——这是历史准确性问题，Property Graph 概念确实来自图数据库（Neo4j 等），与 RAPIDS cuGraph GPU 加速定位有微妙语义张力，但该段只是描述性文字，无代码影响

3. **功能列举段**（第 16-22 行）: 5 个 bullet 列出 PropertyGraph 能力：
   - 多边/节点类型
   - 基于属性的子图提取
   - GPU 内存与 host 存储扩展（GNN-centric storage extensions）
   - 派生数据（分析结果）写回属性图
   - 通过 [CuGraph Service](./cugraph_service.md) 远程共享访问
   
   **注意**: 链接 `./cugraph_service.md` 指向的文件已在上游 `793bc03` 中被删除——此处形成了悬空引用。上游在 `fc5c0e6` 中删除 `property_graph.md` 本身，悬空引用随之消失，属于合理清理

4. **代码示例块**（第 24-54 行）: 完整 Python 示例，演示 PropertyGraph 两阶段分析：
   ```python
   from cugraph.experimental import PropertyGraph
   from cugraph.experimental.datasets import karate
   # 构建图 → edgelist → PropertyGraph → Louvain社区发现
   # → 分区属性写回 → extract_subgraph × 2 → pagerank × 2
   ```
   **注意**: 代码中存在两处空格排版问题：
   - `pG. add_edge_data(...)` — 方法调用 `pG.` 与 `add_edge_data` 之间有多余空格
   - `pG. select_vertices(...)` — 同上
   - `pageranks0.sort_values (by=...)` — 函数调用括号前有多余空格
   
   这些是上游文档本身的排版 bug，从未修复，随文件删除一并消除

**Knuth 审查**:
1. **diff 对比源**:
   | 上游 fc5c0e6 `readme_pages/property_graph.md` | Walpurgis 现有结构 |
   |---|---|
   | 演示 `from cugraph.experimental import PropertyGraph` | Walpurgis 无 `cugraph.experimental` 依赖；PropertyGraph 是 cuGraph 专有 API，Walpurgis 使用 PyG/DGL 图抽象 |
   | 代码示例依赖 `cudf`、`cugraph`、karate 内置数据集 | Walpurgis 使用 `torch`、`numpy`；数据集通过 `datasets/` 目录管理，无 karate 图 |
   | `readme_pages/` 目录结构 | Walpurgis 项目无 `readme_pages/` 目录，文档结构完全不同 |
   | 演示 Louvain 社区发现 → 子图提取 → PageRank 两阶段分析 | Walpurgis 核心是时空图 GNN（METR-LA 交通预测），无社区发现/PageRank 使用场景 |
   | 图片引用 `../img/pg_example.png` | Walpurgis 无 `img/` 目录，引用路径在 Walpurgis 下直接 404 |

2. **用户角度 bug**:
   - 代码示例中 `pG. add_edge_data(df, vertex_col_names=("src", "dst"))` 的多余空格在某些严格 linter（如 `pycodestyle E211`）下会报 `whitespace before '('`；文档示例代码不规范，但随文件删除消除，无遗留影响
   - 代码示例使用 `from cugraph.experimental import PropertyGraph`——`experimental` 命名空间在 cuGraph 新版本中 API 稳定性无保证；上游删除该文档可能与 `PropertyGraph` API 迁移/废弃有关（与同批次删除 `cugraph_service.md` 的模式一致，均为清理已废弃功能文档）
   - 若有用户按文档示例编写代码并已提交到下游项目，删除该页会导致链接 404；但这是上游的文档维护决策，与 Walpurgis 无关

3. **系统角度安全**:
   - 纯 Markdown 文档删除，无运行时代码，无安全影响
   - 代码示例中无硬编码密钥、API token、路径或凭证信息
   - `PropertyGraph` 示例无网络请求、无文件系统写入、无进程间通信——即使迁移该代码也无安全面扩展

**迁移决策**: SKIP — 纯 `readme_pages/property_graph.md` 文档删除，内容依赖 `cugraph.experimental.PropertyGraph` 专有 API 与 `cudf` GPU DataFrame，与 Walpurgis 时空图 GNN 项目技术栈（PyG/DGL + PyTorch + METR-LA 数据集）完全不匹配；Walpurgis 无 `readme_pages/` 目录结构，无任何可迁移内容，无需创建或删除任何文件

## migrate e9eee19: remove pylibcugraph page

- **Upstream commit**: e9eee19 (cugraph-gnn, NVIDIA)
- **Commit message**: `remove pylibcugraph page'`
- **Author**: Alexandria Barghi <abarghi@nvidia.com>
- **Date**: 2024-06-11

- **Upstream diff** (1 文件删除):
  - `readme_pages/pylibcugraph.md` — 删除，25 行：
    - cuGraph logo + 标题
    - 说明 pylibcugraph 是 cuGraph C API 的 Python 包装器，面向集成者而非终端用户/数据科学家
    - 描述与 cython 深度集成以减少 Python 层开销
    - Louvain 算法调用示例（`pylibcugraph.SGGraph` + `cupy` 构建图 → `pylibcugraph.louvain(...)` → 返回 `(vertices, clusters, modularity)`）

**Knuth 审查**:
1. **diff 对比源**:
   | 上游 e9eee19 `readme_pages/pylibcugraph.md` | Walpurgis 现有结构 |
   |---|---|
   | `readme_pages/` 子页面文档，介绍 pylibcugraph C API Python 封装 | Walpurgis 无 `readme_pages/` 目录，项目无子页面文档体系 |
   | 演示 `pylibcugraph.SGGraph` + `pylibcugraph.louvain` 直接调用 | Walpurgis 虽在采样层间接依赖 pylibcugraph，但无此直接调用路径，且无 Louvain 使用场景 |
   | 图片引用 `../img/cugraph_logo_2.png` | Walpurgis 无 `img/` 目录，引用路径在 Walpurgis 下直接 404 |
   | 内容面向 RAPIDS cuGraph 集成者，讲解 cython 绑定架构 | Walpurgis 是时空图 GNN 研究项目（METR-LA 交通预测），无文档站点，无集成者受众 |

2. **用户角度 bug**:
   - 代码示例 `pylibcugraph.louvain(resource_handle, G, 100, 1., False)` 中第 3-5 参数（`max_level=100`、`threshold=1.`、`resolution_param=False`）最后一个参数传入布尔值 `False` 给本应为 float 型的 `resolution_param`，属于上游文档示例 bug；随文件删除消除
   - `store_transposed=True` 但 Louvain 需要对称图，转置对无向图无影响但对有向图静默改变语义——上游文档未说明此约束，删除后不再误导用户

3. **系统角度安全**:
   - 纯 Markdown 文档删除，无运行时代码，无安全影响
   - 代码示例无硬编码密钥、token、凭证或危险系统调用
   - 此 commit 属于与 `fc5c0e6`（property_graph.md 删除）同批次的 `readme_pages/` 文档清理，模式一致：整个 `readme_pages/` 目录逐步移除

**迁移决策**: SKIP — 纯 `readme_pages/pylibcugraph.md` 文档删除；Walpurgis 无 `readme_pages/` 目录结构，无任何可迁移或可删除内容；pylibcugraph 文档面向 RAPIDS cuGraph 集成者受众，与 Walpurgis 时空图 GNN 项目（PyG + PyTorch + METR-LA）技术栈及使用场景完全不匹配；强行创建再删除将制造纯噪音 commit，无任何工程价值

## migrate f6cd9c6: [SKIP] Updates — 纯 build/CI 基础设施重构，Walpurgis 无对应结构

- **Upstream commit**: f6cd9c6 (cugraph-gnn, BradReesWork, 2024-07-01)
- **Commit message**: `Updates`
- **Upstream diff** (5 files changed, 99 insertions, 518 deletions):
  - `.pre-commit-config.yaml` (新增): black/flake8/yesqa/clang-format/copyright/dependency-file-generator 预提交 hook 配置
  - `build.sh`: 大幅裁剪，删除 libcugraph/libcugraph_etl/pylibcugraph/cugraph/cugraph-service/cugraph-equivariant/nx-cugraph 全部 C++ 构建逻辑及对应 cmake 调用；仅保留 cugraph-pyg + cugraph-dgl Python 包安装；默认构建目标从 libcugraph→pylibcugraph→cugraph 改为 cugraph-pyg→cugraph-dgl→wholegraph
  - `conda/environments/all_cuda-118_arch-x86_64.yaml` + `all_cuda-122_arch-x86_64.yaml`: 从头构建依赖（c-compiler/libcugraphops/libraft/openmpi/nvcc 等）全部删除，改为依赖预构建的 `cugraph==24.8.*` conda 包
  - `dependencies.yaml`: 删除 `cpp_build` 节（gcc/cuda-nvcc/libcugraphops/libraft/openmpi）；删除 py_build/py_run/py_test 中 cugraph/pylibcugraph/nx-cugraph/cugraph-equivariant/cugraph-service 六个包的全部 pyproject 配置块；新增 `depends_on_cugraph` 依赖块（conda: cugraph==24.8.*, pypi: cugraph-cu11/cu12==24.8.*）

- **Knuth 审查**:
  1. **diff 对比源**:
     - `build.sh` 变化核心语义：从「从源码编译整个 cugraph 生态」改为「假设 cugraph 已由 conda/pip 安装，仅构建 GNN Python 包层」。这是 cugraph-gnn 仓库从 cugraph 单仓库剥离后的架构调整，不含任何算法或 API 变更
     - `depends_on_cugraph` 新增块新增了 `--extra-index-url=https://pypi.nvidia.com` 的 pip 安装路径，是 NVIDIA private wheel index 配置，与 Walpurgis PyPI 依赖体系无交集
     - `.pre-commit-config.yaml` 中 `rapids-dependency-file-generator` hook 与 `dependencies.yaml` 联动，是 RAPIDS 专用依赖矩阵生成工具，Walpurgis 不使用此工具链
  2. **用户角度 bug**: 无运行时代码改动，无 bug 风险；`build.sh` 裁剪只影响开发者本地构建流程，不影响已安装的 Python 包行为；`cugraph-dgl` 构建条件 `if hasArg cugraph-dgl || buildDefault ||hasArg all` 中 `||hasArg` 前缺少空格（格式 bug），不影响 bash 语义
  3. **系统角度安全**: 无安全影响；删除 openmpi/gcc 等 C++ 构建依赖是正向安全收益（减少编译器/MPI 安装面）；`--extra-index-url=https://pypi.nvidia.com` 是生产 wheel 分发地址，需信任 NVIDIA PyPI 源，与 Walpurgis 无关

- **迁移决策**: SKIP — 纯 build/CI 基础设施重构：删除 C++ 全栈构建逻辑、裁剪 conda 依赖矩阵、新增 pre-commit 配置。Walpurgis 无 `build.sh`、无 `conda/environments/`、无 `dependencies.yaml`、无 RAPIDS CI 体系；`.pre-commit-config.yaml` 中涉及的 `rapids-dependency-file-generator` 是 RAPIDS 专有工具，不适用。无任何可迁移的算法、API 或运行时代码。

---

## db74d87 — Merge pull request #2 from alexbarghi-nv/copy-from-cugraph

**上游描述**: Copy Files From cuGraph Repository（从 rapidsai/cugraph 复制整个 cugraph-dgl 包）

**关键文件分析**:

diff 将 `python/cugraph-dgl/` 整目录（714 行 `cugraph_storage.py`、557 行 `sampling_helpers.py`、353 行 `base.py`、321 行 `dataloader.py` 等）复制进 cugraph-gnn 仓库。绝大多数文件是 DGL 专用接口（CuGraphStorage 鸭子类、DGL DataLoader 封装、DGL nn 卷积层），与 Walpurgis 时空图 GNN 技术栈不直接对应。

以下文件被判定为无价值 SKIP：
- `cugraph_storage.py` / `convert.py` / `cugraph_conversion_utils.py`：DGL 专有 Duck-typed API，Walpurgis 不使用 DGL
- `dataloader.py`（DGL）/ `dataset.py`（HomogenousBulkSamplerDataset）：DGL DataLoader 封装，依赖 `dgl.dataloading`
- `nn/conv/*.py`（GATConv/GATv2Conv/SAGEConv/RelGraphConv/TransformerConv）：DGL 卷积层，依赖 `dgl.DGLHeteroGraph`
- `conda/`/`ci/`/`build.sh`：RAPIDS 构建体系，与 Walpurgis 无关
- `mg_utils/`：Dask 集群辅助脚本，Walpurgis 使用 PyTorch DDP

**迁移文件**:

### 1. `src/walpurgis/tensor/sparse_graph.py`（新增）

**迁移源**: `python/cugraph-dgl/cugraph_dgl/nn/conv/base.py` → `SparseGraph` 类 + `compress_ids` / `decompress_ids`

**迁移价值**: `SparseGraph` 是 pylibcugraphops 算子所需的 CSC/COO/CSR 多格式稀疏图表示，与 Walpurgis `distributed_sampler.py` 中 `pylibcugraphops.pytorch.CSC` 调用路径直接对接。原版依赖 `dgl.DGLHeteroGraph` 作为替代输入，Walpurgis 迁移版移除了 `get_cugraph_ops_CSC/get_cugraph_ops_HeteroCSC`（BaseConv 部分），保留纯格式转换核心。

**20% 改写**:
- `_validate_inputs()` — 集中 6 处散落 if/raise，替代原版 `__init__` 内联校验
- `_build_csc()` — 独立 sort+compress 路径，含 DEBUG 耗时计时
- 全链路 `WALPURGIS_DEBUG=1` 断点：`__init__` → `_build_csc` → 各 lazy 属性首次计算 → `reduce_memory`

### 2. `src/walpurgis/sampler/sampling_csc_helpers.py`（新增）

**迁移源**: `python/cugraph-dgl/cugraph_dgl/dataloading/utils/sampling_helpers.py` → `_process_sampled_df_csc`、`_create_homogeneous_sparse_graphs_from_csc`、`create_homogeneous_sampled_graphs_from_dataframe_csc`

**迁移价值**: BulkSampler `compression="CSR"` 输出的 CSC 格式 DataFrame 后处理，是 `DistributedNeighborSampler` 下游的关键解包步骤。原版 `_process_sampled_df_csc` 是近 200 行单体函数，中间状态完全不可观察。

**20% 改写**:
- `_extract_csc_tensors()` — 独立从 DataFrame 提取张量 + 偏移局部化，原版内联无名
- `_build_per_batch_hop_dict()` — 独立 batch/hop 双层切片，每步 DEBUG 输出切片范围
- 公开接口 `create_homogeneous_sampled_graphs_from_dataframe_csc` 保持上游 API 兼容

### 3. `src/walpurgis/tensor/__init__.py` / `src/walpurgis/sampler/__init__.py`（修改）

暴露新增符号：`SparseGraph`, `compress_ids`, `decompress_ids`, `create_homogeneous_sampled_graphs_from_dataframe_csc`。

**Knuth 审查**:
1. **diff 对比源**: 上游 `SparseGraph` 还含 `get_cugraph_ops_CSC/HeteroCSC`（依赖 `pylibcugraphops.pytorch`）；Walpurgis 版本仅保留格式转换层，不增加新的 pylibcugraphops 调用点，与 `distributed_sampler.py` 已有的使用模式一致。`_process_sampled_df_csc` 的 DataFrame column 约定（`major_offsets`/`minors`/`label_hop_offsets`/`map`）与 BulkSampler `compression="CSR"` 输出完全兼容。
2. **用户角度 bug**: `compress_ids` 内部依赖 `torch._convert_indices_from_coo_to_csr`（PyTorch 私有 API），PyTorch 主版本升级可能 breaking；已在文档注释中标注，后续可换 `torch.sparse`。`_extract_csc_tensors` 中 `int(renumber_map_offsets[batch_id])` cupy/cuda scalar → Python int 转换对 int64 无精度损失，但若 renumber_map 行数超过 `sys.maxsize` 则溢出（实际不会发生）。
3. **系统角度安全**: 两个新文件均为纯张量计算，无网络请求、无文件系统写入、无 IPC。`sampling_csc_helpers.py` 读取 cudf.DataFrame（只读），写入 GPU 张量，无安全面扩展。

**迁移决策**: MIGRATE — 迁移 `SparseGraph` 格式转换类与 CSC 采样后处理三函数，写入 `src/walpurgis/tensor/sparse_graph.py` 和 `src/walpurgis/sampler/sampling_csc_helpers.py`。

## migrate 2776772: [Feature] Add gather/scatter support 1D tensor

- **Upstream commit**: 2776772 (cugraph-gnn, NVIDIA, PR #74)
- **Commit message**: `[Feature] Add gather/scatter support 1D tensor`
- **Upstream diff** (3 files changed):
  - `cpp/src/wholememory_ops/functions/gather_scatter_func.cu` — 1D tensor支持
  - `python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx` — Python绑定扩展
  - `python/pylibwholegraph/pylibwholegraph/torch/wholegraph_env.py` — 环境适配

- **迁移决策**: PARTIAL
- **原因**: 不迁移pylibwholegraph绑定(Walpurgis无此依赖),但将1D/2D自适应模式
  (`view(-1)` / `unsqueeze(1)`)思路应用到`src/walpurgis/tensor/`
- **Knuth审查**:
  1. diff对比源: 上游CUDA kernel增加ndim==1分支,Python绑定增加shape判断
  2. 用户bug: 1D embedding gather之前会silently reshape为[N,1]再gather,额外一次拷贝
  3. 系统安全: CUDA kernel内`if(ndim==1)`分支与2D共享同一stream,无新并发风险

## migrate a9ab8b4: [FEA] Support Heterogeneous Sampling in cuGraph-PyG

- **Upstream commit**: a9ab8b4 (cugraph-gnn, NVIDIA, PR #82)
- **Commit message**: `[FEA] Support Heterogeneous Sampling in cuGraph-PyG`
- **Upstream diff** (13 files changed, 515行大PR):
  - `sampler/sampler.py` — HeterogeneousSampleReader核心: per-edge-type解包
  - `sampler/sampler_utils.py` — 异构图负采样适配
  - `loader/neighbor_loader.py` — 异构图NeighborLoader入口
  - `data/graph_store.py` — 异构图GraphStore扩展
  - 多个测试文件

- **迁移决策**: MIGRATE — HeterogeneousSampleReader核心逻辑直接适用
- **Knuth审查**:
  1. diff对比源: 上游用PyG框架类型,改写为统一`Tuple[str,str,str]`
  2. 用户bug: per-edge-type节点偏移计算若edge_types排序不一致会静默错误
  3. 系统安全: sorted()保证跨rank确定性,但需所有rank注册相同edge_type集合
- **Walpurgis迁移位置**: 已有 `src/walpurgis/dataloader/hetero_sample_reader.py`

## migrate 3f11d45: [BUG] Fix Calculation of # Sampled Nodes/Edges with Zero Input Size

- **Upstream commit**: 3f11d45 (cugraph-gnn, NVIDIA, PR #283)
- **Commit message**: `[BUG] Fix Calculation of # Sampled Nodes/Edges with Zero Input Size`
- **Upstream diff** (1 file changed):
  - `sampler/sampler.py` — 空batch时`num_sampled_nodes/edges`计算修复

- **迁移决策**: MIGRATE — 是a9ab8b4的配套修复
- **Knuth审查**:
  1. diff对比源: 空input_nodes时`max()+1`抛IndexError,修复为`0 if empty else max()+1`
  2. 用户bug: 异构图稀疏batch中某些edge_type可能无节点,直接崩溃
  3. 系统安全: 空tensor的max()是PyTorch已知陷阱,提取为`_safe_max_plus_one()`
- **Walpurgis改写**: 提取为辅助函数,与hetero_sample_reader.py联动

## migrate b860220: [BUG] Fix input type in Taobao example

- **Upstream commit**: b860220 (cugraph-gnn, NVIDIA, PR #301)
- **Commit message**: `[BUG] Fix input type in Taobao example`
- **Upstream diff** (1 file changed):
  - `examples/taobao_mnmg.py` — edge_type参数从裸tensor改为正确的PyG CanonicalEdgeType

- **迁移决策**: PARTIAL — example层修复直接适用
- **Knuth审查**:
  1. diff对比源: 裸张量传给需要(src,rel,dst)三元组的API
  2. 用户bug: 异构图示例运行时TypeError,错误指向库内部而非调用方
  3. 系统安全: 增强Walpurgis loader层参数校验,异构图时裸张量→ValueError
- **Walpurgis迁移位置**: `src/walpurgis/examples/taobao/taobao_mnmg.py`已覆盖

## migrate 4e7a730: [SKIP] remove eggs — 纯.egg-info清理,无迁移价值

## migrate 2aa7b06: [SKIP] update readme — README文字更新,Walpurgis有独立README

## migrate acbddf2: [SKIP] Merge PR #3 — 纯合并commit,0文件

## migrate baacf73: [SKIP] resolve dependency-file-generator warning — RAPIDS CI工具配置

## migrate 996298f: [SKIP] skip CMake 3.30.0 — CMake版本pin,Walpurgis无CMake

## migrate 91b6e85: [SKIP] remove other packages from ci scripts — CI脚本清理

## migrate 02c96b9: [SKIP] fix dgl deps — DGL依赖修复,Walpurgis不用DGL

## migrate ecc22bf: [SKIP] update code — 单文件微调,无实质算法改动

## migrate 666d114: [SKIP] add codeowners — GitHub CODEOWNERS配置

## migrate fca4b79: [SKIP] split CUDA-suffixed deps — RAPIDS依赖矩阵重组

## migrate 27b9bcc: [SKIP] resolve merge conflict — 纯合并冲突解决

## migrate b8b2e76: [SKIP] update codeowners — CODEOWNERS更新

## migrate 43c26b3: [SKIP] fix typo — 单字拼写修复

## migrate 4be0724: [SKIP] update pr — PR更新,无实质改动

## migrate 770ddd4: [SKIP] remove whitespace — 空白字符清理

## migrate 3bbdbb5: [SKIP] conda — conda环境配置,Walpurgis无conda体系

## migrate f8625ce: [SKIP] Updates for v24.10 — 版本号批量更新

## migrate b9db217: [SKIP] Drop Python 3.9 support — Python版本矩阵调整

## migrate d2d9028: [SKIP] update requires-python floor — pyproject.toml版本约束

## migrate 609f725: [SKIP] Remove NumPy <2 pin — NumPy版本解pin

## migrate 3c59e99: [SKIP] upgrade target Python version for black — 格式化工具配置

## migrate 305aa8f: [SKIP] Add support for Python 3.12 — Python版本矩阵

## migrate 8f8b71f: [SKIP] Update flake8 to 7.1.1 — linter版本更新

## migrate a2e3e2c: [SKIP] Fix update-version.sh — 版本脚本修复

## migrate 2798f5e: [SKIP] update cmakelists for VERSION file — CMake配置

## migrate 74c365d: [SKIP] update-version.sh packaging lib — 版本工具

## migrate 37f8629: [SKIP] fix import order — import排序,纯格式

## migrate 429dbc1: [SKIP] add ops bot — GitHub bot配置

## migrate cb6a81f: [SKIP] add copy pr bot — GitHub bot配置

## migrate 7e3182c: [SKIP] introduce minimal CI for PRs — CI配置

## migrate 0ea17b3: [SKIP] add alpha specs, pre-commit hook — RAPIDS pre-commit

## migrate 2e0c143: [SKIP] add comment to dependencies.yaml — 注释添加

## migrate 0013186: [SKIP] Update version references in workflow (#93)
- **Upstream commit**: 0013186 (cugraph-gnn, NVIDIA, PR #93)
- **跳过原因**: 仅修改 `.github/workflows/trigger-breaking-change-alert.yaml` 中的版本号引用（1 个文件，1 处改动）。纯 CI 配置，Walpurgis 无 GitHub Actions 版本管理，无迁移价值。

## migrate cc1bab9: [SKIP] build libwholegraph docs in CI (#96)
- **Upstream commit**: cc1bab9 (cugraph-gnn, NVIDIA, PR #96)
- **跳过原因**: 纯 CI/docs 配置改动：`.github/workflows/build.yaml`、`pr.yaml`、`ci/build_docs.sh`、三个 conda env yaml。Walpurgis 无 RAPIDS CI 体系，无 libwholegraph 文档构建需求，无迁移价值。

## migrate e641496: [SKIP] fix nightly docs workflow dependencies (#97)
- **Upstream commit**: e641496 (cugraph-gnn, NVIDIA, PR #97)
- **跳过原因**: 仅修改 `.github/workflows/build.yaml` 一处（1 行）。CI docs workflow 依赖修复，Walpurgis 无此 workflow，无迁移价值。

## migrate af22a12: [SKIP] remove unused dependencies.yaml entries, other small cleanup (#98)
- **Upstream commit**: af22a12 (cugraph-gnn, NVIDIA, PR #98)
- **跳过原因**: 改动集中在 `dependencies.yaml`（删 52 行）、`ci/release/update-version.sh`、三个 conda env yaml、`cpp/CMakeLists.txt` 注释、`cpp/cmake/thirdparty/get_raft.cmake` 一行。均为 RAPIDS CI/conda/CMake 体系配置。Walpurgis 无 CMake 构建，无 conda 环境矩阵，无迁移价值。

## migrate 71675d8: [SKIP] simplify wholegraph CMake, other small building and testing changes (#102)
- **Upstream commit**: 71675d8 (cugraph-gnn, NVIDIA, PR #102)
- **跳过原因**: 修改 `cpp/CMakeLists.txt`（简化 34 行→11 行）、`python/pylibwholegraph/CMakeLists.txt`（简化 30 行→7 行）、conda recipe yaml、`ci/build_*.sh`。纯 CMake/CI/conda 构建体系简化，Walpurgis 以 Python 包形式使用，无 CMake 构建流程，无迁移价值。

## migrate bb81a18: [SKIP] remove flake8, clang tools from wholegraph CMake (#103)
- **Upstream commit**: bb81a18 (cugraph-gnn, NVIDIA, PR #103)
- **跳过原因**: 修改 `cpp/CMakeLists.txt` 删除 flake8 集成（16 行）、删除 `cpp/cmake/CodeChecker.cmake`（56 行）、修改 `pylibwholegraph/CMakeLists.txt` 删除 3 行。纯 CMake linter/静态分析工具配置删除，Walpurgis 无 CMake，无迁移价值。

## migrate e2b4cf0: [SKIP] Remove invalid conditional (#105)
- **Upstream commit**: e2b4cf0 (cugraph-gnn, NVIDIA, PR #105)
- **跳过原因**: 仅修改 `.github/workflows/test.yaml` 删除 1 行无效条件。纯 CI workflow 修复，Walpurgis 无此 workflow，无迁移价值。

## migrate e332b68: [SKIP] Fix inputs for the workflow (#106)
- **Upstream commit**: e332b68 (cugraph-gnn, NVIDIA, PR #106)
- **跳过原因**: 仅修改 `.github/workflows/test.yaml`（+4/-1 行），修复 workflow inputs。纯 CI 配置，Walpurgis 无此 workflow，无迁移价值。

## migrate 85cab72: [SKIP] Check if nightlies have succeeded recently enough (#100)
- **Upstream commit**: 85cab72 (cugraph-gnn, NVIDIA, PR #100)
- **跳过原因**: 仅修改 `.github/workflows/pr.yaml`（+13 行），新增 nightly 成功检查逻辑。纯 CI 配置，Walpurgis 无 RAPIDS nightly CI 体系，无迁移价值。

## migrate 87455cf: [SKIP] Remove Build Directory (#107)
- **Upstream commit**: 87455cf (cugraph-gnn, NVIDIA, PR #107)
- **跳过原因**: 删除 `python/cugraph-pyg/build/` 目录下 32 个文件（全部为 build artifact 残留，6699 行删除）。纯 git 仓库清理，无算法代码，无迁移价值。Walpurgis 仓库中无对应 build 目录残留。

## migrate a9ab8b4: [FEA] Support Heterogeneous Sampling in cuGraph-PyG

- **Upstream commit**: a9ab8b4 (cugraph-gnn, NVIDIA, PR #82)
- **Commit message**: `[FEA] Support Heterogeneous Sampling in cuGraph-PyG`
- **Upstream diff** (13 个文件变动，515 行增加，196 行删除):
  - `cugraph_pyg/sampler/sampler.py` — 新增 `HeterogeneousSampleReader`（207 行），
    `BaseSampler` 两处 `raise NotImplementedError` 替换为 `HeterogeneousSampleReader` 调用；
    `SampleIterator` 异构路径修复 `.items()` bug；`SampleReader` 新增 `lho_name` 兼容逻辑
  - `cugraph_pyg/data/graph_store.py` — 新增 `_vertex_offset_array` property、
    `_numeric_edge_types` property、`__numeric_edge_types = None` 初始化
  - `cugraph_pyg/loader/neighbor_loader.py` — 异构时强制 `compression="COO"`，
    禁 `directory`，向 `BaseSampler` 传入 `heterogeneous/vertex_type_offsets/num_edge_types`
  - `cugraph_pyg/loader/link_neighbor_loader.py` — 同 neighbor_loader 的异构改动
  - `cugraph_pyg/loader/node_loader.py` — `input_type!=None` 时 `input_nodes+=_vertex_offsets[input_type]`
  - `cugraph_pyg/loader/link_loader.py` — `input_type!=None` 时 `edge_label_index[0/1]+=_vertex_offsets`
  - `gcn_dist_mnmg/sg/snmg.py` — 全面移除 `tempfile.TemporaryDirectory` + `directory` 参数
  - `cugraph_dgl/dataloading/neighbor_sampler.py` — 移除 `tempfile` + `DistSampleWriter`，改 `writer=None`
  - `tests/loader/test_neighbor_loader.py` — 新增 `test_neighbor_loader_hetero_basic/single_etype`

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 a9ab8b4 | Walpurgis 迁移 |
     |---|---|
     | `HeterogeneousSampleReader.__decode_coo` 内 `input_type` 从 `raw_sample_data["input_type"]` 直接读取 | 同上，但 `_safe_max_plus_one()` 替代 3 处裸 `x.max()+1`，空 tensor 安全 |
     | `BaseSampler._choose_reader` 逻辑在 `sample_from_nodes/edges` 中重复 2 次 | 提取为 `_choose_reader()` 私有方法，决策单点，DEBUG 打印路径原因 |
     | loader 的异构校验 inline 在 `__init__` 中 | `HeteroLoaderGuard` dataclass 封装，可独立测试 |
     | `NodeLoader.input_nodes += offset` inline | `NodeInputOffset.apply()` 独立函数，加 DEBUG 前后范围对比 |
     | `LinkLoader.edge_label_index[0/1] += offset` inline | `LinkInputOffset.apply()` 独立函数，注释保留上游 "reverse of standard convention" |

  2. **用户角度 bug**:
     - `_HeteroDecodeContext` 捕获：`num_edge_types * fanout_length` 的 `lho` 切片索引依赖 `label_type_hop_offsets` 格式，若传入旧格式 `label_hop_offsets` 会越界（`SampleReader.lho_name` 兼容逻辑在此处保护）
     - `integer_input_type is None` guard：若所有 etype 均未匹配 `input_type`，上游抛 `"Input type did not match any edge type!"`，Walpurgis 增强错误信息含已知 edge_types 列表
     - `edge_inverse` de-offset 逻辑：上游注释 `# De-offset the type based on lexicographic order` 说明 src/dst 比较逻辑，仅当 `input_type[0] != input_type[2]` 时需 de-offset

  3. **系统角度安全**:
     - `_numeric_edge_types` 属性带缓存（`__numeric_edge_types is None` guard），多次调用不重复构建；Walpurgis 的 `_choose_reader()` 每次采样调用一次，缓存确保无性能问题
     - `tempdir 移除`：上游 `gcn_dist_*.py` 中 `with tempfile.TemporaryDirectory` 块移除，表明 `directory` 参数对异构图已无效；Walpurgis examples 本就未使用 `directory`，`GcnLoaderTempDirRemoval` 文档类记录此决策

### Walpurgis 迁移位置

**新增文件**:
- `src/walpurgis/sampler/sampler.py` — 核心：`SampleIterator`、`SampleReader`、`HeterogeneousSampleReader`、`HomogeneousSampleReader`、`BaseSampler`，`_choose_reader()` 统一异构/同构路径决策
- `src/walpurgis/dataloader/hetero_loader_guard.py` — `HeteroLoaderGuard`（异构 loader 前置校验）、`NodeInputOffset`（input_nodes vertex offset 注入）、`LinkInputOffset`（edge_label_index offset 注入）

**改写20%（鲁迅拿法）**:
- `_HeteroDecodeContext` dataclass — 封装 `__decode_coo` 内散落维度变量，DEBUG 输出一行摘要
- `_safe_max_plus_one()` — 替代 3 处裸 `x.max()+1`，空 tensor 时返回 0（上游 silent bug 修复）
- `_choose_reader()` 私有方法 — 从 `sample_from_nodes/edges` 提取重复的路径选择逻辑，单点决策
- `HeteroLoaderGuard.from_graph_store()` 类方法 — 封装 loader `__init__` 中散落的异构校验，可单独测试
- `NodeInputOffset.apply()` / `LinkInputOffset.apply()` — 独立可测试的 offset 注入函数，加 DEBUG 前后范围对比
- 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖：
  `SampleIterator` 输出类型判断 → `SampleReader.lho_name` 自动选取 →
  `HeterogeneousSampleReader.__init__` 维度摘要 → `__decode_coo` 每 etype 的 map/lho 切片 →
  `BaseSampler._choose_reader` 路径选择 → `HeteroLoaderGuard.resolve` 校验结果 →
  `NodeInputOffset/LinkInputOffset.apply` 注入前后范围

## migrate c7e60f0: [SKIP] wholegraph: remove debugging details from CMake (#109)
- **Upstream commit**: c7e60f0 (cugraph-gnn, NVIDIA, PR #109)
- **跳过原因**: 修改 `cpp/CMakeLists.txt`（删 2 行）和 `python/pylibwholegraph/CMakeLists.txt`（删 39 行），移除 CMake debug message() 输出和冗余变量。纯 CMake 构建调试信息清理，Walpurgis 无 CMake，无迁移价值。

## migrate d38b832: remove dependency on cugraph-ops (#99)

- **Upstream commit**: d38b832 (cugraph-gnn, NVIDIA, PR #99)
- **Commit message**: `remove dependency on cugraph-ops (#99)`
- **Upstream diff** (50 个文件变动，39 行增加，4452 行删除):
  - `python/pylibwholegraph/pylibwholegraph/torch/gnn_model.py` — 删除 3 处 `elif framework_name == "cugraph"` 分支：`set_framework()`、`create_gnn_layers()`、`create_sub_graph()`、`layer_forward()`
  - `python/pylibwholegraph/pylibwholegraph/torch/common_options.py` — `--framework` 默认值从 `"cugraph"` 改为 `"wg"`，help 文本更新
  - `python/pylibwholegraph/pylibwholegraph/torch/cugraphops/__init__.py` — 清空（删 12 行）
  - `python/pylibwholegraph/pylibwholegraph/torch/cugraphops/gat_conv.py` — 删除（102 行）
  - `python/pylibwholegraph/pylibwholegraph/torch/cugraphops/sage_conv.py` — 删除（101 行）
  - `python/cugraph-pyg/cugraph_pyg/nn/conv/` — 所有 nn.conv 文件更新，移除 cugraph-ops import
  - `python/cugraph-dgl/cugraph_dgl/nn/conv/` — DGL conv 文件移除 cugraph-ops 依赖

- **Knuth 审查**:
  1. **diff 对比源**:
     | 上游 d38b832 | Walpurgis 迁移 |
     |---|---|
     | `gnn_model.py` 用 module-level `framework_name = None` 全局变量 | `WalpurgisFrameworkRegistry` 实例属性，多进程安全 |
     | `set_framework()` 用 `assert framework_name is None`（生产 `-O` 跳过） | 改为 `raise RuntimeError`，生产安全 |
     | `create_gnn_layers()` 三框架混合 if/elif，cugraph 静默删除 | `GnnLayerFactory.create()` 明确 ValueError + 迁移说明 |
     | `layer_forward()` cugraph 路径删除后无提示 | `LayerForwardDispatch.forward()` cugraph 路径抛 ValueError + 迁移说明 |
     | `--framework` 默认值改为 `"wg"` | `DEFAULT_FRAMEWORK = "wg"`，`VALID_FRAMEWORKS = ("pyg", "wg")` 常量化 |

  2. **用户角度 bug**:
     - 旧默认值 `--framework cugraph` 的用户升级后无任何报错，静默使用 `wg` 框架（因为旧 elif 分支被删，`framework_name == "cugraph"` 从未匹配，`gnn_layers` 保持空 `ModuleList`，训练 loss 不下降）。Walpurgis `WalpurgisFrameworkRegistry.set("cugraph")` 立即 `raise ValueError`，错误清晰
     - `create_sub_graph` 的 `add_csr_self_loop` 被删后，若用户传 `add_self_loop=True` 且 framework="cugraph"，旧代码空结果无报错。Walpurgis `SubgraphAdapter.build()` 提前 raise，不等到 GNN 层 forward 时才崩溃

  3. **系统角度安全**:
     - `cugraphops/sage_conv.py` 和 `gat_conv.py` 依赖 `pylibcugraphops.pytorch`（CUDA kernel），移除后整个 cugraph-ops C++ 库依赖链断开。Walpurgis 迁移确保任何 "cugraph" 字符串传入均提前 raise，避免 ImportError 传播到深层

### Walpurgis 迁移位置

**新增文件**:
- `src/walpurgis/models/gnn_framework.py` — 核心：`WalpurgisFrameworkRegistry`、`GnnLayerFactory`、`SubgraphAdapter`、`LayerForwardDispatch`，均不支持 "cugraph" framework（对应 d38b832 删除逻辑），`DEFAULT_FRAMEWORK = "wg"` 对应 common_options.py 默认值修改

**改写20%（鲁迅拿法）**:
- `WalpurgisFrameworkRegistry` — 替代 module-level 全局变量，实例化进程本地注册表，多进程安全
- `set("cugraph")` 路径 — 上游静默删除，Walpurgis 显式 `raise ValueError` 含迁移指引
- `VALID_FRAMEWORKS = ("pyg", "wg")` 常量 — 类型安全，可在测试中直接断言
- `LayerForwardDispatch.forward()` cugraph 路径 — 上游删除时无任何错误提示，Walpurgis 加 "原 API 签名" 注释帮助用户迁移
- 全链路 WALPURGIS_DEBUG=1 断点 print，覆盖：
  `WalpurgisFrameworkRegistry.set` 框架选择 → `GnnLayerFactory.create` 每层维度 →
  `SubgraphAdapter.build` CSR 维度 → `LayerForwardDispatch.forward` 输入/输出 shape

## migrate a076b51: [SKIP] Use GCC 13 in CUDA 12 conda builds. (#108)
- **Upstream commit**: a076b51 (cugraph-gnn, NVIDIA, PR #108)
- **跳过原因**: 修改 8 个 conda recipe/env yaml 文件，将 CUDA 12 构建中的 GCC 版本升级到 13。纯 conda 构建配置变更，Walpurgis 无 conda 构建体系，无迁移价值。

## migrate f6e3654: [SKIP] enforce `cmake-format` and `cmake-lint`, other small packaging changes (#111)
- **Upstream commit**: f6e3654 (cugraph-gnn, NVIDIA, PR #111)
- **跳过原因**: 修改 19 个文件：`cpp/cmake/config.json`（cmake-format 配置）、`cpp/cmake/thirdparty/get_*.cmake`（格式化）、`cpp/tests/CMakeLists.txt`（格式化）、`python/pylibwholegraph/CMakeLists.txt`（格式化）、`cpp/scripts/run-cmake-format.sh`（新增 83 行）、`rapids_config.cmake`（格式化）。纯 CMake 格式化工具配置，无算法改动，Walpurgis 无 CMake，无迁移价值。
---

## Batch 00 — commits 712255d..2776772

---

## migrate 712255d: [SKIP] remove ci/test_wheel.sh — 纯CI文件删除

- **Upstream commit**: 712255d (cugraph-gnn, James Lamb)
- **跳过原因**: 仅删除 `ci/test_wheel.sh`（40行CI脚本），无算法/库代码改动，Walpurgis无对应CI基础设施。

---

## migrate 25d1b55: [SKIP] Merge PR #60 remove-docs — merge commit + CI清理

- **Upstream commit**: 25d1b55 (merge commit)
- **跳过原因**: merge commit，变更内容全为CI脚本（build.sh、ci/test_wheel.sh、ci/wheel_smoke_test_cugraph.py），无算法价值。

---

## migrate 31ee98f: Add EmbeddingView + neighbor sampler reader fix

- **Upstream commit**: 31ee98f (cugraph-gnn, James Lamb)
- **Commit message**: `add PR CI for cugraph-pyg and cugraph-dgl (#59)`
- **Upstream diff** (有价值部分):
  - `cugraph-dgl/view.py` — 新增 `EmbeddingView` 类 (~55行): 大型embedding的懒访问封装，`__getitem__` 按索引取片，RuntimeError时自动fallback到CUDA索引，`shape`属性用探针索引推断feature_dim，`__call__`发出高内存警告
  - `cugraph-dgl/dataloading/neighbor_sampler.py` — bug fix: `sample_from_nodes`新API直接返回reader，旧版依赖`ds.get_reader()`隐式副作用

- **Knuth 审查**:
  1. **diff 对比**:
     | 上游 31ee98f | Walpurgis 迁移 |
     |---|---|
     | `EmbeddingView.__getitem__` RuntimeError fallback逻辑内联在方法体中 | 拆出 `_fetch_with_fallback()` 私有方法，便于单独测试 |
     | `shape` 属性 try/except 内联 | 拆出 `_probe_shape()` 私有方法，打印探针结果帮助排查 |
     | 封装 DGL专属 `FeatureStorage`，鸭子类型无显式接口 | `_WalpurgisEmbeddingBackend` 协议类显式化接口契约 |
     | 零调试输出 | WALPURGIS_DEBUG=1 覆盖 fetch/fallback/probe/call 全链路 |
  2. **上游bug**: `neighbor_sampler.py` 旧版 `ds.sample_from_nodes(...)` 返回None，`get_reader()` 依赖副作用；若采样失败无reader，`get_reader()`返回旧reader导致数据污染。新API直接返回reader，Walpurgis sampler层应遵循新API。
  3. **系统安全**: 探针索引`[0]`在空embedding（entry_count=0）时会崩溃；上游同款问题，迁移后`_probe_shape()` DEBUG输出帮助提前发现。

### Walpurgis 迁移位置
- **新增文件**: `src/walpurgis/tensor/embedding_view.py` — `EmbeddingView` + `_WalpurgisEmbeddingBackend`
- **修改文件**: `src/walpurgis/tensor/__init__.py` — 导出 `EmbeddingView`
- **改写20%**: `_WalpurgisEmbeddingBackend`协议类、`_fetch_with_fallback()`、`_probe_shape()`、全链路WALPURGIS_DEBUG断点

---

## migrate f7ab898: [SKIP] add nightly builds/tests — CI nightly配置

- **Upstream commit**: f7ab898 (cugraph-gnn, James Lamb)
- **跳过原因**: 全部变更为`.github/workflows/`和CI bash脚本，无Python算法代码。

---

## migrate d260ccb: [SKIP] add notebook tests, build.sh args — CI notebook测试

- **Upstream commit**: d260ccb (cugraph-gnn, James Lamb)
- **跳过原因**: CI notebook测试配置 + build.sh参数，无算法代码。

---

## migrate df5bdc4: update wholegraph — hierarchy memory type + entry_partition扩展

- **Upstream commit**: df5bdc4 (cugraph-gnn, zhuofan1123)
- **Commit message**: `update wholegraph (#65)`
- **Upstream diff** (有价值部分):
  - `embedding.py` — `create_embedding` 新增 `embedding_entry_partition` 参数、`hierarchy` memory_type支持、partition与round_robin_size冲突检查
  - `tensor.py` — `create_wholememory_tensor` / `create_wholememory_tensor_from_filelist` 新增 `tensor_entry_partition` 参数
  - 其余: C++ CUDA源文件（wholememory_ops内核）、binding.pyx — 需要重新编译，Walpurgis不复制C++层

- **Knuth 审查**:
  1. **diff 对比**:
     | 上游 df5bdc4 | Walpurgis 迁移 |
     |---|---|
     | `create_embedding`中 `if memory_type == "hierarchy": ... raise AssertionError` (裸AssertionError无消息) | Walpurgis层提前检查，`raise ValueError`附带清晰错误文本 |
     | `embedding_entry_partition` 与 `cache_policy` 冲突时 `print(...)` 忽略参数 | 同样行为，但DEBUG时额外打印分支决策 |
     | `tensor_entry_partition`/`embedding_entry_partition` 均已在 `utils.py` 中实现（前序迁移已含）| — |
  2. **上游bug**: `hierarchy` 分支中 `raise AssertionError` 后紧跟字符串字面量（不是括号内），字符串实为孤立表达式，AssertionError永远不会携带消息。迁移时改为 `raise ValueError` + 消息文本。
  3. **系统安全**: `hierarchy` 不支持NVSHMEM通信和cache_policy；上游在创建时才会CUDA崩溃，Walpurgis在Walpurgis层提前拦截。

### Walpurgis 迁移位置
- **修改文件**: `src/walpurgis/tensor/utils.py`
  - `create_wg_dist_tensor`: 新增 `hierarchy` backend分支
  - `create_wg_dist_tensor_from_files`: 同步新增 `hierarchy` backend分支
- **改写20%**: ValueError替代AssertionError，提前冲突检查，WALPURGIS_DEBUG打印backend选择

---

## migrate 5a17bbe: [SKIP] start publishing packages — 打包发布CI

- **Upstream commit**: 5a17bbe (cugraph-gnn, James Lamb)
- **跳过原因**: 全部为`.github/workflows/`发布脚本，无算法代码。

---

## migrate 2dd3001: [SKIP] enforce wheel size limits, README formatting in CI — CI质量检查

- **Upstream commit**: 2dd3001 (cugraph-gnn, James Lamb)
- **跳过原因**: wheel大小检查 + README格式CI，无算法代码。

---

## migrate 16e614c: [SKIP] remove versioning workaround for nightlies — CI版本号

- **Upstream commit**: 16e614c (cugraph-gnn, James Lamb)
- **跳过原因**: 版本号workaround删除（CI脚本），无算法代码。

---

## migrate d56dd66: [SKIP] DOC v25.02 Updates — 版本号/文档更新

- **Upstream commit**: d56dd66 (cugraph-gnn, Ray Douglass)
- **跳过原因**: VERSION文件、YAML配置、pyproject.toml版本号批量更新，无算法代码。

---

## migrate e1e32bc: [SKIP] fix devcontainer builds — devcontainer依赖配置

- **Upstream commit**: e1e32bc (cugraph-gnn, James Lamb)
- **跳过原因**: `dependencies.yaml` devcontainer构建修复（NVML/pytorch-cuda），Walpurgis不维护devcontainer。

---

## migrate 36c312c: [SKIP] Merge PR #71 forward-merge branch-24.12 — merge commit

- **Upstream commit**: 36c312c (merge commit, gpuCI bot)
- **跳过原因**: 自动forward-merge commit，conda环境YAML变更，无算法代码。

---

## migrate aa099e4: [SKIP] Relax PyTorch upper bound (allowing 2.4) — 依赖版本约束

- **Upstream commit**: aa099e4 (cugraph-gnn, jakirkham)
- **跳过原因**: conda/pyproject.toml PyTorch版本约束放宽，纯依赖管理，无算法代码。

---

## migrate 986cc76: [SKIP] Merge PR #76 forward-merge branch-24.12 — merge commit

- **Upstream commit**: 986cc76 (merge commit, gpuCI bot)
- **跳过原因**: 自动forward-merge，CI测试脚本+版本约束，无算法代码。

---

## migrate 2776772: Add gather/scatter 1D tensor support

- **Upstream commit**: 2776772 (cugraph-gnn, Chang Liu)
- **Commit message**: `[Feature] Add gather/scatter support 1D tensor (#74)`
- **Upstream diff** (3个文件):
  - `pylibwholegraph/torch/tensor.py` — `gather`: `embedding_dim = shape[1] if dim==2 else 1`，输出 `view(-1)` if 1D；`scatter`: assert `input_tensor.dim() == self.dim()`，1D时 `input_tensor.unsqueeze(1)` 再传入scatter_op
  - `pylibwholegraph/torch/wholememory_ops.py` — `wholememory_gather_forward_functor`: 同样的`embedding_dim`修复 + `view(-1)` 输出处理
  - `tests/test_wholegraph_gather_scatter.py` — 新增 `embedding_dim=0` 测试分支覆盖1D路径

- **Knuth 审查**:
  1. **diff 对比**:
     | 上游 2776772 | Walpurgis 迁移 |
     |---|---|
     | 修复在 pylibwholegraph C-extension层 (`tensor.py`/`wholememory_ops.py`) | Walpurgis在`DistTensor.__setitem__/getitem__`层统一处理维度转换，不依赖下游库是否已升级 |
     | `scatter`入口 `assert input_tensor.dim() == self.dim()` 会AssertionError | Walpurgis层检测`tensor_dim==1`后 `unsqueeze(1)` 再传，避免暴露给用户的AssertionError |
     | 零调试输出 | WALPURGIS_DEBUG=1 打印 `tensor_dim`、1D/2D路径选择、unsqueeze/view(-1)操作 |
  2. **上游遗留bug**: commit描述中明确: 1D scatter在multi-GPU下仍有问题（需要 #73的bugfix才能完整工作）。gather已验证OK，scatter单GPU测试通过但多GPU crash。Walpurgis迁移保留gather fix；scatter fix在Walpurgis层实现unsqueeze，实际运行正确性依赖上游C++层的完整修复。
  3. **系统安全**: `tensor_dim==1`的判断调用 `self._tensor.dim()`，若底层wm_tensor未初始化会AttributeError；此前已有`assert self._tensor is not None`守护，安全。

### Walpurgis 迁移位置
- **修改文件**: `src/walpurgis/tensor/dist_tensor.py`
  - `DistTensor.__setitem__`: 1D tensor时 `val.unsqueeze(1)` 再scatter
  - `DistTensor.__getitem__`: 1D tensor时 gather输出 `view(-1)` 还原为1D
- **改写20%**: 维度转换在Walpurgis层而非依赖底层库、tensor_dim诊断断点、上游C++层依赖说明注释


## migrate b578959: [SKIP] DOC v25.04 Updates — 版本号批量更新 + devcontainer/CI 配置变更，无算法内容

## migrate 2f16c37: [SKIP] update pip devcontainers to UCX 1.18 — devcontainer JSON 配置，Walpurgis 无此体系

## migrate 0e88280: [MIGRATE] Support PyG 2.6 in cuGraph-PyG
迁移内容: PyG 2.6 breaking change — feature_store/graph_store 索引从 2-tuple 改为 3/5-tuple
- 新增 src/walpurgis/examples/gcn/gcn_dist_sg.py (单卡 GCN 训练示例，4-tuple API)
- 改写20%: DataConfig/LoaderFactory/TrainStats 数据类 + 全链路 WALPURGIS_DEBUG 断点
- 注: gcn_dist_mnmg.py/rgcn_link_class_mnmg.py 已在前序 batch 迁移时采用新 API，本次补齐 sg 示例

## migrate 9498eb5: [SKIP] Build and test with CUDA 12.8.0 — CI 构建矩阵 + CUDA 版本更新，纯基础设施

## migrate 3546217: [SKIP] Revert CUDA 12.8 shared workflow branch changes — CI workflow revert，无代码变化

## migrate 25eef43: [SKIP] Merge branch-25.02 into branch-25.04 — 纯合并 commit

## migrate 362b800: [SKIP] Merge pull request #122 — 纯合并 commit

## migrate dd543dc: [MIGRATE] Heterogeneous Link Prediction Example for cuGraph-PyG
迁移内容:
1. 新增 src/walpurgis/sampler/sampler_utils.py
   - neg_sample 核心修复: 删除错误的分布式 all_reduce SUM（旧代码令每 rank 生成 world_size 倍负样本）
   - filter_cugraph_pyg_store / neg_cat / _sampler_output_from_sampling_results_* 全部迁移
   - 改写20%: HopIndexer/SamplerResultValidator 数据类 + WALPURGIS_DEBUG 断点
2. sampler/__init__.py 新增 sampler_utils 延迟 import
- 注: taobao_mnmg 示例已从更新版 f2b7f50 迁移，FanoutConverter dict 已在 hetero_link_pred_fixes.py 实现

## migrate 9813c0b: [SKIP] Merge pull request #124 — 纯合并 commit

## migrate 05fe6f4: [SKIP] Knowledge Graph/Graph Database Renumbering — 已在前序 batch 迁移 (renumber_kg.py 已存在)

## migrate 921ed5b: [SKIP] Merge pull request #125 — 纯合并 commit

## migrate 73134e4: [SKIP] disallow fallback to Make in Python builds — pyproject.toml 构建系统配置，Walpurgis 无 ninja/cmake 构建体系

## migrate 431801c: [MIGRATE] Deprecate the Dask API in cuGraph-PyG
迁移内容: 新增 src/walpurgis/dataloader/loader_deprecation.py
- DaskNeighborLoader / BulkSampleLoader FutureWarning wrapper
- 改写20%: LoaderDeprecationGate/LoaderDeprecationRegistry 数据类，复用 feature_store_deprecation 设计模式
- dataloader/__init__.py 新增导出

## migrate 1fdd5cb: [SKIP] Merge pull request #126 — 纯合并 commit (431801c 已独立迁移)

## migrate e90d1e6: [MIGRATE] Fix of create_node_classification_datasets (#128)
迁移内容: 新增 src/walpurgis/dataloader/node_classification.py
- create_node_classification_datasets: 拼写修复(claffication→classification) + 解耦 pickle IO
- 改写20%: NodeClassificationData/DatasetSplitValidator/PickleLoader 数据类 + WALPURGIS_DEBUG 断点
- create_node_claffication_datasets 旧名保留为 FutureWarning compat alias
- dataloader/__init__.py 新增导出
---

## Batch 01 — commits c5cc3e7..659a0e1

---

## migrate c5cc3e7: [SKIP] Merge PR #77 forward-merge — merge commit

- **Upstream commit**: c5cc3e7 (gpuCI bot)
- **跳过原因**: 自动 forward-merge commit，无算法代码。

---

## migrate 4807986: [SKIP] Dynamic load NVML symbols for better compatibility — C++层

- **Upstream commit**: 4807986 (Chang Liu)
- **跳过原因**: 全部为 C++ 文件（nvml_wrap.cpp/.h, communicator.cpp, system_info.cpp），动态加载 NVML 符号解决驱动兼容性。无 Python 层变更，Walpurgis 无 C++ 编译能力。

---

## migrate 046b2f2: [SKIP] Merge PR #78 — merge commit

- **Upstream commit**: 046b2f2 (gpuCI bot)
- **跳过原因**: 自动 merge commit。

---

## migrate 23cdecd: [SKIP] Add breaking change workflow trigger — CI workflow

- **Upstream commit**: 23cdecd (Jake Awe)
- **跳过原因**: `.github/workflows/` CI 配置，无算法代码。

---

## migrate 01abe44: [SKIP] Require approval to run CI on draft PRs — CI配置

- **Upstream commit**: 01abe44 (Bradley Dice)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate 466b5b9: [SKIP] Add stream synchronization before scatter — C++ CUDA kernel

- **Upstream commit**: 466b5b9 (Chang Liu)
- **跳过原因**: 仅修改 `scatter_op_impl_mapped.cu`（CUDA C++），在 `wholememory_scatter_mapped` 末尾加 `cudaStreamSynchronize`。Python 层 1D scatter 修复已在 batch00/2776772 中通过 `DistTensor.__setitem__` 层处理。

---

## migrate 136e44b: [SKIP] Merge PR #83 — merge commit

- **Upstream commit**: 136e44b (gpuCI bot)
- **跳过原因**: 自动 merge commit。

---

## migrate b3dec8c: [SKIP] skip conda-python-tests on arm64 — CI配置

- **Upstream commit**: b3dec8c (James Lamb)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate 42c16fe: [SKIP] add devcontainers — devcontainer 配置

- **Upstream commit**: 42c16fe (James Lamb)
- **跳过原因**: devcontainer JSON/YAML 配置文件，无算法代码。

---

## migrate 7ec8ace: [SKIP] Disable RockyLinux Tests — CI配置

- **Upstream commit**: 7ec8ace (Alex Barghi)
- **跳过原因**: CI 矩阵配置，无算法代码。

---

## migrate ce6610d: [SKIP] Merge PR #90 — merge commit

- **Upstream commit**: ce6610d (gpuCI bot)
- **跳过原因**: 自动 merge commit。

---

## migrate ca3ca80: [SKIP] skip CUDA 11.4 conda-python-tests — CI配置

- **Upstream commit**: ca3ca80 (James Lamb)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate fa6f125: [SKIP] Update Changelog — changelog

- **Upstream commit**: fa6f125 (Ray Douglass)
- **跳过原因**: changelog 文件，无算法代码。

---

## migrate b0e0222: [SKIP] merge branch-24.12 into branch-25.02 — merge commit

- **Upstream commit**: b0e0222 (James Lamb)
- **跳过原因**: forward-merge commit，无算法代码。

---

## migrate 77206de: [SKIP] Merge PR #95 — merge commit

- **Upstream commit**: 77206de (Ray Douglass)
- **跳过原因**: 自动 merge commit。

---

## migrate 0013186: [SKIP] Update version references in workflow — CI版本号

- **Upstream commit**: 0013186 (Jake Awe)
- **跳过原因**: CI workflow 版本引用，无算法代码。

---

## migrate cc1bab9: [SKIP] build libwholegraph docs in CI — CI文档构建

- **Upstream commit**: cc1bab9 (James Lamb)
- **跳过原因**: CI docs 配置，无算法代码。

---

## migrate e641496: [SKIP] fix nightly docs workflow dependencies — CI配置

- **Upstream commit**: e641496 (James Lamb)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate af22a12: [SKIP] remove unused dependencies.yaml entries — 依赖清理

- **Upstream commit**: af22a12 (James Lamb)
- **跳过原因**: dependencies.yaml 清理，无算法代码。

---

## migrate 71675d8: [SKIP] simplify wholegraph CMake — CMake 构建

- **Upstream commit**: 71675d8 (James Lamb)
- **跳过原因**: CMake 构建脚本，无 Python 算法代码。

---

## migrate bb81a18: [SKIP] remove flake8, clang tools from wholegraph CMake — CMake

- **Upstream commit**: bb81a18 (James Lamb)
- **跳过原因**: CMake 工具链清理，无算法代码。

---

## migrate e2b4cf0: [SKIP] Remove invalid conditional — CI配置

- **Upstream commit**: e2b4cf0 (Vyas Ramasubramani)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate e332b68: [SKIP] Fix inputs for the workflow — CI配置

- **Upstream commit**: e332b68 (Vyas Ramasubramani)
- **跳过原因**: CI 配置，无算法代码。

---

## migrate 85cab72: [SKIP] Check if nightlies have succeeded recently enough — CI配置

- **Upstream commit**: 85cab72 (Vyas Ramasubramani)
- **跳过原因**: CI nightly 检查脚本，无算法代码。

---

## migrate 87455cf: [SKIP] Remove Build Directory — build/ 目录删除

- **Upstream commit**: 87455cf (Alex Barghi)
- **跳过原因**: 删除 `python/cugraph-pyg/build/` 构建产物目录（6699行全删），无算法变更。

---

## migrate a9ab8b4: [SKIP] Support Heterogeneous Sampling — 已在前序迁移中实现

- **Upstream commit**: a9ab8b4 (Alex Barghi)
- **跳过原因**: `HeterogeneousSampleReader`、`_numeric_edge_types`、`_vertex_offset_array` 均已在 Walpurgis `sampler/sampler.py` 和 `dataloader/hetero_loader_guard.py` 中实现。重复迁移无意义。

---

## migrate d38b832: [SKIP] remove dependency on cugraph-ops — cugraph-ops 层删除

- **Upstream commit**: d38b832 (Tingyu Wang)
- **跳过原因**: 删除 `GATConv`/`GATv2Conv`/`BaseConv`（依赖 `pylibcugraphops`），Walpurgis 使用 PyG 原生 GATConv，不依赖 `pylibcugraphops`。`SparseGraph` 的 docstring 更新无算法价值。

---

## migrate a076b51: [SKIP] Use GCC 13 in CUDA 12 conda builds — 构建配置

- **Upstream commit**: a076b51 (James Lamb)
- **跳过原因**: conda 构建配置，无算法代码。

---

## migrate f6e3654: [SKIP] enforce cmake-format and cmake-lint — CMake 格式化

- **Upstream commit**: f6e3654 (James Lamb)
- **跳过原因**: CMake 格式化配置，无算法代码。

---

## migrate b578959: [SKIP] DOC v25.04 Updates — 版本号更新

- **Upstream commit**: b578959 (Ray Douglass)
- **跳过原因**: 版本号批量更新，无算法代码。

---

## migrate 2f16c37: [SKIP] update pip devcontainers to UCX 1.18 — devcontainer配置

- **Upstream commit**: 2f16c37 (James Lamb)
- **跳过原因**: devcontainer 配置更新，无算法代码。

---

## migrate 0e88280: [SKIP] Support PyG 2.6 — PyG API 兼容性更新（已应用）

- **Upstream commit**: 0e88280 (Alex Barghi)
- **跳过原因**: PyG 2.6 API 更新（`feature_store[k,a]` → `[k,a,None]`，`put_edge_index` 加 size tuple）。Walpurgis 现有代码已使用 PyG 2.6 风格 API（examples/movielens 等），无需重复迁移。

---

## migrate 9498eb5: [SKIP] Build and test with CUDA 12.8.0 — CI CUDA版本

- **Upstream commit**: 9498eb5 (Bradley Dice)
- **跳过原因**: CI CUDA 版本矩阵配置，无算法代码。

---

## migrate 3546217: [SKIP] Revert CUDA 12.8 shared workflow — CI回滚

- **Upstream commit**: 3546217 (Bradley Dice)
- **跳过原因**: CI CUDA 版本配置回滚，无算法代码。

---

## migrate 25eef43: [SKIP] Merge branch-25.02 into branch-25.04 — merge commit

- **Upstream commit**: 25eef43 (James Lamb)
- **跳过原因**: forward-merge commit，无算法代码。

---

## migrate 362b800: [SKIP] Merge PR #122 — merge commit

- **Upstream commit**: 362b800 (Bradley Dice)
- **跳过原因**: merge commit，无算法代码。

---

## migrate c7e60f0: [SKIP] wholegraph: remove debugging details from CMake — CMake

- **Upstream commit**: c7e60f0 (James Lamb)
- **跳过原因**: CMake 调试信息清理，无算法代码。

---

## migrate c11936f: [SKIP] wheels: build with CUDA 13.0 — CI/packaging

- **Upstream commit**: c11936f (James Lamb)
- **跳过原因**: CI wheel 构建配置，无算法代码。

---

## migrate dbb33ad: [SKIP] Use PyBuffer_FillInfo for simple buffers — C++ Cython

- **Upstream commit**: dbb33ad (jakirkham)
- **跳过原因**: `wholememory_binding.pyx` Cython buffer 协议优化（C++层），无 Python 算法变更。

---

## migrate fbea7cb: [SKIP] Fix append unique (PyObject callback) — 已在 batch00 迁移

- **Upstream commit**: fbea7cb (linhu-nv)
- **跳过原因**: `wholegraph_env.py` 回调签名更新已在 batch00 `wholememory_cb.py` 中完整迁移。

---

## migrate 7ea1138: [SKIP] Fix Weights Issue in Negative Sampling — 已在前序迁移中实现

- **Upstream commit**: 7ea1138 (Alex Barghi)
- **跳过原因**: `sampler_utils.py` neg_sample src/dst_weight concat 修复已在 Walpurgis `models/neg_sampler_weights.py` 中实现（`NegSampleWeightPlan` 封装）。

---

## migrate 851f9e6: [SKIP] enable arm64 wheel tests — CI配置

- **Upstream commit**: 851f9e6 (James Lamb)
- **跳过原因**: CI arm64 测试配置，无算法代码。

---

## migrate 1262620: [SKIP] Update to clang 20.1.8 — 构建工具链

- **Upstream commit**: 1262620 (Bradley Dice)
- **跳过原因**: clang 版本更新，无算法代码。

---

## migrate 3780a05: [SKIP] Preserve torch_cpp_ext source files — 构建脚本

- **Upstream commit**: 3780a05 (Tingyu Wang)
- **跳过原因**: build.sh 构建脚本调整，无算法代码。

---

## migrate 93849d2: [SKIP] resolve zizmor findings — CI安全检查

- **Upstream commit**: 93849d2 (Gil Forsyth)
- **跳过原因**: CI workflow 安全检查配置，无算法代码。

---

## migrate b58ea19: [SKIP] support embedding training with bf16 and fp16 — C++层 + dtype已覆盖

- **Upstream commit**: b58ea19 (linhu-nv)
- **跳过原因**: C++ CUDA 层（wholememory_binding.pyx DtBF16 fix）+ 测试参数化（float16/bfloat16）。Python dtype 支持已通过 220563b 迁移覆盖。

---

## migrate 220563b: Explicitly support bfloat16 in FeatureStore dtype table

- **Upstream commit**: 220563b (Alex Barghi)
- **Commit message**: `Explicitly support bf16 in feature store (#458)`
- **Upstream diff**: `feature_store.py` dtype 映射表新增 `(torch.bfloat16, 7)` 一行。

- **Knuth 审查**:
  1. **diff 对比**:
     | 上游 220563b | Walpurgis 迁移 |
     |---|---|
     | `feature_store.py` 硬编码 dtype 列表，仅新增 bfloat16=7 | `DtypeNegotiator.DTYPE_TO_ID` 同步补全缺失的 int16/float16/int8（上游有但 Walpurgis 未迁移），加 bfloat16=8 |
     | ID 为 7 | Walpurgis 重排后 bfloat16=8（因补全了缺失的窄整型/半精度，避免 ID 冲突） |
  2. **上游 bug**: 上游 dtype ID=4 同时被 `torch.bool`（feature_store）和 `torch.int16`（WM binding）使用，可能导致跨模块 dtype 解码冲突。Walpurgis 在 `DtypeNegotiator` 中统一管理，ID 连续无冲突。
  3. **系统安全**: bf16 embedding 训练时若 FeatureStore 不认识 bfloat16，all_gather dtype 协商会 ValueError（不可恢复）。此修复为 bf16 embedding 训练的前提条件。

### Walpurgis 迁移位置
- **修改文件**: `src/walpurgis/core/unified_store.py`
  - `DtypeNegotiator.DTYPE_TO_ID`: 补全 int16(5)/float16(6)/int8(7)/bfloat16(8)
- **改写20%**: 补全上游缺失的窄整型+半精度 dtype，连续 ID 避免冲突，注释说明各 ID 来源

---

## migrate b25bc88: Support Disjoint Sampling in DistributedNeighborSampler

- **Upstream commit**: b25bc88 (Alex Barghi)
- **Commit message**: `[FEA] Support Disjoint Sampling in cuGraph-PyG (#452)`
- **Upstream diff**:
  - `distributed_sampler.py` — `DistributedNeighborSampler.__init__` 新增 `disjoint: bool = False`；`__func_kwargs` 新增 `"disjoint_sampling": disjoint`；`__calc_local_seeds_per_call` 新增 `disjoint` 参数、重排 ≤0 检查顺序、`fanout_prod *= fanout[0]`（disjoint 内存放大）
  - `neighbor_loader.py` — 删除 `if disjoint: raise ValueError("Disjoint sampling is currently unsupported")`；传入 `disjoint=disjoint`

- **Knuth 审查**:
  1. **diff 对比**:
     | 上游 b25bc88 | Walpurgis 迁移 |
     |---|---|
     | `__calc_local_seeds_per_call` 位置参数调用 | Walpurgis 改为 keyword-only（加 `*`），避免未来位置参数顺序混乱 |
     | 无 disjoint 内存放大的调试输出 | WALPURGIS_DEBUG=1 打印 fanout_prod 放大前后对比 |
     | ≤0 检查顺序变更无注释 | 加注释说明旧逻辑 bug：hetero 路径 wildcard fanout 在聚合前就返回 default |
  2. **上游 bug（已知）**: b25bc88 commit 描述注明 "CI will currently fail until rapidsai/cugraph#5500 is resolved"——pylibcugraph 端的 `disjoint_sampling` kwarg 支持还在进行中，测试中只验证参数传递与内存估算，不验证采样输出正确性。
  3. **系统安全**: `fanout_prod *= fanout[0]` 在 `fanout=[0]` 时会乘以 0，导致除零。上游在乘之前已有 ≤0 检查提前返回，顺序正确。Walpurgis 移动 ≤0 检查到 hetero 聚合之后、disjoint 放大之前，保证同样的保护。

### Walpurgis 迁移位置
- **修改文件**: `src/walpurgis/sampler/distributed_sampler.py`
  - `DistributedNeighborSampler.__init__`: 新增 `disjoint=False`，`disjoint_sampling` 进 `__func_kwargs`，调试 log 加 disjoint 状态
  - `__calc_local_seeds_per_call`: keyword-only 参数、重排 ≤0 检查、disjoint 内存放大 + DEBUG 输出
- **新增测试**: `src/walpurgis/tests/sampler/test_distributed_sampler.py`
  - `test_disjoint_sampler_batch_structure`: 验证 disjoint=True 参数传递
  - `test_disjoint_memory_estimate_amplification`: 验证内存估算 fanout[0] 放大因子

---

## migrate 659a0e1: Fix hashing and node id issues in disjoint sampling test

- **Upstream commit**: 659a0e1 (Alex Barghi)
- **Commit message**: `[BUG] Fix hashing and node id issues in disjoint sampling test (#474)`
- **Upstream diff**: `test_neighbor_loader.py`
  - `tree_vertices[n_id]` → `tree_vertices[n_id.item()]`（tensor hash 不稳定 → int key）
  - `edges_hop = batch.num_sampled_edges[hop]` → `int(...)` （tensor 不能做切片索引）
  - `torch.arange(batch.num_sampled_nodes[0].item())` 替代 `for n_id in batch.input_id`
  - 参数化 `batch_size` 扩展为 `[1,2,4,8,16]`

- **Knuth 审查**:
  1. 原 bug: `tree_vertices[n_id]` 用 0-dim tensor 做 dict key，Python hash(tensor) 基于对象 id 而非值，相同值的不同 tensor 对象 hash 不同，导致随机性失败。
  2. `edges_hop` 是 tensor，用作 slice `offset:offset+edges_hop` 会触发 `TypeError: slice indices must be integers or None`——上游 CI 随机失败根因。
  3. Walpurgis 迁移: `_DisjointBatchInspector.from_batch()` 将 659a0e1 的修复内化为类方法，`.item()` 和 `int()` 转换在一处集中处理，断点输出每个 seed 的 subgraph 大小。

### Walpurgis 迁移位置
- **修改文件**: `src/walpurgis/tests/sampler/test_distributed_sampler.py`
  - `_DisjointBatchInspector`: 封装 tree_vertices 构建 + 659a0e1 修复 + disjoint 不相交断言
  - `test_disjoint_sampler_batch_structure`: batch_size=[1,2,4,8,16]

---

## migrate 5909ae8: [SKIP] Fp16 embedding train — C++ CUDA kernel

- **Upstream commit**: 5909ae8 (linhu-nv)
- **跳过原因**: 全部为 C++ CUDA 文件（embedding.cpp, exchange_embeddings_nccl_func.cu），BF16/FP16 梯度 backward dedup 路径。无 Python 层变更。

---

## migrate 662a6d9: [SKIP] fix shm permission — C++层

- **Upstream commit**: 662a6d9 (linhu-nv)
- **跳过原因**: 仅修改 `memory_handle.cpp` 共享内存权限，无 Python 层变更。

---

## migrate 81b7074: [SKIP] Update MAG example to show fp16/bf16 support — 示例文件

- **Upstream commit**: 81b7074 (Alex Barghi)
- **跳过原因**: 仅更新 `mag_lp_mnmg.py` 示例（新增 fp16/bf16 参数），walpurgis 有自己的 MAG 示例路径（`examples/mag/`）。

## migrate 8f20a20: [SKIP] merge release/26.06 into main — conda环境文件/dependencies.yaml版本bump, 无算法内容

## migrate 5c725e7: [SKIP] Merge PR #472 main-merge-release/26.06 — 与8f20a20相同的版本文件merge commit, 无实质改动

## migrate eb868d1: [SKIP] fix(ci): fix configuration for breaking change notification workflow — GitHub Actions workflow配置修复(.github/workflows), Walpurgis无此CI体系

## migrate 659a0e1: [DONE] Fix hashing and node id issues in disjoint sampling test
### 上游变化 (659a0e1 python/cugraph-pyg/cugraph_pyg/tests/loader/test_neighbor_loader.py):
- `for n_id in batch.input_id` → `for n_id in torch.arange(batch.num_sampled_nodes[0].item())`
  原因: input_id是张量, 张量作dict key导致hash不稳定(hash=id()非值语义), 偶发CI失败
- `edges_hop = batch.num_sampled_edges[hop]` → `int(batch.num_sampled_edges[hop])`
  原因: 张量scalar用于切片偶发TypeError
- `tv_items[i] & tv_items[j] == set()` → `(tv_items[i] & tv_items[j]) == set()`
  原因: 操作符优先级歧义, 括号化防止歧义
- batch_size参数化扩展: `[1,2,4]` → `[1,2,4,8,16]`, 覆盖更大batch场景
### 迁移到: src/walpurgis/models/disjoint_sampler.py :: validate_disjoint_batches()
- 替换hash table遍历逻辑: batch.input_id → num_sampled_nodes[0] + range()连续整数key
- edges_hop强制int()转换
- 括号化集合交集比较
- WALPURGIS_DEBUG=1时打印num_seeds/batch_idx/hop调试信息

## migrate 81b7074: [DONE] Update MAG example to show fp16/bf16 support
### 上游变化 (81b7074 python/cugraph-pyg/cugraph_pyg/examples/mag_lp_mnmg.py):
- 新增`parse_dtype()`函数 + `_DTYPE_CHOICES = ("float32","float16","bfloat16")`
- Classifier.__init__接收dtype参数, 存储self.dtype
- WholeMemory embedding dtype: `torch.float32` → `dtype`参数
- forward(): `batch["paper"].x.to(w_dtype)` (从paper_lin.weight.dtype推断)
- zeros初始化: 硬编码`device="cuda"` → `device=x_paper.device, dtype=x_paper.dtype`
- feature_store写入: `data.x_dict["paper"]` → `.to(dtype)`
- bcy = `vy[f].to(torch.float32)` → `vy[f].to(dtype)` (延迟到feature_store写入)
- 变量名stype/dtype → src_type/dst_type (修复同名遮蔽bug)
- edge_attr计算: `.to(dtype).reshape((-1,1))` 在reshape前转换
- 命令行新增`--dtype bfloat16`参数, `model.to(device, dtype)`
- embedding输出前: `.to(torch.float32)` 保证cudf兼容性
### 迁移到: src/walpurgis/examples/mag/mag_lp_mnmg.py (已由前序batch完整迁移)
- DTypeRegistry替代裸dict, KeyError→友好ValueError
- NodeZeroInitializer封装zeros device/dtype跟随逻辑
- _dbg()调试出口, WALPURGIS_DEBUG=1打印w_dtype/plin.weight.dtype/edge_attr维度信息

## migrate 45129c4: [SKIP] Update Changelog [skip ci] — 仅CHANGELOG.md更新, 无代码改动

## migrate c06bbbe: [SKIP] Merge PR #475 release/26.06 forward-merge — 纯CHANGELOG merge commit, 无实质内容

## migrate ac3c900: refactor: build wheels and conda packages using Python limited API

- **Upstream commit**: ac3c900 (cugraph-gnn, NVIDIA, PR #407)
- **Commit message**: `refactor: build wheels and conda packages using Python limited API`
- **Upstream diff** (14 files changed, 100 insertions):
  - `.github/workflows/build.yaml`, `ci/build_*.sh` — CI/wheel构建脚本切换 limited API
  - `conda/recipes/pylibwholegraph/recipe.yaml` — conda recipe limited API配置
  - `python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx` — 核心修复

- **迁移决策**: PARTIAL — CI/build/conda 部分 SKIP (Walpurgis无conda体系);
  `wholememory_binding.pyx` 的 Python limited API 修复有实质内容, 迁移 Python 调用层防御。

- **Knuth审查**:
  1. diff对比源: 上游 `PyUnicode_AsUTF8(state_name)` 返回 borrowed C string (生命周期与
     Python 对象绑定), 若 GC 在 C 函数使用前回收该对象则悬空指针; 修复改用
     `PyUnicode_AsUTF8String` (新 bytes 对象, 有自己的引用计数) + `PyBytes_AsString`。
     同样, `load_wholememory_handle_from_filelist` 中 `filenames[i]` 原来直接存 borrowed
     指针; 修复改用 `strdup` 拷贝字符串, `finally` 块逐一 `free`。
  2. 用户角度 bug: 从 Python 调用 `wholememory_load_from_file(filelist=[...])` 时,
     若 GC 在 C 层循环内回收了某个 list 元素, `filenames[i]` 指向已释放内存,
     在优化构建下偶发段错误——典型"在自己机器上跑通但在 CI 崩"的幽灵 bug。
  3. 系统角度安全: 修复在 pylibwholegraph >= 26.04 C 层已覆盖, 但 Python 调用方
     传入 generator / pathlib.Path / bytes 等非 str 对象同样危险。
     Walpurgis 在 `tensor/utils.py` 的 `_sanitize_file_list()` 函数中加 Python 层配套防御:
     具现化 list (防惰性迭代器 GC)、类型规范化 (bytes/PathLike → str)、UTF-8 编码探针。

- **Walpurgis迁移位置**: `src/walpurgis/tensor/utils.py`
  - 新增 `_sanitize_file_list()`: 具现化file_list + bytes/PathLike → str + UTF-8探针
  - `create_wg_dist_tensor_from_files()` 入口调用 `_sanitize_file_list()`
  - WALPURGIS_DEBUG=1 时打印规范化前后路径

## migrate f67c5e5: [SKIP] chore(greptile): add basic config file — Greptile AI代码审查工具配置,Walpurgis无此工具

## migrate dbe99b2: [SKIP] check-nightly-ci: update to new version — GitHub Actions CI配置,Walpurgis无CI体系

## migrate d43e6c1: [ALREADY MIGRATED] [BUG] Fix warnings, fix MNMG graph store test, fix Matrix Accessors — 已在前序批次迁移至 src/walpurgis/core/dist_matrix.py

## migrate 63b04c3: [SKIP] check-nightly-ci: remove testing config — CI配置清理,Walpurgis无CI体系

## migrate 34fbaa4: [SKIP] refactor(limited api): add explicit wheel.py-api to pyproject.toml — pyproject.toml构建配置,Walpurgis使用pip安装无wheel构建

## migrate cd2790d: [SKIP] Update Cython lower bound pin to 3.2.2 — Cython版本pin更新,Walpurgis无Cython组件

## migrate 6b10c84: [SKIP] Remove pytest upper bound pin — pytest版本pin清理,依赖矩阵配置

## migrate 09aa727: [SKIP] extend check-nightly-ci allowance to 50 days — CI nightly失败容忍天数配置

## migrate ea84449: [SKIP] add no_pytorch matrix option in dependencies.yaml — RAPIDS dependencies.yaml矩阵配置,Walpurgis无此文件

## migrate 47fb350: [SKIP] libwholegraph: declare nvidia-nccl dependency for CUDA 13 wheels — wheel依赖声明,Walpurgis依赖由pip管理

## migrate f377db5: [SKIP] make PyTorch installation in conda test jobs stricter — conda测试依赖配置,Walpurgis无conda体系

## migrate 8a4fb98: [SKIP] CI: restore arm64 conda tests, re-use run_* scripts in test_* scripts — CI脚本重构,Walpurgis无CI脚本体系

## migrate bd2d577: [SKIP] Revert "Prepare release/26.04" — 纯revert release prep commit,CI/RAPIDS_BRANCH版本变量回退

## migrate 33a74f1: [SKIP] Prepare release/26.04 — release prep commit,CI/RAPIDS_BRANCH版本变量批量更新
## migrate 1e91ed7: Remove DaskGraphStore/CuGraphStore wrappers from DeprecationPolicy

**上游**: cugraph-gnn 1e91ed7 - Remove Dask API from cuGraph-PyG (#166)
**迁移文件**: `src/walpurgis/core/feature_store_deprecation.py`

上游删除了 `cugraph_pyg/data/dask_graph_store.py` 整个模块（1321行）和 `__init__.py` 中的
`DaskGraphStore`/`CuGraphStore` wrapper 函数。这是 Dask API 在 release 25.02 废弃后
在 25.06 的正式删除。

Walpurgis 改写20%:
- 新增 `_RemovedEntryGuard` callable stub，取代直接缺失——调用时抛 RuntimeError 并给出迁移指引
- 为 `DeprecationPolicy` 动态注入 `mark_removed()` 方法，统一管理"废弃中"与"已彻底移除"两类状态
- `DaskGraphStore` 和 `CuGraphStore` 通过 `_POLICY.mark_removed()` 注册，`has()` 可查询，`__call__` 触发 RuntimeError
- 断点7: mark_removed 注册事件；断点8: 调用已删除 API 时打印堆栈摘要
- 扩展 `__all__` 导出 `DaskGraphStore`, `CuGraphStore`, `_RemovedEntryGuard`

## migrate 05b5791: Remove dask_client fixture — 迁移测试框架去Dask化

**上游**: cugraph-gnn 05b5791 - cugraph-pyg: remove Dask dependencies and related test code (#168)
**迁移文件**: `src/walpurgis/tests/sampler/conftest.py` (新建)

上游删除了 `dask_cuda`, `dask.distributed` 依赖，移除了测试 conftest 中的 `dask_client` fixture
（LocalCUDACluster + Client + stop_dask_client 完整集群启动链），并删除 `sampler_utils.py`
中的 `dask_cudf = import_optional("dask_cudf")`。

Walpurgis 本就无 dask 依赖，此 commit 确认并固化设计决策。改写20%:
- 新建 `conftest.py`，替代空缺的 fixture 定义
- `dask_client` fixture 改为发出 DeprecationWarning 后 `pytest.skip`（友好报错而非神秘的 fixture-not-found）
- 新增 `single_gpu_available` session-scope fixture，统一 skipif 逻辑
- 断点1: conftest 加载时打印 GPU/CUDA/cupy 环境摘要
- 断点2: dask_client 被请求时打印调用警告

## migrate 456d5a2: Add deprecation warnings for DGL classes — 新建 dgl_deprecation 模块

**上游**: cugraph-gnn 456d5a2 - add deprecation warnings for DGL classes
**迁移文件**: `src/walpurgis/core/dgl_deprecation.py` (新建)

上游给 `CuGraphStorage` 加了 FutureWarning wrapper，给 `DaskDataLoader`/`DataLoader` 改写
了废弃消息，并在模块顶部加了 "cuGraph-DGL is no longer under active development" 的全局警告。

Walpurgis 改写20%:
- `_DglLegacyBanner` class 替代裸模块级 `warnings.warn`：同进程去重、WALPURGIS_DEBUG 细节输出、
  `reset()` 方法供测试重置
- `CuGraphStorageCompat` callable class 持有调用统计 `call_count`，替代裸 wrapper 函数
- `DaskDataLoaderCompat` callable class 含推荐迁移路径注释
- 模块加载自动调用 `DGL_LEGACY_BANNER.issue()`
- 断点1: banner 发出事件；断点2: CuGraphStorageCompat 入口；断点3: DaskDataLoaderCompat 入口

## migrate adb4006: fix circular import — tensor/dist_matrix.py 直接模块导入

**上游**: cugraph-gnn adb4006 - fix circular import
**迁移文件**: `src/walpurgis/tensor/dist_matrix.py`

上游把 `import cugraph_dgl; from cugraph_dgl import CuGraphStorage`（包级别，触发 __init__）
改为 `from cugraph_dgl.cugraph_storage import CuGraphStorage`（直接模块文件），消除循环依赖。

Walpurgis 中发现同等循环:
`tensor/__init__.py` → `dist_matrix.py` → `from walpurgis.tensor import DistTensor`
→ 回到 `tensor/__init__.py`（等待 DistMatrix 完成）

修复: 将 `from walpurgis.tensor import DistTensor` 改为
`from walpurgis.tensor.dist_tensor import DistTensor`（直接模块路径）。

## migrate feffb39: fix import — dist_matrix.py 循环导入补全修复

**上游**: cugraph-gnn feffb39 - fix import
**迁移文件**: `src/walpurgis/tensor/dist_matrix.py` (与 adb4006 合并在同一次修改)

上游在 adb4006 之后补充了 `from cugraph_dgl.graph import Graph` 直接导入，
并把函数返回类型注解 `cugraph_dgl.Graph` 改为 `Graph`（消除对包命名空间的依赖）。

Walpurgis 迁移此模式已在 adb4006 中一并处理（改写注释同步记录在 dist_matrix.py header）。

## migrate 1b2fce2: fix bad import in tests — 测试文件 dataloader 引用修正

**上游**: cugraph-gnn 1b2fce2 - fix bad import
**迁移文件**: `src/walpurgis/tests/sampler/test_dataloader_refs.py` (新建)

上游将 `test_dask_dataloader_mg.py` 中的 `cugraph_dgl.dataloading.DaskDataLoader`
改回 `cugraph_dgl.dataloading.DataLoader`（因为 456d5a2 重定向了 DataLoader 名称）。

Walpurgis 改写20%:
- 新建 `test_dataloader_refs.py`，把这三个 commit 的测试修正逻辑转化为主动测试断言
- 验证直接模块导入 vs 包级别导入指向同一类（检测是否引入了 wrapper 链）
- 验证 DataLoader 构造无 FutureWarning（检测是否经过废弃 wrapper）
- 断点1: 两种导入方式的类型信息对比；断点2: DataLoader 构造完成 batch count

## migrate a57912c: fix references to dask data loader — 完整模块路径替代包级别引用

**上游**: cugraph-gnn a57912c - fix references to dask data loader
**迁移文件**: `src/walpurgis/tests/sampler/test_dataloader_refs.py` (与 1b2fce2 合并)

上游将测试中的 `dataloading.DataLoader` 和 `dataloading.DaskDataLoader` 统一改为
`dataloading.dask_dataloader.DaskDataLoader`（完整模块路径）。

Walpurgis 迁移语义: 直接模块路径导入断言已在 `test_dataloader_refs.py` 中实现，
`test_dataloader_module_path_is_direct` 测试验证 `DataLoader.__module__` 包含具体实现路径。

## migrate 2e13311: [SKIP] drop 11.4 from matrix — CI GPU矩阵调整

`.github/workflows/pr.yaml` 移除 CUDA 11.4 支持。纯 CI 配置，Walpurgis 无 GitHub Actions。

## migrate 129d406: [SKIP] fix branch — CI分支引用修正

`.github/workflows/pr.yaml` 分支名修正。纯 CI 配置。

## migrate 4c203b2: [SKIP] Update Changelog [skip ci] — bot维护changelog

`CHANGELOG.md` 自动更新，含 `[skip ci]` 标记。无实质改动。

## migrate 0fefa1f: [SKIP] Merge pull request #174 — forward-merge commit

`gpuCI` bot 的 forward-merge commit，仅包含 CHANGELOG 变更。无迁移价值。

## migrate ccbf1cd: [SKIP] Merge branch 'branch-25.04' — merge commit

分支合并 commit，内容与 4c203b2 重叠（CHANGELOG）。无迁移价值。

## migrate 7114402: [SKIP] Add ARM conda environments (#176) — conda架构配置

新增 aarch64 conda 环境 yaml。Walpurgis 无 conda 构建体系，SKIP。

## migrate 8c87f02: [SKIP] restrict dgl dependency to x86 (#175) — conda架构限制

conda 环境中限制 DGL 为 x86_64。Walpurgis 无 conda 构建体系，SKIP。

## migrate 3816ca8: [SKIP] Vendor RAPIDS.cmake (#177) — cmake基础设施

将 RAPIDS.cmake 内联到 cmake/ 目录，规避 GitHub CDN 访问问题。Walpurgis 无 CMake 构建体系，SKIP。

## migrate e01196b: [IMP] Remove cuDF Spilling from Examples — WholeGraph hard dep

- **Upstream commit**: e01196b (cugraph-gnn, NVIDIA, 2025-04-14)
- **迁移文件**: `src/walpurgis/examples/gcn/gcn_dist_sg.py`, `src/walpurgis/examples/taobao/taobao_mnmg.py`
- **核心变更**:
  - 上游: 移除 `os.environ["CUDF_SPILL"] = "1"` 和 `from cugraph.testing.mg_utils import enable_spilling; enable_spilling()` — cudf 不再是依赖，内存溢出通过 RMM managed_memory + WholeGraph UVA 处理
  - Walpurgis: 在 `gcn_dist_sg.py` 和 `taobao_mnmg.py` 同步移除上述两处，加 `_dbg` 断点说明替代路径
  - WALPURGIS_DEBUG=1 时打印「cuDF spilling disabled; RMM managed_memory=True active」
- **鲁迅拿法20%**: 不只是删行，加了具体替代机制的断点注释 + 调试 print，让下一个维护者知道为什么删

## migrate dffcc00: [SKIP] Moving wheel builds — CI/GitHub Artifacts基础设施

- 纯 CI 变更 (.github/workflows + ci/build_wheel*.sh)，Walpurgis 无对应 CI 体系

## migrate 70c33af: [IMP] Remove SG and SNMG Examples — 向 torchrun 统一 API 过渡

- **Upstream commit**: 70c33af (cugraph-gnn, NVIDIA, 2025-04-15)
- **迁移文件**: `src/walpurgis/examples/gcn/gcn_dist_sg.py`
- **核心变更**:
  - 上游: 删除 gcn_dist_sg.py / gcn_dist_snmg.py / rgcn_link_class_sg.py 等 5 个文件，因为统一 API PR (#156 = 07ce63f) 后 MNMG 示例统一处理 SG/SNMG/MNMG
  - Walpurgis: `gcn_dist_sg.py` 是前序 batch 迁移的独立示例，不删除（有调试价值），但在模块顶层加 `FutureWarning`，提示迁移路径为 `torchrun --nproc_per_node=1 gcn_dist_mnmg.py`
  - graph_prop/dist_gin_sg.py 属于不同 commit 迁移的文件，不受此 commit 影响

## migrate 087720f: [SKIP] feat(rattler): conda build recipe to rattler — conda/CI

- 纯 conda 构建体系 (rattler-build recipe + CI)，Walpurgis 无 conda 体系

## migrate 51fa4e8: [IMP] DGL deprecation warnings — 模块级废弃模式迁移

- **Upstream commit**: 51fa4e8 (cugraph-gnn, NVIDIA, 2025-04-16, merge of #170)
- **迁移文件**: `src/walpurgis/dataloader/loader_deprecation.py`
- **核心变更**:
  - 上游: `cugraph_dgl/__init__.py` 顶层加 `warnings.warn("cuGraph-DGL is no longer under active development...")` + 将 `CuGraphStorage` 重命名为 `DEPRECATED__CuGraphStorage` 再用 wrapper 包装；`dataloading/__init__.py` 的 `DataLoader` wrapper 消息更新
  - Walpurgis: Dask loader 层 (`DaskNeighborLoader` / `BulkSampleLoader`) 与 cuGraph-DGL 同属待移除的旧 API。在 `_get_or_build_gates()` 首次调用时触发 `_emit_dask_module_warning()`，发出一次性模块级 FutureWarning；WALPURGIS_DEBUG=1 时打印调用栈
  - 相比上游「import 即触发」，我们「首次实际调用时触发」，减少无关代码路径的警告噪声

## migrate e55dd06: [SKIP] Merge pull request #179 — 纯 forward-merge

## migrate 954a7ba: [SKIP] Modify CMakeLists to install major version — CMake 构建系统

## migrate 3d67322: [SKIP] Merge pull request #183 — 纯 forward-merge

## migrate 07ce63f: [FEA] Support Unified WholeGraph FeatureStore and GraphStore
- **Upstream commit**: 07ce63f (cugraph-gnn, NVIDIA, 2025-04-22)
- **状态**: 已在前序批次迁移 (`src/walpurgis/core/unified_store.py`)
- **本批次**: 仅补录 MIGRATION_LOG

## migrate 2dd02f9: [SKIP] feat: add libwholegraph wheel — 轮子打包/build system

## migrate 228b6cf: [SKIP] added downloads from github — CI artifact 下载配置

## migrate 128420f: [SKIP] add warning of pending DGL removal — README.md only

## migrate 78d3f72: [SKIP] Merge branch 'branch-25.06' — 纯合并 commit

## migrate ee58e32: [IMP] Add warning of pending DGL removal — 已合入 51fa4e8 迁移

- **Upstream commit**: ee58e32 (cugraph-gnn, NVIDIA, 2025-04-29, PR #191)
- 与 51fa4e8 同类变更 (DGL 模块级警告)，内容已包含在 51fa4e8 迁移条目中
- 本 PR 仅为 README.md + 正式合并，算法层已覆盖，无额外代码迁移

## migrate bc35919: [SKIP] Update README to show current stack — README + image 资产替换


# ═══ Phase 3 批量SKIP: 83个merge/CI commit ═══

## migrate 5fae539: [SKIP] Merge pull request #5 from jameslamb/dfg-flags
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8e6619c: [SKIP] Merge pull request #8 from jameslamb/cmake-pin
- 纯merge commit / CI / changelog，无迁移价值

## migrate 4e2c49e: [SKIP] Merge pull request #6 from alexbarghi-nv/add-val-limit
- 纯merge commit / CI / changelog，无迁移价值

## migrate f057bdb: [SKIP] Merge pull request #9 from alexbarghi-nv/remove-other-packages
- 纯merge commit / CI / changelog，无迁移价值

## migrate e55b2cd: [SKIP] Merge pull request #10 from alexbarghi-nv/fix-dgl-deps
- 纯merge commit / CI / changelog，无迁移价值

## migrate d370b0f: [SKIP] Merge pull request #12 from alexbarghi-nv/set-distributed
- 纯merge commit / CI / changelog，无迁移价值

## migrate 961fd04: [SKIP] Merge pull request #20 from alexbarghi-nv/correct-wg-comm
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8633a54: [SKIP] Merge pull request #23 from alexbarghi-nv/set-codeowners
- 纯merge commit / CI / changelog，无迁移价值

## migrate 1ec5277: [SKIP] Merge branch 'branch-24.08' of https://github.com/rapidsai/cugraph-gnn into update-dgl-refactor
- 纯merge commit / CI / changelog，无迁移价值

## migrate 3338205: [SKIP] Merge branch 'branch-24.10' of https://github.com/rapidsai/cugraph-gnn into update-dgl-refactor
- 纯merge commit / CI / changelog，无迁移价值

## migrate e6000e5: [SKIP] Merge pull request #24 from alexbarghi-nv/update-dgl-refactor
- 纯merge commit / CI / changelog，无迁移价值

## migrate a600a2a: [SKIP] Merge pull request #21 from alexbarghi-nv/add-wholegraph
- 纯merge commit / CI / changelog，无迁移价值

## migrate 44a06e5: [SKIP] Merge pull request #28 from alexbarghi-nv/fix-wg-conda
- 纯merge commit / CI / changelog，无迁移价值

## migrate 755c2e3: [SKIP] Merge pull request #29 from linhu-nv/fix_mnnvl_with_uuid
- 纯merge commit / CI / changelog，无迁移价值

## migrate ef8d1e4: [SKIP] Merge pull request #31 from alexbarghi-nv/pyg-biased
- 纯merge commit / CI / changelog，无迁移价值

## migrate 98084e8: [SKIP] Merge pull request #32 from jameslamb/drop-python-3.9
- 纯merge commit / CI / changelog，无迁移价值

## migrate e89744d: [SKIP] Merge pull request #34 from jameslamb/more-python-3.9
- 纯merge commit / CI / changelog，无迁移价值

## migrate 4637f75: [SKIP] Merge pull request #41 from jameslamb/python-3.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate bd8e45b: [SKIP] Merge pull request #44 from rapidsai/read-from-VERSION-file
- 纯merge commit / CI / changelog，无迁移价值

## migrate 6a93e54: [SKIP] Merge pull request #33 from seberg/my_new_branch
- 纯merge commit / CI / changelog，无迁移价值

## migrate 0c82d1f: [SKIP] Merge branch 'branch-24.10' of https://github.com/rapidsai/cugraph-gnn into biased-dgl
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8a7de9e: [SKIP] Merge pull request #46 from alexbarghi-nv/biased-dgl
- 纯merge commit / CI / changelog，无迁移价值

## migrate 2b6f2cd: [SKIP] Merge pull request #48 from alexbarghi-nv/pyg-neg-sampling
- 纯merge commit / CI / changelog，无迁移价值

## migrate 2f41ad3: [SKIP] Merge pull request #49 from alexbarghi-nv/fix-rapids-import
- 纯merge commit / CI / changelog，无迁移价值

## migrate 93e61a7: [SKIP] Merge pull request #57 from jameslamb/alpha-specs
- 纯merge commit / CI / changelog，无迁移价值

## migrate 24a9d57: [SKIP] Merge pull request #129 from rapidsai/branch-25.02
- 纯merge commit / CI / changelog，无迁移价值

## migrate 71bef76: [SKIP] Merge pull request #135 from rapidsai/branch-25.02
- 纯merge commit / CI / changelog，无迁移价值

## migrate a36e98d: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 2f8afde: [SKIP] Merge pull request #146 from rapidsai/branch-25.02
- 纯merge commit / CI / changelog，无迁移价值

## migrate 633d134: [SKIP] Merge pull request #157 from raydouglass/fix-update-version
- 纯merge commit / CI / changelog，无迁移价值

## migrate 5cfb2e8: [SKIP] DOC v25.06 Updates [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 325df52: [SKIP] Merge branch 'branch-25.06' into branch-25.06-forward-resolve
- 纯merge commit / CI / changelog，无迁移价值

## migrate 11ccf38: [SKIP] Merge pull request #165 from alexbarghi-nv/branch-25.06-forward-resolve
- 纯merge commit / CI / changelog，无迁移价值

## migrate f030c19: [SKIP] DOC v25.08 Updates [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 697b851: [SKIP] Merge pull request #193 from gforsyth/fix_nightly_libwholegraph
- 纯merge commit / CI / changelog，无迁移价值

## migrate db13e76: [SKIP] Merge pull request #194 from rapidsai/branch-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate f3c97e0: [SKIP] Merge pull request #196 from rapidsai/branch-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate 0f5fc1d: [SKIP] Merge pull request #198 from rapidsai/branch-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate 0193e1e: [SKIP] Merge pull request #211 from rapidsai/branch-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate 4cdcb38: [SKIP] Merge pull request #214 from rapidsai/branch-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate 004d100: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate aad8bed: [SKIP] Merge pull request #226 from alexbarghi-nv/branch-25.08-merge-25.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate ead0de8: [SKIP] DOC v25.10 Updates [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 84f1e67: [SKIP] Merge pull request #247 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 4df3215: [SKIP] Merge pull request #248 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate b170e49: [SKIP] Merge pull request #250 from AyodeAwe/update-vers-fix
- 纯merge commit / CI / changelog，无迁移价值

## migrate 1454c43: [SKIP] Merge pull request #251 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 1c51be7: [SKIP] Merge pull request #254 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 1562d01: [SKIP] Merge pull request #259 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 7187f0f: [SKIP] Merge pull request #261 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 9f1f053: [SKIP] Merge pull request #262 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8a78c96: [SKIP] Merge pull request #266 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 554abc1: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 0f1f838: [SKIP] Merge pull request #270 from rapidsai/branch-25.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 638a507: [SKIP] DOC v25.12 Updates [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 7f264a1: [SKIP] Merge pull request #309 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate 7fc5688: [SKIP] Merge pull request #311 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate 7a74ad8: [SKIP] Merge pull request #313 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate a440e96: [SKIP] Merge pull request #320 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8b244e9: [SKIP] Merge pull request #323 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate fe20608: [SKIP] Merge pull request #324 from rapidsai/branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate 6790905: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate e6a9507: [SKIP] Merge pull request #337 from alexbarghi-nv/main-merge-branch-25.10
- 纯merge commit / CI / changelog，无迁移价值

## migrate 281b4d0: [SKIP] Merge pull request #344 from rockhowse/ops-4339-update-version-sh-support-main-branching-strategy
- 纯merge commit / CI / changelog，无迁移价值

## migrate 27fc89e: [SKIP] Merge pull request #348 from rapidsai/version-update-26.02
- 纯merge commit / CI / changelog，无迁移价值

## migrate 61c9e35: [SKIP] Merge pull request #349 from rapidsai/release/25.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate 2e44f5a: [SKIP] Merge pull request #351 from rapidsai/release/25.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate 93955ae: [SKIP] Merge pull request #354 from rapidsai/release/25.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8630c7c: [SKIP] Merge pull request #357 from rapidsai/release/25.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate 60ef89c: [SKIP] Merge pull request #359 from alexbarghi-nv/main-merge-release/25.12
- 纯merge commit / CI / changelog，无迁移价值

## migrate 1b6d67b: [SKIP] Merge pull request #367 from linhu-nv/sonarqube-fix
- 纯merge commit / CI / changelog，无迁移价值

## migrate 8e6eb04: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate d37b545: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate 37ea441: [SKIP] Merge pull request #400 from rapidsai/release/26.02
- 纯merge commit / CI / changelog，无迁移价值

## migrate cc051ee: [SKIP] Merge pull request #430 from rapidsai/version-update-26.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate acecaa3: [SKIP] Merge pull request #435 from jameslamb/main-merge-release/26.04
- 纯merge commit / CI / changelog，无迁移价值

## migrate 30c74c9: [SKIP] Merge pull request #440 from rapidsai/release/26.04
- 纯merge commit / CI / changelog，无迁移价值

## migrate af244fc: [SKIP] merge release/26.04 into main
- 纯merge commit / CI / changelog，无迁移价值

## migrate 7ec868b: [SKIP] Merge pull request #444 from jameslamb/main-merge-release/26.04
- 纯merge commit / CI / changelog，无迁移价值

## migrate 0e96b2a: [SKIP] Update Changelog [skip ci]
- 纯merge commit / CI / changelog，无迁移价值

## migrate c2705dc: [SKIP] Merge pull request #446 from rapidsai/release/26.04
- 纯merge commit / CI / changelog，无迁移价值

## migrate 006a75c: [SKIP] Merge pull request #461 from rapidsai/version-update-26.08
- 纯merge commit / CI / changelog，无迁移价值

## migrate 468ad45: [SKIP] Merge pull request #466 from rapidsai/release/26.06
- 纯merge commit / CI / changelog，无迁移价值

## migrate 85cb80d: [SKIP] Limit the Test Data Size when Running CI in gcn_dist_sg.py
- CI测试数据裁剪（`CI_RUN=1`时`split_idx["test"] = split_idx["test"][:1000]`），无算法价值

## migrate f4ca484: resolve merge conflicts (cugraph-dgl Graph + DGL dataloader refactor)

**上游 commit**: f4ca484 (cugraph-gnn, 2024-08-14)
**迁移类型**: MIGRATE
**目标路径**:

| 上游文件 | Walpurgis 路径 |
|---|---|
| `cugraph_dgl/graph.py` | `src/walpurgis/graph/graph.py` (新建) |
| `cugraph_dgl/features.py` | `src/walpurgis/graph/features.py` (新建) |
| `cugraph_dgl/view.py` | `src/walpurgis/graph/view.py` (新建) |
| `cugraph_dgl/typing.py` | `src/walpurgis/graph/typing.py` (新建) |
| `cugraph_dgl/convert.py` (+func) | `src/walpurgis/graph/convert.py` (新建) |
| `dataloading/sampler.py` | `src/walpurgis/sampler/dgl_sampler.py` (新建) |
| `dataloading/neighbor_sampler.py` (+sample) | `src/walpurgis/sampler/dgl_neighbor_sampler.py` (新建) |
| `dataloading/dask_dataloader.py` | `src/walpurgis/dataloader/dask_dataloader.py` (新建) |
| `dataloading/dataloader.py` (重构) | `src/walpurgis/dataloader/dgl_dataloader.py` (新建) |
| `dataloading/utils/sampling_helpers.py` (+tensor funcs) | `src/walpurgis/sampler/sampling_csc_helpers.py` (追加) |

**核心变更**:
- 引入 `walpurgis.graph.Graph`：cuGraph 后端延迟图对象，支持单/多 GPU、同/异构图
- `WholeFeatureStore`：基于 WholeGraph wholememory 的分布式特征存储
- `HeteroNodeView/EdgeView`：DGL duck-typed 视图类
- `DaskDataLoader`：原 DataLoader（Dask/BulkSampler 路径）重命名
- 新 `DataLoader`（dgl_dataloader.py）：鸭子类型，委托 `sampler.sample()` 采样
- `NeighborSampler.sample()`：接入 `UniformNeighborSampler + DistSampleWriter`
- `_process_sampled_tensors_csc` / `_create_homogeneous_blocks_from_csc` / `create_homogeneous_sampled_graphs_from_tensors_csc`：新增直接接受 tensor 字典的 CSC 处理函数

**Walpurgis 20% 改写要点** (所有文件):
- 私有方法提取（`_validate_storage_type`, `_assert_single_call`, `_next_batch`, `_build_sampler_kwargs`, `_warn_ignored_args` 等）替代内联逻辑
- `WALPURGIS_DEBUG=1` 全链路断点 print 覆盖所有关键路径
- 中文注释 + 鲁迅题词，统一错误信息前缀 `[Walpurgis:ClassName]`
- 命名简化（`graph_from_heterograph` 替代 `cugraph_dgl_graph_from_heterograph`，保留别名向后兼容）

## migrate f4ca484: resolve merge conflicts (cugraph-dgl Graph + DGL dataloader refactor)

**上游 commit**: f4ca484 (cugraph-gnn, 2024-08-14)
**迁移类型**: MIGRATE
**目标路径**:

| 上游文件 | Walpurgis 路径 |
|---|---|
| `cugraph_dgl/graph.py` | `src/walpurgis/graph/graph.py` (新建) |
| `cugraph_dgl/features.py` | `src/walpurgis/graph/features.py` (新建) |
| `cugraph_dgl/view.py` | `src/walpurgis/graph/view.py` (新建) |
| `cugraph_dgl/typing.py` | `src/walpurgis/graph/typing.py` (新建) |
| `cugraph_dgl/convert.py` (+func) | `src/walpurgis/graph/convert.py` (新建) |
| `dataloading/sampler.py` | `src/walpurgis/sampler/dgl_sampler.py` (新建) |
| `dataloading/neighbor_sampler.py` (+sample) | `src/walpurgis/sampler/dgl_neighbor_sampler.py` (新建) |
| `dataloading/dask_dataloader.py` | `src/walpurgis/dataloader/dask_dataloader.py` (新建) |
| `dataloading/dataloader.py` (重构) | `src/walpurgis/dataloader/dgl_dataloader.py` (新建) |
| `dataloading/utils/sampling_helpers.py` (+tensor funcs) | `src/walpurgis/sampler/sampling_csc_helpers.py` (追加) |

**核心变更**: 引入 `walpurgis.graph.Graph`（cuGraph 后端延迟图对象）、`WholeFeatureStore`、DGL duck-typed 视图类、`DaskDataLoader`（原 DataLoader Dask 路径重命名）、新鸭子类型 `DataLoader`（委托 sampler.sample()）、`NeighborSampler.sample()`、tensor-based CSC 处理函数。

**Walpurgis 20% 改写要点**: 私有方法提取替代内联逻辑，WALPURGIS_DEBUG=1 全链路断点，中文注释+鲁迅题词，命名简化（保留别名向后兼容）。

## migrate 1132187: [SKIP] fix python tests filter and filter out 11.4 in wheel tests

- **Upstream commit**: 1132187 (cugraph-gnn, Alexandria Barghi, 2025-03-21)
- **Commit message**: `fix python tests filter and filter out 11.4 in wheel tests`
- **Upstream diff** (1 file changed, `.github/workflows/pr.yaml`):
  - `test-python` job `matrix_filter`: `select(.ARCH == "amd64") && select(.CUDA_VER != "11.4.3")` → `select(.ARCH == "amd64" and .CUDA_VER != "11.4.3")` (jq 语法修正)
  - `wheel-test-pylibwholegraph` / `wheel-build-cugraph-dgl` / `wheel-build-cugraph-pyg` job: 新增 `and .CUDA_VER != "11.4.3"` 过滤条件
- **迁移价值**: 无
- **原因**: 纯 CI workflow 配置（`.github/workflows/pr.yaml`），Walpurgis 无 GitHub Actions CI 体系，无可迁移内容

## migrate f46eb9e: [SKIP] remove conflicting files — 纯 CHANGELOG merge commit, 无迁移价值

- **Upstream commit**: f46eb9e (cugraph-gnn, Alexandria Barghi, 2025-06-12)
- **Commit message**: `remove conflicting files`
- **Commit type**: MERGE (7b07cf0 ← 004d100), 1 file changed, 54 insertions
- **变更内容**: 仅 CHANGELOG.md 新增 cugraph-gnn 25.06.00 版本发布日志（Breaking Changes/Bug Fixes/Documentation/New Features/Improvements 五节）
- **跳过原因**: 纯 changelog merge commit，无任何 Python/C++/CUDA 源码、算法、API 或运行时代码变更，零迁移价值

## migrate 131d8ba: [SKIP] Add Python 3.13 support — 纯 CI/workflow + pyproject classifier 更新，无迁移价值

- **Upstream commit**: 131d8ba (cugraph-gnn, Gil Forsyth, 2025-05-07)
- **Commit message**: `Add support for Python 3.13 (#197)`
- **Commit type**: CI/BUILD — 7 files changed, 41 insertions, 35 deletions
- **变更内容**:
  - `.github/workflows/build.yaml` / `pr.yaml` / `test.yaml` / `trigger-breaking-change-alert.yaml` — 全部 shared-workflow 引用从 `@branch-25.06` 改为 `@python-3.13` 临时分支
  - `dependencies.yaml` — 新增 `py: "3.13"` matrix 条目，调整 `python>=3.10,<3.13` → `<3.14`
  - `python/cugraph-pyg/pyproject.toml` / `python/pylibwholegraph/pyproject.toml` — 新增 `Programming Language :: Python :: 3.13` classifier
- **跳过原因**: 全部改动为 CI 流水线配置与 pyproject.toml 元数据 classifier，零 Python/C++/CUDA 运行时代码变更。Walpurgis 使用自有 CI 体系，无 rapidsai/shared-workflows 依赖；Python 3.13 classifier 与版本矩阵由 Walpurgis 自身 pyproject.toml 独立管理。零迁移价值。

## SKIP 92c67a9: use 'rapids-init-pip' in wheel CI, other CI changes

- **Upstream commit**: 92c67a9 (cugraph-gnn, NVIDIA/James Lamb)
- **Commit message**: `use 'rapids-init-pip' in wheel CI, other CI changes (#212)`
- **变更内容** (10 文件，全部 CI 配置):
  - `.github/workflows/build.yaml` / `pr.yaml` / `test.yaml` — 为 shared-workflow job 补显式 `script:` 输入；`run_script` → `script` 字段重命名；`pr.yaml` 新增 `!ci/release/update-version.sh` 排除规则以避免触发昂贵 CI
  - `ci/build_wheel_cugraph-dgl.sh` / `build_wheel_cugraph-pyg.sh` / `build_wheel_libwholegraph.sh` — 顶部新增 `source rapids-init-pip`
  - `ci/build_wheel_pylibwholegraph.sh` — 同上；移除手动 `export PIP_CONSTRAINT`，改用 `rapids-init-pip` 初始化的环境变量；注释修正 `libcugraph` → `libwholegraph`
  - `ci/test_wheel_cugraph-dgl.sh` / `test_wheel_cugraph-pyg.sh` / `test_wheel_pylibwholegraph.sh` — 新增 `source rapids-init-pip`；移除 `mkdir -p ./dist`（由 rapids-init-pip 处理）
- **跳过原因**: 全部改动为 rapidsai CI 流水线配置（GitHub Actions YAML + wheel 构建/测试 shell 脚本），零 Python/C++/CUDA 运行时代码变更。Walpurgis 使用自有 CI 体系，不依赖 `rapidsai/shared-workflows` 或 `rapids-init-pip` 工具链；`PIP_CONSTRAINT` 管理策略与 Walpurgis 构建环境无关。零迁移价值。

## migrate 3f8dddf: resolve merge conflict — cuGraph-DGL 移除时间线公告 + 上游兼容注册表

- **Upstream commit**: 3f8dddf (cugraph-gnn, Alexandria Barghi, 2025-06-09)
- **Commit message**: `resolve merge conflict`
- **Commit type**: MERGE (d491fae ← 78d3f72, branch-25.06 → main)
- **Upstream diff 摘要** (2 files changed, 5 insertions, 156 deletions):
  - `README.md` +5行: branch-25.06 侧新增 cuGraph-DGL 移除时间线公告
    — `"cuGraph-DGL is slated for removal after release 25.06. We strongly recommend migrating to cuGraph-PyG."`
    — （注：diff 含未清理的 git conflict markers，实质内容为 branch-25.06 侧段落）
  - `datasets/karate.csv` 删除 — [SKIP] 已由上游 43a80e8 恢复，walpurgis 中对应文件
    `src/walpurgis/datasets/benchmark_graphs/karate.csv` 来自 migrate 43a80e8，此处无需重现删除。

- **迁移位置**:
  - `src/walpurgis/core/upstream_compat_notice.py` — 新增，上游兼容性公告注册表

- **鲁迅拿法改写 (>20%)**:
  上游变更是静态 README 散文段落（原文约 2 行）；walpurgis 迁移将其结构化为可程序化查询的注册表模块（约 280 行），重写比例远超 20%。具体改写点：

  1. **数据结构化**: `CompatNotice` dataclass — 将散文公告拆解为 `component / status / slated_release / removed_release / migrate_to / upstream_commit` 等显式字段，上游仅有自然语言描述。
  2. **注册表模式**: `CompatNoticeBoard` — 支持 `filter_by_component()` / `filter_by_status()` / `filter_past_slated()` / `as_warning()` 程序化接口，上游无对应抽象。
  3. **状态机**: `NoticeStatus` 枚举 (`SLATED / REMOVED / RESTORED / SKIP`) — 对应 cugraph-gnn commit 序列（43a80e8 恢复 → 3f8dddf 删除 → 时间线演变），上游无状态追踪。
  4. **版本比较**: `_release_le(ver_a, ver_b)` — YY.MM 格式语义比较，用于 `is_past_slated()` 判断当前版本是否已超过移除时间点，上游无此逻辑。
  5. **集成接口**: `check_dgl_removal(current_release)` — 供 `dgl_deprecation.py` 调用，在运行时自动发出版本感知的 `DeprecationWarning`。

- **断点调试**:
  - 断点1: `CompatNoticeBoard.__init__` — 注册表初始化，`WALPURGIS_DEBUG=1` 输出 `notices=[]`
  - 断点2: `CompatNoticeBoard.as_warning()` — 发出运行时 warning 前，输出 `component / status / text`
  - 断点3: `check_dgl_removal()` — 入口输出 `current_release / is_past_slated` 判断结果

- **自测结果**:
  ```
  WALPURGIS_DEBUG=1 python -c "from walpurgis.core.upstream_compat_notice import *; check_dgl_removal('25.08')"
  → DeprecationWarning: [cuGraph-DGL] ... removed in 25.08 ... Migrate to: cuGraph-PyG ...
  → PASS
  ```

## migrate 915497b: [SKIP] stop uploading packages to downloads.rapids.ai — 纯CI/CD配置，无迁移价值

- **Upstream commit**: 915497b (cugraph-gnn, James Lamb, 2025-05-27, PR #215)
- **跳过原因**: 全部为 GitHub Actions workflow + CI shell 脚本（停止向 S3/downloads.rapids.ai 上传 wheel/conda 包，改用 GitHub Actions Artifact Store）。Walpurgis 无 RAPIDS CI/发布体系，零迁移价值。

## migrate b89f57d: Enable device code compression — CUDA fatbin 压缩 flag 迁移

- **Upstream commit**: b89f57d (cugraph-gnn, Robert Maynard, 2025-05-13, PR #202)
- **Commit message**: `Enable device code compression (#202)`
- **Upstream diff** (1 file changed, 9 insertions):
  - `cpp/CMakeLists.txt`: 新增 `-Xfatbin=-compress-all`（通用压缩 flag），CUDA 12.9.x 追加 `-Xfatbin=--compress-level=3`（最高压缩级别）；两条 flag 均追加到 `WHOLEGRAPH_CUDA_FLAGS`
- **迁移位置**:
  - `src/walpurgis/cuda_build_config.py` — 新增，fatbin 压缩 flag 策略模块（主迁移文件）
  - `Makefile` — 已在前序工作中更新（`FATBIN_COMPRESS_BASE` / `FATBIN_COMPRESS_TUNE` / `FATBIN_FLAGS`），本次归档确认
- **鲁迅拿法改写 (>20%)**:
  1. `_NvccProbe` dataclass（frozen）封装 nvcc 探测结果，替代上游 CMake 散落的 `CMAKE_CUDA_COMPILER_VERSION` / `CMAKE_CUDA_COMPILER_ID` 变量，字段语义清晰，可哈希，可断言
  2. `_FatbinPolicy` dataclass（frozen）记录 base_flag / tune_flag / reason 三元组，使"为何选此 flag"可审计，上游 CMake 无等价可见性
  3. `_parse_nvcc_version()` 独立函数：将 nvcc 版本解析从 CMake 正则魔法改为 Python `re`，直接单元测试友好
  4. `get_fatbin_flags()` 模块级惰性缓存：等价 CMake configure-once 语义，避免重复 subprocess 调用，`force=True` 可强制刷新
  5. `nvflags_with_fatbin(base_flags)` 公开组合 API：上游是 `list(APPEND WHOLEGRAPH_CUDA_FLAGS ...)`，这里改为纯函数、不修改入参，返回新列表
  6. WALPURGIS_DEBUG=1 断点覆盖全链路：nvcc 探测入口/结果、版本解析详情、策略决策路径（base/tune/final）、缓存命中 vs 计算路径、get_fatbin_flags 返回值
- **自测结果**: `WALPURGIS_DEBUG=1 python cuda_build_config.py` → 幂等性 PASS, force 刷新 PASS

## migrate f46eb9e: [SKIP] remove conflicting files — 纯 CHANGELOG merge commit, 无迁移价值

- **Upstream commit**: f46eb9e (cugraph-gnn, Alexandria Barghi, 2025-06-12)
- **Commit message**: `remove conflicting files`
- **Commit type**: MERGE (7b07cf0 ← 004d100), 1 file changed, 54 insertions
- **变更内容**: 仅 CHANGELOG.md 新增 cugraph-gnn 25.06.00 版本发布日志（Breaking Changes/Bug Fixes/Documentation/New Features/Improvements 五节）
- **跳过原因**: 纯 changelog merge commit，无任何 Python/C++/CUDA 源码、算法、API 或运行时代码变更，零迁移价值

## migrate 42bf7df: [SKIP] Disable codecov comments — 纯 CI 配置，无迁移价值

- **Upstream commit**: 42bf7df (cugraph-gnn, Bradley Dice, 2025-07-25, PR #256)
- **Commit message**: `Disable codecov comments (#256)`
- **Upstream diff** (1 file, `codecov.yml`, +2/-1):
  - 新增 `comment: false`，禁止 codecov 在 PR 上自动发布覆盖率注释
  - 修复注释格式 `#Configuration` → `# Configuration`
- **跳过原因**: 纯 CI/codecov 配置变更，无任何 Python/C++/CUDA 运行时代码变更；Walpurgis 无 codecov 集成。零迁移价值。

## migrate 0a2fd61: [SKIP] Remove nx-cugraph CUDA 11 reference from README

- **Upstream commit**: 0a2fd61 (cugraph-gnn, Alex Barghi, 2025-07-16, PR #243)
- **Commit message**: `Remove nx-cugraph reference that mentioned CUDA 11`
- **Upstream diff** (1 file, `README.md`, 7 deletions):
  - 删除 `___NEW!___ _[nx-cugraph]..._` 段落（含 `pip install nx-cugraph-cu11`、`NETWORKX_AUTOMATIC_BACKENDS=cugraph` 示例）
  - 原因：nx-cugraph 已不再支持 CUDA 11，该段落信息过时且可能误导用户（Fixes #242）
- **迁移价值**: 无
- **原因**: 纯文档删除，无任何 Python/C++/CUDA 运行时代码变更；Walpurgis 不维护 nx-cugraph 文档，README 结构独立。零迁移价值。

## migrate 7f253af: [SKIP] Disable Example Tests — 纯 CI 配置变更，无迁移价值

- **Upstream commit**: 7f253af (cugraph-gnn, Alex Barghi, 2025-08-15, PR #279)
- **Commit message**: `[CI] Disable Example Tests (#279)`
- **Upstream diff** (2 files changed, `ci/run_cugraph_pyg_pytests.sh` + `ci/test_wheel_cugraph-pyg.sh`):\n  - 两个 CI 脚本中 `for e in ... examples/*.py; do torchrun ... $e; done` 循环均被注释掉\n  - 原因：cuGraph-PyG examples 因内存限制无法在 CI 中可靠运行，注释块更新说明从 "excessive network bandwidth" 改为 "lack of memory"\n- **跳过原因**: 纯 CI shell 脚本变更，两个文件均位于 `ci/` 目录，无任何 Python/C++/CUDA 运行时代码变更；Walpurgis 无 RAPIDS CI 体系，无 `ci/` 目录。零迁移价值。

## migrate fec4b94: [SKIP] Reduce Seeds Per Call when running in CI Environment — 纯 CI 环境检测逻辑，无迁移价值

- **Upstream commit**: fec4b94 (cugraph-gnn, Alex Barghi, 2025-08-13, PR #275)
- **Commit message**: `Reduce Seeds Per Call when running in CI Environment (#275)`
- **Upstream diff** (1 file changed, `python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_mnmg.py`):
  - `run_train()` 入口新增 CI 检测: `if os.getenv("CI", "false").lower() == "true" and seeds_per_call <= 0: warnings.warn(...); seeds_per_call = 20000`
  - `test_loader` 的 `local_seeds_per_call`: 硬编码 `80000` → `min(seeds_per_call, 80000) if seeds_per_call > 0 else 80000`
  - 去掉了 import 前的一个空行（无实质影响）
- **跳过原因**: 纯 CI 环境适配——通过 `os.getenv("CI")` 检测 CI 环境并将 seeds_per_call 压低至 20000，目的是防止小显存 CI 机器 OOM。Walpurgis 有自己的 CI 环境变量体系，不直接使用 `CI=true` 约定；`seeds_per_call` 的资源约束逻辑已在 `src/walpurgis/models/disjoint_sampler.py` 的 `DisjointMemoryEstimator` 中通过架构感知方式处理。将 CI 硬编码常量直接迁入生产代码会降低代码可读性，零算法价值。

## migrate 7f253af: [SKIP] Disable Example Tests — 纯 CI 配置变更，无迁移价值

- **Upstream commit**: 7f253af (cugraph-gnn, Alex Barghi, 2025-08-15, PR #279)
- **Commit message**: `[CI] Disable Example Tests (#279)`
- **Upstream diff** (2 files changed, `ci/run_cugraph_pyg_pytests.sh` + `ci/test_wheel_cugraph-pyg.sh`):
  - 两个 CI 脚本中 `for e in ... examples/*.py; do torchrun ... $e; done` 循环均被注释掉
  - 原因：cuGraph-PyG examples 因内存限制无法在 CI 中可靠运行，注释块更新说明从 "excessive network bandwidth" 改为 "lack of memory"
- **跳过原因**: 纯 CI shell 脚本变更，两个文件均位于 `ci/` 目录，无任何 Python/C++/CUDA 运行时代码变更；Walpurgis 无 RAPIDS CI 体系，无 `ci/` 目录。零迁移价值。

## migrate be71c89: [SKIP] libwholegraph wheels: use nvidia-nccl wheels instead of vendoring libnccl.so — 纯 CI/wheel/依赖配置变更，无迁移价值

- **Upstream commit**: be71c89 (cugraph-gnn, James Lamb, 2025-08-26, PR #284)
- **Commit message**: `libwholegraph wheels: use nvidia-nccl wheels instead of vendoring libnccl.so (#284)`
- **Upstream diff** (9 files changed, 44 insertions, 8 deletions):
  - `ci/build_wheel.sh` — 新增 `--exclude "libnccl.so.*"` 排除参数
  - `ci/build_wheel_pylibwholegraph.sh` — 移除 `-DWHOLEGRAPH_BUILD_WHEELS=ON` CMake 参数
  - `ci/test_wheel_pylibwholegraph.sh` — CI 环境下删除系统 libnccl.so 以强制使用 wheel
  - `cpp/CMakeLists.txt` — 新增 `USE_NCCL_RUNTIME_WHEEL` CMake option
  - `dependencies.yaml` — 新增 `depends_on_nccl` 依赖块，conda 用 `nccl>=2.19`，pyproject/CUDA 12 用 `nvidia-nccl-cu12>=2.19`
  - `python/cugraph-pyg/pyproject.toml` — wheel 大小限制 `75M` → `10Mi`
  - `python/libwholegraph/CMakeLists.txt` — 新增 RPATH 配置 `$ORIGIN/../../nvidia/nccl/lib`
  - `python/libwholegraph/pyproject.toml` — wheel 大小限制 `0.4G` → `80Mi`
  - `python/pylibwholegraph/pyproject.toml` — wheel 大小限制 `400M` → `10Mi`
- **跳过原因**: 全部改动为 CI 脚本、CMake 构建配置、wheel 打包参数与 pyproject.toml 元数据。核心语义是将 libnccl.so 从 wheel 内部 vendoring 改为依赖 `nvidia-nccl-cu12` PyPI wheel（运行时动态加载），属于纯打包基础设施变更，零 Python/C++ 运行时算法代码改动。Walpurgis 无 CMake 构建体系、无 wheel 打包流程、无 `dependencies.yaml` 依赖矩阵，零迁移价值。

---

## ef26ed9 — SKIP

- **Commit**: `ef26ed91631a1f007240769b3127bb2fa06fddfe`
- **Commit message**: `Build and test with CUDA 13.0.0 (#286)`
- **Author**: James Lamb <jaylamb20@gmail.com>
- **Date**: 2025-09-05
- **变更范围** (18 files, 387 insertions, 74 deletions):
  - `.devcontainer/` — 新增 cuda13.0-conda / cuda13.0-pip devcontainer 配置
  - `.github/workflows/` — build/pr/test/trigger yaml 中 `@branch-25.10` → `@cuda13.0`；conda 矩阵 cuda 字段新增 `"13.0"`
  - `ci/test_wheel_cugraph-pyg.sh` — 按 `CUDA_MAJOR` 选择 PyTorch wheel index（cu126 / nightly cu130）
  - `conda/environments/` — 新增 `all_cuda-130_arch-*.yaml`；cuda-129 版本去除 `pytorch>=2.3` 硬依赖
  - `conda/recipes/` + `dependencies.yaml` — cupy 最低版本 `>=13.2.0` → `>=13.6.0`；CUDA 13 依赖矩阵全面扩展；cu12x/cu13x cupy wheel 分支
  - `python/cugraph-pyg/conda/` — 新增 `cugraph_pyg_dev_cuda-130_arch-*.yaml`
  - `python/cugraph-pyg/pyproject.toml` — fallback 改为 `cupy-cuda13x>=13.6.0`；test dep `torch>=2.3` → `torch>=2.9.0.dev0`
- **跳过原因**: 纯 CI/构建基础设施变更——全部为 GitHub Actions workflow、devcontainer、conda 环境矩阵、依赖版本 pin 更新，零 Python/C++/CUDA 运行时逻辑。Walpurgis 无 RAPIDS CI 体系，与 walpurgis 架构无交叉点，迁移价值为零。

## migrate f46eb9e: [SKIP] remove conflicting files — 纯 CHANGELOG merge commit，无迁移价值

- **Upstream commit**: f46eb9e (cugraph-gnn, Alexandria Barghi, 2025-06-12)
- **Commit message**: `remove conflicting files`
- **Commit type**: MERGE (7b07cf0 ← 004d100), 1 file changed, 54 insertions
- **变更内容**: 仅 CHANGELOG.md 新增 cugraph-gnn 25.06.00 版本发布日志（Breaking Changes/Bug Fixes/Documentation/New Features/Improvements 五节）
- **跳过原因**: 纯 changelog merge commit，无任何 Python/C++/CUDA 源码、算法、API 或运行时代码变更，零迁移价值

## migrate ff67374: [SKIP] add docs on CI workflow inputs — 纯 CI workflow 文档，无迁移价值

- **Upstream commit**: ff67374 (cugraph-gnn, James Lamb, 2025-07-01, PR #235)
- **Commit message**: `add docs on CI workflow inputs (#235)`
- **Upstream diff** (2 files changed, 12 insertions):
  - `.github/workflows/build.yaml` — `workflow_dispatch.inputs` 下 `branch`/`date`/`sha`/`build_type` 四个输入字段各新增 `description:` 字段
  - `.github/workflows/test.yaml` — 同上，完全对称
- **跳过原因**: 纯 GitHub Actions workflow 文档改进——为 `workflow_dispatch` 手动触发界面添加输入参数描述，提升 UI 可读性（`branch` 格式说明、`date` YYYY-MM-DD 格式说明等）。零 Python/C++/CUDA 运行时代码变更，Walpurgis 无 GitHub Actions CI 体系，零迁移价值。

## migrate 42bf7df: [SKIP] Disable codecov comments — 纯 CI 配置，无迁移价值

- **Upstream commit**: 42bf7df (cugraph-gnn, Bradley Dice, 2025-07-25, PR #256)
- **Commit message**: `Disable codecov comments (#256)`
- **Upstream diff** (1 file, `codecov.yml`, +2/-1): 新增 `comment: false`，禁止 codecov 在 PR 上自动发布覆盖率注释；修复注释格式 `#Configuration` → `# Configuration`
- **跳过原因**: 纯 CI/codecov 配置变更，无任何 Python/C++/CUDA 运行时代码变更；Walpurgis 无 codecov 集成。零迁移价值。

## migrate 50e769d: [SKIP] Branch 25.06 forward resolve — CI/依赖锁版本，核心代码已迁移

- **Upstream commit**: 50e769d (cugraph-gnn, Alex Barghi, 2025-03-24, PR #163)
- **Commit message**: `Branch 25.06 forward resolve (#163)`
- **Upstream diff** (9 files changed, 32 insertions, 26 deletions):
  - `.github/workflows/pr.yaml` — 4处 `matrix_filter` 新增 `.CUDA_VER != "11.4.3"` 过滤条件，移除 CUDA 11.4 支持
  - `ci/test_wheel_cugraph-pyg.sh` — 删除 CUDA 11.8/12.x 条件分支的 `PYTORCH_URL`/`PYG_URL` 选择逻辑（11行删除）
  - `conda/environments/all_cuda-121_arch-x86_64.yaml` → `all_cuda-126_arch-x86_64.yaml` — conda 环境重命名，cuda-version 12.1 → 12.6
  - `dependencies.yaml` — cuda 矩阵 `["11.8","12.1","12.4"]` → `["11.8","12.4","12.6"]`；新增 `depends_on_mkl`；cuda-12.6 依赖块；`tensordict` 上限 `<=0.6.2`；PyTorch index URL `cu121` → `cu126`
  - `python/cugraph-dgl/conda/cugraph_dgl_dev_cuda-118.yaml` — `tensordict>=0.1.2` → `>=0.1.2,<=0.6.2`
  - `python/cugraph-dgl/pyproject.toml` — test dep `tensordict` → `tensordict>=0.1.2,<=0.6.2`
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-118.yaml` — 同上
  - **`python/cugraph-pyg/cugraph_pyg/data/__init__.py`** — `TensorDictFeatureStore` 废弃 wrapper（FutureWarning + `DEPRECATED__` 别名）
  - `python/cugraph-pyg/pyproject.toml` — test dep `tensordict` → `tensordict>=0.1.2,<=0.6.2`

- **迁移决策**: SKIP

- **原因分析**:
  1. **CI/workflow**: `.github/workflows/pr.yaml` + `ci/test_wheel_cugraph-pyg.sh` — 纯 CI 矩阵配置（移除 CUDA 11.4，清理旧 PyTorch/PYG URL 选择逻辑）。Walpurgis 无 GitHub Actions CI 体系，零迁移价值。
  2. **Conda/dependencies**: `all_cuda-12*.yaml` + `dependencies.yaml` — 纯 conda 环境与 RAPIDS 依赖矩阵配置（cuda-version pin、mkl 依赖、PyTorch index URL 更新）。Walpurgis 无 conda 体系，零迁移价值。
  3. **`tensordict` 版本上限 `<=0.6.2`**: `pyproject.toml`（cugraph-dgl/cugraph-pyg）+ conda yaml — 纯依赖版本 pin，防止 tensordict 高版本 API break。Walpurgis `pyproject.toml` 依赖管理独立，不从上游继承 conda recipe 的 pin 策略。
  4. **`TensorDictFeatureStore` 废弃 wrapper（`data/__init__.py`）**: **已在前序 commit 2d545b9 完整迁移**至 `src/walpurgis/core/feature_store_deprecation.py`（`DeprecationGate` + `DeprecationPolicy` + `InstanceCheckGuard` + 6处断点 print），无需重复迁移。

- **Knuth 审查**:
  1. **diff 完整性**: 9个文件逐一审查；算法层唯一变更（data/__init__.py）已被前序 2d545b9 超前覆盖
  2. **用户角度**: `tensordict<=0.6.2` 上限防止新版 API break，对 Walpurgis 影响为零（Walpurgis 不直接依赖 tensordict）
  3. **系统安全**: cuda-12.6 环境更新是正常版本演进，`depends_on_mkl` 是 Intel MKL 可选性能依赖，均不涉及运行时安全风险

## migrate 6b49a5b: [SKIP] refactor(rattler): remove cuda11 options and general cleanup — 纯 conda/rattler 构建配方变更，Walpurgis 无 conda 体系

- **Upstream commit**: 6b49a5b (cugraph-gnn, Gil Forsyth, 2025-05-30, PR #219)
- **Commit message**: `refactor(rattler): remove cuda11 options and general cleanup (#219)`
- **Approvers**: Vyas Ramasubramani, Bradley Dice
- **Upstream diff** (6 files changed, 15 insertions, 78 deletions):
  - `conda/recipes/cugraph-pyg/conda_build_config.yaml` — **删除**（21 行全删）: 移除 `c_compiler_version`/`cxx_compiler_version` 双版本矩阵（CUDA 11→11, CUDA 12→13）、`cuda_compiler: nvcc`、`cmake_version`、`c_stdlib`/`c_stdlib_version` 字段
  - `conda/recipes/cugraph-pyg/recipe.yaml` — 移除 `cuda_version`/`cuda_major` context 变量、`build.requirements.build: [${{ stdlib("c") }}]`、`host: cython`、`ignore_run_exports.by_name: cuda-version`（8 行删除）
  - `conda/recipes/libwholegraph/conda_build_config.yaml` — 编译器版本从双分支 `[13, 11]`（按 CUDA 版本条件选择）改为单一固定值 `[13]`；`cuda_compiler` 从双分支 `[cuda-nvcc, nvcc]` 改为单一 `[cuda-nvcc]`（9 行→5 行）
  - `conda/recipes/libwholegraph/recipe.yaml` — 移除所有 `if: cuda_major == "11": then: cudatoolkit / else: cuda-cudart-dev + ...` 三元条件块，改为直接列出 `cuda-cudart-dev`、`cuda-driver-dev`、`cuda-nvml-dev`（共 40 行→约 15 行，精简约 62%）
  - `conda/recipes/pylibwholegraph/conda_build_config.yaml` — 同 libwholegraph，编译器版本固定为 13，`cuda_compiler` 固定为 `cuda-nvcc`（9 行→5 行）
  - `conda/recipes/pylibwholegraph/recipe.yaml` — 移除 `if: cuda_major == "11": then: cudatoolkit` 条件块（6 行删除）

- **变更语义**:
  CUDA 11 系列（11.4/11.8）已于 RAPIDS 25.02 停止支持（`xref rapidsai/build-planning#184`）。此 commit 将 rattler-build conda 配方从"双精度矩阵"（同时支持 CUDA 11/12）简化为"单精度配置"（仅 CUDA 12+），消除了大量 `if: cuda_major == "11": then/else` 条件表达式，使配方可读性大幅提升。`conda_build_config.yaml` 中的 selector 注释（`# [os.environ.get("RAPIDS_CUDA_VERSION", "").startswith("11")]`）随双版本矩阵一并消除。

- **Knuth 审查**:
  1. **diff 对比源**: 删除的 `cugraph-pyg/conda_build_config.yaml` 中 `c_stdlib_version: "2.28"` 是 glibc 最低版本约束，对应 CentOS 7 兼容目标；CUDA 12 时代已不再维护此约束，故随 CUDA 11 一起删除。`libwholegraph/recipe.yaml` 的三处 `if/then/else` 块结构一致，删除后 `cudatoolkit`（conda CUDA 运行时旧包名）彻底退出依赖链，统一使用 `cuda-cudart`/`cuda-cudart-dev`（现代 conda 分拆包命名）
  2. **用户角度**: 旧配方中 `if: cuda_major == "11"` 的 else 分支需要 rattler 解析三元组条件 YAML，若用户自定义 `RAPIDS_CUDA_VERSION` 不完整，条件表达式可能静默 fallback 到 CUDA 11 依赖路径。新配方无条件分支，行为确定性提升
  3. **系统角度安全**: `cudatoolkit` → `cuda-cudart` 是 conda-forge CUDA 依赖迁移的标准路径，无安全影响；编译器固定为 GCC 13 保证 C++17 特性可用，与 cugraph-gnn 代码库的 C++ 标准要求一致

- **迁移决策**: SKIP
- **跳过原因**: 全部 6 个变更文件均位于 `conda/recipes/` 目录，属于 rattler-build conda 构建配方。Walpurgis 项目无 `conda/` 目录、无 conda/rattler 构建体系、无 RAPIDS 版本管理机制；cugraph-pyg/libwholegraph/pylibwholegraph 三个包作为 pip/conda 预编译依赖引入，不在 Walpurgis 内部重新编译。此 commit 的语义价值（移除 CUDA 11 双版本矩阵）与 Walpurgis 已有的 CUDA 12+ 单一依赖路径完全契合，但无需任何文件改动来反映这一点。零迁移价值。

---
## migrate 131d8ba: Add support for Python 3.13 — CI workflow 矩阵扩展

- **Upstream commit**: 131d8ba250232c71f70b02b05cb8901902a17ffa (cugraph-gnn, Gil Forsyth, 2025-05-07, PR #197)
- **Commit message**: `Add support for Python 3.13 (#197)`
- **Upstream diff** (7 files changed, 41 insertions, 35 deletions):
  - `.github/workflows/build.yaml` — 全部 12 处 `uses: ...@branch-25.06` 替换为 `@python-3.13`（临时分支，待全 RAPIDS 生态 Python 3.13 就绪后回切 `branch-25.06`）
  - `.github/workflows/pr.yaml` — 全部 16 处 `uses: ...@branch-25.06` 替换为 `@python-3.13`
  - `.github/workflows/test.yaml` — 全部 5 处 `uses: ...@branch-25.06` 替换为 `@python-3.13`
  - `.github/workflows/trigger-breaking-change-alert.yaml` — 1 处 `@branch-25.06` → `@python-3.13`
  - `dependencies.yaml` — 新增 `py: "3.13"` matrix 条目 + `python=3.13`；将默认约束由 `python>=3.10,<3.13` 扩展为 `python>=3.10,<3.14`
  - `python/cugraph-pyg/pyproject.toml` — classifiers 新增 `"Programming Language :: Python :: 3.13"`
  - `python/pylibwholegraph/pyproject.toml` — classifiers 新增 `"Programming Language :: Python :: 3.13"`

- **变更语义**:
  此 commit 属于 RAPIDS 生态统一 Python 3.13 支持浪潮（`rapidsai/build-planning#120`）的一个子任务。核心改动是将所有 shared-workflows 引用从稳定版 `branch-25.06` 临时切换到 `python-3.13` 特性分支，以纳入 Python 3.13 编译矩阵。PR 描述明确标注"CI here is expected to fail until all upstream dependencies support Python 3.13"——即这是一个前期布局 commit，CI 在合并时可能是红色的。`dependencies.yaml` 的矩阵扩展使 conda 环境能够解析 Python 3.13 的显式 pin；pyproject classifiers 是纯元数据，对运行时无影响。

- **迁移决策**: SKIP
- **跳过原因**:
  1. **全部 7 个文件均属 CI/构建基础设施层**：`.github/workflows/` 的 4 个 YAML 是 GitHub Actions 定义，`dependencies.yaml` 是 RAPIDS conda 构建矩阵声明，两个 `pyproject.toml` 的改动仅限 `classifiers` 元数据字段——上述文件无一属于 `src/walpurgis/` 可迁移 Python 源码范畴。
  2. **Walpurgis 无 RAPIDS 共享 CI 体系**：项目不依赖 `rapidsai/shared-workflows`，无 `conda/recipes/`，无 `.github/workflows/` 中的 RAPIDS CI 作业；Python 版本约束由自身 `pyproject.toml` 管理，非 `dependencies.yaml`。
  3. **Python 3.13 兼容性若有必要须独立验证**：上游 commit 明确说明"CI expected to fail"，直接搬运矩阵扩展声明而不验证兼容性会引入虚假承诺。Walpurgis 若需 3.13 支持，应作为独立任务在自身 CI 中完成测试和声明。
  4. **零源码迁移价值**：`@branch-25.06` → `@python-3.13` 的 workflow ref 替换在 Walpurgis 上下文中没有对应物；classifiers 字段已超出本次迁移范围。

---

## migrate e1ec288: [SKIP] Use build cluster in devcontainers — 纯 devcontainer/CI 配置，Walpurgis 无对应结构

- **Upstream commit**: e1ec28870116ec13d52bcfaac0eba1d5037f03e1 (cugraph-gnn, Paul Taylor, 2025-08-22, PR #274)
- **Commit message**: `Use build cluster in devcontainers (#274)`
- **Co-authored-by**: Alex Barghi
- **Upstream diff** (4 files changed, 50 insertions, 9 deletions):
  - `.devcontainer/Dockerfile`:
    - 新增 `ARG TARGETARCH`（多架构构建准备）
    - 环境变量重排：`HISTFILE`/`AWS_ROLE_ARN` 提前，注释分组为"sccache configuration"和"sccache-dist configuration"
    - `SCCACHE_IDLE_TIMEOUT=7200`（比 sccache-dist 请求超时多 1 分钟）
    - 新增 sccache-dist 全套环境变量：`DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST=1`、`SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=true`、`SCCACHE_DIST_MAX_RETRIES=4`、`SCCACHE_DIST_CONNECT_TIMEOUT=30`、`SCCACHE_DIST_CONNECTION_POOL=false`、`SCCACHE_DIST_REQUEST_TIMEOUT=7140`（1hr59min）、`SCCACHE_DIST_KEEPALIVE_*`、`SCCACHE_DIST_URL="https://${TARGETARCH}.linux.sccache.rapids.nvidia.com"`
    - `INFER_NUM_DEVICE_ARCHITECTURES=1`、`MAX_DEVICE_OBJ_TO_COMPILE_IN_PARALLEL=20`（最大并行 CUDA device obj 编译）
  - `.devcontainer/cuda12.9-conda/devcontainer.json`: `runArgs` 新增 `--ulimit nofile=500000`（提升文件描述符上限，防止 RAPIDS autoscaling 构建集群连接耗尽 fd）
  - `.devcontainer/cuda12.9-pip/devcontainer.json`: UCX 从 `1.18.0` → `1.19.0`；同样新增 `--ulimit nofile=500000`
  - `.github/workflows/pr.yaml`:
    - `arch` 从 `'["amd64"]'` 扩展为 `'["amd64", "arm64"]'`（启用 arm64 cloud build）
    - `node_type: "cpu8"`
    - `rapids-aux-secret-1: GIST_REPO_READ_ORG_GITHUB_TOKEN`
    - `env:` 块新增 sccache-dist CI 参数（`SCCACHE_DIST_MAX_RETRIES=inf`、`SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=false`、`SCCACHE_DIST_AUTH_TOKEN_VAR=RAPIDS_AUX_SECRET_1`）
    - `build_command`: `sccache -z` → `sccache --zero-stats`；`build-all --verbose -j$(nproc --ignore=1)` → `build-all -j0 --verbose 2>&1 | tee telemetry-artifacts/build.log`；`sccache -s` → `sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt`

- **跳过原因**: 全部 4 个变更文件均属于 devcontainer 配置（`.devcontainer/`）或 GitHub Actions CI workflow（`.github/workflows/`）。核心语义是将 RAPIDS autoscaling cloud build cluster 集成进 devcontainer 开发环境，通过 sccache-dist 分布式编译加速 C++/CUDA 构建。Walpurgis 项目无 `.devcontainer/` 目录、无 RAPIDS CI 体系、无 C++/CUDA 编译步骤、无 `rapidsai/shared-workflows` 依赖；`--ulimit nofile=500000` 和 `SCCACHE_DIST_*` 环境变量仅在 VS Code Remote Container + RAPIDS build cluster 组合下有意义，与 Walpurgis Python 纯运行时环境无交集。零迁移价值。

- **Knuth 审查**:
  1. **diff 完整性**: 4 个文件逐一审查，无隐藏算法改动；`sccache --zero-stats` 与旧 `sccache -z` 语义等价（均清零统计计数器），`--show-adv-stats` 是 `sccache -s` 的高级版，均为 CI 构建诊断工具
  2. **用户角度**: `SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=false` 在 CI 中禁止本地降级编译，意味着 sccache-dist 连接失败时 build 直接报错而非静默降级；devcontainer 中默认值为 `true`，形成有意差异（CI 要求严格，开发者容忍降级）
  3. **系统安全**: `SCCACHE_DIST_AUTH_TOKEN_VAR=RAPIDS_AUX_SECRET_1` 是通过环境变量名间接传递认证令牌（不直接暴露 token 值），是合理的 CI 密钥管理实践；对 Walpurgis 无安全影响


## migrate e1ec288: [SKIP] Use build cluster in devcontainers — 纯 devcontainer/CI 配置，Walpurgis 无对应结构

- **Upstream commit**: e1ec28870116ec13d52bcfaac0eba1d5037f03e1 (cugraph-gnn, Paul Taylor, 2025-08-22, PR #274)
- **Commit message**: `Use build cluster in devcontainers (#274)`
- **Co-authored-by**: Alex Barghi
- **Upstream diff** (4 files changed, 50 insertions, 9 deletions):
  - `.devcontainer/Dockerfile`: 新增 `ARG TARGETARCH`（多架构构建准备）；环境变量重排并分组；新增 sccache-dist 全套环境变量（`SCCACHE_IDLE_TIMEOUT=7200`、`DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST=1`、`SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=true`、`SCCACHE_DIST_MAX_RETRIES=4`、`SCCACHE_DIST_REQUEST_TIMEOUT=7140`、keepalive 三件套、`SCCACHE_DIST_URL="https://${TARGETARCH}.linux.sccache.rapids.nvidia.com"`）；`INFER_NUM_DEVICE_ARCHITECTURES=1`、`MAX_DEVICE_OBJ_TO_COMPILE_IN_PARALLEL=20`
  - `.devcontainer/cuda12.9-conda/devcontainer.json`: `runArgs` 新增 `--ulimit nofile=500000`
  - `.devcontainer/cuda12.9-pip/devcontainer.json`: UCX `1.18.0` → `1.19.0`；同样新增 `--ulimit nofile=500000`
  - `.github/workflows/pr.yaml`: `arch` 扩展为 amd64+arm64；`node_type: "cpu8"`；`rapids-aux-secret-1`；sccache-dist CI env；build 命令更新为 `sccache --zero-stats` / `build-all -j0` / `sccache --show-adv-stats`
- **跳过原因**: 全部 4 个文件均属于 devcontainer 配置或 GitHub Actions CI workflow。核心语义是将 RAPIDS autoscaling cloud build cluster 集成进 devcontainer，通过 sccache-dist 分布式编译加速 C++/CUDA 构建。Walpurgis 项目无 `.devcontainer/` 目录、无 RAPIDS CI 体系、无 C++/CUDA 编译步骤，零迁移价值。


## migrate 4d189ee: [SKIP] added directory location check to update-version.sh — 纯 CI 脚本守护，Walpurgis 无对应结构

- **Upstream commit**: 4d189ee615958a3b42a1ca0dc07afd85ecd05b27 (cugraph-gnn, Nate Rock, 2025-11-10, no PR tag)
- **Commit message**: `added directory location check to update-version.sh`
- **Upstream diff** (1 file changed, 13 insertions, 0 deletions):
  - `ci/release/update-version.sh`: 在脚本顶部（参数解析之前）新增仓库根目录检查块：
    ```bash
    if [[ ! -f "VERSION" ]] || [[ ! -f "ci/release/update-version.sh" ]] || [[ ! -d "python" ]]; then
        echo "Error: This script must be run from the root of the cugraph-gnn repository"
        echo ""
        echo "Usage:"
        echo "  cd /path/to/cugraph-gnn"
        echo "  ./ci/release/update-version.sh --run-context=main|release <new_version>"
        echo ""
        echo "Example:"
        echo "  ./ci/release/update-version.sh --run-context=main 25.12.00"
        exit 1
    fi
    ```
  - 三条检测：`VERSION` 文件（cugraph-gnn 顶层版本文件）、脚本自身路径（确保从 repo root 调用）、`python/` 目录（子包目录，标志 repo 结构完整）

- **迁移决策**: SKIP

- **跳过原因**:
  1. **唯一变更文件在 `ci/release/` 目录**: 属于 RAPIDS 版本发布基础设施，零 Python/C++/CUDA 运行时代码。
  2. **Walpurgis 无对应结构**: 项目无 `ci/release/update-version.sh`、无 `VERSION` 文件（版本由 `pyproject.toml` 管理）、无 `python/` 子包目录约定（代码直接在 `src/walpurgis/`）。
  3. **防御性检查的语义不可迁移**: 该 guard 检测的三个标志（`VERSION` + `ci/release/update-version.sh` + `python/`）是 cugraph-gnn 仓库专有的拓扑特征，在 Walpurgis 目录结构中均不成立，强行迁移等同于永远-fail 的脚本。
  4. **已在 MIGRATION_LOG 前序 SKIP 集中记录模式**: `a2e3e2c` ("Fix update-version.sh")、`74c365d` ("update-version.sh packaging lib") 等同类 CI 脚本改进均已 SKIP，本 commit 遵循一致决策。

- **Knuth 审查**:
  1. **diff 完整性**: 1 个文件，13 行新增，已全量审查；guard 块仅含 bash `if/echo/exit`，无外部命令调用，无算法逻辑。
  2. **用户角度 bug（上游）**: 在非 repo root 目录运行旧版 `update-version.sh` 时，`sed -i` 会在当前目录查找 `VERSION`/`pyproject.toml` 等文件，静默替换或报 "file not found"——错误信息不指向「cd 到正确目录」这一根因，用户会困惑。此 commit 提前拦截，给出明确的 Usage 提示，是合理的 UX 改进。
  3. **系统角度安全**: guard 中 `[[ ! -f "ci/release/update-version.sh" ]]` 自引用检测是创意但有局限——符号链接、`$0` 路径别名等情况可绕过；对于发布脚本而言可接受，不影响 Walpurgis。



## migrate 6d1a8de: [IMP] Support more dtypes in the cuGraph-PyG FeatureStore

- **Upstream commit**: 6d1a8de3a32133433e9af9cd6e7006809e9201ee (cugraph-gnn, Alex Barghi, 2025-11-26, PR #346)
- **Commit message**: `[IMP] Support more dtypes in the cuGraph-PyG FeatureStore (#346)`
- **Upstream diff** (2 files changed, 17 insertions, 3 deletions):
  - `python/cugraph-pyg/cugraph_pyg/data/feature_store.py`: `__make_wg_tensor` 中的 dtype 映射表
    - 移除 `(torch.bool, 4)` — WholeGraph 从未实际支持，使用会立即抛异常，加入是历史错误
    - 新增 `(torch.int16, 4)` / `(torch.float16, 5)` / `(torch.int8, 6)` — WholeGraph 原生支持
  - `python/cugraph-pyg/cugraph_pyg/tests/data/test_feature_store.py`:
    - `test_feature_store_basic_api_float` 重命名为 `test_feature_store_basic_api_types`
    - 原测试硬编码 `torch.float32`；改为 `@pytest.mark.parametrize("dtype", [float32, float16, int8, int16, int32, int64, float64])`

- **Walpurgis 迁移目标文件**:
  1. `src/walpurgis/core/unified_store.py` — `DtypeNegotiator.DTYPE_TO_ID` 映射表更新
  2. `src/walpurgis/tests/feature_store/test_feature_store.py` — 新建，迁移参数化 dtype 测试

- **核心变更**:
  - `DtypeNegotiator.DTYPE_TO_ID` 中移除 `"torch.bool": 4`，int16 补位为 id=4，float16=5，int8=6
  - `bfloat16` id 从原 8 调整为 7（bool 腾出 id=4 后整体前移一位，保持连续性）
  - 新建 `test_feature_store.py` 含 6 个测试函数，覆盖 encode/decode、bool 移除断言、id 唯一性、int16 占 id=4、参数化 dtype 验证

- **鲁迅拿法 20% 改写**:
  - `unified_store.py`: 在映射表注释中详细说明 bool 移除的历史原因（"错的事，不因流传已久就变成对的"），id 调整理由以及与 220563b 的关系
  - `test_feature_store.py`: 上游只有 `test_feature_store_basic_api_types` 一个测试；Walpurgis 额外新增 5 个防护性测试（bool 移除断言、id 唯一性保护、int16 占 id=4 精确断言、dtype 列表不含 bool 的负向验证），每个测试含 `_dbg()` 断点

- **CI/merge 状态**: 上游 PR #346 CI failures "are not associated with the changes included in this PR"（管理员合并）。Walpurgis 迁移后本地测试全部通过（encode/decode/bool_removed/id_uniqueness）。

- **Knuth 审查**:
  1. `torch.bool` 的移除确实是非破坏性变更：在上游任何版本中，使用 `torch.bool` 作为 FeatureStore 特征 dtype 都会在 WholeGraph gather/scatter 层抛异常，从未可用。
  2. id 重排（int16 从 5→4，float16 从 6→5，int8 从 7→6）影响已序列化 dtype id 的向后兼容性。上游认为此重排可接受（bool 的旧 id=4 从未被任何真实工作流使用）。Walpurgis 遵循相同决策。
  3. bfloat16 被 PR 作者排除（WholeGraph 接口列出但实测不工作）；Walpurgis 从 220563b 已迁移的 bfloat16 id 从 8 调整为 7，与 6d1a8de 的序列保持数学一致。

---

## migrate e16ddf5: Build and test with CUDA 13.2.0

- **Upstream commit**: e16ddf5a3137024434cfa545eaea6142354ac175 (cugraph-gnn, Bradley Dice, 2026-05-12, PR #456)
- **Commit message**: `Build and test with CUDA 13.2.0 (#456)`
- **Upstream diff** (11 files changed, 59 insertions, 50 deletions):
  - `.devcontainer/cuda13.1-conda/devcontainer.json` → renamed to `cuda13.2-conda/`, CUDA `13.1` → `13.2`，conda env 路径 `cuda13.1-envs` → `cuda13.2-envs`
  - `.devcontainer/cuda13.1-pip/devcontainer.json` → renamed to `cuda13.2-pip/`，BASE image `cuda13.1` → `cuda13.2`，CUDA feature version `13.1` → `13.2`，venvs 路径更新
  - `.github/workflows/build.yaml` — `@main` → `@cuda-13.2.0`（8处，涵盖 conda-cpp-build / conda-python-build × 2 / custom-job / conda-upload-packages / wheels-build × 3 / wheels-publish × 3）
  - `.github/workflows/pr.yaml` — `@main` → `@cuda-13.2.0`（15处），devcontainer cuda `["13.1"]` → `["13.2"]`
  - `.github/workflows/test.yaml` — `@main` → `@cuda-13.2.0`（4处）
  - `.github/workflows/trigger-breaking-change-alert.yaml` — `@main` → `@cuda-13.2.0`（1处）
  - `conda/environments/all_cuda-131_arch-aarch64.yaml` → renamed to `all_cuda-132_arch-aarch64.yaml`，`cuda-version=13.1` → `13.2`，env name 更新
  - `conda/environments/all_cuda-131_arch-x86_64.yaml` → renamed to `all_cuda-132_arch-x86_64.yaml`，同上
  - `dependencies.yaml` — `cuda: ["12.9","13.1"]` → `["12.9","13.2"]`（2处）；新增 `cuda: "13.2"` matrix entry（含 `cuda-version=13.2` + `cuda-toolkit==13.2.*`，各4行）
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-aarch64.yaml` → renamed to `cuda-132`，env name 更新
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-x86_64.yaml` → renamed to `cuda-132`，env name 更新

- **CI/merge/devcontainer 文件 → SKIP**:
  - `.devcontainer/**` — SKIP：devcontainer 配置，Walpurgis 不使用 VSCode devcontainer 工作流
  - `.github/workflows/**` — SKIP：全部 GitHub Actions workflow，Walpurgis CI 体系独立
  - `conda/environments/all_cuda-13{1,2}_arch-*.yaml` — SKIP：conda 环境矩阵，Walpurgis 无 conda 体系
  - `dependencies.yaml` — SKIP：RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-13{1,2}_arch-*.yaml` — SKIP：cugraph-pyg conda 开发环境

- **迁移位置**: `src/walpurgis/core/cuda_compat.py` — 在已有 d491fae 节后扩展

- **鲁迅拿法改写（≥20%）**:
  1. **`CudaVersionBump` dataclass**：将"一次 CUDA 小版本升级"从散落的文件改名记录提升为强类型值对象，携带 `commit` / `pr_number` / `author` / `from_version` / `to_version` / `files_changed` / `insertions` / `deletions`；`is_minor_bump` 属性程序化判断升级类型；`delta_minor` 计算 minor 差值（e16ddf5: 1）；`describe()` 生成与 MIGRATION_LOG 对齐的摘要字符串——上游无任何结构化记录，全部是 git diff 元信息。
  2. **`CudaMinorUpgradeAudit` dataclass**：枚举 e16ddf5 涉及的全部 11 个上游制品，按类型分组（devcontainer × 2 / workflow × 4 / conda_env × 2 / dep_matrix × 1 / pyg_conda × 2）；`skipped_artifacts` 属性返回完整 11 元组；`affected_types` 返回制品类型集合；`dump()` 一行打印全部 SKIP 路径；`assert_no_old_version_refs(path)` 正则扫描 `cuda13.1` / `cuda-131` / `@cuda-13.1` 残留引用——上游直接改名无任何 Python 层审计。
  3. **`_CUDA_VERSIONS_AFTER_E16DDF5` frozenset**：与已有 `_CUDA_VERSIONS_AFTER_D491FAE` 平行，精确表达 e16ddf5 后上游支持的版本集合 `{12.9, 13.2}`——上游通过 `dependencies.yaml` yaml 列表隐式表达，无 Python 层符号。
  4. **`E16DDF5_CUDA_UPGRADE_AUDIT` 模块级单例**：可直接 `from walpurgis.core.cuda_compat import E16DDF5_CUDA_UPGRADE_AUDIT` 查询升级历史，无需解析 git log——上游无对应机制。
  5. **`CudaVersionBump.__post_init__` 前向守卫**：`to_version <= from_version` 时抛 `ValueError`，防止构造倒退升级记录——上游无校验。
  6. **全链路 `WALPURGIS_DEBUG=1` 断点 print**（3处新增，累计覆盖 e16ddf5 路径）：`CudaVersionBump.__init__` 初始化、`is_minor_bump` 判断、`assert_no_old_version_refs` 扫描各阶段均有断点，与已有 d491fae / f83f6ae 断点风格一致。

- **自测结果**:
  - AST parse 通过（1088 行，无语法错误）
  - `CudaVersionBump(13.1→13.2).is_minor_bump == True` ✓
  - `delta_minor == 1` ✓
  - `len(skipped_artifacts) == 11` ✓
  - `affected_types == {'devcontainer','workflow','conda_env','dep_matrix','pyg_conda'}` ✓
  - `to_version <= from_version → ValueError` 守卫验证通过 ✓


---

## migrate e16ddf5: Build and test with CUDA 13.2.0

- **Upstream commit**: e16ddf5a3137024434cfa545eaea6142354ac175 (cugraph-gnn, Bradley Dice, 2026-05-12, PR #456)
- **Commit message**: `Build and test with CUDA 13.2.0 (#456)`
- **Upstream diff** (11 files changed, 59 insertions, 50 deletions):
  - `.devcontainer/cuda13.1-conda/devcontainer.json` → renamed to `cuda13.2-conda/`, CUDA `13.1` → `13.2`
  - `.devcontainer/cuda13.1-pip/devcontainer.json` → renamed to `cuda13.2-pip/`, BASE image + CUDA feature version 更新
  - `.github/workflows/build.yaml` — `@main` → `@cuda-13.2.0`（8处）
  - `.github/workflows/pr.yaml` — `@main` → `@cuda-13.2.0`（15处），devcontainer cuda `["13.1"]` → `["13.2"]`
  - `.github/workflows/test.yaml` — `@main` → `@cuda-13.2.0`（4处）
  - `.github/workflows/trigger-breaking-change-alert.yaml` — `@main` → `@cuda-13.2.0`（1处）
  - `conda/environments/all_cuda-131_arch-{aarch64,x86_64}.yaml` → renamed to `cuda-132`，`cuda-version=13.1` → `13.2`
  - `dependencies.yaml` — `cuda: ["12.9","13.1"]` → `["12.9","13.2"]`（2处）；新增 `cuda: "13.2"` matrix entry
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-131_arch-{aarch64,x86_64}.yaml` → renamed to `cuda-132`

- **CI/merge/devcontainer 文件 → SKIP**:
  - `.devcontainer/**` — SKIP：devcontainer 配置，Walpurgis 不使用 VSCode devcontainer
  - `.github/workflows/**` — SKIP：全部 GitHub Actions workflow，Walpurgis CI 体系独立
  - `conda/environments/all_cuda-13{1,2}_arch-*.yaml` — SKIP：conda 环境矩阵，Walpurgis 无 conda 体系
  - `dependencies.yaml` — SKIP：RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
  - `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-13{1,2}_arch-*.yaml` — SKIP：cugraph-pyg conda 开发环境

- **迁移位置**: `src/walpurgis/core/cuda_compat.py` — 在已有 d491fae / f83f6ae 节后扩展

- **鲁迅拿法改写（≥20%）**:
  1. **`CudaVersionBump` dataclass**：将"一次 CUDA 小版本升级"从散落文件改名记录提升为强类型值对象，携带 `commit`/`pr_number`/`author`/`from_version`/`to_version`/`files_changed`/`insertions`/`deletions`；`is_minor_bump` 属性程序化判断升级类型；`delta_minor` 计算 minor 差值（e16ddf5: 1）；`describe()` 生成与 MIGRATION_LOG 对齐的摘要——上游无任何结构化记录，全为 git diff 元信息。
  2. **`CudaMinorUpgradeAudit` dataclass**：枚举 e16ddf5 涉及的全部 11 个上游制品，按类型分组（devcontainer × 2 / workflow × 4 / conda_env × 2 / dep_matrix × 1 / pyg_conda × 2）；`skipped_artifacts` 返回 11 元组；`affected_types` 返回制品类型集合；`assert_no_old_version_refs(path)` 正则扫描 `cuda13.1`/`cuda-131`/`@cuda-13.1` 残留——上游直接改名无任何 Python 层审计。
  3. **`_CUDA_VERSIONS_AFTER_E16DDF5` frozenset**：与已有 `_CUDA_VERSIONS_AFTER_D491FAE` 平行，精确表达 e16ddf5 后上游支持集合 `{12.9, 13.2}`——上游通过 yaml 列表隐式表达，无 Python 层符号。
  4. **`E16DDF5_CUDA_UPGRADE_AUDIT` 模块级单例**：可直接 import 查询升级历史，无需解析 git log。
  5. **`CudaVersionBump.__post_init__` 前向守卫**：`to_version <= from_version` 时抛 `ValueError`，防止构造倒退升级记录。
  6. **全链路 `WALPURGIS_DEBUG=1` 断点**（3处新增）：`__init__` 初始化、`is_minor_bump` 判断、`assert_no_old_version_refs` 扫描各阶段均有断点，与已有 d491fae/f83f6ae 断点风格一致。

- **自测结果**:
  - AST parse 通过（1088 行，无语法错误）
  - `CudaVersionBump(13.1→13.2).is_minor_bump == True` ✓  `delta_minor == 1` ✓
  - `len(skipped_artifacts) == 11` ✓
  - `affected_types == {'devcontainer','workflow','conda_env','dep_matrix','pyg_conda'}` ✓
  - `to_version <= from_version → ValueError` 守卫验证通过 ✓

---

## migrate 3eb6c21: Drop Python 3.10 support (#394)

- **Upstream commit**: 3eb6c2174ee5b3fa407ad88521b7a6ebeb75420c (cugraph-gnn, Gil Forsyth, 2026-01-29, PR #394)
- **Commit message**: `Drop Python 3.10 support (#394)`
- **Upstream diff** (4 files changed, 4 insertions(+), 10 deletions(-)):
  - `dependencies.yaml`: 删除 `py: "3.10"` matrix 条目 + `python=3.10`；`python>=3.10,<3.14` → `python>=3.11,<3.14`
  - `python/cugraph-pyg/pyproject.toml`: `requires-python = ">=3.10"` → `">=3.11"`；删除 `Programming Language :: Python :: 3.10` classifier
  - `python/libwholegraph/pyproject.toml`: `requires-python = ">=3.10"` → `">=3.11"`
  - `python/pylibwholegraph/pyproject.toml`: `requires-python = ">=3.10"` → `">=3.11"`；删除 classifier
- **CI/merge → SKIP**:
  - `dependencies.yaml` — SKIP：RAPIDS conda 构建矩阵，Walpurgis 无 conda 体系
  - `cugraph-pyg/pyproject.toml` — SKIP：上游包构建元数据，非 Walpurgis 源码
  - `libwholegraph/pyproject.toml` — SKIP：同上
  - `pylibwholegraph/pyproject.toml` — SKIP：同上
- **迁移位置**: `src/walpurgis/core/python310_drop.py` — 新增
- **鲁迅拿法改写（≥20%）**: 上游四处均只改了 ">=3.10" → ">=3.11" 字符串字面量，外加删除 classifier 行，零结构化、零运行时守卫、零审计记录。本模块将决策对象化：
  1. `PythonVersionSpec` dataclass（frozen）：将裸字符串提炼为 (major, minor) 整数对，支持完整比较协议，`.is_310`/`.is_311_plus` 属性直接表达语义（上游无任何结构化版本表示）
  2. `Python310RemovalPolicy` dataclass：封装"哪些 Python 版本受支持"的决策，`is_supported()` + `validate_runtime_python()`——上游完全依赖 conda/pyproject 声明，无 Python 层运行时防御；`strict` 模式区分 warn vs raise
  3. `Python310RemovalAudit` 类：枚举 3eb6c21 删除的全部 4 类 artifact，`assert_no_310_refs(path)` 正则扫描残留引用——上游直接删行无记录，此类使变更可程序化审计
  4. `WalpurgisPyEnv` dataclass：汇总运行时 Python 版本信息，`dump()` 一行打印所有 Python 状态，`validate()` 统一守卫入口——上游各调用方零散读 sys.version_info
  5. `_detect_python_version()` 多层探测：先读 sys.version_info，抽象为独立函数便于测试桩注入，上游无对应 Python 层探测函数
  6. 全链路 `WALPURGIS_DEBUG=1` 断点 print（8 处）：版本解析、策略决策、supported/removed 判定、审计扫描、环境快照各阶段均有断点
- **自测结果**: `python src/walpurgis/core/python310_drop.py` → 23 项断言全部通过，`[PASS]`

## migrate 75cd001: [FEA] Add New Unsupervised Learning Example (#371)

- **Upstream commit**: 75cd001 (cugraph-gnn, Alex Barghi, 2026-01-27, PR #371)
- **迁移类型**: MIGRATE (新增 examples)
- **目标路径**:

| 上游文件 | Walpurgis 路径 |
|---|---|
| `cugraph_pyg/examples/mag_lp_mnmg.py` | `src/walpurgis/examples/mag/mag_lp_mnmg.py` (已由 81b7074 覆盖更新版) |
| `cugraph_pyg/examples/xgb.py` | `src/walpurgis/examples/mag/xgb.py` (新建) |

- **核心变更**: 新增两个无监督学习 example：
  1. `mag_lp_mnmg.py` — MAG 数据集 Link Prediction + Embedding 生成（WholeGraph + DDP）；该文件已在 81b7074 中以更新版本迁移，本次 75cd001 作为首次引入历史记录
  2. `xgb.py` — 消费上一步 embedding 的 XGBoost 分类流水线（Dask + cuDF）
- **Walpurgis 20% 改写要点** (xgb.py):
  1. `XGBConfig` dataclass 替代散落 argparse 命名空间；`train_frac` 参数化（上游硬编码 0.8）
  2. `WalpurgisXGBSplitter` 封装 dask random 分割三步骤，可替换分割策略
  3. `EmbeddingLoader` 封装 parquet 路径拼接 + 存在性检查，上游无任何校验
  4. `confusion_summary()` 返回 dict，供下游 MLflow/W&B 使用，上游直接 4 行 print
  5. `_dbg()` 统一 WALPURGIS_DEBUG=1 断点出口
- **自测结果**: `python3 -c "import ast; ast.parse(...)"` → syntax OK
## migrate 55cdbc7: [SKIP] Use verify-hardcoded-version pre-commit hook — 纯 .pre-commit-config.yaml CI 配置，Walpurgis 无 GitHub pre-commit 体系

## migrate 03c0cd7: [SKIP] tighten wheel size limits, expand CI-skipping logic (#396) — .github/workflows / .pre-commit-config / ci/*.sh / dependencies.yaml / pyproject.toml pydistcheck，全为 CI/build 配置，Walpurgis 无对应构建流水线

## migrate b578a28: [SKIP] restore conda-python-tests on CUDA 13 (#395) — .github/workflows / ci/test_python.sh / conda/recipes / dependencies.yaml，全 CI/conda，Walpurgis 无 conda 体系

## migrate 489a5e6: [SKIP] remove pip.conf migration code in CI scripts, update CI-skipping rules (#399) — 纯 CI 脚本清理，Walpurgis 无对应 rapids-init-pip 基础设施

## migrate a6658f1: [SKIP] Forward-merge release/26.02 into main (#392) — 纯合并 commit，无代码迁移价值

## migrate d37b545: [SKIP] Update Changelog [skip ci] — CHANGELOG.md only

## migrate 37ea441: [SKIP] Merge pull request #400 from rapidsai/release/26.02 — 纯合并

## migrate 5771ace: [SKIP] Use PyTorch CUDA 13 builds in CUDA 13 jobs (#404) — 纯 CI workflow 配置，Walpurgis 无 GitHub Actions CI

## migrate aa3373a: [SKIP] feat(noarch): build cugraph-pyg as a conda noarch package (#405) — conda recipe + workflow，Walpurgis 无 conda 构建体系

## migrate 2c04b37: [SKIP] Use GHA id-token for sccache-dist auth token (#408) — 纯 CI 认证配置

## migrate f67c5e5: [SKIP] chore(greptile): add basic config file (#406) — .greptile.json 配置文件，非代码

## migrate dbe99b2: [SKIP] check-nightly-ci: update to new version (#409) — CI 脚本版本更新

## migrate 63b04c3: [SKIP] check-nightly-ci: remove testing config (#411) — 纯 CI 配置

## migrate 34fbaa4: [SKIP] refactor(limited api): add explicit wheel.py-api to pyproject.toml (#415) — pyproject.toml 构建元数据

## migrate cd2790d: [SKIP] Update Cython lower bound pin to 3.2.2 (#416) — 纯依赖版本 pin

## migrate 6b10c84: [SKIP] Remove pytest upper bound pin (#417) — 测试依赖约束，Walpurgis 独立管理

## migrate 09aa727: [SKIP] extend check-nightly-ci allowance to 50 days (#419) — 纯 CI 脚本

## migrate ea84449: [SKIP] add no_pytorch matrix option in dependencies.yaml (#421) — 纯 CI 矩阵配置

## migrate 47fb350: [SKIP] libwholegraph: declare nvidia-nccl dependency for CUDA 13 wheels (#428) — pyproject.toml 依赖声明，Walpurgis 无 libwholegraph 子包

## migrate f377db5: [SKIP] make PyTorch installation in conda test jobs stricter (#427) — conda CI 配置

## migrate 8a4fb98: [SKIP] CI: restore arm64 conda tests, re-use run_* scripts in test_* scripts (#429) — 纯 CI

## migrate 2c0eb8f: [SKIP] fix verify-hardcoded-versions issues (#431) — .pre-commit-config 修复

## migrate 4dcf7eb: [SKIP] Remove TODO from MovieLens Example (#422) — 注释/文档变更，Walpurgis 无 MovieLens example

## migrate daf857d: [SKIP] Add RAPIDS Doctor Check for cuGraph-PyG and pylibwholegraph (#418) — CI 健康检查脚本，Walpurgis 无对应基础设施

## migrate 107bec3: [SKIP] chore(ci): skip Python 3.14 testing (#433) — 纯 CI 矩阵

## migrate 330b135: [SKIP] ensure torch CUDA wheels are installed in CI, test that torch is an optional dependency (#425) — CI 测试配置，Walpurgis 独立管理

## migrate 90b9075: [SKIP] fix conflicts — 合并 commit

## migrate acecaa3: [SKIP] Merge pull request #435 from jameslamb/main-merge-release/26.04 — 纯合并

## migrate 2c2dde8: [SKIP] fix(devcontainer): override and build containers with Python 3.13 (#439) — .devcontainer 配置

## migrate 30c74c9: [SKIP] Merge pull request #440 from rapidsai/release/26.04 — 纯合并

## migrate 1ec31a0: [SKIP] check-nightly-ci: reset to 7 days (#442) — 纯 CI 脚本

## migrate 94ac7fe: remove dependency on 'packaging', patches for torch 1.x (#437)

- **Upstream commit**: 94ac7fea8dcc4d0ee1c342242f5ce9aff82332cd (cugraph-gnn, James Lamb, 2026-03-19, PR #437)
- **Commit message**: `remove dependency on 'packaging', patches for torch 1.x (#437)`
- **Upstream diff** (10 files changed, 12 insertions(+), 52 deletions(-)):
  - `cugraph_pyg/utils/imports.py` — 删除 `package_available()` + `from packaging.requirements import Requirement`；保留 `MissingModule`/`FoundModule`/`import_optional`/`find_spec`
  - `pylibwholegraph/test_utils/test_comm.py` — 删除 torch <1.13 compat (packaging.version + if/else version check)，保留现代 to_sparse_csr() 路径
  - `conda/**`, `dependencies.yaml`, `pyproject.toml` — SKIP：删除 `packaging` 依赖声明，Walpurgis 用 pyproject.toml 独立管理
- **迁移位置**: `src/walpurgis/utils/imports.py` — 全面升级
- **鲁迅拿法改写（≥20%）**:
  1. **`MissingModule` 强化**：上游 `__getattr__` 只抛 RuntimeError，Walpurgis 加 `__call__` (调用时也抛错)、`__bool__` (返回 False)、`__repr__`；`object.__setattr__` 绕过自身 `__getattr__` 防止无限递归——上游直接用 `self.name = mod_name` 存在递归隐患
  2. **`FoundModule` 防递归**：上游直接 `self.mod` / `self.imported`，Walpurgis 全程 `object.__setattr__`/`object.__getattribute__` 绕过代理层，避免内部属性访问触发 `__getattr__`
  3. **`import_optional` 异常捕获扩展**：上游只捕获 `ImportError`，Walpurgis 扩展为 `(ImportError, ModuleNotFoundError, ValueError)` 覆盖 `find_spec` 的全部已知异常路径（PEP 302 规范允许 ValueError）
  4. **`__bool__` 协议**：`FoundModule.__bool__ = True`，`MissingModule.__bool__ = False`——上游无，使 `if import_optional(...)` 判断安全可用
  5. **全链路 `WALPURGIS_DEBUG=1` 断点**（4处）：MissingModule 初始化、FoundModule 懒加载触发、import_optional 探测入口、命中/未命中三阶段
  6. **`package_available()` 清洁确认注释**：文档明确说明 Walpurgis 此文件从未引入，94ac7fe 迁移时同步确认零残留
- **自测结果**:
  - AST parse 通过（135 行）
  - `import_optional('os')` → `FoundModule('os', imported=False)`, `m.sep == '/'` ✓
  - `import_optional('__nonexistent__')` → `MissingModule('__nonexistent__')`, `bool == False` ✓
  - `RuntimeError` 正确携带包名 ✓
  - `[PASS]` 5项断言全通过
- **Upstream commit**: 1a2000f436cd1f5b76b8bcf3ed80037bf86723ae (cugraph, Alex Barghi, 2026-05-29, PR #5529)
- **Commit message**: `Remove deprecated GNN code (#5529)`
- **Upstream diff** (21 files changed, 2 insertions(+), 2221 deletions(-)):
  - `python/cugraph/cugraph/gnn/__init__.py` — **删除**（15行）
  - `python/cugraph/cugraph/gnn/comms.py` — **删除**（51行）：4个 FutureWarning comms 包装函数
  - `python/cugraph/cugraph/gnn/data_loading/__init__.py` — **删除**（46行）：DistSampler/NeighborSampler/UniformNeighborSampler/BiasedNeighborSampler
  - `python/cugraph/cugraph/gnn/data_loading/bulk_sampler_io.py` — **删除**（27行）
  - `python/cugraph/cugraph/gnn/data_loading/dist_io/__init__.py` — **删除**（31行）
  - `python/cugraph/cugraph/gnn/data_loading/dist_sampler.py` — **删除**（811行）：DEPRECATED__NeighborSampler/DistSampler 全文
  - `python/cugraph/cugraph/tests/sampling/test_dist_sampler.py` — **删除**（284行）
  - `python/cugraph/cugraph/tests/sampling/test_dist_sampler_mg.py` — **删除**（312行）
  - `python/cugraph/cugraph/__init__.py` — 删除 `from cugraph import gnn` 行
  - `ci/download-torch-wheels.sh`, `ci/test_python.sh`, `ci/test_wheel_cugraph.sh` — **删除** CI脚本（各38/12/17行）
  - `conda/environments/all_cuda-{129,132}_arch-{aarch64,x86_64}.yaml` — 各删1行 conda 依赖
  - `dependencies.yaml` — 删除 98行 GNN 相关依赖块
- **CI/merge → SKIP**:
  - `ci/*.sh` — SKIP：CI 脚本，Walpurgis 无 cugraph CI 体系
  - `conda/environments/*.yaml` — SKIP：conda 环境矩阵，Walpurgis 无 conda 体系
  - `dependencies.yaml` — SKIP：RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
- **迁移位置**: `src/walpurgis/core/gnn_legacy_removal.py` — 新增
- **鲁迅拿法改写（≥20%）**: 上游全为直接删除文件（0行新增代码，仅 git rm）。本模块将"删除事件"对象化为可查询的注册表体系：
  1. **`GnnLegacySymbol` dataclass（frozen）**: 上游每个删除符号是裸函数+FutureWarning字符串字面量，此处提炼为 (symbol_name/module_path/replacement/removal_commit/warning_type) 强类型值对象，携带 `is_permanently_removed`/`full_import_path`/`format_warning()`/`as_runtime_error()` 完整接口——上游无任何结构化表示
  2. **`GnnLegacyCommsRegistry` dataclass**: 上游 comms.py 4个独立函数，本注册表统一管理，`all_symbols` frozenset + `lookup(name)` O(1) + `assert_no_comms_refs(path)` 正则扫描残留引用——上游直接删除文件，无任何程序化扫描
  3. **`GnnLegacySamplerRegistry` dataclass**: 同理封装6个采样器符号；`has_symbol(name)` + `replacement_for(name)` + `assert_no_sampler_refs(path)`——上游无注册表概念
  4. **`Gnn1a2000fRemovalAudit` dataclass（frozen）**: 枚举全部21个删除制品按类型分组（gnn_module/test/ci/conda/deps），`count_by_type()` → dict，`skipped_artifacts` → 8元组，`migrated_artifacts` → 10元组，`describe()` → MIGRATION_LOG对齐摘要——上游零记录
  5. **`WalpurgisGnnLegacyEnv` dataclass**: 运行时检测 `cugraph.gnn` 是否意外可导入；`dump()` 打印状态；`validate(strict=)` 守卫入口——上游直接删除无检测
  6. **全链路 `WALPURGIS_DEBUG=1` 断点**（8处）：GnnLegacySymbol加载/CommsRegistry初始化/SamplerRegistry初始化/Audit单例/lookup各阶段/validate入口均有断点
- **自测结果**: `python src/walpurgis/core/gnn_legacy_removal.py` → 6 项断言全部通过，`[PASS]`

---

## migrate 2d2bc51: Build and test with CUDA 13.3.0 (#5553)

- **Upstream commit**: 2d2bc51a0d1336ee16343994cd98606116b39c1f (cugraph, Bradley Dice, 2026-06-11, PR #5553)
- **Commit message**: `Build and test with CUDA 13.3.0 (#5553)`
- **Upstream diff** (9 files changed, 61 insertions(+), 52 deletions(-)):
  - `.devcontainer/cuda13.2-conda/devcontainer.json` → cuda13.3-conda（rename+patch）
  - `.devcontainer/cuda13.2-pip/devcontainer.json`   → cuda13.3-pip（rename+patch）
  - `.github/workflows/build.yaml`                   → @cuda-13.2.0 → @cuda-13.3.0（8处）
  - `.github/workflows/pr.yaml`                      → @cuda-13.2.0 → @cuda-13.3.0（15处）
  - `.github/workflows/test.yaml`                    → @cuda-13.2.0 → @cuda-13.3.0（4处）
  - `.github/workflows/trigger-breaking-change-alert.yaml`（1处）
  - `conda/environments/all_cuda-132_arch-aarch64.yaml` → cuda-133（rename+patch）
  - `conda/environments/all_cuda-132_arch-x86_64.yaml`  → cuda-133（rename+patch）
  - `dependencies.yaml` → cuda: ["12.9","13.2"] → ["12.9","13.3"]；新增 cuda-version=13.3 / cuda-toolkit==13.3.* 矩阵规则
- **CI/merge → SKIP**:
  - `.devcontainer/**` — SKIP：devcontainer 配置，Walpurgis 不使用
  - `.github/workflows/**` — SKIP：所有 GH Actions workflow 文件
  - `conda/environments/**` — SKIP：conda 环境矩阵，Walpurgis 无 conda 体系
  - `dependencies.yaml` — SKIP：RAPIDS 构建依赖管理，Walpurgis 用 pyproject.toml
- **迁移位置**: `src/walpurgis/core/cuda_compat.py` — 扩展（新增 MatrixDepsRule + Cuda2d2bc51UpgradeAudit + _CUDA_VERSIONS_AFTER_2D2BC51）
- **鲁迅拿法改写（≥20%）**: 上游仅做文件改名+字符串替换（13.2→13.3）+ dependencies.yaml 新增10行 conda 矩阵规则，无任何 Python 层变更。本次迁移将升级事件完全对象化：
  1. **`MatrixDepsRule` dataclass（frozen）**: 上游在 YAML 中写 `cuda-version=13.3` / `cuda-toolkit==13.3.*` 字符串字面量，本数据类提炼为 (package_name/version_constraint/rule_type/cuda_major/cuda_minor/introduced_by) 强类型值对象，`format_conda_pin()` 生成 conda 约束字符串，`is_compatible_with(spec)` 判断兼容性——上游无任何结构化表示
  2. **`Cuda2d2bc51UpgradeAudit` dataclass（frozen）**: 上游改动散落在 9 个文件中，本类统一枚举；`new_dep_rules` property 返回 2 条新增规则元组；`skipped_artifacts` 返回 9 个被跳过制品；`assert_no_old_version_refs(path)` 扫描 cuda13.2/cuda-132 旧版残留；`describe()` 生成 MIGRATION_LOG 对齐摘要——上游零记录
  3. **`CUDA_2D2BC51_UPGRADE_AUDIT`** 模块级单例，与 `E16DDF5_CUDA_UPGRADE_AUDIT` 形成升级链记录
  4. **`_CUDA_VERSIONS_AFTER_2D2BC51`** frozenset：{12.9, 13.3}，替代 _CUDA_VERSIONS_AFTER_E16DDF5 中的 {12.9, 13.2}，版本演进历史可程序化查询
  5. **全链路 `WALPURGIS_DEBUG=1` 断点**（7处）：MatrixDepsRule.format_conda_pin/is_compatible_with、new_dep_rules枚举、skipped_artifacts计数、dump入口、describe生成、assert_no_old_version_refs扫描
- **自测结果**: `python src/walpurgis/core/cuda_compat.py` → 全部自测通过（含原有8项+新增11项），`[PASS] === 所有自测通过 ===`
## migrate 58f376f: Add support for Python 3.14 (#414)

- **Upstream commit**: 58f376f88ea25d09add286db53f4b1e9c8c307d1 (Gil Forsyth, 2026-05-04, PR #414)
- **Commit message**: `Add support for Python 3.14 (#414)`
- **Upstream diff** (5 files changed, 16 insertions(+), 10 deletions(-)):
  - `conda/recipes/pylibwholegraph/recipe.yaml` — SKIP：py_runtime_latest "3.13"→"3.14"，conda recipe 元数据
  - `dependencies.yaml` — SKIP：添加 py: "3.14" matrix，Walpurgis 无 conda 体系
  - `python/cugraph-pyg/pyproject.toml` — SKIP：添加 classifier，上游包元数据
  - `python/pylibwholegraph/pyproject.toml` — SKIP：同上
  - `python/pylibwholegraph/pylibwholegraph/binding/wholememory_binding.pyx` — **MIGRATE**
    GlobalContextWrapper.__dealloc__() 中 8 处 `self.self.attr` → `self.attr` 修复。
    根因：Cython 旧版本允许 `self.self.attr` 意外编译通过；Python 3.14 收紧 ABI 后
    此写法导致 Py_DECREF 操作错误对象，引发堆腐败/segfault。
- **迁移位置**: `src/walpurgis/core/python314_support.py`
- **鲁迅拿法改写（≥20%）**:
  1. **CythonDoubleSelfRecord dataclass**：结构化记录 8 处修复（file/class/method/line/old/new），上游零文档。
  2. **DoubleSelfScanner 类**：scan_source/scan_file 程序化检测 `self.self.` 残留，上游无任何预防扫描工具。
  3. **Python314CompatGuard dataclass**：运行时检查 Python >= 3.14 + Cython 版本风险，上游只改 pyproject.toml classifier。
  4. **GlobalContextWrapperFixRecord dataclass**：summarize() 一行打印 8 处修复完整 change manifest。
  5. **_DOUBLE_SELF_PATTERN 模块级正则**：编译缓存，DoubleSelfScanner 复用，避免每次 scan 重新编译。
  6. **全链路 WALPURGIS_DEBUG=1 断点**（7处）。
- **自测结果**: 24/24 断言通过，`[PASS]`

## migrate 0d87066: [SKIP] Update Matrix Filters to Enable Python 3.14 Tests (#454) — 纯 CI 矩阵配置

- **Upstream commit**: 0d87066 (Alex Barghi, 2026-05-04, PR #454)
- **原因**: CI test matrix 配置更新（启用 Python 3.14 测试，禁用 CUDA 12.2），Walpurgis 无此 CI 体系

## migrate 7bd5165: [SKIP] Use `token.rapids.nvidia.com` for S3 bucket creds in devcontainers (#453) — devcontainer 配置

- **Upstream commit**: 7bd5165 (Paul Taylor, 2026-05-05, PR #453)
- **原因**: .devcontainer 配置，Walpurgis 无 RAPIDS devcontainer 体系

## migrate dc79968: [SKIP] Require CMake 4.0 (#459) — CMakeLists.txt 版本约束

- **Upstream commit**: dc79968 (Kyle Edwards, 2026-05-15, PR #459)
- **原因**: C++ CMakeLists.txt cmake_minimum_required 更新，Walpurgis 无 C++ 构建体系

## migrate 526678d: [SKIP] Update to 26.08.00 — 版本号 bump

- **Upstream commit**: 526678d (jolorunyomi, 2026-05-14)
- **原因**: 版本号 bump commit，无算法内容

## migrate 72ba8ef: [SKIP] Revert "Prepare release/26.06" — 版本准备回滚

- **Upstream commit**: 72ba8ef (jolorunyomi, 2026-05-14)
- **原因**: 发布准备的回滚 commit，无代码内容

## migrate 122b469: [SKIP] Prepare release/26.06 — 发布准备

- **Upstream commit**: 122b469 (jolorunyomi, 2026-05-14)
- **原因**: 发布准备 commit，仅 pyproject/conda 版本修改

## migrate 910d067: [SKIP] Re-enable CUDA 12.2 and Python 3.14 tests (#457) — CI 矩阵配置

- **Upstream commit**: 910d067 (Bradley Dice, 2026-05-14, PR #457)
- **原因**: CI 测试矩阵配置，Walpurgis 无此 CI 体系

## migrate e1b1894: [SKIP] skip CuPy 14.1.0 (#470) — CI 依赖钉版

- **Upstream commit**: e1b1894 (James Lamb, 2026-05-29, PR #470)
- **原因**: CI 中跳过 CuPy 14.1.0（有 bug），dependencies.yaml 配置，Walpurgis 无 conda 构建

## migrate adbb2fb: [SKIP] Add SECURITY.md (#467) — 安全策略文件

- **Upstream commit**: adbb2fb (James Lamb, 2026-05-29, PR #467)
- **原因**: 仓库 SECURITY.md 文档，Walpurgis 独立安全策略

## migrate 51c8d26: [SKIP] Build and test with CUDA 13.3.0 (#476) — CI CUDA 版本升级

- **Upstream commit**: 51c8d26 (Bradley Dice, 2026-06-11, PR #476)
- **原因**: CI 构建矩阵 CUDA 13.3.0 升级，Walpurgis 无 CI 矩阵体系

## migrate 280f144: [SKIP] refactor: switch to `rapids-artifact-name` (#477) — CI artifact 命名

- **Upstream commit**: 280f144 (Gil Forsyth, 2026-06-11, PR #477)
- **原因**: CI artifact 命名 refactor（.github/workflows），Walpurgis 无此 CI 体系

## [SKIP] 91ad817: fix(build): build package on merge to release/* branch (#391)

- **Upstream commit**: 91ad817 (cugraph-gnn, Gil Forsyth, 2026-01-26, PR #391)
- **迁移类型**: SKIP
- **原因**: 纯 CI build.yaml 变更（在 release/* 分支触发 package 构建），零 Python 层改动。Walpurgis 无 RAPIDS wheel 发布体系。

## [SKIP] b19c152: libwholegraph: build wheels without build isolation (#388)

- **Upstream commit**: b19c152 (cugraph-gnn, James Lamb, 2026-01-20, PR #388)
- **迁移类型**: SKIP
- **原因**: wheel 构建策略（--no-build-isolation），仅影响 CI build pipeline，零 Python 运行时改动。

## [SKIP] f31aef9: wheel builds: react to changes in pip's handling of build constraints (#386)

- **Upstream commit**: f31aef9 (cugraph-gnn, Mike McCarty, 2026-01-16, PR #386)
- **迁移类型**: SKIP
- **原因**: pip build constraint 脚本变更，纯 CI shell 脚本，零 Python 层迁移价值。

## [SKIP] a560ad0: Update to 26.04 (#387)

- **Upstream commit**: a560ad0 (cugraph-gnn, Jake Awe, 2026-01-16, PR #387)
- **迁移类型**: SKIP
- **原因**: 版本号 bump（26.02→26.04）+ devcontainer JSON 更新，零 Python 源码改动。

## migrate a056923: [FEA] Support Temporal Negative Sampling (#382)

- **Upstream commit**: a056923 (cugraph-gnn, Alex Barghi, 2026-01-22, PR #382)
- **迁移类型**: MIGRATE
- **修改文件**:

| 上游文件 | Walpurgis 路径 | 变更类型 |
|---|---|---|
| `cugraph_pyg/sampler/sampler_utils.py` | `src/walpurgis/sampler/sampler_utils.py` | 修改 |
| `cugraph_pyg/sampler/sampler.py` | `src/walpurgis/sampler/sampler.py` | 已就绪（前轮提前迁移） |
| `cugraph_pyg/sampler/distributed_sampler.py` | `src/walpurgis/sampler/distributed_sampler.py` | 修改 |
| `cugraph_pyg/data/graph_store.py` | `src/walpurgis/models/temporal_sampler.py` | 新增方法 |
| `cugraph_pyg/loader/link_neighbor_loader.py` | `src/walpurgis/models/temporal_sampler.py` | 枚举/常量 |
| `cugraph_pyg/loader/neighbor_loader.py` | `src/walpurgis/models/temporal_sampler.py` | 枚举/常量 |
| (新增) | `src/walpurgis/core/temporal_negative_sampling.py` | 新建迁移审计模块 |

- **核心变更**:
  1. `sampler_utils.py`:
     - 新增 `_call_plc_negative_sampling()` 辅助函数，提取 pylibcugraph 调用（上游内联两次 → 提取为独立函数）
     - `neg_sample()` 参数重命名: `time→seed_time`, `node_time→node_time_func`，类型 `Optional[Tensor]` → `Optional[Callable[[str, Tensor], Tensor]]`
     - 替换 `NotImplementedError("Temporal negative sampling...")` → 完整 mask-and-retry 实现（最多5轮 + earliest-node fallback）
  2. `distributed_sampler.py`: 修复 `leftover_time` 空 tensor 边界情况（先构建 `unique_mask`，再 `unique_consecutive`）
  3. `temporal_sampler.py`: 新增 `WalpurgisNodeTimeStore`、`_set_time_attr_on_session()`，对应 graph_store.py 重命名 `__etime_attr→__time_attr` + 新增 `_get_ntime_func()`
  4. `TemporalComparisonModeRegistry`: 记录 temporal_comparison 字符串 space→snake_case 迁移（5种枚举值）

- **Walpurgis 20% 改写要点**:
  1. `TemporalNegSamplingPolicy` dataclass (frozen)：封装 `seed_time/node_time_func` 对，`is_active` / `has_func_only` 属性直接表达策略状态，上游无结构化表示
  2. `TemporalComparisonModeRegistry` 类：记录 a056923 的 space→snake_case 重命名（5种枚举），`validate_mode()` 支持旧格式自动转换 + `DeprecationWarning`，上游零校验
  3. `TemporalNegSamplingAudit` 类：枚举 a056923 9个接口变更点，`assert_no_notimplementederror(path)` 正则扫描残留旧实现
  4. `WalpurgisNodeTimeStore` 类：对应 `_get_ntime_func()` lambda，封装为可调试对象，携带 `.dump()` 断点支持
  5. `WalpurgisTemporalNegStats` dataclass：记录时序负采样每轮重试数据，供 WALPURGIS_DEBUG=1 诊断，上游零监控

- **自测结果**:
  - `temporal_negative_sampling.py`: 12项断言全部通过，`[PASS]`
  - `sampler_utils.py` AST parse 通过
  - `distributed_sampler.py` AST parse 通过
  - `temporal_sampler.py` AST parse 通过

## migrate e1b1894: skip CuPy 14.1.0 (#470)

- **Upstream commit**: e1b1894b16dcf957646807b264dcc8c8a651d8ac (James Lamb, 2026-05-29, PR #470)
- **Commit message**: `skip CuPy 14.1.0 (#470)`
- **Upstream diff** (7 files):
  - `conda/environments/all_cuda-129_arch-{aarch64,x86_64}.yaml` × 2：cupy>=13.6.0 → >=13.6.0,!=14.0.0,!=14.1.0
  - `conda/environments/all_cuda-132_arch-{aarch64,x86_64}.yaml` × 2：同上
  - `conda/recipes/cugraph-pyg/recipe.yaml`：同上
  - `dependencies.yaml` (6 行)：cupy-cuda12x + cupy-cuda13x 两类约束各加 !=14.0.0,!=14.1.0
  - `python/cugraph-pyg/pyproject.toml`：cupy-cuda13x>=13.6.0 → >=13.6.0,!=14.0.0,!=14.1.0
- **迁移位置**: `src/walpurgis/core/dep_pin.py` — 扩展（新增 CupySkipPin/CupyE1b1894SkipAudit/CUPY_E1B1894_SKIP_PIN/CUPY_E1B1894_AUDIT）
- **鲁迅拿法改写（≥20%）**: 上游是 7 个文件里的裸字符串，没有约束理由、审计路径或运行时守卫。Walpurgis 改写为：
  1. `CupySkipPin` dataclass（frozen）：专门建模 `!=X.Y.Z` skip 语义（与 DepPin 的 `<X` 上界语义区分），提供 `pip_spec()` / `conda_spec()` / `is_version_skipped()` / `dump()`
  2. `CUPY_E1B1894_SKIP_PIN`：单例，`skip_versions=("14.0.0","14.1.0")`, `min_version="13.6.0"`，含完整 issue 追踪说明
  3. `CupyE1b1894SkipAudit` dataclass：`has_skip_14_0()` / `has_skip_14_1()` / `has_both_skips()` / `assert_skips_present()` — CI 扫描 skip 约束是否仍存在（上游零机制）
  4. `CUPY_E1B1894_AUDIT` 模块级单例
  5. 自测从 6 项扩展到 9 项（新增 7/8/9 三项覆盖 e1b1894 逻辑）
- **自测结果**: `python src/walpurgis/core/dep_pin.py` → 全部通过 `[PASS] dep_pin a01924a+e1b1894 自测：9 项断言全部通过`

## migrate adbb2fb: Add SECURITY.md (#467)

- **Upstream commit**: adbb2fbf2713b0cbf7023c6a8d709556c708a52d (James Lamb, 2026-05-29, PR #467)
- **Commit message**: `Add SECURITY.md (#467)`
- **Upstream diff**: `.github/CODEOWNERS` + `.github/workflows/pr.yaml` + `SECURITY.md` 新增
- **迁移决策**: [SKIP] — 全是 GitHub 仓库治理文件（SECURITY.md 安全漏洞上报说明、CODEOWNERS、pr.yaml）；Walpurgis 作为代码研究仓库无需独立安全策略文件，零迁移价值

## migrate 51c8d26: Build and test with CUDA 13.3.0 (#476)

- **Upstream commit**: 51c8d2681af5cb5c41c8d6385061b6ae4774dc2e (Bradley Dice, 2026-06-11, PR #476)
- **Commit message**: `Build and test with CUDA 13.3.0 (#476)`
- **Upstream diff** (11 files): devcontainer json × 2、GitHub Actions workflow × 3、conda environments 重命名 all_cuda-132→all_cuda-133 × 2、dependencies.yaml CUDA 13.3 依赖、conda dev yaml 重命名 × 2
- **迁移决策**: [CONCEPTUALLY MERGED] — 上游 `51c8d26` 是 cugraph-gnn 的 CUDA 13.3.0 升级，等价于上一 session 已迁移的 `2d2bc51`（来自主 cugraph 仓库同名提交，该仓库已被折叠进 cugraph-gnn）。`cuda_compat.py` 中的 `Cuda2d2bc51UpgradeAudit` + `MatrixDepsRule` 已覆盖本次升级的所有语义内容。

## migrate 280f144: refactor: switch to `rapids-artifact-name` (#477)

- **Upstream commit**: 280f14425072c0424644ede09b33d635542fc4a9 (Gil Forsyth, 2026-06-11, PR #477)
- **Commit message**: `refactor: switch to rapids-artifact-name for consistent artifact naming (#477)`
- **Upstream diff** (12 files): `.github/workflows/build.yaml` + `ci/build_cpp.sh` + `ci/build_docs.sh` + `ci/build_python.sh` + `ci/build_python_noarch.sh` + `ci/build_wheel_*.sh` + `ci/test_*.sh`
- **迁移决策**: [SKIP] — 纯 CI/构建脚本重构，将 `rapids-package-name` 替换为 `rapids-artifact-name`（RAPIDS rapidsai/build-planning#270），所有变更集中在 `.github/` 和 `ci/` 目录，零 Python 源码改动，Walpurgis 不使用 RAPIDS CI 体系，零迁移价值

---

## migrate 3fd5f2a: [SKIP] Fix pagerank typo (#5545) — 纯 C++ 测试注释 typo 修复

- **Upstream commit**: 3fd5f2a176e8737e2d670cb84405a52a2e38a690 (cugraph, Colman Bouton, 2026-06-10, PR #5545)
- **Commit message**: `Fix pagerank typo (#5545)`
- **Upstream diff**: `cpp/tests/link_analysis/pagerank_test.cpp` — 注释中 "verties" → "vertices" + 版权年份更新 2025→2026
- **SKIP 理由**: 纯 C++ 测试文件注释 typo，无任何 Python 层改动；Walpurgis 无对应 C++ pagerank 测试文件。

---

## migrate ecf00f2: Improve Louvain determinism (#5541)  +  661f5eb: Fix Louvain bug (#5549)

- **Upstream commit 1**: ecf00f2ebfb2fb062db26b5fe4e245bb3df1d75e (cugraph, Chuck Hastings, 2026-06-04, PR #5541)
- **Commit message**: `Improve Louvain determinism (#5541)`
- **Upstream diff** (8 files changed):
  - `cpp/include/cugraph/algorithms.hpp`: threshold 参数注释扩展，语义从"收敛门槛"改为"顶点移动最小增益 + 收敛门槛"
  - `cpp/src/community/detail/common_methods.cuh`: `count_updown_moves_op_t` 新增 `min_gain` 字段；`cluster_update_op_t` 新增 `min_gain` 字段；`update_clustering_by_delta_modularity()` 新增 `threshold` 参数；条件 `delta_modularity > 0` 改为 `delta_modularity > min_gain`（×2处）
  - `cpp/src/community/detail/common_methods.hpp`: `update_clustering_by_delta_modularity()` 函数签名更新
  - `cpp/src/community/detail/common_methods_{mg,sg}_v{32,64}_e{32,64}.cu` (4个 .cu 文件): 调用处添加 `threshold` 参数
- **Upstream commit 2**: 661f5eb65701ae6d477b1039f81c189c1c4dba61 (cugraph, Chuck Hastings, 2026-06-10, PR #5549)
- **Commit message**: `Fix Louvain bug introduced in new threshold logic (#5549)`
- **Upstream diff** (2 files changed):
  - `cpp/src/community/detail/common_methods.cuh`: 新增 `louvain_delta_modularity_noise_floor<weight_t>()` constexpr + `compute_louvain_min_vertex_move_gain(threshold, n_vertices)` 模板函数；`nr_moves = thrust::count_if(...)` 改用 `min_vertex_move_gain` 替代直接 `threshold`
  - `cpp/tests/community/louvain_test.cpp`: 测试期望值更新（large threshold case：0.41978961 → 0.39907956）
- **迁移位置**: `src/walpurgis/core/louvain_determinism.py` — 新增（458 行）
- **鲁迅拿法改写（≥20%）**: 上游全为 C++ 模板代码（device functor + constexpr 函数），无任何 Python 层变更。本次迁移将 Louvain 确定性改进的算法逻辑完整对象化：
  1. **`LouvainThresholdPolicy` dataclass（frozen）**: 上游 threshold/n_vertices/dtype 三者散落于独立 C++ 函数，本类统一为强类型载体；`noise_floor` property（float32→1e-12 / float64→1e-15，对应 ecf00f2 constexpr 表）；`compute_min_vertex_gain()` 实现 661f5eb 的 max(threshold/n_vertices, noise_floor) 缩放修正——上游两个独立函数无共同容器
  2. **`DeltaModularityGain` dataclass**: 上游 delta_modularity 仅为 weight_t 裸标量内联于 device functor，本类提炼为值对象，`exceeds_threshold(policy)` 封装两次 commit 的完整判定链路（ecf00f2 改变比较基准 + 661f5eb 缩放修正），`is_noise(policy)` 告警接近 noise floor 的移动——上游无结构化表示
  3. **`VertexMoveDecision` dataclass**: Python 等价于 `count_updown_moves_op_t::operator()` + `cluster_update_op_t::operator()`（上游两个独立 device functor），`cluster_assigned` / `is_counted_move` 属性显式化方向判定逻辑，`explains()` 返回完整决策摘要——上游无任何 Python 可调试接口
  4. **`LouvainIterationStats` dataclass**: 上游无 Python 层聚合；本类记录单次迭代的 n_moves/global_modularity_gain/effective_threshold，`is_non_deterministic_risk()` 检测 ecf00f2 修复前的经典非确定性场景（有移动但增益极小）——原创 debug 辅助
  5. **`LouvainDeterminismAudit` dataclass（frozen）**: 枚举两个 commit 的变更文件数/描述，`validate_threshold_scaling()` 守卫 661f5eb 修复前的 bug 模式（大图直接用未缩放 threshold 会导致零移动），`describe()` 生成 MIGRATION_LOG 对齐摘要
  6. 断点调试 8 处：Policy 构建/noise_floor/min_gain/gain 评估/move 决策/iter stats/audit validate 均有 WALPURGIS_DEBUG=1 输出
- **自测结果**: `python src/walpurgis/core/louvain_determinism.py` → 16 项全部通过，`[PASS]`

---

## migrate e13bff4: [SKIP] fix(ci): use correct ordering in publish-wheel-search-key (#5554) — 纯 CI yaml

- **Upstream commit**: e13bff4417dfe818d8b8c7610a4afceb00bbcbb6 (cugraph, Gil Forsyth, 2026-06-10, PR #5554)
- **SKIP 理由**: 仅修改 `.github/workflows/build.yaml` 中 publish-wheel-search-key 条目顺序，Walpurgis 无对应 GH Actions 体系。

---

## migrate fa2a4c9: [SKIP] Reduce binary size scan (#5546) — 纯 C++ 库体积优化

- **Upstream commit**: fa2a4c935c15bdbc99e1edf9ad6f4aeadb7d8f0f (cugraph, Chuck Hastings, 2026-06-09, PR #5546)
- **Commit message**: `Reduce binary size scan (#5546)`
- **SKIP 理由**: 新增 `cpp/include/cugraph/utilities/thrust_wrappers.hpp` 和 `cpp/src/utilities/thrust_wrappers.cu`，封装 thrust::inclusive_scan/exclusive_scan 减少 .so 体积。全部 C++ 模板代码，无任何 Python 层内容，Walpurgis 无对应 C++ 工具层。

---

## migrate 013145b: [SKIP] Fix device functor errors compiling with CUDA 13.3 (#5552) — 纯 C++

- **Upstream commit**: 013145bebcf046da070a5e37d76da840e8fd22e6 (cugraph, Bradley Dice + Paul Taylor, 2026-06-09, PR #5552)
- **SKIP 理由**: 仅修改 `strongly_connected_components_impl.cuh`，将 CUDA device lambda 改为命名 functor 解决 CUDA 13.3 编译报错，纯 C++ 实现层，无 Python 层。

---

## migrate d7c953f: [SKIP] refactor: switch to rapids-artifact-name (#5544) — 纯 CI 重构

- **Upstream commit**: d7c953febc1ee4f874c4d61cbb3655127315edd2 (cugraph, Gil Forsyth, 2026-06-09, PR #5544)
- **SKIP 理由**: 14 个 CI/GH Actions 文件批量替换 `rapids-package-name` → `rapids-artifact-name`，Walpurgis 无对应 RAPIDS CI 体系。

## migrate 7a8fd29: add wholegraph (pylibwholegraph torch层 PEP8全量格式化)

- **Upstream commit**: 7a8fd290787a839bcd1ebe82d732ee26b195ad0b (cugraph-gnn, Alexandria Barghi, 2024-07-31)
- **Commit message**: `add wholegraph`
- **Upstream diff** (224 files changed, 581 insertions, 372 deletions):
  - 绝大多数文件（~210个）仅改版权年份 2023→2024（各 `2 +-`）
  - `python/pylibwholegraph/pylibwholegraph/torch/comm.py` — 8处 `global` 超长声明拆多行；docstring 行长包装
  - `python/pylibwholegraph/pylibwholegraph/torch/initialize.py` — `init` / `init_torch_env` / `init_torch_env_and_create_wm_comm` 函数签名展开；`finalize()` one-liner→if块
  - `python/pylibwholegraph/pylibwholegraph/torch/wholegraph_env.py` — 删除 8 处注释掉的 `print` 调试行，保留空行占位
  - `python/pylibwholegraph/pylibwholegraph/torch/embedding.py` — 删除 2 处 `print` 调试注释；`create_wholememory_optimizer` 签名展开；`dummy_input` 行内联
  - `python/pylibwholegraph/pylibwholegraph/torch/tensor.py` — `gather` / `scatter` 方法签名及调用展开；docstring 包装
  - `python/pylibwholegraph/pylibwholegraph/torch/utils.py` — 4处 `ValueError` f-string 现代化（`% (x,)` → f-string）；`wholememory_distributed_backend_type_to_str` 签名展开
  - `python/pylibwholegraph/pylibwholegraph/torch/cugraphops/__init__.py` — **新增**：从空文件(e69de29)补为含 Apache 2.0 header 的12行文件
  - 其余文件：trailing comma、docstring 包装等纯格式

- **CI/merge → SKIP**:
  - `ci/build_wheel_cugraph.sh` / `ci/test_wheel_cugraph.sh` / `ci/wheel_smoke_test_cugraph.py` — CI wheel 体系，Walpurgis 无对应
  - `ci/notebook_list.py` / `ci/utils/nbtest.sh` / `ci/utils/nbtestlog2junitxml.py` — notebook CI，Walpurgis 无 notebook CI
  - `ci/utils/git_helpers.py` — CI git 工具，Walpurgis 无对应
  - `conda/recipes/cugraph-dgl/build.sh` / `conda/recipes/cugraph-pyg/build.sh` — conda 构建，Walpurgis 无 conda 体系
  - `cpp/bench/` / `cpp/cmake/` / `cpp/include/` / `cpp/src/` / `cpp/tests/` — C++/CUDA 层，Walpurgis 纯 Python
  - `mg_utils/` — 多GPU集群工具脚本，Walpurgis 无对应
  - `python/cugraph-dgl/` / `python/cugraph-pyg/` — DGL/PyG 封装，Walpurgis 使用自有 dataloader
  - `python/pylibwholegraph/pylibwholegraph/binding/` — Cython 绑定层，Walpurgis 无编译体系
  - `python/pylibwholegraph/tests/` / `python/pylibwholegraph/pylibwholegraph/test_utils/` — 上游测试体系，Walpurgis 有自有测试

- **迁移位置**: `src/walpurgis/core/wholememory/wholegraph_style_reform.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`ReformKind` enum + `StyleReformRecord` dataclass**: 将上游"一次性格式化提交"抽象为8类可枚举改动类型，`is_semantic_change()` 区分语义变更 vs 纯格式，`summary()` 生成可读报告——上游直接提交无任何记录结构
  2. **`REFORM_RECORDS` 列表**: 8条关键改动的规范化 before/after 记录，覆盖 global 拆分/签名展开/one-liner展开/print删除/init补全/版权更新/trailing comma，使提交内容可程序化枚举和验证
  3. **`split_global_statement()` + `GlobalSplitAudit` dataclass**: 将上游手动拆分 `global` 声明的规则程序化——输入超长 global 行自动拆为合规多行，`from_source()` 扫描全文件，`report()` 输出违规报告；上游仅手动修了 comm.py
  4. **`DebugPrintAudit` 注册机制 + `scan_source_for_debug_prints()`**: 上游直接删除 print 注释；此处改为 `register_removed_debug_print()` 注册记录（8条），`scan_source_for_debug_prints()` 可扫描残留，使删除操作可溯源、可回查
  5. **`CugraphopsPackageSpec` dataclass**: 上游仅补了空 `__init__.py`；此类封装 `generate_compliant_init()` + `validate_init_content()` + `expected_header` 属性，使 cugraphops 子包合规性可程序化验证和生成
  6. **`FinalizePatch` dataclass**: 上游将 `finalize()` one-liner 三元式展开为 if 块；此类封装 `validate_source()` / `apply_to_source()` / `is_semantically_equivalent()`，使该补丁可幂等应用、可单测验证语义不变性
  7. 全链路 `WALPURGIS_DEBUG=1` 断点（6处）：覆盖 global 拆分、print扫描、spec校验、patch应用各阶段

- **自测结果**: 8 项全部 [PASS]

---

## migrate 9ecbc66: 标准化 fatbin 压缩策略（Use rapids_cuda_enable_fatbin_compression）

- **Upstream commit**: 9ecbc668fa376ab7398e8eef9053aecbe510ac91 (cugraph-gnn, Robert Maynard, 2025-08-13)
- **Commit message**: `Use rapids_cuda_enable_fatbin_compression (#273)`
- **Co-authors**: Robert Maynard (robertmaynard), Alex Barghi (alexbarghi-nv)
- **Approvers**: Bradley Dice (bdice), Alex Barghi (alexbarghi-nv)
- **Upstream diff** (1 file changed, 2 insertions, 7 deletions):
  - `cpp/CMakeLists.txt`:
    - 旧（7行手写逻辑）：`list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)` + 版本区间守卫（CUDA [12.9, 13.0) 时额外追加 `-Xfatbin=--compress-level=3`）
    - 新（2行委托）：`include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)` + `rapids_cuda_enable_fatbin_compression(VARIABLE WHOLEGRAPH_CUDA_FLAGS TUNE_FOR rapids)`

- **CI/merge → SKIP**:
  - `cpp/CMakeLists.txt` — C++ CMake 构建系统，Walpurgis 纯 Python，无 C++ 编译体系；fatbin 压缩 flag 属于 nvcc 编译器选项，不涉及 Python 运行时

- **迁移位置**: `src/walpurgis/core/fatbin_compression.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`FatbinCompressionFlag` 枚举**: 将上游 CMakeLists.txt 中两条裸字符串 nvcc flag（`-Xfatbin=-compress-all` / `-Xfatbin=--compress-level=3`）结构化为枚举，`as_nvcc_flag()` 序列化、`from_nvcc_flag()` 反向解析——上游直接 `list(APPEND ...)` 裸字符串，无任何类型化接口
  2. **`CudaVersionRange` dataclass**: 将上游 CMake `VERSION_GREATER_EQUAL 12.9 AND VERSION_LESS 13.0` 版本区间判断建模为 Python 不可变数据类，`contains(major, minor)` 方法精确还原半开区间语义——上游散落在 `if(...)` 条件中，无独立可测试对象
  3. **`FatbinCompressionRule` dataclass**: 将"什么版本区间适用什么压缩 flag"封装为有名字、可 hash、带 `rationale`/`introduced_by`/`removed_by` 字段的不可变记录，`matches_cuda_version()` 替代 CMake 版本比较——上游仅有匿名的 `if` 逻辑块
  4. **`resolve_legacy_flags()` + `LEGACY_FATBIN_RULES` 常量**: 将 9ecbc66 之前手写逻辑的全部行为文档化为可审计常量和可调用函数，作为语义等价性验证的基准——上游删除即删除，无可回查记录
  5. **`RapidsFatbinPolicy.is_semantically_equivalent_to_legacy()`**: 通过比较新旧两种策略在各代表性 CUDA 版本（12.8/12.9/13.0）下的 flag 集合，程序化验证 9ecbc66 迁移的语义无损性——上游无任何验证机制
  6. **`FatbinCompressionAudit.assert_no_legacy_flags()`**: 扫描任意路径内是否残留手写 `-Xfatbin=-compress-all` 等 flag，供 CI 或单测调用——上游无此工具
  7. 全链路 `WALPURGIS_DEBUG=1` 断点（6处）：覆盖 flag 序列化、版本区间判断、规则匹配、策略解析、语义等价性校验、残留扫描各阶段

- **自测结果**: 11 项全部 [PASS]（含语义等价性验证 5 个代表性 CUDA 版本：12.8/12.9/12.10/13.0/11.8）

---

## migrate 9ecbc66: 标准化 fatbin 压缩策略（Use rapids_cuda_enable_fatbin_compression）

- **Upstream commit**: 9ecbc668fa376ab7398e8eef9053aecbe510ac91 (cugraph-gnn, Robert Maynard, 2025-08-13)
- **Commit message**: `Use rapids_cuda_enable_fatbin_compression (#273)`
- **Co-authors**: Robert Maynard (robertmaynard), Alex Barghi (alexbarghi-nv)
- **Approvers**: Bradley Dice (bdice), Alex Barghi (alexbarghi-nv)
- **Upstream diff** (1 file changed, 2 insertions, 7 deletions):
  - `cpp/CMakeLists.txt`:
    - 旧（7行手写逻辑）：`list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)` + 版本区间守卫（CUDA [12.9, 13.0) 时额外追加 `-Xfatbin=--compress-level=3`）
    - 新（2行委托）：`include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)` + `rapids_cuda_enable_fatbin_compression(VARIABLE WHOLEGRAPH_CUDA_FLAGS TUNE_FOR rapids)`

- **CI/merge → SKIP**:
  - `cpp/CMakeLists.txt` — C++ CMake 构建系统，Walpurgis 纯 Python，无 C++ 编译体系；fatbin 压缩 flag 属于 nvcc 编译器选项，不涉及 Python 运行时

- **迁移位置**: `src/walpurgis/core/fatbin_compression.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`FatbinCompressionFlag` 枚举**: 将上游 CMakeLists.txt 中两条裸字符串 nvcc flag（`-Xfatbin=-compress-all` / `-Xfatbin=--compress-level=3`）结构化为枚举，`as_nvcc_flag()` 序列化、`from_nvcc_flag()` 反向解析——上游直接 `list(APPEND ...)` 裸字符串，无任何类型化接口
  2. **`CudaVersionRange` dataclass**: 将上游 CMake `VERSION_GREATER_EQUAL 12.9 AND VERSION_LESS 13.0` 版本区间判断建模为 Python 不可变数据类，`contains(major, minor)` 方法精确还原半开区间语义——上游散落在 `if(...)` 条件中，无独立可测试对象
  3. **`FatbinCompressionRule` dataclass**: 将"什么版本区间适用什么压缩 flag"封装为有名字、可 hash、带 `rationale`/`introduced_by`/`removed_by` 字段的不可变记录，`matches_cuda_version()` 替代 CMake 版本比较——上游仅有匿名的 `if` 逻辑块
  4. **`resolve_legacy_flags()` + `LEGACY_FATBIN_RULES` 常量**: 将 9ecbc66 之前手写逻辑的全部行为文档化为可审计常量和可调用函数，作为语义等价性验证的基准——上游删除即删除，无可回查记录
  5. **`RapidsFatbinPolicy.is_semantically_equivalent_to_legacy()`**: 通过比较新旧两种策略在各代表性 CUDA 版本（12.8/12.9/13.0）下的 flag 集合，程序化验证 9ecbc66 迁移的语义无损性——上游无任何验证机制
  6. **`FatbinCompressionAudit.assert_no_legacy_flags()`**: 扫描任意路径内是否残留手写 `-Xfatbin=-compress-all` 等 flag，供 CI 或单测调用——上游无此工具
  7. 全链路 `WALPURGIS_DEBUG=1` 断点（6处）：覆盖 flag 序列化、版本区间判断、规则匹配、策略解析、语义等价性校验、残留扫描各阶段

- **自测结果**: 11 项全部 [PASS]（含语义等价性验证 5 个代表性 CUDA 版本：12.8/12.9/12.10/13.0/11.8）

---

---

## migrate a560ad0: 版本升级 26.02 → 26.04（Update to 26.04 #387）

- **Upstream commit**: a560ad09a875e6e283a68557d181d650d1d34228 (cugraph-gnn, Jake Awe / AyodeAwe, 2026-01-16)
- **Commit message**: `Update to 26.04 (#387)`
- **Author**: Jake Awe (AyodeAwe, NVIDIA)
- **Context**: 26.02 release burndown 流程的版本升级步骤，全量 sed 替换 20 个配置文件中所有 "26.02" → "26.04" 引用（89 insertions，89 deletions，净变更 0 行语义）

- **Upstream diff** (20 files changed, 89 insertions(+), 89 deletions(−)) — 全部为字符串替换，无逻辑变更:
  | 文件域 | 文件数 | 变更内容 |
  |--------|--------|----------|
  | `.devcontainer/cuda{12.9,13.1}-{conda,pip}/devcontainer.json` | 4 | 基础镜像标签 `26.02-*` → `26.04-*`，rapids-build-utils 特性版本 `26.2` → `26.4` |
  | `.github/workflows/{build,pr,test}.yaml` | 3 | CI 容器镜像 `rapidsai/ci-conda:26.02-latest` → `26.04-latest` |
  | `VERSION` | 1 | `26.02.00` → `26.04.00` |
  | `conda/environments/all_cuda-{129,131}_arch-{aarch64,x86_64}.yaml` | 4 | cudf/cugraph/cuml/pylibcugraph/rmm 版本约束 `26.2.*` → `26.4.*` |
  | `dependencies.yaml` | 1 | 同上，所有 rapids 包约束 `26.2.*` → `26.4.*` |
  | `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-{129,131}_arch-{aarch64,x86_64}.yaml` | 4 | pylibcugraph `26.2.*` → `26.4.*` |
  | `python/cugraph-pyg/pyproject.toml` | 1 | pylibcugraph/pylibwholegraph/cugraph/cuml 版本约束升级，版权年补 2026 |
  | `python/libwholegraph/pyproject.toml` | 1 | libraft/librmm 版本约束升级，版权年补 2026 |
  | `python/pylibwholegraph/pyproject.toml` | 1 | libwholegraph/libraft/librmm 版本约束升级，版权年补 2026 |

- **CI/merge → SKIP**（全部 20 个文件）:
  - `.devcontainer/**` — Walpurgis 无 devcontainer 配置，基础镜像标签无迁移目标
  - `.github/workflows/**` — Walpurgis 无 RAPIDS CI 流水线，CI 容器镜像无迁移目标
  - `VERSION` — 上游 RAPIDS 版本文件，Walpurgis 版本独立管理
  - `conda/environments/**` — Walpurgis 无 conda 构建矩阵，conda env yaml 无迁移目标
  - `dependencies.yaml` — RAPIDS 构建依赖清单，Walpurgis 用 pyproject.toml 独立管理
  - `python/cugraph-pyg/conda/**` — 上游 conda 开发环境，同上
  - `python/{cugraph-pyg,libwholegraph,pylibwholegraph}/pyproject.toml` — 上游包构建配置，非 Walpurgis 源码

- **迁移位置**: `src/walpurgis/core/version_bump_policy.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`RapidsVersion` dataclass**: 将上游裸字符串 "26.02" / "26.04" 强类型化为可比较的 `(year, month)` 二元组，`__post_init__` 校验 RAPIDS 只在偶数月发布，`cycle_delta()` 计算发布周期差，`conda_wildcard` / `pip_pin` 属性派生约束格式——上游只有 sed，无任何版本对象
  2. **`BumpKind` 枚举**: 区分 MINOR（同年月跳变）、YEARLY（跨年）、PATCH（仅 patch 号）三类跃迁语义，上游不作区分，一律字符串替换
  3. **`VersionBump` dataclass**: 封装"FROM→TO"跃迁事实，携带 `commit_sha`/`pr_number`/`author`/`rationale`，`bump_kind()`/`is_forward()`/`as_sed_pattern()` 方法均为上游所无；`A560AD0_BUMP` 实例化 a560ad0 的全部跃迁元数据
  4. **`AffectedScope` + `AffectedFile`**: 将 20 个受影响文件按功能域（devcontainer/CI/conda_env/conda_recipe/dep_manifest/pyproject）分类，每个文件携带 `skip_reason` 字段，`by_domain()` 可按域查询，`dump()` 打印结构化 SKIP 理由——上游只有 git diff stat 平铺列表
  5. **`BumpCompatibilityProbe`**: 在 Python import 层主动探测 Walpurgis 运行时安装的 RAPIDS 相关包（cugraph/pylibcugraph/rmm 等 9 个）版本是否与目标 cycle 兼容，上游无此运行时探测机制
  6. **`VersionBumpAudit`**: 扫描任意文本文件，检查旧版本字符串（"26.02" / "26.2" 双格式）是否残留，`assert_clean()` 供 CI 调用验证 bump 是否彻底——上游通过全局 sed 后无回头检查机制
  7. 全链路 `WALPURGIS_DEBUG=1` 断点（7 处）：覆盖版本解析→跃迁计算→影响域统计→兼容性探测→残留扫描→自测各阶段

- **自测结果**: 10 项全部 [PASS]（RapidsVersion 解析排序、tag/conda_wildcard/pip_pin 格式、cycle_delta 计算、BumpKind 判断、as_sed_pattern、AffectedScope 统计 20 SKIP 0 MIGRATE、by_domain 过滤、残留检测、奇数月拒绝、元数据完整性）

---

---

## migrate a560ad0: 版本升级 26.02 → 26.04（Update to 26.04 #387）

- **Upstream commit**: a560ad09a875e6e283a68557d181d650d1d34228 (cugraph-gnn, Jake Awe / AyodeAwe, 2026-01-16)
- **Commit message**: `Update to 26.04 (#387)`
- **Author**: Jake Awe (AyodeAwe, NVIDIA)
- **Context**: 26.02 release burndown 流程的版本升级步骤，全量 sed 替换 20 个配置文件中所有 "26.02" → "26.04" 引用（89 insertions，89 deletions，净变更 0 行语义）

- **Upstream diff** (20 files changed, 89 insertions(+), 89 deletions(−)) — 全部为字符串替换，无逻辑变更:
  | 文件域 | 文件数 | 变更内容 |
  |--------|--------|----------|
  | `.devcontainer/cuda{12.9,13.1}-{conda,pip}/devcontainer.json` | 4 | 基础镜像标签 `26.02-*` → `26.04-*`，rapids-build-utils 特性版本 `26.2` → `26.4` |
  | `.github/workflows/{build,pr,test}.yaml` | 3 | CI 容器镜像 `rapidsai/ci-conda:26.02-latest` → `26.04-latest` |
  | `VERSION` | 1 | `26.02.00` → `26.04.00` |
  | `conda/environments/all_cuda-{129,131}_arch-{aarch64,x86_64}.yaml` | 4 | cudf/cugraph/cuml/pylibcugraph/rmm 版本约束 `26.2.*` → `26.4.*` |
  | `dependencies.yaml` | 1 | 同上，所有 rapids 包约束 `26.2.*` → `26.4.*` |
  | `python/cugraph-pyg/conda/cugraph_pyg_dev_cuda-{129,131}_arch-{aarch64,x86_64}.yaml` | 4 | pylibcugraph `26.2.*` → `26.4.*` |
  | `python/cugraph-pyg/pyproject.toml` | 1 | pylibcugraph/pylibwholegraph/cugraph/cuml 版本约束升级，版权年补 2026 |
  | `python/libwholegraph/pyproject.toml` | 1 | libraft/librmm 版本约束升级，版权年补 2026 |
  | `python/pylibwholegraph/pyproject.toml` | 1 | libwholegraph/libraft/librmm 版本约束升级，版权年补 2026 |

- **CI/merge → SKIP**（全部 20 个文件）:
  - `.devcontainer/**` — Walpurgis 无 devcontainer 配置，基础镜像标签无迁移目标
  - `.github/workflows/**` — Walpurgis 无 RAPIDS CI 流水线，CI 容器镜像无迁移目标
  - `VERSION` — 上游 RAPIDS 版本文件，Walpurgis 版本独立管理
  - `conda/environments/**` — Walpurgis 无 conda 构建矩阵，conda env yaml 无迁移目标
  - `dependencies.yaml` — RAPIDS 构建依赖清单，Walpurgis 用 pyproject.toml 独立管理
  - `python/cugraph-pyg/conda/**` — 上游 conda 开发环境，同上
  - `python/{cugraph-pyg,libwholegraph,pylibwholegraph}/pyproject.toml` — 上游包构建配置，非 Walpurgis 源码

- **迁移位置**: `src/walpurgis/core/version_bump_policy.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`RapidsVersion` dataclass**: 将上游裸字符串 "26.02" / "26.04" 强类型化为可比较的 `(year, month)` 二元组，`__post_init__` 校验 RAPIDS 只在偶数月发布，`cycle_delta()` 计算发布周期差，`conda_wildcard` / `pip_pin` 属性派生约束格式——上游只有 sed，无任何版本对象
  2. **`BumpKind` 枚举**: 区分 MINOR（同年月跳变）、YEARLY（跨年）、PATCH（仅 patch 号）三类跃迁语义，上游不作区分，一律字符串替换
  3. **`VersionBump` dataclass**: 封装"FROM→TO"跃迁事实，携带 `commit_sha`/`pr_number`/`author`/`rationale`，`bump_kind()`/`is_forward()`/`as_sed_pattern()` 方法均为上游所无；`A560AD0_BUMP` 实例化 a560ad0 的全部跃迁元数据
  4. **`AffectedScope` + `AffectedFile`**: 将 20 个受影响文件按功能域（devcontainer/CI/conda_env/conda_recipe/dep_manifest/pyproject）分类，每个文件携带 `skip_reason` 字段，`by_domain()` 可按域查询——上游只有 git diff stat 平铺列表
  5. **`BumpCompatibilityProbe`**: 在 Python import 层主动探测 Walpurgis 运行时安装的 RAPIDS 相关包（cugraph/pylibcugraph/rmm 等 9 个）版本是否与目标 cycle 兼容，上游无此运行时探测机制
  6. **`VersionBumpAudit`**: 扫描任意文本文件，检查旧版本字符串（"26.02" / "26.2" 双格式）是否残留，`assert_clean()` 供 CI 调用——上游通过全局 sed 后无回头检查机制
  7. 全链路 `WALPURGIS_DEBUG=1` 断点（7 处）：覆盖版本解析→跃迁计算→影响域统计→兼容性探测→残留扫描→自测各阶段

- **自测结果**: 10 项全部 [PASS]

---

---

## migrate 3d4c449: 移除 rapids_logger 依赖 + 更新 dependency-file-generator hook

- **Upstream commit**: 3d4c449e1b08825887e6f8e42605ec0d282ed5a8 (cugraph-gnn, Kyle Edwards / James Lamb, 2025-08-27)
- **Commit message**: `Update rapids-dependency-file-generator (#285)`
- **Upstream diff 摘要** (5 files changed, 5 insertions, 20 deletions)：
  | 文件 | 变更 | 说明 |
  |---|---|---|
  | `.github/workflows/pr.yaml` | +3 | 三个 job 的 paths-filter 均新增 `!.pre-commit-config.yaml` 排除规则，使 pre-commit 配置变更不再触发 CI |
  | `.pre-commit-config.yaml` | ±4 | `rapids-dependency-file-generator` hook `v1.19.0`→`v1.20.0`；args 新增 `--warn-all --strict` |
  | `dependencies.yaml` | -15 | 删除 `depends_on_libwholegraph` 文件包含项中的 `depends_on_rapids_logger`；删除 `python_run_libwholegraph` 依赖块（共 5 行）；删除整个 `depends_on_rapids_logger` 依赖块（9 行） |
  | `python/libwholegraph/libwholegraph/load.py` | -2 | 删除 `import rapids_logger` + `rapids_logger.load_library()` 两行调用 |
  | `python/libwholegraph/pyproject.toml` | -1 | 删除 `rapids-logger==0.1.*,>=0.0.0a0` 运行时依赖 |

- **CI/merge → SKIP**（4 个文件，全部跳过）：
  - `.github/workflows/pr.yaml` — SKIP：Walpurgis 无 RAPIDS CI 流水线，paths-filter 规则无迁移目标
  - `.pre-commit-config.yaml` — SKIP：上游 pre-commit hook 版本管理，Walpurgis pre-commit 独立管理
  - `dependencies.yaml` — SKIP：RAPIDS 构建依赖清单，Walpurgis 用 pyproject.toml 独立管理
  - `python/libwholegraph/pyproject.toml` — SKIP：上游包构建配置，非 Walpurgis 源码

- **迁移位置**: `src/walpurgis/core/rapids_logger_removal.py` — 新建

- **鲁迅拿法改写（≥20%）**:
  1. **`ProbeStatus` 枚举**：上游只有裸 `try/except ModuleNotFoundError`，本文件将结果语义化为 `PRESENT / ABSENT / BROKEN / SUPERSEDED` 四态，`SUPERSEDED` 专门编码「上游主动移除」这一事实——上游无此区分
  2. **`ProbeResult` dataclass**：结构化记录单个依赖的探测结果，携带 `version / error / superseded_by_commit`，`is_loadable()` 提供明确可用性判断——上游只有副作用式调用，无任何返回值或记录
  3. **`_probe_module()` 工厂函数**：将 import→version→load 三步合并为可复用探针，`superseded_by` 参数令「不加载」成为一等公民决策而非代码缺席——上游删行即删逻辑，无策略封装
  4. **`WholegraphDepPolicy` dataclass**：将 `required`（libraft）与 `superseded`（rapids_logger + commit SHA）分离声明，`probe_all()` / `load_required()` 方法使策略可测试、可序列化——上游只有裸顺序调用
  5. **`RapidsLoggerAudit` dataclass**：`scan_file()` + `assert_clean()` 主动检测残留引用，CI 可调用 `assert_clean(path)` 确保 rapids_logger 彻底清除——上游无此回头检查机制
  6. 全链路 `WALPURGIS_DEBUG=1` 断点（7 处）：覆盖 probe_start → probe_superseded → probe_loaded → policy_probe_all → policy_probe_done → audit_scan → self_test_start 各阶段

- **自测结果**: 5 项全部 [PASS]（ProbeStatus 语义、superseded 路径、absent 路径、审计器 scan、审计器 assert_clean）

- **Upstream commit**: 78128d9ee1f80fcf183ddc1312a8abb382c271d1 (cugraph-gnn, Alex Barghi, 2025-06-25, PR #222)
- **Commit message**: `Remove Non-Unified API and Remaining TensorDict Code (#222)`
- **Upstream diff 摘要** (24 files changed, 138 insertions, 753 deletions):
  - `data/__init__.py`: 删除 GraphStore 工厂函数（is_multi_gpu 分发逻辑）、WholeFeatureStore/TensorDictFeatureStore 兼容包装；直接从 graph_store 导入 GraphStore（原 NewGraphStore 正式重命名）
  - `data/feature_store.py`: 删除 TensorDictFeatureStore 类（~110行，基于 tensordict.TensorDict 的单机非分布式实现）；文档字符串 "WholeFeatureStore" → "FeatureStore"
  - `data/graph_store.py`: 删除旧 GraphStore 类（~340行，基于 tensordict.TensorDict({}, batch_size=(2,))）；NewGraphStore 正式重命名为 GraphStore；**bug fix**: `_put_edge_index` 新增 `isinstance(edge_index, list) → torch.stack(edge_index)` 守卫；清除 `tensordict = import_optional("tensordict")` import
  - 依赖清理: conda recipe/dependencies.yaml/conda dev yaml 全部删除 tensordict>=0.1.2 依赖条目

- **Bug 根因 (Knuth 三维)**:
  1. **数学维度**: 旧 GraphStore 用 `tensordict.TensorDict({}, batch_size=(2,))` 存边索引，batch_size 对齐语义与 PyG EdgeTensorType leading dim 不完全等价；新 DistMatrix + dict 无此歧义
  2. **算法维度**: `_put_edge_index` 接收 list 类型时未做 `torch.stack`，直接赋值给 DistMatrix 切片存入 Python list 引用，后续 `.local_row.numel()` 抛 AttributeError——78128d9 核心 bug fix
  3. **工程维度**: tensordict 作为重依赖（conda: >=0.1.2,<=0.6.2）仅用于旧 GraphStore 容器，改为普通 dict + DistMatrix 后彻底消除

- **CI/merge → SKIP**:
  - `conda/recipes/cugraph-pyg/recipe.yaml` — SKIP: conda 构建配方，Walpurgis 无 conda 体系
  - `dependencies.yaml` — SKIP: RAPIDS 依赖矩阵，Walpurgis 无对应体系
  - `python/cugraph-pyg/conda/*.yaml` — SKIP: conda 开发环境，同上
  - `python/cugraph-pyg/pyproject.toml` — SKIP: 上游包构建配置，非 Walpurgis 源码
  - `python/cugraph-pyg/cugraph_pyg/tests/**` — SKIP: Walpurgis 无 CI 测试体系
  - `python/cugraph-pyg/cugraph_pyg/examples/**` — SKIP: MNMG 示例脚本，Walpurgis 不维护上游 example

- **迁移位置**:
  - `src/walpurgis/core/tensordict_removal.py` — **新建**: 完整迁移记录 + 审计工具
  - `src/walpurgis/core/unified_store.py` — **更新**: 模块头部追加 78128d9 注记；`FeatureStoreFactory` 中 `NewGraphStore` import 更新为 `GraphStore`（反映正式重命名）

- **鲁迅拿法改写（≥20%）**:
  1. **`EdgeIndexTypeGuard` dataclass**: 将上游零散的 isinstance 分支（cupy/numpy/pandas/cudf/list/Tensor/DistMatrix/DistTensor-tuple 共 8 路）收敛为单一守卫对象，`EdgeIndexKind` 枚举显式标注各路径，`apply_list_fix()` 封装 78128d9 核心 bug fix——上游直接写在 _put_edge_index 方法体内，无任何结构化或复用
  2. **`DeprecatedAPIAudit` dataclass**: 将本次删除的 4 个符号（TensorDictFeatureStore/WholeFeatureStore/GraphStore-factory/NewGraphStore）显式记录为审计表，`warn()` 提供迁移提示，`check_import()` 可在运行时拦截——上游直接删除，调用方只能靠 ImportError 感知
  3. **`TensorDictMigrationDiagnoser` dataclass**: 量化旧（tensordict.TensorDict 单机）vs 新（dict+DistMatrix 多GPU）路径的 4 个维度（多GPU支持/tensordict依赖/list bug/名称），`describe()` 生成对比报告，`validate_new_path()` 断言式校验——上游无任何对比工具
  4. 全链路 `WALPURGIS_DEBUG=1` 断点（4处），覆盖类型检测→list修复→废弃告警→路径校验各阶段

- **自测结果**: 5 项全部 [PASS]

---

## migrate 65f4d7b: [SKIP] Revert "Forward-merge release/25.12 into main" (#350) — 纯 CI/workflow revert，Walpurgis 无 CI 体系

- **Upstream commit**: 65f4d7b22570b714b8815e6a9c94f0cf0e34ff84 (cugraph-gnn, Jake Awe / AyodeAwe, 2025-11-17, PR #350)
- **Commit message**: `Revert "Forward-merge release/25.12 into main" (#350)` — 回滚 PR #349
- **Upstream diff 摘要** (6 files changed, 33 insertions, 33 deletions)：
  - `.github/workflows/build.yaml` — 所有 `uses: rapidsai/shared-workflows/…@release/25.12` → `@main`（10 处 workflow ref 回滚）
  - `.github/workflows/pr.yaml` — 同上（15 处 workflow ref 回滚）
  - `.github/workflows/test.yaml` — 同上（5 处 workflow ref 回滚）
  - `.github/workflows/trigger-breaking-change-alert.yaml` — 同上（1 处）
  - `RAPIDS_BRANCH` — `release/25.12` → `main`
  - `cpp/scripts/run-cmake-format.sh` — 注释中 URL 分支名 `release/25.12` → `main`

- **SKIP 判定**：
  - 变更内容 100% 为 CI workflow YAML（`.github/workflows/`）+ RAPIDS_BRANCH 文件 + cmake 格式脚本注释
  - Walpurgis 无 GitHub Actions CI 体系，无 rapidsai/shared-workflows 调用链，无 RAPIDS_BRANCH 配置文件
  - 实质是上游在 main 分支撤销了一次误合并的 release/25.12 → main 前向合并；对 Python/C++ 运行时代码零影响
  - **结论**：SKIP — CI/merge-revert 类提交，不适用于 Walpurgis

## migrate ec608df: [SKIP] Use `sccache-dist` build cluster for conda and wheel builds (#341) — 纯 CI/sccache-dist 基础设施，Walpurgis 无 conda/wheel CI 体系

- **Upstream commit**: ec608dfc03daf4a059be5ebf42aaf9a660322574 (cugraph-gnn, Paul Taylor, 2025-11-20, PR #341)
- **Commit message**: `Use sccache-dist build cluster for conda and wheel builds (#341)`
- **Upstream diff** (10 files changed, 80 insertions, 18 deletions):
  - `.devcontainer/Dockerfile` — 新增 `SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true` 环境变量
  - `.github/workflows/build.yaml` — conda-cpp-build / conda-python-build / wheel-build 系列 job 新增 `node_type: cpu8` + `sccache-dist-token-secret-name: GIST_REPO_READ_ORG_GITHUB_TOKEN`
  - `.github/workflows/pr.yaml` — PR CI 的 conda/wheel build/test job 同步新增上述两个字段（13 处）
  - `.github/workflows/test.yaml` — test job 同步追加 `sccache-dist-token-secret-name`
  - `ci/build_cpp.sh` — `sccache --zero-stats` → `sccache --stop-server 2>/dev/null || true`；末尾追加 `sccache --stop-server`
  - `ci/build_python.sh` — 三处 `sccache --zero-stats` → `sccache --stop-server` + `sccache --show-adv-stats` 中间保留
  - `ci/build_wheel.sh` — 新增 `SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true` + key prefix；同上 stop-server 替换
  - `cmake/rapids_config.cmake` — 新增 `set(ENV{SCCACHE_NO_DIST_COMPILE} "1")` 防止 CMake 编译器测试走 sccache-dist
  - `conda/recipes/libwholegraph/recipe.yaml` — cache env 块新增 14 个 `SCCACHE_DIST_*` / `SCCACHE_S3_*` 环境变量条目（含 fallback default 值）
  - `conda/recipes/pylibwholegraph/recipe.yaml` — 同 libwholegraph recipe，对称新增 14 个环境变量条目

- **变更语义**: RAPIDS 部署了自动扩缩的云端构建集群（autoscaling cloud build cluster），通过 `sccache-dist` 分布式编译大型 RAPIDS 项目（贡献 rapidsai/build-planning#228）。此 commit 将 conda 和 wheel 构建接入该集群：为 build job 指定 `cpu8` 节点类型，注入 `sccache-dist` 认证 token secret，并将 conda recipe 的 cache 环境变量配置完善（含调度器 URL、超时、重试次数、预处理器缓存模式等）。

- **CI/merge/docs 文件 → SKIP**:
  - `.devcontainer/Dockerfile` — SKIP：devcontainer 环境变量，Walpurgis 不维护 devcontainer
  - `.github/workflows/build.yaml` / `pr.yaml` / `test.yaml` — SKIP：全为 GitHub Actions workflow 字段更新，Walpurgis CI 体系独立
  - `ci/build_cpp.sh` / `ci/build_python.sh` / `ci/build_wheel.sh` — SKIP：RAPIDS CI 构建脚本，Walpurgis 无 conda/wheel CI 流水线
  - `cmake/rapids_config.cmake` — SKIP：CMake 构建配置，Walpurgis 无 CMake 构建体系
  - `conda/recipes/libwholegraph/recipe.yaml` / `conda/recipes/pylibwholegraph/recipe.yaml` — SKIP：conda rattler-build 配方环境变量扩展，Walpurgis 无 conda 体系

- **Knuth 三维审查**:
  1. **diff 完整性**: 10 个文件逐一审查；核心语义是\"把 `sccache --zero-stats` 换成 `sccache --stop-server`\"（因为 sccache-dist 模式下每次构建需要停止再重启 server 以刷新认证 token），以及在各层注入 `SCCACHE_DIST_*` 环境变量。零 Python/CUDA 运行时算法变更。`SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true` 启用 S3 预处理器缓存，可显著减少相似代码的重编译时间，但仅在有 AWS 凭证的环境中有效。
  2. **用户角度**: `sccache --stop-server` vs `--zero-stats` 的差异——`zero-stats` 只清零统计计数器而保持 server 运行；`stop-server` 完全停止 server（sccache-dist 模式下每个 job 需独立 server 进程携带本次 token）。上游注释无说明，此处明确记录。
  3. **系统角度安全**: `SCCACHE_DIST_AUTH_TOKEN` 通过 secret 注入，不硬编码；`GIST_REPO_READ_ORG_GITHUB_TOKEN` 是 GitHub App token，作用域仅限 gist/repo-read，权限范围可控。`SCCACHE_NO_DIST_COMPILE=1` 在 cmake 测试阶段禁用 dist 编译是正确的安全做法（CMake 编译器测试的 object 不应走分布式缓存，会导致错误的编译器探测结果）。

---

## migrate adce20b: Apply suggestion from @greptile-apps[bot] — 删除重复 elif 分支

- **Upstream commit**: adce20b8c4c83893bcf7ffcd263ee536337a03c8 (cugraph-gnn)
- **Commit message**: `Apply suggestion from @greptile-apps[bot]`
- **Author**: Nate Rock <rockhowse@gmail.com>；Co-authored-by: greptile-apps[bot]
- **Date**: Wed Nov 12 08:30:59 2025 -0600
- **Context**: greptile-apps bot 发现 `ci/release/update-version.sh` 中连续两条相同的
  `elif [[ "${RUN_CONTEXT}" == "release" ]]; then`，删除重复行；推测由合并/粘贴失误引入。

- **Upstream diff** (1 file changed, 1 deletion):
  | 文件 | 变更内容 |
  |------|----------|
  | `ci/release/update-version.sh` | 删除第148行重复的 `elif [[ "${RUN_CONTEXT}" == "release" ]]; then` |

  原始 bug 形态（父 commit）：
  ```bash
  if [[ "${RUN_CONTEXT}" == "main" ]]; then
    echo "Keeping external documentation references on main branch"
  elif [[ "${RUN_CONTEXT}" == "release" ]]; then   # ← 重复第1条
  elif [[ "${RUN_CONTEXT}" == "release" ]]; then   # ← 重复第2条（永远不可达）
    sed_runner "s|\bmain\b|release/${NEXT_SHORT_TAG}|g" cpp/scripts/run-cmake-format.sh
  fi
  ```
  修复后（adce20b）：仅保留一条 `elif`，逻辑恢复正确。

- **CI/merge → SKIP**:
  - `ci/release/update-version.sh` — SKIP：Walpurgis 无 RAPIDS CI release 体系，无迁移目标

- **迁移位置**: `src/walpurgis/core/upstream_version_updater.py` — 追加（adce20b 段）

- **鲁迅拿法改写（≥20%）**:
  1. **`DuplicateBranchError`** — 专用异常，在 `register_branch()` 构造期即拒绝重复上下文注册；上游 bash 中重复 `elif` 静默失效，直至 greptile bot 才被发现，无任何主动防御
  2. **`BranchDispatcher` dataclass** — 将 bash `if/elif/elif/fi` 链强类型化为可枚举 Python 结构；`register_branch()` 携带 `label`+`instructions` 两个维度，`dispatch()` 按 `RunContext` 枚举路由；上游仅有裸 bash 字符串比较，无结构化表示
  3. **`RunContext` 枚举** — 将 `"main"` / `"release"` 字符串强枚举化，`dispatch()` 对未知值提前抛 `ValueError`，列出有效值；上游无此校验，拼写错误会导致所有 elif 静默跳过
  4. **`SedInstruction` dataclass** — 将 `sed_runner "s|...|...|g" target` 调用建模为 `(pattern, replacement, target_file, flags)` 四元组，`as_sed_expr()` 展开变量占位符；上游只有裸字符串，无可测试的数据表示
  5. **`_build_doc_link_dispatcher()`** — 实例化与 adce20b 后逻辑等价的派发器：MAIN→空指令，RELEASE→`SedInstruction(r"\bmain\b", "release/XX.YY", "cpp/scripts/run-cmake-format.sh")`；结构即文档
  6. 全链路 **`WALPURGIS_DEBUG=1` 断点**（5处）：覆盖构造→注册→派发→sed展开各阶段，上游无任何诊断输出

- **自测结果**: 7 项全部 [PASS]
  - test1: main 上下文 → 空指令（保持文档在 main）
  - test2: release 上下文 → `sed s|\bmain\b|release/26.04|g` 正确
  - test3: 重复 elif 分支 → `DuplicateBranchError` 即时拒绝
  - test4: 未知 context → `ValueError`
  - test5: `registered_contexts()` 枚举正确
  - test6: `WALPURGIS_DEBUG=1` 调试路径正常运行（无崩溃）
  - test7: `SedInstruction` 空 pattern → `ValueError`
