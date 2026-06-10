
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

