# Walpurgis 技术对标: 10个大厂异构GPU基础设施仓库

> Walpurgis核心: 异构GPU (H100+A6000) 上的GNN训练 + 分层内存管理 + 时序子图处理

## 已在用 (直接上游)

### 1. rapidsai/cugraph-gnn ← 我们的迁移源
- NVIDIA RAPIDS, 450 commits
- cuGraph-PyG + WholeGraph: GPU加速GNN采样/训练
- 异构内存: WholeMemory分布式共享存储 (HBM/GDDR/Host)
- **重合度**: 100% — Walpurgis的tensor/sampler/examples全部从此迁移

### 2. rapidsai/wholegraph
- NVIDIA RAPIDS
- WholeMemory: 多GPU分布式张量存储, NVLink优化
- 核心: P2P内存访问, 多tier存储 (device/pinned/managed)
- **重合度**: 95% — Walpurgis的slab_allocator/tiered_allocator/temp_memory_handle直接对应

## 底层内存与通信

### 3. NVIDIA/nccl
- NVIDIA集体通信库
- 多GPU/多节点all-reduce/broadcast/scatter
- 异构拓扑感知: NVLink/NVSwitch/PCIe路由
- **重合度**: 高 — Walpurgis的communicator.cpp/nvlink_clique.py对齐NCCL拓扑

### 4. NVIDIA/cuda-samples
- CUDA官方示例集
- 多GPU P2P, unified memory, managed memory
- 异构内存带宽基准测试
- **重合度**: 中高 — hetero_bench.cu的E1-E8实验直接参照CUDA samples的multi-GPU模式

### 5. pytorch/pytorch (torch.cuda.use_mem_pool / MemPool)
- PyTorch核心
- CUDA内存池管理, RMM集成, DDP多GPU训练
- **重合度**: 中高 — Walpurgis的memory_pool.py对齐MemPool API

## GNN训练框架

### 6. pyg-team/pytorch_geometric
- PyG: 图神经网络框架
- NeighborLoader, 异构图采样, 时序采样
- **重合度**: 高 — Walpurgis的dataloader/sampler层是PyG的cuGraph后端

### 7. dmlc/dgl (Deep Graph Library)
- 分布式图采样, 多GPU图分区
- CSC/COO格式, mini-batch采样
- **重合度**: 中 — Walpurgis的sampling_csc_helpers.py从DGL后端迁移

## 异构GPU训练调度

### 8. microsoft/DeepSpeed
- 微软分布式训练框架
- ZeRO: 参数/梯度/优化器状态分片
- 异构GPU支持: DeepSpeed-MoE
- **重合度**: 中 — Walpurgis的梯度裁剪/混合精度训练参照DeepSpeed模式

### 9. NVIDIA/Megatron-LM
- NVIDIA大模型训练框架
- 管道并行 + 张量并行 + 数据并行
- 多GPU内存管理, 异构节点支持
- **重合度**: 中 — Walpurgis的模型并行思路参照Megatron的分片策略

## 时序图/异构图专用

### 10. snap-stanford/ogb (Open Graph Benchmark)
- 斯坦福图基准数据集
- 含METR-LA类交通数据评估标准
- GNN性能标杆
- **重合度**: 中 — Walpurgis的METR-LA评估直接使用OGB风格的评估协议

---

## 为什么说"MAE实验不是重点"

Walpurgis的本质是 **异构GPU时序子图引擎** (系统论文), 不是 "又一个交通预测模型"。

| 维度 | 我们做的 | 那些只做MAE的 |
|------|---------|-------------|
| C++引擎 | slab_allocator/partition_skiplist/seqlock | 无 |
| CUDA基准 | hetero_bench.cu 11个实验 (E1-E11) | 无 |
| 多GPU内存 | WholeMemory迁移, NVLink拓扑, tier迁移 | 单GPU |
| GNN后端 | cuGraph-PyG采样/分布式存储 | PyG/DGL默认 |
| 预测 | D2STGNN + 20%改写 = **应用层ablation** | 全部工作 |

MAE 2.90已经平STAEFormer, 预测只是ablation的一个维度。
核心贡献在系统层: 异构内存引擎让同样的模型在H100+A6000混合集群上
比单tier存储快4.1x (窄查询) / 1.75x (中查询), 这才是论文的主打。
