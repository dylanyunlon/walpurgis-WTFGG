## 欺诈检测示例 (fraud)

### 概览

本目录对应上游 940ab01 [FEA] Add Elliptic Bitcoin fraud example 迁移。

包含两个脚本，协同构成完整的欺诈检测流水线:

### bitcoin_mnmg.py — GNN 嵌入生成 (多GPU)

使用 cuGraph-PyG 在 Elliptic Bitcoin 数据集上训练 GNN 模型，生成节点嵌入。

支持 GraphSAGE / GCN / GAT 三种编码器，通过 `--encoder` 参数指定。

`--embedding_dir` 指定时，将嵌入写出为 parquet 文件，供 `bitcoin_rf.py` 使用。

运行方式:
```
WALPURGIS_DEBUG=1 torchrun --nnodes 1 --nproc_per_node 1 \
    bitcoin_mnmg.py \
    --dataset_root /data \
    --embedding_dir /data/bitcoin_embeddings
```

### bitcoin_rf.py — 随机森林分类器

读取 `bitcoin_mnmg.py` 生成的嵌入，训练 cuML RandomForestClassifier，
对比三种方案的混淆矩阵 / 准确率 / ROC AUC:

- RF + GNN 嵌入
- GNN Only (softmax 概率直接分类)
- RF Only (原始94维特征)

运行方式:
```
WALPURGIS_DEBUG=1 python bitcoin_rf.py \
    --dataset_root /data \
    --embedding_dir /data/bitcoin_embeddings
```

### 调试

`WALPURGIS_DEBUG=1` 开启全链路断点调试 print，覆盖所有数据结构体状态:
- 参数解析 dump
- 分布式图构建 (edge_index / feature shape / barrier)
- 每 batch 的 x / edge_index / out shape
- 嵌入推理写回 index
- parquet 读取对齐检查
- RF fit/evaluate dtype 和 class distribution

### Knuth 审查已知问题

1. `bitcoin_rf.py` 的 `cudf.read_parquet(embedding_dir)` 合并目录下所有 parquet，
   多 rank 写出时总行数 != `data.num_nodes`，会触发 `EmbeddingDataset.load()` 中的对齐检查报错
2. `EllipticBitcoin` 含 y=2 (unknown) 节点，cross_entropy (2分类头) 遇到 y=2 行为未定义
3. `bitcoin_mnmg.py` 推理阶段 `drop_last=True` 导致嵌入不完整

作者: dylanyunlon<dogechat@163.com>
