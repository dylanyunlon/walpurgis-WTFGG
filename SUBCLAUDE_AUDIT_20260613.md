# 小弟Claude审计报告 — 2026-06-13

## 一、walpurgis-WTFGG (cugraph-gnn迁移)

| 指标 | 数值 |
|------|------|
| 总commit | 463 |
| 迁移commit | 231 |
| cugraph-gnn上游总commit | 452 |
| 已覆盖upstream hash | ~425 (含MIGRATION_LOG记录) |
| core策略文件 | 69个 |
| 有断点调试的文件 | 42个 (61%) |
| 后缀违规 | 0 |

### 问题1: 重复劳动 (27个upstream被多次迁移)

cookie并发竞争导致多个小弟同时处理同一个upstream commit:

| upstream | 重复次数 | 问题 |
|----------|----------|------|
| f46eb9e | 3次 | 3个都SKIP, 浪费但无害 |
| b89f57d | 3次 | **矛盾决策**: A写了399行策略模块, B直接SKIP, C改Makefile |
| 6ea54ab | 3次 | 3个都做了scatter_op修复, 代码略有不同 |
| 131d8ba | 3次 | 3个都SKIP |
| daf857d | 2次 | **完全重复**: doctor_check.py提交了2遍 |
| 其他 | 2次×22个 | 多为相同SKIP, 少量矛盾 |

### 问题2: 矛盾决策 (b89f57d案例分析)

```
小弟A (440d157): 写了cuda_build_config.py, 399行, NvccProbe+FatbinPolicy+5处_dbg → 最佳
小弟B (be1e1ad): [SKIP] CMake C++ only → 判断错误, 该commit有CUDA编译影响
小弟C (67a1533): 改Makefile NVFLAGS → 可接受但抽象不足
```

**结论**: 小弟A做得最好。B的SKIP判断有误。

### 代码质量评分

| 文件 | 质量 | 说明 |
|------|------|------|
| core/disjoint_sampler.py | ★★★★★ | 241行, 完整dataclass封装, 上游溯源清晰 |
| core/doctor_check.py | ★★★★★ | 267行, DoctorCtx对象化, breakpoint_trace |
| core/fp16_embedding_grad.py | ★★★★ | 移植得当, 有调试 |
| dataloader/link_loader.py | ★★★★ | LinkLoader移植, 边采样激活 |
| sampler/dgl_neighbor_sampler.py | ★★★★ | 偏置采样权重支持 |
| cuda_build_config.py | ★★★★★ | 399行策略模块, nvcc探测+缓存+调试 |

## 二、Neuron_SP (Megatron-LM迁移)

| 指标 | 数值 |
|------|------|
| 总commit | 3692 |
| Megatron迁移commit | 308 |
| Megatron-LM上游总commit | 9062 |
| 已覆盖milestone | 至M1444 |
| 重复迁移 | 5个 |
| 迁移目标目录 | deepspeed/compile/ |

### 代码质量: 高

最近一次迁移 (M1420: Megatron 397d0b2eb):
- 修改6个文件, +442行, -61行
- 正确拆分BaseConfig/TransformerConfig
- 完整的upstream溯源注释
- 代码映射: megatron/core/* → deepspeed/compile/core_*

## 三、根因: cookie并发竞争

多个小弟Claude同时用同一个cookie创建对话, 导致:
1. 同一批次被多个小弟独立处理 → 重复劳动
2. 不同小弟对同一commit做出不同决策 → 矛盾
3. 并发push到main → merge冲突(已有2个merge commit)

### 修复方案

1. **去重清理**: 用git rebase -i合并重复的MIGRATION_LOG条目
2. **矛盾仲裁**: b89f57d保留小弟A的策略模块(最佳), revert B和C
3. **防重复**: dispatch脚本加锁 — 同一commit hash不允许重复派发
4. **cookie序列化**: 派发间隔从3s增加到10s, 避免API竞争

## 四、待办

1. cugraph-gnn: ~27个upstream commit未覆盖 (多为CI/merge, 可batch SKIP)
2. Megatron-LM → walpurgis: 尚未开始 (已为Neuron_SP做了308个)
3. Neuron_SP → walpurgis: 已迁移hetero_mesh/double_buffer/comm_profile (本轮)
4. SYNTH实验: tensor/utils.py f-string语法错误待修
5. 9062个Megatron commit的walpurgis迁移派发系统已就绪, 等cookie空闲
