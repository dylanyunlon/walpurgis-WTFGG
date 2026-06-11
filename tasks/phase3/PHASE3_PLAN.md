# Phase 3 迁移计划: cugraph-gnn → Walpurgis (M501-M750)

## 状态总览

| 指标 | 数值 |
|------|------|
| cugraph-gnn 总commit | 450 |
| 已迁移 | 47 |
| 未迁移 | 403 |
| 其中: 纯Merge(0文件) | 107 → 统一 SKIP |
| 其中: CI/版本/文档 | 214 → 审查后多数 SKIP |
| 其中: 核心运行时代码 | ~82 → 需逐个深度迁移 |

## METR-LA 实验最新结果 (已确认)

```
(On average over 12 horizons) Test MAE: 2.90 | Test RMSE: 5.91 | Test MAPE: 7.91%
Best Val MAE: 2.6585 (epoch 15)
```

距TITAN(2.88) 0.02, 距STAEFormer(2.90) 平手。继续跑多种子+算法改进后预计突破。

## 子Claude分配策略

每位Claude负责一批commit, 按从first commit(64bfd1)到最新的时间顺序。
commit类型决定工作量: SKIP(写一行)→轻松, 核心CODE→需深入diff+Knuth审查+20%改写。

---

### Claude-8 (M501-M525) — 当前主控
**角色**: 计划制定 + imports.py修复 + 任务派发准备
**已完成**:
- ✅ 项目全量审查 (tree/git log/每个关键文件逐行阅读)
- ✅ cugraph-gnn clone + 450 commit分析
- ✅ 修复 src/walpurgis/utils/imports.py (tensor子包import依赖)
- ✅ SYNTH 3-epoch 训练验证通过 (best_val=4.4708)
- ✅ 制定Phase 3迁移计划 + 子Claude任务文件

### Claude-9 (M526-M550) — commits 4e7a730 → 3bbdbb5 (~40 commits)
**范围**: 仓库早期清理 + wholegraph导入 + 24.08→24.10重构
**关键commit**:
- bd703b3 (208 files): add wholegraph to repo → 核心C++ wholegraph源码评估
- 7a8fd29 (224 files): add wholegraph → 第二批wholegraph文件
- 0ea4925 (49 files): refactor → cugraph-pyg代码重构
- 90db89a: use correct wg communicator → Bug fix, 迁移价值
- 其余多为SKIP(merge/CI/conda)

### Claude-10 (M551-M575) — commits 755c2e3 → 2f41ad3 (~40 commits)
**范围**: biased采样 + 负采样 + PyG兼容 + 24.10→24.12
**关键commit**:
- 3e5df7c: pull in changes from cugraph repo
- f57ed88 (20 files): pull in changes → 大批功能更新
- 2b6f2cd: Merge pyg-neg-sampling → 负采样支持
- 其余CI/版本更新多为SKIP

### Claude-11 (M576-M600) — commits 4f250a5 → f6e3654 (~40 commits)
**范围**: CI现代化 + cugraph-ops移除 + 异构采样
**关键commit**:
- a9ab8b4 (13 files): [FEA] Support Heterogeneous Sampling → 高价值
- d38b832 (50 files): remove dependency on cugraph-ops → 架构变更
- 0e88280 (19 files): Support PyG 2.6 → API兼容更新
- df5bdc4 (64 files): update wholegraph → C++代码更新

### Claude-12 (M601-M625) — commits b578959 → 11ccf38 (~40 commits)
**范围**: 25.04→25.06 + PyG统一API + DGL废弃
**关键commit**:
- 431801c: Deprecate Dask API → 架构决策
- 8074120: Deprecate Unbuffered Sampling → 采样器简化
- 1e91ed7 (13 files): Remove Dask API from cuGraph-PyG → 代码移除
- e01196b (10 files): Make WholeGraph Hard Dependency → 依赖变更

### Claude-13 (M626-M650) — commits 70c33af → 0a4c8af (~40 commits)
**范围**: DGL移除 + 非统一API移除 + 25.08→25.10
**关键commit**:
- fb8296e (68 files): Remove cuGraph-DGL → 大规模移除
- 78128d9 (24 files): Remove Non-Unified API → TensorDict清理
- b10f279 (36 files): Remove cugraph Python dependency → 解耦

### Claude-14 (M651-M675) — commits d4dcf7eb → c11936f (~40 commits)
**范围**: 25.10→26.02 + 时序采样完善 + bf16支持
**关键commit**:
- a056923 (8 files): Temporal Negative Sampling → 时序负采样
- 75cd001: Add New Unsupervised Learning Example → 新示例
- b58ea19 (13 files): fp16 embedding train → 精度支持
- 220563b: bf16 in feature store → 精度支持

### Claude-15 (M676-M700) — commits 468ad45 → c06bbbe (~40 commits)
**范围**: 26.04→26.06 + disjoint采样 + 最新bugfix
**关键commit**:
- 659a0e1: Fix disjoint sampling test → Bug fix
- 6d1a8de: Support more dtypes → FeatureStore增强
- 94ac7fe: remove packaging dependency → 依赖清理

### Claude-16 (M701-M725) — 所有SKIP commit批量处理
**角色**: 审查所有merge/CI/版本/文档类commit, 批量写SKIP条目到MIGRATION_LOG
**工作量**: ~300个SKIP条目, 每个一行描述 + 原因

### Claude-17 (M726-M750) — TeX + 实验汇总
**角色**: 汇总所有迁移结果, 更新TeX论文相关章节, 确保pdflatex编译

---

## 铁律 (所有子Claude遵守)

1. 不开新分支, 不用v2/v3/port/alt等后缀
2. 改的是算法, 20%鲁迅拿法, 不是字符串/docstring
3. 每个commit的MIGRATION_LOG条目必须包含:
   - 上游commit hash + message
   - diff分析 (文件数/关键改动)
   - Knuth审查三问 (diff对比源/用户bug/系统安全)
   - Walpurgis迁移位置 (新增文件/改写要点)
4. 作者: dylanyunlon <dogechat@163.com>
5. 断点调试: 每个迁移文件至少3个WALPURGIS_DEBUG断点
6. git push前必须rebase: git pull --rebase origin main
